import os
import sys
import json
import re
from typing import Dict, List, Tuple, Optional

import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from llava.model.builder import load_pretrained_model

MODEL_PATH = "/data/susheel/MM_Neurons-main/llava-llama-2-13b-chat-lightning-preview"
CONV_MODE = "llava_llama_2"
PROMPT = """
You are analyzing whether the animal in the image looks more like a dog or a cat.

Follow the exact output format, but use only features that are actually visible in the current image.
Do not repeat the same feature list for every image.
The reasoning must be image-specific.

Example 1:
Decision: dog
Reasoning: visible muzzle, hanging ears, neck collar
Support class: dog
Consistency: consistent

Example 2:
Decision: dog
Reasoning: compact face, upright ears, cat-like posture
Support class: cat
Consistency: mismatch

Example 3:
Decision: cat
Reasoning: narrow face, upright ears, compact body
Support class: cat
Consistency: consistent

Example 4:
Decision: cat
Reasoning: long muzzle, floppy ears, dog-like body
Support class: dog
Consistency: mismatch

Now answer for the given image.

Strict rules:
- Write exactly 4 lines.
- Do not write a paragraph.
- Use only clues visible in this image.
- At least one clue must be image-specific, such as color, collar, pose, tail, body shape, background, or interaction.
- Do not reuse the exact same clue set across images.
- After 'Decision:' write only one word: dog or cat.
- After 'Support class:' write only one word: dog or cat.
- After 'Consistency:' write only one word: consistent or mismatch.

Output format:
Decision: dog or cat
Reasoning: clue1, clue2, clue3
Support class: dog or cat
Consistency: consistent or mismatch
""".strip()

DELTA_FILE = "analysis/concept_delta_dog_to_cat_causal.pt"
TEST_DIR = "edit_data/test_images"
SAVE_JSON = "analysis/reasoning_clean_compare_20.json"

DEVICE = "cuda"
DTYPE = torch.float16

TOP_K = 20
ALPHA = 10.0
MAX_NEW_TOKENS = 64
ONLY_POSITIVE_CAUSAL = True


def build_prompt(model):
    qs = PROMPT
    if getattr(model.config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def get_llama_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "get_model"):
        inner = model.get_model()
        if hasattr(inner, "layers"):
            return inner.layers
    raise RuntimeError("Could not find transformer layers.")


def list_images(folder):
    out = []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
            out.append(os.path.join(folder, f))
    return out


def build_selected_neurons(delta_data):
    if "causal_ranked" not in delta_data:
        raise KeyError(f"Expected 'causal_ranked' in delta file, got keys={list(delta_data.keys())}")

    ranked = delta_data["causal_ranked"]
    filtered = []
    for item in ranked:
        cscore = float(item["causal_score"])
        if ONLY_POSITIVE_CAUSAL and cscore <= 0:
            continue
        filtered.append({
            "layer": int(item["layer"]),
            "neuron": int(item["neuron"]),
            "delta_value": float(item["delta_value"]),
            "causal_score": cscore,
        })

    total_positive = len(filtered)
    selected = filtered[:TOP_K]
    if not selected:
        raise RuntimeError("No positive causal neurons available.")

    mean_abs_delta = sum(abs(x["delta_value"]) for x in selected) / len(selected)
    mean_abs_delta = mean_abs_delta + 1e-8

    by_layer: Dict[int, List[Tuple[int, float]]] = {}
    for item in selected:
        edit_strength = float(item["delta_value"] / mean_abs_delta)
        by_layer.setdefault(item["layer"], []).append((item["neuron"], edit_strength))

    return by_layer, selected, total_positive


def make_pre_hook(neuron_score_pairs, alpha=5.0):
    neuron_ids = [n for n, _ in neuron_score_pairs]
    score_map = {n: s for n, s in neuron_score_pairs}

    def hook(module, inputs):
        if len(inputs) == 0:
            return inputs
        x = inputs[0].clone()
        for n in neuron_ids:
            x[..., n] = x[..., n] + alpha * score_map[n]
        return (x,)

    return hook


def register_edit_hooks(model, by_layer, alpha=5.0):
    handles = []
    layers = get_llama_layers(model)
    for layer_idx, neuron_score_pairs in by_layer.items():
        if layer_idx < 0 or layer_idx >= len(layers):
            continue
        down_proj = layers[layer_idx].mlp.down_proj
        h = down_proj.register_forward_pre_hook(make_pre_hook(neuron_score_pairs, alpha=alpha))
        handles.append(h)
    return handles


@torch.inference_mode()
def generate_response(model, tokenizer, image_processor, image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].to(device=DEVICE, dtype=DTYPE)

    prompt = build_prompt(model)
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    ).unsqueeze(0).to(DEVICE)

    stop_str = conv_templates[CONV_MODE].sep if conv_templates[CONV_MODE].sep_style != SeparatorStyle.TWO else conv_templates[CONV_MODE].sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=False,
        temperature=0,
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True,
        stopping_criteria=[stopping_criteria],
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    out = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    return out


def parse_response(text: str) -> Dict[str, Optional[str]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    joined = "\n".join(lines)

    def extract(prefix: str) -> Optional[str]:
        m = re.search(rf"^{re.escape(prefix)}\s*(.+)$", joined, flags=re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else None

    decision = extract("Decision:")
    reasoning = extract("Reasoning:")
    support_class = extract("Support class:")
    consistency = extract("Consistency:")

    def norm_label(x: Optional[str]) -> Optional[str]:
        if x is None:
            return None
        xl = x.lower()
        if "dog" in xl and "cat" not in xl:
            return "dog"
        if "cat" in xl and "dog" not in xl:
            return "cat"
        return "ambiguous"

    def norm_consistency(x: Optional[str]) -> Optional[str]:
        if x is None:
            return None
        xl = x.lower()
        if "consistent" in xl and "mismatch" not in xl:
            return "consistent"
        if "mismatch" in xl:
            return "mismatch"
        return "unknown"

    return {
        "raw_text": text,
        "decision": norm_label(decision),
        "reasoning": reasoning,
        "support_class": norm_label(support_class),
        "consistency": norm_consistency(consistency),
        "well_formed": all(x is not None for x in [decision, reasoning, support_class, consistency]),
    }


def pair_category(orig: Dict[str, Optional[str]], edited: Dict[str, Optional[str]]) -> str:
    od, ed = orig.get("decision"), edited.get("decision")
    os, es = orig.get("support_class"), edited.get("support_class")

    if od == "dog" and ed == "cat" and os == "dog" and es == "cat":
        return "dog_to_cat_with_reasoning_change"
    if od == ed and os == es:
        return "same_decision_same_reasoning"
    if od == ed and os != es:
        return "same_decision_changed_reasoning"
    return "other"


def main():
    if not os.path.exists(DELTA_FILE):
        raise FileNotFoundError(f"Missing {DELTA_FILE}. Run find_concept_delta_causal.py first.")

    delta_data = torch.load(DELTA_FILE, map_location="cpu")
    by_layer, selected, total_positive = build_selected_neurons(delta_data)

    print(f"Loaded causal delta: {delta_data.get('source', 'dog')} -> {delta_data.get('target', 'cat')}")
    print(f"Positive causal neurons: {total_positive}")
    print(f"Using TOP_K={len(selected)}, ALPHA={ALPHA}")
    print(f"Neuron percentage used: {100.0 * len(selected) / total_positive:.2f}%")

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        MODEL_PATH,
        model_base=None,
        model_name="llava"
    )
    model = model.to(DEVICE)
    model.eval()

    results = []
    summary_counts = {
        "same_decision_same_reasoning": 0,
        "same_decision_changed_reasoning": 0,
        "dog_to_cat_with_reasoning_change": 0,
        "other": 0,
    }

    for image_path in list_images(TEST_DIR):
        print(f"\nProcessing: {image_path}")
        original_raw = generate_response(model, tokenizer, image_processor, image_path)
        original = parse_response(original_raw)

        handles = register_edit_hooks(model, by_layer, alpha=ALPHA)
        try:
            edited_raw = generate_response(model, tokenizer, image_processor, image_path)
            edited = parse_response(edited_raw)
        finally:
            for h in handles:
                h.remove()

        category = pair_category(original, edited)
        summary_counts[category] += 1

        print("Original:")
        print(original_raw)
        print("Edited:")
        print(edited_raw)
        print("Pair category:", category)

        results.append({
            "image_path": image_path,
            "top_k": len(selected),
            "alpha": ALPHA,
            "total_positive_neurons": total_positive,
            "neuron_percentage_used": 100.0 * len(selected) / total_positive,
            "original": original,
            "edited": edited,
            "pair_category": category,
        })

    num_images = len(results)
    summary = {
        "top_k": len(selected),
        "alpha": ALPHA,
        "total_positive_neurons": total_positive,
        "neuron_percentage_used": 100.0 * len(selected) / total_positive,
        "num_images": num_images,
        "counts": summary_counts,
        "percentages": {k: (100.0 * v / num_images if num_images > 0 else 0.0) for k, v in summary_counts.items()},
    }

    with open(SAVE_JSON, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\nSaved reasoning comparison to {SAVE_JSON}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
