import os
import sys
import json
from typing import Dict, List

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
PROMPT = "What animal is shown in the image? Answer with one word."
DELTA_FILE = "analysis/concept_delta_dog_to_cat_causal.pt"
TEST_DIR = "edit_data/test_images"
SAVE_JSON = "analysis/targeted_edit_neuron_only_15.json"

DEVICE = "cuda"
DTYPE = torch.float16

TOP_K = 15
ALPHA = 10.0
MAX_NEW_TOKENS = 8
ONLY_POSITIVE_CAUSAL = True

TARGET_WORD = "cat"
SOURCE_WORD = "dog"
TARGET_LOGIT_BIAS = 0.0
SOURCE_LOGIT_BIAS = 0.0


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


def build_single_token_ids(tokenizer, word: str) -> List[int]:
    variants = [word, " " + word, word.capitalize(), " " + word.capitalize()]
    ids = set()
    for v in variants:
        tok_ids = tokenizer.encode(v, add_special_tokens=False)
        if len(tok_ids) != 1:
            continue
        tid = tok_ids[0]
        decoded = tokenizer.decode([tid]).strip().lower()
        if decoded == word.lower():
            ids.add(tid)
    if not ids:
        raise RuntimeError(f"Could not find a clean single-token id for word={word!r}")
    return sorted(ids)


def build_selected_neurons(delta_data):
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
        raise RuntimeError("No positive causal neurons available. Check the causal ranking file.")

    # normalize by selected mean abs delta, use delta_value for direction, causal_score for ranking.
    mean_abs_delta = sum(abs(x["delta_value"]) for x in selected) / len(selected)
    mean_abs_delta = mean_abs_delta + 1e-8

    by_layer: Dict[int, List[tuple]] = {}
    for item in selected:
        edit_strength = float(item["delta_value"] / mean_abs_delta)
        by_layer.setdefault(item["layer"], []).append((item["neuron"], edit_strength))

    return by_layer, selected, total_positive


def make_pre_hook(neuron_score_pairs, alpha=5.0):
    neuron_ids = [n for n, _ in neuron_score_pairs]
    score_map = {n: s for n, s in neuron_score_pairs}

    def hook(module, inputs):
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


def build_inputs(model, tokenizer, image_processor, image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].to(device=DEVICE, dtype=DTYPE)

    prompt = build_prompt(model)
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(DEVICE)
    return input_ids, image_tensor


@torch.inference_mode()
def get_next_token_logits(model, tokenizer, image_processor, image_path):
    input_ids, image_tensor = build_inputs(model, tokenizer, image_processor, image_path)
    outputs = model(input_ids=input_ids, images=image_tensor, use_cache=False)
    return outputs.logits[:, -1, :].detach().float().cpu()[0]


@torch.inference_mode()
def generate_caption(model, tokenizer, image_processor, image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].to(device=DEVICE, dtype=DTYPE)

    prompt = build_prompt(model)
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(DEVICE)

    stop_str = conv_templates[CONV_MODE].sep if conv_templates[CONV_MODE].sep_style != SeparatorStyle.TWO else conv_templates[CONV_MODE].sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    generated = input_ids.clone()
    for _ in range(MAX_NEW_TOKENS):
        outputs = model(input_ids=generated, images=image_tensor, use_cache=False)
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        decoded = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=False)
        if stop_str in decoded:
            break

    out = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    return out


def classify_output(text):
    t = text.lower()
    has_dog = "dog" in t
    has_cat = "cat" in t

    if has_dog and has_cat:
        return "both_dog_and_cat"
    elif has_cat and not has_dog:
        return "dog_to_cat_success"
    elif has_dog and not has_cat:
        return "same_dog"
    else:
        return "different_from_both"


def main():
    if not os.path.exists(DELTA_FILE):
        raise FileNotFoundError(f"Missing {DELTA_FILE}. Run find_concept_delta_causal.py first.")

    delta_data = torch.load(DELTA_FILE, map_location="cpu")
    by_layer, selected, total_positive = build_selected_neurons(delta_data)

    print(f"Loaded causal targeted edit: {delta_data['source']} -> {delta_data['target']}")
    print(f"Positive causal neurons: {total_positive}")
    print(f"Using TOP_K={len(selected)}, ALPHA={ALPHA}")
    print(f"Neuron percentage used: {100.0 * len(selected) / total_positive:.2f}%")

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        MODEL_PATH,
        model_base=None,
        model_name="llava",
    )
    model = model.to(DEVICE)
    model.eval()

    target_ids = build_single_token_ids(tokenizer, TARGET_WORD)
    source_ids = build_single_token_ids(tokenizer, SOURCE_WORD)
    print("Target ids:", target_ids)
    print("Source ids:", source_ids)

    results = []
    category_counts = {
        "different_from_both": 0,
        "same_dog": 0,
        "both_dog_and_cat": 0,
        "dog_to_cat_success": 0,
    }

    for image_path in list_images(TEST_DIR):
        print("\nProcessing:", image_path)

        baseline_logits = get_next_token_logits(model, tokenizer, image_processor, image_path)
        baseline_gap = float(baseline_logits[target_ids].max().item() - baseline_logits[source_ids].max().item())
        original = generate_caption(model, tokenizer, image_processor, image_path)

        handles = register_edit_hooks(model, by_layer, alpha=ALPHA)
        try:
            edited_logits = get_next_token_logits(model, tokenizer, image_processor, image_path)
            edited_gap = float(edited_logits[target_ids].max().item() - edited_logits[source_ids].max().item())
            edited = generate_caption(model, tokenizer, image_processor, image_path)
        finally:
            for h in handles:
                h.remove()

        category = classify_output(edited)
        category_counts[category] += 1

        row = {
            "image_path": image_path,
            "source": delta_data["source"],
            "target": delta_data["target"],
            "top_k": len(selected),
            "alpha": ALPHA,
            "target_word": TARGET_WORD,
            "source_word": SOURCE_WORD,
            "target_logit_bias": TARGET_LOGIT_BIAS,
            "source_logit_bias": SOURCE_LOGIT_BIAS,
            "total_positive_neurons": total_positive,
            "neuron_percentage_used": 100.0 * len(selected) / total_positive,
            "baseline_gap_cat_minus_dog": baseline_gap,
            "edited_gap_cat_minus_dog": edited_gap,
            "gap_delta": edited_gap - baseline_gap,
            "original_caption": original,
            "edited_caption": edited,
            "changed": original != edited,
            "category": category,
        }
        results.append(row)

        print("Baseline gap:", baseline_gap)
        print("Edited gap  :", edited_gap)
        print("Gap delta   :", edited_gap - baseline_gap)
        print("Original:", original)
        print("Edited  :", edited)

    num_images = len(results)
    mean_gap_delta = sum(r["gap_delta"] for r in results) / max(num_images, 1)

    summary = {
        "top_k": len(selected),
        "alpha": ALPHA,
        "total_positive_neurons": total_positive,
        "neuron_percentage_used": 100.0 * len(selected) / total_positive,
        "num_images": num_images,
        "mean_gap_delta": mean_gap_delta,
        "counts": category_counts,
        "percentages": {
            "different_from_both": 100.0 * category_counts["different_from_both"] / num_images,
            "same_dog": 100.0 * category_counts["same_dog"] / num_images,
            "both_dog_and_cat": 100.0 * category_counts["both_dog_and_cat"] / num_images,
            "dog_to_cat_success": 100.0 * category_counts["dog_to_cat_success"] / num_images,
        },
    }

    output = {"summary": summary, "results": results}
    os.makedirs(os.path.dirname(SAVE_JSON), exist_ok=True)
    with open(SAVE_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved neuron-only targeted edit results to {SAVE_JSON}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
