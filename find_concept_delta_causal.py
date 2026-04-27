import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

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
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model


MODEL_PATH = "/data/susheel/MM_Neurons-main/llava-llama-2-13b-chat-lightning-preview"
CONV_MODE = "llava_llama_2"
PROMPT = "What animal is shown in the image? Answer with one word."
SOURCE_DIR = "edit_data/source_dog"
TARGET_DIR = "edit_data/target_cat"
SAVE_PATH = "analysis/concept_delta_dog_to_cat_causal.pt"

DEVICE = "cuda"
DTYPE = torch.float16

# Same late semantic layers you were already using.
LAYER_MIN = 20
LAYER_MAX = 39

# Practical settings: start small, then increase if needed.
MEAN_SOURCE_MAX = 30
MEAN_TARGET_MAX = 30
SHORTLIST_TOP_M = 256
EVAL_IMAGES_MAX = 12
PROBE_ALPHA = 8.0
ONLY_POSITIVE_DELTA = True

TARGET_WORD = "cat"
SOURCE_WORD = "dog"


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


def list_images(folder: str) -> List[str]:
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


def load_image_tensor(image_processor, image_path: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    return image_processor.preprocess(image, return_tensors="pt")["pixel_values"].to(device=DEVICE, dtype=DTYPE)


def prepare_inputs(model, tokenizer, image_processor, image_path: str):
    image_tensor = load_image_tensor(image_processor, image_path)
    prompt = build_prompt(model)
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(DEVICE)
    return input_ids, image_tensor


@torch.inference_mode()
def get_next_token_logits(model, tokenizer, image_processor, image_path: str) -> torch.Tensor:
    input_ids, image_tensor = prepare_inputs(model, tokenizer, image_processor, image_path)
    outputs = model(input_ids=input_ids, images=image_tensor, use_cache=False)
    return outputs.logits[:, -1, :].detach().float().cpu()[0]


@torch.inference_mode()
def get_cat_dog_gap(model, tokenizer, image_processor, image_path: str, target_ids: List[int], source_ids: List[int]) -> float:
    logits = get_next_token_logits(model, tokenizer, image_processor, image_path)
    cat_score = logits[target_ids].max().item()
    dog_score = logits[source_ids].max().item()
    return cat_score - dog_score


def collect_single_image_representation(model, tokenizer, image_processor, image_path: str) -> Dict[int, torch.Tensor]:
    layers = get_llama_layers(model)
    captured = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(module, inputs):
            x = inputs[0].detach().float().cpu()
            captured[layer_idx] = x.mean(dim=(0, 1))
        return hook

    for layer_idx in range(len(layers)):
        if LAYER_MIN <= layer_idx <= LAYER_MAX:
            h = layers[layer_idx].mlp.down_proj.register_forward_pre_hook(make_hook(layer_idx))
            handles.append(h)

    input_ids, image_tensor = prepare_inputs(model, tokenizer, image_processor, image_path)
    with torch.inference_mode():
        _ = model(input_ids=input_ids, images=image_tensor, use_cache=False)

    for h in handles:
        h.remove()

    return captured


def collect_folder_mean(model, tokenizer, image_processor, folder: str, max_images: int) -> Dict[int, torch.Tensor]:
    paths = list_images(folder)[:max_images]
    if len(paths) == 0:
        raise RuntimeError(f"No images found in {folder}")

    sums = defaultdict(lambda: None)
    count = 0

    for p in paths:
        rep = collect_single_image_representation(model, tokenizer, image_processor, p)
        for layer_idx, vec in rep.items():
            if sums[layer_idx] is None:
                sums[layer_idx] = vec.clone()
            else:
                sums[layer_idx] += vec
        count += 1
        print("Mean pass:", p)

    means = {}
    for layer_idx, vec in sums.items():
        means[layer_idx] = vec / count
    return means


def build_initial_delta_ranking(source_mean: Dict[int, torch.Tensor], target_mean: Dict[int, torch.Tensor]):
    delta = {}
    ranked = []
    for layer_idx in sorted(target_mean.keys()):
        d = target_mean[layer_idx] - source_mean[layer_idx]
        delta[layer_idx] = d
        for neuron_idx in range(d.shape[0]):
            score = float(d[neuron_idx].item())
            if ONLY_POSITIVE_DELTA and score <= 0:
                continue
            ranked.append((layer_idx, neuron_idx, score))
    ranked.sort(key=lambda x: abs(x[2]), reverse=True)
    return delta, ranked


def make_single_neuron_hook(neuron_idx: int, edit_value: float):
    def hook(module, inputs):
        x = inputs[0].clone()
        x[..., neuron_idx] = x[..., neuron_idx] + edit_value
        return (x,)
    return hook


@torch.inference_mode()
def get_gap_with_single_neuron_edit(
    model,
    tokenizer,
    image_processor,
    image_path: str,
    layer_idx: int,
    neuron_idx: int,
    edit_value: float,
    target_ids: List[int],
    source_ids: List[int],
) -> float:
    layers = get_llama_layers(model)
    h = layers[layer_idx].mlp.down_proj.register_forward_pre_hook(
        make_single_neuron_hook(neuron_idx, edit_value)
    )
    try:
        gap = get_cat_dog_gap(model, tokenizer, image_processor, image_path, target_ids, source_ids)
    finally:
        h.remove()
    return gap


def main():
    os.makedirs("analysis", exist_ok=True)

    print("Loading model...")
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

    print("Collecting source mean (dog)...")
    source_mean = collect_folder_mean(model, tokenizer, image_processor, SOURCE_DIR, MEAN_SOURCE_MAX)
    print("Collecting target mean (cat)...")
    target_mean = collect_folder_mean(model, tokenizer, image_processor, TARGET_DIR, MEAN_TARGET_MAX)

    delta, initial_ranked = build_initial_delta_ranking(source_mean, target_mean)
    shortlist = initial_ranked[:SHORTLIST_TOP_M]
    print(f"Initial positive neurons: {len(initial_ranked)}")
    print(f"Shortlist size for causal re-ranking: {len(shortlist)}")

    eval_paths = list_images(SOURCE_DIR)[:EVAL_IMAGES_MAX]
    if len(eval_paths) == 0:
        raise RuntimeError(f"No eval images found in {SOURCE_DIR}")

    baseline_gaps = {}
    print("Computing baseline cat-dog gaps on dog images...")
    for p in eval_paths:
        baseline_gaps[p] = get_cat_dog_gap(model, tokenizer, image_processor, p, target_ids, source_ids)
        print(f"Baseline gap {os.path.basename(p)}: {baseline_gaps[p]:.6f}")

    # normalize probe magnitude using shortlist delta scale so alpha is meaningful
    shortlist_abs = [abs(s) for _, _, s in shortlist]
    probe_scale = (sum(shortlist_abs) / max(len(shortlist_abs), 1)) + 1e-8

    causal_ranked = []
    print("Evaluating causal effect of shortlist neurons...")
    for idx, (layer_idx, neuron_idx, delta_value) in enumerate(shortlist, start=1):
        edit_value = PROBE_ALPHA * (delta_value / probe_scale)
        deltas = []
        for p in eval_paths:
            edited_gap = get_gap_with_single_neuron_edit(
                model,
                tokenizer,
                image_processor,
                p,
                layer_idx,
                neuron_idx,
                edit_value,
                target_ids,
                source_ids,
            )
            gap_delta = edited_gap - baseline_gaps[p]
            deltas.append(gap_delta)

        causal_score = float(sum(deltas) / len(deltas))
        causal_ranked.append({
            "layer": int(layer_idx),
            "neuron": int(neuron_idx),
            "delta_value": float(delta_value),
            "causal_score": causal_score,
            "probe_edit_value": float(edit_value),
        })

        if idx % 25 == 0 or idx == len(shortlist):
            print(f"  done {idx}/{len(shortlist)}")

    causal_ranked.sort(key=lambda x: x["causal_score"], reverse=True)

    out = {
        "source": SOURCE_WORD,
        "target": TARGET_WORD,
        "prompt": PROMPT,
        "layer_min": LAYER_MIN,
        "layer_max": LAYER_MAX,
        "mean_source_max": MEAN_SOURCE_MAX,
        "mean_target_max": MEAN_TARGET_MAX,
        "shortlist_top_m": SHORTLIST_TOP_M,
        "eval_images_max": EVAL_IMAGES_MAX,
        "probe_alpha": PROBE_ALPHA,
        "source_token_ids": source_ids,
        "target_token_ids": target_ids,
        "source_mean": source_mean,
        "target_mean": target_mean,
        "delta": delta,
        "initial_ranked": initial_ranked,
        "causal_ranked": causal_ranked,
        "selection_method": "rank neurons by average increase in [logit(cat)-logit(dog)] on dog images",
    }

    torch.save(out, SAVE_PATH)
    print(f"Saved causal neuron ranking to {SAVE_PATH}")
    print("Top 20 causal neurons:")
    for i, item in enumerate(causal_ranked[:20], start=1):
        print(
            i,
            f"Layer {item['layer']}, Neuron {item['neuron']}, "
            f"causal_score={item['causal_score']:.6f}, delta_value={item['delta_value']:.6f}"
        )


if __name__ == "__main__":
    main()
