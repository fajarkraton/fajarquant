#!/usr/bin/env python3
"""
tune_thresholds.py — B-fix.2: Auto-tune v3 strategy selector thresholds
using real KV cache reconstruction MSE.

Replaces the original heuristic scoring (path-count-based) with actual
quantize→dequantize MSE on real model KV cache data.

Pipeline:
  1. Load model, forward pass N chunks → extract real KV cache per layer/head
  2. 80/20 train/test split of chunks
  3. For each threshold combo in grid:
     a. Assign strategies per head using threshold-based decision tree
     b. Apply each head's assigned path, compute reconstruction MSE
     c. Sum MSE across all heads (train set)
  4. Select thresholds with lowest train MSE
  5. Evaluate on test set, report both scores

Usage:
    python scripts/tune_thresholds.py \
        --model google/gemma-4-E2B \
        --profile data/calibration/profile_gemma_4_e2b.json \
        --calibration data/calibration/fq_v2_gemma_4_e2b.npz \
        --bits 2 \
        --n-chunks 5 \
        --output data/calibration/thresholds_gemma_2bit.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from strategy_selector import select_strategy, DEFAULT_THRESHOLDS
from quant_attention import (
    apply_kivi_keys_4d,
    apply_kivi_values_4d,
    apply_turboquant_4d_with_outliers,
)
from quant_attention_v2 import load_calibration
from quant_attention_v3 import _path_b_pca
from cache_compat import get_cache_n_layers, get_cache_kv


# Grid values — calibrated to real profile data ranges (B-fix.1)
GRID = {
    "T_cv": [0.5, 1.0, 1.5, 2.0, 3.0],
    "T_svd": [1.2, 1.5, 2.0, 3.0, 5.0],
    "T_kurt": [0.5, 1.0, 2.0, 3.0, 5.0],
    "T_skew": [0.1, 0.2, 0.3, 0.5, 1.0],
    "T_kurt_residual": [0.5, 1.0, 2.0],
}


def load_wikitext2() -> str:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join([t for t in ds["text"] if t.strip()])


def extract_kv_cache(model, tokenizer, text: str, n_chunks: int = 5,
                     seq_len: int = 2048) -> list[list[dict]]:
    """Forward n_chunks through model, extract KV cache per layer.

    Returns: list of n_chunks items, each a list of n_layers dicts
      {"k": Tensor(H, S, D), "v": Tensor(H, S, D)}
    """
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        from transformers import DynamicCache

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    total_len = input_ids.size(1)
    n_avail = total_len // seq_len
    n_chunks = min(n_chunks, n_avail)

    all_chunks = []
    for i in range(n_chunks):
        chunk = input_ids[:, i * seq_len : (i + 1) * seq_len]
        with torch.no_grad():
            cache = DynamicCache()
            model(chunk, past_key_values=cache, use_cache=True)

        layers_kv = []
        for li in range(get_cache_n_layers(cache)):
            k_raw, v_raw = get_cache_kv(cache, li)
            k = k_raw.squeeze(0).float().cpu()   # (H, S, D)
            v = v_raw.squeeze(0).float().cpu()
            layers_kv.append({"k": k, "v": v})
        all_chunks.append(layers_kv)

        del cache
        torch.cuda.empty_cache()

    return all_chunks


def _apply_path_with_cal(
    data_4d: torch.Tensor, path: str, bits: int, is_key: bool,
    layer_idx: int, cal: dict | None,
) -> torch.Tensor:
    """Apply a quantization path with calibration support (for Path B)."""
    if path == "A":
        if is_key:
            return apply_kivi_keys_4d(data_4d, bits)
        return apply_kivi_values_4d(data_4d, bits)
    elif path == "B":
        return _path_b_pca(data_4d, bits, layer_idx, is_key, cal)
    elif path == "C":
        return apply_turboquant_4d_with_outliers(data_4d, bits)
    elif path == "D":
        B, H, S, D = data_4d.shape
        max_q = (1 << (bits - 1)) - 1
        scale1 = data_4d.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / max_q
        q1 = (data_4d / scale1).round().clamp(-max_q, max_q)
        r1 = q1 * scale1
        residual = data_4d - r1
        scale2 = residual.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / max_q
        q2 = (residual / scale2).round().clamp(-max_q, max_q)
        return r1 + q2 * scale2
    elif path == "E":
        B, H, S, D = data_4d.shape
        max_q = (1 << bits) - 1
        result = data_4d.clone()
        for h in range(H):
            head = data_4d[:, h, :, :]
            ch_min = head.amin(dim=(0, 1))
            ch_max = head.amax(dim=(0, 1))
            scale = (ch_max - ch_min).clamp(min=1e-8) / max_q
            zp = ch_min
            q = ((head - zp) / scale).round().clamp(0, max_q)
            result[:, h, :, :] = q * scale + zp
        return result
    else:
        if is_key:
            return apply_kivi_keys_4d(data_4d, bits)
        return apply_kivi_values_4d(data_4d, bits)


def compute_mse_for_thresholds(
    chunks: list[list[dict]],
    profile: dict,
    bits: int,
    thresholds: dict,
    cal: dict | None,
    device: str = "cuda",
) -> float:
    """Compute total reconstruction MSE for a threshold combination.

    For each chunk × layer × head × kv_type:
      1. Look up head's profile stats
      2. Assign strategy via threshold tree
      3. Apply path, compute MSE vs original
      4. Sum all MSE
    """
    n_kv_heads = profile.get("_n_heads", 1)
    total_mse = 0.0
    n_evals = 0

    for chunk_kv in chunks:
        for li, layer_kv in enumerate(chunk_kv):
            if li >= len(profile["layers"]):
                continue
            layer_prof = profile["layers"][li]

            for kv_type in ["k", "v"]:
                original = layer_kv[kv_type]  # (H, S, D)
                H = original.shape[0]

                for hi in range(H):
                    if hi >= len(layer_prof["heads"]):
                        continue
                    head_stats = layer_prof["heads"][hi].get(kv_type, {})
                    path = select_strategy(head_stats, bits, n_kv_heads, thresholds)

                    # Extract single head as 4D: (1, 1, S, D)
                    head_data = original[hi:hi+1].unsqueeze(0).to(device)
                    is_key = (kv_type == "k")

                    reconstructed = _apply_path_with_cal(
                        head_data, path, bits, is_key, li, cal
                    )
                    mse = float(((head_data - reconstructed) ** 2).mean())
                    total_mse += mse
                    n_evals += 1

    return total_mse / max(n_evals, 1)


def grid_search(
    train_chunks: list[list[dict]],
    test_chunks: list[list[dict]],
    profile: dict,
    bits: int,
    cal: dict | None,
    device: str = "cuda",
) -> tuple[dict, float, float]:
    """Find optimal thresholds via exhaustive grid search on train set.

    Returns (best_thresholds, train_mse, test_mse).
    """
    # Build grid — include T_kurt_residual for 2-bit
    grid_keys = ["T_cv", "T_svd", "T_kurt", "T_skew"]
    grid_vals = [GRID[k] for k in grid_keys]
    if bits <= 2:
        grid_keys.append("T_kurt_residual")
        grid_vals.append(GRID["T_kurt_residual"])

    combos = list(itertools.product(*grid_vals))
    print(f"Grid search: {len(combos)} combinations × {len(train_chunks)} train chunks...")

    best_thresholds = dict(DEFAULT_THRESHOLDS)
    best_train_mse = compute_mse_for_thresholds(
        train_chunks, profile, bits, best_thresholds, cal, device
    )
    print(f"  Default thresholds MSE: {best_train_mse:.6f}")

    for ci, combo in enumerate(combos):
        thresholds = dict(zip(grid_keys, combo))
        # Fill in defaults for keys not in grid
        for k, v in DEFAULT_THRESHOLDS.items():
            if k not in thresholds:
                thresholds[k] = v

        train_mse = compute_mse_for_thresholds(
            train_chunks, profile, bits, thresholds, cal, device
        )

        if train_mse < best_train_mse:
            best_train_mse = train_mse
            best_thresholds = thresholds
            print(f"  [{ci+1}/{len(combos)}] New best: MSE={train_mse:.6f} thresholds={thresholds}")

        if (ci + 1) % 200 == 0:
            print(f"  [{ci+1}/{len(combos)}] searched... best so far: {best_train_mse:.6f}")

    # Evaluate best thresholds on test set
    test_mse = compute_mse_for_thresholds(
        test_chunks, profile, bits, best_thresholds, cal, device
    )

    return best_thresholds, best_train_mse, test_mse


def main():
    parser = argparse.ArgumentParser(description="FajarQuant v3 threshold auto-tuner (B-fix.2)")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--profile", required=True, help="Profile JSON from profile_kv_heads.py")
    parser.add_argument("--calibration", default=None, help="v2 calibration .npz (for Path B)")
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--n-chunks", type=int, default=5, help="Total chunks to extract (80/20 split)")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--token", default=None)
    parser.add_argument("--output", default="data/calibration/thresholds_result.json")
    args = parser.parse_args()

    print(f"=== B-fix.2: Real MSE threshold tuner ===")
    print(f"Model: {args.model}")
    print(f"Bits: {args.bits}")
    print(f"Chunks: {args.n_chunks} (train: {int(args.n_chunks * 0.8)}, test: {args.n_chunks - int(args.n_chunks * 0.8)})")
    print()

    # Load profile
    with open(args.profile) as f:
        profile = json.load(f)

    # Load calibration (for Path B)
    cal = None
    if args.calibration and os.path.exists(args.calibration):
        cal = load_calibration(args.calibration)
        print(f"Loaded calibration: {args.calibration} ({cal['n_layers']} layers)")

    # Load model and extract KV cache
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, token=args.token, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, token=args.token, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True,
    ).eval()

    text = load_wikitext2()
    print(f"Extracting {args.n_chunks} KV cache chunks...")
    t0 = time.time()
    all_chunks = extract_kv_cache(model, tokenizer, text, args.n_chunks, args.seq_len)
    print(f"  Extracted in {time.time()-t0:.1f}s")

    # Free model memory
    del model
    torch.cuda.empty_cache()

    # 80/20 split
    n_train = max(1, int(len(all_chunks) * 0.8))
    train_chunks = all_chunks[:n_train]
    test_chunks = all_chunks[n_train:] if n_train < len(all_chunks) else all_chunks[:1]
    print(f"Split: {len(train_chunks)} train, {len(test_chunks)} test chunks")

    # Get default baseline
    from strategy_selector import assign_strategies
    default_assignment = assign_strategies(profile, args.bits)
    print(f"\nDefault thresholds: {DEFAULT_THRESHOLDS}")
    print(f"Default path counts: {default_assignment['path_counts']}")

    # Grid search with real MSE
    print(f"\nStarting grid search...")
    t0 = time.time()
    best_thresholds, train_mse, test_mse = grid_search(
        train_chunks, test_chunks, profile, args.bits, cal,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    elapsed = time.time() - t0

    # Get final assignment with optimal thresholds
    optimal_assignment = assign_strategies(profile, args.bits, best_thresholds)

    print(f"\n{'=' * 60}")
    print(f"Results ({elapsed:.1f}s)")
    print(f"{'=' * 60}")
    print(f"Optimal thresholds: {best_thresholds}")
    print(f"Train MSE: {train_mse:.6f}")
    print(f"Test MSE:  {test_mse:.6f}")
    print(f"Overfit ratio: {test_mse / train_mse:.2f}x" if train_mse > 0 else "N/A")
    print(f"Path counts: {optimal_assignment['path_counts']} ({optimal_assignment['n_unique_paths']} unique)")
    print(f"Default path counts were: {default_assignment['path_counts']}")

    # Save
    result = {
        "_model": profile.get("_model", "unknown"),
        "_bits": args.bits,
        "_n_kv_heads": profile.get("_n_heads", 1),
        "_method": "real_mse_grid_search",
        "_n_train_chunks": len(train_chunks),
        "_n_test_chunks": len(test_chunks),
        "default_thresholds": DEFAULT_THRESHOLDS,
        "default_mse": float(compute_mse_for_thresholds(
            train_chunks, profile, args.bits, DEFAULT_THRESHOLDS, cal,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )),
        "optimal_thresholds": best_thresholds,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "overfit_ratio": round(test_mse / train_mse, 4) if train_mse > 0 else None,
        "optimal_path_counts": optimal_assignment["path_counts"],
        "default_path_counts": default_assignment["path_counts"],
        "grid_size": len(list(itertools.product(*[GRID[k] for k in ["T_cv", "T_svd", "T_kurt", "T_skew"]]))),
        "elapsed_seconds": round(elapsed, 1),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
