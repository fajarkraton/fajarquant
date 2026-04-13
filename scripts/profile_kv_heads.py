#!/usr/bin/env python3
"""
profile_kv_heads.py — Per-head statistical profiler for FajarQuant v3.

Computes per-(layer, head, k/v) statistics: kurtosis, σ1/σ2 ratio,
outlier fraction, channel variance CV, and skewness. These statistics
drive the v3 strategy selector (which quantization path per head).

Usage:
    python scripts/profile_kv_heads.py --model google/gemma-4-E2B \
        --output data/calibration/profile_gemma.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats as scipy_stats


def load_wikitext2() -> str:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join([t for t in ds["text"] if t.strip()])


def compute_head_stats(data: np.ndarray) -> dict:
    """Compute profiling statistics for a single head's KV data.

    Args:
        data: shape (N, D) — N samples, D dimensions per head.

    Returns:
        dict with kurtosis, svd_ratio, outlier_frac, channel_var_cv, skewness, mean_abs.
    """
    N, D = data.shape
    if N < 4 or D < 2:
        return {
            "kurtosis": 0.0,
            "svd_ratio": 1.0,
            "outlier_frac": 0.0,
            "channel_var_cv": 0.0,
            "skewness": 0.0,
            "mean_abs": float(np.mean(np.abs(data))),
        }

    # Per-channel variance
    ch_var = np.var(data, axis=0, ddof=1)
    mean_var = np.mean(ch_var)
    std_var = np.std(ch_var, ddof=0)
    channel_var_cv = float(std_var / mean_var) if mean_var > 1e-15 else 0.0

    # Excess kurtosis (averaged over channels)
    ch_kurt = scipy_stats.kurtosis(data, axis=0, fisher=True)
    kurtosis = float(np.mean(ch_kurt))

    # Skewness (averaged over channels)
    ch_skew = scipy_stats.skew(data, axis=0)
    skewness = float(np.mean(np.abs(ch_skew)))

    # SVD ratio (σ1/σ2)
    try:
        cov = np.cov(data.T)
        eigvals = np.sort(np.abs(np.linalg.eigvalsh(cov)))[::-1]
        s1 = np.sqrt(max(eigvals[0], 0))
        s2 = np.sqrt(max(eigvals[1], 0)) if len(eigvals) > 1 else 1e-15
        svd_ratio = float(s1 / s2) if s2 > 1e-15 else 999.0
    except Exception:
        svd_ratio = 1.0

    # Outlier fraction: channels with variance > 3x median variance
    median_var = np.median(ch_var)
    outlier_frac = float(np.mean(ch_var > 3 * median_var)) if median_var > 1e-15 else 0.0

    return {
        "kurtosis": round(kurtosis, 4),
        "svd_ratio": round(svd_ratio, 4),
        "outlier_frac": round(outlier_frac, 4),
        "channel_var_cv": round(channel_var_cv, 4),
        "skewness": round(skewness, 4),
        "mean_abs": round(float(np.mean(np.abs(data))), 6),
    }


def profile_model(
    model, tokenizer, text: str, n_samples: int = 64, seq_len: int = 2048
) -> dict:
    """Run calibration forward passes and compute per-head stats."""
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        from transformers import DynamicCache

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    total_len = input_ids.size(1)
    n_chunks = min(total_len // seq_len, n_samples)

    print(f"Profiling {n_chunks} chunks × {seq_len} tokens...")

    # First pass: determine architecture
    chunk0 = input_ids[:, :seq_len]
    with torch.no_grad():
        cache = DynamicCache()
        model(chunk0, past_key_values=cache, use_cache=True)

    n_layers = len(cache.layers)
    # Detect per-layer head count and dim
    layer_info = []
    for li in range(n_layers):
        k = cache.layers[li].keys.squeeze(0)  # (H, S, D)
        H, S, D = k.shape
        layer_info.append({"n_heads": H, "head_dim": D})
    del cache
    torch.cuda.empty_cache()

    print(f"  {n_layers} layers, heads={layer_info[0]['n_heads']}, dim={layer_info[0]['head_dim']}")

    # Collect per-head data (streaming: accumulate samples)
    # For memory: store at most 4096 samples per head
    MAX_SAMPLES = 4096
    head_data = {}  # (layer, head, 'k'|'v') → list of (S, D) arrays

    for i in range(n_chunks):
        chunk = input_ids[:, i * seq_len : (i + 1) * seq_len]
        with torch.no_grad():
            cache = DynamicCache()
            model(chunk, past_key_values=cache, use_cache=True)

        for li in range(n_layers):
            k = cache.layers[li].keys.squeeze(0).float().cpu().numpy()  # (H, S, D)
            v = cache.layers[li].values.squeeze(0).float().cpu().numpy()
            H = k.shape[0]

            for h in range(H):
                for kv_type, arr in [("k", k), ("v", v)]:
                    key = (li, h, kv_type)
                    if key not in head_data:
                        head_data[key] = []
                    head_samples = arr[h]  # (S, D)
                    head_data[key].append(head_samples)

        del cache
        torch.cuda.empty_cache()

        if (i + 1) % 8 == 0 or i == n_chunks - 1:
            print(f"  Collected {i + 1}/{n_chunks} chunks")

    # Compute stats
    print("Computing per-head statistics...")
    result = {
        "_model": str(model.config._name_or_path),
        "_n_layers": n_layers,
        "_n_heads": layer_info[0]["n_heads"],
        "_head_dim": layer_info[0]["head_dim"],
        "_n_chunks": n_chunks,
        "_seq_len": seq_len,
        "layers": [],
    }

    for li in range(n_layers):
        H = layer_info[li]["n_heads"]
        layer_entry = {"heads": []}
        for h in range(H):
            head_entry = {}
            for kv_type in ["k", "v"]:
                key = (li, h, kv_type)
                if key in head_data and head_data[key]:
                    all_samples = np.concatenate(head_data[key], axis=0)
                    # Subsample if too large
                    if all_samples.shape[0] > MAX_SAMPLES:
                        idx = np.random.choice(
                            all_samples.shape[0], MAX_SAMPLES, replace=False
                        )
                        all_samples = all_samples[idx]
                    head_entry[kv_type] = compute_head_stats(all_samples)
                else:
                    head_entry[kv_type] = compute_head_stats(
                        np.zeros((1, layer_info[li]["head_dim"]))
                    )
            layer_entry["heads"].append(head_entry)
        result["layers"].append(layer_entry)

    return result


def main():
    parser = argparse.ArgumentParser(description="FajarQuant v3 per-head profiler")
    parser.add_argument("--model", default="google/gemma-4-E2B")
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--token", default=None)
    parser.add_argument(
        "--output", default="data/calibration/profile_result.json"
    )
    args = parser.parse_args()

    print(f"Loading WikiText-2 test set...")
    text = load_wikitext2()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, token=args.token, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, token=args.token, dtype=torch.float16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()

    t0 = time.time()
    profile = profile_model(model, tokenizer, text, args.n_samples, args.seq_len)
    elapsed = time.time() - t0

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(profile, f, indent=2)

    n_heads_total = sum(
        len(layer["heads"]) for layer in profile["layers"]
    )
    print(f"\nDone in {elapsed:.1f}s — {n_heads_total} head profiles saved to {args.output}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Profile Summary: {args.model}")
    print(f"{'='*60}")
    for li, layer in enumerate(profile["layers"][:3]):
        for hi, head in enumerate(layer["heads"][:2]):
            k = head["k"]
            print(
                f"  L{li}H{hi} keys: kurt={k['kurtosis']:.1f} svd={k['svd_ratio']:.1f} "
                f"outlier={k['outlier_frac']:.2f} cv={k['channel_var_cv']:.2f} "
                f"skew={k['skewness']:.2f}"
            )
    if len(profile["layers"]) > 3:
        print(f"  ... ({len(profile['layers'])} layers total)")


if __name__ == "__main__":
    main()
