#!/usr/bin/env python3
"""
calibrate_fq_v2.py — Offline calibration for FajarQuant v2 (F2-D).

Collects per-layer K and V tensors from a calibration dataset, identifies
outlier channels by variance, computes PCA on clean (non-outlier) data,
and saves the result as a .npz file for use by quant_attention_v2.py.

Decision provenance: V26 C1.6 Path B, Phase B2.D → F2-D design.
Algorithm spec: docs/V26_C1_6_V2_DESIGN.md

Usage:
    python scripts/calibrate_fq_v2.py --model google/gemma-4-E2B
    python scripts/calibrate_fq_v2.py --model mistralai/Mistral-7B-v0.1
    python scripts/calibrate_fq_v2.py --model Qwen/Qwen2-7B
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def collect_streaming_stats(
    model,
    tokenizer,
    text: str,
    n_samples: int = 128,
    seq_len: int = 2048,
) -> tuple[list[dict], int, int]:
    """Forward n_samples chunks, accumulating per-layer statistics in
    O(D^2) memory instead of storing all raw K/V tensors.

    For each layer, accumulates:
      - count: number of (H*S) samples seen
      - sum_x: sum of samples, shape (D,)
      - sum_x2: sum of outer products (for covariance), shape (D, D)
      - var_acc: per-channel sum of squares for variance, shape (D,)

    Returns (layer_stats, n_layers, head_dim).
    Memory: O(n_layers * D^2) ≈ 32 * 128^2 * 4 bytes ≈ 2 MB for Mistral.
    """
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        from transformers import DynamicCache

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    total_len = input_ids.size(1)
    n_chunks = min(total_len // seq_len, n_samples)

    if n_chunks == 0:
        raise ValueError(
            f"text too short ({total_len} tokens) for {n_samples} × {seq_len}"
        )

    layer_stats: list[dict] = []
    n_layers_out = 0
    head_dim_out = 0

    for i in range(n_chunks):
        chunk = input_ids[:, i * seq_len : (i + 1) * seq_len]

        with torch.no_grad():
            cache = DynamicCache()
            model(chunk, past_key_values=cache, use_cache=True)

        n_layers = len(cache.layers)
        if i == 0:
            n_layers_out = n_layers
            # Detect head_dim from first layer
            k0 = cache.layers[0].keys.squeeze(0)
            head_dim_out = k0.shape[-1]
            D = head_dim_out
            for _ in range(n_layers):
                layer_stats.append({
                    "k_count": 0,
                    "k_sum": np.zeros(D, dtype=np.float64),
                    "k_sum_sq": np.zeros(D, dtype=np.float64),
                    "k_cov_acc": np.zeros((D, D), dtype=np.float64),
                    "v_count": 0,
                    "v_sum": np.zeros(D, dtype=np.float64),
                    "v_sum_sq": np.zeros(D, dtype=np.float64),
                    "v_cov_acc": np.zeros((D, D), dtype=np.float64),
                })

        for layer_idx in range(n_layers):
            layer = cache.layers[layer_idx]
            k = layer.keys.squeeze(0).float().cpu()   # (H, S, D)
            v = layer.values.squeeze(0).float().cpu()

            k_flat = k.reshape(-1, k.shape[-1]).numpy().astype(np.float64)
            v_flat = v.reshape(-1, v.shape[-1]).numpy().astype(np.float64)
            n = k_flat.shape[0]

            st = layer_stats[layer_idx]
            st["k_count"] += n
            st["k_sum"] += k_flat.sum(axis=0)
            st["k_sum_sq"] += (k_flat ** 2).sum(axis=0)
            st["k_cov_acc"] += k_flat.T @ k_flat

            st["v_count"] += n
            st["v_sum"] += v_flat.sum(axis=0)
            st["v_sum_sq"] += (v_flat ** 2).sum(axis=0)
            st["v_cov_acc"] += v_flat.T @ v_flat

        del cache
        torch.cuda.empty_cache()

        if (i + 1) % 16 == 0 or i == n_chunks - 1:
            print(f"  Collected {i + 1}/{n_chunks} chunks")

    return layer_stats, n_layers_out, head_dim_out


def compute_calibration_from_stats(
    stats: dict,
    kv_type: str,
    outlier_percentile: float = 99.0,
) -> dict[str, np.ndarray]:
    """Compute F2-D calibration from streaming statistics.

    Args:
        stats: accumulated statistics dict for one layer
        kv_type: "k" or "v"
        outlier_percentile: percentile threshold for outlier channels

    Returns dict with:
        outlier_mask: bool array (D,) — True for outlier channels
        pca_rotation: float32 array (D', D') — PCA rotation matrix
        pca_mean: float32 array (D',) — mean of clean data
        per_channel_var: float32 array (D,) — variance of each channel
    """
    prefix = kv_type
    N = stats[f"{prefix}_count"]
    sum_x = stats[f"{prefix}_sum"]
    sum_x2 = stats[f"{prefix}_sum_sq"]
    cov_acc = stats[f"{prefix}_cov_acc"]

    D = len(sum_x)
    mean_all = sum_x / N  # (D,)

    # Step 1: per-channel variance = E[x^2] - E[x]^2
    var_per_channel = (sum_x2 / N - mean_all ** 2).astype(np.float32)

    # Step 2: identify outlier channels (top 1%)
    threshold = np.percentile(var_per_channel, outlier_percentile)
    outlier_mask = var_per_channel >= threshold

    # Step 3: remove outlier channels — recompute stats on clean subset
    clean_idx = np.where(~outlier_mask)[0]
    D_clean = len(clean_idx)

    mean_clean = mean_all[clean_idx].astype(np.float64)

    # Covariance on clean channels:
    # cov = (sum(x_i @ x_i.T) / N) - mean @ mean.T
    # Extract clean submatrix from cov_acc
    cov_clean_acc = cov_acc[np.ix_(clean_idx, clean_idx)]
    cov_clean = (cov_clean_acc / N) - np.outer(mean_clean, mean_clean)

    # Step 4: eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(cov_clean)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    R = eigvecs.T.astype(np.float32)

    return {
        "outlier_mask": outlier_mask.astype(np.bool_),
        "pca_rotation": R,
        "pca_mean": mean_clean.astype(np.float32),
        "per_channel_var": var_per_channel,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate FajarQuant v2 (F2-D) per-layer PCA + outlier channels"
    )
    parser.add_argument("--model", default="google/gemma-4-E2B")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=128,
        help="Number of calibration chunks (default: 128)",
    )
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument(
        "--outlier-percentile",
        type=float,
        default=99.0,
        help="Percentile threshold for outlier channels (default: 99 = top 1%%)",
    )
    parser.add_argument("--token", default=None)
    parser.add_argument(
        "--output",
        default=None,
        help="Output .npz path (default: data/calibration/fq_v2_{model_short}.npz)",
    )
    args = parser.parse_args()

    # Derive short model name for default output path
    model_short = args.model.split("/")[-1].lower().replace("-", "_")
    if args.output is None:
        args.output = f"data/calibration/fq_v2_{model_short}.npz"

    print(f"FajarQuant v2 calibration (F2-D)")
    print(f"  Model: {args.model}")
    print(f"  Samples: {args.n_samples} × {args.seq_len} tokens")
    print(f"  Outlier percentile: {args.outlier_percentile}")
    print(f"  Output: {args.output}")

    # Load dataset
    print("\nLoading WikiText-2 calibration set...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, token=args.token, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        token=args.token,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded on {model.device}")

    # Collect K/V statistics (streaming — O(D^2) memory per layer)
    print(f"\nCollecting K/V statistics from {args.n_samples} chunks (streaming)...")
    t0 = time.time()
    layer_stats, n_layers, head_dim = collect_streaming_stats(
        model, tokenizer, text, n_samples=args.n_samples, seq_len=args.seq_len
    )
    collect_time = time.time() - t0
    n_total = layer_stats[0]["k_count"]
    print(
        f"  Collected {n_layers} layers, head_dim={head_dim}, "
        f"{n_total} samples/layer, {collect_time:.1f}s"
    )

    # Free model GPU memory
    del model
    torch.cuda.empty_cache()

    # Compute calibration per layer from streaming stats
    print(f"\nComputing calibration (outlier identification + PCA)...")
    t0 = time.time()

    save_dict: dict[str, np.ndarray] = {
        "_model": np.array(args.model),
        "_n_samples": np.array(args.n_samples),
        "_seq_len": np.array(args.seq_len),
        "_outlier_percentile": np.array(args.outlier_percentile),
        "_n_layers": np.array(n_layers),
        "_head_dim": np.array(head_dim),
    }

    total_outliers_k = 0
    total_outliers_v = 0

    for layer_idx in range(n_layers):
        st = layer_stats[layer_idx]

        cal_k = compute_calibration_from_stats(
            st, "k", outlier_percentile=args.outlier_percentile
        )
        save_dict[f"k_{layer_idx}_outlier_mask"] = cal_k["outlier_mask"]
        save_dict[f"k_{layer_idx}_pca_rotation"] = cal_k["pca_rotation"]
        save_dict[f"k_{layer_idx}_pca_mean"] = cal_k["pca_mean"]
        save_dict[f"k_{layer_idx}_channel_var"] = cal_k["per_channel_var"]
        total_outliers_k += cal_k["outlier_mask"].sum()

        cal_v = compute_calibration_from_stats(
            st, "v", outlier_percentile=args.outlier_percentile
        )
        save_dict[f"v_{layer_idx}_outlier_mask"] = cal_v["outlier_mask"]
        save_dict[f"v_{layer_idx}_pca_rotation"] = cal_v["pca_rotation"]
        save_dict[f"v_{layer_idx}_pca_mean"] = cal_v["pca_mean"]
        save_dict[f"v_{layer_idx}_channel_var"] = cal_v["per_channel_var"]
        total_outliers_v += cal_v["outlier_mask"].sum()

        if (layer_idx + 1) % 8 == 0 or layer_idx == n_layers - 1:
            print(
                f"  Layer {layer_idx + 1}/{n_layers}: "
                f"K outliers={cal_k['outlier_mask'].sum()}/{head_dim}, "
                f"V outliers={cal_v['outlier_mask'].sum()}/{head_dim}"
            )

    cal_time = time.time() - t0

    # Save
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(args.output, **save_dict)
    file_size = os.path.getsize(args.output)

    print(f"\nCalibration complete ({cal_time:.1f}s)")
    print(f"  Total K outlier channels: {total_outliers_k} across {n_layers} layers")
    print(f"  Total V outlier channels: {total_outliers_v} across {n_layers} layers")
    print(f"  Avg K outliers/layer: {total_outliers_k / n_layers:.1f}/{head_dim}")
    print(f"  Avg V outliers/layer: {total_outliers_v / n_layers:.1f}/{head_dim}")
    print(f"  Saved to {args.output} ({file_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
