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


def collect_kv_tensors(
    model,
    tokenizer,
    text: str,
    n_samples: int = 128,
    seq_len: int = 2048,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Forward n_samples chunks through the model, collecting per-layer
    K and V tensors from the DynamicCache.

    Returns (all_keys, all_values) where each is a list of length
    n_layers, and each element is a tensor of shape
    (n_samples * num_kv_heads * seq_len, head_dim).
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

    # Storage: per-layer list of (num_kv_heads * seq_len, head_dim) tensors
    per_layer_keys: list[list[torch.Tensor]] = []
    per_layer_vals: list[list[torch.Tensor]] = []

    for i in range(n_chunks):
        chunk = input_ids[:, i * seq_len : (i + 1) * seq_len]

        with torch.no_grad():
            cache = DynamicCache()
            model(chunk, past_key_values=cache, use_cache=True)

        n_layers = len(cache.layers)
        if i == 0:
            per_layer_keys = [[] for _ in range(n_layers)]
            per_layer_vals = [[] for _ in range(n_layers)]

        for layer_idx in range(n_layers):
            layer = cache.layers[layer_idx]
            # .keys shape: (1, num_kv_heads, seq_len, head_dim)
            k = layer.keys.squeeze(0).float().cpu()  # (H, S, D)
            v = layer.values.squeeze(0).float().cpu()
            # Flatten to (H*S, D)
            per_layer_keys[layer_idx].append(k.reshape(-1, k.shape[-1]))
            per_layer_vals[layer_idx].append(v.reshape(-1, v.shape[-1]))

        del cache
        torch.cuda.empty_cache()

        if (i + 1) % 16 == 0 or i == n_chunks - 1:
            print(f"  Collected {i + 1}/{n_chunks} chunks")

    # Concatenate per layer
    all_keys = [torch.cat(chunks, dim=0) for chunks in per_layer_keys]
    all_vals = [torch.cat(chunks, dim=0) for chunks in per_layer_vals]

    return all_keys, all_vals


def compute_calibration(
    data: torch.Tensor,
    outlier_percentile: float = 99.0,
) -> dict[str, np.ndarray]:
    """Compute F2-D calibration for a single layer's K or V data.

    Args:
        data: shape (N, D) — concatenated K or V across all samples
        outlier_percentile: percentile threshold for outlier channels

    Returns dict with:
        outlier_mask: bool array (D,) — True for outlier channels
        pca_rotation: float32 array (D', D') — PCA rotation matrix
        pca_mean: float32 array (D',) — mean of clean data
        per_channel_var: float32 array (D,) — variance of each channel
    """
    N, D = data.shape

    # Step 1: per-channel variance
    var_per_channel = data.var(dim=0).numpy()  # (D,)

    # Step 2: identify outlier channels (top 1%)
    threshold = np.percentile(var_per_channel, outlier_percentile)
    outlier_mask = var_per_channel >= threshold  # (D,)
    n_outlier = outlier_mask.sum()

    # Step 3: remove outlier channels
    clean_mask = ~outlier_mask
    data_clean = data[:, clean_mask].numpy()  # (N, D')
    D_clean = data_clean.shape[1]

    # Step 4: PCA on clean data
    mean = data_clean.mean(axis=0)  # (D',)
    centered = data_clean - mean  # (N, D')

    # Use SVD for numerical stability (better than cov + eigh for large N)
    # Only need the right singular vectors (eigenvectors of cov)
    # cov = (centered.T @ centered) / (N-1), eigvecs of cov = V from SVD
    # For memory efficiency with large N, compute cov explicitly
    if N > 10000:
        cov = (centered.T @ centered) / max(N - 1, 1)  # (D', D')
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
    else:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        eigvecs = Vt.T  # columns are eigenvectors, already sorted desc

    # R has eigenvectors as rows: (D', D')
    R = eigvecs.T.astype(np.float32)

    return {
        "outlier_mask": outlier_mask.astype(np.bool_),
        "pca_rotation": R,
        "pca_mean": mean.astype(np.float32),
        "per_channel_var": var_per_channel.astype(np.float32),
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

    # Collect K/V tensors
    print(f"\nCollecting K/V from {args.n_samples} chunks...")
    t0 = time.time()
    all_keys, all_vals = collect_kv_tensors(
        model, tokenizer, text, n_samples=args.n_samples, seq_len=args.seq_len
    )
    collect_time = time.time() - t0
    n_layers = len(all_keys)
    head_dim = all_keys[0].shape[1]
    print(
        f"  Collected {n_layers} layers, head_dim={head_dim}, "
        f"{all_keys[0].shape[0]} samples/layer, {collect_time:.1f}s"
    )

    # Free model GPU memory — we only need the CPU tensors now
    del model
    torch.cuda.empty_cache()

    # Compute calibration per layer
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
        # Keys
        cal_k = compute_calibration(
            all_keys[layer_idx], outlier_percentile=args.outlier_percentile
        )
        save_dict[f"k_{layer_idx}_outlier_mask"] = cal_k["outlier_mask"]
        save_dict[f"k_{layer_idx}_pca_rotation"] = cal_k["pca_rotation"]
        save_dict[f"k_{layer_idx}_pca_mean"] = cal_k["pca_mean"]
        save_dict[f"k_{layer_idx}_channel_var"] = cal_k["per_channel_var"]
        total_outliers_k += cal_k["outlier_mask"].sum()

        # Values
        cal_v = compute_calibration(
            all_vals[layer_idx], outlier_percentile=args.outlier_percentile
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
