#!/usr/bin/env python3
"""
FajarQuant — 3-Way Comparison on Real Gemma 4 KV Cache Data

Compares quantization MSE across three methods:
1. FajarQuant (adaptive PCA rotation + coordinate-wise quantization)
2. KIVI (per-channel keys / per-token values)
3. TurboQuant (random rotation + coordinate-wise quantization)

Usage:
    python scripts/run_comparison.py --data data/kv_cache/
"""

import argparse
import json
import os
import time

import numpy as np
from scipy.stats import ortho_group


# ═══════════════════════════════════════════════════════════════
# TurboQuant: Random rotation + uniform scalar quantization
# ═══════════════════════════════════════════════════════════════

def uniform_quantize(x: np.ndarray, bits: int) -> tuple[np.ndarray, float, float]:
    """Uniform scalar quantization. Returns (indices, scale, zero)."""
    levels = (1 << bits) - 1
    mn, mx = x.min(), x.max()
    rng = mx - mn
    scale = rng / levels if rng > 1e-15 else 1.0
    indices = np.clip(np.round((x - mn) / scale), 0, levels).astype(np.uint8)
    return indices, scale, mn


def uniform_dequantize(indices: np.ndarray, scale: float, zero: float) -> np.ndarray:
    return zero + indices.astype(np.float64) * scale


def turboquant_mse(data: np.ndarray, bits: int, seed: int = 42) -> float:
    """TurboQuant: random orthogonal rotation + per-coordinate quantization."""
    seq_len, d = data.shape
    rng = np.random.default_rng(seed)
    R = ortho_group.rvs(d, random_state=rng)

    # Rotate ALL data
    rotated_data = data @ R.T  # (seq_len, d) in rotated space

    # Per-coordinate quantization in rotated space
    levels = (1 << bits) - 1
    recon_rotated = np.zeros_like(rotated_data)
    for ch in range(d):
        col = rotated_data[:, ch]
        mn, mx = col.min(), col.max()
        rng_val = mx - mn
        scale = rng_val / levels if rng_val > 1e-15 else 1.0
        indices = np.clip(np.round((col - mn) / scale), 0, levels)
        recon_rotated[:, ch] = mn + indices * scale

    recon = recon_rotated @ R
    return np.mean((data - recon) ** 2)


# ═══════════════════════════════════════════════════════════════
# FajarQuant: Adaptive PCA rotation + coordinate-wise quantization
# ═══════════════════════════════════════════════════════════════

def fajarquant_mse(data: np.ndarray, bits: int) -> float:
    """FajarQuant: PCA rotation + per-coordinate quantization (adaptive).

    Key insight: after PCA rotation, each coordinate has different variance.
    Per-coordinate quantization (like KIVI per-channel, but in PCA space)
    allocates scale/zero independently per rotated dimension, which is
    optimal when principal components have very different magnitudes.
    """
    seq_len, d = data.shape

    # Compute PCA rotation from data (calibration = self)
    mean = data.mean(axis=0)
    centered = data - mean
    cov = (centered.T @ centered) / max(seq_len - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    R = eigenvectors[:, idx].T  # d x d rotation

    # Rotate ALL data to PCA space
    rotated_data = (data - mean) @ R.T  # (seq_len, d) in PCA space

    # Per-coordinate quantization in PCA space (like KIVI per-channel)
    levels = (1 << bits) - 1
    recon_rotated = np.zeros_like(rotated_data)
    for ch in range(d):
        col = rotated_data[:, ch]
        mn, mx = col.min(), col.max()
        rng = mx - mn
        scale = rng / levels if rng > 1e-15 else 1.0
        indices = np.clip(np.round((col - mn) / scale), 0, levels)
        recon_rotated[:, ch] = mn + indices * scale

    # Rotate back to original space
    recon = recon_rotated @ R + mean
    return np.mean((data - recon) ** 2)


# ═══════════════════════════════════════════════════════════════
# KIVI: Per-channel key / per-token value quantization
# ═══════════════════════════════════════════════════════════════

def kivi_per_channel_mse(data: np.ndarray, bits: int) -> float:
    """KIVI per-channel: separate scale/zero per dimension across all tokens."""
    seq_len, d = data.shape
    levels = (1 << bits) - 1

    recon = np.zeros_like(data)
    for ch in range(d):
        col = data[:, ch]
        mn, mx = col.min(), col.max()
        rng = mx - mn
        scale = rng / levels if rng > 1e-15 else 1.0
        indices = np.clip(np.round((col - mn) / scale), 0, levels)
        recon[:, ch] = mn + indices * scale

    return np.mean((data - recon) ** 2)


def kivi_per_token_mse(data: np.ndarray, bits: int) -> float:
    """KIVI per-token: separate scale/zero per token across all dimensions."""
    seq_len, d = data.shape
    levels = (1 << bits) - 1

    recon = np.zeros_like(data)
    for t in range(seq_len):
        row = data[t]
        mn, mx = row.min(), row.max()
        rng = mx - mn
        scale = rng / levels if rng > 1e-15 else 1.0
        indices = np.clip(np.round((row - mn) / scale), 0, levels)
        recon[t] = mn + indices * scale

    return np.mean((data - recon) ** 2)


# ═══════════════════════════════════════════════════════════════
# Main Comparison
# ═══════════════════════════════════════════════════════════════

def run_comparison(data_dir: str, num_prompts: int = 10, bits_list: list[int] = None):
    if bits_list is None:
        bits_list = [2, 3, 4]

    meta_path = os.path.join(data_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    print(f"Model: {meta['model']}")
    print(f"d_head: {meta['d_head']}, KV heads: {meta['num_kv_heads']}")
    print(f"Analyzing {num_prompts} prompts, bit widths: {bits_list}")
    print()

    # Collect results per bit width
    results = {b: {"fq_key": [], "fq_val": [], "kivi_key": [], "kivi_val": [],
                    "turbo_key": [], "turbo_val": []} for b in bits_list}

    prompt_dirs = sorted([
        os.path.join(data_dir, d) for d in os.listdir(data_dir)
        if d.startswith("prompt_")
    ])[:num_prompts]

    t0 = time.time()
    for pi, prompt_dir in enumerate(prompt_dirs):
        # Find all layer files
        layer_files = sorted([
            f for f in os.listdir(prompt_dir)
            if f.endswith("_keys.npy")
        ])

        for lf in layer_files:
            layer_name = lf.replace("_keys.npy", "")
            kp = os.path.join(prompt_dir, f"{layer_name}_keys.npy")
            vp = os.path.join(prompt_dir, f"{layer_name}_values.npy")
            if not os.path.exists(vp):
                continue

            keys = np.load(kp).astype(np.float64)    # (kv_heads, seq, d) or (seq, d)
            values = np.load(vp).astype(np.float64)

            # Handle multi-head: flatten to (seq, d) per head
            if keys.ndim == 3:
                for h in range(keys.shape[0]):
                    k = keys[h]
                    v = values[h]
                    if k.shape[0] < 4:
                        continue
                    for bits in bits_list:
                        results[bits]["fq_key"].append(fajarquant_mse(k, bits))
                        results[bits]["fq_val"].append(fajarquant_mse(v, bits))
                        results[bits]["kivi_key"].append(kivi_per_channel_mse(k, bits))
                        results[bits]["kivi_val"].append(kivi_per_token_mse(v, bits))
                        results[bits]["turbo_key"].append(turboquant_mse(k, bits))
                        results[bits]["turbo_val"].append(turboquant_mse(v, bits))
            else:
                if keys.shape[0] < 4:
                    continue
                for bits in bits_list:
                    results[bits]["fq_key"].append(fajarquant_mse(keys, bits))
                    results[bits]["fq_val"].append(fajarquant_mse(values, bits))
                    results[bits]["kivi_key"].append(kivi_per_channel_mse(keys, bits))
                    results[bits]["kivi_val"].append(kivi_per_token_mse(values, bits))
                    results[bits]["turbo_key"].append(turboquant_mse(keys, bits))
                    results[bits]["turbo_val"].append(turboquant_mse(values, bits))

        if (pi + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  [{pi+1}/{num_prompts}] {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\nComparison complete in {elapsed:.1f}s")
    print()

    # Print results table
    print("=" * 80)
    print(f"{'3-WAY COMPARISON: FajarQuant vs KIVI vs TurboQuant':^80}")
    print(f"{'Model: ' + meta['model'] + ', d_head=' + str(meta['d_head']):^80}")
    print("=" * 80)

    print(f"\n{'':>10} {'FajarQuant':>14} {'KIVI':>14} {'TurboQuant':>14} {'FQ vs KIVI':>12} {'FQ vs Turbo':>12}")
    print("-" * 80)

    comparison_data = {}
    for bits in bits_list:
        r = results[bits]
        n = len(r["fq_key"])

        fq_k = np.mean(r["fq_key"])
        fq_v = np.mean(r["fq_val"])
        ki_k = np.mean(r["kivi_key"])
        ki_v = np.mean(r["kivi_val"])
        tq_k = np.mean(r["turbo_key"])
        tq_v = np.mean(r["turbo_val"])

        fq_vs_ki_k = (ki_k - fq_k) / ki_k * 100 if ki_k > 1e-15 else 0
        fq_vs_tq_k = (tq_k - fq_k) / tq_k * 100 if tq_k > 1e-15 else 0
        fq_vs_ki_v = (ki_v - fq_v) / ki_v * 100 if ki_v > 1e-15 else 0
        fq_vs_tq_v = (tq_v - fq_v) / tq_v * 100 if tq_v > 1e-15 else 0

        print(f"\n{bits}-bit Keys (n={n}):")
        print(f"{'  MSE':>10} {fq_k:>14.6f} {ki_k:>14.6f} {tq_k:>14.6f} {fq_vs_ki_k:>+11.1f}% {fq_vs_tq_k:>+11.1f}%")
        print(f"{bits}-bit Values (n={n}):")
        print(f"{'  MSE':>10} {fq_v:>14.6f} {ki_v:>14.6f} {tq_v:>14.6f} {fq_vs_ki_v:>+11.1f}% {fq_vs_tq_v:>+11.1f}%")

        comparison_data[f"{bits}bit"] = {
            "samples": n,
            "key": {"fajarquant": fq_k, "kivi": ki_k, "turboquant": tq_k,
                    "fq_vs_kivi_pct": fq_vs_ki_k, "fq_vs_turbo_pct": fq_vs_tq_k},
            "value": {"fajarquant": fq_v, "kivi": ki_v, "turboquant": tq_v,
                      "fq_vs_kivi_pct": fq_vs_ki_v, "fq_vs_turbo_pct": fq_vs_tq_v},
        }

    print(f"\n{'=' * 80}")
    print("Positive % = FajarQuant better, Negative % = baseline better")
    print(f"{'=' * 80}")

    # Save results
    out_path = os.path.join(data_dir, "comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(comparison_data, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/kv_cache/")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--bits", default="2,3,4", help="Comma-separated bit widths")
    args = parser.parse_args()
    bits_list = [int(b) for b in args.bits.split(",")]
    run_comparison(args.data, args.num_prompts, bits_list)
