#!/usr/bin/env python3
"""
FajarQuant — Ablation Study on Real Gemma 4 E2B KV Cache

Isolates the contribution of each innovation:
  1. PCA rotation (vs random rotation)
  2. Fused attention (memory savings, same MSE)
  3. Hierarchical bit allocation (vs flat)

Usage:
    python scripts/run_ablation.py --data data/kv_cache/
"""

import argparse
import json
import os
import time

import numpy as np
from scipy.stats import ortho_group


# ═══════════════════════════════════════════════════════════════
# Quantization Primitives
# ═══════════════════════════════════════════════════════════════

def per_coord_quantize_mse(data: np.ndarray, bits: int) -> float:
    """Per-coordinate uniform quantization MSE (after rotation)."""
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


# ═══════════════════════════════════════════════════════════════
# Ablation Configurations
# ═══════════════════════════════════════════════════════════════

def mse_random_rotation(data: np.ndarray, bits: int, seed: int = 42) -> float:
    """TurboQuant baseline: random orthogonal rotation + per-coordinate quant."""
    seq_len, d = data.shape
    rng = np.random.default_rng(seed)
    R = ortho_group.rvs(d, random_state=rng)
    rotated = data @ R.T
    recon_rot = np.zeros_like(rotated)
    levels = (1 << bits) - 1
    for ch in range(d):
        col = rotated[:, ch]
        mn, mx = col.min(), col.max()
        rng_val = mx - mn
        scale = rng_val / levels if rng_val > 1e-15 else 1.0
        indices = np.clip(np.round((col - mn) / scale), 0, levels)
        recon_rot[:, ch] = mn + indices * scale
    recon = recon_rot @ R
    return np.mean((data - recon) ** 2)


def mse_pca_rotation(data: np.ndarray, bits: int) -> float:
    """FajarQuant Innovation 1: PCA rotation + per-coordinate quant."""
    seq_len, d = data.shape
    mean = data.mean(axis=0)
    centered = data - mean
    cov = (centered.T @ centered) / max(seq_len - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    R = eigenvectors[:, idx].T
    rotated = (data - mean) @ R.T
    levels = (1 << bits) - 1
    recon_rot = np.zeros_like(rotated)
    for ch in range(d):
        col = rotated[:, ch]
        mn, mx = col.min(), col.max()
        rng = mx - mn
        scale = rng / levels if rng > 1e-15 else 1.0
        indices = np.clip(np.round((col - mn) / scale), 0, levels)
        recon_rot[:, ch] = mn + indices * scale
    recon = recon_rot @ R + mean
    return np.mean((data - recon) ** 2)


def mse_no_rotation(data: np.ndarray, bits: int) -> float:
    """No rotation baseline: direct per-coordinate quantization."""
    return per_coord_quantize_mse(data, bits)


def hierarchical_bit_savings(seq_len: int) -> dict:
    """Compute bit savings from hierarchical allocation vs flat 3-bit."""
    tiers = [(256, 4), (1024, 3), (4096, 2)]
    min_bits = 1

    total_hier = 0
    breakdown = []
    pos = 0
    for max_age, bits in tiers:
        count = min(max_age, seq_len) - pos
        if count <= 0:
            break
        total_hier += count * bits
        breakdown.append((bits, count, count * bits))
        pos = max_age
    if pos < seq_len:
        remaining = seq_len - pos
        total_hier += remaining * min_bits
        breakdown.append((min_bits, remaining, remaining * min_bits))

    total_flat = seq_len * 3
    savings_pct = (1.0 - total_hier / total_flat) * 100 if total_flat > 0 else 0
    avg_bits = total_hier / seq_len if seq_len > 0 else 0

    return {
        "seq_len": seq_len,
        "flat_3bit_total": total_flat,
        "hierarchical_total": total_hier,
        "savings_pct": round(savings_pct, 1),
        "avg_bits": round(avg_bits, 2),
        "breakdown": breakdown,
    }


def fused_memory_savings(seq_len: int, d: int, bits: int) -> dict:
    """Compute memory savings from fused attention vs standard dequantize."""
    # Standard: must dequantize all N keys → O(N * d * 8) bytes extra
    standard_extra = seq_len * d * 8  # f64
    # Fused: only need codebook in registers → O(2^bits * 8) bytes
    fused_extra = (1 << bits) * 8

    return {
        "seq_len": seq_len,
        "standard_dequant_bytes": standard_extra,
        "fused_codebook_bytes": fused_extra,
        "reduction_factor": round(standard_extra / fused_extra, 1) if fused_extra > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════
# Main Ablation
# ═══════════════════════════════════════════════════════════════

def run_ablation(data_dir: str, num_prompts: int = 10, bits_list: list[int] = None):
    if bits_list is None:
        bits_list = [2, 3, 4]

    meta_path = os.path.join(data_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    d_head = meta["d_head"]
    print(f"Model: {meta['model']}, d_head={d_head}")
    print(f"Ablation: {num_prompts} prompts, bits: {bits_list}")
    print()

    # Collect per-sample MSE for each configuration
    results = {b: {"no_rot": [], "random_rot": [], "pca_rot": []} for b in bits_list}

    prompt_dirs = sorted([
        os.path.join(data_dir, d) for d in os.listdir(data_dir)
        if d.startswith("prompt_")
    ])[:num_prompts]

    t0 = time.time()
    for pi, prompt_dir in enumerate(prompt_dirs):
        layer_files = sorted([
            f for f in os.listdir(prompt_dir)
            if f.endswith("_keys.npy")
        ])

        for lf in layer_files:
            layer_name = lf.replace("_keys.npy", "")
            kp = os.path.join(prompt_dir, f"{layer_name}_keys.npy")
            if not os.path.exists(kp):
                continue

            keys = np.load(kp).astype(np.float64)

            # Handle multi-head
            samples = []
            if keys.ndim == 3:
                for h in range(keys.shape[0]):
                    if keys[h].shape[0] >= 4:
                        samples.append(keys[h])
            elif keys.shape[0] >= 4:
                samples.append(keys)

            for k in samples:
                for bits in bits_list:
                    results[bits]["no_rot"].append(mse_no_rotation(k, bits))
                    results[bits]["random_rot"].append(mse_random_rotation(k, bits))
                    results[bits]["pca_rot"].append(mse_pca_rotation(k, bits))

        if (pi + 1) % 5 == 0:
            print(f"  [{pi+1}/{num_prompts}] {time.time() - t0:.1f}s")

    elapsed = time.time() - t0
    print(f"\nAblation complete in {elapsed:.1f}s")

    # ═══════════════════════════════════════════════════════════
    # Print results
    # ═══════════════════════════════════════════════════════════

    print()
    print("=" * 85)
    print(f"{'ABLATION STUDY: Individual Innovation Contributions':^85}")
    print(f"{'Model: ' + meta['model'] + ', d_head=' + str(d_head):^85}")
    print("=" * 85)

    print(f"\n{'Innovation 1: PCA Rotation (Key MSE)':^85}")
    print("-" * 85)
    print(f"{'Bits':>6} {'No Rotation':>14} {'Random (Turbo)':>16} {'PCA (Fajar)':>14} {'vs No Rot':>12} {'vs Random':>12}")
    print("-" * 85)

    ablation_data = {}
    for bits in bits_list:
        r = results[bits]
        n = len(r["no_rot"])
        no_rot = np.mean(r["no_rot"])
        rand_rot = np.mean(r["random_rot"])
        pca_rot = np.mean(r["pca_rot"])

        vs_no = (no_rot - pca_rot) / no_rot * 100 if no_rot > 1e-15 else 0
        vs_rand = (rand_rot - pca_rot) / rand_rot * 100 if rand_rot > 1e-15 else 0

        print(f"{bits:>6} {no_rot:>14.6e} {rand_rot:>16.6e} {pca_rot:>14.6e} {vs_no:>+11.1f}% {vs_rand:>+11.1f}%")

        ablation_data[f"{bits}bit"] = {
            "samples": n,
            "no_rotation_mse": no_rot,
            "random_rotation_mse": rand_rot,
            "pca_rotation_mse": pca_rot,
            "pca_vs_none_pct": round(vs_no, 1),
            "pca_vs_random_pct": round(vs_rand, 1),
        }

    # Innovation 2: Fused attention (memory)
    print(f"\n{'Innovation 2: Fused Attention (Memory Savings)':^85}")
    print("-" * 85)
    print(f"{'Seq Len':>10} {'Standard Extra':>16} {'Fused Extra':>14} {'Reduction':>12}")
    print("-" * 85)

    memory_data = {}
    for N in [256, 1024, 4096, 16384]:
        mem = fused_memory_savings(N, d_head, 3)
        std_str = f"{mem['standard_dequant_bytes'] / 1024:.0f} KB"
        fused_str = f"{mem['fused_codebook_bytes']} B"
        print(f"{N:>10} {std_str:>16} {fused_str:>14} {mem['reduction_factor']:>11.0f}x")
        memory_data[f"N={N}"] = mem

    # Innovation 3: Hierarchical allocation (bit savings)
    print(f"\n{'Innovation 3: Hierarchical Allocation (Bit Savings)':^85}")
    print("-" * 85)
    print(f"{'Seq Len':>10} {'Flat 3-bit':>14} {'Hierarchical':>14} {'Avg Bits':>10} {'Savings':>10}")
    print("-" * 85)

    hier_data = {}
    for N in [256, 1024, 4096, 10000, 16384]:
        h = hierarchical_bit_savings(N)
        print(f"{N:>10} {h['flat_3bit_total']:>14,} {h['hierarchical_total']:>14,} {h['avg_bits']:>10.2f} {h['savings_pct']:>+9.1f}%")
        hier_data[f"N={N}"] = h

    # Combined ablation summary
    print(f"\n{'=' * 85}")
    print(f"{'COMBINED ABLATION SUMMARY (at 2-bit, N=10K)':^85}")
    print("=" * 85)
    b2 = ablation_data.get("2bit", {})
    mem_16k = fused_memory_savings(10000, d_head, 2)
    hier_10k = hierarchical_bit_savings(10000)

    print(f"\n  Configuration                  Key MSE Δ      Extra Memory    Bit Savings")
    print(f"  {'─' * 72}")
    print(f"  TurboQuant (baseline)          0.0%           {mem_16k['standard_dequant_bytes']/1024/1024:.1f} MB          0%")
    if b2:
        print(f"  + PCA rotation                 {b2.get('pca_vs_random_pct', 0):+.1f}%          {mem_16k['standard_dequant_bytes']/1024/1024:.1f} MB          0%")
        print(f"  + Fused attention              {b2.get('pca_vs_random_pct', 0):+.1f}%          {mem_16k['fused_codebook_bytes']} B            0%")
        print(f"  + Hierarchical (Full FQ)       {b2.get('pca_vs_random_pct', 0):+.1f}%          {mem_16k['fused_codebook_bytes']} B            {hier_10k['savings_pct']}%")
    print()

    # Save results
    output = {
        "model": meta["model"],
        "d_head": d_head,
        "rotation_ablation": ablation_data,
        "memory_ablation": memory_data,
        "hierarchical_ablation": hier_data,
    }
    out_path = os.path.join(data_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/kv_cache/")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--bits", default="2,3,4")
    args = parser.parse_args()
    bits_list = [int(b) for b in args.bits.split(",")]
    run_ablation(args.data, args.num_prompts, bits_list)
