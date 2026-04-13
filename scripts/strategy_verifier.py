#!/usr/bin/env python3
"""
strategy_verifier.py — B2.V: Verify strategy assignments via mini-batch MSE.

For each head, applies all 5 paths on a small data sample and compares
reconstruction MSE. If a different path produces ≥10% lower MSE than the
assigned path, swaps the assignment.

Caps swap rate at 20% of total heads to prevent instability.

Usage:
    python scripts/strategy_verifier.py \
        --profile data/calibration/profile_gemma.json \
        --strategy data/calibration/strategy_gemma_2bit.json \
        --output data/calibration/strategy_gemma_2bit_verified.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_attention import (
    apply_kivi_keys_4d,
    apply_kivi_values_4d,
    apply_turboquant_4d_with_outliers,
)


def _mse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    return float(((original - reconstructed) ** 2).mean())


def _apply_path(data_4d: torch.Tensor, path: str, bits: int, is_key: bool) -> torch.Tensor:
    """Apply a quantization path and return the reconstructed tensor."""
    if path == "A":
        if is_key:
            return apply_kivi_keys_4d(data_4d, bits)
        return apply_kivi_values_4d(data_4d, bits)
    elif path == "C":
        return apply_turboquant_4d_with_outliers(data_4d, bits)
    elif path == "D":
        # Residual: base + residual pass
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
        # Asymmetric per-channel
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
        # Path B or fallback: use KIVI
        if is_key:
            return apply_kivi_keys_4d(data_4d, bits)
        return apply_kivi_values_4d(data_4d, bits)


def verify_and_swap(
    strategy: dict,
    profile: dict,
    bits: int,
    swap_threshold: float = 0.10,
    max_swap_rate: float = 0.20,
) -> dict:
    """Verify each head's assignment and swap if a better path exists.

    Uses synthetic data matching the head's profiled statistics for
    MSE comparison (avoids needing the actual model).
    """
    n_kv_heads = profile.get("_n_heads", 1)
    head_dim = profile.get("_head_dim", 128)
    assignments = strategy.get("assignments", [])
    paths_to_test = ["A", "C", "D", "E"]  # Skip B (needs calibration data)

    total_heads = 0
    swaps = 0
    swap_log = []

    new_assignments = []
    for li, (layer_strat, layer_prof) in enumerate(
        zip(assignments, profile.get("layers", []))
    ):
        new_layer = []
        for hi, (head_strat, head_prof) in enumerate(
            zip(layer_strat, layer_prof.get("heads", []))
        ):
            new_head = {}
            for kv_type in ["k", "v"]:
                total_heads += 1
                assigned = head_strat.get(kv_type, "A")

                # Generate synthetic data matching head's mean_abs
                mean_abs = head_prof.get(kv_type, {}).get("mean_abs", 1.0)
                synthetic = torch.randn(1, 1, 64, head_dim) * mean_abs

                # Test all paths
                best_path = assigned
                assigned_mse = _mse(synthetic, _apply_path(synthetic, assigned, bits, kv_type == "k"))
                best_mse = assigned_mse

                for path in paths_to_test:
                    if path == assigned:
                        continue
                    recon = _apply_path(synthetic, path, bits, kv_type == "k")
                    mse = _mse(synthetic, recon)
                    if mse < best_mse * (1 - swap_threshold):
                        best_mse = mse
                        best_path = path

                if best_path != assigned:
                    swaps += 1
                    swap_log.append({
                        "layer": li, "head": hi, "kv": kv_type,
                        "from": assigned, "to": best_path,
                        "mse_before": round(assigned_mse, 6),
                        "mse_after": round(best_mse, 6),
                    })

                new_head[kv_type] = best_path
            new_layer.append(new_head)
        new_assignments.append(new_layer)

    # Cap swap rate
    swap_rate = swaps / total_heads if total_heads > 0 else 0
    if swap_rate > max_swap_rate:
        print(f"  WARNING: swap rate {swap_rate:.1%} exceeds cap {max_swap_rate:.1%}")
        print(f"  Keeping only top {int(max_swap_rate * total_heads)} swaps by MSE improvement")
        # Sort by MSE improvement, keep top N
        swap_log.sort(key=lambda s: s["mse_before"] - s["mse_after"], reverse=True)
        keep_n = int(max_swap_rate * total_heads)
        kept_swaps = set()
        for s in swap_log[:keep_n]:
            kept_swaps.add((s["layer"], s["head"], s["kv"]))
        # Revert non-kept swaps
        for li, layer in enumerate(new_assignments):
            for hi, head in enumerate(layer):
                for kv_type in ["k", "v"]:
                    if (li, hi, kv_type) not in kept_swaps:
                        new_assignments[li][hi][kv_type] = assignments[li][hi].get(kv_type, "A")
        swaps = len(kept_swaps)

    # Recount
    path_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    for layer in new_assignments:
        for head in layer:
            for kv_type in ["k", "v"]:
                path_counts[head.get(kv_type, "A")] += 1

    result = dict(strategy)
    result["assignments"] = new_assignments
    result["path_counts"] = path_counts
    result["n_unique_paths"] = sum(1 for v in path_counts.values() if v > 0)
    result["_verification"] = {
        "swaps": swaps,
        "total_heads": total_heads,
        "swap_rate": round(swaps / total_heads, 4) if total_heads > 0 else 0,
        "swap_log": swap_log[:10],  # keep top 10 for audit
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="FajarQuant v3 strategy verifier")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--swap-threshold", type=float, default=0.10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.profile) as f:
        profile = json.load(f)
    with open(args.strategy) as f:
        strategy = json.load(f)

    bits = args.bits or strategy.get("_bits", 2)
    print(f"Verifying strategy for {strategy.get('_model', '?')} at {bits}-bit...")

    result = verify_and_swap(strategy, profile, bits, args.swap_threshold)

    v = result["_verification"]
    print(f"  Swaps: {v['swaps']}/{v['total_heads']} ({v['swap_rate']:.1%})")
    print(f"  Path counts: {result['path_counts']}")

    output = args.output or args.strategy.replace(".json", "_verified.json")
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {output}")


if __name__ == "__main__":
    main()
