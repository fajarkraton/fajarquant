#!/usr/bin/env python3
"""
strategy_verifier.py — B-fix.3: Verify strategy assignments via real KV cache MSE.

For each head, applies all 5 paths on REAL KV cache data (extracted from model
forward pass) and compares reconstruction MSE. If a different path produces
≥10% lower MSE than the assigned path, swaps the assignment.

Caps swap rate at 20% of total heads to prevent instability.

B-fix.3: replaced torch.randn synthetic data with real model KV cache chunks.
Re-enabled Path B (PCA) in test matrix with calibration data support.

Usage:
    python scripts/strategy_verifier.py \
        --model google/gemma-4-E2B \
        --profile data/calibration/profile_gemma_4_e2b.json \
        --strategy data/calibration/strategy_gemma_4_e2b_2bit.json \
        --calibration data/calibration/fq_v2_gemma_4_e2b.npz \
        --bits 2 \
        --output data/calibration/strategy_gemma_4_e2b_2bit_verified.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_attention import (
    apply_kivi_keys_4d,
    apply_kivi_values_4d,
    apply_turboquant_4d_with_outliers,
)
from quant_attention_v2 import load_calibration
from quant_attention_v3 import _path_b_pca
from tune_thresholds import extract_kv_cache, load_wikitext2


def _mse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    return float(((original - reconstructed) ** 2).mean())


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


def verify_and_swap(
    strategy: dict,
    profile: dict,
    bits: int,
    kv_chunks: list[list[dict]],
    cal: dict | None = None,
    swap_threshold: float = 0.10,
    max_swap_rate: float = 0.20,
    device: str = "cuda",
) -> dict:
    """Verify each head's assignment using real KV cache data.

    For each (layer, head, kv_type):
      1. Collect real KV cache data from all chunks
      2. Apply all 5 paths, compute MSE
      3. If a different path produces ≥swap_threshold lower MSE, swap
    """
    assignments = strategy.get("assignments", [])
    paths_to_test = ["A", "B", "C", "D", "E"]

    total_heads = 0
    swaps = 0
    swap_log = []

    new_assignments = []
    for li, layer_strat in enumerate(assignments):
        if li >= len(profile.get("layers", [])):
            new_assignments.append(layer_strat)
            continue

        new_layer = []
        for hi, head_strat in enumerate(layer_strat):
            new_head = {}
            for kv_type in ["k", "v"]:
                total_heads += 1
                assigned = head_strat.get(kv_type, "A")
                is_key = (kv_type == "k")

                # Collect real KV data for this head from all chunks
                # Average MSE across chunks for stability
                assigned_mse_sum = 0.0
                path_mse_sums = {p: 0.0 for p in paths_to_test}
                n_valid = 0

                for chunk_kv in kv_chunks:
                    if li >= len(chunk_kv):
                        continue
                    original = chunk_kv[li][kv_type]  # (H, S, D)
                    if hi >= original.shape[0]:
                        continue

                    # Single head as 4D: (1, 1, S, D)
                    head_data = original[hi:hi+1].unsqueeze(0).to(device)

                    for path in paths_to_test:
                        recon = _apply_path_with_cal(
                            head_data, path, bits, is_key, li, cal
                        )
                        mse = _mse(head_data, recon)
                        path_mse_sums[path] += mse

                    n_valid += 1

                if n_valid == 0:
                    new_head[kv_type] = assigned
                    continue

                # Average MSE across chunks
                path_mses = {p: path_mse_sums[p] / n_valid for p in paths_to_test}
                assigned_mse = path_mses[assigned]
                best_path = min(path_mses, key=path_mses.get)
                best_mse = path_mses[best_path]

                # Swap if best path is ≥swap_threshold better
                if best_path != assigned and best_mse < assigned_mse * (1 - swap_threshold):
                    swaps += 1
                    swap_log.append({
                        "layer": li, "head": hi, "kv": kv_type,
                        "from": assigned, "to": best_path,
                        "mse_before": round(assigned_mse, 6),
                        "mse_after": round(best_mse, 6),
                        "improvement": round(1 - best_mse / assigned_mse, 4) if assigned_mse > 0 else 0,
                        "all_mses": {p: round(v, 6) for p, v in path_mses.items()},
                    })
                    new_head[kv_type] = best_path
                else:
                    new_head[kv_type] = assigned

            new_layer.append(new_head)
        new_assignments.append(new_layer)

    # Cap swap rate
    swap_rate = swaps / total_heads if total_heads > 0 else 0
    if swap_rate > max_swap_rate:
        print(f"  WARNING: swap rate {swap_rate:.1%} exceeds cap {max_swap_rate:.1%}")
        print(f"  Keeping only top {int(max_swap_rate * total_heads)} swaps by MSE improvement")
        swap_log.sort(key=lambda s: s["mse_before"] - s["mse_after"], reverse=True)
        keep_n = int(max_swap_rate * total_heads)
        kept_swaps = set()
        for s in swap_log[:keep_n]:
            kept_swaps.add((s["layer"], s["head"], s["kv"]))
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
        "method": "real_kv_cache_mse",
        "n_chunks": len(kv_chunks),
        "swap_threshold": swap_threshold,
        "max_swap_rate": max_swap_rate,
        "swaps": swaps,
        "total_heads": total_heads,
        "swap_rate": round(swaps / total_heads, 4) if total_heads > 0 else 0,
        "swap_log": swap_log[:20],  # keep top 20 for audit
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="FajarQuant v3 strategy verifier (B-fix.3)")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--calibration", default=None, help="v2 calibration .npz (for Path B)")
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--n-chunks", type=int, default=3, help="Chunks for verification")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--swap-threshold", type=float, default=0.10)
    parser.add_argument("--token", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.profile) as f:
        profile = json.load(f)
    with open(args.strategy) as f:
        strategy = json.load(f)

    cal = None
    if args.calibration and os.path.exists(args.calibration):
        cal = load_calibration(args.calibration)
        print(f"Loaded calibration: {args.calibration} ({cal['n_layers']} layers)")

    bits = args.bits or strategy.get("_bits", 2)
    print(f"Verifying strategy for {strategy.get('_model', '?')} at {bits}-bit...")
    print(f"  Using {args.n_chunks} real KV cache chunks for MSE comparison")

    # Load model and extract KV cache
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, token=args.token, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, token=args.token, dtype=torch.float16,
        device_map="auto", trust_remote_code=True,
    ).eval()

    text = load_wikitext2()
    print(f"Extracting {args.n_chunks} KV cache chunks...")
    t0 = time.time()
    kv_chunks = extract_kv_cache(model, tokenizer, text, args.n_chunks, args.seq_len)
    print(f"  Extracted in {time.time()-t0:.1f}s")

    del model
    torch.cuda.empty_cache()

    # Verify
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    result = verify_and_swap(
        strategy, profile, bits, kv_chunks, cal,
        args.swap_threshold, device=device,
    )
    elapsed = time.time() - t0

    v = result["_verification"]
    print(f"\n  Swaps: {v['swaps']}/{v['total_heads']} ({v['swap_rate']:.1%}) in {elapsed:.1f}s")
    print(f"  Path counts: {result['path_counts']}")

    if v["swap_log"]:
        print(f"\n  Top swaps:")
        for s in v["swap_log"][:5]:
            print(f"    L{s['layer']}H{s['head']}.{s['kv']}: "
                  f"{s['from']}→{s['to']} (MSE {s['mse_before']:.4f}→{s['mse_after']:.4f}, "
                  f"{s['improvement']:.1%} better)")

    output = args.output or args.strategy.replace(".json", "_verified.json")
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to {output}")


if __name__ == "__main__":
    main()
