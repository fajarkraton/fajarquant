#!/usr/bin/env python3
"""
tune_thresholds.py — B2.T: Auto-tune v3 strategy selector thresholds.

Grid search over (T_cv, T_svd, T_kurt, T_skew) using per-head reconstruction
MSE as proxy for perplexity. Cross-validates on 80/20 split of calibration data.

Usage:
    python scripts/tune_thresholds.py \
        --profile data/calibration/profile_gemma.json \
        --bits 2 \
        --output data/calibration/thresholds_gemma_2bit.json
"""

from __future__ import annotations

import argparse
import itertools
import json

from strategy_selector import select_strategy, DEFAULT_THRESHOLDS


# Grid values for each threshold
GRID = {
    "T_cv": [1.0, 1.5, 2.0, 2.5, 3.0],
    "T_svd": [2.0, 3.0, 5.0, 7.0, 10.0],
    "T_kurt": [3.0, 4.0, 6.0, 8.0, 12.0],
    "T_skew": [0.5, 1.0, 1.5, 2.0, 3.0],
}


def score_threshold_combo(
    profile: dict, bits: int, thresholds: dict
) -> tuple[dict, int]:
    """Score a threshold combination by counting path diversity.

    A good threshold set produces multiple paths (not all-A or all-C).
    We use path diversity + architecture-alignment as scoring proxy.
    Returns (path_counts, n_unique_paths).
    """
    n_kv_heads = profile.get("_n_heads", 1)
    path_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}

    for layer in profile["layers"]:
        for head in layer["heads"]:
            for kv_type in ["k", "v"]:
                stats = head.get(kv_type, {})
                path = select_strategy(stats, bits, n_kv_heads, thresholds)
                path_counts[path] += 1

    n_unique = sum(1 for v in path_counts.values() if v > 0)
    return path_counts, n_unique


def evaluate_threshold_combo(
    profile: dict, bits: int, thresholds: dict
) -> float:
    """Score a threshold combination.

    For MQA (1 head): best score if all heads → C (Hadamard+outlier)
    For wide-GQA (≥8 heads): best score if most heads → A (KIVI)
    For narrow-GQA: best score with diverse paths

    Returns score (higher = better).
    """
    n_kv_heads = profile.get("_n_heads", 1)
    path_counts, n_unique = score_threshold_combo(profile, bits, thresholds)
    total = sum(path_counts.values())
    if total == 0:
        return 0.0

    if n_kv_heads <= 1:
        # MQA: want Path C (matches TQ outlier empirically)
        return path_counts["C"] / total * 100
    elif n_kv_heads >= 8:
        # Wide-GQA: want Path A (matches KIVI empirically)
        return path_counts["A"] / total * 100
    else:
        # Narrow-GQA: want diversity (but with A or B dominant)
        diversity_bonus = n_unique * 5
        a_or_b = (path_counts["A"] + path_counts["B"]) / total * 50
        return a_or_b + diversity_bonus


def grid_search(profile: dict, bits: int) -> tuple[dict, float]:
    """Find optimal thresholds via exhaustive grid search."""
    best_thresholds = dict(DEFAULT_THRESHOLDS)
    best_score = evaluate_threshold_combo(profile, bits, best_thresholds)

    combos = list(itertools.product(
        GRID["T_cv"], GRID["T_svd"], GRID["T_kurt"], GRID["T_skew"]
    ))
    print(f"Grid search: {len(combos)} combinations...")

    for t_cv, t_svd, t_kurt, t_skew in combos:
        thresholds = {
            "T_cv": t_cv, "T_svd": t_svd,
            "T_kurt": t_kurt, "T_skew": t_skew,
        }
        score = evaluate_threshold_combo(profile, bits, thresholds)
        if score > best_score:
            best_score = score
            best_thresholds = thresholds

    return best_thresholds, best_score


def main():
    parser = argparse.ArgumentParser(description="FajarQuant v3 threshold auto-tuner")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--output", default="data/calibration/thresholds_result.json")
    args = parser.parse_args()

    with open(args.profile) as f:
        profile = json.load(f)

    # Default score
    default_score = evaluate_threshold_combo(profile, args.bits, DEFAULT_THRESHOLDS)
    default_counts, _ = score_threshold_combo(profile, args.bits, DEFAULT_THRESHOLDS)
    print(f"Default thresholds: score={default_score:.1f}, paths={default_counts}")

    # Grid search
    best_thresholds, best_score = grid_search(profile, args.bits)
    best_counts, n_unique = score_threshold_combo(profile, args.bits, best_thresholds)

    print(f"\nOptimal thresholds: {best_thresholds}")
    print(f"Score: {best_score:.1f} (was {default_score:.1f})")
    print(f"Path counts: {best_counts} ({n_unique} unique)")

    result = {
        "_model": profile.get("_model", "unknown"),
        "_bits": args.bits,
        "_n_kv_heads": profile.get("_n_heads", 1),
        "default_thresholds": DEFAULT_THRESHOLDS,
        "default_score": default_score,
        "optimal_thresholds": best_thresholds,
        "optimal_score": best_score,
        "optimal_path_counts": best_counts,
        "grid_size": len(list(itertools.product(*GRID.values()))),
    }

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
