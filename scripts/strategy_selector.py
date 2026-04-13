#!/usr/bin/env python3
"""
strategy_selector.py — FajarQuant v3 per-head strategy selection.

Given a profile JSON (from profile_kv_heads.py), assigns each head to
one of 5 quantization paths based on its statistical properties.

Paths:
  A: KIVI-like per-channel symmetric (regular distributions)
  B: PCA rotation + per-coord quantization (structured low-rank)
  C: Hadamard + adaptive outlier extraction (extreme outliers)
  D: Residual quantization (2-bit with high kurtosis)
  E: Asymmetric per-channel (non-zero-centered, high skewness)

Usage:
    python scripts/strategy_selector.py \
        --profile data/calibration/profile_gemma.json \
        --bits 2 \
        --output data/calibration/strategy_gemma_2bit.json
"""

from __future__ import annotations

import argparse
import json


# Default thresholds (tunable via B2.T auto-tuning)
DEFAULT_THRESHOLDS = {
    "T_cv": 2.0,       # channel variance CV → Path A
    "T_svd": 5.0,      # σ1/σ2 ratio → Path B
    "T_kurt": 6.0,     # excess kurtosis → Path C
    "T_skew": 1.5,     # absolute skewness → Path E
}


def select_strategy(
    stats: dict,
    bits: int,
    thresholds: dict | None = None,
) -> str:
    """Select the optimal quantization path for a single head.

    Returns one of: 'A', 'B', 'C', 'D', 'E'.
    """
    t = thresholds or DEFAULT_THRESHOLDS
    cv = stats.get("channel_var_cv", 0.0)
    svd = stats.get("svd_ratio", 1.0)
    kurt = stats.get("kurtosis", 0.0)
    skew = stats.get("skewness", 0.0)

    # Priority order: KIVI > PCA > Hadamard+outlier > Asymmetric > Residual
    if cv > t["T_cv"]:
        return "A"  # KIVI-like: channels have very different variances
    if svd > t["T_svd"]:
        return "B"  # PCA: strong low-rank structure
    if kurt > t["T_kurt"]:
        return "C"  # Hadamard + outlier: heavy-tailed distribution
    if skew > t["T_skew"]:
        return "E"  # Asymmetric: non-zero-centered distribution
    if bits == 2 and kurt > 3.0:
        return "D"  # Residual: 2-bit with moderate kurtosis
    return "A"  # Default: KIVI (safest, works on regular distributions)


def assign_strategies(
    profile: dict,
    bits: int,
    thresholds: dict | None = None,
) -> dict:
    """Assign strategies to all heads in a profile.

    Returns a dict with per-head assignments and summary statistics.
    """
    assignments = []
    path_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}

    for li, layer in enumerate(profile["layers"]):
        layer_assignments = []
        for hi, head in enumerate(layer["heads"]):
            head_assignment = {}
            for kv_type in ["k", "v"]:
                stats = head.get(kv_type, {})
                path = select_strategy(stats, bits, thresholds)
                head_assignment[kv_type] = path
                path_counts[path] += 1
            layer_assignments.append(head_assignment)
        assignments.append(layer_assignments)

    total = sum(path_counts.values())
    return {
        "_model": profile.get("_model", "unknown"),
        "_bits": bits,
        "_thresholds": thresholds or DEFAULT_THRESHOLDS,
        "assignments": assignments,
        "path_counts": path_counts,
        "path_fractions": {
            k: round(v / total, 3) if total > 0 else 0
            for k, v in path_counts.items()
        },
        "n_unique_paths": sum(1 for v in path_counts.values() if v > 0),
    }


def main():
    parser = argparse.ArgumentParser(description="FajarQuant v3 strategy selector")
    parser.add_argument("--profile", required=True, help="Path to profile JSON")
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--output", default="data/calibration/strategy_result.json")
    # Threshold overrides
    parser.add_argument("--T-cv", type=float, default=None)
    parser.add_argument("--T-svd", type=float, default=None)
    parser.add_argument("--T-kurt", type=float, default=None)
    parser.add_argument("--T-skew", type=float, default=None)
    args = parser.parse_args()

    with open(args.profile) as f:
        profile = json.load(f)

    thresholds = dict(DEFAULT_THRESHOLDS)
    if args.T_cv is not None:
        thresholds["T_cv"] = args.T_cv
    if args.T_svd is not None:
        thresholds["T_svd"] = args.T_svd
    if args.T_kurt is not None:
        thresholds["T_kurt"] = args.T_kurt
    if args.T_skew is not None:
        thresholds["T_skew"] = args.T_skew

    result = assign_strategies(profile, args.bits, thresholds)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Strategy assignment for {result['_model']} at {args.bits}-bit:")
    print(f"  Path counts: {result['path_counts']}")
    print(f"  Unique paths used: {result['n_unique_paths']}")
    print(f"  Fractions: {result['path_fractions']}")
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
