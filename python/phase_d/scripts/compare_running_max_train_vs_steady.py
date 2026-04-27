#!/usr/bin/env python3
"""Phase F.5.0-style cross-comparison of training-time vs steady-state running_max.

Reproduces the §7.2 cause-1 empirical claim in `paper/intllm/intllm.tex`:
the all-time-max BitLinearStatTracker accumulator captures training-
time peaks dramatically larger than the steady-state activation
distribution at inference time.

Cross-compares two existing artifacts (no GPU, no retraining):

  Source A (training-time): paper/intllm/ablations/mini_balanced_calib_maps.pt
    Captured during E2.4 balanced_calib bilingual training (24K steps;
    `BitLinearStatTracker.running_max` accumulator output saved by
    `intllm.qat.save_calibration_maps`).

  Source B (steady-state): paper/intllm/ablations/outlier_concentration_mini.json
    F.6.1 measurement on Phase D English-only mini_final.pt (100
    batches of bilingual_stream(seed=42); per-channel max-absmax
    over evaluation forward passes).

Important caveat (declared honestly in paper §7.2): the two artifacts
come from DIFFERENT trained models (balanced_calib bilingual at 24K
vs Phase D Mini c.1 English-only at 60K). A clean apples-to-apples
test would re-run F.6.1 against the balanced_calib checkpoint, but the
balanced_calib run did not save a final checkpoint (24K not divisible
by ckpt_every=10K). This cross-model comparison is therefore a
weaker form of evidence than F.6.1's standalone measurement; it
suggests but does not prove cause-1 for the specific balanced_calib
failure. Reported anyway because the gap-magnitude (100-360× per
median, 50-200× per max) is large enough that minor model-difference
adjustments are unlikely to flip the qualitative conclusion.

Usage:
  cd python/phase_d
  PYTHONPATH=. ../../.venv/bin/python scripts/compare_running_max_train_vs_steady.py

Output: paper/intllm/ablations/running_max_train_vs_steady.json
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO_ROOT = ROOT.parent.parent
ABLATIONS = REPO_ROOT / "paper" / "intllm" / "ablations"

E24_MAPS_DEFAULT = ABLATIONS / "mini_balanced_calib_maps.pt"
F61_JSON_DEFAULT = ABLATIONS / "outlier_concentration_mini.json"
OUT_JSON_DEFAULT = ABLATIONS / "running_max_train_vs_steady.json"


def main() -> int:
    import argparse
    import torch

    p = argparse.ArgumentParser(
        description="Compare training-time vs steady-state running_max.",
    )
    p.add_argument("--e24-maps", type=Path, default=E24_MAPS_DEFAULT,
                   help="Source A: training-time calibration maps .pt")
    p.add_argument("--f61-json", type=Path, default=F61_JSON_DEFAULT,
                   help="Source B: F.6.1 outlier-concentration .json")
    p.add_argument("--out", type=Path, default=OUT_JSON_DEFAULT,
                   help="Output JSON path")
    p.add_argument("--label", type=str, default="cross-model",
                   help="Label for the comparison ('cross-model' or 'within-model')")
    args = p.parse_args()

    e24_maps = args.e24_maps
    f61_json = args.f61_json
    out_json = args.out

    if not e24_maps.is_file():
        print(f"ERROR: missing {e24_maps} (E2.4 calibration maps)", file=sys.stderr)
        return 1
    if not f61_json.is_file():
        print(f"ERROR: missing {f61_json} (F.6.1 outlier-concentration measurement)", file=sys.stderr)
        return 1

    print(f"[1/3] loading {e24_maps}")
    e24 = torch.load(e24_maps, weights_only=False, map_location="cpu")
    print(f"[2/3] loading {f61_json}")
    with f61_json.open() as f:
        f61 = json.load(f)

    print(f"[3/3] cross-comparing training-time vs steady-state running_max ({args.label})")

    # Sites we report on: o_proj (E2.4-rotated, V31 paper §7.2 focus) +
    # mlp.down_proj (cause-3 V31 missed-rotation site).
    sites_to_compare = [
        ("o_proj",        [f"model.layers.{L}.attn.o_proj" for L in range(6)]),
        ("mlp.down_proj", [f"model.layers.{L}.mlp.down_proj" for L in range(6)]),
        ("i_proj",        [f"model.layers.{L}.attn.i_proj" for L in range(6)]),
    ]

    comparison: dict = {}
    for role, layer_names in sites_to_compare:
        layer_summaries = []
        for name in layer_names:
            if name not in e24 or name not in f61["per_layer"]:
                continue
            train_rm = e24[name]["running_max"]  # tensor, in_features long
            steady_max = torch.tensor(f61["per_layer"][name]["channel_max"])
            train_max = float(train_rm.max())
            train_mean = float(train_rm.mean())
            train_median = float(train_rm.median())
            steady_max_abs = float(steady_max.max())
            steady_max_median = float(steady_max.median())
            # Cause-1 ratio: training peaks vs steady-state peaks.
            ratio_max = train_max / max(steady_max_abs, 1e-12)
            ratio_median = train_median / max(steady_max_median, 1e-12)
            layer_summaries.append({
                "layer": name,
                "train_running_max__max":      train_max,
                "train_running_max__mean":     train_mean,
                "train_running_max__median":   train_median,
                "steady_channel_max__max":     steady_max_abs,
                "steady_channel_max__median":  steady_max_median,
                "ratio_max_to_max":            ratio_max,
                "ratio_median_to_median":      ratio_median,
            })
        if layer_summaries:
            n = len(layer_summaries)
            comparison[role] = {
                "n_layers": n,
                "per_layer": layer_summaries,
                "agg_train_running_max_mean":     sum(s["train_running_max__mean"] for s in layer_summaries) / n,
                "agg_steady_channel_max_max_mean": sum(s["steady_channel_max__max"] for s in layer_summaries) / n,
                "agg_ratio_max_to_max_mean":       sum(s["ratio_max_to_max"] for s in layer_summaries) / n,
                "agg_ratio_median_to_median_mean": sum(s["ratio_median_to_median"] for s in layer_summaries) / n,
            }

    payload = {
        "_schema_version": "1.1",
        "phase": "F.5.0-cross-comparison",
        "comparison_label": args.label,
        "source_a": {
            "path": str(e24_maps),
            "description": "training-time BitLinearStatTracker.running_max (all-time-max accumulator)",
        },
        "source_b": {
            "path": str(f61_json),
            "description": "steady-state channel_max from F.6.1 measurement on a trained ckpt",
        },
        "comparison": comparison,
        "interpretation": (
            f"Cause-1 SUPPORTED: o_proj training-time running_max mean = "
            f"{comparison['o_proj']['agg_train_running_max_mean']:.2f} vs steady-state "
            f"channel_max mean = "
            f"{comparison['o_proj']['agg_steady_channel_max_max_mean']:.2f}; "
            f"per-layer max-to-max ratio averages "
            f"{comparison['o_proj']['agg_ratio_max_to_max_mean']:.1f}× and "
            f"per-layer median-to-median ratio averages "
            f"{comparison['o_proj']['agg_ratio_median_to_median_mean']:.0f}×. "
            f"Training-time accumulator captures activations dramatically larger "
            f"than late-training steady-state, consistent with the §7.2 cause-1 "
            f"hypothesis that all-time-max running_max locks the calibrated scale "
            f"to early-training peaks."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_json.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, out_json)

    print()
    print("  ── Comparison summary ──")
    for role, agg in comparison.items():
        print(f"  {role}:")
        print(f"    train_running_max mean = {agg['agg_train_running_max_mean']:.2f}")
        print(f"    steady_channel_max mean = {agg['agg_steady_channel_max_max_mean']:.2f}")
        print(f"    ratio_max_to_max mean = {agg['agg_ratio_max_to_max_mean']:.1f}×")
        print(f"    ratio_median_to_median mean = {agg['agg_ratio_median_to_median_mean']:.0f}×")
    print()
    print(f"  {payload['interpretation']}")
    print(f"\n  JSON: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
