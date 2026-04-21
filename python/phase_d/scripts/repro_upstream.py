#!/usr/bin/env python3
"""V31.C.P2.1 FULL gate — reproduce Zhu et al. Table 1 zero-shot accuracies
on `ridger/MMfreeLM-370M`.

Target: average ≤ ±1 of paper Table 1 value per §6.9 R3 (baseline parity).

Paper Table 1 "MatMul-free 370M" row (Zhu et al. 2024):
    ARC-E:       42.6
    ARC-C:       23.8
    HellaSwag:   32.8
    OpenbookQA:  28.4
    PIQA:        63.0
    Winogrande:  49.2
    Avg:         40.3

Run:
    cd fajarquant/python/phase_d
    ../../.venv/bin/python scripts/repro_upstream.py

Outputs:
    scripts/repro_upstream_results.json   — raw lm_eval output
    scripts/repro_upstream_results.md     — human-readable comparison table
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Do NOT set HF_HUB_ENABLE_HF_TRANSFER=1 here; it requires the optional
# `hf_transfer` package which isn't in the minimum install set.

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parent / "_upstream"
if not UPSTREAM.is_dir():
    sys.exit(f"fatal: vendored snapshot missing at {UPSTREAM}. See UPSTREAM_PIN.md.")
sys.path.insert(0, str(UPSTREAM))

import mmfreelm  # noqa: F401, E402  — registers HGRNBit* model classes
from lm_eval import simple_evaluate  # noqa: E402


MODEL_NAME = "ridger/MMfreeLM-370M"

# Tasks match Zhu et al. Table 1 exactly.
TASKS = ["arc_easy", "arc_challenge", "hellaswag", "openbookqa", "piqa", "winogrande"]

# Paper Table 1 target accuracies (percent).
# Metric = acc_norm for arc_* and hellaswag/openbookqa; acc for piqa/winogrande.
PAPER_TABLE_1 = {
    "arc_easy": 42.6,
    "arc_challenge": 23.8,
    "hellaswag": 32.8,
    "openbookqa": 28.4,
    "piqa": 63.0,
    "winogrande": 49.2,
}
PAPER_AVG = 40.3
TOLERANCE = 1.0  # §6.9 R3: ±1 avg point baseline parity

# Metric key per task — matches lm-eval-harness output conventions.
METRIC_KEY = {
    "arc_easy": "acc_norm",
    "arc_challenge": "acc_norm",
    "hellaswag": "acc_norm",
    "openbookqa": "acc_norm",
    "piqa": "acc",
    "winogrande": "acc",
}


def main() -> int:
    print(f"[1/3] running lm_eval on {MODEL_NAME}, tasks={TASKS}")
    t0 = time.time()
    results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={MODEL_NAME},dtype=float16",
        tasks=TASKS,
        batch_size=8,
        device="cuda",
    )
    elapsed = time.time() - t0
    print(f"      elapsed {elapsed:.1f} s ({elapsed / 60:.1f} min)")

    # Extract per-task accuracy
    our_results = {}
    for task in TASKS:
        metric = METRIC_KEY[task]
        task_scores = results["results"].get(task, {})
        # lm_eval reports keys like "acc_norm,none" or "acc_norm"
        val = None
        for k in (f"{metric},none", metric):
            if k in task_scores:
                val = task_scores[k]
                break
        if val is None:
            print(f"      WARN: task {task} metric {metric} not found, keys: {list(task_scores.keys())}")
            val = float("nan")
        our_results[task] = val * 100.0  # lm_eval returns fraction; paper is percent

    our_avg = sum(our_results.values()) / len(our_results)
    avg_delta = our_avg - PAPER_AVG
    passed = abs(avg_delta) <= TOLERANCE

    # Build markdown comparison
    print(f"\n[2/3] comparison (paper Table 1 vs. our repro, ±{TOLERANCE} avg gate):")
    header = f"| {'Task':<15} | {'Paper':>8} | {'Ours':>8} | {'Δ':>7} |"
    sep = "|" + "-" * 17 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 9 + "|"
    print(header)
    print(sep)
    for task in TASKS:
        paper = PAPER_TABLE_1[task]
        ours = our_results[task]
        delta = ours - paper
        print(f"| {task:<15} | {paper:>8.2f} | {ours:>8.2f} | {delta:>+7.2f} |")
    print(sep)
    print(f"| {'AVERAGE':<15} | {PAPER_AVG:>8.2f} | {our_avg:>8.2f} | {avg_delta:>+7.2f} |")
    print(f"\nGate §6.9 R3: {'PASS ✓' if passed else 'FAIL ✗'} (|Δ avg| = {abs(avg_delta):.2f}, tol {TOLERANCE})")

    # Dump results
    json_path = HERE / "repro_upstream_results.json"
    md_path = HERE / "repro_upstream_results.md"

    artifact = {
        "model": MODEL_NAME,
        "tasks": TASKS,
        "paper_table_1": PAPER_TABLE_1,
        "paper_avg": PAPER_AVG,
        "our_results": our_results,
        "our_avg": our_avg,
        "avg_delta": avg_delta,
        "tolerance": TOLERANCE,
        "gate_pass": passed,
        "elapsed_seconds": elapsed,
    }
    with json_path.open("w") as f:
        json.dump(artifact, f, indent=2)

    lines = [
        "# V31.C.P2.1 FULL gate — MatMul-free 370M repro",
        "",
        f"**Model:** `{MODEL_NAME}`",
        f"**Elapsed:** {elapsed:.1f} s ({elapsed / 60:.1f} min) on RTX 4090 Laptop",
        f"**Gate:** §6.9 R3 baseline parity, ±{TOLERANCE} avg point",
        "",
        "| Task | Paper Table 1 | Our repro | Δ |",
        "|---|---:|---:|---:|",
    ]
    for task in TASKS:
        paper = PAPER_TABLE_1[task]
        ours = our_results[task]
        delta = ours - paper
        lines.append(f"| {task} | {paper:.2f} | {ours:.2f} | {delta:+.2f} |")
    lines.extend([
        f"| **AVERAGE** | **{PAPER_AVG:.2f}** | **{our_avg:.2f}** | **{avg_delta:+.2f}** |",
        "",
        f"**Result:** {'PASS ✓' if passed else 'FAIL ✗'} (|Δ avg| = {abs(avg_delta):.2f})",
    ])
    md_path.write_text("\n".join(lines) + "\n")

    print(f"\n[3/3] artifacts written:\n      {json_path}\n      {md_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
