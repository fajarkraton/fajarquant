#!/usr/bin/env python3
"""
verify_paper_tables.py — Plan Hygiene Rule 3 prevention layer for paper claims.

Cross-checks numerical claims in paper/fajarquant.tex against source data
in data/kv_cache/*.json. Run this whenever the paper or source data changes.

Created: V26 Phase C0 (2026-04-11) as the C0.5 verification deliverable.
Lives in fajarquant repo because the verifications target this repo's data.

Usage:
    python3 scripts/verify_paper_tables.py            # human-readable report
    python3 scripts/verify_paper_tables.py --strict   # exit 1 on any mismatch

Each claim is encoded as a tuple:
    (description, paper_value, source_path, tolerance, rounding_digits)

The script reports both PASS and FAIL for transparency. A mismatch beyond
tolerance is FAIL; a mismatch within tolerance is WARN.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "kv_cache"


@dataclass
class Claim:
    """One numerical claim made in the paper, with how to verify it."""

    name: str                       # human-readable claim name
    paper_value: float              # value the paper states
    source_loader: Callable[[], float]  # function returning the actual source value
    tolerance: float                # absolute tolerance for PASS
    paper_line: int = 0             # paper line number (0 = unknown)


def load_json(rel_path: str) -> dict[str, Any]:
    path = DATA_DIR / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Missing source data: {path}")
    with path.open() as f:
        return json.load(f)


def ppl_v2(model: str, method: str) -> float:
    """Load PPL from canonical v2 results."""
    data = load_json(f"perplexity_v2_{model}.json")
    return float(data[method]["ppl"])


def v1v2_improvement_pct(model: str) -> float:
    """Compute (v1 - v2) / v1 * 100 at 2-bit."""
    data = load_json(f"perplexity_v2_{model}.json")
    v1 = data["fajarquant_2bit"]["ppl"]
    v2 = data["fajarquant_v2_2bit"]["ppl"]
    return (v1 - v2) / v1 * 100


def fqv2_vs_tq_pct(model: str) -> float:
    """Compute (TQ - FQv2) / TQ * 100 at 2-bit (positive = FQ better)."""
    data = load_json(f"perplexity_v2_{model}.json")
    tq = data["turboquant_outlier_2bit"]["ppl"]
    fq = data["fajarquant_v2_2bit"]["ppl"]
    return (tq - fq) / tq * 100


# ─────────────────────────────────────────────────────────────────
# Claims registered for reframed paper (V26 C1.6 B6, 2026-04-13)
# Source: data/kv_cache/perplexity_v2_{gemma,mistral,qwen2}.json
# ─────────────────────────────────────────────────────────────────
CLAIMS: list[Claim] = [
    # ── tab:ppl_crossmodel — Gemma ──
    Claim("Gemma FP16", 28.13, lambda: ppl_v2("gemma", "fp16"), 0.01),
    Claim("Gemma FQ v1 2-bit", 125.19, lambda: ppl_v2("gemma", "fajarquant_2bit"), 0.01),
    Claim("Gemma FQ v2 2-bit", 46.20, lambda: ppl_v2("gemma", "fajarquant_v2_2bit"), 0.01),
    Claim("Gemma KIVI 2-bit", 470.50, lambda: ppl_v2("gemma", "kivi_2bit"), 0.01),
    Claim("Gemma TQ outlier 2-bit", 39.73, lambda: ppl_v2("gemma", "turboquant_outlier_2bit"), 0.01),
    Claim("Gemma KIVI 3-bit", 21.90, lambda: ppl_v2("gemma", "kivi_3bit"), 0.01),
    Claim("Gemma FQ v2 3-bit", 24.55, lambda: ppl_v2("gemma", "fajarquant_v2_3bit"), 0.01),
    Claim("Gemma KIVI 4-bit", 35.17, lambda: ppl_v2("gemma", "kivi_4bit"), 0.01),
    # ── tab:ppl_crossmodel — Mistral ──
    Claim("Mistral FP16", 5.67, lambda: ppl_v2("mistral", "fp16"), 0.01),
    Claim("Mistral KIVI 2-bit", 23.96, lambda: ppl_v2("mistral", "kivi_2bit"), 0.01),
    Claim("Mistral FQ v2 2-bit", 208.16, lambda: ppl_v2("mistral", "fajarquant_v2_2bit"), 0.01),
    Claim("Mistral KIVI 3-bit", 5.99, lambda: ppl_v2("mistral", "kivi_3bit"), 0.01),
    Claim("Mistral KIVI 4-bit", 5.73, lambda: ppl_v2("mistral", "kivi_4bit"), 0.01),
    # ── tab:ppl_crossmodel — Qwen2 ──
    Claim("Qwen2 FP16", 7.55, lambda: ppl_v2("qwen2", "fp16"), 0.01),
    Claim("Qwen2 FQ v2 2-bit", 50.10, lambda: ppl_v2("qwen2", "fajarquant_v2_2bit"), 0.01),
    Claim("Qwen2 KIVI 2-bit", 46.70, lambda: ppl_v2("qwen2", "kivi_2bit"), 0.01),
    Claim("Qwen2 TQ outlier 2-bit", 165.60, lambda: ppl_v2("qwen2", "turboquant_outlier_2bit"), 0.01),
    Claim("Qwen2 KIVI 3-bit", 8.01, lambda: ppl_v2("qwen2", "kivi_3bit"), 0.01),
    Claim("Qwen2 KIVI 4-bit", 7.62, lambda: ppl_v2("qwen2", "kivi_4bit"), 0.01),
    # ── Abstract claims ──
    Claim("Abstract: FQ v2 69.7% better on Qwen2 2-bit", 69.7, lambda: fqv2_vs_tq_pct("qwen2"), 0.5),
    Claim("Abstract: v1→v2 improvement Gemma ≥63%", 63.1, lambda: v1v2_improvement_pct("gemma"), 0.5),
    Claim("Abstract: v1→v2 improvement Mistral ≥76%", 76.0, lambda: v1v2_improvement_pct("mistral"), 0.5),
    Claim("Abstract: v1→v2 improvement Qwen2 ≥80%", 80.9, lambda: v1v2_improvement_pct("qwen2"), 0.5),
    # ── tab:v1v2_ablation (Gemma 2-bit) ──
    Claim("Ablation: FQ v1 Gemma 2-bit", 125.19, lambda: ppl_v2("gemma", "fajarquant_2bit"), 0.01),
    Claim("Ablation: FQ v2a Gemma 2-bit", 161.31, lambda: ppl_v2("gemma", "fajarquant_v2a_2bit"), 0.01),
    Claim("Ablation: FQ v2 Gemma 2-bit", 46.20, lambda: ppl_v2("gemma", "fajarquant_v2_2bit"), 0.01),
    # ── Protocol parameters ──
    Claim("Protocol: 30 chunks", 30, lambda: float(load_json("perplexity_v2_gemma.json")["fp16"]["samples"]), 0),
    Claim("Protocol: 61410 tokens", 61410, lambda: float(load_json("perplexity_v2_gemma.json")["fp16"]["tokens"]), 0),
]


def main() -> int:
    strict = "--strict" in sys.argv

    print(f"verify_paper_tables.py — checking {len(CLAIMS)} claims against {DATA_DIR}")
    print("=" * 78)

    fails = 0
    for claim in CLAIMS:
        try:
            actual = claim.source_loader()
        except Exception as e:
            print(f"  ERROR  {claim.name}: source data load failed → {e}")
            fails += 1
            continue

        diff = abs(actual - claim.paper_value)
        ok = diff <= claim.tolerance
        marker = "  PASS " if ok else "  FAIL "
        line = f" (paper line {claim.paper_line})" if claim.paper_line else ""
        print(
            f"{marker}{claim.name}{line}: "
            f"paper={claim.paper_value} source={actual:.4f} "
            f"diff={diff:.4f} tol={claim.tolerance}"
        )
        if not ok:
            fails += 1

    print("=" * 78)
    if fails == 0:
        print(f"OK — all {len(CLAIMS)} claims verified within tolerance")
        return 0

    print(f"FAIL — {fails} of {len(CLAIMS)} claims failed verification")
    if strict:
        return 1
    print("(re-run with --strict to exit non-zero on mismatches)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
