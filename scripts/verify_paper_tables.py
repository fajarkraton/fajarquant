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


def perplexity(method: str, bits: int) -> float:
    data = load_json("perplexity_results.json")
    return float(data[f"{method}_{bits}bit"]["ppl"])


def hierarchical_savings(seq_len_key: str) -> float:
    data = load_json("ablation_results.json")
    return float(data["hierarchical_ablation"][seq_len_key]["savings_pct"])


def pca_vs_turbo_pct(bits: int) -> float:
    data = load_json("comparison_results.json")
    return float(data[f"{bits}bit"]["key"]["fq_vs_turbo_pct"])


def pca_vs_kivi_pct(bits: int) -> float:
    data = load_json("comparison_results.json")
    return float(data[f"{bits}bit"]["key"]["fq_vs_kivi_pct"])


# ─────────────────────────────────────────────────────────────────
# Claims registered as of paper version 2026-04-11 (V26 Phase C0)
# Add new claims when the paper changes; this list is the source
# of truth for what verify_paper_tables.py guards.
# ─────────────────────────────────────────────────────────────────
CLAIMS: list[Claim] = [
    # Perplexity claims (paper line 39, 211, 290, 291)
    Claim(
        name="PPL 2-bit FajarQuant (abstract + Table)",
        paper_value=80.1,
        source_loader=lambda: perplexity("fajarquant", 2),
        tolerance=0.1,
        paper_line=39,
    ),
    Claim(
        name="PPL 2-bit TurboQuant",
        paper_value=117.1,
        source_loader=lambda: perplexity("turboquant", 2),
        tolerance=0.1,
        paper_line=39,
    ),
    Claim(
        name="PPL 2-bit KIVI",
        paper_value=231.9,
        source_loader=lambda: perplexity("kivi", 2),
        tolerance=0.1,
        paper_line=39,
    ),
    Claim(
        name="PPL 3-bit FajarQuant",
        paper_value=75.6,
        source_loader=lambda: perplexity("fajarquant", 3),
        tolerance=0.1,
        paper_line=291,
    ),
    # Hierarchical savings claims (paper lines 37, 175, 266, 271, 339)
    Claim(
        name="Hierarchical 48.7% at 10K context",
        paper_value=48.7,
        source_loader=lambda: hierarchical_savings("N=10000"),
        tolerance=0.05,
        paper_line=37,
    ),
    Claim(
        name="Hierarchical 55.7% at 16K context",
        paper_value=55.7,
        source_loader=lambda: hierarchical_savings("N=16384"),
        tolerance=0.05,
        paper_line=271,
    ),
    # PCA per-bit improvements (paper line 271)
    Claim(
        name="PCA vs TurboQuant 2-bit (4.9%)",
        paper_value=4.9,
        source_loader=lambda: pca_vs_turbo_pct(2),
        tolerance=0.05,
        paper_line=271,
    ),
    Claim(
        name="PCA vs TurboQuant 3-bit (4.3%)",
        paper_value=4.3,
        source_loader=lambda: pca_vs_turbo_pct(3),
        tolerance=0.05,
        paper_line=271,
    ),
    Claim(
        name="PCA vs TurboQuant 4-bit (4.8%)",
        paper_value=4.8,
        source_loader=lambda: pca_vs_turbo_pct(4),
        tolerance=0.05,
        paper_line=271,
    ),
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
