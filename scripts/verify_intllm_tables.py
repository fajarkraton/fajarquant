#!/usr/bin/env python3
"""
verify_intllm_tables.py — §6.9 R7 pre-publication gate for the IntLLM paper.

Cross-checks every numeric claim made in `paper/intllm/tables/*.md` and
in the paper's LaTeX source (once it lands) against the authoritative
JSON artifacts in `paper/intllm/results/*.json`. Run BEFORE every commit
that touches the paper or the results directory.

Created: Phase 4.3 (2026-04-22) of FJQ_PHASE_D_PRODUCTION_PLAN.md.
Companion to `scripts/verify_paper_tables.py` which covers the older
V26 FajarQuant (KV-cache) paper — the two scripts are independent and
cover different artifact sets.

Usage:
    python3 scripts/verify_intllm_tables.py            # human-readable report
    python3 scripts/verify_intllm_tables.py --strict   # exit 2 on mismatch
    python3 scripts/verify_intllm_tables.py --list     # list registered claims
    python3 scripts/verify_intllm_tables.py --coverage # scan tables for unregistered numbers

Each claim is encoded as a Claim dataclass:
    Claim(name, paper_value, source_loader, tolerance, paper_ref)

Exit codes:
    0  all claims PASS (or no --strict)
    2  at least one claim FAIL and --strict is set (plan §4.3 contract)

§6.8 R3 Prevention layer: wire this into .github/workflows/phase-d-verify.yml
as a required check, and into scripts/git-hooks/pre-commit so local edits
to paper/ files are gated the same way.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "paper" / "intllm" / "results"
TABLES_DIR = REPO_ROOT / "paper" / "intllm" / "tables"

# Strict-mode exit code per FJQ_PHASE_D_PRODUCTION_PLAN.md §4.3.
STRICT_FAIL_EXIT = 2


@dataclass
class Claim:
    """One numerical claim in the IntLLM paper, with how to verify it."""

    name: str                               # human-readable
    paper_value: float                      # value stated in paper/tables
    source_loader: Callable[[], float]       # returns the actual value from the JSON artifact
    tolerance: float                         # absolute tolerance for PASS
    paper_ref: str = ""                     # "Table 2 / Base / wikitext_ppl"-style pointer


def load_json(rel_path: str) -> dict[str, Any]:
    """Load a JSON artifact from paper/intllm/results/. Raises FileNotFoundError
    with a clear error if the file is missing — §6.8 R2 runnable-verify means
    the user needs to know exactly which artifact is missing."""
    path = RESULTS_DIR / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Missing results artifact: {path.relative_to(REPO_ROOT)}")
    with path.open() as f:
        return json.load(f)


def bench_canonical(model: str, task: str, metric: str = "acc") -> float:
    """Load a zero-shot benchmark result from bench_canonical_<model>.json.

    Schema (from `scripts/bench_canonical.py`, Phase 3.2):
        {
          "model": "intllm-base",
          "harness_version": "v0.4.11",
          "results": {
            "<task>": {"acc": 0.xxx, "acc_norm": 0.xxx, "stderr": 0.xxx}
          }
        }
    """
    data = load_json(f"bench_canonical_{model}.json")
    return float(data["results"][task][metric])


def bench_knowledge(model: str, task: str, metric: str = "acc") -> float:
    """Like bench_canonical but for the 5-shot/knowledge benchmarks."""
    data = load_json(f"bench_knowledge_{model}.json")
    return float(data["results"][task][metric])


def training_val_loss(model: str) -> float:
    """Load final val_loss from a training gate result."""
    data = load_json(f"training_{model}.json")
    return float(data["val_loss"])


def kernel_e2e_tokens_per_sec(model: str) -> float:
    """Kernel-path throughput, tokens/sec, from the FajarOS E2E gate."""
    data = load_json(f"kernel_e2e_{model}.json")
    return float(data["tokens_per_second"])


def fp16_parity_summary(model: str, key: str) -> float:
    """Aggregate fp16-vs-ternary per-layer rel-error stats. `key` is one
    of: rel_error_l2_mean / rel_error_l2_max / rel_error_l2_p50 /
    rel_error_l2_p95 / rel_error_l2_min (see intllm.fp16_parity.summary_stats)."""
    data = load_json(f"fp16_parity_{model}.json")
    return float(data["summary"][key])


# ─────────────────────────────────────────────────────────────────
# Claim registry — one entry per numeric cell that appears in the
# paper. Populated incrementally as Phase 4.2 fills the tables.
#
# Placeholder entries point at tests/test_verify_smoke_claim.json
# so the script can RUN today (green) before any real paper table exists.
# Real claims land as Phase 2 training + Phase 3 benchmarks finish.
# ─────────────────────────────────────────────────────────────────
CLAIMS: list[Claim] = [
    # ── Smoke-test claim (proves the script plumbing works) ──
    # Sourced from results/smoke.json — always present in the repo as a
    # trivially-green floor. Failing this means the script can't find
    # its artifact directory at all.
    Claim(
        name="Smoke: pi_ish == 3.14",
        paper_value=3.14,
        source_loader=lambda: float(load_json("smoke.json")["pi_ish"]),
        tolerance=0.01,
        paper_ref="(smoke-only, not in paper)",
    ),
    # ── Table 2 main results: per-config PPL (Phase 4.2 Table 2) ──
    # Registered UNPOPULATED with placeholder 0.0 + infinite tolerance so
    # the script does NOT fail today. Each is flipped to a real number
    # when the corresponding training+eval artifact lands.
    #
    # To activate a placeholder: edit `paper_value` to the paper's number,
    # edit `source_loader` to point at the right JSON key, shrink
    # `tolerance` to 0.01, and commit with `feat(v31-c-p4.2):`.
    Claim("Table 2: IntLLM-Mini wikitext PPL",
          paper_value=342.9784,
          source_loader=lambda: bench_canonical("intllm-mini", "wikitext", "word_perplexity"),
          tolerance=0.01,
          paper_ref="Table 2 / Mini / wikitext_ppl (real, mini_final.pt 21.5M, 8-task lm-eval v0.4.11)"),
    Claim("Table 2: IntLLM-Base wikitext PPL",
          paper_value=201.0904,
          source_loader=lambda: bench_canonical("intllm-base", "wikitext", "word_perplexity"),
          tolerance=0.01,
          paper_ref="Table 2 / Base / wikitext_ppl (real, base_final.pt 46.4M, 8-task lm-eval v0.4.11)"),
    Claim("Table 2: IntLLM-Medium wikitext PPL",
          paper_value=138.3596,
          source_loader=lambda: bench_canonical("intllm-medium", "wikitext", "word_perplexity"),
          tolerance=0.01,
          paper_ref="Table 2 / Medium / wikitext_ppl (real, medium_final.pt 74.5M, 8-task lm-eval v0.4.11)"),
    # HellaSwag per-config zero-shot accuracy (real numbers from same bench runs)
    Claim("Table 2: IntLLM-Mini hellaswag",
          0.2546, lambda: bench_canonical("intllm-mini", "hellaswag", "acc_norm"),
          0.01, "Table 2 / Mini / hellaswag (acc_norm)"),
    Claim("Table 2: IntLLM-Base hellaswag",
          0.2624, lambda: bench_canonical("intllm-base", "hellaswag", "acc_norm"),
          0.01, "Table 2 / Base / hellaswag (acc_norm)"),
    # ── Table 4 FajarOS E2E (Phase 4.2 Table 4) ──
    Claim("Table 4: kernel E2E Mini tok/s",
          0.0, lambda: kernel_e2e_tokens_per_sec("intllm-mini"),
          float("inf"), "Table 4 / Mini / tok_s"),
    # ── Training gates (FJQ_PHASE_D_GATE_CALIBRATION.md) ──
    # These are the val_loss ceiling gates; when a training run lands,
    # flip the paper_value to the actual and tolerance to 0.01.
    Claim("Gate: Mini val_loss",
          4.3822, lambda: training_val_loss("intllm-mini"),
          0.01, "gate Mini<4.5 (PASS by 0.118 nat margin)"),
    Claim("Gate: Base val_loss (c.1 982M tok Chinchilla)",
          3.9903, lambda: training_val_loss("intllm-base"),
          0.01, "gate Base<4.2 (c.1 PASS by 0.21 nat; supersedes c.2 which PASSed by only 0.071 nat)"),
    Claim("Gate: Medium val_loss (c.1 1.819B tok ~Chinchilla)",
          3.7211, lambda: training_val_loss("intllm-medium"),
          0.01, "gate Medium<4.0 (c.1 PASS by 0.28 nat margin; widest in Phase D scaling chain)"),
    # ── Table 3 fp16-vs-ternary parity (Phase 3.5 IntLLM differentiator) ──
    # First measured 2026-04-22 on Mini v2. BitNet 2B4T did not ship a
    # parity gate; high absolute rel-error is expected for 1.58-bit
    # weights, but the *distribution* + the gate scaffold IS the
    # contribution. Threshold-setting deferred to follow-up commit.
    Claim("Table 3: Mini fp16 parity rel_l2 mean",
          2.3104, lambda: fp16_parity_summary("intllm-mini", "rel_error_l2_mean"),
          0.01, "Table 3 / Mini / rel_l2_mean (37 hooks)"),
    Claim("Table 3: Mini fp16 parity rel_l2 p50",
          0.8162, lambda: fp16_parity_summary("intllm-mini", "rel_error_l2_p50"),
          0.01, "Table 3 / Mini / rel_l2_p50 (median, scale-free)"),
    Claim("Table 3: Mini fp16 parity rel_l2 max",
          19.3093, lambda: fp16_parity_summary("intllm-mini", "rel_error_l2_max"),
          0.01, "Table 3 / Mini / rel_l2_max (worst layer)"),
]


def cmd_list(_args: argparse.Namespace) -> int:
    print(f"Registered claims ({len(CLAIMS)}):")
    for i, c in enumerate(CLAIMS, 1):
        ref = c.paper_ref or "(no ref)"
        tol = "unpopulated" if c.tolerance == float("inf") else f"tol={c.tolerance}"
        print(f"  [{i:2d}] {c.name}  →  {ref}  [{tol}]")
    return 0


def cmd_coverage(_args: argparse.Namespace) -> int:
    """Scan `paper/intllm/tables/*.md` for numbers and report any that
    aren't covered by at least one registered Claim.

    Heuristic: a "numeric cell" is any standalone float literal inside a
    markdown-table row. Integers that are purely structural (2, 4, 8, d=128)
    are best excluded by listing them in a per-table whitelist — not
    implemented in the skeleton; Phase 4.2 adds them.
    """
    if not TABLES_DIR.exists():
        print(f"[INFO] {TABLES_DIR.relative_to(REPO_ROOT)} does not exist yet; nothing to scan.")
        return 0

    md_files = sorted(TABLES_DIR.glob("*.md"))
    if not md_files:
        print(f"[INFO] {TABLES_DIR.relative_to(REPO_ROOT)} is empty; nothing to scan.")
        return 0

    float_re = re.compile(r"\|\s*(-?\d+\.\d+)\s*\|")
    registered_values = {round(c.paper_value, 4) for c in CLAIMS if c.tolerance != float("inf")}

    unregistered: list[tuple[Path, int, str]] = []
    for f in md_files:
        for lineno, line in enumerate(f.read_text().splitlines(), 1):
            for m in float_re.finditer(line):
                v = round(float(m.group(1)), 4)
                if v not in registered_values:
                    unregistered.append((f, lineno, m.group(1)))

    if not unregistered:
        print(f"[OK] Every number in {TABLES_DIR.relative_to(REPO_ROOT)} is covered.")
        return 0

    print(f"[WARN] {len(unregistered)} number(s) in tables are NOT in CLAIMS:")
    for f, lineno, val in unregistered:
        print(f"  {f.relative_to(REPO_ROOT)}:{lineno}  {val}")
    print("\nFix: register these in scripts/verify_intllm_tables.py::CLAIMS or "
          "move them outside the tables/ scope.")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    print(f"verify_intllm_tables.py — checking {len(CLAIMS)} claims against "
          f"{RESULTS_DIR.relative_to(REPO_ROOT)}")
    print("=" * 78)

    fails = 0
    unpopulated = 0
    for claim in CLAIMS:
        try:
            actual = claim.source_loader()
        except FileNotFoundError as e:
            if claim.tolerance == float("inf"):
                print(f"  PEND   {claim.name}: (placeholder — "
                      f"{Path(str(e).split(': ', 1)[-1]).name} not yet committed)")
                unpopulated += 1
                continue
            print(f"  ERROR  {claim.name}: source artifact missing → {e}")
            fails += 1
            continue
        except (KeyError, json.JSONDecodeError) as e:
            print(f"  ERROR  {claim.name}: source data malformed → {e}")
            fails += 1
            continue

        diff = abs(actual - claim.paper_value)
        ok = diff <= claim.tolerance
        ref = f" ({claim.paper_ref})" if claim.paper_ref else ""
        if claim.tolerance == float("inf"):
            # Placeholder — we don't assert anything yet, just report the
            # observed value so reviewers can see the pipeline works.
            print(f"  OBS    {claim.name}{ref}: source={actual:.4f} "
                  f"(placeholder — populate paper_value + tolerance to activate)")
            unpopulated += 1
            continue

        marker = "  PASS " if ok else "  FAIL "
        print(f"{marker}{claim.name}{ref}: "
              f"paper={claim.paper_value} source={actual:.4f} "
              f"diff={diff:.4f} tol={claim.tolerance}")
        if not ok:
            fails += 1

    print("=" * 78)
    activated = len(CLAIMS) - unpopulated
    if fails == 0:
        print(f"OK — {activated} activated claims verified within tolerance; "
              f"{unpopulated} placeholder(s) pending Phase 2/3 results.")
        return 0

    print(f"FAIL — {fails} of {activated} activated claim(s) failed verification")
    if args.strict:
        return STRICT_FAIL_EXIT
    print(f"(re-run with --strict to exit {STRICT_FAIL_EXIT} on mismatches)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--strict", action="store_true",
                    help=f"exit {STRICT_FAIL_EXIT} on any mismatch (pre-commit hook mode)")
    ap.add_argument("--list", dest="subcommand", action="store_const", const="list",
                    help="list registered claims instead of verifying")
    ap.add_argument("--coverage", dest="subcommand", action="store_const", const="coverage",
                    help="scan paper tables for unregistered numbers")
    args = ap.parse_args()

    if getattr(args, "subcommand", None) == "list":
        return cmd_list(args)
    if getattr(args, "subcommand", None) == "coverage":
        return cmd_coverage(args)
    return cmd_verify(args)


if __name__ == "__main__":
    sys.exit(main())
