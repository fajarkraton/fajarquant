#!/usr/bin/env python3
"""V31.C.P3.2 — canonical zero-shot benchmark runner (scaffold stage).

Produces paper/intllm/results/bench_canonical_<tag>.json in the schema
consumed by scripts/verify_intllm_tables.py::bench_canonical
(see verify_intllm_tables.py:73-86 for the authoritative schema doc).

Phase 3.2 of FJQ_PHASE_D_PRODUCTION_PLAN.md §3.2. Covers the 8 canonical
zero-shot tasks from lm-evaluation-harness v0.4.11 (pinned in
eval/HARNESS_SHA):

  wikitext, lambada_openai, hellaswag, piqa, winogrande, arc_easy,
  arc_challenge, openbookqa

Two modes:

  --scaffold   Emit a schema-valid JSON with placeholder 0.0 values and
               status='scaffold'. CPU-only, deterministic, <1s. Unblocks
               verify_intllm_tables.py claim registry (whose Table 2
               entries today use tolerance=inf, so a 0.0 scaffold value
               PASSes — tolerance gets tightened by Phase 4.2 when the
               cell is populated with a real number.)

  (default)    Real lm_eval run on --checkpoint. Requires an LM wrapper
               around HGRNBitForCausalLM exposing lm_eval's LM interface
               (loglikelihood + generate_until). The wrapper lands in a
               follow-up commit (Phase 3.2.1) once Base c.1 finishes;
               for now the real path raises NotImplementedError with a
               clear pointer to the TODO.

Scaffold mode is the deliverable for THIS commit — it guarantees
`make bench-canonical` has a green, runnable target today, so downstream
Phase 4 work (paper Table 2) can iterate on the verification plumbing
independently of GPU training schedules.

Usage:
    cd fajarquant/python/phase_d
    PYTHONPATH=. ../../.venv/bin/python scripts/bench_canonical.py \\
        --tag intllm-mini --scaffold

    # Or via top-level make (Phase 3.2):
    make bench-canonical              # scaffolds all 3 sizes
    make bench-canonical TAG=mini     # scaffold one tag

Exit codes:
    0  artifact written (scaffold or real)
    2  real mode requested but LM wrapper not yet implemented
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO_ROOT = ROOT.parent.parent

# Canonical task list — Phase 3.2 spec line FJQ_PHASE_D_PRODUCTION_PLAN.md:283.
# Per-task metric set mirrors lm-evaluation-harness v0.4.11 output keys for
# each task. Scaffold mode emits all of these; real mode subsets to what
# simple_evaluate actually returns.
CANONICAL_TASKS: dict[str, list[str]] = {
    "wikitext":       ["word_perplexity", "byte_perplexity", "bits_per_byte"],
    "lambada_openai": ["acc", "perplexity"],
    "hellaswag":      ["acc", "acc_norm"],
    "piqa":           ["acc", "acc_norm"],
    "winogrande":     ["acc"],
    "arc_easy":       ["acc", "acc_norm"],
    "arc_challenge":  ["acc", "acc_norm"],
    "openbookqa":     ["acc", "acc_norm"],
}

# Pinned harness version — must stay in sync with eval/HARNESS_SHA.
HARNESS_VERSION = "v0.4.11"


def load_harness_ref() -> str:
    """Read the pinned lm-eval ref from eval/HARNESS_SHA. Returns e.g. 'v0.4.11'."""
    sha_file = REPO_ROOT / "eval" / "HARNESS_SHA"
    if not sha_file.exists():
        return HARNESS_VERSION
    for line in sha_file.read_text().splitlines():
        s = line.strip()
        if s.startswith("ref:"):
            return s.split(None, 1)[1].strip()
    return HARNESS_VERSION


def scaffold_results(tasks: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    """Emit placeholder results for every (task, metric) pair.

    Uses 0.0 (not NaN or None) so verify_intllm_tables.py can load the JSON
    without raising — its Table 2 claims currently register tolerance=inf,
    so 0.0 passes trivially. When a claim gets tightened, the scaffold
    value will be overwritten by a real bench run.
    """
    out: dict[str, dict[str, float]] = {}
    for task, metrics in tasks.items():
        out[task] = {}
        for m in metrics:
            out[task][m] = 0.0
            # stderr is a standard lm-eval companion key; emit it so
            # downstream code doesn't KeyError.
            out[task][f"{m}_stderr"] = 0.0
    return out


def run_real(checkpoint: Path, tasks: list[str]) -> dict[str, dict[str, float]]:
    """Run lm-evaluation-harness against a trained IntLLM checkpoint.

    NOT YET IMPLEMENTED. Wrapping HGRNBitForCausalLM as an lm_eval.LM requires
    implementing loglikelihood() + generate_until() with the model's
    non-standard linear-attention forward. Lands in Phase 3.2.1 follow-up.

    The stub below documents the expected call shape so the LM-wrapper
    author can drop in without restructuring this runner.
    """
    raise NotImplementedError(
        "Real lm_eval run not yet implemented — LM wrapper for "
        "HGRNBitForCausalLM lands in Phase 3.2.1 follow-up. "
        "Track: FJQ_PHASE_D_PRODUCTION_PLAN.md §3.2. "
        "For now, re-run with --scaffold to emit schema-valid placeholder "
        "JSON so verify_intllm_tables.py plumbing stays green."
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tag", required=True,
                   help="Output tag, e.g. 'intllm-mini' for "
                        "paper/intllm/results/bench_canonical_intllm-mini.json")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Path to .pt checkpoint (real mode). If omitted or "
                        "missing, falls back to scaffold unless --strict.")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Subset of canonical tasks to run. Default: all 8.")
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "paper" / "intllm" / "results",
                   help="Where to write bench_canonical_<tag>.json")
    p.add_argument("--scaffold", action="store_true",
                   help="Force scaffold mode (placeholder results, no GPU).")
    p.add_argument("--strict", action="store_true",
                   help="Fail if scaffold mode would be chosen implicitly.")
    args = p.parse_args()

    # Resolve task set
    if args.tasks:
        selected = {t: CANONICAL_TASKS[t] for t in args.tasks if t in CANONICAL_TASKS}
        missing = [t for t in args.tasks if t not in CANONICAL_TASKS]
        if missing:
            print(f"WARN: unknown tasks ignored: {missing}", file=sys.stderr)
        if not selected:
            print(f"FATAL: no valid tasks selected from {args.tasks}", file=sys.stderr)
            return 2
    else:
        selected = CANONICAL_TASKS

    # Mode selection
    use_scaffold = args.scaffold or args.checkpoint is None or not args.checkpoint.exists()
    if use_scaffold and args.strict:
        reason = "no --checkpoint" if args.checkpoint is None else f"checkpoint {args.checkpoint} missing"
        print(f"FATAL (--strict): would scaffold because {reason}", file=sys.stderr)
        return 2

    harness_ref = load_harness_ref()
    t0 = time.time()

    if use_scaffold:
        results = scaffold_results(selected)
        status = "scaffold"
        print(f"[scaffold] tag={args.tag} tasks={list(selected)} "
              f"(harness={harness_ref})")
    else:
        try:
            results = run_real(args.checkpoint, list(selected))
        except NotImplementedError as e:
            print(f"FATAL: {e}", file=sys.stderr)
            return 2
        status = "real"

    elapsed = time.time() - t0

    payload = {
        "_description": (
            "Canonical zero-shot benchmark results. Schema consumed by "
            "scripts/verify_intllm_tables.py::bench_canonical. "
            "Run via scripts/bench_canonical.py."
        ),
        "_phase": "FJQ_PHASE_D_PRODUCTION_PLAN.md §3.2 (canonical benchmarks)",
        "_plan_ref": "paper Table 2",
        "model": args.tag,
        "harness_version": harness_ref,
        "status": status,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "tasks": list(selected),
        "elapsed_seconds": round(elapsed, 3),
        "results": results,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"bench_canonical_{args.tag}.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"artifact: {out_path.relative_to(REPO_ROOT) if out_path.is_relative_to(REPO_ROOT) else out_path}")
    print(f"status:   {status}")
    print(f"tasks:    {len(selected)}  ({', '.join(selected)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
