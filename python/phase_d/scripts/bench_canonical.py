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

  (default)    Real lm_eval run on --checkpoint. Uses intllm.lm_eval_wrapper
               to load the checkpoint into an HFLM adapter (Phase 3.2.1),
               then runs lm_eval.simple_evaluate on the selected tasks.
               Emits JSON with status='real' and populated metric values.

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


def run_real(
    checkpoint: Path,
    tasks: list[str],
    device: str = "cuda",
    batch_size: int = 4,
    max_length: int | None = None,
    limit: int | float | None = None,
) -> dict[str, dict[str, float]]:
    """Run lm-evaluation-harness against a trained IntLLM checkpoint.

    Phase 3.2.1: HGRNBitForCausalLM is HF-compatible (inherits from
    `transformers.PreTrainedModel` + `GenerationMixin`) so we pass it
    directly to lm_eval's HFLM adapter rather than writing a custom LM
    subclass. See `intllm.lm_eval_wrapper.build_hflm` for details.

    Returns results in the canonical bench_canonical schema.
    """
    # Deferred imports — heavy (torch, lm_eval, transformers). Keeps
    # --scaffold path fast + avoids forcing a GPU env on CI.
    from intllm.lm_eval_wrapper import build_hflm, extract_results
    from lm_eval import simple_evaluate

    lm = build_hflm(
        checkpoint,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    out = simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=0,
        limit=limit,
        log_samples=False,
        bootstrap_iters=100,  # default 100_000 is overkill for our sample sizes
    )
    return extract_results(out)


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
    p.add_argument("--device", default="cuda",
                   help="torch device for real mode (default: cuda)")
    p.add_argument("--batch-size", type=int, default=4,
                   help="lm_eval HFLM batch_size for real mode (default: 4)")
    p.add_argument("--max-length", type=int, default=None,
                   help="override max_length passed to HFLM (default: "
                        "arch.max_position_embeddings from checkpoint)")
    p.add_argument("--limit", type=float, default=None,
                   help="lm_eval `limit` — cap docs per task (smoke-test). "
                        "If <1, treated as a fraction; if ≥1, absolute count.")
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
        print(f"[real] tag={args.tag} tasks={list(selected)} "
              f"device={args.device} batch_size={args.batch_size} "
              f"limit={args.limit} (harness={harness_ref})")
        real = run_real(
            args.checkpoint,
            list(selected),
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            limit=args.limit,
        )
        # Merge into a full-schema result so partial-task runs (e.g. smoke
        # with --tasks piqa) still preserve placeholder entries for the
        # other tasks — keeps verify_intllm_tables.py able to find every
        # registered (task, metric) pair.
        results = scaffold_results(CANONICAL_TASKS)
        for task, metrics in real.items():
            results[task] = metrics
        status = "real" if set(selected) == set(CANONICAL_TASKS) else "real-partial"

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
        "limit": args.limit,
        "batch_size": args.batch_size if status == "real" else None,
        "device": args.device if status == "real" else None,
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
