#!/usr/bin/env python3
"""V31.C.P3.3 — knowledge / few-shot benchmark runner (scaffold + real).

Covers the 3-task knowledge suite matching BitNet 2B4T Table 1:

  - mmlu      (5-shot, summary metric: acc)
  - triviaqa  (5-shot, metric: exact_match)
  - boolq     (0-shot, metric: acc)

Phase 3.3 of FJQ_PHASE_D_PRODUCTION_PLAN.md §3.3. Output schema consumed
by scripts/verify_intllm_tables.py::bench_knowledge (line 89).

Schema (matches bench_canonical, different task set + fewshot):
    {
      "model": "intllm-base",
      "harness_version": "v0.4.11",
      "status": "scaffold" | "real-partial" | "real",
      "results": {
        "<task>": {"<metric>": float, "<metric>_stderr": float, ...}
      },
      ...
    }

Two modes — same pattern as bench_canonical.py:

  --scaffold   Placeholder 0.0 values, CPU-only, deterministic, <1s.
  (default)    Real lm_eval run via intllm.lm_eval_wrapper.build_hflm.
               Runs TWO simple_evaluate calls — one for 5-shot tasks
               (mmlu + triviaqa) and one for 0-shot (boolq) — merged.

Usage:
    cd fajarquant/python/phase_d
    PYTHONPATH=. ../../.venv/bin/python scripts/bench_knowledge.py \\
        --tag intllm-mini --scaffold

    # via top-level make:
    make bench-knowledge                         # scaffold all 3 sizes
    make bench-knowledge-real TAG=mini LIMIT=10 TASKS="boolq"

Exit codes:
    0   artifact written
    2   --strict but checkpoint missing
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

# Per-task metric set + default fewshot. Metric names track lm-eval-harness
# v0.4.11 output keys. Fewshot matches BitNet 2B4T Table 1 (mmlu 5-shot,
# triviaqa 5-shot, boolq 0-shot).
KNOWLEDGE_TASKS: dict[str, dict] = {
    "mmlu":      {"metrics": ["acc"],          "num_fewshot": 5},
    "triviaqa":  {"metrics": ["exact_match"],  "num_fewshot": 5},
    "boolq":     {"metrics": ["acc"],          "num_fewshot": 0},
}

HARNESS_VERSION = "v0.4.11"


def load_harness_ref() -> str:
    sha_file = REPO_ROOT / "eval" / "HARNESS_SHA"
    if not sha_file.exists():
        return HARNESS_VERSION
    for line in sha_file.read_text().splitlines():
        s = line.strip()
        if s.startswith("ref:"):
            return s.split(None, 1)[1].strip()
    return HARNESS_VERSION


def scaffold_results(tasks: dict[str, dict]) -> dict[str, dict[str, float]]:
    """Placeholder 0.0 values for every (task, metric) pair."""
    out: dict[str, dict[str, float]] = {}
    for task, spec in tasks.items():
        out[task] = {}
        for m in spec["metrics"]:
            out[task][m] = 0.0
            out[task][f"{m}_stderr"] = 0.0
    return out


def run_real(
    checkpoint: Path,
    tasks: dict[str, dict],
    device: str = "cuda",
    batch_size: int = 4,
    max_length: int | None = None,
    limit: int | float | None = None,
) -> dict[str, dict[str, float]]:
    """Run lm_eval on a trained IntLLM checkpoint, grouped by fewshot value.

    We do up to 2 `simple_evaluate` calls (5-shot group + 0-shot group)
    because lm_eval's `num_fewshot` is per-call, not per-task. The HFLM
    wrapper is instantiated once and reused across both calls.
    """
    from intllm.lm_eval_wrapper import build_hflm, extract_results
    from lm_eval import simple_evaluate

    lm = build_hflm(
        checkpoint,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )

    # Group tasks by num_fewshot so we issue one simple_evaluate per group.
    by_fewshot: dict[int, list[str]] = {}
    for task, spec in tasks.items():
        by_fewshot.setdefault(spec["num_fewshot"], []).append(task)

    merged: dict[str, dict[str, float]] = {}
    for n_shot, task_list in by_fewshot.items():
        out = simple_evaluate(
            model=lm,
            tasks=task_list,
            num_fewshot=n_shot,
            limit=limit,
            log_samples=False,
            bootstrap_iters=100,
        )
        for t, m in extract_results(out).items():
            merged[t] = m
    return merged


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tag", required=True,
                   help="Output tag, e.g. 'intllm-mini'")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Path to .pt checkpoint (real mode)")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Subset of knowledge tasks (default: mmlu triviaqa boolq)")
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "paper" / "intllm" / "results",
                   help="Where to write bench_knowledge_<tag>.json")
    p.add_argument("--scaffold", action="store_true",
                   help="Force scaffold mode")
    p.add_argument("--strict", action="store_true",
                   help="Fail if scaffold mode would be chosen implicitly")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--limit", type=float, default=None)
    args = p.parse_args()

    # Task selection
    if args.tasks:
        selected = {t: KNOWLEDGE_TASKS[t] for t in args.tasks if t in KNOWLEDGE_TASKS}
        missing = [t for t in args.tasks if t not in KNOWLEDGE_TASKS]
        if missing:
            print(f"WARN: unknown tasks ignored: {missing}", file=sys.stderr)
        if not selected:
            print(f"FATAL: no valid tasks from {args.tasks}", file=sys.stderr)
            return 2
    else:
        selected = KNOWLEDGE_TASKS

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
            selected,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            limit=args.limit,
        )
        # Merge into a full-schema result (overlay on top of scaffold) so
        # partial-task smoke runs preserve placeholder entries for tasks
        # we didn't touch this invocation.
        results = scaffold_results(KNOWLEDGE_TASKS)
        for task, metrics in real.items():
            results[task] = metrics
        status = "real" if set(selected) == set(KNOWLEDGE_TASKS) else "real-partial"

    elapsed = time.time() - t0

    payload = {
        "_description": (
            "Knowledge / few-shot benchmark results (mmlu 5-shot, "
            "triviaqa 5-shot, boolq 0-shot — matching BitNet 2B4T Table 1). "
            "Schema consumed by scripts/verify_intllm_tables.py::bench_knowledge. "
            "Run via scripts/bench_knowledge.py."
        ),
        "_phase": "FJQ_PHASE_D_PRODUCTION_PLAN.md §3.3 (knowledge benchmarks)",
        "_plan_ref": "paper Table 2 knowledge column / BitNet 2B4T Table 1 parity",
        "model": args.tag,
        "harness_version": harness_ref,
        "status": status,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "tasks": list(selected),
        "num_fewshot_map": {t: selected[t]["num_fewshot"] for t in selected},
        "limit": args.limit,
        "batch_size": args.batch_size if status != "scaffold" else None,
        "device": args.device if status != "scaffold" else None,
        "elapsed_seconds": round(elapsed, 3),
        "results": results,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"bench_knowledge_{args.tag}.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"artifact: {out_path.relative_to(REPO_ROOT) if out_path.is_relative_to(REPO_ROOT) else out_path}")
    print(f"status:   {status}")
    print(f"tasks:    {len(selected)}  ({', '.join(selected)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
