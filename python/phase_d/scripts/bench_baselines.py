#!/usr/bin/env python3
"""V31.C.P3.4 — baseline re-run benchmark runner (scaffold + real).

Covers the 7 baselines committed in docs/FJQ_PHASE_D_3_4_B0_FINDINGS.md §2:

  microsoft/bitnet-b1.58-2B-4T      (bitnet, native HF, post-SFT+DPO)
  ridger/MMfreeLM-370M              (hgrn_bit, trust_remote_code, pretrain-only)
  ridger/MMfreeLM-1.3B              (hgrn_bit, trust_remote_code, pretrain-only)
  ridger/MMfreeLM-2.7B              (hgrn_bit, trust_remote_code, pretrain-only)
  HuggingFaceTB/SmolLM2-135M        (llama, native HF, pretrain-only base)
  HuggingFaceTB/SmolLM2-360M        (llama, native HF, pretrain-only base)
  EleutherAI/pythia-160m            (gpt_neox, native HF, pretrain-only)

Every baseline is evaluated on the SAME 11 tasks the IntLLM models run:
  8 canonical zero-shot (bench_canonical.CANONICAL_TASKS)
  3 knowledge few-shot (bench_knowledge.KNOWLEDGE_TASKS)

Rationale (§6.9 R3): reusing the identical `HFLM → simple_evaluate →
extract_results` plumbing that IntLLM uses eliminates silent disagreement
from wrapper behavior. The only axes of difference that remain are:
model weights, tokenizer, and compute kernel — which is what Phase D
is *trying* to measure.

Two modes (same pattern as bench_canonical + bench_knowledge):

  --scaffold   Emit schema-valid JSON with placeholder 0.0 values and
               status='scaffold'. CPU-only, deterministic, <1s. Unblocks
               verify_intllm_tables.py PEND rows without GPU.

  (default)    Real lm_eval run. Loads the HF repo via
               `lm_eval.models.huggingface.HFLM(pretrained=<repo>, ...)`.
               Emits JSON with status='real' and populated metric values.

Usage:
    cd fajarquant/python/phase_d
    PYTHONPATH=_upstream:. ../../.venv/bin/python scripts/bench_baselines.py \\
        --repo microsoft/bitnet-b1.58-2B-4T --scaffold

    # via top-level make (§7 decisions):
    make bench-baselines                    # scaffold all 7
    make bench-baselines-real REPO=EleutherAI/pythia-160m LIMIT=20

Exit codes:
    0   artifact written (scaffold or real)
    2   real mode requested but prerequisites missing (hard failure, no
        silent fall-back to scaffold per FJQ_PHASE_D_3_4_B0_FINDINGS.md §7.4)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO_ROOT = ROOT.parent.parent

# ─────────────────────────────────────────────────────────────────────
# Registry — per-baseline metadata committed in FJQ_PHASE_D_3_4_B0_FINDINGS.md §6
# ─────────────────────────────────────────────────────────────────────
#
# training_regime: {"pretrain_only", "pretrain_sft_dpo"}
#     Drives paper Table 2 footnote: pretrain_sft_dpo baselines (currently
#     only BitNet 2B4T) MUST carry an SFT-lift caveat when compared to
#     pretrain-only IntLLM on zero-shot tasks. PPL/BPB stays fair.
#
# tokenizer_family: human-readable key, for paper Table 2 tokenizer column.
#     PPL is not cross-comparable across tokenizer families — paper uses
#     bits_per_byte (BPB) as headline metric, PPL as supplementary.
#
# trust_remote_code: HF AutoModel flag. True ⇒ the repo ships custom
#     modeling code + we must ensure the custom package is importable
#     (see BASELINE_UNPORTED_FEATURES.md for hgrn_bit / _upstream/ wire-up).
#
# default_batch_size: per-repo default, overridable via CLI. 2.7B and
#     2B baselines default to 2 to keep KV cache inside 16 GB VRAM on
#     RTX 4090 Laptop (B0 §5 GPU budget).
#
# kernel_note: string that appears in the JSON artifact + paper footnote,
#     explicitly flagging unported-feature risks (§6.9 R3).

BASELINE_REGISTRY: dict[str, dict] = {
    "microsoft/bitnet-b1.58-2B-4T": {
        "arch": "bitnet",
        "params_approx": 2_000_000_000,
        "training_regime": "pretrain_sft_dpo",
        "tokenizer_family": "llama3-128k",
        "trust_remote_code": False,
        "default_batch_size": 2,
        "kernel_note": (
            "transformers-native BitNetForCausalLM (native since v4.54+); "
            "ternary-packed U8 weights unpacked on-the-fly in forward. "
            "Published efficiency numbers (29ms CPU decode) come from "
            "bitnet.cpp, NOT this path — Table 4 throughput/energy row "
            "stays N/A per BASELINE_UNPORTED_FEATURES.md."
        ),
        "sft_lift_caveat": True,
    },
    "ridger/MMfreeLM-370M": {
        "arch": "hgrn_bit",
        "params_approx": 370_000_000,
        "training_regime": "pretrain_only",
        "tokenizer_family": "mistral-v3-32k",
        "trust_remote_code": True,
        "default_batch_size": 4,
        "kernel_note": (
            "Vendored mmfreelm/ (UPSTREAM_PIN.md commit f24cfe5). "
            "Same arch cousin + same tokenizer as IntLLM — cleanest "
            "apples-to-apples baseline in the set."
        ),
        "sft_lift_caveat": False,
    },
    "ridger/MMfreeLM-1.3B": {
        "arch": "hgrn_bit",
        "params_approx": 1_300_000_000,
        "training_regime": "pretrain_only",
        "tokenizer_family": "mistral-v3-32k",
        "trust_remote_code": True,
        "default_batch_size": 4,
        "kernel_note": "Vendored mmfreelm/. No upstream-published Table 1 for this size.",
        "sft_lift_caveat": False,
    },
    "ridger/MMfreeLM-2.7B": {
        "arch": "hgrn_bit",
        "params_approx": 2_700_000_000,
        "training_regime": "pretrain_only",
        "tokenizer_family": "mistral-v3-32k",
        "trust_remote_code": True,
        "default_batch_size": 2,
        "kernel_note": (
            "Vendored mmfreelm/, sharded safetensors. Largest baseline; "
            "batch_size=2 to fit 16 GB VRAM alongside KV cache."
        ),
        "sft_lift_caveat": False,
    },
    "HuggingFaceTB/SmolLM2-135M": {
        "arch": "llama",
        "params_approx": 135_000_000,
        "training_regime": "pretrain_only",
        "tokenizer_family": "smollm2-49k",
        "trust_remote_code": False,
        "default_batch_size": 4,
        "kernel_note": "Base (non-Instruct) checkpoint.",
        "sft_lift_caveat": False,
    },
    "HuggingFaceTB/SmolLM2-360M": {
        "arch": "llama",
        "params_approx": 360_000_000,
        "training_regime": "pretrain_only",
        "tokenizer_family": "smollm2-49k",
        "trust_remote_code": False,
        "default_batch_size": 4,
        "kernel_note": "Base (non-Instruct) checkpoint.",
        "sft_lift_caveat": False,
    },
    "EleutherAI/pythia-160m": {
        "arch": "gpt_neox",
        "params_approx": 160_000_000,
        "training_regime": "pretrain_only",
        "tokenizer_family": "gpt-neox-50k",
        "trust_remote_code": False,
        "default_batch_size": 4,
        "kernel_note": "Classic Pythia-160M — Pile pretrain, no SFT confound.",
        "sft_lift_caveat": False,
    },
}

# ─────────────────────────────────────────────────────────────────────
# Task matrix — union of bench_canonical + bench_knowledge, with fewshot.
# ─────────────────────────────────────────────────────────────────────
# Single source-of-truth: we re-declare here (not import) because the
# scaffold path must work without torch/lm_eval/mmfreelm importable. The
# values MUST stay in sync with bench_canonical.CANONICAL_TASKS +
# bench_knowledge.KNOWLEDGE_TASKS. If this drift, CI should catch via the
# cross-validation test (see tests/test_bench_baselines_task_sync.py —
# Phase 3.4 follow-up).

CANONICAL_TASKS: dict[str, dict] = {
    "wikitext":       {"metrics": ["word_perplexity", "byte_perplexity", "bits_per_byte"], "num_fewshot": 0},
    "lambada_openai": {"metrics": ["acc", "perplexity"],                                    "num_fewshot": 0},
    "hellaswag":      {"metrics": ["acc", "acc_norm"],                                      "num_fewshot": 0},
    "piqa":           {"metrics": ["acc", "acc_norm"],                                      "num_fewshot": 0},
    "winogrande":     {"metrics": ["acc"],                                                  "num_fewshot": 0},
    "arc_easy":       {"metrics": ["acc", "acc_norm"],                                      "num_fewshot": 0},
    "arc_challenge":  {"metrics": ["acc", "acc_norm"],                                      "num_fewshot": 0},
    "openbookqa":     {"metrics": ["acc", "acc_norm"],                                      "num_fewshot": 0},
}
KNOWLEDGE_TASKS: dict[str, dict] = {
    "mmlu":      {"metrics": ["acc"],         "num_fewshot": 5},
    "triviaqa":  {"metrics": ["exact_match"], "num_fewshot": 5},
    "boolq":     {"metrics": ["acc"],         "num_fewshot": 0},
}
ALL_TASKS: dict[str, dict] = {**CANONICAL_TASKS, **KNOWLEDGE_TASKS}

HARNESS_VERSION = "v0.4.11"


def load_harness_ref() -> str:
    """Read the pinned lm-eval ref from eval/HARNESS_SHA."""
    sha_file = REPO_ROOT / "eval" / "HARNESS_SHA"
    if not sha_file.exists():
        return HARNESS_VERSION
    for line in sha_file.read_text().splitlines():
        s = line.strip()
        if s.startswith("ref:"):
            return s.split(None, 1)[1].strip()
    return HARNESS_VERSION


def repo_to_tag(repo: str) -> str:
    """Turn 'microsoft/bitnet-b1.58-2B-4T' → 'microsoft__bitnet-b1_58-2B-4T'.

    Filesystem-safe, collision-safe, reversible by rule. Dots become
    underscores (avoid extension confusion); slashes become '__'.
    """
    return re.sub(r"\.", "_", repo.replace("/", "__"))


def scaffold_results(tasks: dict[str, dict]) -> dict[str, dict[str, float]]:
    """Placeholder 0.0 values for every (task, metric) pair.

    Mirrors bench_canonical.scaffold_results / bench_knowledge.scaffold_results.
    """
    out: dict[str, dict[str, float]] = {}
    for task, spec in tasks.items():
        out[task] = {}
        for m in spec["metrics"]:
            out[task][m] = 0.0
            out[task][f"{m}_stderr"] = 0.0
    return out


def _ensure_mmfreelm_importable() -> None:
    """For hgrn_bit (MMfreeLM) baselines, make `_upstream/mmfreelm` importable.

    Called at the top of run_real() but BEFORE any torch/transformers
    imports that might dispatch on the custom class. Per
    BASELINE_UNPORTED_FEATURES.md §MMfreeLM, absence is a hard failure —
    no silent fall-back.
    """
    upstream = ROOT / "_upstream"
    if not upstream.exists():
        raise FileNotFoundError(
            f"Vendored mmfreelm snapshot not found at {upstream}. "
            f"Run the recreate-snapshot steps in UPSTREAM_PIN.md, "
            f"or skip hgrn_bit baselines for this invocation."
        )
    p = str(upstream)
    if p not in sys.path:
        sys.path.insert(0, p)
    # eager import so dispatch-time errors surface at script start, not
    # mid-task 30 min into a bench run
    import mmfreelm  # noqa: F401


def run_real(
    repo: str,
    meta: dict,
    tasks: dict[str, dict],
    device: str = "cuda",
    batch_size: int | None = None,
    limit: int | float | None = None,
) -> dict[str, dict[str, float]]:
    """Run lm_eval simple_evaluate against a HF baseline repo.

    Groups tasks by num_fewshot so we issue one simple_evaluate per
    group (mirrors bench_knowledge.run_real pattern). The HFLM
    instance is built once per repo and reused across fewshot groups.
    """
    # Eager vendor wire-up for hgrn_bit baselines (§3.3 of B0 findings).
    if meta["arch"] == "hgrn_bit":
        _ensure_mmfreelm_importable()

    # Deferred imports — heavy (torch, lm_eval, transformers). Keep
    # --scaffold fast + avoid forcing a GPU env on CI.
    from intllm.lm_eval_wrapper import extract_results
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    bs = batch_size if batch_size is not None else meta["default_batch_size"]

    # NOTE (§6.9 R3): we pass the HF repo id directly as `pretrained=`.
    # HFLM's own AutoModelForCausalLM.from_pretrained call handles both
    # native + trust_remote_code paths. Downloaded weights land in
    # ~/.cache/huggingface/hub as usual; no side-effect on our disk budget
    # beyond what B0 §1 already accounts for.
    lm = HFLM(
        pretrained=repo,
        trust_remote_code=meta["trust_remote_code"],
        backend="causal",
        device=device,
        batch_size=bs,
        dtype="bfloat16",
    )

    # Group tasks by num_fewshot (0 + 5 for our suite).
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
    p.add_argument("--repo", required=True,
                   help=f"HF repo id. Must be one of: {list(BASELINE_REGISTRY)}")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Subset of the 11 tasks. Default: all 11.")
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "paper" / "intllm" / "results",
                   help="Where to write bench_baselines_<tag>.json")
    p.add_argument("--scaffold", action="store_true",
                   help="Force scaffold mode (no GPU, no HF download, <1s).")
    p.add_argument("--strict", action="store_true",
                   help="Fail if scaffold mode would be chosen implicitly.")
    p.add_argument("--device", default="cuda",
                   help="torch device for real mode (default: cuda)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="override default_batch_size from registry")
    p.add_argument("--limit", type=float, default=None,
                   help="lm_eval `limit` — cap docs per task (smoke-test)")
    args = p.parse_args()

    if args.repo not in BASELINE_REGISTRY:
        print(f"FATAL: unknown --repo {args.repo!r}. "
              f"Registered: {sorted(BASELINE_REGISTRY)}", file=sys.stderr)
        return 2
    meta = BASELINE_REGISTRY[args.repo]

    # Task selection
    if args.tasks:
        selected = {t: ALL_TASKS[t] for t in args.tasks if t in ALL_TASKS}
        missing = [t for t in args.tasks if t not in ALL_TASKS]
        if missing:
            print(f"WARN: unknown tasks ignored: {missing}", file=sys.stderr)
        if not selected:
            print(f"FATAL: no valid tasks from {args.tasks}", file=sys.stderr)
            return 2
    else:
        selected = ALL_TASKS

    # Mode selection — real mode needs no local checkpoint (HF repo is
    # the checkpoint). We only drop to scaffold if explicitly asked, or
    # if we detect the run would fail prerequisites.
    use_scaffold = args.scaffold
    if not use_scaffold:
        # Prerequisite check (BASELINE_UNPORTED_FEATURES.md §pre-flight):
        # for hgrn_bit, the vendored _upstream/ must exist. Don't try
        # imports here (expensive) — just filesystem probe.
        if meta["arch"] == "hgrn_bit" and not (ROOT / "_upstream" / "mmfreelm").exists():
            msg = (f"real mode requires vendored mmfreelm (see UPSTREAM_PIN.md); "
                   f"{ROOT / '_upstream' / 'mmfreelm'} not found")
            if args.strict:
                print(f"FATAL (--strict): {msg}", file=sys.stderr)
                return 2
            print(f"WARN: {msg}. Falling back to scaffold.", file=sys.stderr)
            use_scaffold = True

    if use_scaffold and args.strict and args.scaffold is False:
        # Only error on implicit scaffold; explicit --scaffold is OK.
        print(f"FATAL (--strict): would scaffold implicitly", file=sys.stderr)
        return 2

    harness_ref = load_harness_ref()
    t0 = time.time()

    if use_scaffold:
        results = scaffold_results(selected)
        status = "scaffold"
        print(f"[scaffold] repo={args.repo} tasks={list(selected)} "
              f"(harness={harness_ref})")
    else:
        bs = args.batch_size if args.batch_size is not None else meta["default_batch_size"]
        print(f"[real] repo={args.repo} tasks={list(selected)} "
              f"device={args.device} batch_size={bs} "
              f"limit={args.limit} (harness={harness_ref})")
        print(f"[real] kernel_note: {meta['kernel_note']}")
        real = run_real(
            args.repo,
            meta,
            selected,
            device=args.device,
            batch_size=args.batch_size,
            limit=args.limit,
        )
        # Merge into full schema so partial-task runs preserve placeholders
        # for tasks we didn't touch this invocation.
        results = scaffold_results(ALL_TASKS)
        for task, metrics in real.items():
            results[task] = metrics
        status = "real" if set(selected) == set(ALL_TASKS) else "real-partial"

    elapsed = time.time() - t0
    tag = repo_to_tag(args.repo)

    payload = {
        "_description": (
            "Baseline re-run benchmark results. Same 11-task suite + same "
            "HFLM plumbing as IntLLM itself — §6.9 R3 baseline parity. "
            "Schema consumed by scripts/verify_intllm_tables.py. "
            "Run via scripts/bench_baselines.py."
        ),
        "_phase": "FJQ_PHASE_D_PRODUCTION_PLAN.md §3.4 (baseline re-runs)",
        "_plan_ref": "paper Table 2 baseline rows",
        "_b0_findings": "docs/FJQ_PHASE_D_3_4_B0_FINDINGS.md",
        "repo": args.repo,
        "tag": tag,
        "arch": meta["arch"],
        "params_approx": meta["params_approx"],
        "training_regime": meta["training_regime"],
        "tokenizer_family": meta["tokenizer_family"],
        "sft_lift_caveat": meta["sft_lift_caveat"],
        "kernel_note": meta["kernel_note"],
        "harness_version": harness_ref,
        "status": status,
        "tasks": list(selected),
        "num_fewshot_map": {t: selected[t]["num_fewshot"] for t in selected},
        "limit": args.limit,
        "batch_size": (args.batch_size if args.batch_size is not None else meta["default_batch_size"]) if status != "scaffold" else None,
        "device": args.device if status != "scaffold" else None,
        "elapsed_seconds": round(elapsed, 3),
        "results": results,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"bench_baselines_{tag}.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    rel = out_path.relative_to(REPO_ROOT) if out_path.is_relative_to(REPO_ROOT) else out_path
    print(f"artifact: {rel}")
    print(f"status:   {status}")
    print(f"tasks:    {len(selected)}  ({', '.join(selected)})")
    print(f"repo:     {args.repo} ({meta['params_approx']/1e6:.0f}M, {meta['training_regime']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
