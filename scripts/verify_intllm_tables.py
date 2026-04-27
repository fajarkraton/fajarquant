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


def bench_baselines(tag: str, task: str, metric: str) -> float:
    """Load a baseline-re-run benchmark result from bench_baselines_<tag>.json.

    Schema (from `scripts/bench_baselines.py`, Phase 3.4 — see
    `docs/BASELINE_UNPORTED_FEATURES.md`):
        {
          "repo": "microsoft/bitnet-b1.58-2B-4T",
          "tag": "microsoft__bitnet-b1_58-2B-4T",
          "training_regime": "pretrain_sft_dpo" | "pretrain_only",
          "tokenizer_family": "mistral-v3-32k" | "llama3-128k" | ...,
          "sft_lift_caveat": bool,  // Table 2 footnote gate
          "results": {
            "<task>": {"<metric>": 0.xxx, "<metric>_stderr": 0.xxx, ...}
          },
          ...
        }

    The `tag` here is the filesystem-safe derivative produced by
    `bench_baselines.repo_to_tag()` (slashes → '__', dots → '_'), NOT the
    HF repo id itself.
    """
    data = load_json(f"bench_baselines_{tag}.json")
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
# Path A Phase E paper loaders (added 2026-04-27)
# ─────────────────────────────────────────────────────────────────
#
# These cover the tables introduced by the Path A re-scoped paper
# (`paper/intllm/intllm.tex`) — Q5 bilingual baseline (§6 tab:q5-baseline),
# E2.4 + E2.1 negative-result rows (§7 tab:negatives-summary), Phase E1
# corpus composition (§6 tab:corpus-comp). Sourced from the ablation
# JSONs in `paper/intllm/ablations/` rather than `paper/intllm/results/`,
# so we add a parallel ABLATIONS_DIR-relative loader.

ABLATIONS_DIR = REPO_ROOT / "paper" / "intllm" / "ablations"


def load_ablation(rel_path: str) -> dict[str, Any]:
    """Load a JSON artifact from paper/intllm/ablations/. Same error-path
    contract as load_json (clear FileNotFoundError on missing file)."""
    path = ABLATIONS_DIR / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Missing ablations artifact: {path.relative_to(REPO_ROOT)}")
    with path.open() as f:
        return json.load(f)


def q5_baseline(key: str) -> float:
    """Load a value from the Q5 bilingual baseline JSON.
    Keys: val_loss_id, val_loss_en, ratio, ppl_id, ppl_en,
    training_seconds, final_loss, n_steps."""
    data = load_ablation("q5_bilingual_baseline.json")
    return float(data[key])


def ablation_row(tag: str, key: str = "val_loss") -> float:
    """Load a metric from a Mini-scale ablation row JSON.
    `tag` is the train_mini_ablation.py --tag value (e.g. 'hadamard',
    'balanced_calib'). Default key is val_loss; other useful keys:
    final_loss, training_seconds, model_params."""
    data = load_ablation(f"mini_{tag}.json")
    return float(data[key])


def quant_error_metric(tag: str, key: str = "outlier_global_reduction") -> float:
    """Load a value from the E2.4.C metric JSON output.
    Keys: outlier_global_reduction (the gate metric), global_mean_reduction,
    gate_threshold, gate_pass (bool — caller responsible for float-coercion
    if needed; here we expect float-valued keys only)."""
    data = load_ablation(f"mini_{tag}_quant_error.json")
    return float(data[key])


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
    # ── Table 1 knowledge benchmarks (Phase 3.3) ──
    # mmlu (5-shot, 4-choice, random=0.25); triviaqa (5-shot exact_match);
    # boolq (0-shot binary, random=0.50). Sub-100M models near-random on
    # mmlu/triviaqa as expected per Chinchilla — included for honest
    # negative-result reporting. boolq shows clear monotonic scaling
    # (0.378 → 0.449 → 0.620) with Medium meaningfully above random.
    Claim("Table 1: IntLLM-Mini mmlu (5-shot)",
          0.2295, lambda: bench_knowledge("intllm-mini", "mmlu"),
          0.01, "Table 1 / Mini / mmlu_acc (random 0.25, model below threshold)"),
    Claim("Table 1: IntLLM-Base mmlu (5-shot)",
          0.2297, lambda: bench_knowledge("intllm-base", "mmlu"),
          0.01, "Table 1 / Base / mmlu_acc (random 0.25, model below threshold)"),
    Claim("Table 1: IntLLM-Medium mmlu (5-shot)",
          0.2374, lambda: bench_knowledge("intllm-medium", "mmlu"),
          0.01, "Table 1 / Medium / mmlu_acc (random 0.25, model marginally above Mini)"),
    Claim("Table 1: IntLLM-Mini triviaqa (5-shot exact)",
          0.0002, lambda: bench_knowledge("intllm-mini", "triviaqa", "exact_match"),
          0.01, "Table 1 / Mini / triviaqa_em (sub-100M cannot recall trivia)"),
    Claim("Table 1: IntLLM-Base triviaqa (5-shot exact)",
          0.0002, lambda: bench_knowledge("intllm-base", "triviaqa", "exact_match"),
          0.01, "Table 1 / Base / triviaqa_em (sub-100M cannot recall trivia)"),
    Claim("Table 1: IntLLM-Medium triviaqa (5-shot exact)",
          0.0003, lambda: bench_knowledge("intllm-medium", "triviaqa", "exact_match"),
          0.01, "Table 1 / Medium / triviaqa_em (sub-100M cannot recall trivia)"),
    Claim("Table 1: IntLLM-Mini boolq (0-shot)",
          0.3783, lambda: bench_knowledge("intllm-mini", "boolq"),
          0.01, "Table 1 / Mini / boolq_acc (random 0.50, below)"),
    Claim("Table 1: IntLLM-Base boolq (0-shot)",
          0.4492, lambda: bench_knowledge("intllm-base", "boolq"),
          0.01, "Table 1 / Base / boolq_acc (random 0.50, approaching)"),
    Claim("Table 1: IntLLM-Medium boolq (0-shot)",
          0.6202, lambda: bench_knowledge("intllm-medium", "boolq"),
          0.01, "Table 1 / Medium / boolq_acc (0.62 vs random 0.50 — clear scaling signal, +24pts vs Mini)"),
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
    # ── Table 2 baseline rows (Phase 3.4) ──
    # 7 baselines × 3 headline metrics = 21 placeholder claims. All start
    # with paper_value=0.0 + tolerance=inf so the script runs green today
    # against the scaffold JSONs produced by `make bench-baselines`. When
    # the real Phase 3.4 sweep lands (`make bench-baselines-real-all`,
    # ~13 GPU-hours), each cell is activated by setting paper_value to the
    # measured number and tolerance to 0.01.
    #
    # Headline metric selection (§6.9 R3 + BASELINE_UNPORTED_FEATURES.md §2):
    #   1. wikitext / bits_per_byte  — tokenizer-invariant text-modeling metric
    #      (PPL is NOT cross-comparable across the 5 tokenizer families in
    #      the set; BPB is the fix).
    #   2. hellaswag / acc_norm      — zero-shot reasoning headline, matches
    #      IntLLM Table 2 hellaswag row for direct side-by-side.
    #   3. mmlu / acc (5-shot)       — knowledge headline, matches BitNet
    #      2B4T Table 1 column + IntLLM Table 1 mmlu row.
    #
    # Footnote reminders (enforced by the Phase 4 pre-commit hook):
    #   - BitNet 2B4T row is post-SFT+DPO — zero-shot accuracy cells must
    #     carry the SFT-lift footnote (BASELINE_UNPORTED_FEATURES.md §2.1).
    #   - MMfreeLM 1.3B / 2.7B have no upstream-published Table 1 to
    #     cross-check against (§3.2).
    # ─── BitNet 2B4T (post-SFT+DPO — zero-shot accuracies need SFT-lift footnote) ───
    Claim("Table 2 Baseline: microsoft/bitnet-b1.58-2B-4T wikitext BPB",
          0.0, lambda: bench_baselines("microsoft__bitnet-b1_58-2B-4T", "wikitext", "bits_per_byte"),
          float("inf"), "Table 2 / BitNet-2B4T / wikitext_bpb (placeholder — populate from Phase 3.4 real run)"),
    Claim("Table 2 Baseline: microsoft/bitnet-b1.58-2B-4T hellaswag",
          0.0, lambda: bench_baselines("microsoft__bitnet-b1_58-2B-4T", "hellaswag", "acc_norm"),
          float("inf"), "Table 2 / BitNet-2B4T / hellaswag_acc_norm (placeholder; SFT-lift footnote required)"),
    Claim("Table 2 Baseline: microsoft/bitnet-b1.58-2B-4T mmlu (5-shot)",
          0.0, lambda: bench_baselines("microsoft__bitnet-b1_58-2B-4T", "mmlu", "acc"),
          float("inf"), "Table 2 / BitNet-2B4T / mmlu_acc (placeholder; SFT-lift footnote required)"),
    # ─── MMfreeLM 370M (cleanest apples-to-apples baseline — same arch family + tokenizer) ───
    Claim("Table 2 Baseline: ridger/MMfreeLM-370M wikitext BPB",
          0.0, lambda: bench_baselines("ridger__MMfreeLM-370M", "wikitext", "bits_per_byte"),
          float("inf"), "Table 2 / MMfreeLM-370M / wikitext_bpb (placeholder)"),
    Claim("Table 2 Baseline: ridger/MMfreeLM-370M hellaswag",
          0.0, lambda: bench_baselines("ridger__MMfreeLM-370M", "hellaswag", "acc_norm"),
          float("inf"), "Table 2 / MMfreeLM-370M / hellaswag_acc_norm (placeholder)"),
    Claim("Table 2 Baseline: ridger/MMfreeLM-370M mmlu (5-shot)",
          0.0, lambda: bench_baselines("ridger__MMfreeLM-370M", "mmlu", "acc"),
          float("inf"), "Table 2 / MMfreeLM-370M / mmlu_acc (placeholder)"),
    # ─── MMfreeLM 1.3B (first-party eval; no upstream Table 1 for cross-check) ───
    Claim("Table 2 Baseline: ridger/MMfreeLM-1.3B wikitext BPB",
          0.0, lambda: bench_baselines("ridger__MMfreeLM-1_3B", "wikitext", "bits_per_byte"),
          float("inf"), "Table 2 / MMfreeLM-1.3B / wikitext_bpb (placeholder; no upstream cross-check)"),
    Claim("Table 2 Baseline: ridger/MMfreeLM-1.3B hellaswag",
          0.0, lambda: bench_baselines("ridger__MMfreeLM-1_3B", "hellaswag", "acc_norm"),
          float("inf"), "Table 2 / MMfreeLM-1.3B / hellaswag_acc_norm (placeholder; no upstream cross-check)"),
    Claim("Table 2 Baseline: ridger/MMfreeLM-1.3B mmlu (5-shot)",
          0.0, lambda: bench_baselines("ridger__MMfreeLM-1_3B", "mmlu", "acc"),
          float("inf"), "Table 2 / MMfreeLM-1.3B / mmlu_acc (placeholder; no upstream cross-check)"),
    # ─── MMfreeLM 2.7B (largest baseline; VRAM-constrained at batch_size=2) ───
    Claim("Table 2 Baseline: ridger/MMfreeLM-2.7B wikitext BPB",
          0.0, lambda: bench_baselines("ridger__MMfreeLM-2_7B", "wikitext", "bits_per_byte"),
          float("inf"), "Table 2 / MMfreeLM-2.7B / wikitext_bpb (placeholder; no upstream cross-check)"),
    Claim("Table 2 Baseline: ridger/MMfreeLM-2.7B hellaswag",
          0.0, lambda: bench_baselines("ridger__MMfreeLM-2_7B", "hellaswag", "acc_norm"),
          float("inf"), "Table 2 / MMfreeLM-2.7B / hellaswag_acc_norm (placeholder; no upstream cross-check)"),
    Claim("Table 2 Baseline: ridger/MMfreeLM-2.7B mmlu (5-shot)",
          0.0, lambda: bench_baselines("ridger__MMfreeLM-2_7B", "mmlu", "acc"),
          float("inf"), "Table 2 / MMfreeLM-2.7B / mmlu_acc (placeholder; no upstream cross-check)"),
    # ─── SmolLM2-135M (pretrain-only base; native Llama arch) ───
    Claim("Table 2 Baseline: HuggingFaceTB/SmolLM2-135M wikitext BPB",
          0.0, lambda: bench_baselines("HuggingFaceTB__SmolLM2-135M", "wikitext", "bits_per_byte"),
          float("inf"), "Table 2 / SmolLM2-135M / wikitext_bpb (placeholder)"),
    Claim("Table 2 Baseline: HuggingFaceTB/SmolLM2-135M hellaswag",
          0.0, lambda: bench_baselines("HuggingFaceTB__SmolLM2-135M", "hellaswag", "acc_norm"),
          float("inf"), "Table 2 / SmolLM2-135M / hellaswag_acc_norm (placeholder)"),
    Claim("Table 2 Baseline: HuggingFaceTB/SmolLM2-135M mmlu (5-shot)",
          0.0, lambda: bench_baselines("HuggingFaceTB__SmolLM2-135M", "mmlu", "acc"),
          float("inf"), "Table 2 / SmolLM2-135M / mmlu_acc (placeholder)"),
    # ─── SmolLM2-360M ───
    Claim("Table 2 Baseline: HuggingFaceTB/SmolLM2-360M wikitext BPB",
          0.0, lambda: bench_baselines("HuggingFaceTB__SmolLM2-360M", "wikitext", "bits_per_byte"),
          float("inf"), "Table 2 / SmolLM2-360M / wikitext_bpb (placeholder)"),
    Claim("Table 2 Baseline: HuggingFaceTB/SmolLM2-360M hellaswag",
          0.0, lambda: bench_baselines("HuggingFaceTB__SmolLM2-360M", "hellaswag", "acc_norm"),
          float("inf"), "Table 2 / SmolLM2-360M / hellaswag_acc_norm (placeholder)"),
    Claim("Table 2 Baseline: HuggingFaceTB/SmolLM2-360M mmlu (5-shot)",
          0.0, lambda: bench_baselines("HuggingFaceTB__SmolLM2-360M", "mmlu", "acc"),
          float("inf"), "Table 2 / SmolLM2-360M / mmlu_acc (placeholder)"),
    # ─── Pythia-160m (classic Pile pretrain) ───
    Claim("Table 2 Baseline: EleutherAI/pythia-160m wikitext BPB",
          0.0, lambda: bench_baselines("EleutherAI__pythia-160m", "wikitext", "bits_per_byte"),
          float("inf"), "Table 2 / Pythia-160m / wikitext_bpb (placeholder)"),
    Claim("Table 2 Baseline: EleutherAI/pythia-160m hellaswag",
          0.0, lambda: bench_baselines("EleutherAI__pythia-160m", "hellaswag", "acc_norm"),
          float("inf"), "Table 2 / Pythia-160m / hellaswag_acc_norm (placeholder)"),
    Claim("Table 2 Baseline: EleutherAI/pythia-160m mmlu (5-shot)",
          0.0, lambda: bench_baselines("EleutherAI__pythia-160m", "mmlu", "acc"),
          float("inf"), "Table 2 / Pythia-160m / mmlu_acc (placeholder)"),

    # ─────────────────────────────────────────────────────────────────
    # Path A Phase E paper claims (added 2026-04-27 — paper/intllm/intllm.tex)
    # ─────────────────────────────────────────────────────────────────

    # ── §6 Table tab:q5-baseline (Q5 bilingual baseline) ──
    Claim("Q5 baseline: val_loss(ID, seed=999, 50 batches)",
          paper_value=2.679,
          source_loader=lambda: q5_baseline("val_loss_id"),
          tolerance=0.001,
          paper_ref="§6 Table tab:q5-baseline / val_loss(ID)"),
    Claim("Q5 baseline: val_loss(EN, seed=998, 50 batches)",
          paper_value=4.732,
          source_loader=lambda: q5_baseline("val_loss_en"),
          tolerance=0.001,
          paper_ref="§6 Table tab:q5-baseline / val_loss(EN)"),
    Claim("Q5 baseline: cross-lingual coherence ratio",
          paper_value=1.77,
          source_loader=lambda: q5_baseline("ratio"),
          tolerance=0.01,
          paper_ref="§6 Table tab:q5-baseline / rho"),
    Claim("Q5 baseline: PPL(ID)",
          paper_value=14.6,
          source_loader=lambda: q5_baseline("ppl_id"),
          tolerance=0.05,
          paper_ref="§6 Table tab:q5-baseline / PPL(ID)"),
    Claim("Q5 baseline: PPL(EN)",
          paper_value=113.5,
          source_loader=lambda: q5_baseline("ppl_en"),
          tolerance=0.05,
          paper_ref="§6 Table tab:q5-baseline / PPL(EN)"),

    # ── §7 Table tab:negatives-summary (E2.4 balanced_calib + E2.1 Hadamard) ──
    Claim("Negatives: balanced_calib val_loss(EN)",
          paper_value=4.834,
          source_loader=lambda: ablation_row("balanced_calib", "val_loss"),
          tolerance=0.001,
          paper_ref="§7 Table tab:negatives-summary / balanced_calib val_loss(EN)"),
    Claim("Negatives: balanced_calib outlier_global_reduction (E2.4 gate)",
          paper_value=-82.13,
          source_loader=lambda: quant_error_metric("balanced_calib", "outlier_global_reduction"),
          tolerance=0.05,
          paper_ref="§7 Table tab:negatives-summary / balanced_calib gate value"),
    Claim("Negatives: balanced_calib global_mean_reduction (diagnostic)",
          paper_value=-148.27,
          source_loader=lambda: quant_error_metric("balanced_calib", "global_mean_reduction"),
          tolerance=0.05,
          paper_ref="§7 §7.2 / balanced_calib global_mean_reduction"),
    Claim("Negatives: hadamard val_loss(EN)",
          paper_value=4.852,
          source_loader=lambda: ablation_row("hadamard", "val_loss"),
          tolerance=0.001,
          paper_ref="§7 Table tab:negatives-summary / Hadamard val_loss(EN)"),

    # Wall-clock training time claims (validate the §4 + §6 + §7 numbers)
    Claim("Q5 baseline: training wall-clock (seconds; matches §6 42.0 min)",
          paper_value=2519.17,
          source_loader=lambda: q5_baseline("training_seconds"),
          tolerance=10.0,  # ±10s for system-load jitter
          paper_ref="§6 Table tab:q5-baseline / training wall-clock"),
    Claim("Negatives: hadamard training wall-clock (seconds; matches §7 61.8 min)",
          paper_value=3706.50,
          source_loader=lambda: ablation_row("hadamard", "training_seconds"),
          tolerance=10.0,
          paper_ref="§7.3 / Hadamard training_seconds"),

    # ── §7.3 F.6.1 outlier-concentration measurement (NEW 2026-04-27 post-V31) ──
    Claim("F.6.1: o_proj per-layer-max max/mean ratio (paper says 421×)",
          paper_value=421.08,
          source_loader=lambda: float(load_ablation("outlier_concentration_mini.json")
                                      ["by_role"]["o_proj"]["max_of_ratio_max"]),
          tolerance=0.5,
          paper_ref="§7.3 cause-2 measured F.6.1 / o_proj max_of_ratio_max"),
    Claim("F.6.1: o_proj mean ratio across 6 layers (paper says 51.6×)",
          paper_value=51.63,
          source_loader=lambda: float(load_ablation("outlier_concentration_mini.json")
                                      ["by_role"]["o_proj"]["mean_of_ratio_mean"]),
          tolerance=0.05,
          paper_ref="§7.3 cause-2 measured F.6.1 / o_proj mean_of_ratio_mean"),
    Claim("F.6.1: mlp.down_proj mean ratio (paper says 40.2×)",
          paper_value=40.18,
          source_loader=lambda: float(load_ablation("outlier_concentration_mini.json")
                                      ["by_role"]["mlp.down_proj"]["mean_of_ratio_mean"]),
          tolerance=0.05,
          paper_ref="§7.3 cause-2 measured F.6.1 / mlp.down_proj mean_of_ratio_mean"),
    Claim("F.6.1: i_proj/f_proj/g_proj mean ratio (paper says 5.30×)",
          paper_value=5.30,
          source_loader=lambda: float(load_ablation("outlier_concentration_mini.json")
                                      ["by_role"]["i_proj"]["mean_of_ratio_mean"]),
          tolerance=0.05,
          paper_ref="§7.3 cause-2 measured F.6.1 / i_proj mean_of_ratio_mean"),
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
