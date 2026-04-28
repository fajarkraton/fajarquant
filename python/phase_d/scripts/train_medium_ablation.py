#!/usr/bin/env python3
"""
V32-prep F.6.4 — Medium-scale ablation harness.

Per Phase F roadmap §4.1 F.6.4 (Hadamard rotation at Base/Medium scale,
post-Phase-E E2.1 negative result): this driver produces
`paper/intllm/ablations/medium_<TAG>.json` artifacts so any feature
demoted from Mini-scale Phase E2 to F.6.4-scale re-evaluation can be
re-run at Medium scale (d=512, L=12, 73 BitLinear sites = 6×12 attn/MLP
+ 1 lm_head) without duplicating the Mini ablation harness.

NOTE: This is SCAFFOLD work (V32-prep). Entry condition for F.6.x
execution is "Phase E paper accepted (or at least submitted)" — see
docs/FJQ_PHASE_F_TAX_VERTICAL_ROADMAP.md §4.1. Until then this script
is exercised only via `--proof-of-life` and `--dry-run` paths.

Driver pattern: thin shim around `intllm.train.train_loop` with feature
toggles. Mirrors `train_medium.py` (Phase D) closely so behavior is
predictable; ablation flag wiring mirrors `train_mini_ablation.py`
(E2.1.0). Only adds ablation-specific TAG + feature flags + ablation
JSON output at Medium scale.

Currently no E2.x feature is implemented. Toggling any flag emits a
clear warning and proceeds with baseline behavior so the harness wiring
(driver -> JSON -> verify_paper_tables) is testable BEFORE feature code
lands. As each E2.x feature ships, the corresponding flag in this
driver is wired to the real implementation in intllm.{quant,qat,model}.

Smoke (no GPU required, ~30 s on CPU with --dry-run):

    cd python/phase_d
    PYTHONPATH=. ../../.venv/bin/python scripts/train_medium_ablation.py \
        --tag baseline --dry-run

Proof-of-life (200 steps, ~3-5 min on RTX 4090):

    PYTHONPATH=. ../../.venv/bin/python scripts/train_medium_ablation.py \
        --tag baseline --proof-of-life

Full (~11 h on RTX 4090 per CONFIG matrix §3.2, baseline equivalent to Phase D Medium c.1):

    PYTHONPATH=. ../../.venv/bin/python scripts/train_medium_ablation.py \
        --tag baseline

Real ablation (once features land, e.g. E2.4 BilingualCalibrationSampler):

    PYTHONPATH=. ../../.venv/bin/python scripts/train_medium_ablation.py \
        --tag balanced_calib --balanced-calib

JSON schema written to `paper/intllm/ablations/medium_<TAG>.json`:

    {
      "tag": "<TAG>",
      "phase": "E2.x",
      "features_active": [...],   # list of feature names ON
      "features_requested": [...], # list of feature flags requested by caller
      "n_steps": int,
      "val_loss": float,           # NaN if --proof-of-life or --dry-run
      "ppl": float,                # NaN if --proof-of-life or --dry-run
      "gate_pass": bool | null,    # null if val skipped
      "training_seconds": float,
      "initial_loss": float,
      "final_loss": float,
      "model_params": int,
      "timestamp": str (ISO 8601),
      "_schema_version": "1.0"
    }
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Force line-buffering for nohup-friendly logs (memory/feedback_nohup_python_buffering.md).
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPO_ROOT = ROOT.parent.parent  # python/phase_d → python → fajarquant root
ABLATIONS_DIR = REPO_ROOT / "paper" / "intllm" / "ablations"

# Medium gate per FJQ_PHASE_D_GATE_CALIBRATION.md (calibrated 2026-04-21).
MEDIUM_VAL_LOSS_GATE = 4.0

# E2.4 Option-C calibration: top-K outlier-channel %, low/high bit widths.
# Defaults match `intllm.qat.QATConfig.adaptive_bits_*` so paper claims
# can cite a single source. Override via CLI for sensitivity analysis.
E2_4_DEFAULT_TOP_K_PCT = 5.0
E2_4_DEFAULT_LOW_BITS = 8
E2_4_DEFAULT_HIGH_BITS = 10

# Set of feature flags that DO have a real implementation in this driver.
# Stub-warning loop skips entries here; the corresponding feature_active
# entry is set when real wire-up runs.
E2_REAL_FEATURES = {"balanced_calib", "hadamard"}

# E2 ablation runs match Q5 baseline (24K steps × 8K tok/step ≈ 196M
# tokens at 60:40 bilingual mix); see `q5_bilingual_baseline.py:DEFAULT_N_STEPS_FOR_200M_TOK`
# and `FJQ_PHASE_E_E2_FINDINGS.md` v1.1 §3. This is the LIKE-FOR-LIKE
# comparison budget — every E2.x ablation row must use this token
# budget so deltas reflect the FEATURE, not training depth.
#
# `MediumTrainConfig.n_steps = 60000` is the original Phase D Medium-full
# target (~491M tokens, EN-only); use that only by passing --n-steps
# 60000 explicitly when reproducing Phase D Medium gates, not when
# computing E2 ablations.
E2_DEFAULT_N_STEPS = 24_000

# E2.x feature flag → human-readable name mapping. The driver only
# RECORDS these in JSON; feature implementation lands in intllm.* as
# E2.x sub-tasks complete and replaces the corresponding stub block.
E2_FEATURE_FLAGS = {
    "hadamard": "E2.1 Hadamard rotation (QuaRot/SpinQuant) on attn.o_proj + lm_head",
    "fp8_lmhead": "E2.2 FP8 (E4M3) for lm_head + attn.o_proj",
    "distill": "E2.3 FP16 distillation from Phase D Medium c.1 teacher",
    "balanced_calib": "E2.4 BilingualCalibrationSampler (60:40 ID:EN)",
    "lang_cond_a": "E2.5 (a) Per-language RMSNorm γ params",
    "lang_cond_b": "E2.5 (b) Per-language LoRA adapter",
    "lang_cond_c": "E2.5 (c) Language-aware routing (MoE-light)",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_ablation_json(
    out_path: Path,
    tag: str,
    *,
    features_active: list[str],
    features_requested: list[str],
    n_steps: int,
    val_loss: float,
    initial_loss: float,
    final_loss: float,
    training_seconds: float,
    model_params: int,
    proof_of_life: bool,
    dry_run: bool,
    tracker_maps_path: Path | None = None,
    quant_error_path: Path | None = None,
) -> None:
    """Write the ablation result JSON. Schema v1.2 (E2.4.C.3:
    `quant_error_path` field added — null if --balanced-calib not run
    or if running in POL mode).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    val_skipped = proof_of_life or dry_run or math.isnan(val_loss)
    payload = {
        "_schema_version": "1.2",
        "tag": tag,
        "phase": "E2.x",
        "features_active": features_active,
        "features_requested": features_requested,
        "n_steps": n_steps,
        "val_loss": val_loss,
        "ppl": math.exp(val_loss) if not val_skipped else float("nan"),
        "gate_pass": (val_loss < MEDIUM_VAL_LOSS_GATE) if not val_skipped else None,
        "gate_threshold": MEDIUM_VAL_LOSS_GATE,
        "training_seconds": training_seconds,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "model_params": model_params,
        "proof_of_life": proof_of_life,
        "dry_run": dry_run,
        "tracker_maps_path": str(tracker_maps_path) if tracker_maps_path else None,
        "quant_error_path": str(quant_error_path) if quant_error_path else None,
        "timestamp": _now_iso(),
    }
    tmp = out_path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, out_path)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Phase E E2 Medium-scale ablation harness (E2.1.0 scaffold).",
    )
    p.add_argument(
        "--tag",
        required=True,
        help="Ablation tag — output goes to paper/intllm/ablations/medium_<TAG>.json. "
        "Standard tags: baseline, hadamard, fp8_lmhead, distill, balanced_calib, "
        "lang_cond_{a,b,c}, combined.",
    )
    p.add_argument(
        "--proof-of-life",
        action="store_true",
        help="Cap at 200 steps, skip val eval, do not save model checkpoint. "
        "Used for E2.1.0 harness smoke + per-feature wiring sanity.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build model + write placeholder JSON without any training. "
        "Validates JSON schema + Makefile wiring with no GPU.",
    )
    p.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Override training steps (e.g. 10000 for ~45 min intermediate).",
    )
    p.add_argument(
        "--val-steps",
        type=int,
        default=50,
        help="Number of batches to compute val_loss over (default 50).",
    )
    p.add_argument(
        "--quant-error-n-batches",
        type=int,
        default=1000,
        help="E2.4.C.3 metric: # batches for the quant-error pass (default 1000 per "
        "FJQ_PHASE_E_E2_4_C_METRIC_SPEC.md §2). Only runs when --balanced-calib is set.",
    )
    p.add_argument(
        "--skip-warmup-calibration",
        type=int,
        default=0,
        metavar="N",
        help="V32-prep F.5.3: defer BitLinearStatTracker.running_max accumulation "
        "until training step N (i.e. drop the first N training steps from the "
        "calibration accumulator). 0 = legacy behavior (record from step 0, the "
        "default for E2.4 balanced_calib). Useful values match the LR-scheduler "
        "warmup_steps so chaotic early-training peaks do not pollute the "
        "all-time-max statistic — see paper/intllm/intllm.tex §7.2 cause-1 + the "
        "F.5.0 cross-comparison evidence in "
        "paper/intllm/ablations/running_max_train_vs_steady*.json. Only meaningful "
        "with --balanced-calib (which is the only path that attaches trackers).",
    )
    p.add_argument(
        "--ema-calibration",
        action="store_true",
        help="V32-prep F.5.2: switch BitLinearStatTracker accumulator from "
        "elementwise max (legacy) to exponential moving average. Under EMA, "
        "running_max ← α·running_max + (1-α)·per_channel; the first post-warmup "
        "observation seeds the estimate directly. Late-training stability "
        "dominates over early-training peaks (partial mitigation of §7.2 cause-1; "
        "see paper/intllm/intllm.tex §7.2). Composes with --skip-warmup-calibration. "
        "Only meaningful with --balanced-calib.",
    )
    p.add_argument(
        "--ema-alpha",
        type=float,
        default=0.99,
        metavar="A",
        help="V32-prep F.5.2: smoothing factor for --ema-calibration. Must be in "
        "[0.0, 1.0]. Default 0.99 (literature standard, ~100-step half-life). "
        "α=0 → pure tracking of latest observation; α=1 → freezes after the "
        "bootstrap call (only useful for diagnostics). Ignored when "
        "--ema-calibration is not set.",
    )
    p.add_argument("--ckpt-dir", type=Path, default=HERE.parent / "checkpoints" / "medium_ablations")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Override output JSON path. Default: paper/intllm/ablations/medium_<TAG>.json",
    )

    # E2.x feature flags — currently stubs. Each toggles a placeholder
    # warning and a "features_active" entry in the JSON. Wire to real
    # impl as the corresponding E2.x sub-task lands.
    p.add_argument("--hadamard", action="store_true", help=E2_FEATURE_FLAGS["hadamard"])
    p.add_argument(
        "--hadamard-sites",
        choices=["o", "igfo", "igf"],
        default="o",
        help="V32-prep F.6.3: which HGRN attn projection inputs to rotate. "
        "'o' (default, E2.1 baseline) = o_proj only — preserves the failed "
        "Phase E E2.1 ablation behavior bit-exact. 'igfo' = i/f/g/o_proj all "
        "four — F.6.3 hypothesis that QKV-equivalent paths in HGRN benefit "
        "more from rotation than the gated output. 'igf' = i/f/g only — "
        "ablation-isolation control to attribute any igfo gain. Audit at "
        "paper/intllm/ablations/igf_proj_dim_audit.json confirms shared "
        "HadamardRotation(hidden_size) is dimensionally safe at "
        "expand_ratio=1 across Phase D scales (R1+R2+R3 PASS). Only "
        "meaningful with --hadamard.",
    )
    p.add_argument("--fp8-lmhead", action="store_true", help=E2_FEATURE_FLAGS["fp8_lmhead"])
    p.add_argument("--distill", action="store_true", help=E2_FEATURE_FLAGS["distill"])
    p.add_argument("--balanced-calib", action="store_true", help=E2_FEATURE_FLAGS["balanced_calib"])
    p.add_argument(
        "--bilingual-data",
        action="store_true",
        help="Use bilingual_stream(id_share=0.6) for both train and val. "
        "Default OFF (= slimpajama EN-only, matches Phase D Medium gates). "
        "Implicitly ON when --balanced-calib is set (E2.4 always implies "
        "bilingual). For apples-to-apples vs Q5 baseline (commit 1074883), "
        "E2.x ablations should pass --bilingual-data alongside the feature "
        "flag (e.g., `--hadamard --bilingual-data`).",
    )
    p.add_argument(
        "--lang-cond",
        choices=["a", "b", "c"],
        default=None,
        help="E2.5 language-conditioned design option. (a) per-lang RMSNorm γ "
        "(preliminary recommendation per E2.0 findings Q2), (b) per-lang LoRA, "
        "(c) language-aware routing.",
    )

    args = p.parse_args()

    if args.dry_run and args.proof_of_life:
        p.error("--dry-run and --proof-of-life are mutually exclusive")

    if args.skip_warmup_calibration < 0:
        p.error("--skip-warmup-calibration must be ≥ 0")
    if args.skip_warmup_calibration > 0 and not args.balanced_calib:
        p.error(
            "--skip-warmup-calibration is only meaningful with --balanced-calib "
            "(only that path attaches BitLinearStatTracker hooks)",
        )
    if not 0.0 <= args.ema_alpha <= 1.0:
        p.error(f"--ema-alpha must be in [0.0, 1.0], got {args.ema_alpha}")
    if args.ema_calibration and not args.balanced_calib:
        p.error(
            "--ema-calibration is only meaningful with --balanced-calib "
            "(only that path attaches BitLinearStatTracker hooks)",
        )
    if args.hadamard_sites != "o" and not args.hadamard:
        p.error(
            f"--hadamard-sites={args.hadamard_sites!r} is only meaningful "
            f"with --hadamard (no rotation hooks attached otherwise)",
        )

    out_path = args.out or (ABLATIONS_DIR / f"medium_{args.tag}.json")

    # Collect requested features (from CLI flags) before any stub-warning.
    features_requested: list[str] = []
    if args.hadamard:
        features_requested.append("hadamard")
    if args.fp8_lmhead:
        features_requested.append("fp8_lmhead")
    if args.distill:
        features_requested.append("distill")
    if args.balanced_calib:
        features_requested.append("balanced_calib")
    if args.lang_cond is not None:
        features_requested.append(f"lang_cond_{args.lang_cond}")

    # As each E2.x sub-task ships, the relevant flag is added to
    # E2_REAL_FEATURES so the warning is suppressed; the actual
    # behavior is wired further down. Features still in stub-mode emit
    # a [WARN] and proceed with baseline behavior so the harness keeps
    # producing JSON for verify_paper_tables to consume.
    features_active: list[str] = []
    for feat in features_requested:
        if feat in E2_REAL_FEATURES:
            features_active.append(feat)
            continue
        print(
            f"  [WARN] feature '{feat}' requested but NOT YET IMPLEMENTED — "
            f"falling through to baseline behavior. "
            f"This will be wired when the corresponding E2.x sub-task ships.",
            file=sys.stderr,
        )

    # ── DRY RUN: build model briefly, write JSON with placeholder values ──
    if args.dry_run:
        # Lazy imports — keep dry-run fast even if torch isn't fully loaded.
        import torch  # noqa: F401
        from configs.medium import MediumArchConfig
        from intllm.model import HGRNBitConfig, HGRNBitForCausalLM

        arch = MediumArchConfig()
        cfg = HGRNBitConfig(
            vocab_size=arch.vocab_size,
            hidden_size=arch.hidden_size,
            num_hidden_layers=arch.num_hidden_layers,
            max_position_embeddings=arch.max_position_embeddings,
        )
        model = HGRNBitForCausalLM(cfg)
        n_params = sum(p.numel() for p in model.parameters())

        _write_ablation_json(
            out_path,
            args.tag,
            features_active=features_active,
            features_requested=features_requested,
            n_steps=0,
            val_loss=float("nan"),
            initial_loss=float("nan"),
            final_loss=float("nan"),
            training_seconds=0.0,
            model_params=n_params,
            proof_of_life=False,
            dry_run=True,
        )
        print(f"\n  [DRY-RUN] wrote placeholder ablation JSON: {out_path}")
        print(f"  [DRY-RUN] model_params: {n_params:,}")
        return 0

    # ── REAL TRAINING PATH (proof-of-life or full) ──
    import torch  # noqa: E402
    from configs.medium import MediumArchConfig, MediumTrainConfig  # noqa: E402
    from intllm.data import bilingual_stream, slimpajama_stream  # noqa: E402
    from intllm.eval import compute_quant_error_per_channel, run_held_out_loss  # noqa: E402
    from intllm.model import HGRNBitConfig, HGRNBitForCausalLM  # noqa: E402
    from intllm.qat import (  # noqa: E402
        advance_trackers,
        attach_stat_trackers,
        detach_stat_trackers,
        is_bitlinear,
        save_calibration_maps,
    )
    from intllm.quant import HadamardRotation  # noqa: E402
    from intllm.tokenizer import get_tokenizer  # noqa: E402
    from intllm.train import TrainConfig, find_latest_checkpoint, train_loop  # noqa: E402  # noqa: F401
    sys.path.insert(0, str(REPO_ROOT / "python" / "phase_e"))
    from intllm_en import BILINGUAL_RATIO_DEFAULT  # noqa: E402

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    arch = MediumArchConfig()
    train_hp = MediumTrainConfig()
    if args.proof_of_life:
        train_hp = replace(train_hp, n_steps=200, log_every=25)
    elif args.n_steps is not None:
        train_hp = replace(train_hp, n_steps=args.n_steps)
    else:
        # Default to Q5-baseline-comparable token budget (24K steps),
        # NOT MediumTrainConfig's 60K. See E2_DEFAULT_N_STEPS docstring.
        train_hp = replace(train_hp, n_steps=E2_DEFAULT_N_STEPS)

    print(f"[1/4] config: arch={asdict(arch)}")
    print(f"      train: {asdict(train_hp)}")
    print(f"      device: {device}")
    print(f"      tag: {args.tag}")
    print(f"      features_requested: {features_requested or '(baseline only)'}")
    print(f"      features_active   : {features_active or '(baseline only)'}")

    print("[2/4] tokenizer + data stream")
    tok = get_tokenizer()
    use_bilingual = args.balanced_calib or args.bilingual_data
    if use_bilingual:
        # Bilingual stream when EITHER --balanced-calib (E2.4 implicit
        # bilingual sampling) OR --bilingual-data (explicit, used by
        # E2.1+ ablations that want apples-to-apples vs Q5 baseline).
        reason = "balanced_calib" if args.balanced_calib else "bilingual_data"
        print(f"      stream: bilingual (id_share={BILINGUAL_RATIO_DEFAULT:.2f}, reason={reason})")
        batches = bilingual_stream(
            tokenizer=tok,
            seq_len=train_hp.seq_len,
            batch_size=train_hp.batch_size,
            id_share=BILINGUAL_RATIO_DEFAULT,
            device=device,
            seed=0,
        )
    else:
        print("      stream: slimpajama (EN-only, Phase D Medium-gate baseline)")
        batches = slimpajama_stream(
            tokenizer=tok,
            seq_len=train_hp.seq_len,
            batch_size=train_hp.batch_size,
            device=device,
            seed=0,
        )

    print("[3/4] model")
    cfg = HGRNBitConfig(
        vocab_size=arch.vocab_size,
        hidden_size=arch.hidden_size,
        num_hidden_layers=arch.num_hidden_layers,
        max_position_embeddings=arch.max_position_embeddings,
    )
    model = HGRNBitForCausalLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      params: {n_params:,}")

    # E2.4.A.1: attach BitLinearStatTracker forward hooks BEFORE
    # training begins so every forward pass contributes to per-channel
    # running_max accumulation. Only enabled when --balanced-calib is
    # set; baseline path stays hook-free.
    #
    # F.5.3 (V32-prep): pass start_recording_at_step so the trackers
    # drop forward calls until the warmup boundary. A top-level
    # forward-post-hook (registered below) advances each tracker's
    # current_step once per training-mode forward, so the threshold
    # progresses naturally with step_idx.
    trackers: dict | None = None
    step_advance_handle = None
    if args.balanced_calib:
        if args.skip_warmup_calibration >= train_hp.n_steps:
            raise RuntimeError(
                f"--skip-warmup-calibration ({args.skip_warmup_calibration}) "
                f">= n_steps ({train_hp.n_steps}); the calibration accumulator "
                f"would record 0 forward calls. Reduce the warmup or increase "
                f"n_steps so at least one post-warmup step exists.",
            )
        accumulator_mode = "ema" if args.ema_calibration else "max"
        trackers = attach_stat_trackers(
            model,
            start_recording_at_step=args.skip_warmup_calibration,
            accumulator_mode=accumulator_mode,
            ema_alpha=args.ema_alpha,
        )
        ema_suffix = (
            f", accumulator=ema(α={args.ema_alpha})"
            if args.ema_calibration
            else ", accumulator=max"
        )
        print(
            f"      stat trackers attached to {len(trackers)} BitLinear sites "
            f"(start_recording_at_step={args.skip_warmup_calibration}{ema_suffix})",
        )

        # Advance each tracker's step counter once per training-mode
        # top-level forward. Gated on `module.training` so val passes
        # do not bump it (and `compute_quant_error_per_channel` cannot
        # bump it either: trackers are detached before quant-error runs).
        def _step_advance_post_hook(module, _inputs, _output):
            if module.training:
                advance_trackers(trackers)

        step_advance_handle = model.register_forward_hook(_step_advance_post_hook)

    # E2.1.2: attach HadamardRotation pre-hook on every block.attn.o_proj
    # BitLinear. Per Q1 closure (FJQ_PHASE_E_E2_FINDINGS.md v1.1 §4),
    # rotation applies ONLY to o_proj — i/f/g_proj are HGRN gated-linear-
    # recurrence projections, structurally distinct from QKV. The hook
    # REPLACES the input tensor with H @ x before BitLinear sees it,
    # so quantization operates on outlier-spread activations.
    #
    # Single shared HadamardRotation instance — all o_proj sites at Medium
    # share `hidden_size` (=512), so one matrix in memory suffices. Hook
    # handles tracked for clean teardown post-val.
    hadamard_module: HadamardRotation | None = None
    hadamard_handles: list = []
    if args.hadamard:
        hadamard_module = HadamardRotation(arch.hidden_size).to(device)

        def _hadamard_pre_hook(_module, inputs):
            x = inputs[0]
            return (hadamard_module(x),)

        # F.6.3: which attn projection sites to rotate. Audit at
        # paper/intllm/ablations/igf_proj_dim_audit.json confirms shared
        # HadamardRotation(hidden_size) is dimensionally safe at
        # expand_ratio=1 (R1+R2+R3 all PASS for Mini/Base/Medium).
        site_suffixes = {
            "o":    [".attn.o_proj"],
            "igfo": [".attn.i_proj", ".attn.f_proj", ".attn.g_proj", ".attn.o_proj"],
            "igf":  [".attn.i_proj", ".attn.f_proj", ".attn.g_proj"],
        }[args.hadamard_sites]

        n_attached = 0
        for name, module in model.named_modules():
            if any(name.endswith(suf) for suf in site_suffixes) and is_bitlinear(module):
                handle = module.register_forward_pre_hook(_hadamard_pre_hook)
                hadamard_handles.append(handle)
                n_attached += 1
        print(
            f"      Hadamard rotation attached to {n_attached} sites "
            f"(mode={args.hadamard_sites}, dim={arch.hidden_size})",
        )
        if n_attached == 0:
            raise RuntimeError(
                f"args.hadamard set but no BitLinear modules matched "
                f"hadamard_sites={args.hadamard_sites!r} (suffixes={site_suffixes}) — "
                f"model layout may not match expected HGRNBit topology",
            )

    print(f"[4/4] training {train_hp.n_steps} steps (tag={args.tag})")

    def capped_batches():
        for i, b in enumerate(batches):
            if i >= train_hp.n_steps:
                break
            yield b

    t0 = time.time()
    if args.proof_of_life:
        warmup, total = 0, 0
    else:
        warmup, total = train_hp.warmup_steps, train_hp.n_steps
    ckpt_every = 0 if args.proof_of_life else train_hp.ckpt_every

    result = train_loop(
        model,
        capped_batches(),
        config=TrainConfig(
            lr=train_hp.lr,
            weight_decay=train_hp.weight_decay,
            grad_clip_norm=train_hp.grad_clip_norm,
            log_every=train_hp.log_every,
            warmup_steps=warmup,
            total_steps=total,
            min_lr_ratio=0.1,
            ckpt_every=ckpt_every,
            ckpt_dir=str(args.ckpt_dir / args.tag) if ckpt_every > 0 else None,
            keep_last_n_ckpts=3,
            watchdog_idle_seconds=0 if args.proof_of_life else 1800,
        ),
    )
    elapsed = time.time() - t0
    print(f"\n  elapsed       : {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    print(f"  initial loss  : {result.initial_loss:.4f}")
    print(f"  final loss    : {result.final_loss:.4f}")

    # F.5.3: drop the step-advance post-hook before model state is
    # consumed downstream (quant-error pass, val_loss). The trackers'
    # current_step is now frozen at result.steps; the BitLinear hooks
    # are still attached but `detach_stat_trackers` (after save) will
    # clear them.
    if step_advance_handle is not None:
        step_advance_handle.remove()

    # E2.4.A.2: post-training, compute bit allocation + channel
    # permutation per BitLinear and serialize to a single .pt artifact
    # alongside the ablation JSON. Hook teardown happens after save so
    # the trackers' running_max stays valid through the dump.
    tracker_maps_path: Path | None = None
    if trackers is not None:
        # 1 ckpt per HGRN block has 6 BitLinears (4 attn + 2 MLP) per Q1
        # closure in FJQ_PHASE_E_E2_FINDINGS.md §4 + 1 lm_head BitLinear,
        # so for L=12 (Medium) we expect 6×12 + 1 = 73. Print the layer
        # count for smoke confirmation.
        tracker_maps_path = ABLATIONS_DIR / f"medium_{args.tag}_maps.pt"
        save_calibration_maps(
            trackers,
            tracker_maps_path,
            top_k_pct=E2_4_DEFAULT_TOP_K_PCT,
            low_bits=E2_4_DEFAULT_LOW_BITS,
            high_bits=E2_4_DEFAULT_HIGH_BITS,
            extra_meta={
                "tag": args.tag,
                "id_share": float(BILINGUAL_RATIO_DEFAULT),
                "n_steps": int(result.steps),
                "n_layers_arch": int(arch.num_hidden_layers),
                "skip_warmup_calibration_cli": int(args.skip_warmup_calibration),
                "ema_calibration_cli": bool(args.ema_calibration),
                "ema_alpha_cli": float(args.ema_alpha),
            },
        )
        print(f"  calibration maps : {tracker_maps_path}  ({len(trackers)} BitLinear sites)")
        detach_stat_trackers(model)

    # E2.4.C.3: when --balanced-calib is set AND we're not in POL/dry-run,
    # run the offline quantization-error metric IN-PROCESS so the trained
    # model state is reused without checkpoint reload. Spec §4-§7 of
    # FJQ_PHASE_E_E2_4_C_METRIC_SPEC.md v1.0.
    quant_error_path: Path | None = None
    if args.balanced_calib and not args.proof_of_life and tracker_maps_path is not None:
        n_qe_batches = args.quant_error_n_batches
        print(f"\n  running quant-error metric ({n_qe_batches} batches, seed=42) ...")
        qe_stream = bilingual_stream(
            tokenizer=tok,
            seq_len=train_hp.seq_len,
            batch_size=train_hp.batch_size,
            id_share=BILINGUAL_RATIO_DEFAULT,
            device=device,
            seed=42,
        )
        quant_error_path = ABLATIONS_DIR / f"medium_{args.tag}_quant_error.json"
        qe_t0 = time.time()
        qe_result = compute_quant_error_per_channel(
            model,
            batches=qe_stream,
            n_batches=n_qe_batches,
            bit_map_path=tracker_maps_path,
            device=device,
            out_path=quant_error_path,
        )
        qe_elapsed = time.time() - qe_t0
        print(f"  quant-error elapsed       : {qe_elapsed:.1f} s ({qe_elapsed/60:.1f} min)")
        print(f"  global_mean_reduction     : {qe_result['global_mean_reduction']:+.4f}")
        print(
            f"  outlier_global_reduction  : {qe_result['outlier_global_reduction']:+.4f}  "
            f"(gate ≥ {qe_result['gate_threshold']:.2f})",
        )
        print(
            f"  E2.4.C gate               : {'PASS ✓' if qe_result['gate_pass'] else 'FAIL ✗ (or OBSERVATION if 0.05-0.10)'}",
        )
        print(f"  quant-error JSON          : {quant_error_path}")

    val_loss = float("nan")
    if not args.proof_of_life:
        print(f"\n  measuring val_loss over {args.val_steps} batches ...")
        val_stream = slimpajama_stream(
            tokenizer=tok,
            seq_len=train_hp.seq_len,
            batch_size=train_hp.batch_size,
            device=device,
            seed=999,
        )
        val_loss = run_held_out_loss(
            model, batches=val_stream, n_steps=args.val_steps, device=device,
        )
        print(f"  val_loss      : {val_loss:.4f}  (PPL {math.exp(val_loss):,.1f})")
        print(
            f"  Medium gate     : val_loss < {MEDIUM_VAL_LOSS_GATE} -> "
            f"{'PASS ✓' if val_loss < MEDIUM_VAL_LOSS_GATE else 'FAIL ✗'}"
        )

    _write_ablation_json(
        out_path,
        args.tag,
        features_active=features_active,
        features_requested=features_requested,
        n_steps=result.steps,
        val_loss=val_loss,
        initial_loss=result.initial_loss,
        final_loss=result.final_loss,
        training_seconds=elapsed,
        model_params=n_params,
        proof_of_life=args.proof_of_life,
        dry_run=False,
        tracker_maps_path=tracker_maps_path,
        quant_error_path=quant_error_path,
    )
    print(f"\n  ablation JSON : {out_path}")

    # E2.1.2 teardown: remove Hadamard pre-hooks before model is freed,
    # so PyTorch's hook-handle cleanup path doesn't need to traverse a
    # half-deleted module tree.
    for h in hadamard_handles:
        h.remove()

    # Explicit teardown
    del model
    del tok
    if args.proof_of_life:
        return 0  # POL is harness-wiring sanity, not a gate
    return 0 if val_loss < MEDIUM_VAL_LOSS_GATE else 1


if __name__ == "__main__":
    raise SystemExit(main())
