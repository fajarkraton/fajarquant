#!/usr/bin/env python3
"""V31.C.P2.2 — Medium config training driver.

Trains a 71.3M-param HGRNBit model on SlimPajama-6B for ~122K steps
(≈ 2.0B tokens, Chinchilla-optimal ≈28 tok/p) on RTX 4090 Laptop.

Per FJQ_PHASE_D_GATE_CALIBRATION.md Medium gate:
  PASS condition: val_loss < 4.0 on 1M held-out tokens after training
  (PPL < 55; matches Chinchilla-optimal expectation for 71M × 2B tokens)

Pre-requisite per FJQ_PHASE_D_PRODUCTION_PLAN.md §2.2: Base c.1 PASS
(val_loss < 4.2). If Medium FAILS, Stretch is aborted and arch ablation
(RWKV-7) per FJQ_PHASE_D_ARCH.md §3 is triggered.

Run:
    cd fajarquant/python/phase_d
    PYTHONPATH=. ../../.venv/bin/python scripts/train_medium.py --proof-of-life

    # or full ~11h run:
    PYTHONPATH=. ../../.venv/bin/python scripts/train_medium.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Force line-buffering: nohup-redirected stdout is block-buffered by default,
# which makes long runs unmonitorable until the buffer flushes (every ~4-8 KB).
# See memory/feedback_nohup_python_buffering.md (V31.C medium resume, 2026-04-23).
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from configs.medium import MediumArchConfig, MediumTrainConfig  # noqa: E402
from intllm.data import slimpajama_stream  # noqa: E402
from intllm.eval import run_held_out_loss  # noqa: E402
from intllm.model import HGRNBitConfig, HGRNBitForCausalLM  # noqa: E402
from intllm.tokenizer import get_tokenizer  # noqa: E402
from intllm.train import TrainConfig, find_latest_checkpoint, train_loop  # noqa: E402


def build_model(arch: MediumArchConfig) -> HGRNBitForCausalLM:
    cfg = HGRNBitConfig(
        vocab_size=arch.vocab_size,
        hidden_size=arch.hidden_size,
        num_hidden_layers=arch.num_hidden_layers,
        max_position_embeddings=arch.max_position_embeddings,
    )
    return HGRNBitForCausalLM(cfg)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--proof-of-life", action="store_true",
                   help="Cap at 200 steps, skip val eval, save no checkpoints")
    p.add_argument("--n-steps", type=int, default=None,
                   help="Override training steps (e.g. 20000 for an intermediate run)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch size (default from MediumTrainConfig)")
    p.add_argument("--val-steps", type=int, default=50,
                   help="Number of batches to compute val_loss over (default 50)")
    p.add_argument("--ckpt-dir", type=Path, default=HERE.parent / "checkpoints" / "medium")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    resume_grp = p.add_mutually_exclusive_group()
    resume_grp.add_argument("--resume", type=Path, default=None,
                            help="Resume from a specific ckpt_step_*.pt file")
    resume_grp.add_argument("--resume-auto", action="store_true",
                            help="Resume from the highest-step checkpoint in --ckpt-dir")
    p.add_argument("--watchdog-idle-seconds", type=int, default=1800,
                   help="SIGTERM if step counter idle > N seconds. 0=disabled.")
    args = p.parse_args()

    arch = MediumArchConfig()
    train_hp = MediumTrainConfig()
    from dataclasses import replace
    if args.proof_of_life:
        # POL: 200 steps, no checkpoint, no val
        train_hp = replace(train_hp, n_steps=200, log_every=25)
    elif args.n_steps is not None:
        train_hp = replace(train_hp, n_steps=args.n_steps)
    if args.batch_size is not None:
        train_hp = replace(train_hp, batch_size=args.batch_size)

    print(f"[1/4] config: arch={asdict(arch)}")
    print(f"      train: {asdict(train_hp)}")
    print(f"      device: {args.device}")

    print("[2/4] tokenizer + data stream")
    tok = get_tokenizer()  # cached after first call
    batches = slimpajama_stream(
        tokenizer=tok,
        seq_len=train_hp.seq_len,
        batch_size=train_hp.batch_size,
        device=args.device,
        seed=0,
    )

    print("[3/4] model")
    model = build_model(arch).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      params: {n_params:,}")

    print(f"[4/4] training {train_hp.n_steps} steps")

    # Cap the iterator so train_loop only consumes n_steps batches.
    def capped_batches():
        for i, b in enumerate(batches):
            if i >= train_hp.n_steps:
                break
            yield b

    t0 = time.time()
    # POL skips the LR schedule (200 steps too few for warmup/decay to matter).
    if args.proof_of_life:
        warmup, total = 0, 0
    else:
        warmup, total = train_hp.warmup_steps, train_hp.n_steps
    # Track B step 1 (V31.C.P6.1): interruption-safe intermediate ckpts.
    ckpt_every = 0 if args.proof_of_life else train_hp.ckpt_every

    # Track B step 2 (V31.C.P6.2): resume resolution.
    resume_path: Path | None = None
    if args.resume is not None:
        resume_path = args.resume
    elif args.resume_auto:
        resume_path = find_latest_checkpoint(args.ckpt_dir)
        if resume_path is None:
            print(f"  --resume-auto: no ckpt_step_*.pt in {args.ckpt_dir}; starting fresh")
    if resume_path is not None:
        print(f"  resume requested: {resume_path}")

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
            ckpt_dir=str(args.ckpt_dir) if ckpt_every > 0 else None,
            keep_last_n_ckpts=3,
            resume_from=str(resume_path) if resume_path else None,
            watchdog_idle_seconds=0 if args.proof_of_life else args.watchdog_idle_seconds,
        ),
    )
    if result.checkpoints_written:
        print(f"  intermediate checkpoints written: {len(result.checkpoints_written)}")
        print(f"  last intermediate: {result.checkpoints_written[-1]}")
    elapsed = time.time() - t0

    print(f"\n  elapsed       : {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    print(f"  steps         : {result.steps}")
    print(f"  initial loss  : {result.initial_loss:.4f}")
    print(f"  final loss    : {result.final_loss:.4f}")
    init_ppl = math.exp(result.initial_loss)
    final_ppl = math.exp(result.final_loss)
    print(f"  initial PPL   : {init_ppl:,.1f}")
    print(f"  final PPL     : {final_ppl:,.1f}")

    # Held-out val_loss measurement (skip on POL — too few steps to be meaningful)
    val_loss = float("nan")
    if not args.proof_of_life:
        print(f"\n  measuring val_loss over {args.val_steps} batches (held-out seed) ...")
        val_stream = slimpajama_stream(
            tokenizer=tok,
            seq_len=train_hp.seq_len,
            batch_size=train_hp.batch_size,
            device=args.device,
            seed=999,  # different seed than train (=0)
        )
        val_t0 = time.time()
        val_loss = run_held_out_loss(model, batches=val_stream, n_steps=args.val_steps,
                                      device=args.device)
        print(f"  val_loss      : {val_loss:.4f}  (PPL {math.exp(val_loss):,.1f})")
        print(f"  val elapsed   : {time.time() - val_t0:.1f} s")
        # FJQ_PHASE_D_GATE_CALIBRATION.md Medium gate
        gate = 4.0
        gate_pass = val_loss < gate
        print(f"  Medium gate   : val_loss < {gate} -> "
              f"{'PASS ✓' if gate_pass else 'FAIL ✗'}")

    if not args.proof_of_life:
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = args.ckpt_dir / "medium_final.pt"
        torch.save({
            "arch": asdict(arch),
            "train": asdict(train_hp),
            "state_dict": model.state_dict(),
            "result": {
                "steps": result.steps,
                "initial_loss": result.initial_loss,
                "final_loss": result.final_loss,
                "elapsed_seconds": elapsed,
            },
        }, ckpt_path)
        print(f"  checkpoint    : {ckpt_path}")

    # Save loss trace alongside script (small)
    if args.proof_of_life:
        trace_name = "train_medium_pol_trace.json"
    elif args.n_steps is not None:
        trace_name = f"train_medium_{args.n_steps}_trace.json"
    else:
        trace_name = "train_medium_full_trace.json"
    trace_path = HERE / trace_name
    with trace_path.open("w") as f:
        json.dump({
            "arch": asdict(arch),
            "train": asdict(train_hp),
            "elapsed_seconds": elapsed,
            "losses": result.losses,
            "initial_loss": result.initial_loss,
            "final_loss": result.final_loss,
            "val_loss": val_loss,
        }, f, indent=2)
    print(f"  trace         : {trace_path}")

    # Soft gate for proof-of-life: loss should drop ≥ 1.0 nat
    if args.proof_of_life:
        drop = result.initial_loss - result.final_loss
        gate_pass = drop >= 1.0
        print(f"\n  POL gate (Δ ≥ 1.0 nat): {'PASS ✓' if gate_pass else 'FAIL ✗'} (Δ={drop:.2f})")
        rc = 0 if gate_pass else 1
    else:
        # Medium gate per FJQ_PHASE_D_GATE_CALIBRATION.md
        rc = 0 if val_loss < 4.0 else 1

    # Explicit teardown to silence sentencepiece's atexit
    # `PyGILState_Release` warning observed in V31.C.P4.0 logs.
    del model
    del tok
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
