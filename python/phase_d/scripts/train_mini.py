#!/usr/bin/env python3
"""V31.C.P4.1 — Mini config training driver.

Trains a 21.5M-param HGRNBit model on SlimPajama-6B for ~60K steps
(≈ 500M tokens) on RTX 4090 Laptop.

Per FJQ_PHASE_D_CONFIG.md §5.1 Mini gate:
  PASS condition: val_loss < 4.0 on 1M held-out tokens after training

For C.P4.0 proof-of-life (this commit), pass `--proof-of-life` to
limit to a few hundred steps + skip val eval. Full Mini gate run is
launched without that flag in a subsequent session.

Run:
    cd fajarquant/python/phase_d
    PYTHONPATH=. ../../.venv/bin/python scripts/train_mini.py --proof-of-life

    # or full ~1h run:
    PYTHONPATH=. ../../.venv/bin/python scripts/train_mini.py
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from configs.mini import MiniArchConfig, MiniTrainConfig  # noqa: E402
from intllm.data import slimpajama_stream  # noqa: E402
from intllm.model import HGRNBitConfig, HGRNBitForCausalLM  # noqa: E402
from intllm.tokenizer import get_tokenizer  # noqa: E402
from intllm.train import TrainConfig, train_loop  # noqa: E402


def build_model(arch: MiniArchConfig) -> HGRNBitForCausalLM:
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
    p.add_argument("--ckpt-dir", type=Path, default=HERE.parent / "checkpoints" / "mini")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    arch = MiniArchConfig()
    train_hp = MiniTrainConfig()
    if args.proof_of_life:
        # POL: 200 steps, no checkpoint, no val
        train_hp_overrides = {"n_steps": 200, "log_every": 25}
        from dataclasses import replace
        train_hp = replace(train_hp, **train_hp_overrides)

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
    result = train_loop(
        model,
        capped_batches(),
        config=TrainConfig(
            lr=train_hp.lr,
            weight_decay=train_hp.weight_decay,
            grad_clip_norm=train_hp.grad_clip_norm,
            log_every=train_hp.log_every,
        ),
    )
    elapsed = time.time() - t0

    print(f"\n  elapsed       : {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    print(f"  steps         : {result.steps}")
    print(f"  initial loss  : {result.initial_loss:.4f}")
    print(f"  final loss    : {result.final_loss:.4f}")
    init_ppl = math.exp(result.initial_loss)
    final_ppl = math.exp(result.final_loss)
    print(f"  initial PPL   : {init_ppl:,.1f}")
    print(f"  final PPL     : {final_ppl:,.1f}")

    if not args.proof_of_life:
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = args.ckpt_dir / "mini_final.pt"
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
    trace_path = HERE / ("train_mini_pol_trace.json" if args.proof_of_life
                          else "train_mini_full_trace.json")
    with trace_path.open("w") as f:
        json.dump({
            "arch": asdict(arch),
            "train": asdict(train_hp),
            "elapsed_seconds": elapsed,
            "losses": result.losses,
            "initial_loss": result.initial_loss,
            "final_loss": result.final_loss,
        }, f, indent=2)
    print(f"  trace         : {trace_path}")

    # Soft gate for proof-of-life: loss should drop ≥ 1.0 nat
    if args.proof_of_life:
        drop = result.initial_loss - result.final_loss
        gate_pass = drop >= 1.0
        print(f"\n  POL gate (Δ ≥ 1.0 nat): {'PASS ✓' if gate_pass else 'FAIL ✗'} (Δ={drop:.2f})")
        return 0 if gate_pass else 1
    # Full run gate is val-loss based; not yet implemented (C.P4.1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
