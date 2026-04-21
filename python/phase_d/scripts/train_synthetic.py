#!/usr/bin/env python3
"""V31.C.P2.3 — proof-of-life training on synthetic random tokens.

Builds a tiny HGRNBit model (vocab=256, d=64, L=2, ~80K params), trains
for 200 steps on synthetic uniform-random batches, and verifies loss
drops from random init (~ln(256) ≈ 5.55) to a meaningfully lower value.

This is NOT the C.P4.1 Mini config (d=384, L=12, V=32K, 1B tokens).
This is a fast (<60s) sanity check that the whole intllm.train +
intllm.data + intllm.model stack converges end-to-end.

Run:
    cd fajarquant/python/phase_d
    PYTHONPATH=. ../../.venv/bin/python scripts/train_synthetic.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from intllm.data import overfit_token_batches  # noqa: E402
from intllm.model import HGRNBitConfig, HGRNBitForCausalLM  # noqa: E402
from intllm.train import TrainConfig, smoothed_min, train_loop  # noqa: E402


# Tiny config — fast sanity check, NOT C.P4.1 Mini
TINY_VOCAB = 256
TINY_HIDDEN = 64
TINY_LAYERS = 2
TINY_SEQLEN = 64
BATCH = 8
N_STEPS = 200
LR = 3e-3


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    cfg = HGRNBitConfig(
        vocab_size=TINY_VOCAB,
        hidden_size=TINY_HIDDEN,
        num_hidden_layers=TINY_LAYERS,
        max_position_embeddings=TINY_SEQLEN,
        # MLGRU/GLU defaults inherit from HGRNBitConfig
    )
    model = HGRNBitForCausalLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: vocab={TINY_VOCAB} d={TINY_HIDDEN} L={TINY_LAYERS}, params={n_params:,}")

    batches = overfit_token_batches(
        vocab_size=TINY_VOCAB,
        seq_len=TINY_SEQLEN,
        batch_size=BATCH,
        n_batches=N_STEPS,
        seed=0,
        device=device,
    )

    print(f"training {N_STEPS} steps lr={LR}")
    t0 = time.time()
    result = train_loop(model, batches, config=TrainConfig(lr=LR, log_every=50))
    elapsed = time.time() - t0
    print(f"elapsed: {elapsed:.1f}s ({elapsed / N_STEPS * 1000:.1f}ms / step)")

    init_loss = result.initial_loss
    final_loss = result.final_loss
    floor = smoothed_min(result.losses)
    drop_pct = 100.0 * (init_loss - floor) / init_loss

    print(f"\n  initial loss     : {init_loss:.4f}")
    print(f"  final loss       : {final_loss:.4f}")
    print(f"  smoothed-min     : {floor:.4f}")
    print(f"  drop vs initial  : {drop_pct:.1f}%")

    # Gate: smoothed-min loss must drop ≥30% from initial.
    # Looser than the C.P4 Mini gate (loss < 2.0), tighter than "no NaN".
    GATE = 30.0
    passed = drop_pct >= GATE and torch.isfinite(torch.tensor(floor)).item()
    print(f"\n  gate (drop ≥ {GATE}%): {'PASS ✓' if passed else 'FAIL ✗'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
