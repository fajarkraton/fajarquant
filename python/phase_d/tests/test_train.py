"""Training-loop proof-of-life tests (C.P2.3).

Verifies that `intllm.model + intllm.data + intllm.train` compose into
a working end-to-end training stack. The gate is "loss drops
substantially on an overfit-a-single-batch test" — the simplest
possible signal that the stack isn't silently broken.

These tests run in ~10 s on GPU, ~30 s on CPU. They skip if CUDA isn't
available on machines where the upstream mmfreelm Triton kernels won't
build (CPU-only).
"""

from __future__ import annotations

import pytest
import torch

from intllm.data import overfit_token_batches, synthetic_token_batches
from intllm.model import HGRNBitConfig, HGRNBitForCausalLM
from intllm.train import TrainConfig, _lr_lambda, smoothed_min, train_loop


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="mmfreelm Triton kernels require CUDA",
)

# Tiny config — 165K params, trains in ~10 s on RTX 4090 Laptop.
_TINY_VOCAB = 256
_TINY_HIDDEN = 64
_TINY_LAYERS = 2
_TINY_SEQLEN = 64
_BATCH = 8


def _make_tiny_model(device: str = "cuda") -> HGRNBitForCausalLM:
    cfg = HGRNBitConfig(
        vocab_size=_TINY_VOCAB,
        hidden_size=_TINY_HIDDEN,
        num_hidden_layers=_TINY_LAYERS,
        max_position_embeddings=_TINY_SEQLEN,
    )
    return HGRNBitForCausalLM(cfg).to(device)


def test_training_loop_overfits_a_single_batch() -> None:
    """Smoke test: model memorizes a fixed synthetic batch.

    Gate: loss drops ≥80% from random-init (`ln(V) ≈ 5.55`) over 200
    steps. 80% is deliberately loose — if the stack works at all the
    observed drop is >99%, and the test goal is catching "silently
    broken" not "perfectly trained."
    """
    torch.manual_seed(0)
    model = _make_tiny_model()
    batches = overfit_token_batches(
        vocab_size=_TINY_VOCAB,
        seq_len=_TINY_SEQLEN,
        batch_size=_BATCH,
        n_batches=200,
        seed=0,
        device="cuda",
    )
    result = train_loop(model, batches, config=TrainConfig(lr=3e-3, log_every=0))

    assert result.steps == 200
    assert torch.isfinite(torch.tensor(result.final_loss)).item(), (
        f"final loss not finite: {result.final_loss}"
    )

    floor = smoothed_min(result.losses)
    drop_pct = 100.0 * (result.initial_loss - floor) / result.initial_loss
    assert drop_pct >= 80.0, (
        f"loss only dropped {drop_pct:.1f}%; initial={result.initial_loss:.3f} floor={floor:.3f}"
    )


def test_training_loop_stays_finite_on_random_batches() -> None:
    """Stack-stability test: on random tokens (no learnable signal),
    loss hovers near `ln(V)` ≈ 5.55 but NEVER goes NaN/Inf.

    This catches a different class of bugs than the overfit test —
    e.g., unstable quantization + numerical saturation in the MLGRU
    state update would manifest as NaN over enough steps.
    """
    torch.manual_seed(1)
    model = _make_tiny_model()
    batches = synthetic_token_batches(
        vocab_size=_TINY_VOCAB,
        seq_len=_TINY_SEQLEN,
        batch_size=_BATCH,
        n_batches=50,
        seed=42,
        device="cuda",
    )
    result = train_loop(model, batches, config=TrainConfig(lr=1e-3, log_every=0))

    assert result.steps == 50
    assert all(
        torch.isfinite(torch.tensor(x)).item() for x in result.losses
    ), "non-finite loss during random-token training"
    import math

    expected = math.log(_TINY_VOCAB)
    # Loss should stay within ±1 of ln(V) — random tokens have no
    # learnable signal, so big deviations mean instability.
    assert all(abs(x - expected) < 1.0 for x in result.losses[-10:]), (
        f"random-token loss drifted; last 10 = {result.losses[-10:]}"
    )


# -----------------------------------------------------------------
# LR scheduler (C.P4.1 H1 fix)
# -----------------------------------------------------------------

def test_lr_lambda_no_schedule_when_both_zero() -> None:
    """warmup=0, total=0 → constant 1.0 (regression-safe default)."""
    for s in (0, 100, 10_000):
        assert _lr_lambda(s, warmup=0, total=0, min_ratio=0.1) == 1.0


def test_lr_lambda_linear_warmup() -> None:
    """warmup_steps=1000 → linear ramp 0 → 1 over [0, 1000)."""
    assert _lr_lambda(0, warmup=1000, total=2000, min_ratio=0.1) == 0.0
    assert _lr_lambda(500, warmup=1000, total=2000, min_ratio=0.1) == 0.5
    # at warmup boundary, ramp completes; cosine starts at 1.0
    assert abs(_lr_lambda(1000, warmup=1000, total=2000, min_ratio=0.1) - 1.0) < 1e-6


def test_lr_lambda_cosine_decay_to_min_ratio() -> None:
    """At total_steps, lr should equal min_ratio (clamped)."""
    val = _lr_lambda(2000, warmup=1000, total=2000, min_ratio=0.1)
    assert abs(val - 0.1) < 1e-6
    # Past total — clamp
    val = _lr_lambda(5000, warmup=1000, total=2000, min_ratio=0.1)
    assert abs(val - 0.1) < 1e-6


def test_lr_lambda_midpoint_decay() -> None:
    """Halfway through cosine decay → average of 1.0 and min_ratio."""
    # warmup=0, total=1000, midpoint 500
    val = _lr_lambda(500, warmup=0, total=1000, min_ratio=0.1)
    expected = 0.1 + (1.0 - 0.1) * 0.5  # cos(π/2)=0 → 0.5*(1+0)=0.5 cosine
    assert abs(val - expected) < 1e-6
