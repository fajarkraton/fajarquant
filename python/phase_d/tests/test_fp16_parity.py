"""Tests for intllm.fp16_parity (Phase 3.5 IntLLM differentiator).

Verifies the monkey-patch context manager + per-layer hook capture
work end-to-end on a tiny HGRNBit model. The actual rel-error number
on a real trained checkpoint is measured by scripts/run_fp16_parity.py;
these tests only verify mechanics.
"""
from __future__ import annotations

import pytest
import torch

from intllm.fp16_parity import (
    LayerParity,
    fp16_reference_mode,
    measure_parity,
    summary_stats,
)
from intllm.model import HGRNBitConfig, HGRNBitForCausalLM


_DEVICE = "cuda" if torch.cuda.is_available() else None


@pytest.fixture(scope="module")
def tiny_model() -> HGRNBitForCausalLM:
    """A small 2-layer model to keep tests fast.

    Upstream HGRNBit uses fused triton kernels (FusedRMSNormSwishGate)
    that require CUDA tensors at runtime. Tests skip cleanly on CPU-only
    boxes per the module-level skip below.
    """
    if _DEVICE is None:
        pytest.skip("HGRNBit forward requires CUDA (triton kernels)")
    cfg = HGRNBitConfig(
        vocab_size=64,
        hidden_size=64,    # too small triggers triton tile-shape errors
        num_hidden_layers=2,
        max_position_embeddings=32,
    )
    torch.manual_seed(0)
    return HGRNBitForCausalLM(cfg).to(_DEVICE).eval()


@pytest.fixture
def tiny_input() -> torch.Tensor:
    if _DEVICE is None:
        pytest.skip("HGRNBit forward requires CUDA (triton kernels)")
    torch.manual_seed(0)
    return torch.randint(0, 64, (1, 8), device=_DEVICE)


def test_fp16_mode_changes_logits(tiny_model, tiny_input):
    """The fp16 reference path MUST produce different logits from the
    ternary path on at least one element — otherwise the monkey-patch
    is silently a no-op."""
    with torch.no_grad():
        out_ternary = tiny_model(tiny_input).logits.clone()
        with fp16_reference_mode():
            out_fp16 = tiny_model(tiny_input).logits.clone()
    assert not torch.allclose(out_ternary, out_fp16, atol=1e-6, rtol=0), (
        "fp16_reference_mode produced identical output to ternary — "
        "monkey-patch is not effective"
    )


def test_fp16_mode_restores_forward_on_exit(tiny_model, tiny_input):
    """Exiting the context manager must restore the original BitLinear
    forward (so subsequent unscoped forwards are quantized again)."""
    with torch.no_grad():
        before = tiny_model(tiny_input).logits.clone()
        with fp16_reference_mode():
            _ = tiny_model(tiny_input).logits
        after = tiny_model(tiny_input).logits.clone()
    assert torch.allclose(before, after, atol=0, rtol=0), (
        "Forward did not match itself before/after context exit — "
        "monkey-patch leaked across the boundary"
    )


def test_fp16_mode_restores_on_exception(tiny_model, tiny_input):
    """If the wrapped block raises, the context must still restore the
    original BitLinear forward (try/finally contract)."""
    with torch.no_grad():
        baseline_a = tiny_model(tiny_input).logits.clone()
        with pytest.raises(RuntimeError, match="sentinel"):
            with fp16_reference_mode():
                raise RuntimeError("sentinel")
        baseline_b = tiny_model(tiny_input).logits.clone()
    assert torch.allclose(baseline_a, baseline_b, atol=0, rtol=0)


def test_measure_parity_returns_per_layer_results(tiny_model, tiny_input):
    results = measure_parity(tiny_model, tiny_input)
    assert len(results) > 0, "at least one BitLinear hook must fire"
    for name, parity in results.items():
        assert isinstance(parity, LayerParity)
        assert parity.name == name
        assert parity.n_elements > 0
        assert parity.fp16_norm_l2 >= 0.0
        assert parity.abs_error_l2 >= 0.0
        # rel_error can be > 1 for ternary models — no upper assertion.
        assert parity.rel_error_l2 >= 0.0


def test_measure_parity_max_layers_truncates(tiny_model, tiny_input):
    full = measure_parity(tiny_model, tiny_input)
    truncated = measure_parity(tiny_model, tiny_input, max_layers=1)
    assert len(truncated) == 1
    assert len(full) > 1
    # Same first layer should appear in both
    assert next(iter(truncated)) == next(iter(full))


def test_summary_stats_aggregates_correctly():
    fake_results = {
        "a": LayerParity("a", 1.0, 0.10, 0.5, 0.5, 10.0, 4),
        "b": LayerParity("b", 2.0, 0.20, 1.0, 1.0, 10.0, 4),
        "c": LayerParity("c", 3.0, 0.30, 1.5, 1.5, 10.0, 4),
    }
    stats = summary_stats(fake_results)
    assert stats["n_layers"] == 3
    assert stats["rel_error_l2_mean"] == pytest.approx(0.20, rel=0, abs=1e-9)
    assert stats["rel_error_l2_min"] == pytest.approx(0.10)
    assert stats["rel_error_l2_max"] == pytest.approx(0.30)


def test_summary_stats_empty():
    assert summary_stats({}) == {"n_layers": 0}
