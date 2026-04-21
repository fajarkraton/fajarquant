"""Tests for `intllm.quant` — verify forward matches FP reference within
spec tolerance, and backward is autograd-correct.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from intllm.quant import (
    SIGMOID_LUT_MAX,
    SIGMOID_LUT_MAX_ABS_ERROR,
    SIGMOID_LUT_MIN,
    SILU_LUT_MAX_ABS_ERROR,
    IntRMSNorm,
    SigmoidLUT,
    SiLULUT,
    fake_quantize_absmax_ste,
)


# -----------------------------------------------------------------
# SigmoidLUT
# -----------------------------------------------------------------

def test_sigmoid_lut_forward_within_spec_tolerance() -> None:
    sigmoid = SigmoidLUT()
    # Sweep the full LUT range plus 1.5x past saturation
    x = torch.linspace(SIGMOID_LUT_MIN * 1.5, SIGMOID_LUT_MAX * 1.5, 4096)
    y_lut = sigmoid(x)
    y_ref = torch.sigmoid(x)
    err = (y_lut - y_ref).abs().max().item()
    assert err <= SIGMOID_LUT_MAX_ABS_ERROR, (
        f"SigmoidLUT max abs error {err:.4e} exceeds spec {SIGMOID_LUT_MAX_ABS_ERROR:.4e}"
    )


def test_sigmoid_lut_endpoints_saturate() -> None:
    sigmoid = SigmoidLUT()
    # Past saturation, should be very close to 0 / 1
    assert sigmoid(torch.tensor([-100.0])).item() < 1e-3
    assert sigmoid(torch.tensor([+100.0])).item() > 1.0 - 1e-3


def test_sigmoid_lut_backward_matches_torch_sigmoid_grad() -> None:
    sigmoid = SigmoidLUT()
    # Use a sample inside the LUT range so saturation doesn't interfere
    x = torch.linspace(-4.0, 4.0, 200, requires_grad=True)
    y = sigmoid(x).sum()
    y.backward()
    expected = torch.sigmoid(x).detach() * (1 - torch.sigmoid(x).detach())
    assert torch.allclose(x.grad, expected, atol=1e-4), (
        f"backward grad max diff = {(x.grad - expected).abs().max().item():.4e}"
    )


# -----------------------------------------------------------------
# SiLULUT
# -----------------------------------------------------------------

def test_silu_lut_forward_within_spec_tolerance() -> None:
    silu = SiLULUT()
    x = torch.linspace(-8.0, 8.0, 4096)
    y_lut = silu(x)
    y_ref = F.silu(x)
    err = (y_lut - y_ref).abs().max().item()
    assert err <= SILU_LUT_MAX_ABS_ERROR, (
        f"SiLULUT max abs error {err:.4e} exceeds spec {SILU_LUT_MAX_ABS_ERROR:.4e}"
    )


def test_silu_lut_backward_finite_and_close_to_torch() -> None:
    silu = SiLULUT()
    x = torch.linspace(-4.0, 4.0, 200, requires_grad=True)
    y = silu(x).sum()
    y.backward()
    # Reference: SiLU' = σ(x) + x·σ(x)·(1-σ(x))
    s = torch.sigmoid(x).detach()
    expected = s + x.detach() * s * (1 - s)
    diff = (x.grad - expected).abs().max().item()
    assert torch.isfinite(x.grad).all(), "non-finite SiLU LUT gradient"
    # SiLU backward composes the LUT-approximation with x — slightly
    # looser tolerance than pure sigmoid.
    assert diff <= 1e-3, f"SiLULUT backward max diff = {diff:.4e}"


# -----------------------------------------------------------------
# IntRMSNorm
# -----------------------------------------------------------------

def test_int_rmsnorm_matches_torch_rmsnorm_no_gamma() -> None:
    norm = IntRMSNorm(eps=1e-6)
    x = torch.randn(4, 16, 384)
    y = norm(x)
    # Reference: torch's nn.functional.rms_norm exists in 2.4+; otherwise
    # build the formula manually.
    ms = x.pow(2).mean(dim=-1, keepdim=True)
    y_ref = x / torch.sqrt(ms + 1e-6)
    assert torch.allclose(y, y_ref, atol=1e-6)


def test_int_rmsnorm_fixed_point_scale_round_trip() -> None:
    """When fixed_point_scale > 0, output should be on the int grid.

    Tolerance 5e-4 absorbs fp32 round-trip noise from `(round(y*1000)/1000)`
    — the actual quantization is to 1/1000 grid, well coarser than fp32
    machine epsilon at unit-magnitude values.
    """
    norm = IntRMSNorm(eps=1e-6, fixed_point_scale=1000)
    x = torch.randn(2, 8, 64)
    y = norm(x)
    on_grid = ((y * 1000) - (y * 1000).round()).abs() < 5e-4
    assert on_grid.all(), "fixed-point scale=1000 output is not on the integer grid"


def test_int_rmsnorm_backward_finite() -> None:
    norm = IntRMSNorm(eps=1e-6)
    x = torch.randn(2, 4, 32, requires_grad=True)
    norm(x).sum().backward()
    assert torch.isfinite(x.grad).all()


# -----------------------------------------------------------------
# fake_quantize_absmax_ste
# -----------------------------------------------------------------

def test_fake_quantize_absmax_ste_forward_clips_to_grid() -> None:
    x = torch.linspace(-2.0, 2.0, 100)
    x_q = fake_quantize_absmax_ste(x, n_bits=8)
    # Recoverable scale: 127 / max(|x|) = 127 / 2.0 = 63.5
    # Each output should be representable as k/63.5 for k ∈ [-127, 127]
    scale = 127.0 / x.abs().max().item()
    on_grid = ((x_q * scale) - (x_q * scale).round()).abs() < 1e-4
    assert on_grid.all(), "fake-quant 8-bit output not on the integer grid"
    assert x_q.abs().max().item() <= x.abs().max().item() + 1e-6


def test_fake_quantize_absmax_ste_backward_passes_gradient() -> None:
    """STE: gradient should flow through the fake-quant op as identity
    (within saturation)."""
    x = torch.linspace(-1.5, 1.5, 50, requires_grad=True)
    x_q = fake_quantize_absmax_ste(x, n_bits=8)
    # Sum + backward — d(sum(x_q))/dx should be ~all-ones via STE
    x_q.sum().backward()
    assert torch.isfinite(x.grad).all()
    # Within the absmax range, every input was in saturation mask, so
    # gradient is identity.
    assert torch.allclose(x.grad, torch.ones_like(x.grad), atol=1e-6), (
        f"STE backward should be identity within abs-range; got max diff "
        f"{(x.grad - 1).abs().max().item():.4e}"
    )


def test_fake_quantize_absmax_ste_zero_input_no_nan() -> None:
    """Edge case: all-zero tensor should not divide by zero (clamp_min)."""
    x = torch.zeros(10, requires_grad=True)
    x_q = fake_quantize_absmax_ste(x)
    assert torch.isfinite(x_q).all()
    x_q.sum().backward()
    assert torch.isfinite(x.grad).all()
