"""Tests for `intllm.quant` — verify forward matches FP reference within
spec tolerance, and backward is autograd-correct.
"""

from __future__ import annotations

import pytest
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
    activation_quant_per_channel,
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


# -----------------------------------------------------------------
# E2.4.C.2.1 — activation_quant_per_channel
# -----------------------------------------------------------------

def test_activation_quant_per_channel_hand_computed_reference() -> None:
    """Tiny known input + known map → output matches hand-computed.

    in_features = 4. Two channels at 8-bit, two at 10-bit.
    running_max chosen so the per-channel scales are simple integers.
    """
    # 8-bit:  qmax = 127, scale_c = 127 / running_max[c]
    # 10-bit: qmax = 511, scale_c = 511 / running_max[c]
    running_max = torch.tensor([1.0, 2.0, 1.0, 2.0])
    bits = torch.tensor([8, 8, 10, 10], dtype=torch.int32)

    # Hand inputs, all within [-running_max, running_max] → no clip.
    # row 0: [0.5, 1.0, 0.25, 0.5]
    # row 1: [-1.0, -0.5, -0.5, -1.0]
    x = torch.tensor([[0.5, 1.0, 0.25, 0.5], [-1.0, -0.5, -0.5, -1.0]])

    y = activation_quant_per_channel(x, running_max, bits)

    # Hand-quantize:
    #   chan 0: scale = 127/1 = 127, q = round(x*127), then /127
    #     row 0: round(0.5*127)=64 → 64/127 ≈ 0.5039
    #     row 1: round(-1.0*127)=-127 → -127/127 = -1.0
    #   chan 1: scale = 127/2 = 63.5
    #     row 0: round(1.0*63.5)=64 (banker's rounds 63.5→64? PyTorch rounds half-to-even)
    #            → 64/63.5 ≈ 1.0079
    #     row 1: round(-0.5*63.5)=round(-31.75)=-32 → -32/63.5 ≈ -0.5039
    #   chan 2 (10-bit): scale = 511/1 = 511
    #     row 0: round(0.25*511)=128 → 128/511 ≈ 0.2505
    #     row 1: round(-0.5*511)=round(-255.5)=-256 (half-to-even) → -256/511 ≈ -0.5009
    #   chan 3 (10-bit): scale = 511/2 = 255.5
    #     row 0: round(0.5*255.5)=round(127.75)=128 → 128/255.5 ≈ 0.5010
    #     row 1: round(-1.0*255.5)=-256 → -256/255.5 ≈ -1.0020
    expected = torch.tensor([
        [64 / 127.0, 64 / 63.5, 128 / 511.0, 128 / 255.5],
        [-127 / 127.0, -32 / 63.5, -256 / 511.0, -256 / 255.5],
    ])
    assert torch.allclose(y, expected, atol=1e-4), (
        f"hand-computed reference mismatch:\n  got: {y}\n  want: {expected}"
    )


def test_activation_quant_per_channel_preserves_shape_and_dtype() -> None:
    """Input shape `(B, T, in_features)` round-trips unchanged."""
    in_features = 8
    running_max = torch.full((in_features,), 5.0)
    bits = torch.full((in_features,), 8, dtype=torch.int32)

    x = torch.randn(2, 16, in_features)
    y = activation_quant_per_channel(x, running_max, bits)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_activation_quant_per_channel_supports_2d_and_3d_input() -> None:
    """Same channel count, different leading shapes — both must work."""
    in_features = 4
    running_max = torch.tensor([1.0, 1.0, 1.0, 1.0])
    bits = torch.tensor([8, 8, 8, 8], dtype=torch.int32)

    x_2d = torch.tensor([[0.5, -0.5, 0.25, -0.25]])
    x_3d = x_2d.unsqueeze(0)  # (1, 1, 4)

    y_2d = activation_quant_per_channel(x_2d, running_max, bits)
    y_3d = activation_quant_per_channel(x_3d, running_max, bits)
    # Same content, just different leading shape
    assert torch.allclose(y_2d, y_3d.squeeze(0))


def test_activation_quant_per_channel_clips_out_of_range() -> None:
    """Inputs above running_max should clamp to ±qmax/scale, not blow up."""
    in_features = 1
    running_max = torch.tensor([1.0])
    bits = torch.tensor([8], dtype=torch.int32)

    # x[0] = 100 wildly above running_max=1.0 → scale=127, q = clamp(round(100*127),-128,127)=127
    # → output = 127/127 = 1.0, not 100
    x = torch.tensor([[100.0], [-100.0], [0.5]])
    y = activation_quant_per_channel(x, running_max, bits)
    assert y[0, 0].item() == pytest.approx(1.0, abs=1e-5)
    # Lower bound is -128/127, slightly below -1.0
    assert y[1, 0].item() == pytest.approx(-128 / 127.0, abs=1e-5)
    # In-range stays close to its rounded grid point
    assert abs(y[2, 0].item() - 0.5) < 0.01


def test_activation_quant_per_channel_higher_bits_lower_error() -> None:
    """10-bit quantization MUST have lower MSE than 8-bit on the same
    calibrated channel — fundamental bit-width property. This is the
    spec §6 adoption-gate intuition: more bits → finer grid → less error.

    Input must stay inside the calibrated range (no clipping), otherwise
    clipping error dominates and both bit widths look equally bad. Uses
    uniform-in-range input so rounding error is the only noise source.
    """
    torch.manual_seed(42)
    in_features = 16
    rmax = 4.0
    running_max = torch.full((in_features,), rmax)
    # Uniform in (-rmax, rmax) → no clipping, only rounding error.
    x = (torch.rand(8192, in_features) * 2.0 - 1.0) * rmax * 0.95

    bits_8 = torch.full((in_features,), 8, dtype=torch.int32)
    bits_10 = torch.full((in_features,), 10, dtype=torch.int32)

    y_8 = activation_quant_per_channel(x, running_max, bits_8)
    y_10 = activation_quant_per_channel(x, running_max, bits_10)

    mse_8 = (x - y_8).pow(2).mean().item()
    mse_10 = (x - y_10).pow(2).mean().item()
    # 10-bit step is 4× finer than 8-bit step → MSE ∝ step² → ~16× smaller.
    # Allow generous slack for rounding noise; require ≥4× improvement.
    assert mse_10 < mse_8 / 4.0, (
        f"10-bit MSE ({mse_10:.6e}) not significantly lower than 8-bit ({mse_8:.6e}); "
        f"ratio = {mse_8/max(mse_10,1e-12):.1f}× (expected ≥4×)"
    )


def test_activation_quant_per_channel_zero_input_no_nan() -> None:
    """All-zero input + zero running_max → eps floor prevents NaN."""
    in_features = 4
    running_max = torch.zeros(in_features)  # forces eps clamp
    bits = torch.full((in_features,), 8, dtype=torch.int32)
    x = torch.zeros(10, in_features)
    y = activation_quant_per_channel(x, running_max, bits)
    assert torch.isfinite(y).all()
    assert (y == 0).all()  # zero in → zero out


def test_activation_quant_per_channel_idempotent_within_grid() -> None:
    """Quantizing twice with the same map produces the same output —
    quantization is a projection onto the grid; once on it, you stay."""
    torch.manual_seed(0)
    in_features = 8
    running_max = torch.full((in_features,), 3.0)
    bits = torch.tensor([8] * 4 + [10] * 4, dtype=torch.int32)
    x = torch.randn(64, in_features)
    y1 = activation_quant_per_channel(x, running_max, bits)
    y2 = activation_quant_per_channel(y1, running_max, bits)
    assert torch.allclose(y1, y2, atol=1e-5), (
        f"q(q(x)) != q(x); max abs diff {(y1-y2).abs().max().item():.2e}"
    )


def test_activation_quant_per_channel_channel_dim_mismatch_raises() -> None:
    """Mismatched in_features in x vs running_max → ValueError before any compute."""
    x = torch.randn(4, 8)  # in_features=8
    running_max = torch.ones(4)  # mismatch
    bits = torch.full((8,), 8, dtype=torch.int32)
    with pytest.raises(ValueError, match="channel-dim mismatch"):
        activation_quant_per_channel(x, running_max, bits)

    # Mismatched bits_per_channel separately
    running_max_ok = torch.ones(8)
    bits_bad = torch.full((4,), 8, dtype=torch.int32)
    with pytest.raises(ValueError, match="channel-dim mismatch"):
        activation_quant_per_channel(x, running_max_ok, bits_bad)
