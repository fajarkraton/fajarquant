"""IntLLM quantization primitives — differentiable simulators of the
integer ops cataloged in `FJQ_PHASE_D_OPS.md`.

Every op here has two roles:
  - **Forward:** numerically simulate what the FajarOS i64 kernel will
    do, including LUT approximations and saturation. Lets QAT see the
    same precision floor the deployed model will see.
  - **Backward:** flow gradients through as if the op were the smooth
    FP reference. Standard straight-through estimator (STE) for
    quantization, exact derivative for sigmoid/SiLU (their LUTs
    approximate smooth functions whose derivatives are well known —
    no need to push gradient through the LUT itself).

These modules are the foundation for the QAT recipe (C.P2.4). They
are framework-agnostic — no FajarOS coupling — so they can also be
unit-tested against torch reference impls.

Op coverage in this file (per `FJQ_PHASE_D_OPS.md` §2):
  - Op 2  IntRMSNorm           — γ-less RMSNorm with optional fixed-point sim
  - Op 4  SigmoidLUT           — 257-entry LUT + linear interp, ±2e-3 max err
  - Op 5  SiLULUT              — composed x · σ(x)
  - Op 10 fake_quantize_absmax_ste — 8-bit absmax fake-quant for activations,
                                     STE backward
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# §1. Sigmoid LUT (FJQ_PHASE_D_OPS.md §2 Op 4)
# ---------------------------------------------------------------------------

# 257 entries spanning [-8, +8]; step = 1/16 = 0.0625. Max LUT-spacing error
# bound: (0.0625)² · max_derivative(σ) / 8 ≤ 1.2e-4. Linear interpolation
# between consecutive entries collapses that further. Spec: max error ≤2e-3
# (well above the analytical bound — leaves room for fp16 rounding inside
# the LUT lookup itself).
SIGMOID_LUT_MIN: float = -8.0
SIGMOID_LUT_MAX: float = 8.0
SIGMOID_LUT_STEP: float = 1.0 / 16.0
SIGMOID_LUT_SIZE: int = 257  # (8 - (-8)) / (1/16) + 1 = 256 + 1


def _build_sigmoid_lut(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Construct the 257-entry sigmoid lookup table.

    Returned tensor is the FP reference values; downstream code may
    cast or round-to-fixed-point as needed.
    """
    xs = torch.arange(SIGMOID_LUT_SIZE, dtype=dtype) * SIGMOID_LUT_STEP + SIGMOID_LUT_MIN
    return torch.sigmoid(xs)


class SigmoidLUT(nn.Module):
    """Sigmoid via 257-entry LUT + linear interpolation.

    Forward matches the FajarOS kernel's `km_sigmoid_lut` to within
    spec tolerance (≤2e-3 absolute error vs torch.sigmoid).
    Backward uses the analytical sigmoid derivative — `σ(x) · (1 - σ(x))`
    — applied to the LUT-interpolated value; this is the standard
    pattern for quant-aware modules whose forward is a smooth-function
    approximation. No STE needed because the function being approximated
    is already smooth.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("lut", _build_sigmoid_lut(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return _SigmoidLUTFn.apply(x, self.lut)  # type: ignore[no-any-return]


class _SigmoidLUTFn(torch.autograd.Function):
    """Custom autograd: forward = LUT, backward = exact σ′(x) · grad."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        lut: torch.Tensor,
    ) -> torch.Tensor:
        # Saturate to LUT range; outside [-8, 8] sigmoid is < 4e-4 from 0/1
        # and the LUT's endpoints already hold those saturated values.
        x_clamped = x.clamp(SIGMOID_LUT_MIN, SIGMOID_LUT_MAX)
        # Index-into-LUT in float space; integer floor + frac for interp
        idx_f = (x_clamped - SIGMOID_LUT_MIN) / SIGMOID_LUT_STEP
        idx_lo = idx_f.floor().long().clamp_max(SIGMOID_LUT_SIZE - 2)
        frac = idx_f - idx_lo.to(idx_f.dtype)
        lo = lut[idx_lo]
        hi = lut[idx_lo + 1]
        y = lo + (hi - lo) * frac
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        (y,) = ctx.saved_tensors  # type: ignore[attr-defined]
        # σ′(x) = σ(x) · (1 - σ(x)) — well-known identity, doesn't need
        # actual x because we already have y = σ(x).
        grad_x = grad_output * y * (1.0 - y)
        return grad_x, None


# ---------------------------------------------------------------------------
# §2. SiLU LUT (FJQ_PHASE_D_OPS.md §2 Op 5)
# ---------------------------------------------------------------------------

class SiLULUT(nn.Module):
    """SiLU (Swish) via composed sigmoid LUT.

    Forward: y = x · σ_lut(x). Backward: y′ = σ(x) + x · σ′(x), the
    standard SiLU derivative.

    Composes `SigmoidLUT` so the LUT is shared if multiple SiLU sites
    exist in the same module.
    """

    def __init__(self, sigmoid_lut: SigmoidLUT | None = None) -> None:
        super().__init__()
        self.sigmoid = sigmoid_lut if sigmoid_lut is not None else SigmoidLUT()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return x * self.sigmoid(x)


# ---------------------------------------------------------------------------
# §3. IntRMSNorm (FJQ_PHASE_D_OPS.md §2 Op 2)
# ---------------------------------------------------------------------------

class IntRMSNorm(nn.Module):
    """γ-less RMSNorm matching the BitNet/MatMul-Free convention.

    Standard RMSNorm scales by `√(E[x²])`, then multiplies by a
    learnable γ vector. Phase D (and BitNet b1.58, MatMul-Free LM)
    drop the γ — the post-normalize scaling is absorbed downstream
    into the BitLinear's per-matrix `β/Q_b` rescale. Doing so removes
    the per-channel γ-magnification that V31.R3 isolated as the root
    cause of Gemma 3 pad-collapse in fixed-point.

    The optional `fixed_point_scale` parameter simulates the kernel's
    x1000 fixed-point precision floor. When > 0, both the input and
    intermediate sums are rounded to that scale to mimic the truncate-
    toward-zero behavior of the FajarOS i64 kernel.

    Forward shape: (..., d) → (..., d). Backward is autograd-standard
    for the squared-mean + isqrt chain — no STE.
    """

    def __init__(self, eps: float = 1e-6, fixed_point_scale: int = 0) -> None:
        super().__init__()
        self.eps = eps
        self.fixed_point_scale = fixed_point_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fixed_point_scale
        if scale > 0:
            # Simulate x1000 fixed-point precision by round-trip through
            # integer-scaled space. Differentiable via STE (the rounding is
            # treated as identity in backward).
            x_q = torch.round(x * scale) / scale
            x = x + (x_q - x).detach()
        # Standard RMSNorm: divide by RMS, no learnable γ.
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        denom = torch.sqrt(ms + self.eps)
        y = x / denom
        if scale > 0:
            y_q = torch.round(y * scale) / scale
            y = y + (y_q - y).detach()
        return y


# ---------------------------------------------------------------------------
# §4. Fake-quantize absmax (FJQ_PHASE_D_OPS.md §2 Op 10)
# ---------------------------------------------------------------------------

class _FakeQuantizeAbsmaxSTE(torch.autograd.Function):
    """8-bit absmax fake-quantization with straight-through backward.

    Forward:
        γ_x = max_i |x_i|   (per call, not learned)
        s   = Q_b / γ_x
        x̃   = clip(round(x · s), -Q_b, Q_b) / s   ← back to FP scale
    Backward:
        STE — pass gradient through unchanged inside [-γ_x, γ_x],
        zero outside (saturation is non-differentiable).
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        n_bits: int,
    ) -> torch.Tensor:
        q_b = (1 << (n_bits - 1)) - 1  # 127 for n_bits=8
        # Per-tensor absmax. Using a small floor avoids division-by-zero
        # on exactly-zero inputs (rare, but happens during init).
        gamma_x = x.abs().max().clamp_min(1e-8)
        s = q_b / gamma_x
        x_q = torch.round(x * s).clamp(-q_b, q_b) / s
        # Save the saturation mask for the STE pass-through gradient
        sat_mask = (x.abs() <= gamma_x)
        ctx.save_for_backward(sat_mask)
        return x_q

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        (sat_mask,) = ctx.saved_tensors  # type: ignore[attr-defined]
        return grad_output * sat_mask.to(grad_output.dtype), None


def fake_quantize_absmax_ste(x: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """Functional API for absmax fake-quantization with STE backward.

    `n_bits=8` matches the BitNet/MatMul-Free Q_b=127 activation
    quantization convention. `n_bits=10` is reserved for the per-coord
    adaptive-bit-allocation branch (§6.9 R5 contribution, deferred to
    C.P2.4 QAT harness).
    """
    return _FakeQuantizeAbsmaxSTE.apply(x, n_bits)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# §4.5. Per-channel adaptive quantization (E2.4.C.2)
# ---------------------------------------------------------------------------

def activation_quant_per_channel(
    x: torch.Tensor,
    running_max: torch.Tensor,
    bits_per_channel: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Per-channel calibrated quantize-cast-dequantize.

    Implements `q_calibrated` from `FJQ_PHASE_E_E2_4_C_METRIC_SPEC.md`
    §3.2:

      scale_c   = (2^(bits_c - 1) - 1) / max(running_max[c], eps)
      clip_lo_c = -(2^(bits_c - 1))
      clip_hi_c = (2^(bits_c - 1) - 1)
      y[..., c] = round(x[..., c] * scale_c).clamp(clip_lo_c, clip_hi_c) / scale_c

    Args:
        x: Activation tensor of shape `(..., in_features)`. Last
            dimension is the channel axis. Any leading shape is
            preserved (caller should `.reshape(-1, in_features)` if a
            flat 2D view is preferred for streaming SSE accumulation,
            but this function does NOT require it).
        running_max: Float tensor of shape `(in_features,)`; per-channel
            absmax accumulated during calibration (output of
            `BitLinearStatTracker.running_max`).
        bits_per_channel: Int tensor of shape `(in_features,)`;
            per-channel bit width (e.g. 8 for the bulk, 10 for the
            top-K outlier channels). Output of
            `compute_bit_allocation`.
        eps: Floor on `running_max` to avoid division by zero on
            channels that never saw nonzero activation. Default 1e-5
            matches upstream `activation_quant`.

    Returns:
        Tensor of the same shape and dtype as `x`, quantized then
        dequantized.

    Notes:
        - Standalone math function; does NOT modify `BitLinear.forward`.
          Used only by `intllm.eval.compute_quant_error_per_channel`
          for offline metric evaluation under E2.4 Option C scope.
          Wiring this into `BitLinear.forward` is Option B work,
          deferred per `FJQ_PHASE_E_E2_4_FINDINGS.md` v1.2 §6.1.
        - No autograd customization; `round()` is non-differentiable.
          STE training-time use should go through
          `fake_quantize_absmax_ste` instead. This function is for
          INFERENCE-time error measurement only.
        - All ops broadcast over the leading dims of `x`.
    """
    if x.shape[-1] != running_max.shape[0]:
        raise ValueError(
            f"channel-dim mismatch: x.shape[-1]={x.shape[-1]} but "
            f"running_max.shape[0]={running_max.shape[0]}",
        )
    if x.shape[-1] != bits_per_channel.shape[0]:
        raise ValueError(
            f"channel-dim mismatch: x.shape[-1]={x.shape[-1]} but "
            f"bits_per_channel.shape[0]={bits_per_channel.shape[0]}",
        )

    # Move calibration tensors onto x's device once (no-op if already
    # there). Caller usually attaches via `.to(device)` before the
    # streaming loop; this is defense in depth.
    rmax = running_max.to(device=x.device, dtype=torch.float32)
    bits = bits_per_channel.to(device=x.device, dtype=torch.long)

    # Use long arithmetic for `1 << (bits - 1)` to avoid float silently
    # being produced via Python `**` semantics on tensors. Cast to
    # x.dtype only at the multiplicative-scale step.
    pow2_minus_1 = (1 << (bits - 1)).to(dtype=torch.float32)  # 2^(bits-1)
    qmax = pow2_minus_1 - 1                                   # high clip = 2^(bits-1) - 1
    qmin = -pow2_minus_1                                       # low  clip = -2^(bits-1)

    scale = qmax / rmax.clamp(min=eps)                        # shape (in_features,)
    # Promote to x.dtype for arithmetic; clamp params stay float32
    scale_x = scale.to(dtype=x.dtype)

    # Quantize, clamp per-channel (broadcast over leading dims), dequant.
    q = (x * scale_x).round()
    q = torch.maximum(q, qmin.to(dtype=x.dtype))
    q = torch.minimum(q, qmax.to(dtype=x.dtype))
    return q / scale_x


# ---------------------------------------------------------------------------
# §5. Spec-compliance constants (read by tests)
# ---------------------------------------------------------------------------

#: Maximum absolute error of `SigmoidLUT` vs `torch.sigmoid` over the
#: domain [-8, 8]. Per FJQ_PHASE_D_OPS.md §2 Op 4 spec: ≤ 1 LSB at
#: x1000 fixed point ⇒ ≤ 1e-3 absolute. We assert ≤ 2e-3 in tests to
#: leave headroom for fp16 LUT rounding.
SIGMOID_LUT_MAX_ABS_ERROR: float = 2e-3

#: Maximum absolute error of `SiLULUT` vs `torch.nn.functional.silu`.
#: SiLU = x · σ(x); error scales with |x|, so |error| ≤ |x|·SIGMOID_LUT_MAX_ABS_ERROR
#: + |σ(x)|·O(0). For |x| ≤ 8: |error| ≤ 8 · 2e-3 = 1.6e-2.
SILU_LUT_MAX_ABS_ERROR: float = 1.6e-2


__all__ = [
    "IntRMSNorm",
    "SiLULUT",
    "SigmoidLUT",
    "SIGMOID_LUT_MAX_ABS_ERROR",
    "SIGMOID_LUT_MAX",
    "SIGMOID_LUT_MIN",
    "SIGMOID_LUT_SIZE",
    "SIGMOID_LUT_STEP",
    "SILU_LUT_MAX_ABS_ERROR",
    "activation_quant_per_channel",
    "fake_quantize_absmax_ste",
]


def _next_power_of_two(n: int) -> int:
    """Helper used by tests + downstream calibration code."""
    if n <= 1:
        return 1
    return 1 << (math.ceil(math.log2(n)))
