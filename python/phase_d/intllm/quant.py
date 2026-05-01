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
# §4.4. Walsh-Hadamard rotation for outlier suppression (E2.1)
# ---------------------------------------------------------------------------

def _walsh_hadamard_matrix(dim: int) -> torch.Tensor:
    """Build the dim×dim normalized Walsh-Hadamard matrix H.

    Construction by recursive doubling:
      H_1 = [[1]]
      H_{2n} = (1/sqrt(2)) [[H_n, H_n], [H_n, -H_n]]

    Properties (verified by unit tests):
      - Orthogonal: H @ H^T = I (symmetric → H^T = H)
      - Norm-preserving: ||H x|| = ||x|| for any x
      - Energy-spreading: per-coordinate variance of H x is bounded by
        the *average* per-coordinate variance of x, regardless of how
        concentrated x's energy is. This is the property QuaRot/SpinQuant
        exploit to reduce activation outliers before quantization.

    `dim` must be a power of 2; raises ValueError otherwise. Matrix is
    constructed in float32 and stored on CPU; caller moves to device
    via the parent module's `.to(device)`.
    """
    if dim < 1 or (dim & (dim - 1)) != 0:
        raise ValueError(f"Hadamard dim must be a positive power of 2, got {dim}")
    h = torch.tensor([[1.0]])
    while h.shape[0] < dim:
        n = h.shape[0]
        top = torch.cat([h, h], dim=1)
        bot = torch.cat([h, -h], dim=1)
        h = torch.cat([top, bot], dim=0) / math.sqrt(2.0)
    return h


class HadamardRotation(nn.Module):
    """Fixed (non-learned) orthogonal rotation by the Walsh-Hadamard matrix.

    Per QuaRot (arxiv 2404.00456) and SpinQuant (arxiv 2405.16406),
    rotating activations by an orthogonal matrix BEFORE quantization
    spreads outlier energy across coordinates. The fixed Walsh-Hadamard
    rotation is the canonical no-learnable-params baseline; SpinQuant
    extends with a learned rotation on top, deferred for a future
    iteration.

    Mathematical equivalence (training-from-scratch case):

      Original:    y = quant(W) · quant(x)
      Rotated:     y' = quant(W) · quant(H x)
                      ≈ W · H x = (W H) x

    For end-to-end training from random init, the model learns
    W_eff = W H instead of W. The OUTPUT semantics are identical at
    convergence; the QUANTIZATION error is reduced because (H x) has
    flatter per-coordinate magnitude distribution than x.

    For inference-time application to a pre-trained model (NOT this
    sub-task — see E2.1.4 deferred), one would fuse H into the upstream
    weights to preserve numerics; that's the QuaRot inference recipe.

    Wiring: this module is a pure forward function. It does NOT modify
    any BitLinear; E2.1.2 attaches an instance via `register_forward_pre_hook`
    on `block.attn.o_proj` so the rotation runs as a transparent
    pre-processing step. The Q1 closure in `FJQ_PHASE_E_E2_FINDINGS.md`
    v1.1 §4 establishes that ONLY o_proj should be rotated in HGRNBit
    (the i/f/g_proj are gated-linear-recurrence projections, structurally
    distinct from QKV — rotating them is non-canonical).

    Args:
        dim: Hidden dimension to rotate over (must be a power of 2).

    Buffers:
        H: (dim, dim) float32 normalized Walsh-Hadamard matrix.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 1 or (dim & (dim - 1)) != 0:
            raise ValueError(f"HadamardRotation dim must be a positive power of 2, got {dim}")
        self.dim = dim
        self.register_buffer("H", _walsh_hadamard_matrix(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate `x` along its last dim: returns x @ H, shape unchanged."""
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"HadamardRotation: x.shape[-1]={x.shape[-1]} but module dim={self.dim}",
            )
        # Cast H to x's dtype for fp16/bf16 training compatibility.
        return x @ self.H.to(dtype=x.dtype)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


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
    "HadamardRotation",
    "IntRMSNorm",
    "SiLULUT",
    "SigmoidLUT",
    "SIGMOID_LUT_MAX_ABS_ERROR",
    "SIGMOID_LUT_MAX",
    "SIGMOID_LUT_MIN",
    "SIGMOID_LUT_SIZE",
    "SIGMOID_LUT_STEP",
    "SILU_LUT_MAX_ABS_ERROR",
    "SmoothQuantCalibrator",
    "SparseFusedBitLinear",
    "activation_quant_per_channel",
    "fake_quantize_absmax_ste",
    "load_smoothquant_maps",
    "save_smoothquant_maps",
]


def _next_power_of_two(n: int) -> int:
    """Helper used by tests + downstream calibration code."""
    if n <= 1:
        return 1
    return 1 << (math.ceil(math.log2(n)))


# ---------------------------------------------------------------------------
# §5. SmoothQuant PTQ calibration (V32-prep F.5.1)
# ---------------------------------------------------------------------------
# See `docs/FJQ_PHASE_F_F5_1_SMOOTHQUANT_PTQ_DESIGN.md` v1.0 for full spec.
# Per §3.3 the fusion identity is one-line per BitLinear:
#     b.norm.weight.data /= s          (γ ← γ/s)
#     b.weight.data *= s.unsqueeze(0)  (W ← s·W, broadcast along output dim)
# Mathematical FP equivalence preserved (s⁻¹·s = I).

_F51_SCHEMA_VERSION = "1.0"

# §3.5: site-restriction presets — match suffix patterns against
# `model.named_modules()` names.
SMOOTHQUANT_SITE_PRESETS: dict[str, list[str] | None] = {
    "o":      [".attn.o_proj"],
    "igfo":   [".attn.i_proj", ".attn.f_proj", ".attn.g_proj", ".attn.o_proj"],
    "igf":    [".attn.i_proj", ".attn.f_proj", ".attn.g_proj"],
    "o_down": [".attn.o_proj", ".mlp.down_proj"],
    "all":    None,  # special: every BitLinear regardless of name
}


def _resolve_sites(sites: list[str] | str | None) -> list[str] | None:
    """Resolve `sites` argument to a list of name-suffix patterns or
    None (= all BitLinear sites). Accepts preset name string or list.
    """
    if sites is None:
        return SMOOTHQUANT_SITE_PRESETS["o"]
    if isinstance(sites, str):
        if sites not in SMOOTHQUANT_SITE_PRESETS:
            raise ValueError(
                f"unknown site preset {sites!r}; valid: {list(SMOOTHQUANT_SITE_PRESETS)}",
            )
        return SMOOTHQUANT_SITE_PRESETS[sites]
    return list(sites)


def _is_bitlinear(module: nn.Module) -> bool:
    """Local copy of `intllm.qat.is_bitlinear` to avoid a circular
    import (qat.py imports from quant.py). Matches BitLinear-family
    classes by class name regardless of which upstream module path
    imports them (`bitnet.py` BitLinear vs `fusedbitnet.py`
    FusedBitLinear vs BitLinear_wonorm_bmm). All have a ``.norm``
    attribute set in ``__init__`` per upstream impl; the hasattr
    check is a defensive sanity guard for future variants.
    """
    return (
        module.__class__.__name__ in {
            "BitLinear",
            "FusedBitLinear",
            "BitLinear_wonorm_bmm",
            "SparseFusedBitLinear",  # F.10.2 — sparse extension
        }
        and hasattr(module, "norm")
    )


class SmoothQuantCalibrator:
    """Static-calibration SmoothQuant scale generator + apply utility.

    Per V32-prep F.5.1 design v1.0 §3-§4. Workflow:

      1. ``__init__``: bind to model, alpha, sites
      2. ``calibrate(batches)``: forward-pre-hook captures
         ``max(|x|)`` per-channel for each matched BitLinear; weight
         scan computes ``max(|w|)`` per-input-channel. Combined into
         ``s_j = max_act_j ** alpha / max_w_j ** (1 - alpha)`` per
         §4.4 with edge-case clamping.
      3. ``validate(calibration_dict)``: §4.6 mechanical gates
         (finite, range, median, condition number).
      4. ``apply(model, calibration_dict)``: §3.3 fusion identity
         in-place: ``γ ← γ/s`` AND ``W ← s·W``.

    Args:
        model: HGRNBitForCausalLM (or any nn.Module containing
            BitLinear children).
        alpha: SmoothQuant migration strength ∈ [0, 1]. Default 0.5
            (literature). Per §3.2 ternary may want α<0.5; sweep
            via the eval harness.
        sites: site-preset name (str ∈ ``SMOOTHQUANT_SITE_PRESETS``)
            OR list of name-suffix patterns. Default ``"o"`` (only
            ``.attn.o_proj``).
        device: torch device for tracker tensors. Defaults to model's
            first parameter device.

    Example:
        >>> cal = SmoothQuantCalibrator(model, alpha=0.4, sites="igfo")
        >>> calibration = cal.calibrate(val_batches, n_batches=64)
        >>> cal.validate(calibration)["_aggregate"]["all_layers_pass"]
        True
        >>> cal.apply(model, calibration)  # in-place mutation
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        alpha: float = 0.5,
        sites: list[str] | str | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.model = model
        self.alpha = float(alpha)
        self.sites = _resolve_sites(sites)
        if device is None:
            device = next(model.parameters()).device
        self.device = torch.device(device)

    def _matched_modules(self):
        """Yield ``(name, module)`` pairs for every BitLinear matching
        ``self.sites``. ``self.sites is None`` matches every BitLinear.
        """
        for name, module in self.model.named_modules():
            if not _is_bitlinear(module):
                continue
            if self.sites is None or any(name.endswith(suf) for suf in self.sites):
                yield name, module

    def calibrate(self, batches, n_batches: int = 64) -> dict:
        """Run forward-pre-hook capture + weight scan; compute s_j per
        matched BitLinear.

        Args:
            batches: iterable of token-id tensors (or whatever the
                model's forward expects). Same protocol as
                ``intllm.eval.run_held_out_loss``.
            n_batches: cap on number of calibration batches consumed.

        Returns:
            dict mapping layer name → ``{'s', 'max_act', 'max_w', 'edge_cases'}``.
        """
        trackers: dict[str, dict] = {}
        for name, module in self._matched_modules():
            in_features = module.in_features
            trackers[name] = {
                "max_act": torch.zeros(in_features, device=self.device),
                "max_w": module.weight.detach().abs().max(dim=0)[0].to(self.device),
                "handle": None,
            }

        def _make_pre_hook(layer_name: str):
            def _hook(_mod, inputs):
                x = inputs[0]
                x_max = x.detach().abs().reshape(-1, x.shape[-1]).max(dim=0)[0]
                trackers[layer_name]["max_act"] = torch.maximum(
                    trackers[layer_name]["max_act"], x_max,
                )
                return None
            return _hook

        for name, module in self._matched_modules():
            trackers[name]["handle"] = module.register_forward_pre_hook(_make_pre_hook(name))

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for i, batch in enumerate(batches):
                    if i >= n_batches:
                        break
                    self.model(batch)
        finally:
            for name in trackers:
                trackers[name]["handle"].remove()
            if was_training:
                self.model.train()

        result: dict[str, dict] = {}
        for name, t in trackers.items():
            max_act_raw = t["max_act"]
            max_w_raw = t["max_w"]
            max_act = max_act_raw.clamp(min=1e-5)
            max_w = max_w_raw.clamp(min=1e-5)
            s = (max_act ** self.alpha) / (max_w ** (1.0 - self.alpha))
            s_clamped = s.clamp(min=1e-3, max=1e3)
            result[name] = {
                "s": s_clamped.cpu(),
                "max_act": max_act_raw.cpu(),
                "max_w": max_w_raw.cpu(),
                "edge_cases": {
                    "n_act_clamped": int((max_act_raw < 1e-5).sum().item()),
                    "n_w_clamped": int((max_w_raw < 1e-5).sum().item()),
                    "n_s_clamped_lo": int((s_clamped == 1e-3).sum().item()),
                    "n_s_clamped_hi": int((s_clamped == 1e3).sum().item()),
                },
            }
        return result

    def validate(self, calibration_dict: dict) -> dict:
        """Apply §4.6 mechanical gates to a calibration result. Returns
        a dict ``{layer_name: {gate_name: passed}}`` plus an aggregate.
        """
        gates: dict[str, dict] = {}
        for name, layer in calibration_dict.items():
            s = layer["s"]
            finite = bool(s.isfinite().all().item())
            range_valid = bool(((s >= 1e-3) & (s <= 1e3)).all().item())
            median_in_band = bool(0.1 <= s.median().item() <= 10.0)
            cond_num_ok = (
                bool((s.max() / s.min()).item() <= 1e6)
                if s.min().item() > 0
                else False
            )
            layer_gates = {
                "finite": finite,
                "range_valid": range_valid,
                "median_in_band": median_in_band,
                "condition_number_ok": cond_num_ok,
            }
            layer_gates["all_pass"] = all(layer_gates.values())
            gates[name] = layer_gates

        gates["_aggregate"] = {
            "all_layers_pass": all(
                g["all_pass"] for n, g in gates.items() if n != "_aggregate"
            ),
            "n_layers": len(calibration_dict),
        }
        return gates

    def validate_forward_equivalence(
        self,
        calibration_dict: dict,
        *,
        probe_batch,
        threshold_nat: float = 0.05,
        loss_fn=None,
    ) -> dict:
        """G5 gate per F.5.1.6 §3.3: pre-mutation forward equivalence.

        Catches Run-7-style cumulative multi-site saturation where every
        per-layer §4.6 gate (G1-G4) passes yet the calibration regresses
        ``val_loss`` catastrophically once applied — the case that nearly
        shipped silently in F.5.1.5 (148/148 layer gates PASS, +0.27 nat
        regression on sites=all × α=0.3). Per-layer recipe-sanity gates
        are NOT sufficient when SmoothQuant runs on many sites at once.

        Approach: deepcopy ``self.model``, compute baseline loss on
        ``probe_batch``, apply the calibration to the copy, recompute,
        compare. The original ``self.model`` is left untouched. If the
        absolute loss delta exceeds ``threshold_nat``, the calibration
        is flagged unsafe.

        Args:
            calibration_dict: output of :meth:`calibrate`.
            probe_batch: a single batch (token IDs or whatever the model's
                forward expects). One batch is enough — this is the cheap
                pre-apply check, not the full multi-batch val measurement.
            threshold_nat: max permissible absolute Δ-loss in nats.
                Default 0.05 per design §4.6 / findings §3.3.
            loss_fn: optional callable ``(model, batch) -> float``. If
                ``None``, uses HF causal-LM convention
                ``model(input_ids=batch, labels=batch).loss`` — same
                formula as :func:`intllm.eval.run_held_out_loss`.

        Returns:
            dict with keys ``loss_pre``, ``loss_post``, ``delta_nat``,
            ``threshold_nat``, ``passed``.
        """
        import copy

        if loss_fn is None:
            def loss_fn(model, batch):  # noqa: E306
                out = model(input_ids=batch, labels=batch)
                return float(out.loss.detach())

        probe_model = copy.deepcopy(self.model)
        was_training = probe_model.training
        probe_model.eval()
        try:
            with torch.no_grad():
                loss_pre = loss_fn(probe_model, probe_batch)
            self.apply(probe_model, calibration_dict, in_place=True)
            with torch.no_grad():
                loss_post = loss_fn(probe_model, probe_batch)
        finally:
            if was_training:
                probe_model.train()

        delta = loss_post - loss_pre
        return {
            "loss_pre": float(loss_pre),
            "loss_post": float(loss_post),
            "delta_nat": float(delta),
            "threshold_nat": float(threshold_nat),
            "passed": bool(abs(delta) < threshold_nat) and bool(
                math.isfinite(delta)
            ),
        }

    def apply(
        self,
        model: nn.Module,
        calibration_dict: dict,
        *,
        in_place: bool = True,
    ) -> nn.Module:
        """Apply §3.3 fusion identity per BitLinear site:
        ``γ ← γ/s`` AND ``W ← s·W``.

        Args:
            model: target model (typically same as ``self.model``, but
                supports passing a fresh copy for non-destructive usage).
            calibration_dict: output of ``calibrate()``.
            in_place: if False, deepcopy the model before mutation.

        Returns:
            The mutated model.
        """
        if not in_place:
            import copy
            model = copy.deepcopy(model)

        modules_by_name = dict(model.named_modules())
        for name, layer in calibration_dict.items():
            if name not in modules_by_name:
                raise KeyError(f"calibration layer {name!r} not found in target model")
            module = modules_by_name[name]
            if not _is_bitlinear(module):
                raise TypeError(f"target {name!r} is not a BitLinear")
            s = layer["s"].to(module.weight.device).to(module.weight.dtype)
            module.norm.weight.data.div_(s)
            module.weight.data.mul_(s.unsqueeze(0))
        return model


def save_smoothquant_maps(
    out_path,
    calibration_dict: dict,
    *,
    alpha: float,
    sites: list[str] | str | None,
    ckpt_path: str,
    n_calibration_sequences: int,
    calibration_seed: int,
    calibration_stream: str,
):
    """Side-car file writer per F.5.1 design §3.4 Option B schema v1.0.

    Atomic write via .tmp + os.replace.
    """
    import os
    from datetime import datetime, timezone
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_schema_version": _F51_SCHEMA_VERSION,
        "ckpt_path": str(ckpt_path),
        "alpha": float(alpha),
        "sites": sites,
        "n_calibration_sequences": int(n_calibration_sequences),
        "calibration_seed": int(calibration_seed),
        "calibration_stream": str(calibration_stream),
        "per_layer": calibration_dict,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, out_path)
    return out_path


def load_smoothquant_maps(path) -> dict:
    """Reverse of :func:`save_smoothquant_maps`. Validates schema version."""
    payload = torch.load(path, weights_only=False, map_location="cpu")
    schema = payload.get("_schema_version")
    if schema != _F51_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported smoothquant maps schema version {schema!r}; "
            f"expected {_F51_SCHEMA_VERSION!r}",
        )
    return payload


# ---------------------------------------------------------------------------
# §6. Sparse extension of FusedBitLinear (V32-prep F.10.2)
#
# Per docs/FJQ_PHASE_F_F10_2_BITLINEAR_DESIGN.md "Approach D" — subclass
# FusedBitLinear with a forward override that replicates the parent's
# RMSNorm + activation_quant + weight_quant + linear path, with an
# optional N:M sparse mask multiplied onto the quantized weight.
#
# Default `sparse_n=0` disables sparsity → forward() defers to parent's
# (fused, optimized) FusedBitLinear.forward → bit-exact dense path.
# `sparse_n=2, sparse_m=4` enables 2:4 structured sparsity for Sparse-BitNet
# recipe; trades the fused autograd kernel optimization for explicit STE
# clarity (5-15% slower training on sparse runs only; dense unaffected).
#
# Mask is computed FRESH PER FORWARD from the current ternary-quantized
# weight via the vendored Sparse-BitNet Triton kernel (see
# python/phase_d/intllm/sparse_kernel.py for upstream attribution).
# Mask is treated as a constant w.r.t. autograd: gradients flow to `w`
# only on positions where mask==1, exactly Sparse-BitNet recipe.
# ---------------------------------------------------------------------------


class SparseFusedBitLinear(nn.Module):
    """FusedBitLinear with optional N:M structured sparsity mask.

    Subclass-by-composition of upstream `FusedBitLinear` (we do NOT
    inherit because upstream's `__init__` signature locks in fused-
    autograd-Function as the forward kernel; we want the option to
    bypass it on the sparse path). This class wraps an inner
    FusedBitLinear and adds an optional sparse-mask multiplication
    after weight quantization.

    Args:
        in_features: input feature dim (passed to inner FusedBitLinear).
        out_features: output feature dim.
        bias: include bias term (default False, matches upstream).
        sparse_n: number of nonzeros per group of `sparse_m`. Default
            0 = disabled (dense path = bit-exact upstream behavior).
        sparse_m: group size for N:M sparsity (default 4 → with N=2
            this is the canonical 2:4 pattern accelerated by Ada
            Lovelace 4th-gen Tensor Cores).

    Example:
        >>> dense = SparseFusedBitLinear(256, 512)  # default → dense
        >>> sparse = SparseFusedBitLinear(256, 512, sparse_n=2, sparse_m=4)

    F.10 chain context: F.10.5 PoL smoke uses this class via the
    `--sparse-2-4` flag in `train_mini_ablation.py` (F.10.4 wiring).
    F.10.6 full Mini run produces the val_loss number that F.10.6 G1
    gate evaluates against Q5 baseline.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        sparse_n: int = 0,
        sparse_m: int = 4,
        mask_refresh_interval: int = 0,
    ) -> None:
        super().__init__()
        # Lazy import to avoid circular import + to defer triton dependency
        # to first instantiation. mmfreelm.ops.fusedbitnet imports triton
        # at module-load time, which is fine for training but problematic
        # for stat-tracker test fixtures that don't need it.
        # Match the sys.path injection pattern from intllm/model.py so this
        # class works without requiring intllm.model to be imported first.
        import sys
        from pathlib import Path
        _upstream = Path(__file__).resolve().parent.parent / "_upstream"
        if str(_upstream) not in sys.path:
            sys.path.insert(0, str(_upstream))
        from mmfreelm.ops.fusedbitnet import FusedBitLinear

        self._inner = FusedBitLinear(in_features, out_features, bias=bias)
        if sparse_n < 0 or sparse_n > sparse_m:
            raise ValueError(
                f"sparse_n must be in [0, sparse_m={sparse_m}]; got {sparse_n}"
            )
        if sparse_m <= 0:
            raise ValueError(f"sparse_m must be positive; got {sparse_m}")
        if mask_refresh_interval < 0:
            raise ValueError(
                f"mask_refresh_interval must be ≥0; got {mask_refresh_interval}"
            )
        self.sparse_n = sparse_n
        self.sparse_m = sparse_m
        # F.10.3 — refresh schedule.
        # 0 = recompute every forward (matches F.10.2 baseline; correct but
        #     pays Triton kernel launch per forward per layer).
        # N>0 = recompute every N forwards; reuse cached mask for the other
        #       N-1 forwards. Trades Triton launch cost for slightly stale
        #       mask between refreshes; Sparse-BitNet recipe says masks
        #       should evolve "dynamically" but doesn't quote a frequency.
        #       Empirically (F.10.5 PoL) we expect 100-1000 to be safe.
        self.mask_refresh_interval = mask_refresh_interval
        # Step counter as buffer (saved with state_dict, but not learnable).
        self.register_buffer(
            "_mask_step_counter",
            torch.zeros(1, dtype=torch.long),
            persistent=False,
        )
        # Cached mask buffer; lazy-allocated on first sparse forward to avoid
        # holding GPU memory before sparse_n>0 is exercised.
        self._cached_mask: torch.Tensor | None = None

    @property
    def weight(self) -> torch.Tensor:
        """Pass-through to inner FusedBitLinear's weight parameter.

        Required for `_is_bitlinear` matching + stat-tracker hooks +
        optimizer state (which sees `self.named_parameters()`).
        """
        return self._inner.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self._inner.bias

    @property
    def norm(self) -> nn.Module:
        """Pass-through to inner FusedBitLinear's RMSNorm."""
        return self._inner.norm

    @property
    def in_features(self) -> int:
        return self._inner.in_features

    @property
    def out_features(self) -> int:
        return self._inner.out_features

    def refresh_mask(self) -> torch.Tensor:
        """Compute + cache mask from current quantized weight. Returns the
        new mask. F.10.3: explicit method allows external schedulers
        (e.g. training loop callback) to control refresh timing.

        Caller must hold the relevant context (no_grad if not training, etc.).
        Method itself is idempotent + side-effect-only-on-cache.
        """
        from mmfreelm.ops.fusedbitnet import weight_quant
        from intllm.sparse_kernel import mask_creator_triton_optimized

        if self.sparse_n == 0:
            return torch.ones_like(self._inner.weight)

        with torch.no_grad():
            w_q = weight_quant(self._inner.weight)
            mask = mask_creator_triton_optimized(
                w_q.detach(), N=self.sparse_n, M=self.sparse_m
            )
        self._cached_mask = mask
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparse_n == 0:
            # Dense path — UNCHANGED, bit-exact to upstream FusedBitLinear.
            return self._inner(x)

        # Sparse path: replicate base BitLinear.forward semantics + mask.
        # Reference: mmfreelm/ops/fusedbitnet.py:560-585 (BitLinear.forward).
        # Imports here (not at module top) keep the dense path lighter
        # for users who never enable sparsity.
        from mmfreelm.ops.fusedbitnet import activation_quant, weight_quant
        from intllm.sparse_kernel import mask_creator_triton_optimized

        x_norm = self._inner.norm(x)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w = self._inner.weight
        w_quant = w + (weight_quant(w) - w).detach()

        # F.10.3 mask refresh schedule:
        #   interval=0 → recompute every forward (F.10.2 baseline behavior)
        #   interval>0 → recompute every Nth forward; cache reused otherwise
        # Cached mask is stale-by-design between refreshes; Sparse-BitNet
        # recipe says masks should evolve "dynamically" but tolerates
        # short-window staleness empirically.
        step = int(self._mask_step_counter.item())
        if (
            self.mask_refresh_interval == 0
            or self._cached_mask is None
            or step % self.mask_refresh_interval == 0
        ):
            with torch.no_grad():
                mask = mask_creator_triton_optimized(
                    w_quant.detach(),
                    N=self.sparse_n,
                    M=self.sparse_m,
                )
            self._cached_mask = mask
        else:
            mask = self._cached_mask
        # Increment counter AFTER mask decision so step=0 always refreshes.
        self._mask_step_counter += 1

        w_quant_sparse = w_quant * mask

        return torch.nn.functional.linear(x_quant, w_quant_sparse, self._inner.bias)
