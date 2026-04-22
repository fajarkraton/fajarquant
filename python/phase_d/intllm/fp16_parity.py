"""fp16-vs-ternary parity measurement (Phase 3.5 IntLLM differentiator).

Per FJQ_PHASE_D_PRODUCTION_PLAN.md §3.5, this module measures the
per-layer relative error between two forward passes through the SAME
trained checkpoint:

  - Ternary forward: BitLinear applies activation_quant + weight_quant
    via STE — this is the "production" forward path that the kernel
    will replicate bit-exactly.
  - fp16 reference: BitLinear short-circuits the quantization step,
    using `F.linear(self.norm(x), self.weight)` directly.

The fp16 reference asks "what would the model output if its weights
+ activations were never quantized?" For a model trained with quant-
in-loop (STE), the model has learned to be robust to ternary noise,
so divergence from the fp16 oracle should be bounded — but BitNet
2B4T did not ship a parity gate at all, so this is a publishable
differentiator regardless of where the bound lands.

Usage:

    from intllm.fp16_parity import measure_parity, fp16_reference_mode

    rel_errors = measure_parity(model, input_ids, max_layers=None)
    # rel_errors: dict[layer_name → float]

    # Or use the context manager directly to temporarily disable quant:
    with fp16_reference_mode():
        out_fp16 = model(input_ids).logits
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F


# Both upstream BitLinear classes inherit from nn.Linear and expose:
#   self.norm   (RMSNorm)
#   self.weight (full-precision shadow weight)
#   self.bias   (None for bitnet.BitLinear; may be set for fusedbitnet)
# We monkey-patch their forward to skip quantization. Source:
# python/phase_d/_upstream/mmfreelm/ops/{bitnet,fusedbitnet}.py
def _import_bitlinears() -> tuple[type, type, type]:
    """Locate the upstream BitLinear classes. Imported lazily so this
    module is safe to import in environments without the full upstream."""
    from mmfreelm.ops.bitnet import BitLinear as BitLinearA
    from mmfreelm.ops.fusedbitnet import BitLinear as BitLinearB
    from mmfreelm.ops.fusedbitnet import FusedBitLinear
    return BitLinearA, BitLinearB, FusedBitLinear


def _fp16_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Replacement forward that skips quantization. Matches the upstream
    BitLinear forward up to (and including) RMSNorm, then takes a plain
    F.linear with the unquantized weight."""
    x_norm = self.norm(x)
    bias = getattr(self, "bias", None)
    return F.linear(x_norm, self.weight, bias)


def _fp16_fused_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Replacement for FusedBitLinear which uses a fused triton kernel
    upstream — we bypass it entirely here for the fp16 reference."""
    x_norm = self.norm(x)
    bias = getattr(self, "bias", None)
    return F.linear(x_norm, self.weight, bias)


@contextlib.contextmanager
def fp16_reference_mode() -> Iterator[None]:
    """Temporarily monkey-patch BitLinear.forward across all upstream
    classes to skip quantization. Restores originals on exit even if
    the wrapped block raises."""
    BitLinearA, BitLinearB, FusedBitLinear = _import_bitlinears()
    orig_a = BitLinearA.forward
    orig_b = BitLinearB.forward
    orig_fused = FusedBitLinear.forward
    BitLinearA.forward = _fp16_forward
    BitLinearB.forward = _fp16_forward
    FusedBitLinear.forward = _fp16_fused_forward
    try:
        yield
    finally:
        BitLinearA.forward = orig_a
        BitLinearB.forward = orig_b
        FusedBitLinear.forward = orig_fused


@dataclass
class LayerParity:
    """Per-layer parity result. rel_error is computed L2-normalized:
        ||y_ternary - y_fp16||_2 / max(||y_fp16||_2, eps)
    so it is a scale-free fraction (0.0 = identical, 1.0 = orthogonal-magnitude)."""
    name: str
    abs_error_l2: float
    rel_error_l2: float
    abs_error_max: float
    rel_error_max: float
    fp16_norm_l2: float
    n_elements: int


def _capture_outputs(model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], list]:
    """Register forward hooks on every BitLinear-like module to capture
    its output tensor. Returns (outputs_dict, hook_handles)."""
    BitLinearA, BitLinearB, FusedBitLinear = _import_bitlinears()
    bitlinear_types = (BitLinearA, BitLinearB, FusedBitLinear)

    captured: dict[str, torch.Tensor] = {}
    handles: list = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            captured[name] = output.detach().to(torch.float32).cpu()
        return hook

    for name, module in model.named_modules():
        if isinstance(module, bitlinear_types):
            handles.append(module.register_forward_hook(make_hook(name)))

    return captured, handles


def measure_parity(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    max_layers: int | None = None,
    eps: float = 1e-12,
) -> dict[str, LayerParity]:
    """Run two forward passes through `model` on `input_ids` (one with
    ternary quant, one with the fp16 reference) and return per-layer
    relative errors.

    Args:
        model: A loaded HGRNBitForCausalLM (or any nn.Module containing
            BitLinear submodules).
        input_ids: Input token IDs, shape [batch, seq_len].
        max_layers: If set, truncate the result to the first N hooks
            captured (useful for debugging deep models cheaply).
        eps: Floor for the denominator of the relative error.

    Returns:
        dict mapping layer name to LayerParity. Iteration order matches
        depth (PyTorch's named_modules walk).
    """
    model.eval()

    # Pass 1: ternary forward (the production path)
    captured_ternary, handles = _capture_outputs(model)
    with torch.no_grad():
        _ = model(input_ids)
    for h in handles:
        h.remove()

    # Pass 2: fp16 reference forward
    captured_fp16, handles = _capture_outputs(model)
    with torch.no_grad(), fp16_reference_mode():
        _ = model(input_ids)
    for h in handles:
        h.remove()

    # Sanity: the same hook names should be captured in both passes.
    if set(captured_ternary.keys()) != set(captured_fp16.keys()):
        missing = set(captured_ternary.keys()) ^ set(captured_fp16.keys())
        raise RuntimeError(
            f"hook name mismatch between ternary and fp16 passes: {missing}"
        )

    results: dict[str, LayerParity] = {}
    for name in captured_ternary:
        y_t = captured_ternary[name]
        y_f = captured_fp16[name]
        if y_t.shape != y_f.shape:
            raise RuntimeError(
                f"shape mismatch at {name}: ternary {tuple(y_t.shape)} "
                f"vs fp16 {tuple(y_f.shape)}"
            )
        diff = y_t - y_f
        abs_l2 = float(diff.norm())
        fp16_l2 = float(y_f.norm())
        rel_l2 = abs_l2 / max(fp16_l2, eps)
        abs_max = float(diff.abs().max())
        # Element-wise rel error, taking the max over elements with
        # |y_fp16| > eps to avoid dividing by ~0.
        denom = y_f.abs().clamp(min=eps)
        rel_elem_max = float((diff.abs() / denom).max())
        results[name] = LayerParity(
            name=name,
            abs_error_l2=abs_l2,
            rel_error_l2=rel_l2,
            abs_error_max=abs_max,
            rel_error_max=rel_elem_max,
            fp16_norm_l2=fp16_l2,
            n_elements=y_t.numel(),
        )
        if max_layers is not None and len(results) >= max_layers:
            break

    return results


def summary_stats(results: dict[str, LayerParity]) -> dict[str, float]:
    """Aggregate per-layer rel_error_l2 into a single-row summary
    suitable for paper Table 3 / verify_intllm_tables.py."""
    if not results:
        return {"n_layers": 0}
    rels = [r.rel_error_l2 for r in results.values()]
    return {
        "n_layers": len(rels),
        "rel_error_l2_mean": sum(rels) / len(rels),
        "rel_error_l2_max": max(rels),
        "rel_error_l2_min": min(rels),
        "rel_error_l2_p50": sorted(rels)[len(rels) // 2],
        "rel_error_l2_p95": sorted(rels)[max(0, int(0.95 * len(rels)) - 1)],
    }
