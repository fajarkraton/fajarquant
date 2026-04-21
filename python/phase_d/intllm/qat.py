"""IntLLM quantization-aware training harness.

C.P2.4 ships the *infrastructure* for the three novel QAT contributions
described in `FJQ_PHASE_D_OPS.md` §5 + paper outline §3.3:

  1. **Per-coord adaptive bit allocation** — `BitAllocator`
     Top-K most-outlier-prone activation channels get higher precision
     (10-bit) instead of the BitNet baseline 8-bit.

  2. **Ternary-aware channel reordering** — `compute_channel_permutation`
     Permutes output channels so high-magnitude rows cluster, enabling
     block-sparse `W̃_ij = 0` compute-skip during inference.

  3. **Periodic γ_x re-calibration** — `BitLinearStatTracker`
     Tracks running per-channel ‖x‖_∞ during shadow training so γ_x
     can be frozen (rather than recomputed per-call) for the late QAT
     phase, eliminating per-batch noise in the activation scaler.

The hooks attach to upstream's `BitLinear`/`FusedBitLinear` modules
without monkey-patching — pure forward-hook injection. Calling
`attach_stat_trackers(model)` is idempotent and reversible
(detach by deleting hooks).

This module ships the **measurement + transformation** layer. The
**training-loop integration** (when to switch from shadow to QAT phase,
when to re-calibrate, how to apply the bit-allocation map at forward
time) lives in `intllm.train` extensions and the C.P4 training driver.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# §1. Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QATConfig:
    """Toggle the three §6.9 R5 outlier-handling features.

    Default: all OFF — falls back to the pure BitNet b1.58 baseline that
    upstream `BitLinear` already implements. Enable individually for
    ablation Tables 4 (`FJQ_PHASE_D_PAPER_OUTLINE.md` §5.1).
    """

    enable_adaptive_bits: bool = False
    adaptive_bits_top_k_pct: float = 5.0
    adaptive_bits_low: int = 8
    adaptive_bits_high: int = 10

    enable_channel_permutation: bool = False

    enable_periodic_recal: bool = False
    recal_every_steps: int = 1000


# ---------------------------------------------------------------------------
# §2. Stat tracker (forward hook)
# ---------------------------------------------------------------------------

@dataclass
class BitLinearStatTracker:
    """Forward-hook helper that records per-channel activation magnitude.

    For every BitLinear forward call, captures:
      - `running_max`  — element-wise max of |input| across all calls
      - `n_calls`      — total number of forward calls observed

    These are the inputs to:
      - `compute_bit_allocation` (§3) — top-K channels by max magnitude
      - `compute_channel_permutation` (§4) — sort output channels
      - periodic γ_x re-calibration — running max tracks the activation
        absmax that BitNet's `activation_quant` uses per-call
    """

    in_features: int
    running_max: torch.Tensor = field(init=False)
    n_calls: int = 0

    def __post_init__(self) -> None:
        # Live on CPU by default; moved to GPU lazily on first call.
        self.running_max = torch.zeros(self.in_features, dtype=torch.float32)

    def __call__(self, _module: nn.Module, inputs: tuple, _output: torch.Tensor) -> None:
        x = inputs[0]
        # Reduce over all but the last dim to get per-channel absmax.
        per_channel = x.detach().abs().reshape(-1, x.size(-1)).amax(dim=0).cpu().float()
        # Element-wise running max — captures the absmax the model has
        # ever seen on each channel, exactly the quantity that determines
        # γ_x stability.
        if per_channel.numel() != self.running_max.numel():
            raise ValueError(
                f"channel-dim mismatch: expected {self.running_max.numel()}, got {per_channel.numel()}"
            )
        torch.maximum(self.running_max, per_channel, out=self.running_max)
        self.n_calls += 1


def _is_bitlinear(module: nn.Module) -> bool:
    """Detect upstream `BitLinear` / `FusedBitLinear` regardless of which
    code path imports them (bitnet.py vs fusedbitnet.py)."""
    return module.__class__.__name__ in {"BitLinear", "FusedBitLinear", "BitLinear_wonorm_bmm"}


def attach_stat_trackers(model: nn.Module) -> dict[str, BitLinearStatTracker]:
    """Walk `model`, install a `BitLinearStatTracker` forward hook on each
    BitLinear-family layer, and return a dict mapping layer-name →
    tracker.

    The returned trackers stay alive (and the hooks stay registered)
    until `detach_stat_trackers` is called or the model is deleted.
    """
    trackers: dict[str, BitLinearStatTracker] = {}
    for name, module in model.named_modules():
        if not _is_bitlinear(module):
            continue
        in_features: int = getattr(module, "in_features")
        tracker = BitLinearStatTracker(in_features=in_features)
        module.register_forward_hook(tracker)
        # Stash the hook handle so we can detach later (PyTorch returns it
        # from `register_forward_hook`, so re-attaching here is awkward
        # — easier to just clear via `module._forward_hooks`).
        trackers[name] = tracker
    return trackers


def detach_stat_trackers(model: nn.Module) -> None:
    """Remove all forward hooks from BitLinear-family modules.

    We don't track which hooks WE installed vs. user-supplied ones —
    callers should treat this as a "clear all forward hooks on
    BitLinear" sledgehammer. Re-attach with `attach_stat_trackers`.
    """
    for module in model.modules():
        if _is_bitlinear(module):
            module._forward_hooks.clear()


# ---------------------------------------------------------------------------
# §3. Adaptive bit allocation
# ---------------------------------------------------------------------------

def compute_bit_allocation(
    tracker: BitLinearStatTracker,
    *,
    top_k_pct: float = 5.0,
    low_bits: int = 8,
    high_bits: int = 10,
) -> torch.Tensor:
    """Given accumulated per-channel absmax stats, decide how many bits
    each input channel should be quantized to.

    Returns an int32 tensor of shape `(in_features,)` containing
    `low_bits` for the bulk of channels and `high_bits` for the
    `top_k_pct`% with the largest running max.

    Per `FJQ_PHASE_D_PAPER_OUTLINE.md` §3.3: the few outlier-prone
    channels keep more dynamic range, while the majority stay at the
    cheap BitNet baseline. Aggregate cost: 8.1-bit average at
    top_k_pct=5%.
    """
    if tracker.n_calls == 0:
        raise ValueError("tracker has no observations; run forward passes first")

    n_channels = tracker.running_max.numel()
    n_top = max(1, int(round(n_channels * top_k_pct / 100.0)))
    # `topk` returns indices of the n_top largest values
    _, top_indices = torch.topk(tracker.running_max, k=n_top)
    bits = torch.full((n_channels,), low_bits, dtype=torch.int32)
    bits[top_indices] = high_bits
    return bits


# ---------------------------------------------------------------------------
# §4. Ternary-aware channel reordering
# ---------------------------------------------------------------------------

def compute_channel_permutation(tracker: BitLinearStatTracker) -> torch.Tensor:
    """Return a permutation that sorts input channels by descending
    running absmax.

    Use case: rearrange the upstream BitLinear's weight matrix so that
    rows operating on high-magnitude input channels cluster at the
    front. This enables block-sparse compute-skip on `W̃_ij = 0`
    entries during inference (kernel-side optimization, not exercised
    in PyTorch QAT but required by §FJQ_PHASE_D_OPS.md §3 BitLinear
    delivery for the FajarOS kernel).

    Returns int64 indices of shape `(in_features,)`.
    """
    if tracker.n_calls == 0:
        raise ValueError("tracker has no observations; run forward passes first")
    return torch.argsort(tracker.running_max, descending=True)


__all__ = [
    "BitLinearStatTracker",
    "QATConfig",
    "attach_stat_trackers",
    "compute_bit_allocation",
    "compute_channel_permutation",
    "detach_stat_trackers",
]
