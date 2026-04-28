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
      - `n_calls`      — total number of forward calls *recorded*
      - `n_skipped`    — total number of forward calls dropped because
                        `current_step < start_recording_at_step`

    These are the inputs to:
      - `compute_bit_allocation` (§3) — top-K channels by max magnitude
      - `compute_channel_permutation` (§4) — sort output channels
      - periodic γ_x re-calibration — running max tracks the activation
        absmax that BitNet's `activation_quant` uses per-call

    F.5.3 skip-warmup calibration: when `start_recording_at_step > 0`,
    the tracker drops every forward call until the externally-managed
    `current_step` field reaches that threshold. Use this to defer
    accumulator initialization until past the LR-scheduler warmup phase
    so chaotic early-training peaks do not poison the all-time-max
    statistic — see `paper/intllm/intllm.tex` §7.2 cause-1 + the F.5.0
    cross-comparison evidence in
    `paper/intllm/ablations/running_max_train_vs_steady*.json`.

    The training loop is responsible for advancing `current_step`
    (typically once per optimizer step). Use `advance_trackers(...)`
    for the common "advance all attached trackers by 1" path, or set
    `tracker.current_step = global_step` directly when tighter control
    is needed.
    """

    in_features: int
    start_recording_at_step: int = 0
    running_max: torch.Tensor = field(init=False)
    n_calls: int = 0
    n_skipped: int = 0
    current_step: int = 0

    def __post_init__(self) -> None:
        # Live on CPU by default; moved to GPU lazily on first call.
        self.running_max = torch.zeros(self.in_features, dtype=torch.float32)
        if self.start_recording_at_step < 0:
            raise ValueError(
                f"start_recording_at_step must be ≥ 0, got {self.start_recording_at_step}"
            )

    def __call__(self, _module: nn.Module, inputs: tuple, _output: torch.Tensor) -> None:
        # F.5.3: drop every call before the warmup boundary. We still
        # increment n_skipped so the audit trail captures the dropped
        # count; the running_max stays untouched.
        if self.current_step < self.start_recording_at_step:
            self.n_skipped += 1
            return
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


def is_bitlinear(module: nn.Module) -> bool:
    """Detect upstream `BitLinear` / `FusedBitLinear` regardless of which
    code path imports them (bitnet.py vs fusedbitnet.py).

    Matched by class name rather than `isinstance` so a model that wraps
    or subclasses upstream BitLinear without preserving the exact import
    path still gets recognized. This is the canonical surface for any
    code that needs to walk a model's BitLinear-family layers.
    """
    return module.__class__.__name__ in {"BitLinear", "FusedBitLinear", "BitLinear_wonorm_bmm"}


# Backwards-compat alias — the leading-underscore form was internal to
# this module before E2.4.C.2; downstream code that imported the
# private form keeps working until the next refactor sweep.
_is_bitlinear = is_bitlinear


def attach_stat_trackers(
    model: nn.Module,
    *,
    start_recording_at_step: int = 0,
) -> dict[str, BitLinearStatTracker]:
    """Walk `model`, install a `BitLinearStatTracker` forward hook on each
    BitLinear-family layer, and return a dict mapping layer-name →
    tracker.

    All installed trackers share the same `start_recording_at_step`
    threshold (F.5.3 skip-warmup calibration). For per-site warmup,
    construct trackers manually and call
    `module.register_forward_hook(tracker)` directly.

    The returned trackers stay alive (and the hooks stay registered)
    until `detach_stat_trackers` is called or the model is deleted.
    """
    trackers: dict[str, BitLinearStatTracker] = {}
    for name, module in model.named_modules():
        if not _is_bitlinear(module):
            continue
        in_features: int = getattr(module, "in_features")
        tracker = BitLinearStatTracker(
            in_features=in_features,
            start_recording_at_step=start_recording_at_step,
        )
        module.register_forward_hook(tracker)
        # Stash the hook handle so we can detach later (PyTorch returns it
        # from `register_forward_hook`, so re-attaching here is awkward
        # — easier to just clear via `module._forward_hooks`).
        trackers[name] = tracker
    return trackers


def advance_trackers(
    trackers: dict[str, BitLinearStatTracker],
    *,
    global_step: int | None = None,
) -> None:
    """Advance every tracker's `current_step` counter.

    Two modes:
      - `global_step=None` (default) — increment each tracker's
        `current_step` by 1. Use after each optimizer step in a normal
        training loop.
      - `global_step=k` — set every tracker's `current_step` to `k`. Use
        when resuming from a checkpoint so trackers re-sync to the true
        step counter regardless of the in-memory drift.

    F.5.3: the warmup-boundary check `current_step >= start_recording_at_step`
    runs at every forward call; advancing the counter here is what
    eventually transitions the tracker out of the skip phase.
    """
    if global_step is None:
        for t in trackers.values():
            t.current_step += 1
    else:
        if global_step < 0:
            raise ValueError(f"global_step must be ≥ 0, got {global_step}")
        for t in trackers.values():
            t.current_step = global_step


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


# ---------------------------------------------------------------------------
# §5. Calibration-map serialization (E2.4.A.2)
# ---------------------------------------------------------------------------

def save_calibration_maps(
    trackers: dict[str, BitLinearStatTracker],
    out_path: "str | __import__('pathlib').Path",
    *,
    top_k_pct: float = 5.0,
    low_bits: int = 8,
    high_bits: int = 10,
    extra_meta: dict | None = None,
) -> dict:
    """Compute per-BitLinear bit allocation + channel permutation maps from
    a dict of attached `BitLinearStatTracker`s, and atomically serialize
    them via `torch.save` to `out_path`.

    The resulting `.pt` file is the deliverable of E2.4 Option-A
    calibration — one entry per BitLinear, each containing:

      - `running_max`         (float32, shape `(in_features,)`)
      - `n_calls`             (int)
      - `bits`                (int32, shape `(in_features,)`) — output of
                              `compute_bit_allocation` (top-K → high_bits,
                              rest → low_bits)
      - `permutation`         (int64, shape `(in_features,)`) — output of
                              `compute_channel_permutation`

    The dict structure is `{layer_name: {**maps_above}}` plus a
    top-level `"_meta"` key containing `top_k_pct`, `low_bits`,
    `high_bits`, `n_layers`, plus any keys from `extra_meta`. The map
    file is consumed by E2.4.C.2 quantization-error metric and
    (eventually) by an `IntLLMBitLinear` wrapper if Option B is later
    adopted.

    Atomic write via `os.replace` so a SIGKILL mid-save cannot leave a
    partial map on disk (matches §6.11 R1 atomicity convention).
    """
    import os
    from pathlib import Path

    if not trackers:
        raise ValueError("save_calibration_maps: trackers dict is empty")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {}
    for layer_name, tracker in trackers.items():
        if tracker.n_calls == 0:
            raise ValueError(
                f"save_calibration_maps: tracker '{layer_name}' has 0 observations; "
                f"run forward passes before saving"
            )
        bits = compute_bit_allocation(
            tracker, top_k_pct=top_k_pct, low_bits=low_bits, high_bits=high_bits,
        )
        perm = compute_channel_permutation(tracker)
        payload[layer_name] = {
            "running_max": tracker.running_max.detach().clone(),
            "n_calls": int(tracker.n_calls),
            "n_skipped": int(tracker.n_skipped),
            "start_recording_at_step": int(tracker.start_recording_at_step),
            "bits": bits,
            "permutation": perm,
            "in_features": int(tracker.in_features),
        }

    # F.5.3 audit: warmup-skip is a model-wide policy. We record the
    # min/max across trackers so the consumer can sanity-check that
    # `attach_stat_trackers` was called with a single warmup setting.
    warmup_steps = [t.start_recording_at_step for t in trackers.values()]
    skipped_counts = [t.n_skipped for t in trackers.values()]
    meta: dict = {
        "top_k_pct": float(top_k_pct),
        "low_bits": int(low_bits),
        "high_bits": int(high_bits),
        "n_layers": len(trackers),
        "start_recording_at_step_min": min(warmup_steps),
        "start_recording_at_step_max": max(warmup_steps),
        "n_skipped_min": min(skipped_counts),
        "n_skipped_max": max(skipped_counts),
        "_schema_version": "1.1",
    }
    if extra_meta:
        meta.update(extra_meta)
    payload["_meta"] = meta

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, out_path)
    return payload


__all__ = [
    "BitLinearStatTracker",
    "QATConfig",
    "advance_trackers",
    "attach_stat_trackers",
    "compute_bit_allocation",
    "compute_channel_permutation",
    "detach_stat_trackers",
    "is_bitlinear",
    "save_calibration_maps",
]
