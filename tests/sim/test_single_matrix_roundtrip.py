"""P1.5 — Single-matrix quant→dequant round-trip regression.

Ports the V28.2 Day 1 Gate (fajaros-x86/scripts/export_gemma3_v8.py
`validate_roundtrip`) to use kernel_sim for the dequant path. The
kernel-integer dequant must reach the same error envelope (<5% max
abs error) as the numpy float dequant that originally passed V28.2's
gate at 2.40%.

Two scenarios covered:
1. **Synthetic matrix** (always runs) — 6912×1152 Gaussian with
   std=0.031 matching V28.2 observed distribution of
   `model.layers.0.mlp.gate_proj.weight`.
2. **Real HF weight** (conditional, skipped without torch +
   safetensors + local Gemma 3 checkpoint) — mirrors V28.2 directly.

Additionally verifies: kernel_sim integer dequant == numpy float
dequant at 0-ULP (after ×1e6 integer cast), confirming the Python
port's dequantization path is bit-exact with the kernel's.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

import numpy as np
import pytest

from kernel_sim import dequant_groupwise_v8_x1M, pack_nibbles, V8_GROUP_SIZE


# ── Quantizer (mirror of export_gemma3_v8.groupwise_quantize_4bit) ────

SCALE_FIXED_POINT = 1_000_000


def _groupwise_quantize_4bit(flat: np.ndarray, group_size: int = V8_GROUP_SIZE):
    """Asymmetric per-group 4-bit quantization matching
    fajaros-x86/scripts/export_gemma3_v8.py. Returns (indices, scales_int, zeros)."""
    assert flat.dtype == np.float32
    n = len(flat)
    n_groups = (n + group_size - 1) // group_size
    scales_int = np.empty(n_groups, dtype=np.int32)
    zeros = np.empty(n_groups, dtype=np.uint8)
    indices = np.empty(n, dtype=np.uint8)
    for g in range(n_groups):
        s = g * group_size
        e = min(s + group_size, n)
        grp = flat[s:e]
        mn, mx = float(grp.min()), float(grp.max())
        rng = mx - mn
        if rng < 1e-9:
            scale_real = 1.0
            zero = max(0, min(15, int(round(-mn))))
            q = np.full(e - s, zero, dtype=np.uint8)
        else:
            scale_real = rng / 15.0
            zero = max(0, min(15, int(round(-mn / scale_real))))
            q = np.clip(
                np.round(grp / scale_real + zero).astype(np.int32),
                0, 15,
            ).astype(np.uint8)
        indices[s:e] = q
        scales_int[g] = int(round(scale_real * SCALE_FIXED_POINT))
        zeros[g] = zero
    return indices, scales_int, zeros


def _dequant_numpy(indices, scales_int, zeros, group_size=V8_GROUP_SIZE):
    """Numpy float reconstruction (original V28.2 path for comparison)."""
    n = len(indices)
    out = np.empty(n, dtype=np.float32)
    n_groups = len(scales_int)
    for g in range(n_groups):
        s = g * group_size
        e = min(s + group_size, n)
        scale = float(scales_int[g]) / SCALE_FIXED_POINT
        zero = int(zeros[g])
        out[s:e] = (indices[s:e].astype(np.float32) - zero) * scale
    return out


# ── Core gate: synthetic-matrix round-trip ────────────────────────────

class TestSyntheticRoundTrip:
    """Deterministic synthetic matrix reproducing V28.2 observed
    statistics. Runs in ~2s without external dependencies."""

    def test_v28_2_gate_synthetic(self):
        rng = np.random.default_rng(20260416)
        # V28.2 observed: shape=(6912,1152), mean≈0, std≈0.031
        # Smaller dims keep the test fast (~1s) while still spanning
        # hundreds of groups.
        shape = (1024, 128)
        w = rng.normal(0, 0.031, shape).astype(np.float32)
        flat = w.flatten()
        n = flat.size
        assert n % 2 == 0, "total must be even for 4-bit packing"

        # Quantize (numpy-based — same as V28.2 script)
        indices, scales_int, zeros = _groupwise_quantize_4bit(flat)
        assert len(scales_int) == (n + V8_GROUP_SIZE - 1) // V8_GROUP_SIZE

        # Reconstruct two ways: numpy float + kernel_sim integer
        recon_np = _dequant_numpy(indices, scales_int, zeros)

        packed = pack_nibbles([int(x) for x in indices])
        recon_x1M = dequant_groupwise_v8_x1M(
            packed, list(scales_int), bytes(zeros), n
        )
        recon_sim = np.array(recon_x1M, dtype=np.float64) / SCALE_FIXED_POINT

        # Parity: numpy and kernel_sim must agree within tiny rounding
        # (numpy does float mul; kernel_sim does int mul then /1e6
        # float cast — difference only in last 1-2 float bits)
        parity_err = np.abs(recon_np.astype(np.float64) - recon_sim)
        assert parity_err.max() < 1e-6, (
            f"numpy vs kernel_sim parity broken: "
            f"max diff {parity_err.max()}"
        )

        # V28.2 gate: max abs error < 5% of weight range
        err = recon_sim - flat.astype(np.float64)
        max_abs = float(np.abs(err).max())
        mae = float(np.abs(err).mean())
        rng_w = float(flat.max() - flat.min())
        gate_pct = max_abs / rng_w * 100.0
        print(
            f"\n  Synthetic {shape}: "
            f"max_abs={max_abs:.6f} ({gate_pct:.2f}% of range), "
            f"MAE={mae:.6f}"
        )
        # Synthetic gaussian typically gives ~3-4% (single-codebook
        # Lloyd-Max 4-bit quant ceiling on normal data). Accept ≤5%
        # matching the V28.2 gate.
        assert gate_pct < 5.0, (
            f"Synthetic round-trip exceeded 5% gate: {gate_pct:.2f}%"
        )

    def test_constant_group_edge_case(self):
        """Group with range<1e-9 takes the constant-group branch.
        Must round-trip exactly (since all nibbles are the zero)."""
        flat = np.zeros(V8_GROUP_SIZE * 2, dtype=np.float32)
        flat[0] = 1e-10  # tiny perturbation < threshold
        flat[V8_GROUP_SIZE] = 1e-10
        indices, scales_int, zeros = _groupwise_quantize_4bit(flat)

        packed = pack_nibbles([int(x) for x in indices])
        recon_x1M = dequant_groupwise_v8_x1M(
            packed, list(scales_int), bytes(zeros), len(flat)
        )
        recon = np.array(recon_x1M) / SCALE_FIXED_POINT
        # Constant-group branch: zero=round(-mn)=0, scale=1, all
        # indices==zero → (zero-zero)*1 = 0. So reconstructed values
        # should all be 0.
        assert np.all(recon == 0), (
            f"Constant-group branch reconstructed non-zero: {recon[recon != 0]}"
        )

    def test_kernel_sim_vs_numpy_zero_ulp_integer(self):
        """Before the float cast, the integer ×1e6 values must match
        numpy's integer intermediate byte-for-byte. Confirms kernel_sim
        and numpy agree on every per-element (q - zero) * scale."""
        rng = np.random.default_rng(42)
        flat = rng.normal(0, 0.05, V8_GROUP_SIZE * 4).astype(np.float32)
        indices, scales_int, zeros = _groupwise_quantize_4bit(flat)
        n = len(flat)
        n_groups = len(scales_int)

        # Numpy integer intermediate: (q - zero) * scale (no /1e6 yet)
        expected = np.zeros(n, dtype=np.int64)
        for g in range(n_groups):
            s = g * V8_GROUP_SIZE
            e = min(s + V8_GROUP_SIZE, n)
            q = indices[s:e].astype(np.int64)
            z = np.int64(zeros[g])
            sc = np.int64(scales_int[g])
            expected[s:e] = (q - z) * sc

        packed = pack_nibbles([int(x) for x in indices])
        got = np.array(
            dequant_groupwise_v8_x1M(
                packed, list(scales_int), bytes(zeros), n
            ),
            dtype=np.int64,
        )
        assert np.array_equal(got, expected), (
            f"Integer-stage dequant diverged from numpy at "
            f"{np.where(got != expected)[0][:10]}"
        )


# ── Real HF weight case (conditional) ─────────────────────────────────

def _resolve_safetensors_path():
    candidates = [
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--google--gemma-3-1b-pt"
        ),
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--unsloth--gemma-3-1b-it"
        ),
        os.path.expanduser(
            "~/Documents/fajaros-x86/models/gemma-3-1b-pt"
        ),
    ]
    for root in candidates:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if f.endswith(".safetensors"):
                    return os.path.join(dirpath, f)
    return None


@pytest.mark.skipif(
    _resolve_safetensors_path() is None,
    reason="Gemma 3 1B safetensors not found locally — real weight test skipped",
)
class TestRealHFRoundTrip:
    """Reproduces V28.2 gate using the exact matrix from
    model.layers.0.mlp.gate_proj.weight. Requires HF torch +
    safetensors + local Gemma 3 checkpoint."""

    def test_v28_2_gate_real_weight(self):
        try:
            import torch
            from safetensors import safe_open
        except ImportError:
            pytest.skip("torch / safetensors not installed")

        path = _resolve_safetensors_path()
        with safe_open(path, framework="pt") as f:
            w = f.get_tensor(
                "model.layers.0.mlp.gate_proj.weight"
            ).to(torch.float32).numpy().astype(np.float32)
        flat = w.flatten()
        n = flat.size

        indices, scales_int, zeros = _groupwise_quantize_4bit(flat)

        packed = pack_nibbles([int(x) for x in indices])
        recon_x1M = dequant_groupwise_v8_x1M(
            packed, list(scales_int), bytes(zeros), n
        )
        recon_sim = np.array(recon_x1M, dtype=np.float64) / SCALE_FIXED_POINT

        err = recon_sim - flat.astype(np.float64)
        max_abs = float(np.abs(err).max())
        rng_w = float(flat.max() - flat.min())
        gate_pct = max_abs / rng_w * 100.0
        print(
            f"\n  Real HF gate_proj.weight {w.shape}: "
            f"max_abs={max_abs:.6f} ({gate_pct:.2f}% of range)"
        )
        # V28.2 recorded 2.40%. Allow headroom up to the 5% gate.
        assert gate_pct < 5.0, (
            f"Real HF round-trip exceeded 5% gate: {gate_pct:.2f}% "
            f"(V28.2 recorded 2.40%)"
        )
        # Stricter assertion: should be in the same ballpark as V28.2
        assert gate_pct < 3.0, (
            f"Real HF round-trip regressed vs V28.2 baseline: "
            f"{gate_pct:.2f}% (V28.2 was 2.40%)"
        )
