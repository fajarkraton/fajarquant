"""Parity tests: Python int_ops vs FajarOS kernel i64 semantics.

Every test here encodes a kernel arithmetic invariant. Breaking any of
these is grounds to block a kernel commit that changes arithmetic
policy (will be enforced by pre-commit check in V30.SIM P5).
"""

from __future__ import annotations

import sys
import os

# Make tools/ importable without installing
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

import pytest

from kernel_sim.int_ops import (
    INT8_MAX, INT16_MIN, INT32_MAX, INT32_MIN, INT64_MAX, INT64_MIN,
    widen_u8, widen_u32,
    add_i64, sub_i64, mul_i64, trunc_div_i64,
    ashr_i64, lshr_i64, shl_i64,
    bitand_i64, bitor_i64, bitxor_i64,
    clamp_i64, abs_i64,
    volatile_read_u8, volatile_read_u32, volatile_read_u64,
    read_nibble,
)


# ── Widening (u8 / u32 zero-extend to i64) ────────────────────────────

class TestWidening:
    def test_u8_zero(self):
        assert widen_u8(0) == 0

    def test_u8_max(self):
        # u8(0xFF) widens to i64 255, NOT -1 (zero-extend, not sign-extend)
        assert widen_u8(0xFF) == 255

    def test_u8_masks_high_bits(self):
        # Kernel u8 reads only see low 8 bits; simulator mirrors that
        assert widen_u8(0x1FF) == 0xFF
        assert widen_u8(-1) == 0xFF

    def test_u32_max(self):
        # u32(0xFFFFFFFF) widens to i64 4294967295, NOT -1
        assert widen_u32(0xFFFFFFFF) == 4294967295

    def test_u32_masks_high_bits(self):
        assert widen_u32(0x1_00000000) == 0
        assert widen_u32(0xFFFFFFFF_00000000) == 0


# ── Truncated division (LLVM sdiv) vs Python floor division ───────────

class TestTruncDiv:
    """Critical: Python `//` rounds toward -inf; LLVM `sdiv` truncates
    toward 0. For mixed-sign operands they disagree by 1. The kernel
    uses LLVM semantics throughout."""

    def test_positive_exact(self):
        assert trunc_div_i64(7, 2) == 3
        assert 7 // 2 == 3  # Python matches for positive

    def test_negative_dividend(self):
        # Python: -7 // 2 == -4; kernel: -7 / 2 == -3
        assert trunc_div_i64(-7, 2) == -3
        assert -7 // 2 == -4  # confirms divergence from Python

    def test_negative_divisor(self):
        # Python: 7 // -2 == -4; kernel: 7 / -2 == -3
        assert trunc_div_i64(7, -2) == -3
        assert 7 // -2 == -4

    def test_both_negative(self):
        # Python: -7 // -2 == 3; kernel: -7 / -2 == 3
        assert trunc_div_i64(-7, -2) == 3
        assert -7 // -2 == 3

    def test_zero_dividend(self):
        assert trunc_div_i64(0, 5) == 0
        assert trunc_div_i64(0, -5) == 0

    def test_divide_by_zero_raises(self):
        with pytest.raises(ZeroDivisionError):
            trunc_div_i64(10, 0)

    def test_large_values(self):
        # Matches kernel's V8_SCALE_FP = 1_000_000 divide
        assert trunc_div_i64(5_820_000_000, 1_000_000) == 5820
        assert trunc_div_i64(-5_820_000_000, 1_000_000) == -5820

    def test_rmsnorm_k_scale_pattern(self):
        # Matches kernel: `x_rs = x * k_scale / max_abs`
        k_scale = 10000
        # Case: x = -3, max_abs = 7 → kernel path
        x = -3
        max_abs = 7
        got = trunc_div_i64(mul_i64(x, k_scale), max_abs)
        # Expected: trunc(-30000/7) = -4285 (Python gives -4286)
        assert got == -4285
        assert mul_i64(x, k_scale) // max_abs == -4286  # confirms divergence


# ── Arithmetic wrap semantics ─────────────────────────────────────────

class TestWrap:
    def test_add_overflow_wraps_to_min(self):
        assert add_i64(INT64_MAX, 1) == INT64_MIN

    def test_sub_underflow_wraps_to_max(self):
        assert sub_i64(INT64_MIN, 1) == INT64_MAX

    def test_mul_overflow_wraps(self):
        # (1<<33) * (1<<33) = 2^66 wraps to 0 in i64
        assert mul_i64(1 << 33, 1 << 33) == 0

    def test_mul_near_boundary(self):
        # INT64_MAX * 2 wraps — useful sanity for kernel accumulator
        result = mul_i64(INT64_MAX, 2)
        assert result == -2  # two's complement wrap

    def test_add_stays_in_range_nominal(self):
        # Kernel per-term accumulator pattern, well below overflow
        assert add_i64(5_820, 3_900) == 9_720


# ── Bit ops ───────────────────────────────────────────────────────────

class TestShifts:
    def test_ashr_positive(self):
        assert ashr_i64(16, 2) == 4

    def test_ashr_negative_sign_extends(self):
        # -16 >> 2 == -4 (arithmetic shift preserves sign)
        assert ashr_i64(-16, 2) == -4

    def test_ashr_byte_pattern(self):
        # Matches kernel read_nibble: (raw_byte >> bit_off) & 15
        assert ashr_i64(widen_u8(0xAB), 4) == 0x0A  # high nibble
        assert ashr_i64(widen_u8(0xAB), 0) == 0xAB

    def test_lshr_zero_fills(self):
        # -16 as u64 then >>2 yields 0x3FFFFFFFFFFFFFFC
        got = lshr_i64(-16, 2)
        assert got == 0x3FFFFFFFFFFFFFFC

    def test_shl_wraps(self):
        # 1 << 63 == INT64_MIN
        assert shl_i64(1, 63) == INT64_MIN
        # 1 << 64 is UB in LLVM; we return 0 (defensive)
        assert shl_i64(1, 64) == 0

    def test_v8_group_shift(self):
        # Kernel: `g = flat_idx >> V8_GROUP_SHIFT` where V8_GROUP_SHIFT=7
        assert ashr_i64(255, 7) == 1    # group 1
        assert ashr_i64(128, 7) == 1    # boundary
        assert ashr_i64(127, 7) == 0    # group 0


class TestBitAnd:
    def test_nibble_mask(self):
        # Kernel: `q = (raw >> bit_off) & 15`
        assert bitand_i64(0xAB, 15) == 0x0B
        assert bitand_i64(0xA0, 15) == 0x00

    def test_parity_bit(self):
        # Kernel: `bit_off = (flat_idx & 1) * 4`
        assert bitand_i64(0, 1) == 0
        assert bitand_i64(1, 1) == 1
        assert bitand_i64(100, 1) == 0
        assert bitand_i64(101, 1) == 1


# ── abs_i64 (kernel pattern: `if x < 0 { 0 - x } else { x }`) ─────────

class TestAbs:
    def test_positive_unchanged(self):
        assert abs_i64(42) == 42

    def test_negative_flips(self):
        assert abs_i64(-42) == 42

    def test_zero(self):
        assert abs_i64(0) == 0

    def test_int64_min_wraps(self):
        # INT64_MIN has no positive counterpart; kernel wraps to INT64_MIN
        assert abs_i64(INT64_MIN) == INT64_MIN


# ── Memory-read mirrors ───────────────────────────────────────────────

class TestMemoryReads:
    def test_read_u8(self):
        buf = bytes([0x00, 0xFF, 0x7F, 0x80])
        assert volatile_read_u8(buf, 0) == 0
        assert volatile_read_u8(buf, 1) == 255
        assert volatile_read_u8(buf, 2) == 127
        assert volatile_read_u8(buf, 3) == 128

    def test_read_u32_little_endian(self):
        # 0xDEADBEEF stored LE: EF BE AD DE
        buf = bytes([0xEF, 0xBE, 0xAD, 0xDE])
        assert volatile_read_u32(buf, 0) == 0xDEADBEEF

    def test_read_u32_zero_extended(self):
        # u32 max → positive i64, not -1
        buf = bytes([0xFF, 0xFF, 0xFF, 0xFF])
        assert volatile_read_u32(buf, 0) == 0xFFFFFFFF

    def test_read_u64_negative_roundtrip(self):
        # Kernel stores -5000 via volatile_write_u64, then reads back
        # Must reconstruct as i64 -5000, not u64 UINT64-5000
        import struct
        buf = bytearray(8)
        struct.pack_into("<q", buf, 0, -5000)
        assert volatile_read_u64(buf, 0) == -5000

    def test_read_u64_positive(self):
        import struct
        buf = bytearray(8)
        struct.pack_into("<q", buf, 0, 12345678901234)
        assert volatile_read_u64(buf, 0) == 12345678901234


# ── read_nibble: full end-to-end match of kernel pattern ──────────────

class TestReadNibble:
    def test_low_nibble(self):
        # flat_idx = 0 → byte_idx = 0, bit_off = 0 → low nibble
        buf = bytes([0xAB])
        assert read_nibble(buf, 0) == 0x0B

    def test_high_nibble(self):
        # flat_idx = 1 → byte_idx = 0, bit_off = 4 → high nibble
        buf = bytes([0xAB])
        assert read_nibble(buf, 1) == 0x0A

    def test_adjacent_bytes(self):
        buf = bytes([0x12, 0x34])
        assert read_nibble(buf, 0) == 0x02  # 0x12 low
        assert read_nibble(buf, 1) == 0x01  # 0x12 high
        assert read_nibble(buf, 2) == 0x04  # 0x34 low
        assert read_nibble(buf, 3) == 0x03  # 0x34 high

    def test_all_zero(self):
        buf = bytes([0x00] * 10)
        for i in range(20):
            assert read_nibble(buf, i) == 0

    def test_all_full(self):
        buf = bytes([0xFF] * 10)
        for i in range(20):
            assert read_nibble(buf, i) == 0x0F


# ── Kernel patterns composed (sanity) ─────────────────────────────────

class TestKernelPatterns:
    """These mirror specific multi-op patterns that appear verbatim in
    the kernel hot paths. If any fail, the composite operation has
    diverged from kernel semantics."""

    def test_dequant_formula(self):
        # Kernel: w_x_1M = (q - zero) * scale; w = w_x_1M / V8_SCALE_FP
        # with q=10, zero=4, scale=38802 → (6) * 38802 = 232812 (×1e6)
        q = 10
        zero = 4
        scale = 38802
        w_x_1M = mul_i64(sub_i64(q, zero), scale)
        assert w_x_1M == 232812

    def test_dequant_negative(self):
        # q=2, zero=14 → (2-14) = -12; scale=38802 → -465624
        q = 2
        zero = 14
        scale = 38802
        w_x_1M = mul_i64(sub_i64(q, zero), scale)
        assert w_x_1M == -465624

    def test_lmhead_per_term_bound(self):
        # Kernel: sum += (x_val * w_x_1M) / 1_000_000
        # With x_val=10000 (max post-rmsnorm), w_x_1M=582030 (max from analysis)
        x_val = 10000
        w_x_1M = 582030
        term = trunc_div_i64(mul_i64(x_val, w_x_1M), 1_000_000)
        # 10000 * 582030 / 1e6 = 5820300 / 1000 = 5820
        assert term == 5820

    def test_rmsnorm_gemma_gamma(self):
        # Gemma-mode: `(normed * (1000 + g)) / 1000`
        # With normed=10000, g=55750 (gamma max 55.75 × 1000) → 567500
        normed = 10000
        g = 55750
        out = trunc_div_i64(mul_i64(normed, add_i64(1000, g)), 1000)
        assert out == 567500

    def test_argmax_init_sentinel(self):
        # Kernel: `best_score: i64 = -999_999_999`
        # Any plausible per-vocab sum must be >> -999M to update it
        best_score = -999_999_999
        candidate = -1_000_000  # -1M
        assert candidate > best_score
