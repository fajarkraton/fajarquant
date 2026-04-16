"""Parity tests for `kernel_sim.vecmat_v8`.

Verification strategy (P1.2 gate):
1. **Golden manual cases** — hand-computed small inputs verify the
   kernel formula is implemented correctly (no off-by-one, nibble
   ordering matches, group indexing matches).
2. **Independent int-reference** — a parallel implementation that uses
   only raw Python `int` + explicit trunc-div helper (no kernel_sim
   imports). If both implementations agree for 100 random inputs, the
   port is internally consistent. Full kernel-trace parity is deferred
   to V30.SIM P2/P3.
3. **Invariants** — zero weights → zero output; identity weights + unit
   scale → dot product of x with dequantized 1s; etc.
"""

from __future__ import annotations

import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

import pytest

from kernel_sim.vecmat_v8 import (
    V8_GROUP_SIZE, V8_SCALE_FP,
    vecmat_v8, pack_nibbles, unpack_nibbles,
)


# ── Independent int reference (no int_ops imports) ────────────────────

def _trunc_div(a: int, b: int) -> int:
    """Standalone trunc-div-toward-zero for cross-check."""
    q = abs(a) // abs(b)
    return -q if (a < 0) ^ (b < 0) else q


def _wrap_i64(x: int) -> int:
    x &= (1 << 64) - 1
    if x >= (1 << 63):
        x -= 1 << 64
    return x


def _vecmat_v8_ref(x_vec, packed, scales, zeros, m, n):
    """Independent port of km_vecmat_packed_v8 — only uses Python int
    and the helpers above. No dependency on kernel_sim.int_ops."""
    total = m * n
    out = [0] * n
    for j in range(n):
        sum_acc = 0
        for k in range(m):
            flat_idx = k * n + j
            byte_idx = flat_idx >> 1
            bit_off = (flat_idx & 1) * 4
            raw = packed[byte_idx] & 0xFF
            q = (raw >> bit_off) & 0xF

            g = flat_idx >> 7
            scale = scales[g] & 0xFFFFFFFF
            zero = zeros[g] & 0xFF

            w_x_1M = _wrap_i64((q - zero) * scale)
            v = x_vec[k]
            term = _trunc_div(_wrap_i64(v * w_x_1M), V8_SCALE_FP)
            sum_acc = _wrap_i64(sum_acc + term)
        out[j] = sum_acc
    return out


# ── Pack/unpack round-trip ────────────────────────────────────────────

class TestPacking:
    def test_pack_unpack_roundtrip(self):
        vals = [0, 15, 1, 14, 7, 8, 3, 12, 5, 10, 2, 13]
        packed = pack_nibbles(vals)
        assert len(packed) == len(vals) // 2
        assert unpack_nibbles(packed, len(vals)) == vals

    def test_pack_low_high(self):
        # [0x0B, 0x0A] should pack to byte 0xAB (low=0x0B, high=0x0A)
        packed = pack_nibbles([0x0B, 0x0A])
        assert packed == bytes([0xAB])

    def test_pack_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            pack_nibbles([16, 0])
        with pytest.raises(ValueError):
            pack_nibbles([-1, 0])

    def test_pack_rejects_odd_length(self):
        with pytest.raises(ValueError):
            pack_nibbles([0, 1, 2])


# ── Golden manual cases (m, n small enough to audit by hand) ──────────

class TestGolden:
    def test_zero_weights(self):
        # All weights = 0, zero=0, scale=1e6 → w = 0 → output all 0
        m, n = 4, 2  # total=8, 1 group (<=128)
        packed = pack_nibbles([0] * (m * n))
        scales = [V8_SCALE_FP]  # doesn't matter, weights are 0
        zeros = bytes([0])
        x = [1000, -500, 2000, -1500]  # arbitrary
        out = vecmat_v8(x, packed, scales, zeros, m, n)
        assert out == [0, 0]

    def test_unit_weights(self):
        # q=1 for all, zero=0, scale=1e6 → w=1 → out[j] = sum_k x[k]
        m, n = 4, 2
        packed = pack_nibbles([1] * (m * n))
        scales = [V8_SCALE_FP]
        zeros = bytes([0])
        x = [1000, 2000, 3000, 4000]
        expected_sum = 10000
        out = vecmat_v8(x, packed, scales, zeros, m, n)
        assert out == [expected_sum, expected_sum]

    def test_identity_single_column(self):
        # m=2, n=1. Weights [q0=5, q1=3]; zero=2, scale=1e6
        # → w0 = (5-2)*1e6 / 1e6 = 3; w1 = (3-2)*1e6 / 1e6 = 1
        # out[0] = x[0]*3 + x[1]*1 after the /1e6 step
        # Kernel: sum += (v * w_x_1M) / 1e6
        #   term0 = (x[0] * 3e6) / 1e6 = x[0] * 3
        #   term1 = (x[1] * 1e6) / 1e6 = x[1] * 1
        m, n = 2, 1
        packed = pack_nibbles([5, 3])  # byte 0x35
        scales = [V8_SCALE_FP]
        zeros = bytes([2])
        x = [7, 11]
        # expected: 7*3 + 11*1 = 21 + 11 = 32
        out = vecmat_v8(x, packed, scales, zeros, m, n)
        assert out == [32]

    def test_negative_dequant(self):
        # q=0, zero=5 → (0-5) * scale = -5e6 → w = -5
        m, n = 2, 1
        packed = pack_nibbles([0, 0])
        scales = [V8_SCALE_FP]
        zeros = bytes([5])
        x = [100, 200]
        # term0 = (100 * -5e6) / 1e6 = -500  (trunc toward 0)
        # term1 = (200 * -5e6) / 1e6 = -1000
        out = vecmat_v8(x, packed, scales, zeros, m, n)
        assert out == [-1500]

    def test_trunc_toward_zero_path(self):
        # Force a case where trunc-div ≠ floor-div.
        # (v * w_x_1M) = -3e6 + 1 = -2_999_999 → / 1e6
        # trunc toward 0: -2 ; floor: -3 → must be -2
        m, n = 1, 1
        # Want w_x_1M = -2_999_999. (q - zero)*scale = -2_999_999.
        # Let q=0, zero=1, scale=2_999_999.
        packed = pack_nibbles([0, 0])  # m*n=1 odd; bump to 2
        # Actually m*n=1 is odd → rejected by impl. Use m=2,n=1 with
        # second row zero'd so it contributes 0.
        packed = pack_nibbles([0, 0])  # row0 q=0, row1 q=0
        scales = [2_999_999]
        zeros = bytes([1])
        x = [1, 0]  # only row0 contributes
        m, n = 2, 1
        # row0: w_x_1M = (0-1)*2_999_999 = -2_999_999
        #       term = (1 * -2_999_999) / 1e6 = trunc(-2.999) = -2
        # row1: same w_x_1M but x=0 → term=0
        out = vecmat_v8(x, packed, scales, zeros, m, n)
        assert out == [-2]  # NOT -3 (would be Python floor-div)

    def test_multi_group_boundary(self):
        # total = 256 → 2 groups (group_size=128). Different scales
        # per group should be picked up correctly.
        m, n = 16, 16  # total = 256
        # Group 0 (flat_idx 0..127): q=1, zero=0, scale=1e6 → w=1
        # Group 1 (flat_idx 128..255): q=2, zero=0, scale=2e6 → w=4
        nibbles = [1] * 128 + [2] * 128
        packed = pack_nibbles(nibbles)
        scales = [V8_SCALE_FP, 2 * V8_SCALE_FP]
        zeros = bytes([0, 0])
        x = [1] * m
        # For each output column j:
        #   k in [0..8): flat_idx = k*16+j in [0..127] → group 0 (w=1)
        #   k in [8..16): flat_idx in [128..255] → group 1 (w=4)
        # so out[j] = 8*1 + 8*4 = 40 for every j
        out = vecmat_v8(x, packed, scales, zeros, m, n)
        assert out == [40] * n


# ── Invariants ────────────────────────────────────────────────────────

class TestInvariants:
    def test_output_dim_matches_n(self):
        m, n = 8, 12
        packed = pack_nibbles([0] * (m * n))
        scales = [V8_SCALE_FP]
        zeros = bytes([0])
        out = vecmat_v8([0] * m, packed, scales, zeros, m, n)
        assert len(out) == n

    def test_rejects_odd_total(self):
        with pytest.raises(ValueError):
            vecmat_v8([0], bytes([0]), [1], bytes([0]), 1, 1)

    def test_rejects_short_scales(self):
        # total=256 needs 2 groups; supplying 1 must raise
        m, n = 16, 16
        packed = pack_nibbles([0] * (m * n))
        with pytest.raises(ValueError):
            vecmat_v8([0] * m, packed, [V8_SCALE_FP], bytes([0, 0]), m, n)

    def test_rejects_short_zeros(self):
        m, n = 16, 16
        packed = pack_nibbles([0] * (m * n))
        with pytest.raises(ValueError):
            vecmat_v8([0] * m, packed, [V8_SCALE_FP, V8_SCALE_FP], bytes([0]), m, n)

    def test_determinism(self):
        random.seed(42)
        m, n = 8, 4
        nibbles = [random.randint(0, 15) for _ in range(m * n)]
        packed = pack_nibbles(nibbles)
        scales = [random.randint(1, 40000)]
        zeros = bytes([random.randint(0, 15)])
        x = [random.randint(-10_000, 10_000) for _ in range(m)]
        a = vecmat_v8(x, packed, scales, zeros, m, n)
        b = vecmat_v8(x, packed, scales, zeros, m, n)
        assert a == b


# ── Parity against independent int reference (100 random cases) ───────

class TestParityAgainstReference:
    @pytest.mark.parametrize("seed", list(range(100)))
    def test_random_matches_reference(self, seed):
        """For 100 random (m, n, weights, scales, zeros, x) tuples, the
        kernel_sim.vecmat_v8 output must match the standalone reference
        byte-for-byte.

        The reference uses only Python int + a local trunc-div helper,
        so it shares no implementation surface with int_ops — making a
        genuine cross-check.
        """
        rng = random.Random(seed)
        # Small dims; keep m*n even and modest so random exhaustion is
        # meaningful in 0.01s per case.
        m = rng.choice([2, 4, 6, 8, 16])
        n = rng.choice([1, 2, 3, 4, 5])
        if m * n % 2 != 0:
            # Ensure total even (4-bit packing requirement)
            m += 1

        total = m * n
        n_groups = (total + V8_GROUP_SIZE - 1) // V8_GROUP_SIZE

        nibbles = [rng.randint(0, 15) for _ in range(total)]
        packed = pack_nibbles(nibbles)
        # Scales in the observed kernel calibration range (3556..38802 per
        # P0.3 gate log) + some outliers
        scales = [rng.randint(1, 2**31 - 1) for _ in range(n_groups)]
        zeros = bytes([rng.randint(0, 15) for _ in range(n_groups)])
        # x in realistic post-rmsnorm ×1000 fp range (with headroom for
        # Gemma-mode gamma up to ×56.7)
        x = [rng.randint(-600_000, 600_000) for _ in range(m)]

        got = vecmat_v8(x, packed, scales, zeros, m, n)
        want = _vecmat_v8_ref(x, packed, scales, zeros, m, n)
        assert got == want, (
            f"seed={seed} m={m} n={n} diverged\n"
            f"  got={got}\n  want={want}"
        )

    def test_large_case_realistic_bounds(self):
        """One medium-size case with realistic Gemma-scale bounds to
        catch any overflow that small cases miss."""
        rng = random.Random(12345)
        m, n = 128, 8  # total = 1024 = 8 groups
        n_groups = 1024 // V8_GROUP_SIZE
        nibbles = [rng.randint(0, 15) for _ in range(m * n)]
        packed = pack_nibbles(nibbles)
        scales = [rng.randint(3556, 38802) for _ in range(n_groups)]  # P0.3 observed range
        zeros = bytes([rng.randint(2, 14) for _ in range(n_groups)])
        x = [rng.randint(-10_000, 10_000) for _ in range(m)]
        got = vecmat_v8(x, packed, scales, zeros, m, n)
        want = _vecmat_v8_ref(x, packed, scales, zeros, m, n)
        assert got == want
