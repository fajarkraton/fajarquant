"""Parity tests for `kernel_sim.rmsnorm` + `kernel_sim.km_isqrt`.

P1.3 gate (from plan): 50 random inputs match independent int-reference
at 0 ULP. This suite adds golden small cases, Gemma-gamma magnification
measurement (relevant to S3 HIGH-priority hypothesis), edge cases
(all-zero, single value, dim>8192), and km_isqrt-standalone tests.
"""

from __future__ import annotations

import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

import pytest

from kernel_sim.rmsnorm import (
    K_SCALE,
    GAMMA_MODE_NONE, GAMMA_MODE_LLAMA, GAMMA_MODE_GEMMA,
    km_isqrt, rmsnorm,
)


# ── Independent int reference (no int_ops imports) ────────────────────

def _trunc_div(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return -q if (a < 0) ^ (b < 0) else q


def _wrap_i64(x: int) -> int:
    x &= (1 << 64) - 1
    if x >= (1 << 63):
        x -= 1 << 64
    return x


def _km_isqrt_ref(x: int) -> int:
    if x <= 0:
        return 1
    if x == 1:
        return 1
    guess = _trunc_div(x, 2)
    if guess == 0:
        return 1
    prev = guess + 1
    while guess < prev:
        prev = guess
        guess = _trunc_div(guess + _trunc_div(x, guess), 2)
    return 1 if prev == 0 else prev


def _rmsnorm_ref(data, gamma=None, gamma_mode=0, k_scale=K_SCALE):
    dim = len(data)
    if dim <= 0 or dim > 8192:
        return list(data)
    max_abs = 0
    for x in data:
        ax = -x if x < 0 else x
        if ax > max_abs:
            max_abs = ax
    if max_abs <= 0:
        return list(data)

    rss = 0
    for x in data:
        x_rs = _trunc_div(_wrap_i64(x * k_scale), max_abs)
        rss = _wrap_i64(rss + _trunc_div(_wrap_i64(x_rs * x_rs), dim))

    rms_rs = _km_isqrt_ref(rss + 1)
    if rms_rs <= 0:
        return list(data)

    out = [0] * dim
    for i, x in enumerate(data):
        x_rs = _trunc_div(_wrap_i64(x * k_scale), max_abs)
        normed = _trunc_div(_wrap_i64(x_rs * 1000), rms_rs)
        if gamma is None or gamma_mode == 0:
            out[i] = normed
        elif gamma_mode == 1:
            out[i] = _trunc_div(_wrap_i64(normed * gamma[i]), 1000)
        elif gamma_mode == 2:
            out[i] = _trunc_div(
                _wrap_i64(normed * (1000 + gamma[i])), 1000
            )
        else:
            raise ValueError(f"bad gamma_mode {gamma_mode}")
    return out


# ── km_isqrt standalone ───────────────────────────────────────────────

class TestIsqrt:
    def test_zero_returns_one(self):
        # Kernel guard: isqrt(0) returns 1 (avoids div-by-zero in caller)
        assert km_isqrt(0) == 1

    def test_one(self):
        assert km_isqrt(1) == 1

    def test_perfect_squares(self):
        for i in range(1, 200):
            assert km_isqrt(i * i) == i, f"isqrt({i*i})"

    def test_off_by_one_below(self):
        for i in range(2, 200):
            # floor(sqrt(i*i - 1)) == i - 1
            assert km_isqrt(i * i - 1) == i - 1, f"isqrt({i*i - 1})"

    def test_off_by_one_above(self):
        for i in range(2, 200):
            # floor(sqrt(i*i + 1)) == i
            assert km_isqrt(i * i + 1) == i, f"isqrt({i*i + 1})"

    def test_random_vs_math_isqrt(self):
        rng = random.Random(99)
        for _ in range(500):
            x = rng.randint(1, 10**15)
            assert km_isqrt(x) == math.isqrt(x), f"diverged at x={x}"

    def test_negative_returns_one(self):
        # Kernel guard
        assert km_isqrt(-1) == 1
        assert km_isqrt(-999999) == 1


# ── Golden manual cases ───────────────────────────────────────────────

class TestGolden:
    def test_all_zero(self):
        # max|x|=0 → kernel early-returns without writing; port preserves input
        out = rmsnorm([0, 0, 0, 0])
        assert out == [0, 0, 0, 0]

    def test_constant_vector(self):
        # All equal positive: max_abs = 1000, x_rs = k_scale for every i
        # rss = K² / 1 (dim=1, one entry) → rms_rs = K → normed = K·1000/K = 1000
        # Actually rss accumulates across dim entries and divides by dim:
        # each term = K²/dim; sum = dim × K²/dim = K² ; rms_rs = K
        # So normed[i] = K × 1000 / K = 1000 for all i
        data = [1000] * 8
        out = rmsnorm(data)
        assert all(v == 1000 for v in out)

    def test_symmetric_signs(self):
        # [+v, -v, +v, -v] has same max_abs and equal magnitudes
        # Post-norm should be symmetric in sign, same magnitude
        data = [500, -500, 500, -500]
        out = rmsnorm(data)
        assert len(out) == 4
        assert out[0] == -out[1]
        assert out[2] == -out[3]
        assert abs(out[0]) == abs(out[1]) == abs(out[2]) == abs(out[3])

    def test_gamma_none_equivalent_to_mode_zero(self):
        # With mode=LLAMA but gamma=None, port falls back to no-gamma path
        data = [100, 200, 300, 400]
        a = rmsnorm(data)
        b = rmsnorm(data, gamma=None, gamma_mode=GAMMA_MODE_LLAMA)
        c = rmsnorm(data, gamma=None, gamma_mode=GAMMA_MODE_GEMMA)
        assert a == b == c

    def test_llama_gamma_scaling(self):
        # Uniform gamma of 2000 means y = normed * 2000 / 1000 = 2× normed
        data = [100, 200, 300, 400]
        gamma = [2000, 2000, 2000, 2000]
        base = rmsnorm(data)
        scaled = rmsnorm(data, gamma=gamma, gamma_mode=GAMMA_MODE_LLAMA)
        for i in range(4):
            # Trunc-div introduces up to ±1 ULP, so allow that tolerance
            assert abs(scaled[i] - 2 * base[i]) <= 1, (
                f"i={i} base={base[i]} scaled={scaled[i]}"
            )

    def test_gemma_gamma_near_zero_is_identity(self):
        # Gemma-mode gamma of 0 means y = normed * (1000 + 0) / 1000 = normed
        data = [100, 200, 300, 400]
        gamma = [0, 0, 0, 0]
        base = rmsnorm(data)
        gemma = rmsnorm(data, gamma=gamma, gamma_mode=GAMMA_MODE_GEMMA)
        assert gemma == base


# ── S3 hypothesis: Gemma gamma ×56.7 magnification ────────────────────

class TestGemmaMagnification:
    """P0.2 analysis: Gemma 3 gamma max=55.75 produces (1000 + 55750) =
    56750× multiplier; post-rmsnorm magnitude can exceed k_scale by 56.7×.
    These tests measure that empirically."""

    def test_large_gamma_amplifies_by_expected_factor(self):
        # Uniform input [1000]×8; with gamma=55750 (55.75 real)
        # base normed ≈ 1000 (from test_constant_vector)
        # Gemma: y = 1000 × (1000 + 55750) / 1000 = 56750
        data = [1000] * 8
        gamma = [55750] * 8
        base = rmsnorm(data)
        gemma = rmsnorm(data, gamma=gamma, gamma_mode=GAMMA_MODE_GEMMA)
        # Each entry should be ~56750 (the formal bound from P0.2)
        for v in gemma:
            assert 56000 <= v <= 57500, (
                f"gemma output {v} outside expected ~56750 "
                f"band (base was {base[0]})"
            )

    def test_magnification_factor(self):
        # With base=1000 and gamma=55750, factor vs base should be ~56.75
        data = [1000] * 8
        gamma = [55750] * 8
        base = rmsnorm(data)
        gemma = rmsnorm(data, gamma=gamma, gamma_mode=GAMMA_MODE_GEMMA)
        factor = gemma[0] / base[0] if base[0] else 0
        assert 56.0 < factor < 57.5, (
            f"Gemma magnification factor {factor:.2f} outside "
            f"expected 56.75× band"
        )

    def test_negative_gamma_attenuates(self):
        # Gemma: y = normed × (1000 + g) / 1000. If g=-500, factor=0.5
        data = [1000] * 8
        gamma = [-500] * 8
        base = rmsnorm(data)
        gemma = rmsnorm(data, gamma=gamma, gamma_mode=GAMMA_MODE_GEMMA)
        # Each entry ≈ base × 0.5
        for i in range(8):
            expected = base[i] // 2
            assert abs(gemma[i] - expected) <= 2, (
                f"i={i} gemma={gemma[i]} expected~{expected}"
            )


# ── Invariants ────────────────────────────────────────────────────────

class TestInvariants:
    def test_rms_is_scale_invariant(self):
        """Kernel's rescaling assumption: `rms(α·x) == α·rms(x)` so
        normalized output should be invariant to uniform input scale.

        In practice trunc-div introduces ~1 ULP per operation, so we
        allow a small tolerance — but the direction should be preserved
        and magnitude within a few %.
        """
        base_x = [100, -200, 300, -400, 500, -600, 700, -800]
        out_small = rmsnorm(base_x)
        out_large = rmsnorm([x * 100 for x in base_x])
        # Normalized result should have same sign pattern
        for s, l in zip(out_small, out_large):
            assert (s > 0) == (l > 0) or (s == 0 == l), (
                f"sign mismatch: small={s} large={l}"
            )
        # Magnitudes should match within ~5% (allowing trunc-div drift)
        for s, l in zip(out_small, out_large):
            if s != 0:
                rel = abs(l - s) / abs(s)
                assert rel < 0.05, (
                    f"scale-invariance broken >5%: small={s} large={l}"
                )

    def test_rejects_dim_zero(self):
        with pytest.raises(ValueError):
            rmsnorm([])

    def test_dim_over_8192_returns_unchanged(self):
        # Kernel early-return; port mirrors by returning input copy
        data = list(range(8193))
        out = rmsnorm(data)
        assert out == data

    def test_rejects_mismatched_gamma_length(self):
        with pytest.raises(ValueError):
            rmsnorm([1, 2, 3, 4], gamma=[1, 2], gamma_mode=GAMMA_MODE_LLAMA)

    def test_determinism(self):
        rng = random.Random(7)
        data = [rng.randint(-10000, 10000) for _ in range(64)]
        a = rmsnorm(data)
        b = rmsnorm(data)
        assert a == b


# ── Parity vs independent reference (50+ random cases) ────────────────

class TestParityAgainstReference:
    @pytest.mark.parametrize("seed", list(range(50)))
    def test_no_gamma(self, seed):
        rng = random.Random(seed)
        dim = rng.choice([4, 8, 16, 64, 128, 1152])  # 1152 = Gemma 3 d_model
        data = [rng.randint(-100_000, 100_000) for _ in range(dim)]
        got = rmsnorm(data)
        want = _rmsnorm_ref(data)
        assert got == want, (
            f"seed={seed} dim={dim} diverged (no-gamma path)"
        )

    @pytest.mark.parametrize("seed", list(range(50)))
    def test_llama_gamma(self, seed):
        rng = random.Random(seed + 1000)
        dim = rng.choice([4, 8, 16, 64, 128])
        data = [rng.randint(-50_000, 50_000) for _ in range(dim)]
        gamma = [rng.randint(0, 3000) for _ in range(dim)]  # llama-style scales
        got = rmsnorm(data, gamma=gamma, gamma_mode=GAMMA_MODE_LLAMA)
        want = _rmsnorm_ref(data, gamma=gamma, gamma_mode=1)
        assert got == want, f"seed={seed} dim={dim} diverged (llama)"

    @pytest.mark.parametrize("seed", list(range(50)))
    def test_gemma_gamma(self, seed):
        rng = random.Random(seed + 2000)
        dim = rng.choice([4, 8, 16, 64, 128])
        data = [rng.randint(-50_000, 50_000) for _ in range(dim)]
        # Gemma 3 observed range: mean=4.55, max=55.75 → ×1000 fp = up to 55_750
        # Non-zero-centered; sample uniformly across that range
        gamma = [rng.randint(-5000, 56_000) for _ in range(dim)]
        got = rmsnorm(data, gamma=gamma, gamma_mode=GAMMA_MODE_GEMMA)
        want = _rmsnorm_ref(data, gamma=gamma, gamma_mode=2)
        assert got == want, f"seed={seed} dim={dim} diverged (gemma)"

    def test_large_dim_realistic(self):
        """One dim=1152 case matching Gemma 3 1B d_model with realistic
        non-zero-centered gamma (mean=4.55, max=55.75). Catches drift
        that small cases might miss."""
        rng = random.Random(42)
        dim = 1152
        data = [rng.randint(-30_000, 30_000) for _ in range(dim)]
        # Normal-ish around mean 4.55 × 1000 = 4550, tail up to 55_750
        gamma = [
            rng.randint(-2000, 10000) if rng.random() > 0.02
            else rng.randint(40_000, 55_750)
            for _ in range(dim)
        ]
        got = rmsnorm(data, gamma=gamma, gamma_mode=GAMMA_MODE_GEMMA)
        want = _rmsnorm_ref(data, gamma=gamma, gamma_mode=2)
        assert got == want
