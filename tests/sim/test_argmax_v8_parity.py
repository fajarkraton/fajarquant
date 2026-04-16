"""Parity + diagnostic tests for `kernel_sim.argmax_v8`.

P1.4 gate (from plan): 10 test logit arrays match kernel reference.
This suite adds:
- Golden cases (vocab=4, d_model=2 — hand-computed argmax outcomes)
- Tie-break semantics (strict `>`, first-seen wins)
- Sentinel replacement on first positive sum
- Independent reference cross-check (50 random inputs @ 0 ULP)
- **Pad-collapse diagnostic tests** — directly probe the failure mode
  observed on the live kernel (all 64 tokens → token 0):
    * Near-uniform negative sums → argmax=0 by tie-break
    * Row 0 genuinely dominates → argmax=0 legitimately
    * Distinct row dominates → argmax != 0 (sanity: simulator can
      find non-pad winners, so any kernel-vs-sim mismatch on real
      weights is a real divergence)
"""

from __future__ import annotations

import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

import pytest

from kernel_sim.argmax_v8 import (
    ARGMAX_SENTINEL, argmax_v8, argmax_v8_full_logits,
)
from kernel_sim.vecmat_v8 import (
    V8_GROUP_SIZE, V8_SCALE_FP, pack_nibbles,
)


# ── Independent reference (no int_ops imports) ────────────────────────

def _trunc_div(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return -q if (a < 0) ^ (b < 0) else q


def _wrap_i64(x: int) -> int:
    x &= (1 << 64) - 1
    if x >= (1 << 63):
        x -= 1 << 64
    return x


def _argmax_v8_ref(x_vec, packed, scales, zeros, vocab_size, d_model):
    best_token = 0
    best_score = ARGMAX_SENTINEL
    for v in range(vocab_size):
        row_start = v * d_model
        sum_acc = 0
        for i in range(d_model):
            flat_idx = row_start + i
            byte_idx = flat_idx >> 1
            bit_off = (flat_idx & 1) * 4
            raw = packed[byte_idx] & 0xFF
            q = (raw >> bit_off) & 0xF
            g = flat_idx >> 7
            scale = scales[g] & 0xFFFFFFFF
            zero = zeros[g] & 0xFF
            w_x_1M = _wrap_i64((q - zero) * scale)
            x_val = x_vec[i]
            term = _trunc_div(_wrap_i64(x_val * w_x_1M), V8_SCALE_FP)
            sum_acc = _wrap_i64(sum_acc + term)
        if sum_acc > best_score:
            best_score = sum_acc
            best_token = v
    return best_token, best_score


# ── Golden manual cases ───────────────────────────────────────────────

class TestGolden:
    def test_all_zero_weights_returns_token_zero(self):
        # Every row scores 0; sentinel is -999M; first row beats it.
        vocab, dm = 4, 2
        packed = pack_nibbles([0] * (vocab * dm))
        scales = [V8_SCALE_FP]
        zeros = bytes([0])
        x = [1000, 2000]
        tok, score = argmax_v8(x, packed, scales, zeros, vocab, dm)
        assert tok == 0
        assert score == 0

    def test_distinct_row_dominates(self):
        # Design: vocab=4, d_model=2; only row 2 has non-zero weights.
        # Row 2 contributes a large positive score, others ~0.
        # Expected: argmax = 2.
        vocab, dm = 4, 2
        # flat_idx: row 0 → [0,1]; row 1 → [2,3]; row 2 → [4,5]; row 3 → [6,7]
        # Give row 2 weights q=10 (with zero=0 → w=scale/1e6)
        nibbles = [0] * 8
        nibbles[4] = 10  # row 2, col 0
        nibbles[5] = 10  # row 2, col 1
        packed = pack_nibbles(nibbles)
        scales = [V8_SCALE_FP]  # 1 group; scale = 1e6 → dequant w=10
        zeros = bytes([0])
        x = [1000, 2000]
        # sum for row 2: (1000*10e6)/1e6 + (2000*10e6)/1e6 = 10000 + 20000 = 30000
        tok, score = argmax_v8(x, packed, scales, zeros, vocab, dm)
        assert tok == 2
        assert score == 30000

    def test_ties_first_seen_wins(self):
        # All rows produce equal scores. Argmax must return token 0.
        vocab, dm = 5, 2
        # q=5 everywhere, zero=0, scale=1e6 → w=5
        packed = pack_nibbles([5] * (vocab * dm))
        scales = [V8_SCALE_FP]
        zeros = bytes([0])
        x = [100, 200]
        # Every row scores (100*5 + 200*5) = 1500
        tok, score = argmax_v8(x, packed, scales, zeros, vocab, dm)
        assert tok == 0
        assert score == 1500

    def test_sentinel_beaten_by_zero(self):
        # With all-zero weights, row 0 scores 0 which > -999_999_999
        vocab, dm = 3, 2
        packed = pack_nibbles([0] * 6)
        scales = [V8_SCALE_FP]
        zeros = bytes([0])
        x = [0, 0]
        tok, score = argmax_v8(x, packed, scales, zeros, vocab, dm)
        assert tok == 0
        # First row replaces sentinel with its own sum (0)
        assert score == 0

    def test_all_scores_negative(self):
        # Inputs negative, weights positive → all rows negative-scored.
        # But still argmax = row with least-negative sum.
        vocab, dm = 3, 2
        # Row 0: q=[10,10] ; Row 1: q=[5,5] ; Row 2: q=[1,1]
        # With x=[-100, -200] and scale=1e6, zero=0:
        #   row 0: (-100*10 + -200*10) = -3000 (most negative)
        #   row 1: (-100*5 + -200*5)   = -1500
        #   row 2: (-100*1 + -200*1)   = -300 (least negative → wins)
        nibbles = [10, 10, 5, 5, 1, 1]
        packed = pack_nibbles(nibbles)
        scales = [V8_SCALE_FP]
        zeros = bytes([0])
        x = [-100, -200]
        tok, score = argmax_v8(x, packed, scales, zeros, vocab, dm)
        assert tok == 2
        assert score == -300


# ── Pad-collapse diagnostic scenarios ─────────────────────────────────

class TestPadCollapseDiagnostics:
    """Probe the specific failure modes that could produce the live
    kernel symptom (64/64 tokens → pad). These aren't parity tests;
    they confirm the simulator correctly models each hypothesized
    mechanism so P3 analysis can distinguish them."""

    def test_near_uniform_negative_produces_token_zero(self):
        """If rmsnorm collapse produces hidden states that cause
        every vocab row to score near the same (large negative) value,
        argmax ties break to token 0 by strict `>` comparison. This
        is the most benign pad-collapse mechanism: kernel math is
        correct, but input to argmax is degenerate."""
        vocab, dm = 10, 4
        # q=1 for all → w=1; with x=[0,0,0,0] → all scores = 0 (tied)
        packed = pack_nibbles([1] * (vocab * dm))
        scales = [V8_SCALE_FP]
        zeros = bytes([0])
        x = [0, 0, 0, 0]
        tok, score = argmax_v8(x, packed, scales, zeros, vocab, dm)
        # All rows score 0 → first-seen (tok=0) wins
        assert tok == 0
        assert score == 0

    def test_row_zero_genuinely_dominates(self):
        """If vocab row 0 (pad token) happens to have weights that
        genuinely score highest for the given x, argmax=0 is correct.
        Distinguishable from the near-uniform case by logit spread."""
        vocab, dm = 8, 4
        # Row 0 gets q=15 (max positive after dequant); others q=1
        nibbles = [15] * 4 + [1] * (7 * 4)
        packed = pack_nibbles(nibbles)
        scales = [V8_SCALE_FP]
        zeros = bytes([0])
        x = [100, 100, 100, 100]
        tok, score, logits = argmax_v8_full_logits(
            x, packed, scales, zeros, vocab, dm
        )
        assert tok == 0
        # Row 0: 4 × (100 × 15) = 6000 ; other rows: 4 × (100 × 1) = 400
        assert logits[0] == 6000
        assert logits[1] == 400
        # Spread = 6000 - 400 = 5600 (not near-uniform)
        assert max(logits) - min(logits) > 1000

    def test_logit_spread_distinguishes_mechanisms(self):
        """Near-uniform scenarios have tight logit spread; genuine-
        dominance scenarios have wide spread. P3 diagnostic gate:
        if kernel shows tok=0 with spread < 100, that's tie-break
        pad-collapse (S2/S3 rmsnorm-upstream bug). If spread > 10000,
        that's genuine pad dominance (S1 overflow or weight-layout bug)."""
        vocab, dm = 10, 4

        # Near-uniform: all q=1, zero=0, x=[1,1,1,1]
        # Every row scores 4 → spread=0
        packed_uniform = pack_nibbles([1] * (vocab * dm))
        _, _, logits_uniform = argmax_v8_full_logits(
            [1, 1, 1, 1], packed_uniform, [V8_SCALE_FP], bytes([0]), vocab, dm
        )
        spread_uniform = max(logits_uniform) - min(logits_uniform)
        assert spread_uniform == 0, f"expected tight spread, got {spread_uniform}"

        # Genuine-dominance: row 5 has q=15, others q=0
        nibbles_dom = [0] * (vocab * dm)
        for i in range(dm):
            nibbles_dom[5 * dm + i] = 15
        packed_dom = pack_nibbles(nibbles_dom)
        tok_dom, _, logits_dom = argmax_v8_full_logits(
            [100, 100, 100, 100], packed_dom,
            [V8_SCALE_FP], bytes([0]), vocab, dm,
        )
        assert tok_dom == 5
        spread_dom = max(logits_dom) - min(logits_dom)
        assert spread_dom >= 6000, f"expected wide spread, got {spread_dom}"


# ── Invariants ────────────────────────────────────────────────────────

class TestInvariants:
    def test_rejects_mismatched_x_length(self):
        with pytest.raises(ValueError):
            argmax_v8(
                [1, 2, 3], pack_nibbles([0] * 4),
                [V8_SCALE_FP], bytes([0]), 2, 2,
            )

    def test_rejects_odd_total(self):
        with pytest.raises(ValueError):
            # vocab=1, d_model=1 → total=1, odd
            argmax_v8([0], b"\x00", [V8_SCALE_FP], bytes([0]), 1, 1)

    def test_rejects_short_scales(self):
        # vocab*d_model=256 → 2 groups; supply 1
        vocab, dm = 16, 16
        packed = pack_nibbles([0] * (vocab * dm))
        with pytest.raises(ValueError):
            argmax_v8(
                [0] * dm, packed, [V8_SCALE_FP], bytes([0, 0]), vocab, dm,
            )

    def test_determinism(self):
        rng = random.Random(123)
        vocab, dm = 16, 8
        nibbles = [rng.randint(0, 15) for _ in range(vocab * dm)]
        packed = pack_nibbles(nibbles)
        scales = [rng.randint(1, 40000)]
        zeros = bytes([rng.randint(0, 15)])
        x = [rng.randint(-10000, 10000) for _ in range(dm)]
        r1 = argmax_v8(x, packed, scales, zeros, vocab, dm)
        r2 = argmax_v8(x, packed, scales, zeros, vocab, dm)
        assert r1 == r2

    def test_full_logits_matches_argmax(self):
        # The two APIs must return the same best_token + best_score
        rng = random.Random(456)
        vocab, dm = 32, 16
        nibbles = [rng.randint(0, 15) for _ in range(vocab * dm)]
        packed = pack_nibbles(nibbles)
        n_groups = (vocab * dm + V8_GROUP_SIZE - 1) // V8_GROUP_SIZE
        scales = [rng.randint(1, 40000) for _ in range(n_groups)]
        zeros = bytes([rng.randint(0, 15) for _ in range(n_groups)])
        x = [rng.randint(-5000, 5000) for _ in range(dm)]
        tok_a, score_a = argmax_v8(x, packed, scales, zeros, vocab, dm)
        tok_b, score_b, logits = argmax_v8_full_logits(
            x, packed, scales, zeros, vocab, dm
        )
        assert tok_a == tok_b
        assert score_a == score_b
        assert logits[tok_b] == score_b


# ── Parity vs independent reference (50 random cases) ────────────────

class TestParityAgainstReference:
    @pytest.mark.parametrize("seed", list(range(50)))
    def test_random_matches_reference(self, seed):
        rng = random.Random(seed + 3000)
        vocab = rng.choice([2, 4, 8, 16, 32, 64])
        dm = rng.choice([2, 4, 8, 16])
        if (vocab * dm) % 2 != 0:
            dm += 1

        total = vocab * dm
        n_groups = (total + V8_GROUP_SIZE - 1) // V8_GROUP_SIZE
        nibbles = [rng.randint(0, 15) for _ in range(total)]
        packed = pack_nibbles(nibbles)
        scales = [rng.randint(1, 38802) for _ in range(n_groups)]
        zeros = bytes([rng.randint(0, 15) for _ in range(n_groups)])
        x = [rng.randint(-10000, 10000) for _ in range(dm)]

        got = argmax_v8(x, packed, scales, zeros, vocab, dm)
        want = _argmax_v8_ref(x, packed, scales, zeros, vocab, dm)
        assert got == want, (
            f"seed={seed} vocab={vocab} dm={dm} "
            f"got={got} want={want}"
        )

    def test_realistic_gemma_scale_case(self):
        """Small but representative case: vocab=256, d_model=128.
        Tests that the port scales correctly before attempting the
        live Gemma 3 (262144 × 1152 ~= 302M inner iterations, way
        too slow for a unit test)."""
        rng = random.Random(789)
        vocab, dm = 256, 128  # total = 32768 = 256 groups
        n_groups = 32768 // V8_GROUP_SIZE
        nibbles = [rng.randint(0, 15) for _ in range(vocab * dm)]
        packed = pack_nibbles(nibbles)
        scales = [rng.randint(3556, 38802) for _ in range(n_groups)]
        zeros = bytes([rng.randint(2, 14) for _ in range(n_groups)])
        x = [rng.randint(-10_000, 10_000) for _ in range(dm)]
        got = argmax_v8(x, packed, scales, zeros, vocab, dm)
        want = _argmax_v8_ref(x, packed, scales, zeros, vocab, dm)
        assert got == want
