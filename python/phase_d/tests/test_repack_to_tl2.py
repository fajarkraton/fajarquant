"""V32-prep F.11.0 — Unit tests for the V31 → TL2 weight repacker.

Coverage targets per F.11 design v1.0 §3 sub-task F.11.0:
  - Round-trip: encode → decode reproduces the original ternary stream
  - All-zeros: all-0 input encodes as all-0 (idx=0 everywhere)
  - All-±1: trivial uniform inputs encode + decode correctly
  - Plus property: all 27 (w0, w1, w2) triplets round-trip individually
  - Plus storage: ~1.67 bpw matches F.11 design §1.5 claim
  - Plus tail: K mod 3 ∈ {0, 1, 2} all round-trip
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure scripts/ is on the import path
_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPT_DIR))

from repack_to_tl2 import (  # noqa: E402
    THREE_LUT_CANONICAL_TO_IDX,
    TL2WireLayout,
    TWO_LUT_TO_IDX,
    decode_pair,
    decode_triplet,
    decode_v31_packed,
    encode_pair,
    encode_singleton,
    encode_triplet,
    encode_v31_packed,
    pack_tl2_wire,
    repack_v31_to_tl2,
    tl2_storage_bytes,
    unpack_tl2_wire,
)


# -----------------------------------------------------------------
# §1. Triplet encoding (canonical 14-entry index)
# -----------------------------------------------------------------

def test_three_lut_has_14_canonical_entries() -> None:
    """The canonical-to-index LUT must have exactly 14 entries (indices 0-13)."""
    assert len(THREE_LUT_CANONICAL_TO_IDX) == 14
    assert set(THREE_LUT_CANONICAL_TO_IDX.values()) == set(range(14))


def test_encode_triplet_all_zero_returns_idx_zero() -> None:
    """(0, 0, 0) → idx=0, sign=0 (sign immaterial for all-zero)."""
    idx, sign = encode_triplet(0, 0, 0)
    assert idx == 0
    assert sign == 0


def test_encode_triplet_all_27_round_trip() -> None:
    """Property: every (w0, w1, w2) ∈ {-1, 0, +1}^3 round-trips."""
    for w0 in (-1, 0, 1):
        for w1 in (-1, 0, 1):
            for w2 in (-1, 0, 1):
                idx, sign = encode_triplet(w0, w1, w2)
                assert 0 <= idx <= 13, f"idx {idx} out of [0,13] for ({w0},{w1},{w2})"
                assert sign in (0, 1)
                w0r, w1r, w2r = decode_triplet(idx, sign)
                assert (w0, w1, w2) == (w0r, w1r, w2r), (
                    f"round-trip failed: ({w0},{w1},{w2}) → "
                    f"(idx={idx}, sign={sign}) → ({w0r},{w1r},{w2r})"
                )


def test_encode_triplet_negation_flips_sign_only() -> None:
    """For any nonzero triplet, negating it must flip the sign bit but
    leave the magnitude index unchanged."""
    for w0 in (-1, 0, 1):
        for w1 in (-1, 0, 1):
            for w2 in (-1, 0, 1):
                if (w0, w1, w2) == (0, 0, 0):
                    continue
                idx_pos, sign_pos = encode_triplet(w0, w1, w2)
                idx_neg, sign_neg = encode_triplet(-w0, -w1, -w2)
                assert idx_pos == idx_neg, (
                    f"magnitude mismatch on negation for ({w0},{w1},{w2})"
                )
                assert sign_pos != sign_neg, (
                    f"sign should flip for negated ({w0},{w1},{w2})"
                )


def test_encode_triplet_rejects_non_ternary() -> None:
    with pytest.raises(ValueError, match="not in"):
        encode_triplet(0, 0, 2)
    with pytest.raises(ValueError, match="not in"):
        encode_triplet(-2, 0, 0)


# -----------------------------------------------------------------
# §2. Two_LUT (K mod 3 tail)
# -----------------------------------------------------------------

def test_two_lut_has_9_entries() -> None:
    assert len(TWO_LUT_TO_IDX) == 9
    assert set(TWO_LUT_TO_IDX.values()) == set(range(9))


def test_encode_pair_all_9_round_trip() -> None:
    """All 9 (w0, w1) pairs must round-trip."""
    for w0 in (-1, 0, 1):
        for w1 in (-1, 0, 1):
            idx = encode_pair(w0, w1)
            assert 0 <= idx <= 8
            w0r, w1r = decode_pair(idx)
            assert (w0, w1) == (w0r, w1r)


def test_encode_singleton_round_trip() -> None:
    """K mod 3 = 1 case: singleton encoded as (w, 0) in Two_LUT."""
    for w in (-1, 0, 1):
        idx = encode_singleton(w)
        w_r, second_r = decode_pair(idx)
        assert w_r == w
        assert second_r == 0


# -----------------------------------------------------------------
# §3. V31 2-bit packed decode/encode (round-trip)
# -----------------------------------------------------------------

def test_v31_decode_known_byte() -> None:
    """Known V31 byte: bits {00, 01, 10, 01} = weights (-1, 0, +1, 0)."""
    # Per spec: bit 1-0 = W[0], 3-2 = W[1], 5-4 = W[2], 7-6 = W[3]
    # 01_10_01_00 (msb left) = 0x64
    byte = 0b01_10_01_00  # = 0x64
    packed = bytes([byte])
    decoded = decode_v31_packed(packed, 4)
    np.testing.assert_array_equal(decoded, np.array([-1, 0, 1, 0], dtype=np.int8))


def test_v31_round_trip_random() -> None:
    """encode → decode reproduces the original ternary array."""
    rng = np.random.default_rng(42)
    n = 100  # not a multiple of 4 to exercise tail
    weights = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=n)
    packed = encode_v31_packed(weights)
    decoded = decode_v31_packed(packed, n)
    np.testing.assert_array_equal(decoded, weights)


# -----------------------------------------------------------------
# §4. Wire-layout pack/unpack round-trip — the headline F.11.0 gate
# -----------------------------------------------------------------

def test_wire_round_trip_aligned() -> None:
    """K = 3·n round-trips bit-exact (no tail)."""
    rng = np.random.default_rng(0)
    weights = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=3 * 16)  # 48 weights
    layout = pack_tl2_wire(weights)
    assert layout.tail_kind == "none"
    assert layout.n_full_triplets == 16
    recovered = unpack_tl2_wire(layout)
    np.testing.assert_array_equal(recovered, weights)


def test_wire_round_trip_singleton_tail() -> None:
    """K mod 3 == 1: singleton tail."""
    rng = np.random.default_rng(1)
    weights = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=3 * 16 + 1)  # 49
    layout = pack_tl2_wire(weights)
    assert layout.tail_kind == "singleton"
    recovered = unpack_tl2_wire(layout)
    np.testing.assert_array_equal(recovered, weights)


def test_wire_round_trip_pair_tail() -> None:
    """K mod 3 == 2: 2-weight tail."""
    rng = np.random.default_rng(2)
    weights = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=3 * 16 + 2)  # 50
    layout = pack_tl2_wire(weights)
    assert layout.tail_kind == "pair"
    recovered = unpack_tl2_wire(layout)
    np.testing.assert_array_equal(recovered, weights)


def test_wire_all_zeros() -> None:
    """All-zero input → all-zero magnitude + sign buffers."""
    n = 3 * 8  # 24 weights = 8 triplets
    weights = np.zeros(n, dtype=np.int8)
    layout = pack_tl2_wire(weights)
    # 8 triplets / 2 per byte = 4 magnitude bytes; all should be 0
    # 8 triplets / 8 per byte = 1 sign byte; should be 0
    assert all(b == 0 for b in layout.magnitudes), (
        f"magnitudes not all zero: {layout.magnitudes!r}"
    )
    assert all(b == 0 for b in layout.signs), (
        f"signs not all zero: {layout.signs!r}"
    )
    recovered = unpack_tl2_wire(layout)
    np.testing.assert_array_equal(recovered, weights)


def test_wire_all_positive_one() -> None:
    """All-+1 input round-trips; every triplet is (1,1,1) → idx=13, sign=0."""
    n = 3 * 8
    weights = np.ones(n, dtype=np.int8)
    layout = pack_tl2_wire(weights)
    # Every triplet → idx=13 (=0xD); 2 per byte = upper 0xD | lower 0xD = 0xDD
    assert all(b == 0xDD for b in layout.magnitudes), (
        f"expected all 0xDD: {layout.magnitudes.hex()}"
    )
    # Every sign bit = 0 (leading nonzero is +1)
    assert all(b == 0 for b in layout.signs)
    recovered = unpack_tl2_wire(layout)
    np.testing.assert_array_equal(recovered, weights)


def test_wire_all_negative_one() -> None:
    """All-(-1) input: every triplet is (-1,-1,-1) → idx=13, sign=1.

    Sign byte: every bit set → 0xFF.
    """
    n = 3 * 8  # 8 triplets
    weights = -np.ones(n, dtype=np.int8)
    layout = pack_tl2_wire(weights)
    # Magnitudes still 0xDD (idx=13 since canonical of (-1,-1,-1) is (1,1,1))
    assert all(b == 0xDD for b in layout.magnitudes)
    # Signs: 8 bits set → 0xFF for the one sign byte
    assert layout.signs == bytes([0xFF])
    recovered = unpack_tl2_wire(layout)
    np.testing.assert_array_equal(recovered, weights)


# -----------------------------------------------------------------
# §5. End-to-end V31 → TL2 (the integration entry point)
# -----------------------------------------------------------------

def test_repack_v31_to_tl2_round_trip() -> None:
    """V31 bytes → TL2 layout → unpack → identical ternary array."""
    rng = np.random.default_rng(7)
    n = 384  # divisible by 3 to keep tail empty
    weights = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=n)
    v31_bytes = encode_v31_packed(weights)

    layout = repack_v31_to_tl2(v31_bytes, n)
    recovered = unpack_tl2_wire(layout)
    np.testing.assert_array_equal(recovered, weights)


# -----------------------------------------------------------------
# §6. Storage accounting (matches F.11 design §1.5 1.67 bpw claim)
# -----------------------------------------------------------------

def test_storage_bpw_matches_design_doc() -> None:
    """At large N, bits-per-weight should approach 5/3 ≈ 1.667 (design §1.5)."""
    n = 3 * 10_000  # 30K weights, divisible by 3 (no tail overhead)
    storage = tl2_storage_bytes(n)
    # 5 bits per 3 weights = 5/3 bpw exactly
    assert abs(storage["bits_per_weight"] - 5.0 / 3.0) < 0.01, (
        f"bpw {storage['bits_per_weight']:.4f} too far from 5/3 = {5.0/3.0:.4f}"
    )


def test_storage_strictly_smaller_than_v31() -> None:
    """TL2 wire format must be strictly smaller than V31 (2 bpw) at scale."""
    n = 3 * 1024
    tl2_bytes = tl2_storage_bytes(n)["total_bytes"]
    v31_bytes = (n + 3) // 4  # V31: 2 bpw = 1 byte per 4 weights
    assert tl2_bytes < v31_bytes, (
        f"TL2 ({tl2_bytes}) not smaller than V31 ({v31_bytes})"
    )


# -----------------------------------------------------------------
# §7. Type/contract guards
# -----------------------------------------------------------------

def test_decode_v31_rejects_short_buffer() -> None:
    with pytest.raises(ValueError, match="too short"):
        decode_v31_packed(b"\x00", 100)


def test_decode_triplet_reserved_indices_yield_zero() -> None:
    """Indices 14, 15 are reserved; spec says decode as all-zero."""
    assert decode_triplet(14, 0) == (0, 0, 0)
    assert decode_triplet(15, 0) == (0, 0, 0)


def test_decode_triplet_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of"):
        decode_triplet(16, 0)
    with pytest.raises(ValueError, match="not in"):
        decode_triplet(0, 2)


def test_layout_namedtuple_immutable() -> None:
    """TL2WireLayout is a NamedTuple — immutable."""
    layout = TL2WireLayout(b"", b"", b"", 0, 0, "none")
    with pytest.raises(AttributeError):
        layout.n_weights = 99  # type: ignore[misc]


# -----------------------------------------------------------------
# §X. Tile-organized encoder (V32-prep F.11.4 Branch X-real, scaffold)
# -----------------------------------------------------------------

from repack_to_tl2 import (  # noqa: E402
    TL2TileLayout,
    pack_tl2_tile_organized,
    tl2_tile_layout_sizes,
)


def test_tl2_tile_layout_sizes_mini_256_256() -> None:
    """Per kernel strides for the smallest Mini shape (256, 256)
    with BM=64 / BBK=128:
        a_bytes    = 4 tiles × 2 k_outer × (2 × 21 × 32) = 10,752 B
        sign_bytes = 4 tiles × 2 k_outer × (2 × 5  × 32) = 2,560 B
    """
    sizes = tl2_tile_layout_sizes(m=256, k=256, bm=64, bbk=128)
    assert sizes["a_bytes"] == 10_752, (
        f"expected 10,752 A bytes for (256, 256), got {sizes['a_bytes']}"
    )
    assert sizes["sign_bytes"] == 2_560, (
        f"expected 2,560 sign bytes for (256, 256), got {sizes['sign_bytes']}"
    )
    assert sizes["n_tiles"] == 4
    assert sizes["n_kouter"] == 2
    assert sizes["a_per_kouter"] == 1_344
    assert sizes["sign_per_kouter"] == 320


def test_tl2_tile_layout_sizes_mini_1536_256() -> None:
    """Larger Mini shape: gate_proj (1536, 256). 24 tiles × 2 k_outer."""
    sizes = tl2_tile_layout_sizes(m=1536, k=256, bm=64, bbk=128)
    assert sizes["n_tiles"] == 24
    assert sizes["n_kouter"] == 2
    assert sizes["a_bytes"] == 24 * 2 * 1_344  # = 64,512
    assert sizes["sign_bytes"] == 24 * 2 * 320  # = 15,360


def test_tl2_tile_layout_sizes_rejects_non_divisible() -> None:
    with pytest.raises(ValueError, match="not divisible by BM"):
        tl2_tile_layout_sizes(m=200, k=256, bm=64, bbk=128)
    with pytest.raises(ValueError, match="not divisible by BBK"):
        tl2_tile_layout_sizes(m=256, k=200, bm=64, bbk=128)


def test_pack_tl2_tile_organized_scaffold_returns_correct_sizes() -> None:
    """Scaffold check: byte-count matches tl2_tile_layout_sizes
    (content is zero-initialized; populated in next iteration)."""
    weights = np.zeros((256, 256), dtype=np.int8)
    layout = pack_tl2_tile_organized(weights, bm=64, bbk=128)
    assert isinstance(layout, TL2TileLayout)
    assert len(layout.magnitudes) == 10_752
    assert len(layout.signs) == 2_560
    assert layout.m == 256
    assert layout.k == 256
    assert layout.bm == 64
    assert layout.bbk == 128
    assert layout.n_tiles == 4
    assert layout.n_kouter == 2


def test_pack_tl2_tile_organized_rejects_wrong_dim() -> None:
    """1-D weight arrays are rejected; only 2-D (m, k) supported."""
    weights_1d = np.zeros(256 * 256, dtype=np.int8)
    with pytest.raises(ValueError, match="must be 2-D"):
        pack_tl2_tile_organized(weights_1d, bm=64, bbk=128)


def test_pack_tl2_tile_organized_zero_input_yields_zero_magnitudes() -> None:
    """All-zero input → magnitude index 0 everywhere → all bytes zero.
    (Index 0 is the canonical encoding for triplet (0, 0, 0).)"""
    weights = np.zeros((256, 256), dtype=np.int8)
    layout = pack_tl2_tile_organized(weights, bm=64, bbk=128)
    assert all(b == 0 for b in layout.magnitudes)


def test_pack_tl2_tile_organized_all_positive_yields_idx13() -> None:
    """All-+1 input → every triplet is (+1, +1, +1) → idx=13 = 0xD.
    Per the kernel's nibble interpretation:
        upper nibble = first triplet of pair = 0xD
        lower nibble = second triplet of pair = 0xD
    → every byte = 0xDD.
    """
    weights = np.full((256, 256), 1, dtype=np.int8)
    layout = pack_tl2_tile_organized(weights, bm=64, bbk=128)
    # Note: kk=42 triplets per k_outer × 2 k_outer = 84 triplets per row.
    # 84 * 3 = 252 weights covered. The remaining 4 weights per row
    # (256 - 252 = 4) are dropped by the kernel's KK = BBK/3 truncation.
    # So we expect all populated bytes to be 0xDD; no zero gaps for
    # this all-+1 input.
    assert all(b == 0xDD for b in layout.magnitudes), (
        f"expected all 0xDD; first non-DD byte at "
        f"{next((i for i, b in enumerate(layout.magnitudes) if b != 0xDD), -1)}"
    )


def test_pack_tl2_tile_organized_known_offset_first_triplet() -> None:
    """Hand-computed offset for output_row=0, triplet_col_idx=0:
        tile_idx=0, row_in_tile=0, i_group=0, lane=0
        k_outer=0, triplet_in_kouter=0, ai=0, nibble=0 (upper)
    → byte_offset = 0
    → upper nibble holds magnitude index for this triplet.

    Construct an array where ONLY the first triplet (cols 0..2 of
    row 0) is (+1, +1, +1) → idx=13 = 0xD. All other triplets are
    zero → idx=0.

    Expected: byte 0 has upper nibble = 0xD, lower nibble = 0
    (next triplet pair, also zero in our setup) → byte 0 = 0xD0.
    All other bytes should be zero.
    """
    weights = np.zeros((256, 256), dtype=np.int8)
    weights[0, 0] = 1
    weights[0, 1] = 1
    weights[0, 2] = 1

    layout = pack_tl2_tile_organized(weights, bm=64, bbk=128)
    assert layout.magnitudes[0] == 0xD0, (
        f"expected byte 0 = 0xD0 (upper nibble = idx 13 for triplet "
        f"(+1,+1,+1) at lane 0, lower nibble = 0); got "
        f"0x{layout.magnitudes[0]:02X}"
    )
    # Spot-check: bytes 1..31 (rest of the ai=0 vector for output row 0,
    # but those lanes are output rows 1..31) should be zero — those
    # rows have no non-zero triplets in our setup.
    assert all(b == 0 for b in layout.magnitudes[1:32])


def test_pack_tl2_tile_organized_signs_still_zero_iter2() -> None:
    """Iteration 2 STATUS: sign byte stream is still zero (sign
    packing deferred to iteration 3). Once iteration 3 lands, this
    test will need updating — likely flips to a hand-computed
    sign-bit position check or replaced by parity_real_mlp."""
    weights = np.full((256, 256), -1, dtype=np.int8)  # all sign-bit-1
    layout = pack_tl2_tile_organized(weights, bm=64, bbk=128)
    assert all(b == 0 for b in layout.signs), (
        "iteration 2 expected sign bytes still zero; if non-zero, "
        "iteration 3 sign-packing landed — replace this test with a "
        "real sign-bit-offset check or parity test"
    )
