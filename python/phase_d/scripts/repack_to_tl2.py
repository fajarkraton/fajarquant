"""V32-prep F.11.0 — Offline V31 Phase D → microsoft/BitNet TL2 weight repacker.

This script reads a V31 Phase D `.fjm` v9 ternary-packed weight tensor
(2 bits per weight, 4 weights per byte LE, encoding `00=-1, 01=0,
10=+1`) and emits the **TL2 native** wire layout that
`ggml_qgemm_lut(GGML_BITNET_X86_TL2)` expects: a packed 4-bit
magnitude index stream + parallel 1-bit sign stream + secondary
9-entry index stream for the `K mod 3` tail.

References:
  - V31 source format: docs/FJQ_PHASE_D_FJM_V9_SPEC.md §5
  - TL2 target ABI: github.com/microsoft/BitNet
    include/ggml-bitnet.h (`ggml_qgemm_lut`, `ggml_preprocessor`)
  - TL2 codegen / canonical LUT: github.com/microsoft/BitNet
    utils/codegen_tl2.py — `three_lut_ctor`, `two_lut_ctor`
  - F.11 design: docs/FJQ_PHASE_F_F11_BITNETCPP_TL2_PORT_DESIGN.md v1.0

Per F.11 §3 (port plan) this is sub-task F.11.0. Run-once per
checkpoint at export time; output consumed by `ggml_qgemm_lut` via
the F.11.4 FFI binding.

The wire layout choices in this file (specifically: 1-bit-per-triplet
sign packing, byte-aligned, LE) match the algebraic intent of TL2's
codegen. F.11.1 (vendor microsoft/BitNet source) will verify against
the upstream `__attribute__((target("avx2")))` layout produced by
`utils/codegen_tl2.py` and patch any wire-level discrepancies. The
14-entry magnitude index LUT and the 9-entry Two_LUT are reproduced
bit-exact from the upstream codegen.

Apache-2.0 (fajarquant). Vendors NO upstream code; this is an encoder
spec written from scratch from the publicly-documented LUTs.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


# ---------------------------------------------------------------------------
# §1. V31 Phase D ternary 2-bit decoding
# ---------------------------------------------------------------------------

V31_PACK_DECODE: dict[int, int] = {
    0b00: -1,
    0b01: 0,
    0b10: +1,
    0b11: 0,  # reserved per spec; treat as 0 if encountered (producer MUST never emit)
}


def decode_v31_packed(packed: bytes | bytearray | np.ndarray, n_weights: int) -> np.ndarray:
    """Decode V31 Phase D 2-bit-packed ternary stream → int8 array.

    Per `FJQ_PHASE_D_FJM_V9_SPEC.md` §5: each byte holds 4 weights at
    bit offsets {0-1, 2-3, 4-5, 6-7} (little-endian). Values
    `{00=-1, 01=0, 10=+1, 11=reserved}`.

    Args:
        packed: byte stream of length `ceil(n_weights / 4)`.
        n_weights: number of ternary weights to decode.

    Returns:
        int8 ndarray of shape `(n_weights,)` with values in `{-1, 0, +1}`.
    """
    expected_bytes = (n_weights + 3) // 4
    buf = bytes(packed) if not isinstance(packed, bytes) else packed
    if len(buf) < expected_bytes:
        raise ValueError(
            f"V31 packed buffer too short: got {len(buf)} bytes, expected at least "
            f"{expected_bytes} for {n_weights} weights"
        )
    out = np.empty(n_weights, dtype=np.int8)
    for i in range(n_weights):
        byte = buf[i // 4]
        nibble_pos = (i % 4) * 2
        code = (byte >> nibble_pos) & 0b11
        out[i] = V31_PACK_DECODE[code]
    return out


def encode_v31_packed(weights: np.ndarray) -> bytes:
    """Inverse of :func:`decode_v31_packed` — used for round-trip tests.

    Args:
        weights: int8 ndarray of values in `{-1, 0, +1}`.

    Returns:
        V31-format byte stream of length `ceil(n / 4)`.
    """
    if weights.dtype != np.int8:
        weights = weights.astype(np.int8)
    encode_map = {-1: 0b00, 0: 0b01, 1: 0b10}
    n = weights.shape[0]
    out = bytearray((n + 3) // 4)
    for i in range(n):
        v = int(weights[i])
        if v not in encode_map:
            raise ValueError(f"weights[{i}] = {v} not in {{-1, 0, +1}}")
        code = encode_map[v]
        byte_idx = i // 4
        bit_pos = (i % 4) * 2
        out[byte_idx] |= code << bit_pos
    return bytes(out)


# ---------------------------------------------------------------------------
# §2. TL2 canonical magnitude-index LUT (3-weight triplets)
# ---------------------------------------------------------------------------

# Per `utils/codegen_tl2.py` `three_lut_ctor`: the 14 unsigned
# canonical (w0, w1, w2) magnitudes mapped to indices 0-13. Indices
# 14 and 15 are reserved (encode as 0). Sign of the leading nonzero
# is factored out into the parallel sign array.
THREE_LUT_CANONICAL_TO_IDX: dict[tuple[int, int, int], int] = {
    (0, 0, 0): 0,
    (0, 0, 1): 1,
    (0, 1, -1): 2,
    (0, 1, 0): 3,
    (0, 1, 1): 4,
    (1, -1, -1): 5,
    (1, -1, 0): 6,
    (1, -1, 1): 7,
    (1, 0, -1): 8,
    (1, 0, 0): 9,
    (1, 0, 1): 10,
    (1, 1, -1): 11,
    (1, 1, 0): 12,
    (1, 1, 1): 13,
}

# Inverse for round-trip: index → canonical triplet (sign-stripped).
THREE_LUT_IDX_TO_CANONICAL: dict[int, tuple[int, int, int]] = {
    v: k for k, v in THREE_LUT_CANONICAL_TO_IDX.items()
}


def encode_triplet(w0: int, w1: int, w2: int) -> tuple[int, int]:
    """Encode a (w0, w1, w2) ternary triplet into (magnitude_index, sign_bit).

    Sign convention: the sign of the **leading nonzero** weight is
    factored out. The magnitude index identifies the
    sign-canonicalized form (where the leading nonzero is +1).

    All-zero triplet → idx=0, sign=0.

    Returns:
        Tuple ``(magnitude_index, sign_bit)`` where
        ``magnitude_index ∈ [0, 13]`` and ``sign_bit ∈ {0, 1}``.
    """
    for v in (w0, w1, w2):
        if v not in (-1, 0, +1):
            raise ValueError(f"ternary weight {v!r} not in {{-1, 0, +1}}")

    # Find leading nonzero
    if w0 != 0:
        sign_factor = 1 if w0 > 0 else -1
    elif w1 != 0:
        sign_factor = 1 if w1 > 0 else -1
    elif w2 != 0:
        sign_factor = 1 if w2 > 0 else -1
    else:
        sign_factor = 1  # all-zero; sign immaterial

    canonical = (w0 * sign_factor, w1 * sign_factor, w2 * sign_factor)
    idx = THREE_LUT_CANONICAL_TO_IDX[canonical]
    sign = 0 if sign_factor == 1 else 1
    return idx, sign


def decode_triplet(idx: int, sign: int) -> tuple[int, int, int]:
    """Inverse of :func:`encode_triplet` for round-trip tests."""
    if idx not in THREE_LUT_IDX_TO_CANONICAL:
        if idx in (14, 15):
            return (0, 0, 0)  # reserved → all-zero per spec
        raise ValueError(f"magnitude index {idx} out of [0, 15]")
    if sign not in (0, 1):
        raise ValueError(f"sign bit {sign} not in {{0, 1}}")
    canonical = THREE_LUT_IDX_TO_CANONICAL[idx]
    factor = 1 if sign == 0 else -1
    return (canonical[0] * factor, canonical[1] * factor, canonical[2] * factor)


# ---------------------------------------------------------------------------
# §3. TL2 Two_LUT (K mod 3 tail; sign embedded in the 9-entry index)
# ---------------------------------------------------------------------------

# Per `utils/codegen_tl2.py` `two_lut_ctor`: a 9-entry LUT for 2-weight
# tails. Encodes sign DIRECTLY in the index (no parallel sign array).
TWO_LUT_TO_IDX: dict[tuple[int, int], int] = {
    (-1, -1): 0,
    (-1, 0): 1,
    (-1, 1): 2,
    (0, -1): 3,
    (0, 0): 4,
    (0, 1): 5,
    (1, -1): 6,
    (1, 0): 7,
    (1, 1): 8,
}
TWO_LUT_IDX_TO_PAIR: dict[int, tuple[int, int]] = {v: k for k, v in TWO_LUT_TO_IDX.items()}


def encode_pair(w0: int, w1: int) -> int:
    """Encode a 2-weight tail (w0, w1) → Two_LUT index ∈ [0, 8]."""
    for v in (w0, w1):
        if v not in (-1, 0, +1):
            raise ValueError(f"ternary weight {v!r} not in {{-1, 0, +1}}")
    return TWO_LUT_TO_IDX[(w0, w1)]


def decode_pair(idx: int) -> tuple[int, int]:
    """Inverse of :func:`encode_pair`."""
    if idx not in TWO_LUT_IDX_TO_PAIR:
        raise ValueError(f"Two_LUT index {idx} not in [0, 8]")
    return TWO_LUT_IDX_TO_PAIR[idx]


def encode_singleton(w: int) -> int:
    """K mod 3 = 1 case: encode single weight as (w, 0) in Two_LUT.

    Maps `-1 → 1`, `0 → 4`, `+1 → 7`. The decoder discards the
    second component.
    """
    return encode_pair(w, 0)


# ---------------------------------------------------------------------------
# §4. Wire-layout packing (magnitudes + signs + tail)
# ---------------------------------------------------------------------------

class TL2WireLayout(NamedTuple):
    """Packed TL2 wire format for one ternary weight matrix.

    Attributes:
        magnitudes: byte stream; 2 magnitude indices per byte (upper
            nibble = first triplet of the pair, lower nibble = second).
            Length = ceil(n_full_triplets / 2).
        signs: byte stream; 1 bit per triplet, packed 8 per byte LE
            (bit 0 = first triplet of the byte). Length =
            ceil(n_full_triplets / 8).
        tail: byte stream for the `K mod 3 ∈ {1, 2}` remainder.
            Two_LUT index packed 2 per byte (upper/lower nibbles),
            sign embedded in the index. Empty if `n_weights % 3 == 0`.
        n_weights: total ternary weights encoded (for round-trip
            verification).
        n_full_triplets: number of complete 3-weight triplets =
            n_weights // 3.
        tail_kind: ``"none"``, ``"singleton"``, or ``"pair"``.
    """

    magnitudes: bytes
    signs: bytes
    tail: bytes
    n_weights: int
    n_full_triplets: int
    tail_kind: str  # noqa: VNE003


def pack_tl2_wire(weights: np.ndarray) -> TL2WireLayout:
    """Encode a ternary weight array into TL2 wire format.

    See :class:`TL2WireLayout` for output schema.
    """
    if weights.dtype != np.int8:
        weights = weights.astype(np.int8)
    n = int(weights.shape[0])
    n_full_triplets = n // 3
    tail_len = n % 3

    # Magnitude buffer: one nibble per triplet, 2 triplets per byte
    mag_bytes = bytearray((n_full_triplets + 1) // 2)
    # Sign buffer: 1 bit per triplet, 8 triplets per byte LE
    sign_bytes = bytearray((n_full_triplets + 7) // 8)

    for t in range(n_full_triplets):
        w0, w1, w2 = (
            int(weights[3 * t]),
            int(weights[3 * t + 1]),
            int(weights[3 * t + 2]),
        )
        idx, sign = encode_triplet(w0, w1, w2)
        # Magnitude: upper nibble for even-position triplet, lower
        # nibble for odd-position. (Pair = bytes representing 2
        # triplets; first triplet of pair lives in upper nibble.)
        byte_idx = t // 2
        if t % 2 == 0:
            mag_bytes[byte_idx] |= (idx & 0xF) << 4
        else:
            mag_bytes[byte_idx] |= (idx & 0xF)
        # Sign: bit-position t % 8 within byte t // 8
        sign_byte_idx = t // 8
        sign_bit_pos = t % 8
        sign_bytes[sign_byte_idx] |= (sign & 0x1) << sign_bit_pos

    # Tail handling
    if tail_len == 0:
        tail_bytes = b""
        tail_kind = "none"
    elif tail_len == 1:
        w = int(weights[n - 1])
        idx = encode_singleton(w)
        tail_bytes = bytes([(idx & 0xF) << 4])
        tail_kind = "singleton"
    else:  # tail_len == 2
        w0, w1 = int(weights[n - 2]), int(weights[n - 1])
        idx = encode_pair(w0, w1)
        tail_bytes = bytes([(idx & 0xF) << 4])
        tail_kind = "pair"

    return TL2WireLayout(
        magnitudes=bytes(mag_bytes),
        signs=bytes(sign_bytes),
        tail=tail_bytes,
        n_weights=n,
        n_full_triplets=n_full_triplets,
        tail_kind=tail_kind,
    )


def unpack_tl2_wire(layout: TL2WireLayout) -> np.ndarray:
    """Inverse of :func:`pack_tl2_wire`. Used for round-trip tests."""
    out = np.zeros(layout.n_weights, dtype=np.int8)
    for t in range(layout.n_full_triplets):
        byte_idx = t // 2
        if t % 2 == 0:
            idx = (layout.magnitudes[byte_idx] >> 4) & 0xF
        else:
            idx = layout.magnitudes[byte_idx] & 0xF
        sign_byte_idx = t // 8
        sign_bit_pos = t % 8
        sign = (layout.signs[sign_byte_idx] >> sign_bit_pos) & 0x1
        w0, w1, w2 = decode_triplet(idx, sign)
        out[3 * t] = w0
        out[3 * t + 1] = w1
        out[3 * t + 2] = w2

    if layout.tail_kind == "singleton":
        idx = (layout.tail[0] >> 4) & 0xF
        w0, _ = decode_pair(idx)  # discards spare component
        out[layout.n_weights - 1] = w0
    elif layout.tail_kind == "pair":
        idx = (layout.tail[0] >> 4) & 0xF
        w0, w1 = decode_pair(idx)
        out[layout.n_weights - 2] = w0
        out[layout.n_weights - 1] = w1

    return out


# ---------------------------------------------------------------------------
# §5. Top-level repack: V31 .fjm → TL2 wire layout
# ---------------------------------------------------------------------------

def repack_v31_to_tl2(packed_v31: bytes, n_weights: int) -> TL2WireLayout:
    """Top-level F.11.0 entry point: V31 bytes → TL2 wire layout.

    Args:
        packed_v31: raw byte stream from `.fjm` v9 weight block,
            length `ceil(n_weights / 4)`.
        n_weights: total ternary weights (M × K for an M-by-K matrix).

    Returns:
        TL2WireLayout with magnitudes + signs + tail buffers.
    """
    decoded = decode_v31_packed(packed_v31, n_weights)
    return pack_tl2_wire(decoded)


# ---------------------------------------------------------------------------
# §6. Storage size accounting (for design-doc verification)
# ---------------------------------------------------------------------------

def tl2_storage_bytes(n_weights: int) -> dict[str, int]:
    """Compute TL2 storage breakdown for a ternary weight matrix.

    Useful as a sanity gate vs §1.3 of the design doc which claims
    1.67 bpw on the wire (5 bits per 3-weight triplet).
    """
    n_full_triplets = n_weights // 3
    tail_len = n_weights % 3
    mag_bytes = (n_full_triplets + 1) // 2
    sign_bytes = (n_full_triplets + 7) // 8
    tail_bytes = 0 if tail_len == 0 else 1
    total = mag_bytes + sign_bytes + tail_bytes
    return {
        "magnitudes_bytes": mag_bytes,
        "signs_bytes": sign_bytes,
        "tail_bytes": tail_bytes,
        "total_bytes": total,
        "bits_per_weight": 8.0 * total / max(1, n_weights),
    }


# ---------------------------------------------------------------------------
# §7. Tile-organized encoder (V32-prep F.11.4 Branch X-real)
# ---------------------------------------------------------------------------
#
# Per `cpu_kernels/bitnet_tl2/bitnet-lut-kernels-tl2.h`
# `three_tbl_impl_<m>_<k>` inner loops, the AVX2 kernel reads:
#
#     vec_as[ai] = a[i * KK / 2 + ai * 32]    for ai ∈ [0, KK/2)
#     vec_signs[as] = sign[i * KK / 8 + as * 32]  for as ∈ [0, KK/8)
#
# where KK = BBK / 3 and i ∈ {0, 32} (two i-groups for BM=64).
#
# This means the magnitude buffer A and sign buffer are NOT
# row-major; they're organized by (output-tile, k-outer block,
# i-group, ai-vector, lane). The encoder below emits them in
# this layout so a single `fjq_tl2_qgemm_lut(A, sign, ...)` call
# Just Works on real ternary data.
#
# Tile geometry per F.11.4 Branch X v1.1 (`e038acf`):
#
#     tile_idx          = output_row // BM            (e.g., 4 tiles for m=256, BM=64)
#     row_in_tile       = output_row %  BM
#     i_group           = row_in_tile // 32
#     lane              = row_in_tile %  32
#
#     k_outer           = triplet_col_idx // KK       (KK=42 for BBK=128)
#     triplet_in_kouter = triplet_col_idx %  KK
#     ai                = triplet_in_kouter // 2
#     nibble_in_byte    = triplet_in_kouter %  2      (0=upper, 1=lower)
#
# Per (tile, k_outer) sub-block, A buffer length:
#     2 i-groups × 21 ai-vectors × 32 bytes   = 1344 bytes
# Per matrix: tile_count × kouter_count × 1344 bytes.
#
# Sign buffer per (tile, k_outer) sub-block:
#     2 i-groups × 5  as-vectors × 32 bytes   = 320  bytes
#
# **CURRENT STATUS: SCAFFOLD ONLY.** The byte-count math is
# verified by §7-tests; the byte CONTENT logic is TODO. F.11.4
# Branch X-real next iteration will populate the per-byte
# magnitude+sign emission per the indexing rule above.

class TL2TileLayout(NamedTuple):
    """Tile-organized TL2 wire layout produced by
    :func:`pack_tl2_tile_organized`. Output dtype is bytes; the
    kernel reads via uint8_t* so endianness is not relevant.

    Attributes:
        magnitudes: A buffer (4-bit magnitude indices, 2 per byte
            via upper/lower nibbles), tile-organized.
        signs: parallel sign buffer (1 bit per triplet, packed).
        m, k: matrix dimensions (rows × cols).
        bm: row-tile dimension (typically 64 for Mini shapes).
        bbk: inner k-block dimension (typically 128 for Mini shapes).
        n_tiles: m // bm.
        n_kouter: K // BBK (number of inner k blocks per tile).
    """

    magnitudes: bytes
    signs: bytes
    m: int
    k: int
    bm: int
    bbk: int
    n_tiles: int
    n_kouter: int


def tl2_tile_layout_sizes(m: int, k: int, bm: int = 64, bbk: int = 128) -> dict[str, int]:
    """Compute byte counts for the tile-organized layout.

    Per the kernel stride formulas at the top of §7:

        kk            = bbk // 3
        a_per_kouter  = 2 * (kk // 2) * 32      (2 i-groups; ceiling for KK/2)
        s_per_kouter  = 2 * (kk // 8) * 32      (2 i-groups; floor for KK/8)
        a_total       = (m // bm) * (k // bbk) * a_per_kouter
        s_total       = (m // bm) * (k // bbk) * s_per_kouter

    For (m=256, k=256, BM=64, BBK=128) — the smallest Mini shape:
        kk            = 42
        a_per_kouter  = 2 * 21 * 32 = 1344
        s_per_kouter  = 2 * 5  * 32 = 320
        a_total       = 4 * 2 * 1344 = 10,752 bytes
        s_total       = 4 * 2 * 320  = 2,560 bytes

    Returns:
        Dict with keys ``a_bytes``, ``sign_bytes``, ``a_per_kouter``,
        ``sign_per_kouter``, ``n_tiles``, ``n_kouter``.

    Raises:
        ValueError: if m % bm != 0 or k % bbk != 0 (kernel preconditions).
    """
    if m % bm != 0:
        raise ValueError(f"m={m} not divisible by BM={bm}")
    if k % bbk != 0:
        raise ValueError(f"k={k} not divisible by BBK={bbk}")
    kk = bbk // 3
    a_per_kouter = 2 * (kk // 2) * 32
    s_per_kouter = 2 * (kk // 8) * 32
    n_tiles = m // bm
    n_kouter = k // bbk
    return {
        "a_bytes": n_tiles * n_kouter * a_per_kouter,
        "sign_bytes": n_tiles * n_kouter * s_per_kouter,
        "a_per_kouter": a_per_kouter,
        "sign_per_kouter": s_per_kouter,
        "n_tiles": n_tiles,
        "n_kouter": n_kouter,
    }


def pack_tl2_tile_organized(
    weights: np.ndarray,
    bm: int = 64,
    bbk: int = 128,
) -> TL2TileLayout:
    """Pack ternary weights into TL2 tile-organized wire format.

    F.11.4 Branch X-real iteration 2: populates the magnitude
    byte stream per the indexing rule documented at §7's docstring.
    Sign byte stream remains ZERO (deferred to iteration 3 — the
    in-byte sign-bit position requires unfolding the kernel's
    `_mm256_slli_epi16(vec_sign, 4*k+sub)` pattern, easier to
    debug against a real parity test that catches sign-flip
    failures).

    Algorithm: walk every full triplet `(w0, w1, w2)` at
    `(output_row, k_idx*3 .. k_idx*3+2)`; encode via
    :func:`encode_triplet` to get `(magnitude_index, sign_bit)`;
    place `magnitude_index` into the byte at the computed offset
    + nibble. K-mod-3 leftover (e.g., 256 mod 3 == 1) is dropped —
    the kernel's KK = BBK/3 truncation matches this behavior.

    Args:
        weights: int8 ndarray, shape ``(m, k)``, values in
            ``{-1, 0, +1}``.
        bm: row-tile dim (default 64 — matches all 4 Mini shapes).
        bbk: inner k-block dim (default 128 — same).

    Returns:
        :class:`TL2TileLayout` with magnitude bytes populated +
        zero sign bytes. Use the parity test in
        `src/cpu_kernels/tl2.rs` (iteration 3) to validate.
    """
    if weights.ndim != 2:
        raise ValueError(f"weights must be 2-D (m, k); got shape {weights.shape}")
    m, k = int(weights.shape[0]), int(weights.shape[1])
    sizes = tl2_tile_layout_sizes(m, k, bm, bbk)
    kk = bbk // 3
    a_per_kouter = sizes["a_per_kouter"]      # bytes per (tile, k_outer) sub-block
    s_per_kouter = sizes["sign_per_kouter"]
    a_per_tile = sizes["n_kouter"] * a_per_kouter
    s_per_tile = sizes["n_kouter"] * s_per_kouter

    a_buf = bytearray(sizes["a_bytes"])
    s_buf = bytearray(sizes["sign_bytes"])  # zero — sign deferred to iter 3

    n_triplets_per_kouter = kk            # 42 for BBK=128
    n_kouter = sizes["n_kouter"]

    for output_row in range(m):
        tile_idx = output_row // bm
        row_in_tile = output_row % bm
        i_group = row_in_tile // 32
        lane = row_in_tile % 32

        for k_outer in range(n_kouter):
            for triplet_in_kouter in range(n_triplets_per_kouter):
                # Source-side k coordinate: where we read 3 weights from
                k_base = k_outer * bbk + triplet_in_kouter * 3
                if k_base + 2 >= k:
                    # K-mod-3 tail — kernel truncates KK = BBK/3 too,
                    # so this triplet is also implicitly dropped.
                    continue
                w0 = int(weights[output_row, k_base])
                w1 = int(weights[output_row, k_base + 1])
                w2 = int(weights[output_row, k_base + 2])
                mag_idx, sign_bit = encode_triplet(w0, w1, w2)

                ai = triplet_in_kouter // 2
                nibble_in_byte = triplet_in_kouter % 2

                # F.11.4 Path B step 4: kernel unpacklo/unpackhi
                # crosses AVX2 lanes for bytes 8..23. Encoder must
                # place row r's mag at the input byte position the
                # kernel will route to output row r. Empirical sweep
                # (Rust harness derive_byte_to_row_mapping) confirmed:
                #   row 0..7   ↔ input byte 0..7
                #   row 8..15  ↔ input byte 16..23   (lane-crossed)
                #   row 16..23 ↔ input byte 8..15    (lane-crossed)
                #   row 24..31 ↔ input byte 24..31
                if lane < 8:
                    byte_in_vec = lane
                elif lane < 16:
                    byte_in_vec = lane + 8
                elif lane < 24:
                    byte_in_vec = lane - 8
                else:
                    byte_in_vec = lane

                byte_offset = (
                    tile_idx * a_per_tile
                    + k_outer * a_per_kouter
                    + i_group * (a_per_kouter // 2)
                    + ai * 32
                    + byte_in_vec
                )
                # Per the kernel: vec_v_top = (a >> 4) & 0xF
                #                  vec_v_bot = a & 0xF
                # So upper nibble = "top" path = first triplet of pair (even index)
                #    lower nibble = "bot" path = second triplet of pair (odd index)
                if nibble_in_byte == 0:
                    a_buf[byte_offset] |= (mag_idx & 0xF) << 4
                else:
                    a_buf[byte_offset] |= (mag_idx & 0xF)

                # Iteration 6 sign packing — DERIVED from kernel
                # `slli_epi16(vec_sign, 4*k+sub) → srai_epi16(_, 15)`
                # bit-extraction pattern, refining iteration 4 guess.
                #
                # Per outer-k iteration (= as_idx), kernel processes 4
                # ai's × 2 nibbles = 8 triplets per lane. Within one
                # 16-bit lane of `vec_signs[as_idx]`, bit positions are:
                #
                #   For ai (= triplet_in_as // 2) ∈ {0..3}, kernel uses
                #   bits {15, 14, 13, 12} - 4*ai. Of these:
                #     - bit 15-4*ai = top_hi sign (top nibble, rows 0-15)
                #     - bit 14-4*ai = top_lo sign (top nibble, rows 16-31)
                #     - bit 13-4*ai = bot_hi sign (bot nibble, rows 0-15)
                #     - bit 12-4*ai = bot_lo sign (bot nibble, rows 16-31)
                #
                # Compact:
                #   bit_pos_in_lane = 15 - 4*ai - 2*nibble - row_hi_lo
                #   where row_hi_lo = 0 if (row_in_i_group < 16) else 1
                #
                # 16-bit lane is `lane % 16` (within i_group's 32 rows,
                # rows 0-15 share lanes 0-15 with rows 16-31; the lo/hi
                # split is encoded in bit_pos_in_lane via row_hi_lo).
                #
                # 16-bit lane → 2 bytes in vec_signs. Byte = bit_pos // 8;
                # bit_in_byte = bit_pos % 8. So:
                #   byte_offset = i_group_s_off + (lane % 16) * 2
                #                                + (bit_pos_in_lane // 8)
                #   bit_set     = bit_pos_in_lane % 8
                if sign_bit:
                    as_idx = triplet_in_kouter // 8
                    triplet_in_as = triplet_in_kouter % 8
                    if as_idx < (kk // 8):
                        ai_in_as = triplet_in_as // 2
                        nibble_for_sign = triplet_in_as % 2  # 0=top, 1=bot
                        row_hi_lo = 1 if lane >= 16 else 0
                        bit_pos_in_lane = 15 - 4 * ai_in_as - 2 * nibble_for_sign - row_hi_lo

                        byte_in_lane_pair = bit_pos_in_lane // 8  # 0=low, 1=high
                        bit_in_byte = bit_pos_in_lane % 8
                        lane_in_16 = lane % 16  # rows 0-15 and 16-31 share

                        sign_byte_off = (
                            tile_idx * s_per_tile
                            + k_outer * s_per_kouter
                            + i_group * (s_per_kouter // 2)
                            + as_idx * 32
                            + lane_in_16 * 2
                            + byte_in_lane_pair
                        )
                        s_buf[sign_byte_off] |= (1 << bit_in_byte)

    return TL2TileLayout(
        magnitudes=bytes(a_buf),
        signs=bytes(s_buf),
        m=m,
        k=k,
        bm=bm,
        bbk=bbk,
        n_tiles=sizes["n_tiles"],
        n_kouter=sizes["n_kouter"],
    )


# ---------------------------------------------------------------------------
# §8. CLI driver — repack a Mini-ckpt MLP weight tensor (smoke validation)
# ---------------------------------------------------------------------------

def _main() -> int:  # pragma: no cover  (CLI entry; tests cover library API)
    import argparse
    import json
    import sys
    from pathlib import Path

    p = argparse.ArgumentParser(
        description="V32-prep F.11.0: repack V31 Phase D ternary → TL2 wire format. "
        "Reads a raw 2-bit-packed weight buffer from disk, emits magnitudes + signs "
        "+ tail buffers + a JSON manifest with storage stats.",
    )
    p.add_argument("--in-bytes", type=Path, required=True,
                   help="Path to raw V31 2-bit packed weight buffer.")
    p.add_argument("--n-weights", type=int, required=True,
                   help="Number of ternary weights in the buffer (= M × K).")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output directory; writes magnitudes.bin / signs.bin / "
                   "tail.bin / manifest.json.")
    args = p.parse_args()

    if not args.in_bytes.exists():
        print(f"[error] input not found: {args.in_bytes}", file=sys.stderr)
        return 2
    raw = args.in_bytes.read_bytes()
    layout = repack_v31_to_tl2(raw, args.n_weights)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "magnitudes.bin").write_bytes(layout.magnitudes)
    (args.out_dir / "signs.bin").write_bytes(layout.signs)
    (args.out_dir / "tail.bin").write_bytes(layout.tail)
    manifest = {
        "n_weights": layout.n_weights,
        "n_full_triplets": layout.n_full_triplets,
        "tail_kind": layout.tail_kind,
        "magnitudes_bytes": len(layout.magnitudes),
        "signs_bytes": len(layout.signs),
        "tail_bytes": len(layout.tail),
        "storage": tl2_storage_bytes(layout.n_weights),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[F.11.0] repacked {layout.n_weights} weights → {args.out_dir}")
    print(f"         storage: {manifest['storage']['bits_per_weight']:.2f} bpw "
          f"(target ~1.67 bpw per F.11 design §1.5)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
