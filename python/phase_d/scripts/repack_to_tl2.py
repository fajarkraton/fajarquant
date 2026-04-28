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
# §7. CLI driver — repack a Mini-ckpt MLP weight tensor (smoke validation)
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
