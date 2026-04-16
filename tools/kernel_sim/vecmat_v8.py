"""Python port of `km_vecmat_packed_v8` (kernel/compute/kmatrix.fj:823).

Group-wise 4-bit quantized vector-matrix multiply. For each output
column j in [0, n):

    sum = 0
    for k in [0, m):
        flat_idx = k * n + j
        q        = nibble at flat_idx in packed weights (low/high per parity)
        g        = flat_idx >> 7                 # group_size=128
        scale    = u32 scales[g]
        zero     = u8  zeros[g]
        w_x_1M   = (q - zero) * scale            # weight × 1_000_000
        v        = x[k]                          # i64 fp×1000
        sum     += (v * w_x_1M) / V8_SCALE_FP    # trunc div
    out[j] = sum

Every arithmetic op goes through `int_ops` so the result is byte-exact
with the kernel's LLVM lowering. Input/output are i64 values (lists or
numpy arrays); memory is modeled as bytes-likes.
"""

from __future__ import annotations

from . import int_ops

V8_GROUP_SIZE = 128
V8_GROUP_SHIFT = 7
V8_SCALE_FP = 1_000_000


def vecmat_v8(
    x_vec,
    packed,
    scales,
    zeros,
    m: int,
    n: int,
):
    """Pure-int bit-exact vecmat matching kernel formula.

    Parameters
    ----------
    x_vec : sequence of int
        Input vector, length `m`. Each element is an i64 (typically
        a fixed-point ×1000 value coming out of an rmsnorm).
    packed : bytes-like
        Packed 4-bit weights, length = m*n/2 bytes. Low nibble at
        index 2k, high nibble at index 2k+1.
    scales : sequence of int
        Per-group u32 scales, length = n_groups = ceil(m*n / 128).
        Each scale is already zero-extended to i64.
    zeros : bytes-like
        Per-group u8 zero-points, length = n_groups.
    m, n : int
        Input and output dimensions. Must satisfy m*n % 2 == 0.

    Returns
    -------
    list[int]
        Length-n output vector, each element an i64.
    """
    if m * n % 2 != 0:
        raise ValueError(
            f"vecmat_v8 requires m*n even (got m={m}, n={n}); "
            "4-bit packing needs full bytes"
        )
    total = m * n
    n_groups = (total + V8_GROUP_SIZE - 1) // V8_GROUP_SIZE
    if len(scales) < n_groups:
        raise ValueError(
            f"scales too short: got {len(scales)}, need {n_groups}"
        )
    if len(zeros) < n_groups:
        raise ValueError(
            f"zeros too short: got {len(zeros)}, need {n_groups}"
        )

    out = [0] * n

    for j in range(n):
        sum_acc = 0  # i64
        for k in range(m):
            flat_idx = int_ops.add_i64(int_ops.mul_i64(k, n), j)

            # byte_idx = flat_idx >> 1 ; bit_off = (flat_idx & 1) * 4
            # q = (raw_byte >> bit_off) & 15
            q = int_ops.read_nibble(packed, flat_idx)

            # g = flat_idx >> V8_GROUP_SHIFT   (== flat_idx / 128)
            g = int_ops.ashr_i64(flat_idx, V8_GROUP_SHIFT)

            # scale: u32 widened to i64 (caller already provides i64, but
            # mask defensively to mirror volatile_read_u32 zero-extend)
            scale = int_ops.widen_u32(scales[g])
            zero = int_ops.widen_u8(zeros[g])

            # w_x_1M = (q - zero) * scale
            w_x_1M = int_ops.mul_i64(int_ops.sub_i64(q, zero), scale)

            v = x_vec[k]

            # sum += (v * w_x_1M) / V8_SCALE_FP (trunc toward 0)
            term = int_ops.trunc_div_i64(
                int_ops.mul_i64(v, w_x_1M), V8_SCALE_FP
            )
            sum_acc = int_ops.add_i64(sum_acc, term)

        out[j] = sum_acc

    return out


# ── Helpers for constructing packed-weight buffers ──────────────────

def pack_nibbles(nibbles):
    """Pack a sequence of 4-bit values (each in [0, 15]) into bytes.

    `nibbles[2k]` becomes low nibble of byte k; `nibbles[2k+1]` becomes
    high nibble. Length must be even.

    Useful for constructing test inputs that match the kernel's layout.
    """
    if len(nibbles) % 2 != 0:
        raise ValueError(f"pack_nibbles needs even length, got {len(nibbles)}")
    out = bytearray(len(nibbles) // 2)
    for i, q in enumerate(nibbles):
        if not 0 <= q <= 15:
            raise ValueError(f"nibble {i} out of range [0,15]: {q}")
        byte_idx = i >> 1
        bit_off = (i & 1) * 4
        out[byte_idx] |= (q & 0x0F) << bit_off
    return bytes(out)


def unpack_nibbles(packed, count):
    """Inverse of `pack_nibbles`. Returns list of `count` nibbles."""
    out = []
    for i in range(count):
        out.append(int_ops.read_nibble(packed, i))
    return out
