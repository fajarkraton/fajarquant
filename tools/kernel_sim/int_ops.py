"""Bit-exact int64 primitives matching FajarOS kernel LLVM lowering.

Every op here is a Python mirror of a single operation in
`kernel/compute/kmatrix.fj` / `model_loader.fj` / `transformer.fj`. The
goal is 0-ULP parity between this module and the kernel for all
inputs. Tests in `tests/sim/test_int_ops_parity.py` enforce that.

Design rules:
- All arithmetic uses Python int internally for correctness, then
  `_wrap_i64` clamps to 2's-complement int64 to match LLVM defined wrap.
- Division is truncated toward zero (LLVM `sdiv`), NOT Python `//`
  which rounds toward -infinity.
- Right shift is arithmetic for signed values (`ashr`), sign-extending.
  Unsigned right shift (`lshr`) is also provided for bit-level work.
- Reads from memory (u8, u32, u64) widen to i64 by zero-extend, matching
  Fajar Lang's `volatile_read_u*` builtins which return i64.
"""

from __future__ import annotations

INT8_MIN = -(1 << 7)
INT8_MAX = (1 << 7) - 1
INT16_MIN = -(1 << 15)
INT16_MAX = (1 << 15) - 1
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1

_UINT64_MOD = 1 << 64
_UINT32_MASK = (1 << 32) - 1
_UINT8_MASK = 0xFF


def _wrap_i64(x: int) -> int:
    """Wrap a Python int into int64 two's-complement range."""
    x &= _UINT64_MOD - 1
    if x >= (1 << 63):
        x -= _UINT64_MOD
    return x


# ── Widening reads (zero-extend, kernel convention) ───────────────────

def widen_u8(byte) -> int:
    """u8 → i64 zero-extend. Accepts any int-like (incl. numpy scalars);
    only low 8 bits used."""
    return int(byte) & _UINT8_MASK


def widen_u32(word) -> int:
    """u32 → i64 zero-extend. Accepts any int-like (incl. numpy scalars);
    only low 32 bits used."""
    return int(word) & _UINT32_MASK


# ── Arithmetic (defined wrap at i64) ──────────────────────────────────

def add_i64(a: int, b: int) -> int:
    return _wrap_i64(a + b)


def sub_i64(a: int, b: int) -> int:
    return _wrap_i64(a - b)


def mul_i64(a: int, b: int) -> int:
    return _wrap_i64(a * b)


def trunc_div_i64(a: int, b: int) -> int:
    """LLVM `sdiv`: truncation toward zero. Undefined for b == 0.

    Differs from Python `//` for mixed-sign operands:
        trunc_div_i64(-7, 2) == -3     (Python -7 // 2 == -4)
        trunc_div_i64(7, -2) == -3     (Python  7 // -2 == -4)
    """
    if b == 0:
        raise ZeroDivisionError("trunc_div_i64: divide by zero (UB in LLVM)")
    q = abs(a) // abs(b)
    if (a < 0) ^ (b < 0):
        q = -q
    return _wrap_i64(q)


# ── Bit ops ───────────────────────────────────────────────────────────

def ashr_i64(x: int, n: int) -> int:
    """Arithmetic shift right (sign-extending).

    Matches LLVM `ashr`. For x < 0, high bits fill with 1. Python's `>>`
    already behaves this way for int, but we wrap the result to i64
    range to ensure parity when feeding back into mul/add.
    """
    if n < 0 or n >= 64:
        # Undefined per LLVM; kernel code never does this. Mirror Python.
        return _wrap_i64(x >> n if n > 0 else x)
    return _wrap_i64(x >> n)


def lshr_i64(x: int, n: int) -> int:
    """Logical shift right (zero-fill). Matches LLVM `lshr` on i64."""
    if n < 0 or n >= 64:
        return 0
    x_u = x & (_UINT64_MOD - 1)  # reinterpret as u64
    return _wrap_i64(x_u >> n)


def shl_i64(x: int, n: int) -> int:
    """Left shift with i64 wrap. Matches LLVM `shl`."""
    if n < 0 or n >= 64:
        return 0
    return _wrap_i64(x << n)


def bitand_i64(a: int, b: int) -> int:
    return _wrap_i64(a & b)


def bitor_i64(a: int, b: int) -> int:
    return _wrap_i64(a | b)


def bitxor_i64(a: int, b: int) -> int:
    return _wrap_i64(a ^ b)


# ── Utility ───────────────────────────────────────────────────────────

def clamp_i64(x: int, lo: int, hi: int) -> int:
    """Clamp; not part of kernel core but useful for i32 / i16 simulation."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def abs_i64(x: int) -> int:
    """Branchless-style absolute value matching kernel pattern
    `if x < 0 { 0 - x } else { x }`. At INT64_MIN this wraps to INT64_MIN
    (same behavior as kernel; documented UB-adjacent case)."""
    return _wrap_i64(-x) if x < 0 else _wrap_i64(x)


# ── Memory read mirrors ───────────────────────────────────────────────
# These accept a bytes-like buffer + offset and return i64 per kernel
# volatile_read_u* semantics. For the simulator, "memory" is just a
# numpy.uint8 array or `bytes`.

def volatile_read_u8(buf, offset: int) -> int:
    """Read 1 byte from buf[offset], zero-extend to i64."""
    return widen_u8(buf[offset])


def volatile_read_u32(buf, offset: int) -> int:
    """Read 4 little-endian bytes from buf[offset:offset+4], zero-extend to i64."""
    b0 = buf[offset] & 0xFF
    b1 = buf[offset + 1] & 0xFF
    b2 = buf[offset + 2] & 0xFF
    b3 = buf[offset + 3] & 0xFF
    val = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    return widen_u32(val)


def volatile_read_u64(buf, offset: int) -> int:
    """Read 8 little-endian bytes from buf[offset:offset+8] as i64.

    Unlike u8/u32 this is NOT zero-extend — it's a bit-cast to i64, so a
    stored negative value (e.g., post-rmsnorm `normed = -5000`) round-trips.
    """
    val = 0
    for i in range(8):
        val |= (buf[offset + i] & 0xFF) << (8 * i)
    return _wrap_i64(val)


def read_nibble(buf, flat_idx: int) -> int:
    """Read the 4-bit nibble at flat_idx into an i64 in [0, 15].

    Matches the kernel pattern:
        byte_idx = flat_idx >> 1
        bit_off  = (flat_idx & 1) * 4
        raw      = volatile_read_u8(base + byte_idx)
        q        = (raw >> bit_off) & 15
    """
    byte_idx = ashr_i64(flat_idx, 1)
    bit_off = mul_i64(bitand_i64(flat_idx, 1), 4)
    raw = widen_u8(buf[byte_idx])
    return bitand_i64(ashr_i64(raw, bit_off), 15)
