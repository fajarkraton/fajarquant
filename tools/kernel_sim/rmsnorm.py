"""Python port of `km_rmsnorm` (kernel/compute/kmatrix.fj:558) and the
`km_isqrt` Newton-Raphson integer square root it depends on.

Algorithm (V28.2 "max-abs rescaling" variant):

    rms is scale-invariant:  rms(α·x) == α·rms(x)
    Rescale x to [-K, K]:    x_rs = x × K / max|x|
    RSS:                     rss = Σ (x_rs² / dim)
    RMS in rescaled space:   rms_rs = sqrt(rss + 1)
    Normalized (×1000 fp):   normed = x_rs × 1000 / rms_rs
    Apply gamma:             y = normed × γ / 1000           (Llama, gamma_mode=1)
                         or  y = normed × (1000 + γ) / 1000  (Gemma, gamma_mode=2)

K is a fixed scaling constant (10000 in the kernel). This keeps all
intermediate products safely under i64 limits even for dim=8192.

Every op goes through int_ops for bit-exact kernel parity.
"""

from __future__ import annotations

from . import int_ops

K_SCALE = 10000

GAMMA_MODE_NONE = 0     # no gamma; just normed
GAMMA_MODE_LLAMA = 1    # y = normed * g / 1000
GAMMA_MODE_GEMMA = 2    # y = normed * (1000 + g) / 1000


def km_isqrt(x: int) -> int:
    """Integer square root via Newton's method, matching kernel
    `km_isqrt` (kernel/compute/kmatrix.fj:335).

    Returns floor(sqrt(x)) for x > 0, and 1 for x <= 1 (kernel
    guard; avoids div-by-zero in caller).
    """
    if x <= 0:
        return 1
    if x == 1:
        return 1
    guess = int_ops.trunc_div_i64(x, 2)
    if guess == 0:
        return 1
    prev = int_ops.add_i64(guess, 1)
    # Kernel loop: `while guess < prev { prev = guess;
    #                                    guess = (guess + x/guess) / 2 }`
    while guess < prev:
        prev = guess
        guess = int_ops.trunc_div_i64(
            int_ops.add_i64(guess, int_ops.trunc_div_i64(x, guess)),
            2,
        )
    return 1 if prev == 0 else prev


def rmsnorm(
    data,
    gamma=None,
    gamma_mode: int = GAMMA_MODE_NONE,
    k_scale: int = K_SCALE,
):
    """Apply RMSNorm in place-semantically (returns a new list).

    Parameters
    ----------
    data : sequence of int
        Input vector (i64, typically fp×1000). dim = len(data).
    gamma : sequence of int or None
        Per-element gamma weights. If None, no gamma is applied.
    gamma_mode : int
        - 0 (`GAMMA_MODE_NONE`): ignore gamma (treats as identity).
        - 1 (`GAMMA_MODE_LLAMA`): y = normed × γ / 1000.
        - 2 (`GAMMA_MODE_GEMMA`): y = normed × (1000 + γ) / 1000.
    k_scale : int
        Rescaling constant. Kernel uses 10000; exposed for testing.

    Returns
    -------
    list[int]
        Normalized output, same length as `data`.

    Notes
    -----
    - If dim <= 0 or dim > 8192, the kernel early-returns without
      writing. This port raises for dim <= 0 (Python guard) and
      mirrors the dim > 8192 early return by returning a copy of
      the input unchanged.
    - If `max|x|` is 0 (all-zero input), the kernel early-returns
      without writing. This port returns a copy of the input.
    """
    dim = len(data)
    if dim <= 0:
        raise ValueError("rmsnorm: dim must be > 0")
    if dim > 8192:
        # Mirror kernel early-return: do not normalize, return unchanged
        return list(data)
    if gamma is not None and len(gamma) != dim:
        raise ValueError(
            f"rmsnorm: gamma length {len(gamma)} != dim {dim}"
        )

    # Pass 1: max|x|
    max_abs = 0
    for i in range(dim):
        x = data[i]
        ax = int_ops.abs_i64(x)
        if ax > max_abs:
            max_abs = ax
    if max_abs <= 0:
        # All-zero input — kernel writes nothing; we return input unchanged
        return list(data)

    # Pass 2: rss = Σ (x_rs² / dim), where x_rs = x × k_scale / max_abs
    rss = 0
    for i in range(dim):
        x = data[i]
        x_rs = int_ops.trunc_div_i64(
            int_ops.mul_i64(x, k_scale), max_abs
        )
        rss = int_ops.add_i64(
            rss,
            int_ops.trunc_div_i64(int_ops.mul_i64(x_rs, x_rs), dim),
        )

    rms_rs = km_isqrt(int_ops.add_i64(rss, 1))
    if rms_rs <= 0:
        return list(data)

    # Pass 3: normalize + gamma
    out = [0] * dim
    for i in range(dim):
        x = data[i]
        x_rs = int_ops.trunc_div_i64(
            int_ops.mul_i64(x, k_scale), max_abs
        )
        normed = int_ops.trunc_div_i64(
            int_ops.mul_i64(x_rs, 1000), rms_rs
        )
        if gamma is None or gamma_mode == GAMMA_MODE_NONE:
            out[i] = normed
        elif gamma_mode == GAMMA_MODE_LLAMA:
            out[i] = int_ops.trunc_div_i64(
                int_ops.mul_i64(normed, gamma[i]), 1000
            )
        elif gamma_mode == GAMMA_MODE_GEMMA:
            out[i] = int_ops.trunc_div_i64(
                int_ops.mul_i64(normed, int_ops.add_i64(1000, gamma[i])),
                1000,
            )
        else:
            raise ValueError(
                f"rmsnorm: unknown gamma_mode {gamma_mode}; "
                "expected 0, 1, or 2"
            )

    return out
