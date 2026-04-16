"""Python port of `mdl_ram_lmhead_argmax_v8_tied`
(kernel/compute/model_loader.fj:2154).

Tied LM head argmax. For Gemma 3 the input embedding matrix is re-used
as the LM head, so this iterates over the SAME group-wise 4-bit packed
weight buffer that embed lookup uses.

Per-vocab dot product:

    for v in [0, vocab_size):
        sum = 0
        for i in [0, d_model):
            flat_idx = v * d_model + i
            q        = nibble at flat_idx
            g        = flat_idx >> 7            # group_size = 128
            scale    = u32 scales[g]
            zero     = u8  zeros[g]
            w_x_1M   = (q - zero) * scale
            sum     += (x[i] * w_x_1M) / 1_000_000    # trunc div
        if sum > best_score:
            best_score = sum
            best_token = v

Tie-break: first-seen wins (`>` not `>=`). best_score sentinel starts at
-999_999_999 so any plausible dot product wins on the first iteration.

IMPORTANT difference from vecmat_v8 layout: argmax stores weights as
(vocab_size, d_model) row-major so `flat_idx = v * d_model + i`. This
is reversed from vecmat's (m, n) orientation where the inner loop index
is k and `flat_idx = k * n + j`. Same packing (low nibble at even flat
index, high nibble at odd), same group_size, same dequant formula.
"""

from __future__ import annotations

from . import int_ops
from .vecmat_v8 import V8_GROUP_SIZE, V8_GROUP_SHIFT, V8_SCALE_FP


# Sentinel matching kernel: `let mut best_score: i64 = -999999999`
ARGMAX_SENTINEL = -999_999_999


def argmax_v8(
    x_vec,
    packed,
    scales,
    zeros,
    vocab_size: int,
    d_model: int,
):
    """Return (best_token, best_score) after scanning all vocab rows.

    Parameters
    ----------
    x_vec : sequence of int
        Hidden state, length `d_model`. Each element i64 (post-final-rmsnorm,
        typically fp×1000).
    packed : bytes-like
        Packed 4-bit weights, length = vocab_size * d_model / 2.
        Row v of the weight matrix is at flat_idx v*d_model .. v*d_model+d_model-1.
    scales : sequence of int
        u32 per-group scales, length = ceil(vocab_size * d_model / 128).
    zeros : bytes-like
        u8 per-group zero-points, same length as scales.
    vocab_size : int
    d_model : int

    Returns
    -------
    (best_token, best_score) : (int, int)
        best_token ∈ [0, vocab_size); best_score is the i64 dot-product
        accumulator for that winning row.
    """
    total = vocab_size * d_model
    if total % 2 != 0:
        raise ValueError(
            f"argmax_v8 requires vocab_size*d_model even "
            f"(got {vocab_size}×{d_model}); 4-bit packing needs whole bytes"
        )
    if len(x_vec) != d_model:
        raise ValueError(
            f"x_vec length {len(x_vec)} != d_model {d_model}"
        )
    n_groups = (total + V8_GROUP_SIZE - 1) // V8_GROUP_SIZE
    if len(scales) < n_groups:
        raise ValueError(
            f"scales too short: got {len(scales)}, need {n_groups}"
        )
    if len(zeros) < n_groups:
        raise ValueError(
            f"zeros too short: got {len(zeros)}, need {n_groups}"
        )

    best_token = 0
    best_score = ARGMAX_SENTINEL

    for v in range(vocab_size):
        row_start = int_ops.mul_i64(v, d_model)
        sum_acc = 0
        for i in range(d_model):
            flat_idx = int_ops.add_i64(row_start, i)

            q = int_ops.read_nibble(packed, flat_idx)
            g = int_ops.ashr_i64(flat_idx, V8_GROUP_SHIFT)

            scale = int_ops.widen_u32(scales[g])
            zero = int_ops.widen_u8(zeros[g])

            w_x_1M = int_ops.mul_i64(int_ops.sub_i64(q, zero), scale)

            x_val = x_vec[i]
            term = int_ops.trunc_div_i64(
                int_ops.mul_i64(x_val, w_x_1M), V8_SCALE_FP
            )
            sum_acc = int_ops.add_i64(sum_acc, term)

        # Kernel: `if sum > best_score { ... }` — strict inequality
        # means first-seen wins on ties.
        if sum_acc > best_score:
            best_score = sum_acc
            best_token = v

    return best_token, best_score


def argmax_v8_full_logits(
    x_vec,
    packed,
    scales,
    zeros,
    vocab_size: int,
    d_model: int,
):
    """Same as argmax_v8 but also returns the full logit vector.

    Useful for P3 divergence analysis: dumping the top-K logit
    distribution + tie-break order reveals whether the kernel's
    pad-collapse is pad dominating (`logit[0]` is genuinely max) or
    near-uniform logits (tie-break to first == argmax=0 trivially).
    """
    total = vocab_size * d_model
    if total % 2 != 0:
        raise ValueError(
            f"argmax_v8 requires vocab_size*d_model even "
            f"(got {vocab_size}×{d_model})"
        )
    if len(x_vec) != d_model:
        raise ValueError(
            f"x_vec length {len(x_vec)} != d_model {d_model}"
        )
    n_groups = (total + V8_GROUP_SIZE - 1) // V8_GROUP_SIZE
    if len(scales) < n_groups or len(zeros) < n_groups:
        raise ValueError("scales/zeros too short")

    logits = [0] * vocab_size
    best_token = 0
    best_score = ARGMAX_SENTINEL

    for v in range(vocab_size):
        row_start = int_ops.mul_i64(v, d_model)
        sum_acc = 0
        for i in range(d_model):
            flat_idx = int_ops.add_i64(row_start, i)
            q = int_ops.read_nibble(packed, flat_idx)
            g = int_ops.ashr_i64(flat_idx, V8_GROUP_SHIFT)
            scale = int_ops.widen_u32(scales[g])
            zero = int_ops.widen_u8(zeros[g])
            w_x_1M = int_ops.mul_i64(int_ops.sub_i64(q, zero), scale)
            x_val = x_vec[i]
            term = int_ops.trunc_div_i64(
                int_ops.mul_i64(x_val, w_x_1M), V8_SCALE_FP
            )
            sum_acc = int_ops.add_i64(sum_acc, term)
        logits[v] = sum_acc
        if sum_acc > best_score:
            best_score = sum_acc
            best_token = v

    return best_token, best_score, logits
