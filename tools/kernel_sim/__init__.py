"""kernel_sim — bit-exact Python mirror of FajarOS kernel integer math.

Per V30.SIM plan: this package models FajarOS v8 quantization hot paths
(embedding lookup, group-wise vecmat, rmsnorm, argmax) using explicit
int64 arithmetic that matches the kernel's truncated-toward-zero division,
arithmetic right shift, and defined-wrap overflow.

Why not Python int or NumPy default? Python `//` rounds toward -inf, LLVM
`sdiv` truncates toward 0. For negative dividends these disagree. Also
Python int is unbounded; i64 wraps at 2^63. The `int_ops` module locks
dtype to `int64` and implements trunc-div + ashr explicitly.
"""

from .int_ops import (
    INT8_MIN, INT8_MAX,
    INT16_MIN, INT16_MAX,
    INT32_MIN, INT32_MAX,
    INT64_MIN, INT64_MAX,
    widen_u8, widen_u32,
    add_i64, sub_i64, mul_i64,
    trunc_div_i64, ashr_i64, lshr_i64, shl_i64,
    bitand_i64, bitor_i64, bitxor_i64,
    clamp_i64, abs_i64,
    volatile_read_u8, volatile_read_u32, volatile_read_u64,
    read_nibble,
)
from .vecmat_v8 import (
    V8_GROUP_SIZE, V8_GROUP_SHIFT, V8_SCALE_FP,
    vecmat_v8, pack_nibbles, unpack_nibbles,
    dequant_groupwise_v8_x1M,
)
from .rmsnorm import (
    K_SCALE,
    GAMMA_MODE_NONE, GAMMA_MODE_LLAMA, GAMMA_MODE_GEMMA,
    km_isqrt, rmsnorm,
)
from .argmax_v8 import (
    ARGMAX_SENTINEL, argmax_v8, argmax_v8_full_logits,
)
from .transformer import (
    MODEL_TYPE_LLAMA, MODEL_TYPE_SMOLLM,
    MODEL_TYPE_GEMMA3_1B, MODEL_TYPE_GEMMA3_270M,
    TransformerConfig, LayerWeights, ForwardResult,
    activation_identity, activation_relu,
    embed_lookup, simplified_attention_single_pos, gated_ffn,
    layer_forward, forward, forward_with_logits,
)
from .trace import (
    SCHEMA_VERSION, OP_NAMES, LAYER_SCOPED_OPS,
    FNV_OFFSET_64, FNV_PRIME_64, fnv1a_u64_bytes,
    TraceWriter, NoopTracer, NOOP_TRACER,
)

__all__ = [
    "INT8_MIN", "INT8_MAX",
    "INT16_MIN", "INT16_MAX",
    "INT32_MIN", "INT32_MAX",
    "INT64_MIN", "INT64_MAX",
    "widen_u8", "widen_u32",
    "add_i64", "sub_i64", "mul_i64",
    "trunc_div_i64", "ashr_i64", "lshr_i64", "shl_i64",
    "bitand_i64", "bitor_i64", "bitxor_i64",
    "clamp_i64", "abs_i64",
    "volatile_read_u8", "volatile_read_u32", "volatile_read_u64",
    "read_nibble",
    "V8_GROUP_SIZE", "V8_GROUP_SHIFT", "V8_SCALE_FP",
    "vecmat_v8", "pack_nibbles", "unpack_nibbles",
    "dequant_groupwise_v8_x1M",
    "K_SCALE",
    "GAMMA_MODE_NONE", "GAMMA_MODE_LLAMA", "GAMMA_MODE_GEMMA",
    "km_isqrt", "rmsnorm",
    "ARGMAX_SENTINEL", "argmax_v8", "argmax_v8_full_logits",
    "MODEL_TYPE_LLAMA", "MODEL_TYPE_SMOLLM",
    "MODEL_TYPE_GEMMA3_1B", "MODEL_TYPE_GEMMA3_270M",
    "TransformerConfig", "LayerWeights", "ForwardResult",
    "activation_identity", "activation_relu",
    "embed_lookup", "simplified_attention_single_pos", "gated_ffn",
    "layer_forward", "forward", "forward_with_logits",
    "SCHEMA_VERSION", "OP_NAMES", "LAYER_SCOPED_OPS",
    "FNV_OFFSET_64", "FNV_PRIME_64", "fnv1a_u64_bytes",
    "TraceWriter", "NoopTracer", "NOOP_TRACER",
]
