"""Forward-pass orchestration for the FajarOS v8 kernel path.

Composes the P1 primitives (embed lookup, group-wise vecmat, rmsnorm,
argmax) into a full prefill → argmax cycle. Mirrors the layer
structure of `kernel/compute/transformer.fj::tfm_layer_stream` and
`tfm_forward_stream`.

── P2.1 scope + deliberate simplifications ─────────────────────────

The plan §P2.1 gate is "Python forward runs on 'hello' prompt to
completion without error". Pure Python cannot reasonably execute real
Gemma 3 1B (262144 vocab × 1152 d_model × 26 layers ≈ 5 minutes per
token). P2.1 ships the orchestration shell + toy-config test; P2.2–P2.5
build on this:

  * Attention: simplified to single-position case
    `attn(x) = W_O · (W_V · x)`. For prefill of a 1-token prompt the
    softmax weights over a length-1 sequence are trivially [1.0], so
    the full multi-head attention collapses to W_O·W_V·x. This gives
    a structurally correct forward pass for structural testing
    without implementing KV cache, RoPE, GQA, or softmax — which P2.3
    kernel FJTRACE capture will re-introduce honestly.
  * Q norm / K norm / RoPE / GQA: skipped (not structurally required
    for argmax at single position with identity attention).
  * FFN: gated with optional activation fn; default is identity (no
    nonlinearity) so toy-weights tests are deterministic. Real kernel
    path uses GELU tanh approximation (`km_tanh_approx`).
  * Dequant policy: per-matrix vecmat via kernel_sim.vecmat_v8
    bit-exactly (0-ULP kernel parity already established in P1.2).

The P2.1 test `test_e2e_hello_prompt_runs` runs a tiny synthetic
config (vocab=32, d_model=4, n_heads=1, n_layers=2) so a full forward
completes in under a second.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from . import int_ops
from .vecmat_v8 import vecmat_v8
from .rmsnorm import (
    rmsnorm,
    GAMMA_MODE_LLAMA, GAMMA_MODE_GEMMA,
)
from .argmax_v8 import argmax_v8, argmax_v8_full_logits


# Kernel model_type enum values (from kernel/compute/model_loader.fj)
MODEL_TYPE_LLAMA = 1
MODEL_TYPE_SMOLLM = 2
MODEL_TYPE_GEMMA3_1B = 10
MODEL_TYPE_GEMMA3_270M = 11


@dataclass(frozen=True)
class TransformerConfig:
    """Architecture parameters. Matches kernel `mdl_get_*` queries."""
    vocab_size: int
    d_model: int
    n_heads: int
    n_kv_heads: int   # GQA: may be < n_heads (stubbed in P2.1)
    d_head: int       # usually d_model // n_heads
    n_layers: int
    ffn_dim: int
    model_type: int = MODEL_TYPE_GEMMA3_1B

    @property
    def is_gemma3(self) -> bool:
        return self.model_type in (MODEL_TYPE_GEMMA3_1B, MODEL_TYPE_GEMMA3_270M)

    @property
    def gamma_mode(self) -> int:
        return GAMMA_MODE_GEMMA if self.is_gemma3 else GAMMA_MODE_LLAMA


@dataclass
class LayerWeights:
    """One transformer block's quantized weights + norms.

    For P2.1 each projection is stored as (packed_bytes, scales, zeros)
    where the shape is implicit (`m × n`) per kernel tfm_layer_stream.
    For real Gemma 3: Q is (d_model, n_heads*d_head), K/V are
    (d_model, n_kv*d_head), O is (n_heads*d_head, d_model), gate/up
    are (d_model, ffn_dim), down is (ffn_dim, d_model).

    Norm gammas are stored as lists of i64 (fp×1000 post-V28.2
    convention). In Gemma mode the actual multiplier is 1 + γ/1000.
    """
    q_packed: bytes
    q_scales: Sequence[int]
    q_zeros: bytes
    k_packed: bytes
    k_scales: Sequence[int]
    k_zeros: bytes
    v_packed: bytes
    v_scales: Sequence[int]
    v_zeros: bytes
    o_packed: bytes
    o_scales: Sequence[int]
    o_zeros: bytes
    gate_packed: bytes
    gate_scales: Sequence[int]
    gate_zeros: bytes
    up_packed: bytes
    up_scales: Sequence[int]
    up_zeros: bytes
    down_packed: bytes
    down_scales: Sequence[int]
    down_zeros: bytes

    # Four RMSNorm gammas per Gemma-3 layer:
    #   pre_attn, post_attn, pre_ffn, post_ffn
    # Non-Gemma layers use only pre_attn + pre_ffn; post_* may be None.
    pre_attn_gamma: Sequence[int]
    post_attn_gamma: Optional[Sequence[int]]
    pre_ffn_gamma: Sequence[int]
    post_ffn_gamma: Optional[Sequence[int]]


# ── Activation functions (pluggable; kernel uses km_tanh_approx) ──────

def activation_identity(x: int) -> int:
    return x


def activation_relu(x: int) -> int:
    return x if x > 0 else 0


# ── Forward-pass stages ───────────────────────────────────────────────

def embed_lookup(
    token_id: int,
    embed_packed: bytes,
    embed_scales: Sequence[int],
    embed_zeros: bytes,
    vocab_size: int,
    d_model: int,
) -> List[int]:
    """Dequantize one row of the group-wise 4-bit embedding matrix.

    Mirrors `mdl_stream_embed_lookup_raw_v8` (kernel/compute/
    model_loader.fj:2111). Output units: i64 fp×1000 (kernel divides
    by 1000 after the ×1e6 dequant stage to match the rest of the
    ×1000 fixed-point pipeline).
    """
    if not 0 <= token_id < vocab_size:
        raise ValueError(
            f"token_id {token_id} out of range [0, {vocab_size})"
        )
    row_start = token_id * d_model
    out = [0] * d_model
    for i in range(d_model):
        flat = int_ops.add_i64(row_start, i)
        q = int_ops.read_nibble(embed_packed, flat)
        g = int_ops.ashr_i64(flat, 7)  # group_size=128
        scale = int_ops.widen_u32(embed_scales[g])
        zero = int_ops.widen_u8(embed_zeros[g])
        w_x_1M = int_ops.mul_i64(int_ops.sub_i64(q, zero), scale)
        # Convert ×1e6 → ×1000 (matches kernel L2137)
        out[i] = int_ops.trunc_div_i64(w_x_1M, 1000)
    return out


def simplified_attention_single_pos(
    x: List[int],
    layer: LayerWeights,
    cfg: TransformerConfig,
) -> List[int]:
    """Attention for a length-1 sequence — collapses to `W_O · W_V · x`.

    For real prefill over N tokens this would be full scaled dot-product
    attention with RoPE, KV cache, and softmax. P2.1 simplifies by
    noting that softmax over a length-1 sequence is trivially [1.0],
    so the attention output equals V. Composition W_O ∘ V gives a
    well-formed structural transformer without implementing attention
    machinery.

    Q is computed but unused (present for weight-touch parity); future
    P2.3 revision will reintroduce real multi-pos attention.
    """
    # Compute V = W_V · x   shape: (d_model)  (treating n_kv*d_head = d_model for P2.1)
    kv_d = cfg.n_kv_heads * cfg.d_head
    v_out = vecmat_v8(
        x, layer.v_packed, layer.v_scales, layer.v_zeros,
        cfg.d_model, kv_d,
    )
    # Touch Q and K to ensure weights are shape-consistent (helps catch
    # construction bugs in tests even though the values are unused here)
    q_dim = cfg.n_heads * cfg.d_head
    _q = vecmat_v8(
        x, layer.q_packed, layer.q_scales, layer.q_zeros,
        cfg.d_model, q_dim,
    )
    _k = vecmat_v8(
        x, layer.k_packed, layer.k_scales, layer.k_zeros,
        cfg.d_model, kv_d,
    )

    # O projection: W_O · V → (d_model)
    # In real Gemma: W_O has shape (n_heads*d_head, d_model). For P2.1
    # with kv_d == d_model this is (d_model, d_model).
    attn_in_dim = q_dim if q_dim == kv_d else kv_d
    o_out = vecmat_v8(
        v_out, layer.o_packed, layer.o_scales, layer.o_zeros,
        attn_in_dim, cfg.d_model,
    )
    return o_out


def gated_ffn(
    x: List[int],
    layer: LayerWeights,
    cfg: TransformerConfig,
    activation: Callable[[int], int] = activation_identity,
) -> List[int]:
    """Gated FFN: `down(act(gate·x) * (up·x))`.

    Mirrors `tfm_ffn_gated` (kernel/compute/transformer.fj:447). For
    P2.1 the element-wise product of gate and up is computed in
    ×1000 fp and divided back by 1000. Activation defaults to identity
    (deterministic for tests). Real kernel uses GELU-tanh approx.
    """
    gate_out = vecmat_v8(
        x, layer.gate_packed, layer.gate_scales, layer.gate_zeros,
        cfg.d_model, cfg.ffn_dim,
    )
    up_out = vecmat_v8(
        x, layer.up_packed, layer.up_scales, layer.up_zeros,
        cfg.d_model, cfg.ffn_dim,
    )
    # Element-wise: activation(gate) * up / 1000   (keeps fp×1000 scale)
    hidden = [
        int_ops.trunc_div_i64(
            int_ops.mul_i64(activation(g), u),
            1000,
        )
        for g, u in zip(gate_out, up_out)
    ]
    return vecmat_v8(
        hidden, layer.down_packed, layer.down_scales, layer.down_zeros,
        cfg.ffn_dim, cfg.d_model,
    )


def layer_forward(
    x: List[int],
    layer: LayerWeights,
    cfg: TransformerConfig,
    activation: Callable[[int], int] = activation_identity,
) -> List[int]:
    """One transformer block. Matches Gemma 3 4-norm flow when
    `cfg.is_gemma3`, otherwise falls back to Llama-style 2-norm flow."""
    dim = cfg.d_model

    # === Pre-attention RMSNorm ===
    res = list(x)
    x_normed = rmsnorm(
        x, gamma=layer.pre_attn_gamma, gamma_mode=cfg.gamma_mode,
    )

    # === Attention block ===
    attn_out = simplified_attention_single_pos(x_normed, layer, cfg)

    # === Gemma 3: post-attention RMSNorm before residual ===
    if cfg.is_gemma3 and layer.post_attn_gamma is not None:
        attn_out = rmsnorm(
            attn_out, gamma=layer.post_attn_gamma,
            gamma_mode=cfg.gamma_mode,
        )

    # === Residual ===
    x = [int_ops.add_i64(a, b) for a, b in zip(res, attn_out)]

    # === Pre-FFN RMSNorm ===
    res = list(x)
    x_normed = rmsnorm(
        x, gamma=layer.pre_ffn_gamma, gamma_mode=cfg.gamma_mode,
    )

    # === FFN ===
    ffn_out = gated_ffn(x_normed, layer, cfg, activation=activation)

    # === Gemma 3: post-FFN RMSNorm before residual ===
    if cfg.is_gemma3 and layer.post_ffn_gamma is not None:
        ffn_out = rmsnorm(
            ffn_out, gamma=layer.post_ffn_gamma,
            gamma_mode=cfg.gamma_mode,
        )

    # === Residual ===
    x = [int_ops.add_i64(a, b) for a, b in zip(res, ffn_out)]

    assert len(x) == dim
    return x


@dataclass
class ForwardResult:
    """Output of a single-prompt forward pass."""
    last_token: int            # argmax over final lmhead logits
    last_score: int            # best_score at winning token
    last_hidden: List[int]     # hidden state after final norm (d_model)
    per_token_hidden: List[List[int]] = field(default_factory=list)
    """If `capture_all=True` in forward(), one hidden state per prompt
    token after all layers+final_norm (useful for P3 trajectory analysis)."""


def forward(
    tokens: Sequence[int],
    embed_packed: bytes,
    embed_scales: Sequence[int],
    embed_zeros: bytes,
    layers: Sequence[LayerWeights],
    final_norm_gamma: Sequence[int],
    cfg: TransformerConfig,
    activation: Callable[[int], int] = activation_identity,
    capture_all: bool = False,
) -> ForwardResult:
    """Run a full prefill + single-position argmax.

    For P2.1 the "prefill" is actually independent per-token forward
    passes (no KV cache, no cross-token attention because single-pos
    attention is used). The returned argmax is for the LAST prompt
    token's hidden state — matching how argmax is used in
    `tfm_generate_stream` at prefill completion.
    """
    if not tokens:
        raise ValueError("tokens must be non-empty")

    per_token = []
    last_hidden = None

    for tok in tokens:
        # === Embed lookup ===
        x = embed_lookup(
            tok, embed_packed, embed_scales, embed_zeros,
            cfg.vocab_size, cfg.d_model,
        )

        # === 26× (or cfg.n_layers) layers ===
        for layer in layers:
            x = layer_forward(x, layer, cfg, activation=activation)

        # === Final RMSNorm ===
        x = rmsnorm(
            x, gamma=final_norm_gamma, gamma_mode=cfg.gamma_mode,
        )
        last_hidden = x
        if capture_all:
            per_token.append(list(x))

    assert last_hidden is not None

    # === LM head (tied with embedding) argmax on LAST prompt token ===
    best_token, best_score = argmax_v8(
        last_hidden, embed_packed, embed_scales, embed_zeros,
        cfg.vocab_size, cfg.d_model,
    )

    return ForwardResult(
        last_token=best_token,
        last_score=best_score,
        last_hidden=last_hidden,
        per_token_hidden=per_token if capture_all else [],
    )


def forward_with_logits(
    tokens: Sequence[int],
    embed_packed: bytes,
    embed_scales: Sequence[int],
    embed_zeros: bytes,
    layers: Sequence[LayerWeights],
    final_norm_gamma: Sequence[int],
    cfg: TransformerConfig,
    activation: Callable[[int], int] = activation_identity,
):
    """Like forward(), but also returns the full logit vector from the
    LM head. Used by P3 spread analysis."""
    if not tokens:
        raise ValueError("tokens must be non-empty")

    last_hidden = None
    for tok in tokens:
        x = embed_lookup(
            tok, embed_packed, embed_scales, embed_zeros,
            cfg.vocab_size, cfg.d_model,
        )
        for layer in layers:
            x = layer_forward(x, layer, cfg, activation=activation)
        x = rmsnorm(
            x, gamma=final_norm_gamma, gamma_mode=cfg.gamma_mode,
        )
        last_hidden = x

    best_token, best_score, logits = argmax_v8_full_logits(
        last_hidden, embed_packed, embed_scales, embed_zeros,
        cfg.vocab_size, cfg.d_model,
    )
    return best_token, best_score, logits, last_hidden
