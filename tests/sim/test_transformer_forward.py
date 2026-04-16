"""P2.1 structural tests for transformer.forward orchestration.

Gate (per plan): "Python forward runs on 'hello' prompt to completion
without error". Uses tiny synthetic weights so a full prefill completes
in under a second. Does NOT attempt real Gemma 3 1B — that requires
P2.3 kernel FJTRACE capture for any meaningful comparison.
"""

from __future__ import annotations

import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

import pytest

from kernel_sim import (
    TransformerConfig, LayerWeights, ForwardResult,
    MODEL_TYPE_LLAMA, MODEL_TYPE_GEMMA3_1B,
    V8_GROUP_SIZE, V8_SCALE_FP,
    pack_nibbles,
    activation_identity, activation_relu,
    embed_lookup, simplified_attention_single_pos, gated_ffn,
    layer_forward, forward, forward_with_logits,
)


# ── Synthetic-weight helpers ──────────────────────────────────────────

def _make_packed(rng, m: int, n: int):
    """Build (packed_bytes, scales, zeros) for a random m×n quantized
    weight. Total must be even. Uses small scales / zero ~ 7 (mid-range)
    to keep arithmetic magnitudes reasonable."""
    total = m * n
    assert total % 2 == 0, f"m*n must be even, got {m}*{n}"
    n_groups = (total + V8_GROUP_SIZE - 1) // V8_GROUP_SIZE
    nibbles = [rng.randint(0, 15) for _ in range(total)]
    packed = pack_nibbles(nibbles)
    # Small scales so values stay in manageable range
    scales = [rng.randint(1_000, 50_000) for _ in range(n_groups)]
    zeros = bytes([rng.randint(5, 10) for _ in range(n_groups)])
    return packed, scales, zeros


def _make_layer(rng, cfg: TransformerConfig) -> LayerWeights:
    """Build one random LayerWeights for the given config."""
    dm = cfg.d_model
    kv_d = cfg.n_kv_heads * cfg.d_head
    q_dim = cfg.n_heads * cfg.d_head
    ff = cfg.ffn_dim

    q_p, q_s, q_z = _make_packed(rng, dm, q_dim)
    k_p, k_s, k_z = _make_packed(rng, dm, kv_d)
    v_p, v_s, v_z = _make_packed(rng, dm, kv_d)
    # O projection's input dim matches attention output (kv_d in our
    # simplified attention)
    o_in = kv_d
    o_p, o_s, o_z = _make_packed(rng, o_in, dm)
    gate_p, gate_s, gate_z = _make_packed(rng, dm, ff)
    up_p, up_s, up_z = _make_packed(rng, dm, ff)
    down_p, down_s, down_z = _make_packed(rng, ff, dm)

    # Norm gammas (fp×1000). Small magnitudes — keep close to identity.
    def _gamma():
        return [rng.randint(-100, 100) for _ in range(dm)]

    return LayerWeights(
        q_packed=q_p, q_scales=q_s, q_zeros=q_z,
        k_packed=k_p, k_scales=k_s, k_zeros=k_z,
        v_packed=v_p, v_scales=v_s, v_zeros=v_z,
        o_packed=o_p, o_scales=o_s, o_zeros=o_z,
        gate_packed=gate_p, gate_scales=gate_s, gate_zeros=gate_z,
        up_packed=up_p, up_scales=up_s, up_zeros=up_z,
        down_packed=down_p, down_scales=down_s, down_zeros=down_z,
        pre_attn_gamma=_gamma(),
        post_attn_gamma=_gamma(),
        pre_ffn_gamma=_gamma(),
        post_ffn_gamma=_gamma(),
    )


def _tiny_config() -> TransformerConfig:
    """Minimum viable Gemma-3-like config that keeps all math intact
    but completes a full forward in ~0.1s of Python."""
    return TransformerConfig(
        vocab_size=32,
        d_model=4,
        n_heads=1,
        n_kv_heads=1,
        d_head=4,
        n_layers=2,
        ffn_dim=16,
        model_type=MODEL_TYPE_GEMMA3_1B,
    )


def _small_llama_config() -> TransformerConfig:
    """Same sizes, Llama-style gamma mode — sanity check the branch."""
    return TransformerConfig(
        vocab_size=32,
        d_model=4,
        n_heads=1,
        n_kv_heads=1,
        d_head=4,
        n_layers=2,
        ffn_dim=16,
        model_type=MODEL_TYPE_LLAMA,
    )


# ── Embed-lookup tests ────────────────────────────────────────────────

class TestEmbedLookup:
    def test_shape(self):
        rng = random.Random(0)
        packed, scales, zeros = _make_packed(rng, 32, 4)
        out = embed_lookup(5, packed, scales, zeros, 32, 4)
        assert len(out) == 4

    def test_different_tokens_give_different_rows(self):
        rng = random.Random(1)
        # 32 × 4 = 128 elements, 1 group → all rows share scale/zero
        # but nibbles differ per row → rows differ.
        packed, scales, zeros = _make_packed(rng, 32, 4)
        row_3 = embed_lookup(3, packed, scales, zeros, 32, 4)
        row_7 = embed_lookup(7, packed, scales, zeros, 32, 4)
        # Not all four values must differ, but typically at least one
        # nibble differs between rows 3 and 7
        assert row_3 != row_7

    def test_out_of_range_raises(self):
        rng = random.Random(2)
        packed, scales, zeros = _make_packed(rng, 32, 4)
        with pytest.raises(ValueError):
            embed_lookup(32, packed, scales, zeros, 32, 4)
        with pytest.raises(ValueError):
            embed_lookup(-1, packed, scales, zeros, 32, 4)


# ── Attention + FFN stage tests ───────────────────────────────────────

class TestAttentionStage:
    def test_attention_single_pos_shape(self):
        rng = random.Random(3)
        cfg = _tiny_config()
        layer = _make_layer(rng, cfg)
        x = [rng.randint(-1000, 1000) for _ in range(cfg.d_model)]
        out = simplified_attention_single_pos(x, layer, cfg)
        assert len(out) == cfg.d_model

    def test_attention_deterministic(self):
        rng = random.Random(4)
        cfg = _tiny_config()
        layer = _make_layer(rng, cfg)
        x = [100, -200, 300, -400]
        a = simplified_attention_single_pos(x, layer, cfg)
        b = simplified_attention_single_pos(x, layer, cfg)
        assert a == b


class TestFFNStage:
    def test_ffn_shape(self):
        rng = random.Random(5)
        cfg = _tiny_config()
        layer = _make_layer(rng, cfg)
        x = [rng.randint(-1000, 1000) for _ in range(cfg.d_model)]
        out = gated_ffn(x, layer, cfg)
        assert len(out) == cfg.d_model

    def test_ffn_identity_vs_relu(self):
        """With negative inputs, ReLU must differ from identity in the
        gate path — confirms the activation hook is actually applied."""
        rng = random.Random(6)
        cfg = _tiny_config()
        layer = _make_layer(rng, cfg)
        # Force the gate·x product to be negative somewhere by biasing x
        x = [-500] * cfg.d_model
        a = gated_ffn(x, layer, cfg, activation=activation_identity)
        b = gated_ffn(x, layer, cfg, activation=activation_relu)
        # Not required that every element differs, but for random
        # weights the two almost always produce different outputs
        assert a != b, (
            "ReLU and identity FFN produced identical outputs — either "
            "weights are too sparse or activation hook is broken"
        )


# ── Layer + full forward tests ────────────────────────────────────────

class TestLayerForward:
    def test_gemma3_layer_shape(self):
        rng = random.Random(7)
        cfg = _tiny_config()
        layer = _make_layer(rng, cfg)
        x = [rng.randint(-1000, 1000) for _ in range(cfg.d_model)]
        out = layer_forward(x, layer, cfg)
        assert len(out) == cfg.d_model

    def test_llama_layer_skips_post_norms(self):
        """In Llama mode, post_attn_gamma / post_ffn_gamma must NOT be
        applied (branch guard). If they were applied, swapping them
        would change output. If they're skipped (correct), output is
        unchanged."""
        rng = random.Random(8)
        cfg = _small_llama_config()
        layer_a = _make_layer(rng, cfg)
        # Build layer_b identical to layer_a except with post gammas
        # replaced by very different values. Output should be identical
        # in Llama mode (post gammas unused).
        layer_b = LayerWeights(
            q_packed=layer_a.q_packed, q_scales=layer_a.q_scales, q_zeros=layer_a.q_zeros,
            k_packed=layer_a.k_packed, k_scales=layer_a.k_scales, k_zeros=layer_a.k_zeros,
            v_packed=layer_a.v_packed, v_scales=layer_a.v_scales, v_zeros=layer_a.v_zeros,
            o_packed=layer_a.o_packed, o_scales=layer_a.o_scales, o_zeros=layer_a.o_zeros,
            gate_packed=layer_a.gate_packed, gate_scales=layer_a.gate_scales,
            gate_zeros=layer_a.gate_zeros,
            up_packed=layer_a.up_packed, up_scales=layer_a.up_scales,
            up_zeros=layer_a.up_zeros,
            down_packed=layer_a.down_packed, down_scales=layer_a.down_scales,
            down_zeros=layer_a.down_zeros,
            pre_attn_gamma=layer_a.pre_attn_gamma,
            post_attn_gamma=[9999] * cfg.d_model,  # large outlier
            pre_ffn_gamma=layer_a.pre_ffn_gamma,
            post_ffn_gamma=[9999] * cfg.d_model,
        )
        x = [100, -200, 300, -400]
        out_a = layer_forward(x, layer_a, cfg)
        out_b = layer_forward(x, layer_b, cfg)
        assert out_a == out_b, (
            "Llama mode is applying post_*_gamma — should skip them"
        )


class TestForwardEndToEnd:
    """The P2.1 headline gate: forward runs on 'hello' prompt without
    error. 'hello' here is represented as the token IDs [1, 2, 3, 4, 5]
    since the simulator is tokenizer-agnostic."""

    def test_e2e_hello_prompt_runs(self):
        rng = random.Random(9)
        cfg = _tiny_config()
        # Build the whole model: embed + N layers + final norm
        embed_packed, embed_scales, embed_zeros = _make_packed(
            rng, cfg.vocab_size, cfg.d_model
        )
        layers = [_make_layer(rng, cfg) for _ in range(cfg.n_layers)]
        final_gamma = [rng.randint(-100, 100) for _ in range(cfg.d_model)]

        # "hello" → 5 tokens (stubbed as [1, 2, 3, 4, 5])
        tokens = [1, 2, 3, 4, 5]

        result = forward(
            tokens, embed_packed, embed_scales, embed_zeros,
            layers, final_gamma, cfg,
            activation=activation_identity,
        )
        assert isinstance(result, ForwardResult)
        assert 0 <= result.last_token < cfg.vocab_size
        assert len(result.last_hidden) == cfg.d_model

    def test_capture_all_yields_per_token_hidden(self):
        rng = random.Random(10)
        cfg = _tiny_config()
        embed_packed, embed_scales, embed_zeros = _make_packed(
            rng, cfg.vocab_size, cfg.d_model
        )
        layers = [_make_layer(rng, cfg) for _ in range(cfg.n_layers)]
        final_gamma = [rng.randint(-100, 100) for _ in range(cfg.d_model)]
        tokens = [1, 2, 3, 4, 5]

        result = forward(
            tokens, embed_packed, embed_scales, embed_zeros,
            layers, final_gamma, cfg,
            capture_all=True,
        )
        assert len(result.per_token_hidden) == len(tokens)
        for h in result.per_token_hidden:
            assert len(h) == cfg.d_model

    def test_forward_with_logits_reports_spread(self):
        """Validates the P3 diagnostic hook end-to-end: we should be
        able to get the full logit vector after forward() and measure
        spread. For random weights we expect a wide spread (>> 0)."""
        rng = random.Random(11)
        cfg = _tiny_config()
        embed_packed, embed_scales, embed_zeros = _make_packed(
            rng, cfg.vocab_size, cfg.d_model
        )
        layers = [_make_layer(rng, cfg) for _ in range(cfg.n_layers)]
        final_gamma = [rng.randint(-100, 100) for _ in range(cfg.d_model)]
        tokens = [1, 2, 3, 4, 5]

        tok, score, logits, hidden = forward_with_logits(
            tokens, embed_packed, embed_scales, embed_zeros,
            layers, final_gamma, cfg,
        )
        assert len(logits) == cfg.vocab_size
        assert logits[tok] == score
        spread = max(logits) - min(logits)
        # Random weights must produce non-degenerate logit distribution
        assert spread > 0, (
            "Forward produced all-zero logit spread — something is "
            "collapsing hidden state (this is the exact pad-collapse "
            "mechanism we're debugging; if it happens on random weights "
            "then the orchestration itself is broken)"
        )

    def test_determinism(self):
        """Same weights + tokens → same argmax, byte-for-byte."""
        rng = random.Random(12)
        cfg = _tiny_config()
        embed_packed, embed_scales, embed_zeros = _make_packed(
            rng, cfg.vocab_size, cfg.d_model
        )
        layers = [_make_layer(rng, cfg) for _ in range(cfg.n_layers)]
        final_gamma = [rng.randint(-100, 100) for _ in range(cfg.d_model)]
        tokens = [1, 2, 3, 4, 5]

        r1 = forward(
            tokens, embed_packed, embed_scales, embed_zeros,
            layers, final_gamma, cfg,
        )
        r2 = forward(
            tokens, embed_packed, embed_scales, embed_zeros,
            layers, final_gamma, cfg,
        )
        assert r1.last_token == r2.last_token
        assert r1.last_score == r2.last_score
        assert r1.last_hidden == r2.last_hidden

    def test_empty_prompt_raises(self):
        rng = random.Random(13)
        cfg = _tiny_config()
        embed_packed, embed_scales, embed_zeros = _make_packed(
            rng, cfg.vocab_size, cfg.d_model
        )
        layers = [_make_layer(rng, cfg) for _ in range(cfg.n_layers)]
        final_gamma = [0] * cfg.d_model
        with pytest.raises(ValueError):
            forward(
                [], embed_packed, embed_scales, embed_zeros,
                layers, final_gamma, cfg,
            )

    def test_gemma_mode_and_llama_mode_diverge(self):
        """Gemma mode applies (1 + γ/1000), Llama mode applies γ/1000.
        For non-zero gamma these must produce different outputs."""
        rng = random.Random(14)
        cfg_g = TransformerConfig(
            vocab_size=32, d_model=4, n_heads=1, n_kv_heads=1,
            d_head=4, n_layers=1, ffn_dim=16,
            model_type=MODEL_TYPE_GEMMA3_1B,
        )
        cfg_l = TransformerConfig(
            vocab_size=32, d_model=4, n_heads=1, n_kv_heads=1,
            d_head=4, n_layers=1, ffn_dim=16,
            model_type=MODEL_TYPE_LLAMA,
        )
        embed_packed, embed_scales, embed_zeros = _make_packed(
            rng, 32, 4
        )
        # Force determinism: same layer weights for both
        rng_layer = random.Random(15)
        layer_g = _make_layer(rng_layer, cfg_g)
        rng_layer = random.Random(15)
        layer_l = _make_layer(rng_layer, cfg_l)
        final_gamma = [100] * 4

        r_g = forward(
            [1, 2, 3], embed_packed, embed_scales, embed_zeros,
            [layer_g], final_gamma, cfg_g,
        )
        r_l = forward(
            [1, 2, 3], embed_packed, embed_scales, embed_zeros,
            [layer_l], final_gamma, cfg_l,
        )
        # Same inputs + weights, different gamma modes → hidden
        # states should differ (and likely argmax too, but at
        # minimum the hidden state must not be identical).
        assert r_g.last_hidden != r_l.last_hidden, (
            "Gemma and Llama modes produced identical hidden state — "
            "gamma_mode branch is not being applied"
        )
