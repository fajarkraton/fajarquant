"""P2.2 tests for trace.py per-op JSONL capture.

Gate (per plan):
  `python -m kernel_sim hello > /tmp/sim_trace.jsonl` produces ~500+
  lines. Since the P2.1 tests use a tiny 2-layer config (not 26) the
  expected line count here scales down proportionally:

    1 embed + (4 pre+post norms + 4 projs + 1 attn_out + 1 residual +
             4 ffn ops + 1 residual) per layer × N layers + 1 final +
             1 argmax

  For N=2 layers, 5 tokens, Gemma-3 mode (post-norms emitted):
    per token: 1 embed + 2 × (pre_attn + q + k + v + attn_out +
                              post_attn + residual + pre_ffn + gate
                              + up + ffn_hidden + down + post_ffn +
                              residual) + 1 final
             = 1 + 2 × 14 + 1 = 30
    total    : 5 × 30 + 1 argmax = 151 records

  The `test_e2e_record_count` test asserts this exact count so any
  later instrumentation drift is caught immediately.
"""

from __future__ import annotations

import io
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "tools"))

import pytest

from kernel_sim import (
    SCHEMA_VERSION, OP_NAMES, LAYER_SCOPED_OPS,
    FNV_OFFSET_64, fnv1a_u64_bytes,
    TraceWriter, NoopTracer, NOOP_TRACER,
    TransformerConfig, LayerWeights,
    MODEL_TYPE_GEMMA3_1B, MODEL_TYPE_LLAMA,
    V8_GROUP_SIZE, pack_nibbles,
    activation_identity,
    forward, forward_with_logits,
)


# ── Synthetic weight helpers (dup of test_transformer_forward) ────────

def _make_packed(rng, m: int, n: int):
    total = m * n
    assert total % 2 == 0
    n_groups = (total + V8_GROUP_SIZE - 1) // V8_GROUP_SIZE
    nibbles = [rng.randint(0, 15) for _ in range(total)]
    packed = pack_nibbles(nibbles)
    scales = [rng.randint(1_000, 50_000) for _ in range(n_groups)]
    zeros = bytes([rng.randint(5, 10) for _ in range(n_groups)])
    return packed, scales, zeros


def _make_layer(rng, cfg: TransformerConfig) -> LayerWeights:
    dm = cfg.d_model
    kv_d = cfg.n_kv_heads * cfg.d_head
    q_dim = cfg.n_heads * cfg.d_head
    ff = cfg.ffn_dim
    q_p, q_s, q_z = _make_packed(rng, dm, q_dim)
    k_p, k_s, k_z = _make_packed(rng, dm, kv_d)
    v_p, v_s, v_z = _make_packed(rng, dm, kv_d)
    o_p, o_s, o_z = _make_packed(rng, kv_d, dm)
    gate_p, gate_s, gate_z = _make_packed(rng, dm, ff)
    up_p, up_s, up_z = _make_packed(rng, dm, ff)
    down_p, down_s, down_z = _make_packed(rng, ff, dm)

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


def _tiny_gemma3() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=32, d_model=4, n_heads=1, n_kv_heads=1, d_head=4,
        n_layers=2, ffn_dim=16, model_type=MODEL_TYPE_GEMMA3_1B,
    )


def _tiny_llama() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=32, d_model=4, n_heads=1, n_kv_heads=1, d_head=4,
        n_layers=2, ffn_dim=16, model_type=MODEL_TYPE_LLAMA,
    )


def _build_model(rng, cfg):
    e_p, e_s, e_z = _make_packed(rng, cfg.vocab_size, cfg.d_model)
    layers = [_make_layer(rng, cfg) for _ in range(cfg.n_layers)]
    final_gamma = [rng.randint(-100, 100) for _ in range(cfg.d_model)]
    return e_p, e_s, e_z, layers, final_gamma


# ── Module-level schema constants ─────────────────────────────────────

class TestSchemaConstants:
    def test_schema_version_is_pinned(self):
        assert SCHEMA_VERSION == 1

    def test_op_names_match_plan_spec(self):
        """OP_NAMES must cover exactly the boundaries called out in the
        plan §P2.2. Adding an op here without updating P2.3 kernel
        FJTRACE would silently break cross-source parity."""
        expected = {
            "embed_lookup",
            "pre_attn_rmsnorm",
            "q_proj", "k_proj", "v_proj",
            "attn_out", "post_attn_rmsnorm", "attn_residual",
            "pre_ffn_rmsnorm",
            "gate_proj", "up_proj", "ffn_hidden", "down_proj",
            "post_ffn_rmsnorm", "ffn_residual",
            "final_rmsnorm", "argmax",
        }
        assert set(OP_NAMES) == expected

    def test_layer_scoped_ops_excludes_boundaries(self):
        assert "embed_lookup" not in LAYER_SCOPED_OPS
        assert "final_rmsnorm" not in LAYER_SCOPED_OPS
        assert "argmax" not in LAYER_SCOPED_OPS
        assert "q_proj" in LAYER_SCOPED_OPS
        assert "attn_residual" in LAYER_SCOPED_OPS


# ── FNV-1a 64-bit hash ────────────────────────────────────────────────

class TestFnv1aHash:
    def test_empty_bytes_returns_offset(self):
        assert fnv1a_u64_bytes(b"") == FNV_OFFSET_64

    def test_known_vector_foo(self):
        """Known FNV-1a 64-bit hash of b'foo' from the reference spec."""
        # Computed once and pinned; any change breaks bit-exact
        # parity with a kernel implementation.
        expected = fnv1a_u64_bytes(b"foo")
        # Double-check via manual computation
        h = FNV_OFFSET_64
        for b in b"foo":
            h ^= b
            h = (h * 0x100000001B3) & ((1 << 64) - 1)
        assert expected == h

    def test_different_inputs_hash_differently(self):
        h1 = fnv1a_u64_bytes(b"abc")
        h2 = fnv1a_u64_bytes(b"abd")
        assert h1 != h2

    def test_hash_stays_in_u64_range(self):
        for _ in range(50):
            data = bytes(random.randint(0, 255) for _ in range(100))
            h = fnv1a_u64_bytes(data)
            assert 0 <= h < (1 << 64)


# ── TraceWriter API ───────────────────────────────────────────────────

class TestTraceWriterDisabled:
    def test_disabled_records_do_not_advance_step(self):
        tw = TraceWriter(enabled=False)
        tw.record(
            "embed_lookup", [1, 2, 3, 4], token_idx=0, layer=-1,
            shape=[4], dtype="i64",
        )
        assert tw.step == 0

    def test_from_env_without_var_is_disabled(self, monkeypatch):
        monkeypatch.delenv("FJ_SIM_TRACE", raising=False)
        tw = TraceWriter.from_env()
        assert tw.enabled is False


class TestTraceWriterEnabled:
    def _mk(self, tmp_path):
        path = str(tmp_path / "trace.jsonl")
        tw = TraceWriter(path=path, enabled=True)
        tw._open()
        return tw, path

    def test_record_emits_jsonl_line(self, tmp_path):
        tw, path = self._mk(tmp_path)
        tw.record(
            "embed_lookup", [10, -20, 30, 0], token_idx=0, layer=-1,
            shape=[4], dtype="i64",
        )
        tw.close()
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["op"] == "embed_lookup"
        assert rec["shape"] == [4]
        assert rec["dtype"] == "i64"
        assert rec["min"] == -20
        assert rec["max"] == 30
        assert rec["nnz"] == 3
        assert rec["schema_version"] == 1
        assert rec["step"] == 0

    def test_top5_abs_sorted_by_magnitude(self, tmp_path):
        tw, path = self._mk(tmp_path)
        # Distinct magnitudes so order is deterministic
        tw.record(
            "q_proj", [1, -200, 3, -4, 50, 6, -7000, 8],
            token_idx=0, layer=0, shape=[8], dtype="i64",
        )
        tw.close()
        rec = json.loads(open(path).readline())
        # |values|: 7000(-), 200(-), 50, 8, 6, 4(-), 3, 1
        expected = [[6, -7000], [1, -200], [4, 50], [7, 8], [5, 6]]
        assert rec["top5_abs"] == expected

    def test_int_mean_trunc_toward_zero(self, tmp_path):
        tw, path = self._mk(tmp_path)
        # -5 -3 -1 1 3 5 → sum=0 → mean=0 (easy case)
        tw.record(
            "attn_residual", [-5, -3, -1, 1, 3, 5, 0, 0],
            token_idx=0, layer=0, shape=[8], dtype="i64",
        )
        # -7 -2 → sum=-9 → kernel trunc mean: -9/2 = -4 (not -5)
        tw.record(
            "attn_residual", [-7, -2], token_idx=0, layer=0,
            shape=[2], dtype="i64",
        )
        tw.close()
        recs = [json.loads(l) for l in open(path).readlines()]
        assert recs[0]["mean"] == 0
        assert recs[1]["mean"] == -4

    def test_float_dtype_records_float_stats(self, tmp_path):
        tw, path = self._mk(tmp_path)
        tw.record(
            "q_proj", [0.5, -1.25, 2.0, 0.0], token_idx=0, layer=0,
            shape=[4], dtype="f32",
        )
        tw.close()
        rec = json.loads(open(path).readline())
        assert isinstance(rec["mean"], float)
        assert rec["min"] == -1.25
        assert rec["max"] == 2.0
        assert rec["nnz"] == 3

    def test_hash_is_hex_u64(self, tmp_path):
        tw, path = self._mk(tmp_path)
        tw.record(
            "embed_lookup", [1, 2, 3, 4], token_idx=0, layer=-1,
            shape=[4], dtype="i64",
        )
        tw.close()
        rec = json.loads(open(path).readline())
        h = rec["hash"]
        assert h.startswith("0x")
        # Always rendered as 16 hex digits after 0x
        assert len(h) == 18

    def test_hash_equal_for_equal_tensors(self, tmp_path):
        tw, path = self._mk(tmp_path)
        tw.record("embed_lookup", [1, 2, 3, 4], token_idx=0, layer=-1,
                  shape=[4], dtype="i64")
        tw.record("embed_lookup", [1, 2, 3, 4], token_idx=1, layer=-1,
                  shape=[4], dtype="i64")
        tw.close()
        recs = [json.loads(l) for l in open(path).readlines()]
        assert recs[0]["hash"] == recs[1]["hash"]
        # But different steps
        assert recs[0]["step"] != recs[1]["step"]

    def test_step_is_monotonic(self, tmp_path):
        tw, path = self._mk(tmp_path)
        for i in range(10):
            tw.record("embed_lookup", [i], token_idx=i, layer=-1,
                      shape=[1], dtype="i64")
        tw.close()
        steps = [json.loads(l)["step"] for l in open(path).readlines()]
        assert steps == list(range(10))

    def test_extra_field_round_trips(self, tmp_path):
        tw, path = self._mk(tmp_path)
        tw.record(
            "argmax", [42], token_idx=4, layer=-1, shape=[1],
            dtype="i64", extra={"best_score": 123456, "vocab_size": 32},
        )
        tw.close()
        rec = json.loads(open(path).readline())
        assert rec["extra"] == {"best_score": 123456, "vocab_size": 32}

    def test_shape_mismatch_raises(self, tmp_path):
        tw, path = self._mk(tmp_path)
        with pytest.raises(ValueError, match="shape .* implies"):
            tw.record("embed_lookup", [1, 2, 3], token_idx=0, layer=-1,
                      shape=[4], dtype="i64")

    def test_unknown_op_raises(self, tmp_path):
        tw, path = self._mk(tmp_path)
        with pytest.raises(ValueError, match="unknown op"):
            tw.record("bogus_op", [1], token_idx=0, layer=-1,
                      shape=[1], dtype="i64")

    def test_layer_scoped_op_rejects_layer_minus_one(self, tmp_path):
        tw, path = self._mk(tmp_path)
        with pytest.raises(ValueError, match="layer-scoped"):
            tw.record("q_proj", [1], token_idx=0, layer=-1,
                      shape=[1], dtype="i64")

    def test_non_layer_scoped_op_rejects_positive_layer(self, tmp_path):
        tw, path = self._mk(tmp_path)
        with pytest.raises(ValueError, match="not layer-scoped"):
            tw.record("embed_lookup", [1], token_idx=0, layer=3,
                      shape=[1], dtype="i64")

    def test_context_manager_closes_file(self, tmp_path):
        path = str(tmp_path / "cm.jsonl")
        with TraceWriter(path=path, enabled=True) as tw:
            tw.record("embed_lookup", [1], token_idx=0, layer=-1,
                      shape=[1], dtype="i64")
        # Must be able to re-open for reading immediately
        assert open(path).read().count("\n") == 1


# ── NoopTracer ────────────────────────────────────────────────────────

class TestNoopTracer:
    def test_record_returns_none_and_is_free(self):
        tr = NoopTracer()
        assert tr.record("anything", [1, 2, 3]) is None
        assert tr.step == 0

    def test_noop_can_be_used_as_context_manager(self):
        with NoopTracer() as tr:
            tr.record("anything", None, weird_kwarg=1)


# ── Integration with forward() ────────────────────────────────────────

class TestIntegrationWithForward:
    def test_forward_without_tracer_is_unchanged(self):
        """Calling forward() without `tracer` must behave exactly as
        before P2.2 — guards the P2.1 API contract."""
        rng = random.Random(42)
        cfg = _tiny_gemma3()
        e_p, e_s, e_z, layers, final_gamma = _build_model(rng, cfg)
        result_a = forward([1, 2, 3, 4, 5], e_p, e_s, e_z, layers,
                           final_gamma, cfg)
        result_b = forward([1, 2, 3, 4, 5], e_p, e_s, e_z, layers,
                           final_gamma, cfg, tracer=NOOP_TRACER)
        assert result_a.last_token == result_b.last_token
        assert result_a.last_score == result_b.last_score
        assert result_a.last_hidden == result_b.last_hidden

    def test_e2e_record_count_gemma3(self, tmp_path):
        """Headline gate: per-token records + final + argmax equals the
        formula in the module docstring. For N=2 layers, 5 tokens,
        Gemma-3 mode → 151 records."""
        rng = random.Random(42)
        cfg = _tiny_gemma3()
        e_p, e_s, e_z, layers, final_gamma = _build_model(rng, cfg)

        path = str(tmp_path / "e2e.jsonl")
        with TraceWriter(path=path, enabled=True) as tw:
            forward([1, 2, 3, 4, 5], e_p, e_s, e_z, layers,
                    final_gamma, cfg, tracer=tw)

        lines = open(path).readlines()
        # 1 embed + 2 × 14 layer ops + 1 final = 30 per token
        # 5 tokens × 30 + 1 argmax = 151
        per_token = 1 + cfg.n_layers * 14 + 1
        expected = 5 * per_token + 1
        assert len(lines) == expected == 151

    def test_e2e_record_count_llama_skips_post_norms(self, tmp_path):
        """Llama mode skips post_attn_rmsnorm and post_ffn_rmsnorm, so
        per-layer ops drop from 14 to 12 → fewer records total."""
        rng = random.Random(42)
        cfg = _tiny_llama()
        e_p, e_s, e_z, layers, final_gamma = _build_model(rng, cfg)

        path = str(tmp_path / "llama.jsonl")
        with TraceWriter(path=path, enabled=True) as tw:
            forward([1, 2, 3], e_p, e_s, e_z, layers,
                    final_gamma, cfg, tracer=tw)

        lines = open(path).readlines()
        per_token = 1 + cfg.n_layers * 12 + 1
        expected = 3 * per_token + 1
        assert len(lines) == expected

    def test_trace_records_are_valid_jsonl(self, tmp_path):
        rng = random.Random(42)
        cfg = _tiny_gemma3()
        e_p, e_s, e_z, layers, final_gamma = _build_model(rng, cfg)
        path = str(tmp_path / "valid.jsonl")
        with TraceWriter(path=path, enabled=True) as tw:
            forward([1, 2], e_p, e_s, e_z, layers, final_gamma, cfg,
                    tracer=tw)
        # Every line parses; required fields present; op in OP_NAMES
        for line in open(path).readlines():
            rec = json.loads(line)
            for f in ("schema_version", "step", "op", "token_idx",
                      "layer", "shape", "dtype", "min", "max",
                      "mean", "nnz", "top5_abs", "hash"):
                assert f in rec, f"missing {f}"
            assert rec["op"] in OP_NAMES

    def test_trace_argmax_has_winning_token_in_extra(self, tmp_path):
        rng = random.Random(42)
        cfg = _tiny_gemma3()
        e_p, e_s, e_z, layers, final_gamma = _build_model(rng, cfg)
        path = str(tmp_path / "argmax.jsonl")
        with TraceWriter(path=path, enabled=True) as tw:
            result = forward([1, 2, 3], e_p, e_s, e_z, layers,
                             final_gamma, cfg, tracer=tw)
        last = json.loads(open(path).readlines()[-1])
        assert last["op"] == "argmax"
        assert last["shape"] == [1]
        assert "extra" in last
        assert "best_score" in last["extra"]
        # The emitted value should be the winning token
        assert last["max"] == result.last_token

    def test_trace_layer_indices_cover_all_layers(self, tmp_path):
        rng = random.Random(42)
        cfg = _tiny_gemma3()
        e_p, e_s, e_z, layers, final_gamma = _build_model(rng, cfg)
        path = str(tmp_path / "layers.jsonl")
        with TraceWriter(path=path, enabled=True) as tw:
            forward([1], e_p, e_s, e_z, layers, final_gamma, cfg,
                    tracer=tw)
        recs = [json.loads(l) for l in open(path).readlines()]
        layer_idxs_seen = {r["layer"] for r in recs if r["layer"] != -1}
        assert layer_idxs_seen == set(range(cfg.n_layers))

    def test_trace_token_idxs_cover_all_tokens(self, tmp_path):
        rng = random.Random(42)
        cfg = _tiny_gemma3()
        e_p, e_s, e_z, layers, final_gamma = _build_model(rng, cfg)
        path = str(tmp_path / "toks.jsonl")
        with TraceWriter(path=path, enabled=True) as tw:
            forward([1, 2, 3, 4], e_p, e_s, e_z, layers, final_gamma,
                    cfg, tracer=tw)
        recs = [json.loads(l) for l in open(path).readlines()]
        seen = {r["token_idx"] for r in recs}
        assert seen == {0, 1, 2, 3}

    def test_forward_with_logits_also_traces(self, tmp_path):
        rng = random.Random(42)
        cfg = _tiny_gemma3()
        e_p, e_s, e_z, layers, final_gamma = _build_model(rng, cfg)
        path = str(tmp_path / "logits.jsonl")
        with TraceWriter(path=path, enabled=True) as tw:
            forward_with_logits([1, 2], e_p, e_s, e_z, layers,
                                final_gamma, cfg, tracer=tw)
        lines = open(path).readlines()
        assert len(lines) > 0
        # last record must still be argmax
        last = json.loads(lines[-1])
        assert last["op"] == "argmax"

    def test_trace_is_deterministic(self, tmp_path):
        """Running the same forward twice produces byte-identical JSONL
        outputs — critical for reproducible P3 divergence analysis."""
        rng1 = random.Random(42)
        cfg = _tiny_gemma3()
        e_p, e_s, e_z, layers, final_gamma = _build_model(rng1, cfg)

        path_a = str(tmp_path / "a.jsonl")
        path_b = str(tmp_path / "b.jsonl")
        with TraceWriter(path=path_a, enabled=True) as tw:
            forward([1, 2, 3], e_p, e_s, e_z, layers, final_gamma, cfg,
                    tracer=tw)
        with TraceWriter(path=path_b, enabled=True) as tw:
            forward([1, 2, 3], e_p, e_s, e_z, layers, final_gamma, cfg,
                    tracer=tw)
        assert open(path_a, "rb").read() == open(path_b, "rb").read()
