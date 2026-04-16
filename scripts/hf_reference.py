#!/usr/bin/env python3
"""hf_reference.py — V30.SIM Phase P2.5

HuggingFace Gemma-3 float-reference forward, emitting per-op JSONL
records at the SAME 17 op boundaries as the Python simulator
(`fajarquant/tools/kernel_sim/trace.py`) and the kernel FJTRACE
(`fajaros-x86/kernel/compute/fjtrace.fj`).

Output schema is identical to the Python-sim schema v1, with
`dtype="f32"` instead of `"i64"`. This lets `diff.py` (planned P3.1)
compare three trace files line-for-line:

  * Python-int sim (kernel mirror)  — i64, from kernel_sim.forward()
  * Kernel-actual                   — i64, from FJTRACE serial → parser
  * HuggingFace float reference     — f32, from THIS script

── Capture strategy ─────────────────────────────────────────────

For ops that correspond to a single PyTorch submodule's output, a
standard `register_forward_hook` is used (14 of 17 ops). For the
remaining 3 ops that fall between submodules, we monkey-patch the
forward method of `Gemma3MLP` (ffn_hidden) and `Gemma3DecoderLayer`
(attn_residual, ffn_residual). Patching is scoped via a context
manager so the global class remains intact after the script exits.

Hook map:

  | Kernel op          | HF source                                 | Mech   |
  |--------------------|-------------------------------------------|--------|
  | embed_lookup       | Gemma3Model.embed_tokens output           | hook   |
  | pre_attn_rmsnorm   | DecoderLayer.input_layernorm out          | hook   |
  | q_proj             | Gemma3Attention.q_proj out                | hook   |
  | k_proj             | Gemma3Attention.k_proj out                | hook   |
  | v_proj             | Gemma3Attention.v_proj out                | hook   |
  | attn_out           | Gemma3Attention.o_proj out                | hook   |
  | post_attn_rmsnorm  | DecoderLayer.post_attention_layernorm out | hook   |
  | attn_residual      | DecoderLayer.forward (manual)             | patch  |
  | pre_ffn_rmsnorm    | DecoderLayer.pre_feedforward_layernorm    | hook   |
  | gate_proj          | Gemma3MLP.gate_proj out                   | hook   |
  | up_proj            | Gemma3MLP.up_proj out                     | hook   |
  | ffn_hidden         | Gemma3MLP.forward intermediate            | patch  |
  | down_proj          | Gemma3MLP.down_proj out                   | hook   |
  | post_ffn_rmsnorm   | DecoderLayer.post_feedforward_layernorm   | hook   |
  | ffn_residual       | DecoderLayer.forward (manual)             | patch  |
  | final_rmsnorm      | Gemma3TextModel.norm out                  | hook   |
  | argmax             | lm_head out → argmax(-1)                  | hook   |

The attention op boundary maps to `o_proj` output here, matching the
kernel's semantics where `attn_out` is the value after O-projection
(kernel/compute/transformer.fj:1483 `tfm_vecmat_auto(attn_data,
o_addr, q_dim, d_model, bits, cb_o, STFM_FFN_OUT)` emits attn_out).

── Usage ─────────────────────────────────────────────────────────

  # With a real checkpoint (any Gemma-3 text model):
  python3 scripts/hf_reference.py \\
      --model google/gemma-3-1b-it \\
      --prompt "hello" \\
      --output /tmp/hf_trace.jsonl

  # Synthetic tiny model for CI / smoke-test (no HF download):
  python3 scripts/hf_reference.py --dry-run -o /tmp/hf_trace.jsonl

  # Self-test (schema + hook-count invariants):
  python3 scripts/hf_reference.py --self-test

── Design notes ─────────────────────────────────────────────────

Per-token vs batched: the kernel runs one forward per prompt token
(no KV cache, single-position attention). HF natively batches all
prompt tokens through full attention. For kernel parity the HF
script runs prompt tokens ONE AT A TIME, disabling the KV cache, so
the recorded records are 1:1 comparable with the kernel emits.

Float serialization: TraceWriter's `_tensor_stats_float` already
computes IEEE754 f32 byte-level FNV-1a, so f32 records hash on a
different surface than i64 records (by design — diff tool compares
fields, not hashes, across dtypes).

`ffn_hidden` units: the kernel scales its fp×1000 hidden by /1000
after `gate * up`; HF's float is the raw `act(gate) * up`. For diff
comparison, one side must scale (multiply HF by 1000 or divide
kernel by 1000) — that's a P3.1 responsibility, not P2.5's.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "tools"))

# kernel_sim.trace is authoritative for the schema — reuse its
# OP_NAMES and TraceWriter so the output is guaranteed schema-valid.
from kernel_sim.trace import (  # noqa: E402
    OP_NAMES, SCHEMA_VERSION, TraceWriter,
)


# ── HF capture helpers ──────────────────────────────────────────

def _flatten_to_list(t) -> List[float]:
    """Detach, move to CPU, cast to float32, flatten, and return a
    Python list of floats. Emit-ready for TraceWriter."""
    import torch
    assert isinstance(t, torch.Tensor), type(t)
    flat = t.detach().to(dtype=torch.float32, device="cpu").contiguous().view(-1)
    return flat.tolist()


def _emit(
    tw: TraceWriter,
    op: str,
    tensor,
    *,
    token_idx: int,
    layer: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    values = _flatten_to_list(tensor)
    shape = [len(values)]
    tw.record(
        op, values, token_idx=token_idx, layer=layer,
        shape=shape, dtype="f32", extra=extra,
    )


class _PerTokenContext:
    """Holds mutable token_idx for hook closures (hooks are registered
    once but must know which prompt token the current forward is on)."""

    def __init__(self) -> None:
        self.token_idx: int = 0
        self.total_tokens: int = 0   # set by run_forward before loop


def _register_standard_hooks(model, ctx: _PerTokenContext, tw: TraceWriter) -> List[Any]:
    """Register 14 forward hooks mapping submodule outputs → ops."""
    handles = []

    # Find the text model (works for Gemma3ForCausalLM and TextModel).
    text_model = model.model if hasattr(model, "model") else model
    layers = text_model.layers

    def hook_emit(op: str, layer_idx: int):
        def _hook(_module, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            _emit(tw, op, t, token_idx=ctx.token_idx, layer=layer_idx)
        return _hook

    handles.append(text_model.embed_tokens.register_forward_hook(
        hook_emit("embed_lookup", -1)
    ))
    for layer_idx, layer in enumerate(layers):
        handles.append(layer.input_layernorm.register_forward_hook(
            hook_emit("pre_attn_rmsnorm", layer_idx)
        ))
        attn = layer.self_attn
        handles.append(attn.q_proj.register_forward_hook(
            hook_emit("q_proj", layer_idx)
        ))
        handles.append(attn.k_proj.register_forward_hook(
            hook_emit("k_proj", layer_idx)
        ))
        handles.append(attn.v_proj.register_forward_hook(
            hook_emit("v_proj", layer_idx)
        ))
        handles.append(attn.o_proj.register_forward_hook(
            hook_emit("attn_out", layer_idx)
        ))
        handles.append(layer.post_attention_layernorm.register_forward_hook(
            hook_emit("post_attn_rmsnorm", layer_idx)
        ))
        handles.append(layer.pre_feedforward_layernorm.register_forward_hook(
            hook_emit("pre_ffn_rmsnorm", layer_idx)
        ))
        mlp = layer.mlp
        handles.append(mlp.gate_proj.register_forward_hook(
            hook_emit("gate_proj", layer_idx)
        ))
        handles.append(mlp.up_proj.register_forward_hook(
            hook_emit("up_proj", layer_idx)
        ))
        handles.append(mlp.down_proj.register_forward_hook(
            hook_emit("down_proj", layer_idx)
        ))
        handles.append(layer.post_feedforward_layernorm.register_forward_hook(
            hook_emit("post_ffn_rmsnorm", layer_idx)
        ))

    handles.append(text_model.norm.register_forward_hook(
        hook_emit("final_rmsnorm", -1)
    ))

    if hasattr(model, "lm_head"):
        def argmax_hook(_module, _inp, out):
            import torch
            # Emit argmax only on the FINAL prompt token, matching
            # kernel semantics (kernel emits argmax once at end of
            # tfm_forward_stream after final_rmsnorm on last pos).
            if ctx.token_idx != ctx.total_tokens - 1:
                return
            t = out[0] if isinstance(out, (tuple, list)) else out
            logits = t[..., -1, :] if t.ndim >= 2 else t
            winner = int(torch.argmax(logits, dim=-1).item())
            best_score = float(logits.flatten()[winner].item())
            tw.record(
                "argmax", [float(winner)],
                token_idx=ctx.token_idx, layer=-1,
                shape=[1], dtype="f32",
                extra={"best_score": best_score,
                       "vocab_size": int(logits.shape[-1])},
            )
        handles.append(model.lm_head.register_forward_hook(argmax_hook))

    return handles


@contextlib.contextmanager
def _patch_decoder_layer_and_mlp(ctx: _PerTokenContext, tw: TraceWriter):
    """Monkey-patch Gemma3MLP.forward (for ffn_hidden) and
    Gemma3DecoderLayer.forward (for attn_residual + ffn_residual).
    Restored on context exit so other modules using these classes
    are unaffected after the script finishes."""
    from transformers.models.gemma3 import modeling_gemma3
    import torch

    orig_mlp_forward = modeling_gemma3.Gemma3MLP.forward
    orig_layer_forward = modeling_gemma3.Gemma3DecoderLayer.forward

    def patched_mlp_forward(self, x):
        # Re-implement forward inline so we can capture the hidden.
        # Mirrors the original: down_proj(act(gate_proj(x)) * up_proj(x)).
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = self.act_fn(gate) * up
        layer_idx = getattr(self, "_fjtrace_layer_idx", -1)
        _emit(tw, "ffn_hidden", hidden,
              token_idx=ctx.token_idx, layer=layer_idx)
        return self.down_proj(hidden)

    def patched_layer_forward(self, hidden_states, position_embeddings=None,
                              attention_mask=None, position_ids=None,
                              past_key_values=None, **kwargs):
        # Publish layer_idx onto the MLP so patched_mlp_forward knows
        # which layer to tag its ffn_hidden record with.
        self.mlp._fjtrace_layer_idx = self.layer_idx

        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h, _ = self.self_attn(
            hidden_states=h,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        h = self.post_attention_layernorm(h)
        h = residual + h
        _emit(tw, "attn_residual", h,
              token_idx=ctx.token_idx, layer=self.layer_idx)

        residual = h
        h = self.pre_feedforward_layernorm(h)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        h = residual + h
        _emit(tw, "ffn_residual", h,
              token_idx=ctx.token_idx, layer=self.layer_idx)
        return h

    modeling_gemma3.Gemma3MLP.forward = patched_mlp_forward
    modeling_gemma3.Gemma3DecoderLayer.forward = patched_layer_forward
    try:
        yield
    finally:
        modeling_gemma3.Gemma3MLP.forward = orig_mlp_forward
        modeling_gemma3.Gemma3DecoderLayer.forward = orig_layer_forward


def run_forward(
    model, tokenizer, prompt: str, tw: TraceWriter,
    *, max_tokens: int = 0,
) -> Tuple[List[int], List[int]]:
    """Run the HF model on `prompt` one token at a time, no KV cache,
    emitting all 17 ops per prompt token. Returns (prompt_ids,
    generated_ids). `max_tokens=0` means prefill-only."""
    import torch
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()

    ctx = _PerTokenContext()
    ctx.total_tokens = len(input_ids)
    handles = _register_standard_hooks(model, ctx, tw)
    try:
        with _patch_decoder_layer_and_mlp(ctx, tw):
            with torch.no_grad():
                # Prefill one token at a time to mirror kernel's
                # tfm_forward_stream loop. Pos embeddings derived from
                # the token's position, but for a tiny-sim smoke test
                # we just drive a sequence of single-token forwards.
                for i, tok in enumerate(input_ids):
                    ctx.token_idx = i
                    _ = model(
                        input_ids=torch.tensor([[tok]]),
                        use_cache=False,
                    )
                # max_tokens generation loop omitted for P2.5 (not
                # needed for prefill-vs-prefill diff).
    finally:
        for h in handles:
            h.remove()

    return input_ids, []


# ── Synthetic tiny config for --dry-run ─────────────────────────

def _build_synthetic_model():
    """Build a Gemma3ForCausalLM with a tiny random-initialized
    config so `run_forward` can exercise all 17 ops in <1s without
    downloading a real checkpoint. Size is chosen to keep LM head
    tractable and avoid HF tokenizer complexity."""
    from transformers.models.gemma3 import modeling_gemma3
    from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
    import torch

    cfg = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling={"rope_type": "default"},
        hidden_activation="gelu_pytorch_tanh",
        attn_logit_softcapping=None,
        query_pre_attn_scalar=8,
        sliding_window=64,
        layer_types=["full_attention", "full_attention"],
        attention_bias=False,
        use_bidirectional_attention=False,
        tie_word_embeddings=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = modeling_gemma3.Gemma3ForCausalLM(cfg)
    model.eval()

    class _Tok:
        """Minimal tokenizer stub — enough for run_forward."""
        def __call__(self, s, **kw):
            import torch
            ids = [1, 2, 3, 4, 5][:len(s) + 1] if s else [1]
            return type("_X", (), {"input_ids": torch.tensor([ids])})()

    return model, _Tok()


# ── Self-test ────────────────────────────────────────────────────

def run_self_test() -> int:
    """Full round-trip: tiny synthetic Gemma3 + run_forward + parse
    the emitted JSONL. Asserts all 17 ops hit, schema valid, record
    count matches the plan's 14-ops-per-layer formula."""
    import io
    import importlib.util

    # Ensure parser is importable — it carries the authoritative
    # validator. Parser lives in fajaros-x86 scripts/.
    parser_candidates = [
        os.path.join(_ROOT, "..", "fajaros-x86", "scripts",
                     "parse_kernel_trace.py"),
        os.path.join(os.path.expanduser("~"), "Documents",
                     "fajaros-x86", "scripts", "parse_kernel_trace.py"),
    ]
    parser_path = next((p for p in parser_candidates
                        if os.path.isfile(p)), None)
    parser = None
    if parser_path is not None:
        spec = importlib.util.spec_from_file_location(
            "fjtrace_parser", parser_path)
        parser = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parser)  # type: ignore[union-attr]

    model, tok = _build_synthetic_model()

    import tempfile
    with tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False
    ) as fh:
        path = fh.name
    try:
        with TraceWriter(path=path, enabled=True) as tw:
            run_forward(model, tok, "hi", tw)
        with open(path, encoding="utf-8") as f:
            text = f.read()
    finally:
        os.unlink(path)
    lines = [l for l in text.splitlines() if l.strip()]

    n_layers = 2
    n_tokens = 3  # _Tok stub yields 3 ids for "hi"
    expected = n_tokens * (1 + n_layers * 14 + 1) + 1
    expected_ops = set(OP_NAMES)

    ok = True

    def check(cond, msg):
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        print(f"[hf-ref self-test] {status} {msg}", file=sys.stderr)
        if not cond:
            ok = False

    check(len(lines) == expected,
          f"expected {expected} records (N={n_tokens}, L={n_layers}, "
          f"14 ops/layer + embed + final_norm + argmax), got {len(lines)}")

    records = [json.loads(l) for l in lines]
    ops_seen = {r["op"] for r in records}
    check(ops_seen == expected_ops,
          f"missing ops: {expected_ops - ops_seen}; "
          f"extra: {ops_seen - expected_ops}")

    # All records schema_version=1
    check(all(r["schema_version"] == 1 for r in records),
          "not all records schema_version=1")
    # All records dtype=f32
    check(all(r["dtype"] == "f32" for r in records),
          "not all records dtype=f32")
    # All hashes are hex
    check(all(r["hash"].startswith("0x") and len(r["hash"]) == 18
              for r in records),
          "hash format invalid")

    # Parser round-trip if available: feed our JSONL back through it.
    if parser is not None:
        in_buf = io.StringIO(text)
        out_buf = io.StringIO()
        stats = parser.parse_stream(in_buf, out_buf)
        check(stats.records == len(lines),
              f"parser passed through {stats.records}/{len(lines)} records")
        check(stats.malformed == 0,
              f"parser flagged {stats.malformed} malformed")
    else:
        print("[hf-ref self-test] skip parser round-trip "
              "(parse_kernel_trace.py not found)", file=sys.stderr)

    return 0 if ok else 1


# ── CLI ──────────────────────────────────────────────────────────

def _main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(
        description="HF Gemma-3 float-reference forward with per-op "
                    "JSONL trace capture.",
    )
    p.add_argument("--model", default=None,
                   help="HF checkpoint path or HF hub id "
                        "(e.g. google/gemma-3-1b-it)")
    p.add_argument("--prompt", default="hello",
                   help="prompt string (default: 'hello')")
    p.add_argument("-o", "--output", default="-",
                   help="output JSONL path, or '-' for stdout "
                        "(default: -)")
    p.add_argument("--dry-run", action="store_true",
                   help="use a tiny synthetic Gemma-3 config instead "
                        "of loading a real checkpoint")
    p.add_argument("--self-test", action="store_true",
                   help="run internal test against synthetic model")
    args = p.parse_args(argv)

    if args.self_test:
        return run_self_test()

    out_path = args.output
    if out_path == "-":
        print("[hf-ref] Please pass -o <path>. Stdout streaming requires "
              "TraceWriter file-handle abstraction (future work).",
              file=sys.stderr)
        return 2

    if args.dry_run:
        model, tokenizer = _build_synthetic_model()
    else:
        if args.model is None:
            print("[hf-ref] --model REQUIRED (or pass --dry-run for smoke test)",
                  file=sys.stderr)
            return 2
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)
        model.eval()

    with TraceWriter(path=out_path, enabled=True) as tw:
        ids, _ = run_forward(model, tokenizer, args.prompt, tw)

    print(f"[hf-ref] {len(ids)} prompt tokens traced → {out_path}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
