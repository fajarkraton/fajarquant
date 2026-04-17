#!/usr/bin/env python3
"""run_sim_trace.py — V30.SIM Phase P3.3

Run kernel_sim.forward() on real Gemma-3-1B weights loaded from disk_v8.img
and emit a JSONL trace file compatible with diff.py for kernel↔sim comparison.

Usage:
    python scripts/run_sim_trace.py --tokens 104 -o /tmp/sim_trace.jsonl
    python scripts/run_sim_trace.py --tokens 104,101,108,108,111 -o /tmp/sim_trace.jsonl  # "hello"
    python scripts/run_sim_trace.py --self-test  # synthetic tiny model, fast

Runtime warning: 1 token through real Gemma-3-1B (26 layers, d_model=1152,
ffn_dim=6912, vocab=262144) takes ~10-15 minutes in pure Python.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "tools"))
sys.path.insert(0, _HERE)

from kernel_sim import (  # noqa: E402
    TraceWriter, forward, activation_identity,
    TransformerConfig, LayerWeights,
    MODEL_TYPE_GEMMA3_1B,
)
from load_v8_weights import load_v8_model  # noqa: E402


def run_trace(
    tokens: List[int],
    model_path: str,
    output_path: str,
    *,
    verbose: bool = False,
) -> int:
    """Load weights, run forward, write trace JSONL. Returns 0 on success."""
    t0 = time.monotonic()
    if verbose:
        print(f"[sim-trace] Loading weights from {model_path}...", flush=True)
    m = load_v8_model(model_path)
    t_load = time.monotonic() - t0
    if verbose:
        print(f"[sim-trace] Loaded in {t_load:.1f}s — "
              f"{m.cfg.vocab_size} vocab, {m.cfg.d_model} d_model, "
              f"{m.cfg.n_layers} layers, {m.cfg.ffn_dim} ffn_dim")
        print(f"[sim-trace] Running forward for tokens={tokens} "
              f"({len(tokens)} token(s))...", flush=True)

    t1 = time.monotonic()
    with TraceWriter(path=output_path, enabled=True) as tw:
        result = forward(
            tokens=tokens,
            embed_packed=m.embed_packed,
            embed_scales=m.embed_scales,
            embed_zeros=m.embed_zeros,
            layers=m.layers,
            final_norm_gamma=m.final_norm_gamma,
            cfg=m.cfg,
            activation=activation_identity,
            tracer=tw,
        )
    t_fwd = time.monotonic() - t1

    if verbose:
        print(f"[sim-trace] Forward done in {t_fwd:.1f}s")
        print(f"[sim-trace] Result: token={result.last_token}, "
              f"score={result.last_score}")
        # Count records
        with open(output_path) as f:
            n_records = sum(1 for _ in f)
        print(f"[sim-trace] Wrote {n_records} trace records to {output_path}")
        print(f"[sim-trace] Total time: {time.monotonic() - t0:.1f}s")

    return 0


def run_self_test() -> int:
    """Quick smoke test with a tiny synthetic model."""
    import tempfile
    from kernel_sim import pack_nibbles

    cfg = TransformerConfig(
        vocab_size=32, d_model=4, n_heads=1, n_kv_heads=1,
        d_head=4, n_layers=2, ffn_dim=8,
        model_type=MODEL_TYPE_GEMMA3_1B,
    )
    # Tiny synthetic weights
    embed_n = cfg.vocab_size * cfg.d_model
    embed_packed = pack_nibbles([7] * embed_n)
    n_g = (embed_n + 127) // 128
    embed_scales = [1000] * n_g  # scale=0.001 → ×1e6 = 1000
    embed_zeros = bytes([8] * n_g)  # zero_point=8, so q-z = 7-8 = -1

    def make_layer() -> LayerWeights:
        def _mat(rows, cols):
            n = rows * cols
            p = pack_nibbles([8] * n)  # q=8, z=8 → w=0 → identity-like
            ng = (n + 127) // 128
            return p, [1000] * ng, bytes([8] * ng)

        qp, qs, qz = _mat(cfg.d_model, cfg.n_heads * cfg.d_head)
        kp, ks, kz = _mat(cfg.d_model, cfg.n_kv_heads * cfg.d_head)
        vp, vs, vz = _mat(cfg.d_model, cfg.n_kv_heads * cfg.d_head)
        op_, os_, oz = _mat(cfg.n_kv_heads * cfg.d_head, cfg.d_model)
        gp, gs, gz = _mat(cfg.d_model, cfg.ffn_dim)
        up, us, uz = _mat(cfg.d_model, cfg.ffn_dim)
        dp, ds, dz = _mat(cfg.ffn_dim, cfg.d_model)
        gamma = [1000] * cfg.d_model  # γ=1.0 in ×1000
        return LayerWeights(
            q_packed=qp, q_scales=qs, q_zeros=qz,
            k_packed=kp, k_scales=ks, k_zeros=kz,
            v_packed=vp, v_scales=vs, v_zeros=vz,
            o_packed=op_, o_scales=os_, o_zeros=oz,
            gate_packed=gp, gate_scales=gs, gate_zeros=gz,
            up_packed=up, up_scales=us, up_zeros=uz,
            down_packed=dp, down_scales=ds, down_zeros=dz,
            pre_attn_gamma=gamma,
            post_attn_gamma=gamma,
            pre_ffn_gamma=gamma,
            post_ffn_gamma=gamma,
        )

    layers = [make_layer() for _ in range(cfg.n_layers)]
    final_gamma = [1000] * cfg.d_model

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
        out_path = tf.name

    try:
        with TraceWriter(path=out_path, enabled=True) as tw:
            result = forward(
                tokens=[5],
                embed_packed=embed_packed,
                embed_scales=embed_scales,
                embed_zeros=embed_zeros,
                layers=layers,
                final_norm_gamma=final_gamma,
                cfg=cfg,
                activation=activation_identity,
                tracer=tw,
            )

        with open(out_path) as f:
            records = f.readlines()

        import json
        ops_seen = set()
        for line in records:
            r = json.loads(line)
            ops_seen.add(r["op"])

        # Expected: embed_lookup + 2 layers × 14 ops + final_rmsnorm + argmax
        expected = 1 + 2 * 14 + 1 + 1  # = 31
        ok = True

        def check(cond, msg):
            nonlocal ok
            status = "PASS" if cond else "FAIL"
            print(f"[sim-trace self-test] {status} {msg}")
            if not cond:
                ok = False

        check(len(records) == expected,
              f"record count = {len(records)} (expected {expected})")
        check("embed_lookup" in ops_seen, "embed_lookup present")
        check("pre_attn_rmsnorm" in ops_seen, "pre_attn_rmsnorm present")
        check("argmax" in ops_seen, "argmax present")
        check(result.last_token >= 0, f"argmax token={result.last_token} >= 0")

        return 0 if ok else 1
    finally:
        os.unlink(out_path)


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(
        description="Run kernel_sim forward and emit trace JSONL.",
    )
    p.add_argument("--tokens", type=str, default="104",
                   help="Comma-separated token IDs (default: 104 = 'h')")
    p.add_argument("--model", default=os.path.expanduser(
        "~/Documents/fajaros-x86/disk_v8.img"),
        help="Path to .fjm v8 model file")
    p.add_argument("-o", "--output", default="/tmp/sim_trace.jsonl",
                   help="Output JSONL path (default: /tmp/sim_trace.jsonl)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--self-test", action="store_true",
                   help="Run quick self-test with synthetic model")
    args = p.parse_args(argv)

    if args.self_test:
        return run_self_test()

    tokens = [int(t.strip()) for t in args.tokens.split(",")]
    return run_trace(tokens, args.model, args.output, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
