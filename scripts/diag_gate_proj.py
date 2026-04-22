#!/usr/bin/env python3
"""V30 P3.6 diagnostic: verify sim gate_proj internal consistency.

Runs the sim forward for BOS token through layer 0 ONLY, captures raw
pre_ffn_rmsnorm and gate_proj vectors, and verifies:
1. pre_ffn_rmsnorm FNV hash matches trace (0x4c878ba6b9caa162)
2. gate_proj recomputed from captured pre_ffn_rmsnorm matches trace
3. Element-level gate_proj output for comparison with kernel

Runtime: ~5 min (1 layer of Gemma-3-1B in pure Python).
"""
import os, sys, struct, time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "tools"))
sys.path.insert(0, _HERE)

from kernel_sim import int_ops
from kernel_sim.vecmat_v8 import vecmat_v8, V8_GROUP_SHIFT, V8_SCALE_FP
from kernel_sim.rmsnorm import rmsnorm, km_isqrt, GAMMA_MODE_GEMMA
from kernel_sim.transformer import (
    embed_lookup, simplified_attention_single_pos, layer_forward,
    gated_ffn, activation_gelu_tanh, NOOP_TRACER,
)
from kernel_sim.trace import FNV_OFFSET_64 as FNV_OFFSET, FNV_PRIME_64 as FNV_PRIME
from load_v8_weights import load_v8_model


def fnv_hash_i64_list(vals):
    """FNV-1a hash of a list of i64 values (matches kernel fjtrace_hash_i64_region)."""
    MASK64 = (1 << 64) - 1
    h = FNV_OFFSET & MASK64
    for v in vals:
        bs = struct.pack("<q", v)
        for b in bs:
            h = ((h ^ b) * FNV_PRIME) & MASK64
    return h


class VectorCapture:
    """Tracer that captures raw vectors by op name."""
    def __init__(self):
        self.vectors = {}

    def record(self, op, data, **kw):
        layer = kw.get("layer", -1)
        key = (layer, op)
        self.vectors[key] = list(data)


def main():
    disk = os.environ.get("DISK", "/home/primecore/Documents/fajaros-x86/disk_v8.img")
    print(f"[diag] Loading {disk}...")
    m = load_v8_model(disk)
    L = m.layers[0]
    cfg = m.cfg
    print(f"[diag] Loaded: {cfg.vocab_size} vocab, {cfg.d_model} d_model, {cfg.ffn_dim} ffn_dim")

    # Step 1: Embed BOS + Gemma scaling
    print("[diag] Step 1: embed + Gemma scaling...")
    x = embed_lookup(2, m.embed_packed, m.embed_scales, m.embed_zeros,
                     cfg.vocab_size, cfg.d_model)
    scale = km_isqrt(int_ops.mul_i64(cfg.d_model, 1_000_000))
    x = [int_ops.trunc_div_i64(int_ops.mul_i64(v, scale), 1000) for v in x]

    # Step 2: Run layer_forward for layer 0 with vector capture
    print("[diag] Step 2: layer_forward(layer=0) with GELU-tanh...")
    t0 = time.monotonic()
    cap = VectorCapture()
    x_out = layer_forward(
        x, L, cfg, activation=activation_gelu_tanh,
        tracer=cap, token_idx=0, layer_idx=0,
    )
    dt = time.monotonic() - t0
    print(f"[diag] Layer 0 done in {dt:.1f}s")

    # Step 3: Verify pre_ffn_rmsnorm hash
    preffn = cap.vectors.get((0, "pre_ffn_rmsnorm"))
    if preffn is None:
        print("[FAIL] pre_ffn_rmsnorm not captured!")
        return 1

    h_preffn = fnv_hash_i64_list(preffn)
    expected_preffn = 0x4c878ba6b9caa162
    match_preffn = h_preffn == expected_preffn
    print(f"\n[diag] pre_ffn_rmsnorm hash: 0x{h_preffn:016x}")
    print(f"[diag] expected (from trace): 0x{expected_preffn:016x}")
    print(f"[diag] MATCH: {match_preffn}")
    print(f"[diag] min={min(preffn)} max={max(preffn)} mean={sum(preffn)//len(preffn)}")

    # Step 4: Verify gate_proj hash
    gate = cap.vectors.get((0, "gate_proj"))
    if gate is None:
        print("[FAIL] gate_proj not captured!")
        return 1

    h_gate = fnv_hash_i64_list(gate)
    expected_gate = 0xcbece5fb526ebba9
    match_gate = h_gate == expected_gate
    print(f"\n[diag] gate_proj hash: 0x{h_gate:016x}")
    print(f"[diag] expected (from trace): 0x{expected_gate:016x}")
    print(f"[diag] MATCH: {match_gate}")
    print(f"[diag] min={min(gate)} max={max(gate)} mean={sum(gate)//len(gate)}")
    print(f"[diag] gate[0]={gate[0]} gate[1]={gate[1]} gate[-1]={gate[-1]}")

    # Step 5: If pre_ffn_rmsnorm matches, recompute gate_proj independently
    if match_preffn:
        print("\n[diag] Step 5: recompute gate_proj from verified pre_ffn_rmsnorm...")
        t1 = time.monotonic()
        gate2 = vecmat_v8(
            preffn, L.gate_packed, L.gate_scales, L.gate_zeros,
            cfg.d_model, cfg.ffn_dim,
        )
        dt2 = time.monotonic() - t1
        h_gate2 = fnv_hash_i64_list(gate2)
        match2 = h_gate2 == expected_gate
        print(f"[diag] recomputed gate hash: 0x{h_gate2:016x}")
        print(f"[diag] matches trace: {match2}")
        print(f"[diag] matches layer_forward gate: {h_gate2 == h_gate}")
        print(f"[diag] recomputed in {dt2:.1f}s")

        if h_gate2 != h_gate:
            # Find first differing element
            for i in range(len(gate)):
                if gate[i] != gate2[i]:
                    print(f"[diag] first diff at j={i}: layer_fwd={gate[i]} recomputed={gate2[i]}")
                    break
    else:
        print("\n[diag] pre_ffn_rmsnorm doesn't match — sim's attention path differs!")
        print("[diag] Cannot verify gate_proj independently.")

    # Step 5.5: Save raw vectors to files for later analysis
    import struct as st
    with open("/tmp/sim_preffn_L0.bin", "wb") as f:
        for v in preffn:
            f.write(st.pack("<q", v))
    with open("/tmp/sim_gate_L0.bin", "wb") as f:
        for v in gate:
            f.write(st.pack("<q", v))
    print(f"\n[diag] Saved /tmp/sim_preffn_L0.bin ({len(preffn)*8} bytes)")
    print(f"[diag] Saved /tmp/sim_gate_L0.bin ({len(gate)*8} bytes)")

    # Step 6: Dump first 20 + last 5 gate elements for kernel comparison
    print(f"\n[diag] gate_proj elements (first 20):")
    for i in range(20):
        print(f"  [{i:5d}] = {gate[i]}")
    print(f"  ...")
    for i in range(len(gate) - 5, len(gate)):
        print(f"  [{i:5d}] = {gate[i]}")

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"pre_ffn_rmsnorm hash match: {match_preffn}")
    print(f"gate_proj hash match:       {match_gate}")
    if match_preffn and match_gate:
        print("SIM IS INTERNALLY CONSISTENT — gate_proj matches trace.")
    elif match_preffn and not match_gate:
        print("BUG IN SIM: pre_ffn_rmsnorm correct but gate_proj doesn't match trace!")
    elif not match_preffn:
        print("ATTENTION PATH DIFFERS: pre_ffn_rmsnorm hash mismatch.")
        print("This means layer_forward computes a different attention result")
        print("than when the trace was generated. Check activation function.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
