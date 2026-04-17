#!/usr/bin/env python3
"""load_v8_weights.py — V30.SIM Phase P3.2.P1

Load a FajarOS .fjm v8 model file (Gemma-3-1B group-wise 4-bit
quantization) and return a structure compatible with
`kernel_sim.forward()`. This is the missing piece that lets the
Python simulator run on the SAME weights the kernel uses — a
prerequisite for meaningful sim↔kernel parity in P3.3+.

── Source of truth ──────────────────────────────────────────────

`fajaros-x86/scripts/export_gemma3_v8.py` writes the .fjm file;
this script inverts that layout. Byte offsets and sizes are cross-
referenced against `kernel/compute/model_loader.fj` FJM_OFF_*
constants. Any drift between exporter, kernel, and this loader
will surface when the loader's asserts fire at load time.

── File layout ──────────────────────────────────────────────────

  [0..176)    Header (176 bytes, little-endian)
      offset  size   field
      0       4      magic                   = 0x314D4A46 ("FJM1")
      4       4      version                 = 8
      8       4      model_type              = 10 (Gemma3-1B)
      12      4      n_layers                = 26
      16      4      d_model                 = 1152
      20      4      n_heads                 = 4
      24      4      d_head                  = 256
      28      4      vocab_size              = 262144
      32      4      quant_bits              = 4
      36      4      total_size
      40      4      embed_off               = 176
      44      4      layer0_off
      48      4      lmhead_off              = embed_off (tied)
      52      4      n_kv_heads              = 1
      56      4      ffn_type                = 1 (gated)
      60      4      norm_type               = 1 (RMSNorm)
      64      4      ffn_dim                 = 6912
      68      4      rope_theta_local/1000   = 10
      72      4      eos_token               = 106
      140     4      final_norm_off
      172     2      quant_format            = 1 (group-wise)
      174     2      group_size              = 128

  [embed_off..layer0_off)   Embedding matrix (v8 serialization)
  [layer0_off..final_norm_off)
      26× layer blocks, each:
        16B header:   layer_id:i32, total_size:i32,
                      qkv_size:i32, ffn_size:i32
        qkv_size:     Q + K + V + O  (each v8-serialized)
        ffn_size:     gate + up + down  (each v8-serialized)
        norms:        input_layernorm (d_model × i64 ×1000)
                      post_attention_layernorm (d_model × i64 ×1000)
                      pre_feedforward_layernorm (d_model × i64 ×1000)
                      post_feedforward_layernorm (d_model × i64 ×1000)
                      q_norm (d_head × i64 ×1000)  [unused by sim]
                      k_norm (d_head × i64 ×1000)  [unused by sim]
  [final_norm_off..total_size)
      final RMSNorm gamma (d_model × i64 ×1000)

  Per matrix (v8 serialization):
      packed:  n/2 bytes (2 nibbles per byte, low nibble first)
      scales:  n_groups × 4 bytes (int32 ×1e6 fixed-point)
      zeros:   n_groups × 1 byte (u8, values 0..15)
      where n_groups = ceil(n_elems / 128)

── Why return raw bytes/lists, not decoded tensors ──────────────

`kernel_sim.LayerWeights` mirrors the kernel's memory layout: packed
bytes for quantized matrices + parallel scales/zeros arrays. The
simulator reads nibbles from the packed bytes via `int_ops
.read_nibble`, so keeping the bytes raw preserves 0-ULP parity
with the kernel.

── Usage ────────────────────────────────────────────────────────

  from load_v8_weights import load_v8_model

  weights = load_v8_model("disk_v8.img")
  # weights.cfg is a TransformerConfig
  # weights.embed_packed / .embed_scales / .embed_zeros
  # weights.layers is a list[LayerWeights] (len == n_layers)
  # weights.final_norm_gamma is Sequence[int]

  from kernel_sim import forward, TraceWriter
  with TraceWriter(path="/tmp/sim_trace.jsonl", enabled=True) as tw:
      result = forward(
          tokens=[1, 2, 3, 4, 5],
          embed_packed=weights.embed_packed,
          embed_scales=weights.embed_scales,
          embed_zeros=weights.embed_zeros,
          layers=weights.layers,
          final_norm_gamma=weights.final_norm_gamma,
          cfg=weights.cfg,
          tracer=tw,
      )
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from dataclasses import dataclass
from typing import List, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "tools"))

from kernel_sim import (  # noqa: E402
    TransformerConfig, LayerWeights,
    MODEL_TYPE_GEMMA3_1B, MODEL_TYPE_LLAMA, MODEL_TYPE_SMOLLM,
)


FJM_MAGIC = 0x314D4A46           # b"FJM1" little-endian
FJM_HEADER_SIZE = 176
GROUP_SIZE = 128


@dataclass
class V8Model:
    cfg: TransformerConfig
    embed_packed: bytes
    embed_scales: Sequence[int]
    embed_zeros: bytes
    layers: List[LayerWeights]
    final_norm_gamma: Sequence[int]


def _u32(data: bytes, off: int) -> int:
    return struct.unpack_from("<I", data, off)[0]


def _u16(data: bytes, off: int) -> int:
    return struct.unpack_from("<H", data, off)[0]


def _i64(data: bytes, off: int) -> int:
    return struct.unpack_from("<q", data, off)[0]


def _matrix_size(n_elems: int) -> int:
    """Compute on-disk size of a v8 serialized matrix in bytes."""
    packed = n_elems // 2
    n_groups = (n_elems + GROUP_SIZE - 1) // GROUP_SIZE
    return packed + n_groups * 5   # 4 B scale + 1 B zero per group


def _split_matrix(data: memoryview, off: int, n_elems: int
                  ) -> Tuple[int, bytes, List[int], bytes]:
    """Slice a v8 matrix out of `data` starting at `off`. Returns
    (next_off, packed_bytes, scales_list, zeros_bytes)."""
    packed_sz = n_elems // 2
    n_groups = (n_elems + GROUP_SIZE - 1) // GROUP_SIZE
    packed = bytes(data[off : off + packed_sz])
    off += packed_sz
    # Scales are int32 ×1e6 — stored as 4-byte little-endian. We keep
    # them as plain Python ints since kernel_sim.vecmat_v8 consumes
    # Sequence[int] via widen_u32 at read time.
    scales = list(struct.unpack_from(
        f"<{n_groups}i", data, off
    ))
    off += n_groups * 4
    zeros = bytes(data[off : off + n_groups])
    off += n_groups
    return off, packed, scales, zeros


def _read_norm(data: memoryview, off: int, length: int
               ) -> Tuple[int, List[int]]:
    """Read `length` i64 little-endian gamma values. Returns
    (next_off, gamma_list)."""
    gamma = list(struct.unpack_from(f"<{length}q", data, off))
    return off + length * 8, gamma


def load_v8_model(path: str) -> V8Model:
    """Parse a .fjm v8 file from disk and return all weights in a
    `kernel_sim`-compatible structure."""
    with open(path, "rb") as f:
        raw = f.read()
    data = memoryview(raw)

    # ── Header ────────────────────────────────────────────────
    magic = _u32(raw, 0)
    if magic != FJM_MAGIC:
        raise ValueError(
            f"Bad magic 0x{magic:08x} at {path!r}; expected FJM1 "
            f"(0x{FJM_MAGIC:08x}). Is this a valid .fjm file?"
        )
    version = _u32(raw, 4)
    if version != 8:
        raise ValueError(
            f"Unsupported FJM version {version}; loader only handles v8. "
            f"Use scripts/export_gemma3_v8.py to produce a v8 file."
        )
    model_type = _u32(raw, 8)
    n_layers   = _u32(raw, 12)
    d_model    = _u32(raw, 16)
    n_heads    = _u32(raw, 20)
    d_head     = _u32(raw, 24)
    vocab_size = _u32(raw, 28)
    quant_bits = _u32(raw, 32)
    total_size = _u32(raw, 36)
    embed_off  = _u32(raw, 40)
    layer0_off = _u32(raw, 44)
    lmhead_off = _u32(raw, 48)
    n_kv_heads = _u32(raw, 52)
    ffn_type   = _u32(raw, 56)
    ffn_dim    = _u32(raw, 64)
    final_norm_off = _u32(raw, 140)
    quant_fmt  = _u16(raw, 172)
    group_sz   = _u16(raw, 174)

    if quant_bits != 4:
        raise ValueError(f"v8 loader expects quant_bits=4, got {quant_bits}")
    if ffn_type != 1:
        raise ValueError(f"v8 loader expects gated FFN (ffn_type=1), got {ffn_type}")
    if quant_fmt != 1:
        raise ValueError(f"v8 loader expects quant_format=1 (group-wise), got {quant_fmt}")
    if group_sz != GROUP_SIZE:
        raise ValueError(f"v8 loader expects group_size={GROUP_SIZE}, got {group_sz}")
    # Some .fjm blobs (e.g. inside a 1 GB disk image) are larger than
    # total_size because the model is a prefix followed by padding /
    # unrelated bytes. Only fail if the blob is SHORTER than declared.
    if len(raw) < total_size:
        raise ValueError(
            f"File too small: {len(raw)} bytes, header says "
            f"total_size={total_size}"
        )
    if lmhead_off != embed_off:
        raise ValueError(
            f"v8 loader expects tied lmhead (lmhead_off==embed_off); "
            f"got {lmhead_off} vs {embed_off}"
        )

    cfg = TransformerConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_head=d_head,
        n_layers=n_layers,
        ffn_dim=ffn_dim,
        model_type=(MODEL_TYPE_GEMMA3_1B if model_type == 10
                    else MODEL_TYPE_LLAMA if model_type == 1
                    else MODEL_TYPE_SMOLLM),
    )

    # ── Embedding matrix ──────────────────────────────────────
    embed_n_elems = vocab_size * d_model
    embed_end, embed_packed, embed_scales, embed_zeros = _split_matrix(
        data, embed_off, embed_n_elems
    )
    # Sanity check: embed block should end at layer0_off.
    if embed_end != layer0_off:
        raise ValueError(
            f"embed block ends at {embed_end} but layer0_off={layer0_off}"
        )

    # ── 26 layer blocks ───────────────────────────────────────
    layers: List[LayerWeights] = []
    cur = layer0_off
    q_cols  = n_heads * d_head            # 1024
    kv_cols = n_kv_heads * d_head         # 256

    for li in range(n_layers):
        hdr = struct.unpack_from("<iiii", raw, cur)
        layer_id, total_sz, qkv_sz, ffn_sz = hdr
        if layer_id != li:
            raise ValueError(
                f"layer[{li}] header says id={layer_id} (mismatch)"
            )
        block_start = cur
        after_hdr = cur + 16

        # QKV + O
        off, q_p, q_s, q_z = _split_matrix(
            data, after_hdr, d_model * q_cols
        )
        off, k_p, k_s, k_z = _split_matrix(data, off, d_model * kv_cols)
        off, v_p, v_s, v_z = _split_matrix(data, off, d_model * kv_cols)
        off, o_p, o_s, o_z = _split_matrix(data, off, q_cols * d_model)
        if off - after_hdr != qkv_sz:
            raise ValueError(
                f"layer[{li}] QKV read {off - after_hdr} B, header qkv_size={qkv_sz}"
            )

        # FFN: gate + up + down
        off, g_p, g_s, g_z = _split_matrix(data, off, d_model * ffn_dim)
        off, u_p, u_s, u_z = _split_matrix(data, off, d_model * ffn_dim)
        off, dn_p, dn_s, dn_z = _split_matrix(data, off, ffn_dim * d_model)
        if off - after_hdr - qkv_sz != ffn_sz:
            raise ValueError(
                f"layer[{li}] FFN read {off - after_hdr - qkv_sz} B, header ffn_size={ffn_sz}"
            )

        # 4 block RMSNorms: input, post_attn, pre_ffn, post_ffn
        off, pre_attn_g   = _read_norm(data, off, d_model)
        off, post_attn_g  = _read_norm(data, off, d_model)
        off, pre_ffn_g    = _read_norm(data, off, d_model)
        off, post_ffn_g   = _read_norm(data, off, d_model)
        # q_norm + k_norm (d_head each) — unused by kernel_sim simplified
        # attention, but we skip past them so cur advances correctly.
        off += d_head * 8   # q_norm
        off += d_head * 8   # k_norm

        if off - block_start != total_sz:
            raise ValueError(
                f"layer[{li}] total {off - block_start} B, "
                f"header total_size={total_sz}"
            )
        cur = off

        layers.append(LayerWeights(
            q_packed=q_p, q_scales=q_s, q_zeros=q_z,
            k_packed=k_p, k_scales=k_s, k_zeros=k_z,
            v_packed=v_p, v_scales=v_s, v_zeros=v_z,
            o_packed=o_p, o_scales=o_s, o_zeros=o_z,
            gate_packed=g_p, gate_scales=g_s, gate_zeros=g_z,
            up_packed=u_p, up_scales=u_s, up_zeros=u_z,
            down_packed=dn_p, down_scales=dn_s, down_zeros=dn_z,
            pre_attn_gamma=pre_attn_g,
            post_attn_gamma=post_attn_g,
            pre_ffn_gamma=pre_ffn_g,
            post_ffn_gamma=post_ffn_g,
        ))

    if cur != final_norm_off:
        raise ValueError(
            f"layers end at {cur} but final_norm_off={final_norm_off}"
        )

    # ── Final RMSNorm ─────────────────────────────────────────
    _, final_norm_gamma = _read_norm(data, final_norm_off, d_model)

    return V8Model(
        cfg=cfg,
        embed_packed=embed_packed,
        embed_scales=embed_scales,
        embed_zeros=embed_zeros,
        layers=layers,
        final_norm_gamma=final_norm_gamma,
    )


# ── Self-test ───────────────────────────────────────────────────

def run_self_test(path: str) -> int:
    """Load disk_v8.img (or supplied path) and assert Gemma-3-1B
    invariants without running a full forward (which is slow in
    pure Python)."""
    if not os.path.isfile(path):
        print(f"[load-v8 self-test] fixture not found: {path}", file=sys.stderr)
        print(f"[load-v8 self-test] pass --path /path/to/v8.img to override",
              file=sys.stderr)
        return 1

    ok = True

    def check(cond: bool, msg: str) -> None:
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        print(f"[load-v8 self-test] {status} {msg}", file=sys.stderr)
        if not cond:
            ok = False

    m = load_v8_model(path)

    # Config matches Gemma-3-1B per export_gemma3_v8.py constants
    check(m.cfg.vocab_size == 262144,
          f"vocab_size = {m.cfg.vocab_size} (expected 262144)")
    check(m.cfg.d_model == 1152,
          f"d_model = {m.cfg.d_model} (expected 1152)")
    check(m.cfg.n_layers == 26,
          f"n_layers = {m.cfg.n_layers} (expected 26)")
    check(m.cfg.n_heads == 4,
          f"n_heads = {m.cfg.n_heads} (expected 4)")
    check(m.cfg.n_kv_heads == 1,
          f"n_kv_heads = {m.cfg.n_kv_heads} (expected 1)")
    check(m.cfg.d_head == 256,
          f"d_head = {m.cfg.d_head} (expected 256)")
    check(m.cfg.ffn_dim == 6912,
          f"ffn_dim = {m.cfg.ffn_dim} (expected 6912)")
    check(m.cfg.is_gemma3, "model_type should be Gemma3")

    # Embedding matrix sizes
    expected_embed_packed = (m.cfg.vocab_size * m.cfg.d_model) // 2
    check(len(m.embed_packed) == expected_embed_packed,
          f"embed_packed = {len(m.embed_packed)} "
          f"(expected {expected_embed_packed})")
    embed_n_groups = (m.cfg.vocab_size * m.cfg.d_model + 127) // 128
    check(len(m.embed_scales) == embed_n_groups,
          f"embed_scales len = {len(m.embed_scales)} "
          f"(expected {embed_n_groups})")
    check(len(m.embed_zeros) == embed_n_groups,
          f"embed_zeros len = {len(m.embed_zeros)} "
          f"(expected {embed_n_groups})")

    # Each layer shape
    check(len(m.layers) == m.cfg.n_layers,
          f"layers len = {len(m.layers)} (expected {m.cfg.n_layers})")

    L = m.layers[0]
    q_elems = m.cfg.d_model * m.cfg.n_heads * m.cfg.d_head     # 1152*1024
    kv_elems = m.cfg.d_model * m.cfg.n_kv_heads * m.cfg.d_head  # 1152*256
    ffn_elems = m.cfg.d_model * m.cfg.ffn_dim                    # 1152*6912

    check(len(L.q_packed) == q_elems // 2,
          f"layer0.q_packed = {len(L.q_packed)} (expected {q_elems // 2})")
    check(len(L.k_packed) == kv_elems // 2,
          f"layer0.k_packed = {len(L.k_packed)} (expected {kv_elems // 2})")
    check(len(L.gate_packed) == ffn_elems // 2,
          f"layer0.gate_packed = {len(L.gate_packed)} (expected {ffn_elems // 2})")
    check(len(L.pre_attn_gamma) == m.cfg.d_model,
          f"layer0.pre_attn_gamma len = {len(L.pre_attn_gamma)} (expected {m.cfg.d_model})")
    check(L.post_attn_gamma is not None and
          len(L.post_attn_gamma) == m.cfg.d_model,
          f"layer0.post_attn_gamma populated for Gemma-3")

    # Final norm
    check(len(m.final_norm_gamma) == m.cfg.d_model,
          f"final_norm_gamma len = {len(m.final_norm_gamma)} "
          f"(expected {m.cfg.d_model})")

    # Scale range sanity: int32 ×1e6, so for Gemma-3 weights (typical
    # range ~0.01-0.1) the int-scale should be in roughly 10..100000.
    # Any scale == 0 is a bug (would zero-out the group).
    n_zero_scales = sum(1 for s in L.q_scales if s == 0)
    check(n_zero_scales == 0,
          f"layer0.q_scales has {n_zero_scales} zero scales (should be 0)")

    return 0 if ok else 1


# ── CLI ─────────────────────────────────────────────────────────

def _main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(
        description="Load a FajarOS .fjm v8 model for kernel_sim.",
    )
    p.add_argument("--path", default=os.path.expanduser(
        "~/Documents/fajaros-x86/disk_v8.img"),
        help="path to .fjm v8 file (default: ~/Documents/fajaros-x86/disk_v8.img)")
    p.add_argument("--self-test", action="store_true",
                   help="run invariant assertions against --path")
    args = p.parse_args(argv)

    if args.self_test:
        return run_self_test(args.path)

    m = load_v8_model(args.path)
    print(f"Loaded {args.path}:")
    print(f"  cfg: {m.cfg}")
    print(f"  embed_packed: {len(m.embed_packed) / 1024 / 1024:.1f} MB")
    print(f"  layers: {len(m.layers)}")
    total_layer_packed = sum(
        len(L.q_packed) + len(L.k_packed) + len(L.v_packed) +
        len(L.o_packed) + len(L.gate_packed) + len(L.up_packed) +
        len(L.down_packed)
        for L in m.layers
    )
    print(f"  total layer-packed: "
          f"{total_layer_packed / 1024 / 1024:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
