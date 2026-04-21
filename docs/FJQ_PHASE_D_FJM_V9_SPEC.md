# FJM v9 Format Spec — Phase D / MatMul-Free LM

> **Date:** 2026-04-21 | **Audience:** FajarOS kernel + Phase D `.fjm` exporter | **Depends on:** v8 spec in `fajaros-x86/kernel/compute/model_loader.fj` lines 60-76 + `FJQ_PHASE_D_OPS.md` + `FJQ_PHASE_D_ARCH.md` | **Replaces:** nothing (additive — v1-v8 readers continue to work)

## 0. Why v9 exists

Phase D's MatMul-Free architecture (HGRNBit / MLGRU + GLU + BitLinear)
is structurally different from v8's Gemma 3 (attention + RoPE + GQA +
SWA). v8 readers cannot interpret a Phase D model:

- No QKV projection — replaced by 4 MLGRU gate matrices (f, c, g, o)
- No FFN — replaced by 3 GLU matrices (g, u, d) at width ratio 8/3
- No RoPE / sliding window
- All matrix weights are ternary {-1, 0, +1}, not 2/4-bit codebook
- Per-matrix β (absmean) + per-channel γ_x (frozen QAT calibration)

v9 adds a new MODEL_TYPE + restructured per-layer block layout while
preserving the v1-v8 header skeleton (additive, not breaking).

## 1. Header changes (offset 0x00 — 0xC0)

v9 keeps the v8 176-byte header skeleton exactly. Only two bytes
change semantics:

| Offset | v8 meaning | v9 meaning |
|---|---|---|
| 0x04 | version = 8 | **version = 9** |
| 0x05 | model_type = 10 (Gemma 3) | **model_type = 11 (MatMul-Free LM)** |
| 0xAC | quant_format (0=single, 1=group-wise) | **quant_format = 2 (ternary BitNet absmean)** |
| 0xAE | group_size (128 for v8) | **group_size = 0 (per-matrix β, not group-wise)** |
| 0x98 (rope_global) | 1,000,000 (Gemma 3) | **0 (NoPE, recurrence carries position)** |
| 0xA0 (sliding_window) | 512 (Gemma 3) | **0 (no sliding window)** |
| 0xA4 (sliding_pattern) | 6 (Gemma 3 every-6th-layer global) | **0** |

All other v8 fields keep their v8 semantics. n_layers / d_model /
vocab_size / n_heads etc. read identically.

**v9 readers detect Phase D models by the tuple `(version=9,
model_type=11, quant_format=2)`.** Mismatched tuples MUST surface a
load error rather than fall through to v8 logic.

## 2. Per-matrix β scalars (new section, written before layer blocks)

v9 introduces a fixed-position table of FP32 absmean scalars β_i, one
per BitLinear in the model. Computed once at QAT freeze, baked into
the export, never re-derived at inference time (per §6.9 R4
calibration once).

Layout immediately following the 176-byte header, before the first
layer block:

```
0x00B0  table_size : u32   = 4 × n_betas    (bytes)
0x00B4  n_betas    : u32   = (4 × L) + (3 × L) + 1   = 7L + 1
                              ↑ MLGRU: 4 BitLinears (f,c,g,o) per layer
                              ↑ GLU:   3 BitLinears (g,u,d) per layer
                              ↑ +1 for the LM head (also a BitLinear)
0x00B8  β[0]       : f32
0x00BC  β[1]       : f32
        ...
0x00B8 + 4·n_betas
```

Indexing convention (deterministic, matches PyTorch state_dict order):

```
For layer ℓ in 0..L:
  index 7ℓ + 0  : MLGRU.f_proj.β
  index 7ℓ + 1  : MLGRU.c_proj.β
  index 7ℓ + 2  : MLGRU.g_proj.β
  index 7ℓ + 3  : MLGRU.o_proj.β
  index 7ℓ + 4  : GLU.g_proj.β
  index 7ℓ + 5  : GLU.u_proj.β
  index 7ℓ + 6  : GLU.d_proj.β
index 7L + 0    : LM_head.β
```

Total table size at Stretch (L=24): 4 × (7·24 + 1) = 676 bytes. At
Mini: 4 × 43 = 172 bytes.

## 3. Per-channel γ_x calibration (new section, optional)

If `quant_format = 2`, Phase D readers MUST also accept an optional γ_x
table immediately after the β table. The γ_x table holds frozen
per-channel activation absmax values from the QAT calibration pass —
needed when the model uses periodic-recal QAT (§6.9 R5 contribution
3).

Layout:

```
γ_x_offset = 0x00B8 + 4·n_betas

γ_x_offset + 0x00  table_size  : u32   = 4 × Σ(in_features per BitLinear)
γ_x_offset + 0x04  n_entries   : u32   = same as Σ
γ_x_offset + 0x08  γ_x[0]      : f32
                   ...
```

Same indexing as β (per-BitLinear), but each entry is a vector of
length `in_features` (e.g. `d_model` or `8d/3`), packed contiguously.

If `n_entries = 0` (empty table written explicitly), γ_x is computed
per-call at inference (BitNet baseline behavior).

## 4. Per-layer block layout (replaces v8 QKV + FFN block)

v8 layer block: 16-byte header + QKV-packed weights + FFN-packed
weights. v9 replaces this entirely:

```
+0x00  layer_id        : u32     = ℓ
+0x04  total_size      : u32     = bytes from this header to next
+0x08  mlgru_size      : u32     = bytes for the 4 MLGRU BitLinears
+0x0C  glu_size        : u32     = bytes for the 3 GLU BitLinears
+0x10  rmsnorm_offset  : u32     = byte offset (within block) to RMSNorm γ
+0x14  reserved        : u32     = 0 (alignment)
+0x18  ── start of MLGRU weights ──
       MLGRU.f_proj : ternary [d, d]
       MLGRU.c_proj : ternary [d, d]
       MLGRU.g_proj : ternary [d, d]
       MLGRU.o_proj : ternary [d, d]
       ── start of GLU weights ──
       GLU.g_proj   : ternary [d, 8d/3]
       GLU.u_proj   : ternary [d, 8d/3]
       GLU.d_proj   : ternary [8d/3, d]
       ── RMSNorm parameters ──
       (no γ — Phase D drops learnable γ per FJQ_PHASE_D_OPS.md §2 Op 2)
       (only RMSNorm.eps stored, as a single FP32 scalar)
+ rmsnorm_offset       :  rmsnorm_eps : f32  (always 1e-6)
```

Per-layer header size: **24 bytes** (vs v7/v8 16 bytes — added
`rmsnorm_offset` + reserved padding).

## 5. Ternary packing convention

Each ternary matrix entry is **2 bits**, packed 4-per-byte little-endian:

```
byte 0:  bits 1-0 = W[0],  bits 3-2 = W[1],  bits 5-4 = W[2],  bits 7-6 = W[3]
byte 1:  bits 1-0 = W[4], ...
```

Encoding:
```
00 = -1
01 =  0
10 = +1
11 = reserved (treated as 0 if encountered; producer MUST never write)
```

Total bytes per matrix = `ceil(out_features × in_features / 4)`.
Matrices are stored row-major (`out_features` outer, `in_features`
inner) to match PyTorch's `nn.Linear.weight` shape `(out, in)`.

Total weight bytes for Stretch (L=24, d=1024, V=32K):
- Per-layer MLGRU: 4 × 1024² × 2 / 8 = 1.0 MiB × 4 = 4.0 MiB
- Per-layer GLU: 3 × 1024 × 2731 × 2 / 8 = 2.0 MiB × 3 = 6.0 MiB
- Per-layer total: 10 MiB
- All layers: 240 MiB
- LM head: 1024 × 32768 × 2 / 8 = 8.0 MiB
- Embedding (FP16): 32768 × 1024 × 2 = 64 MiB
- **Total .fjm v9 file size at Stretch: ~315 MiB** (vs v8 Gemma 3 1B at ~500 MiB)

## 6. Embedding storage

Phase D uses an FP16 embedding table — same as v7/v8 — stored
immediately after the last layer block. Shape `(V, d)`,
row-major. **Not** ternary; the embedding lookup is too magnitude-
sensitive for ternary in BitNet/MatMul-Free papers.

LM head: ternary BitLinear, stored in the per-layer-style block at the
end of the file (model_loader treats it as "layer L+1" with mlgru_size=0
+ glu_size = 0 + lm_head_size = ternary[d, V]). Tied with embedding is
NOT supported in v9 (ternary head ≠ FP embedding).

## 7. Reader detection logic (Fajar Lang kernel pseudocode)

```fajar
@kernel fn mdl_parse_header(src_addr: i64) -> i32 {
    let version = mdl_read_u32(src_addr + FJM_OFF_VERSION) & 0xFF
    let model_type = mdl_read_u32(src_addr + FJM_OFF_MODEL_TYPE) & 0xFF
    let quant_format = mdl_read_u16(src_addr + FJM_OFF_QUANT_FORMAT)

    if version == 9 && model_type == 11 && quant_format == 2 {
        return MDL_FORMAT_PHASE_D    // Route to tfm_matmulfree forward
    } else if version == 8 && quant_format == 1 {
        return MDL_FORMAT_GEMMA3_V8  // Existing Gemma 3 path
    } else if version >= 7 {
        return MDL_FORMAT_V7         // SmolLM/legacy path
    } else if version >= 3 {
        return MDL_FORMAT_LEGACY     // v3-v6
    } else {
        return -1                    // Unsupported
    }
}
```

## 8. Exporter side (Phase D Python)

`intllm/export.py` (C.P5.4 deliverable, NOT in this commit) will:

1. Read trained PyTorch state_dict
2. Quantize each BitLinear weight to ternary (using upstream's
   `weight_quant` or our QAT-frozen β values)
3. Pack ternary into 2-bit format per §5
4. Compute / read per-matrix β + per-channel γ_x (if periodic-recal
   ablation enabled)
5. Write header + β table + (optional) γ_x table + per-layer blocks +
   embedding + LM head
6. Emit a `.fjm.v9.json` sidecar with shapes + sizes for diagnostic
   loading verification

## 9. Backwards compatibility check

v9 readers MUST handle v1-v8 files (all readers handle older versions
in FajarOS Nova). Since v9 only ADDS new code paths (MDL_FORMAT_PHASE_D
branch), existing v1-v8 logic is untouched — pure additive change.

A v8 reader that encounters a v9 file MUST surface a clean error,
not silently fall through to v7 logic. Current `mdl_parse_header` in
`model_loader.fj:218` uses `version >= 7` → would fall through; **v9
exporter must verify v8 reader rejects v9 with a load error before
shipping**.

## 10. Implementation cost estimate

Kernel-side (fajaros-x86) per FJQ_PHASE_D_OPS.md §6:
- `kernel/compute/matmulfree.fj` — ~300 LOC (BitLinear ⊛, MLGRU update, absmax quant)
- `kernel/compute/matmulfree_lut.fj` — ~80 LOC (σ + SiLU LUT + tables)
- `kernel/model/fjm_v9.fj` — ~120 LOC (parser routing + Phase D layer block layout)
- `kernel/model/tfm_matmulfree.fj` — ~200 LOC (forward orchestration peer to `tfm_gemma3_stream`)
- Extends `kmatrix.fj` with `km_rmsnorm_no_gamma` — ~30 LOC

Total ~700 LOC. Estimated 2 weeks to write + 2 weeks to debug to
kernel-path cleanliness, mirroring V28.1+V28.2 Gemma 3 pace.

Python-side (fajarquant/python/phase_d):
- `intllm/export.py` — ~150 LOC (state_dict → .fjm v9)
- `tests/test_export.py` — ~100 LOC (round-trip + numerical-parity tests)

## 11. §6.8 Plan Hygiene checklist

- [x] R1 — Pre-flight: read v8 source, confirmed v9 is purely additive
- [x] R2 — Verifications via runnable check: a "v9 sample export + v8
       reader-rejection test" is the gate for shipping (C.P5.4)
- [x] R3 — Prevention layer: §9 mandates v8 reader rejection test
       before v9 ships, otherwise silent format ambiguity could mask
       export bugs
- [x] R4 — Numbers cross-checked: total file size at Stretch (315 MiB)
       hand-computed against §5 weight breakdown
- [x] R5 — Variance tag in commit
- [x] R6 — This file IS the .fjm v9 mechanical decision gate
- [x] R7 — No public artifact implications (internal design doc)
- [x] R8 — Multi-repo state confirmed pre-commit

## 12. What this un-gates

- C.P5.4 (.fjm v9 parser in fajaros-x86) — has a concrete spec to
  implement against
- C.P5.5 (model-load → embed-load → tok-load → ask workflow) — has
  the format the loader will read
- intllm/export.py (Python exporter) — has the byte layout to write
- Future: format negotiation between Phase D Python + FajarOS kernel
  is now mechanically defined, not informal
