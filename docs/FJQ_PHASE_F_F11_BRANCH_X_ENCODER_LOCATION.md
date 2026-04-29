# Phase F F.11 Branch X — Encoder Location Findings (v1.0)

> **Status:** Branch X audit COMPLETE. Located the upstream
> llama.cpp fork; the TL2 weight-encoder logic is **NOT** in any
> file we found, BUT a structural insight emerged: 3 of 4 Mini
> shapes use a 2-bpw format that matches V31's existing wire
> format. Branch X may need to handle only 1 shape, not 4.

---

## 1. Upstream fork located

`microsoft/BitNet/.gitmodules`:

```
[submodule "3rdparty/llama.cpp"]
    path = 3rdparty/llama.cpp
    url = https://github.com/Eddie-Wang1120/llama.cpp.git
    branch = merge-dev
```

Searched 5 branches (`merge-dev`, `ms-i2`, `i2`, `bitnet`,
`master`). Two branches surface: `ms-i2` and `merge-dev`. Both
contain BitNet-related conversion + .py scripts but **no
identifiable TL2 wire-encoder C/C++ function**.

| Path | Branch | Relevance |
|---|---|---|
| `convert-ms-to-gguf-bitnet.py` (67KB) | `ms-i2` | Has `transform_to_i2(x)` — maps ternary {-1,0,+1}→uint8 {1,2,3} (8 bpw, NOT TL2's 1.67 bpw) |
| `convert_hf_to_gguf.py` (186KB) | `merge-dev` | Has `BitnetModel.weight_quant` — float→ternary, no packing |
| `ggml/src/ggml-quants.c` (652KB) | `merge-dev` | Standard llama.cpp IQ2_S/IQ2_XXS/IQ2_XS — NOT TL2 |
| `ggml.c:2602` (732KB) | `ms-i2` | Type 30 (`GGML_TYPE_I2`): `nbytes = nbytes / 4 + 32` (= 2 bpw + 32-byte trailer). Matches V31 Phase D's 2-bit packing! |
| `spm-sources/ggml-bitnet-lut.cpp` (32 B) | `merge-dev` | SwiftPM stub, not real source |

**Where the TL2 1.67-bpw encoder actually lives is still unclear.**
microsoft/BitNet's `ggml_bitnet_transform_tensor` (which we
vendored) only stores `tensor->data` as-is — does NOT permute. So
the layout was already permuted upstream of `transform_tensor`,
either at GGUF write-time or by a C function we haven't found.

## 2. Load-bearing insight: 3 of 4 Mini shapes use 2-bpw format

Per the format formula at `codegen_tl2.py:659` /
`bitnet-lut-kernels-tl2.h:1444`:

```c
int nbytes = (k - 256) * m / 3 * 5 / 8 + 256 * m / 2 * 4 / 8;
```

For each Mini shape:

| Shape | k | TL2 region (k − 256) | Tail region (last 256) | Total format |
|---|---|---|---|---|
| (256, 256) | 256 | 0 cols | 256 cols × 2 bpw | **All 2 bpw** = 16,384 B |
| (1536, 256) | 256 | 0 cols | 256 cols × 2 bpw | **All 2 bpw** = 98,304 B |
| (256, 768) | 768 | 512 cols × 1.67 bpw | 256 cols × 2 bpw | **Mixed** = 26,624 + 16,384 = 43,008 B |
| (32768, 256) | 256 | 0 cols | 256 cols × 2 bpw | **All 2 bpw** = 2,097,152 B |

**Three of four Mini shapes (attn projections × 4 sites + lm_head)
have NO TL2 1.67-bpw region.** They're entirely in the "two-path"
2-bit format, which is structurally identical to V31 Phase D's
2-bit-per-weight packing (4 weights/byte LE).

Confirmed via `ggml.c:2602` on `ms-i2` branch: type 30 has
`nbytes = nbytes / 4 + 32` — 4 elements/byte = 2 bpw, plus
32-byte trailer. **Same as V31's `(out × in / 4)` byte count
plus a footer.**

The exact byte permutation INSIDE may still differ (V31 row-major
vs TL2 tile-organized), but the COMPRESSION RATIO is identical.

## 3. Implications for Branch X scope

Branch X originally scoped at ~8-12h to port a generic encoder
covering all 4 shapes. With this insight:

- **(256, 256), (1536, 256), (32768, 256):** Maybe directly
  usable from V31 with a tile-permute step (no triplet
  encoding). If V31's row-major is already the 2-bpw format
  the kernel expects, only the tile-tiling needs adjustment.
  Estimated: 2-3h to verify + adapt.
- **(256, 768) — mlp.down_proj:** Has 512 columns of true TL2
  1.67-bpw region. Encoder needed there OR fall back to scalar
  for this single shape. mlp.down_proj is 1 of ~6 BitLinear
  sites per layer × 6 layers = 6 sites total at this shape.
  The other 36 sites use 2-bpw shapes.

**Three concrete branches under "Branch X":**

### 3.1 Branch X-narrow (~2-3h)

Adapt the V31 row-major bytes to TL2 tile-organized layout
(reshuffle bytes by tile blocks; no triplet encoding because
the 2-bpw format IS the data). Test on a (256, 256) parity
case. If parity passes, ship for the 3 V31-compatible shapes.
Use scalar fallback for mlp.down_proj.

**Risk:** assumption "V31 2-bpw == TL2 two-path 2-bpw at the
byte level" may fail if upstream uses a different bit ordering
within the byte (e.g., big-endian per-byte instead of LE). Then
an extra bit-permutation pass is needed.

**Verdict if it works:** F.11.5 + F.11.6 unblock for 3 of 4
Mini shapes (the most numerically common ones), ~80% of
inference time covered by TL2.

### 3.2 Branch X-mid (~5-6h)

X-narrow + add the triplet encoder for the mlp.down_proj
shape. Covers all 4 Mini shapes.

### 3.3 Branch X-full (~8-12h, original scope)

Find the upstream encoder definitively (audit submodules in
the merge-dev branch's `3rdparty/`, look at `examples/` build
hooks, possibly check `gguf-py/` Python utilities) OR
reverse-engineer fully from kernel strides. Port everything.

## 4. What WAS confirmed

- Upstream fork: `Eddie-Wang1120/llama.cpp` (active, multiple
  branches)
- `transform_to_i2`: simple Python ternary→{1,2,3} uint8 mapping
  at `convert-ms-to-gguf-bitnet.py:723` (8 bpw, doesn't help us)
- `BitnetModel.weight_quant`: ternary quantization without
  packing at `convert_hf_to_gguf.py:1610` (32-bit signs)
- `GGML_TYPE_I2` size: 2 bpw + 32-byte trailer (matches V31)
- No `quantize_row_tl2` or `three_lut_ctor`-style encoder
  function found in the searched files

## 5. What WAS NOT found

- The C++ function (if any) that emits TL2 tile-organized
  (`a[i * KK/2 + ai * 32]`) byte ordering from row-major
  ternary input
- Whether V31's `00=-1, 01=0, 10=+1` 2-bit encoding matches
  TL2's two-path 2-bit encoding bit-for-bit
- Whether the in-byte ordering matches (LE 4-elements/byte vs
  some other layout)

## 6. Recommendation

**Branch X-narrow first** (~2-3h):

1. Verify byte-level layout match between V31 2-bpw and TL2
   two-path 2-bpw. Method: encode a known 32-element ternary
   array via V31, run it through `fjq_tl2_qgemm_lut` for
   (256, 256), compare output to scalar baseline. ~1h.
2. If layouts match: write the tile-permute pass (row-major
   → tile-organized; no triplet encoding) in Rust, ~50-100
   LOC. ~1h.
3. Write `parity_real_mlp` test at (256, 256). ~30min.

If step 1 surfaces a byte-layout mismatch: add a bit-swap pass
(~30min). If still failing: escalate to Branch X-mid (port the
triplet encoder for the (256, 768) case and accept the test
gap for shapes 1-3).

**Default per resume protocol:** Branch X-narrow as the next
"lanjutkan" increment. ~3h fits comfortably in remaining 11h
F.11 budget.

If user prefers to STOP F.11 here and pivot, Branch Z (F.13
dispatch) remains a clean exit per
`FJQ_PHASE_F_F11_CHAIN_CLOSURE.md` v1.0.

---

*v1.0 — F.11 Branch X audit COMPLETE 2026-04-29. Encoder still
not located, but a path-of-least-resistance subset (Branch
X-narrow, ~3h) is identified.*

*Last updated: 2026-04-29 (V32-prep F.11 Branch X scoping).*
