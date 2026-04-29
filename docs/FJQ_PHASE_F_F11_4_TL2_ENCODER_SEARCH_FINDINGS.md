# Phase F F.11.4 — TL2 Weight-Encoder Search Findings (v1.0)

> **Status:** F.11.4(a') BLOCKED on encoder availability. Decision
> needed at next session entry — branch options at §5.
> **Origin:** F.11.3 + F.11.4(a) PARTIAL identified that
> `fjq_tl2_qgemm_lut` requires A/sign buffers in tile-organized
> layout NOT produced by F.11.0's flat repacker. F.11.4(a') was
> scoped to "port upstream tile encoder to Rust + write
> `parity_real_mlp` test." This doc surfaces the encoder is NOT
> in any vendorable upstream Python — it lives in compiled
> `llama-quantize` C++ from microsoft/BitNet's llama.cpp fork.

---

## 1. What we know about the wire layout

From hands-on reading of `cpu_kernels/bitnet_tl2/bitnet-lut-kernels-tl2.h`
generated for the `fajarquant_mini` ModelShapeDict entry:

### 1.1 Per-tile constants (256×256 shape)

```c
#define BM256_256   64    // rows per output tile (4 tiles for m=256)
#define BBK256_256 128    // K-cols per inner block (2 inner blocks for k=256)
const int KK = BBK / 3;   // = 42 (integer division; 128 % 3 = 2 leftover)
```

### 1.2 `three_qgemm_lut_256_256` driver (kernel-side caller)

```c
for (int32_t k_outer = 0; k_outer < 256 / BBK256_256; ++k_outer) {
    three_tbl_impl_256_256<BS, 256>(
        (int32_t*)CBits,
        (int8_t*)LUT      + k_outer * BBK / 3 * 32,        // = k_outer * 42 * 32 = k_outer * 1344 bytes
        (uint8_t*)A       + k_outer * BBK / 3 / 2 * BM,    // = k_outer * 21 * 64 = k_outer * 1344 bytes
        (uint8_t*)sign    + k_outer * BBK / 3 / 8 * BM     // = k_outer * 5  * 64 = k_outer *  320 bytes
    );
}
```

### 1.3 `three_tbl_impl_256_256` per-tile inner loops

```c
for (int i = 0; i < BM; i += 32) {              // 2 i-groups (i=0, i=32) for BM=64
    __m256i vec_as[KK / 2];                      // 21 magnitude vectors of 32 bytes each
    __m256i vec_signs[KK / 8];                   // 5 sign vectors of 32 bytes each
    for (int ai = 0; ai < KK / 2; ai++) {
        vec_as[ai] = _mm256_loadu_si256(a + i * KK / 2 + ai * 32);
                                                 // = a[i * 21 + ai * 32]
    }
    for (int as = 0; as < KK / 8; as++) {
        vec_signs[as] = _mm256_loadu_si256(sign + i * KK / 8 + as * 32);
                                                 // = sign[i * 5 + as * 32]
    }
    // ... 4 unrolled k iterations using vec_as[k*4+0..3] + vec_signs[k] ...
}
```

### 1.4 Implied byte semantics

Each magnitude byte holds **2 magnitude indices** (4-bit upper nibble +
4-bit lower nibble), gating on `_mm256_shuffle_epi8` lookups against
8 separate LUT slabs (vec_k1..vec_k4 × top/bot). One byte therefore
covers 2 triplets = 6 weights. One 32-byte ymm vector covers 64
triplets = 192 weights. KK=42 triplets per inner block × 32 outputs
parallelism = 1344 triplet-output combinations per block per i-group.

Per output row in one BBK block (128 weights / 42 triplets,
truncated): 21 magnitude bytes + ~5 sign bytes. Total per tile per
k_outer: 64 rows × 21 bytes = 1344 magnitude bytes + 64 × 5 = 320
sign bytes.

### 1.5 Format formula (codegen `gen_transform_code`)

```c
int nbytes = (k - 256) * m / 3 * 5 / 8 + 256 * m / 2 * 4 / 8;
```

Sized for the BitNet 2B4T variant (k > 256): first `k - 256` columns
are 3-path encoded (5 bits per 3 weights = 5/24 bytes per weight),
last 256 columns are 2-path (4 bits per 2 weights = 1/4 byte per
weight). For our (m=256, k=256) shape this degenerates: zero 3-path
bytes, 16,384 2-path bytes — INCONSISTENT with the kernel's actual
3-path-only call sequence (preprocessor uses `three_k=256, two_k=0`).

This formula is NOT a reliable size guide for our config. Empirical
sizing from the kernel strides above:
  - A buffer total: 4 row-tiles × 2 k_outer × 1344 = **10,752 bytes**
  - sign buffer total: 4 × 2 × 320 = **2,560 bytes**

---

## 2. Where the encoder is NOT

Searched 2026-04-29:

| Source | Verdict | Evidence |
|---|---|---|
| `cpu_kernels/bitnet_tl2/bitnet-lut-kernels-tl2.h` | NO | `ggml_bitnet_transform_tensor` (line 1124+) just stores `tensor->data` as-is; no permutation |
| `utils/convert-helper-bitnet.py` | NO | Pure orchestration wrapper; calls 3 external tools |
| `utils/convert-ms-to-gguf-bitnet.py` | NO | 72KB but contains zero TL2/lut/BM references; top-level functions are `permute`, `transform_to_i2`, `weight_quant`, etc. — none ternary-tile-permute |
| `gpu/pack_weight.py` | NO | 2-bit GPU layout, NOT TL2 (`compress_int2_to_int8` + `B_global_16x32_to_shared_load_16x32_layout`; row-blocks 16×32, not BM×BBK) |
| `utils/codegen_tl2.py` `gen_transform_code` | NO | Generates ABI stub that just stores `tensor->data` |

## 3. Where the encoder IS (almost certainly)

The microsoft/BitNet `convert-helper-bitnet.py` calls
**`llama-quantize`** as the final step. `llama-quantize` is a binary
from microsoft/BitNet's llama.cpp fork — **a separate repository**
(or sibling `3rdparty/llama.cpp` submodule). The Q-format conversion
path that emits `GGML_TYPE_TL2` tensors lives in C++ ggml-quants.c
(or similar) inside that fork.

To port the encoder, we'd need to:
1. Locate microsoft/BitNet's llama.cpp fork (likely
   github.com/microsoft/llama.cpp at a specific commit pinned in
   their build).
2. Find the `quantize_row_tl2` / `ggml_compute_forward_tl2` /
   `convert_q_to_tl2` C function.
3. Read its tile-permutation logic.
4. Port to Rust (~100-200 LOC depending on complexity).
5. Test against synthetic input + a known-good vendored TL2 tensor
   (which we don't have).

**Estimated effort: 1-2 days.** Doesn't fit a single "first step
only" increment under our discipline.

---

## 4. What's still working

F.11.0–F.11.3.5 + F.11.4(a) achievements remain valid:

- F.11.0: V31 → flat-TL2-format Python repacker (algorithmic encoder for
  the LUT _semantics_, not the tile-permuted wire format)
- F.11.1: build chain for vendored TL2 AVX2 kernel
- F.11.2: safe Rust shim with alignment + shape validation
- F.11.3: scalar baseline as bit-exact reference
- F.11.3.5: Mini-shape codegen unblocking shape-locking
- F.11.4(a): preprocessor FFI smoke (LUT_Scales + Three_QLUT
  populated correctly through AVX2)

What's STILL un-validated end-to-end: the `qgemm_lut` AVX2 path
producing bit-exact-equal output to scalar on real ternary weights.
Without the encoder, this stays an open question.

---

## 5. Decision required at next session entry

Three branches:

### Branch X (recommended, conditional): port encoder from llama.cpp fork

- Locate microsoft/BitNet's llama.cpp fork (1-2h)
- Read `quantize_row_tl2` (1-2h)
- Port to Rust (~4-6h)
- Write `parity_real_mlp` test (~2h)
- Total: 1-1.5d

If this lands, F.11.4(a') CLOSES bit-exactly. Strongest
F.11 chain validation possible. Downside: significant
session-time commitment.

### Branch Y: skip parity, ship F.11.4(b) FFI binding with
`make test-intllm-kernel-path` as safety net

- Wire `@kernel @ffi extern fn fjq_tl2_qgemm_lut` in
  `fajaros-x86/kernel/compute/matmulfree.fj`
- Replace scalar `km_mf_bitlinear_packed` Op 3 with FFI call
- Run existing 4-inv kernel-path test
- If green: empirical safety; if red: bug surfaces here

Pros: progresses F.11 chain without 1-2d encoder work. Cons:
parity is empirical, not bit-exact-formal. A subtle bug
producing wrong logits within tolerance would slip past the
4-inv test (which checks for crashes + sane output magnitude,
not exact-match).

### Branch Z: STOP F.11 chain at F.11.4(a) PARTIAL; pivot to F.13

F.13 (CPU-vs-GPU dispatch heuristic) is independent of TL2's
correctness — it's a runtime decision about which path to use.
We could ship F.13 as a standalone V32-prep deliverable while
deferring F.11.4(b) until Branch X resolves.

Pros: clean exit from the F.11 rabbit hole. Cons: F.11 chain
ships incomplete (no production TL2 deployment in
matmulfree.fj).

---

## 6. Recommendation

**Default per resume protocol: Branch Y.** F.11.4(b) FFI binding
gives empirical TL2-in-production validation via the existing
`make test-intllm-kernel-path` regression. If it passes, we have
strong-enough evidence for ship; if it fails, we know parity is
the issue and Branch X gets prioritized.

The 4-invariant test is robust enough to catch correctness
regressions (it checks output magnitude, sentinel tokens, kernel
crash freedom, and decode-loop stability). Bit-exact parity is
"nicer" but not strictly required for production confidence.

Branch X is the right path if user wants ALGORITHMIC certainty
before deployment, OR if a separate audit (paper v2 review,
external collaborator) requires it. Otherwise Branch Y unblocks
the chain at lower cost.

---

*v1.0 — F.11.4(a') BLOCKED 2026-04-29; decision branches X/Y/Z
spec'd. Default = Branch Y.*

*Last updated: 2026-04-29 (V32-prep F.11.4 encoder search).*
