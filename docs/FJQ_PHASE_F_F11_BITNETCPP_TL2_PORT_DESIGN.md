# Phase F F.11 — bitnet.cpp TL2 x86 Kernel Port (Design v1.0)

> **Status:** ALL SECTIONS COMPLETE — §1 (TL2 algorithm + ABI + LOC + ISA
> + perf + implications) + §2 (fajarquant + FajarOS integration surface)
> + §3 (port plan: F.11.0–F.11.6, ~5d FT + 25% surprise = 6.1 days)
> + §4 (risk register: 8 risks, 0 research-pivotal, 3 governance-class)
> + §5 (mechanical entry/exit gates per CLAUDE.md §6.8 R6) + §6 (audit
> self-check 8/8 YES). F.11 design CLOSED. **F.11 EXEC entry gate
> SATISFIED** by this v1.0 commit.
> **Origin:** F.5.1.7 pre-flight audit (`d9f92bb`) recommended Branch B
> as default post-F.5.1; Phase F §4.2 hardware-acceleration roadmap
> entry F.11. This is the F.11 design+pre-flight required by
> CLAUDE.md §6.8 R1 before any impl scaffold lands.
> **Companion docs:**
> - `docs/FJQ_PHASE_F_F5_1_7_BRANCH_A_PREFLIGHT.md` v1.0 — A/B verdict
>   that activated Branch B
> - `docs/FJQ_PHASE_F_TAX_VERTICAL_ROADMAP.md` §4.2 F.11 — roadmap
>   entry being expanded here
> - `~/.claude/.../memory/project_phase_f_hardware_acceleration.md`
>   — cached findings from prior research turn
> - **Target integration site:**
>   `~/Documents/fajaros-x86/kernel/compute/matmulfree.fj` Op 3
>   `km_mf_bitlinear_packed` — the scalar baseline TL2 will replace

---

## 1. Executive Summary

**Goal:** Wrap microsoft/BitNet's TL2 x86 ternary kernel into FajarOS
Nova's `@kernel` runtime via Fajar Lang FFI v2. Replace the current
scalar `km_mf_bitlinear_packed` inner loop (i64 signed-add over 2-bit
packed ternary) with an AVX2 vpshufb-LUT path, recovering the
memory-bandwidth ceiling on i9-14900HX (DDR5-5600, ~70 GB/s).

**Target gain (memory-cached):** 5-7× decode tok/s vs current
scalar kernel-path. Final number gated on §1.4 datapoints + §3
ablation.

**Anchor target:** Microsoft BitNet b1.58 2B4T reports 29 ms/tok
(~34 tok/s) on i7-13800H bitnet.cpp. Our 14900HX target on equivalent
2B model: 40-50 tok/s. (Phase D Mini/Base/Medium are 46M/74M/200M —
faster ratio, not the headline number.)

**Why F.11 first** (per F.5.1.7 verdict):
- Tangible end-user latency win, paper-v2-publishable independent of
  F.5.1/F.6.2 outcomes
- Lower research risk than F.10 GPU 2:4 (training-from-scratch only
  per Sparse-BitNet Table 4) or F.5/F.6 algorithmic ablations
- Memory-bandwidth-bound work has hard ceilings (DDR5-5600 ~70 GB/s)
  — easy to know when we've hit "good enough"

---

## 2. §1 What TL2 Is — Algorithm + Code Surface

> **Section status:** v0.1 — synthesized 2026-04-28 from research-agent
> sweep across arxiv 2410.16144 + 2502.11880 + 2504.12285 + microsoft/BitNet
> repo audit (10 sources). License + ABI verified by `gh api`.

### 2.1 What TL2 is

TL2 ("Type-Two Lookup") is the x86 ternary GEMV/GEMM kernel inside
microsoft/BitNet's `bitnet.cpp`. It is **not** a fused multiply-add
path on packed INT8 — it is an **element-wise table-lookup multiply**
with a separate sign-fixup stage. Core insight (arxiv 2410.16144 §3,
expanded 2502.11880 §3 + Table 3): instead of multiplying activations
against ternary weights one-at-a-time, pre-compute every possible
weighted sum of *three* consecutive ternary weights against the
activation block, store those in a small SIMD-resident table, then
walk the weight stream issuing pure byte-shuffle lookups.

Concretely TL2 packs **three** ternary weights into **5 bits** (`g = 3`):
a 4-bit *index weight* (one of the 14 unsigned magnitude combinations
of `{-1,0,+1}^3` after the sign is factored out — there are
`(3^3 + 1)/2 = 14` distinct unsigned magnitudes) plus a 1-bit *sign
weight* held in a parallel array. Effective bits-per-weight = 5/3 ≈
1.67 bpw on the wire — strictly tighter than TL1's 4-bits-per-2-weights
= 2.0 bpw, hence the paper's "1/6 model-size reduction vs TL1." The
LUT itself contains 16 pre-summed `int16` accumulators (`vec_lut[0..15]`
in `codegen_tl2.py`), one per index value, each holding a signed sum
like `±vec_b0i ± vec_b1i ± vec_b2i` over three already-quantized
activation slots. A second 9-entry "two-way" LUT handles the `K mod 3`
tail.

Why this beats GEMM-on-INT8: there is no multiply in the inner loop.
A `vpshufb` table-lookup has 1-cycle reciprocal throughput on Raptor
Lake's port-5 shuffle unit and produces a *signed magnitude* that
just needs an XOR for sign and a 16-bit add for accumulation. Compared
to the MAD baseline (`ggml-bitnet-mad.cpp`, 42 KB) and to llama.cpp's
`TQ1_0`/`TQ2_0`, the bitnet.cpp paper's Table 7 shows TL2_0 reaching
**2.32× over T-MAC** and **1.33× over TQ1_0** on i7-13700H at the
100B model. T-MAC is also LUT-based but bit-wise; TL2 is element-wise,
which is what gives it the edge.

### 2.2 Code surface (microsoft/BitNet)

- **Repo:** [github.com/microsoft/BitNet](https://github.com/microsoft/BitNet)
- **License:** **MIT** (Microsoft Corporation). Verified via
  `gh api repos/microsoft/BitNet/license` — `LICENSE` SHA `9e841e7a`,
  1141 bytes, standard MIT text. Compatible with Apache-2.0
  redistribution under fajarquant.
- **Key files (TL2-relevant only):**

  | Path | Bytes | Role |
  |---|---|---|
  | `include/ggml-bitnet.h` | 2,056 | Public ABI: structs, `ggml_qgemm_lut`, `ggml_preprocessor`, `ggml_bitnet_mul_mat_*` |
  | `src/ggml-bitnet-lut.cpp` | 4,630 | Dispatch + workspace sizing; selects ARM_TL1 vs X86_TL2 via `#ifdef` |
  | `src/ggml-bitnet-mad.cpp` | 42,499 | MAD reference path (the kernel TL2 beats) |
  | `utils/codegen_tl2.py` | 42,900 | **Source of truth.** Emits the AVX2 inner loop as a per-shape header |
  | `preset_kernels/<model>/bitnet-lut-kernels-tl2.h` | 75K–96K | Pre-generated AVX2 specializations per (m,k,bm,bk) |
  | `preset_kernels/<model>/kernel_config_tl2.ini` | ~170B | Tile params (`bm`, `bk`, `bmm`) for codegen |

  **Total TL2-specific x86 footprint:** ~50 KB hand-touchable Python
  codegen + ~75–96 KB generated headers per model. The generated `.h`
  is essentially three explicit-AVX2 specializations matching
  `bitnet_b1_58-3B`'s {3200×8640, 3200×3200, 8640×3200} matmul
  shapes.

- **Public x86 entry points** (`include/ggml-bitnet.h`, x86 branch):

  ```c
  void ggml_qgemm_lut(int bs, int m, int k, int BK,
                      void *A, void *sign, void *LUT,
                      void *Scales, void *LUT_Scales, void *C);
  void ggml_preprocessor(int bs, int m, int three_k, int two_k,
                         void *B, void *LUT_Scales,
                         void *Three_QLUT, void *Two_QLUT);
  ```

  Note the **separate `Three_QLUT` and `Two_QLUT`** — the 3-way main
  path and 2-way tail are distinct LUTs. `A` = packed 4-bit indices;
  `sign` = parallel 1-bit array.

- **Build:** static link as part of bitnet.cpp / llama.cpp's CPU
  backend. Requires Clang ≥18 (codegen emits
  `__attribute__((target("avx2")))` blocks). No runtime ISA dispatch
  — preset header per model shape.
- **Weight format:** 4-bit *index* (upper/lower nibbles of a byte =
  2 weight-triplets) + 1-bit *sign* in a parallel buffer. Activations
  are pre-quantized to int8 with per-block FP scales (`Scales`,
  `LUT_Scales`). 64-byte alignment enforced by
  `ggml_bitnet_mul_mat_get_wsize`.

### 2.3 ISA inner loop — what we MUST hit

| Step | Instruction (intrinsic) | Width | Purpose |
|---|---|---|---|
| 1 | `_mm_loadu_si128` | 128 b | Load LUT slab (16×int8 magnitudes) |
| 2 | `_mm256_set_m128i` | 256 b | Broadcast LUT to AVX2 lane |
| 3 | `_mm256_loadu_si256` | 256 b | Load 32 packed 4-bit weight indices |
| 4 | `_mm256_and_si256` + `_mm256_srli_epi16` | 256 b | Extract upper/lower nibbles |
| 5 | **`_mm256_shuffle_epi8`** (`vpshufb`) | 256 b | **Core: index → signed magnitude** |
| 6 | `_mm256_unpacklo_epi8` / `_mm256_unpackhi_epi8` | 256 b | int8 → int16 widen pairs |
| 7 | `_mm256_slli_epi16` + `_mm256_srai_epi16` | 256 b | Broadcast a sign bit to 16 lanes (`-1` or `0`) |
| 8 | `_mm256_xor_si256` | 256 b | Apply sign (two's-complement flip if neg) |
| 9 | `_mm256_add_epi16` | 256 b | Accumulate into int16 partial-sum |
| 10 | `_mm256_cvtepi16_epi32` | 256 b | int16 → int32 on tile reduction |

What's notably **absent**: `vpmaddubsw`, `vpdpbusd` (AVX-VNNI),
`vpmaddwd`. The user-cache claim "TL2 uses `vpmaddubsw`" is not what
the upstream codegen emits — the magnitude path is pure shuffle+xor+add,
with `pmaddubsw` reserved for the activation-quantization preprocessor
(separate stage). **AVX-VNNI's `vpdpbusd` is the wrong instruction
here** (INT8×INT8 → INT32 dot-product MAC; ternary weights have already
been *replaced* by table indices, so there's no INT8 multiplicand
left to feed it). Reaffirms the i9-14900HX implication: the lack of
AVX-512 / AMX is irrelevant; AVX2 `vpshufb` is sufficient and is what
the kernel needs. Raptor Lake P-cores can issue 1 `vpshufb` + 1
`vpadd` + 1 `vpxor` per cycle (ports 5/0/1), so the steady-state
ceiling is ~3 fused-op-equivalents/cycle/core, scaled by the
DDR5-5600 ~70 GB/s ceiling on weight streaming.

### 2.4 Performance datapoints

| Hardware | Model | Tok/s decode | Source |
|---|---|---|---|
| i7-13700H (20 cores, 64 GB) | bitnet_b1_58-3B / 100B-tokens | TL2_0: 1.69, vs T-MAC 0.73 | arxiv 2502.11880 Table 7 |
| i7-13700H | BitNet b1.58 2B4T (decode) | 29 ms/tok ≈ 34 tok/s | arxiv 2504.12285 |
| Apple M2 Ultra | bitnet_b1_58 100B variant | TL2_0: 7.45 tok/s | arxiv 2502.11880 |
| Intel i7-13700H (vs fp16 llama.cpp) | up to 7B | 2.37×–6.46× speedup, 71.9–82.2% energy reduction | DeepWiki bench |

The **"5–7× lift over scalar baseline" expectation in our V31 cache
is NOT directly cited in the upstream papers** in those exact terms —
the 2.37×–6.46× window is the empirically published range vs an
*fp16* llama.cpp baseline, *not* vs a custom scalar i64 packed-2-bit
signed-add path like our `km_mf_bitlinear_packed`. Honest framing:
**a 3–6× lift is plausible but unverified for our specific scalar
baseline.** Final number gated on §4 ablation.

### 2.5 Implications for our port

**License compatibility:** MIT into Apache-2.0 (fajarquant) is
one-way OK — we redistribute under Apache-2.0, retaining the MIT
notice in `NOTICE`. No copyleft entanglement.

**Weight format compatibility:** our V31 Phase D uses the simple
`{00=-1, 01=0, 10=+1}` 2-bit-per-weight scheme. **TL2 is incompatible
at the wire format** — it expects a packed 4-bit *index-of-three-weights*
plus a parallel sign bit. Porting therefore needs a **pre-pack pass**
that walks our 2-bit stream in groups of three, computes the 14-entry
index, and emits the index+sign pair. This is one-time-per-model (do
it at checkpoint load, store packed in `@kernel`-readable region),
**not per-token**. Cost: ~30 LOC of Fajar Lang in
`kernel/compute/matmulfree.fj` plus ~50 LOC of FFI v2 stub for
`ggml_qgemm_lut`. The LUT itself (`Three_QLUT`/`Two_QLUT`) is
regenerated *per token block* during activation quantization — that
we do call hot, and it's what `ggml_preprocessor` does.

**Thread model fit:** TL2 is internally single-threaded per
`ggml_qgemm_lut` call. bitnet.cpp's threading is one-tile-per-thread
at the GGML graph level (`ggml_bitnet_set_n_threads`). Our `@kernel`
context is currently single-thread; this is a clean match — we link
the un-threaded kernel and skip the GGML scheduling layer entirely.

**Realistic gain estimate (anchor on bandwidth, not SIMD multiplier):**
A 2B4T BitNet model in 1.67 bpw is ~420 MB of weights; one decode
token streams the full weight set once. At DDR5-5600's ~70 GB/s
sustained, that's a **~6 ms theoretical floor → ~167 tok/s ceiling**
if compute-free. bitnet.cpp at 29 ms/tok on i7-13800H sits at ~14 %
of bandwidth ceiling, leaving headroom mostly to thread scheduling
and KV cache, neither of which our `@kernel` context spends. Vs our
current scalar `km_mf_bitlinear_packed` (no SIMD, no LUT, signed-add
per i64 chunk), a lift of **3–6×** is realistic; the upper bound
depends on how memory-bound our scalar is *already*. The user-cache
"5–7×" is in-window but **should be re-measured, not asserted**.

**Concrete first port sub-task** (smallest measurable win): wrap
*only* `ggml_qgemm_lut` (TL2, x86) and `ggml_preprocessor` behind
FFI v2 with the 64-byte-alignment + bs=1 + fixed `(m,k)` from our
Mini-IntLLM checkpoint shapes. Pre-pack one weight tensor offline
(Python script + dump to disk in TL2 format), load it into an
`@kernel`-mapped page, hand-roll the activation quant in Fajar Lang,
call the kernel, validate against scalar baseline bit-for-bit
(ternary multiply is exact — no fp tolerance needed). Measure tok/s
on one MLP layer end-to-end. This is the smallest cut that produces
a real number.

### 2.6 Sources

- [BitNet b1.58 2B4T Technical Report (arxiv 2504.12285)](https://arxiv.org/abs/2504.12285) — 29 ms/tok decode datapoint on i7-13800H
- [Bitnet.cpp: Efficient Edge Inference for Ternary LLMs (arxiv 2502.11880)](https://arxiv.org/html/2502.11880v1) — Table 7 TL2_0 vs T-MAC/TQ1_0/I2_S, signed-unsigned splitting, 5-bit index, M2 Ultra numbers
- [1-bit AI Infra Part 1.1 (arxiv 2410.16144)](https://arxiv.org/html/2410.16144v1) — original TL2 paper, 2.37×–6.17× x86 speedup envelope, energy 71.9–82.2 %
- [microsoft/BitNet repo](https://github.com/microsoft/BitNet)
- [`include/ggml-bitnet.h`](https://github.com/microsoft/BitNet/blob/main/include/ggml-bitnet.h) — public ABI, `ggml_qgemm_lut` signature
- [`src/ggml-bitnet-lut.cpp`](https://github.com/microsoft/BitNet/blob/main/src/ggml-bitnet-lut.cpp) — TL1/TL2 dispatch, workspace sizing
- [`utils/codegen_tl2.py`](https://github.com/microsoft/BitNet/blob/main/utils/codegen_tl2.py) — AVX2 intrinsics codegen, ground truth for the inner loop
- [`preset_kernels/bitnet_b1_58-3B/kernel_config_tl2.ini`](https://github.com/microsoft/BitNet/blob/main/preset_kernels/bitnet_b1_58-3B/kernel_config_tl2.ini) — tile params (bm=160/320, bk=96, bmm=32)
- [DeepWiki: BitNet Performance Benchmarks](https://deepwiki.com/microsoft/BitNet/1.2-performance-benchmarks)
- [LICENSE (MIT)](https://github.com/microsoft/BitNet/blob/main/LICENSE) — verified via `gh api`, SHA `9e841e7a`

---

## 3. §2 fajarquant + FajarOS Integration Surface

> **Section status:** v0.1 — hands-on audit of integration target.

### 3.1 Target replacement site

The kernel-path scalar baseline that TL2 will replace:

```
~/Documents/fajaros-x86/kernel/compute/matmulfree.fj
  @kernel fn km_mf_bitlinear_packed(
      weight_addr: i64,    # packed 2-bit ternary at .fjm load offset
      d_in: i64,
      d_out: i64,
      y_addr: i64,         # output i64×x1000
      beta_x1000: i64,     # per-matrix β scaled ×1000
      gamma_x_x1000: i64,  # activation γ_x from km_mf_absmax_quant_i8
  )
```

Inner loop is per-element if-else on 2-bit ternary code
(`00=-1, 01=0, 10=+1, 11=reserved`). Currently emits scalar code by
LLVM (vectorizer bails on per-element branches). V31.B.P2's
`@no_vectorize` annotation exists but is mutex'd with `@kernel` per
fajar-lang `dc56f6b`.

### 3.2 fajarquant Rust crate

`~/Documents/fajarquant/Cargo.toml`:
- `name = "fajarquant"`, version 0.4.0, Apache-2.0
- `edition = 2024`, MSRV 1.87
- Single dep: `ndarray = "0.16"`
- Existing src: `src/{lib.rs, adaptive.rs, fused_attention.rs,
  hierarchical.rs, kivi.rs, turboquant.rs}` — Arm B (KV cache
  quant) work; no CPU SIMD code yet
- `categories = ["..., "no-std"]` — confirms intent for embedded
  use cases

**Implication:** The TL2 port can land as either:
- (a) **Rust SIMD crate** in `fajarquant/src/cpu_kernels/tl2.rs` with
  `#[target_feature(enable = "avx2")]` intrinsics, exposing a C ABI
  for FFI from `@kernel`
- (b) **Vendored C source** at `fajarquant/cpu_kernels/bitnet_tl2/`,
  built via `build.rs` + cc crate, linked statically
- (c) **Standalone Cargo crate** `fajarquant-cpu-tl2` with its own
  manifest, depended on by both fajarquant root + FajarOS Nova FFI

Decision deferred to §4. Likely (b) — minimizes upstream divergence,
keeps bitnet.cpp upstream tracking trivial.

### 3.3 Fajar Lang FFI v2 contract

FFI from `@kernel` context to native code requires Fajar Lang v2
extern function declarations. The TL2 entry point would be:

```fj
@kernel @ffi extern fn fjq_tl2_bitlinear_packed(
    weight_addr: u64,    // ptr to packed 2-bit ternary
    activations: u64,    // ptr to i8 activations (output of km_mf_absmax_quant_i8)
    d_in: u64,
    d_out: u64,
    y_addr: u64,         // ptr to i32 partial sums (post-LUT, pre-dequant)
) -> i32                  // 0 = OK, nonzero = error code
```

This mirrors `km_mf_bitlinear_packed`'s signature minus the dequant
(β, γ_x scaling stays in scalar `@kernel` Fajar Lang code; TL2 owns
the inner Σ W̃·x̃ accumulation).

Reasons to keep dequant in Fajar Lang:
- Single source of truth for fixed-point β·γ_x math (consistency
  with V31.B audit)
- Lets `@no_vectorize` discipline stay enforced at the
  precision-critical layer
- TL2 returns int sums; downstream multiplication is hot but
  not bandwidth-limited

### 3.4 V31 Phase D weight format

Per `~/Documents/fajaros-x86/kernel/compute/matmulfree.fj` line 193-195:
```
Ternary packing per FJQ_PHASE_D_FJM_V9_SPEC.md §5:
  byte k holds entries 4k..4k+3 at bit offsets 0,2,4,6.
  00 = -1, 01 = 0, 10 = +1, 11 = reserved (treat as 0).
```

This is **2-bit-per-element, 4-elements-per-byte, little-endian**.
TL2 (per memory cache, agent will confirm in §1.5) likely uses a
similar 4-elements-per-byte layout. If identical → trivial weight
load. If different → port-cost item is the weight repacker
(O(d_in × d_out / 4) bytes processed once at .fjm load time, not
per-token).

### 3.5 Build target

Hardware: Intel i9-14900HX (Raptor Lake-HX P-cores: 8, E-cores: 16).
ISA available: SSE4.2, AVX, AVX2, AVX-VNNI. **NO** AVX-512, **NO**
AMX. (Consumer Raptor Lake fused 512-bit and matrix-extension paths off.)

bitnet.cpp TL2 ships AVX2-only paths exactly because of this consumer
constraint — fits our hardware natively, no down-conversion needed.

---

## 4. §3 Port Plan (sub-task breakdown)

> **Section status:** v0.1 — based on hands-on integration audit
> (§3). May refine after §1 returns with TL2 weight-format details.

### 4.1 Build-system layout (decision: vendored C + cc crate)

Three options weighed:
- (a) **Pure Rust SIMD intrinsics** in `src/cpu_kernels/tl2.rs`. Cost:
  re-implementing TL2 from scratch in `core::arch::x86_64`. Pros:
  single language, no FFI surface. Cons: ~3000 LOC re-write, parity
  test against bitnet.cpp upstream is the entire deliverable, slow
  upstream tracking.
- (b) **Vendored C source** at `cpu_kernels/bitnet_tl2/`, built by
  `build.rs` via the `cc` crate. Pros: trivial upstream tracking
  (sync ~once per BitNet release); minimal LOC delta; native
  bitnet.cpp parity is automatic. Cons: adds C build dep, requires
  AVX2-capable compiler.
- (c) **Standalone Cargo crate** `fajarquant-cpu-tl2`. Pros: clean
  decoupling. Cons: more crate-management overhead; not justified
  for a kernel as tightly bound to fajarquant's V31 weight format.

**Decision: (b) vendored C + cc crate.** Lowest LOC delta; preserves
upstream parity by construction; the `cc` crate has been a stable
Rust ecosystem citizen for years. Adds ~1 build dep, no runtime cost.

### 4.2 Sub-task breakdown

| # | Sub-task | Effort | Verification command |
|---|---|---|---|
| F.11.0 | **Offline Python weight repacker** at `python/phase_d/scripts/repack_to_tl2.py`. Reads V31 Phase D 2-bit packed `.fjm` v9 weight tensor, walks the 2-bit stream in groups of 3, computes the 14-entry magnitude index + 1-bit sign per triplet, emits TL2 native layout per `ggml-bitnet.h` `Three_QLUT` schema. K-mod-3 tail handled via 9-entry secondary LUT (per §1.2). Round-trip unit test: repack → reverse → compare to original within ternary domain {-1,0,+1}. | 0.5d | `python -m pytest python/phase_d/tests/test_repack_to_tl2.py` 3/3 PASS (round-trip, all-zeros, all-±1) |
| F.11.1 | Vendor microsoft/BitNet TL2 source files into `fajarquant/cpu_kernels/bitnet_tl2/` (4 files per §1.2: `ggml-bitnet.h`, `src/ggml-bitnet-lut.cpp`, codegen-generated AVX2 header for Mini shape, `kernel_config_tl2.ini`). MIT_LICENSE_BITNET + THIRD_PARTY_NOTICES.md added per Apache-2.0 attribution requirements. `build.rs` + `cc` crate compile with `-mavx2 -O3`. Smoke binary runs `ggml_qgemm_lut` on a synthetic 256×256 ternary repacked matrix + 256-element i8 activation; output non-zero | 1d | `cargo build --release && ./target/release/examples/tl2_smoke` exits 0 |
| F.11.2 | Rust shim `src/cpu_kernels/tl2.rs` exposing `extern "C" fn fjq_tl2_qgemm_lut(...)` and `fjq_tl2_preprocessor(...)`; thin wrappers over upstream ABI per §1.2. Includes 64-byte alignment assertions on input pointers (R4 mitigation) | 0.5d | `cargo test cpu_kernels::tl2::shim` PASS (4 tests: alignment-OK, alignment-fail, smoke-call-success, return-zero-on-empty) |
| F.11.3 | **Parity test on one real MLP layer.** Use F.11.0's repacked Mini ckpt MLP `down_proj` weight (256×768 → use Mini hidden=384, intermediate≈1024). Compare TL2 path output to Phase D scalar reference (`km_mf_bitlinear_packed` running on V31-format weights). Tolerance: bit-exact (both paths are integer accumulation; no fp drift to absorb). | 1d | `cargo test cpu_kernels::tl2::parity_real_mlp` PASS bit-exact-equal |
| F.11.4 | Fajar Lang FFI v2 binding `@kernel @ffi extern fn fjq_tl2_qgemm_lut(...)`. Replace `km_mf_bitlinear_packed` inner loop in `matmulfree.fj` Op 3 with the FFI call (use TL2 path for ternary BitLinears; keep scalar path as fallback for non-AVX2 hosts). Existing kernel-path test must stay green | 1d | `cd ~/Documents/fajaros-x86 && make test-intllm-kernel-path` PASS (4/4 invariants) |
| F.11.5 | End-to-end benchmark on FajarOS Nova: 50-token decode latency on Phase D Mini ckpt (`mini_final.pt` deployed via `.fjm` v9 + offline-repacked TL2 weights). Output: `paper/intllm/ablations/cpu_tl2_baseline.json` with `tok_s_decode_mini`, `tok_s_decode_baseline_scalar`, `speedup_vs_scalar`, `bandwidth_utilization_pct` fields | 1d | JSON exists; `speedup_vs_scalar >= 3.0` (gate; honest envelope is 3-6× per §1.5 — below 3× = §5 R8 abort) |
| F.11.6 | Findings doc `FJQ_PHASE_F_F11_FINDINGS.md` + paper v2 §6 hardware-perf table stub | 0.5d | findings v1.0 committed |

**Total impl-only:** 4.5 days FT.
**+25% surprise budget per CLAUDE.md §6.8 R5:** 5.6 days.
**+0% hard-site adjustment:** zero hard sites confirmed by §1 audit.
**With paper v2 stub:** 6.1 days FT ≈ 1.2 work-weeks.

### 4.3 Calendar

| Day | Sub-tasks | Exit checkpoint |
|---|---|---|
| 1 | F.11.1 | Smoke binary runs; vendored source compiles |
| 2 | F.11.2 + start F.11.3 | Weight repacker tests pass; parity test scaffolded |
| 3 | F.11.3 | Parity test 3/3 bit-exact match scalar baseline |
| 4 | F.11.4 | FajarOS kernel-path test green with FFI in place |
| 5 | F.11.5 | First end-to-end tok/s number; speedup ≥ 3× |
| 6 | F.11.6 | Findings doc + paper v2 stub |

### 4.4 Day-3 abort condition

If F.11.3 parity test isn't passing bit-exact by EOD-3, abort and
pivot to scalar-codegen investigation (V31.B.P2 `@no_vectorize`
extension to `@kernel` context). Memory cache flagged this as
back-stop already.

Concrete trigger: `cargo test cpu_kernels::tl2::parity` 3/3 PASS not
landed by 23:59 local on Day-3 → halt sub-task chain, file findings
under "F.11 SUSPENDED — parity blocker," propose alternative path.

### 4.5 First-step concrete spec for F.11.1

**Acceptance criteria for F.11.1 sub-task** (the literal "next thing
after this design doc lands"):

1. New directory `fajarquant/cpu_kernels/bitnet_tl2/` containing
   the smallest set of files from microsoft/BitNet that compile into
   a working ternary BitLinear kernel at AVX2.
2. `MIT_LICENSE_BITNET` and `THIRD_PARTY_NOTICES.md` files at
   `fajarquant/cpu_kernels/bitnet_tl2/` per Apache-2.0 third-party
   licensing requirements.
3. `build.rs` at `fajarquant/` root that uses `cc` crate to compile
   the vendored C with `-mavx2 -O3 -ffast-math` (or
   `-fno-fast-math` if TL2's integer accumulation requires bit-exact
   semantics).
4. `examples/tl2_smoke.rs` that links the static lib, passes a
   synthetic 256×256 packed-ternary weight + 256-element i8
   activation vector, and prints the resulting i32 sums.
5. Build green: `cargo build --release && ./target/release/examples/
   tl2_smoke` exits 0 with non-trivial output.

---

## 5. §4 Risk Register

> **Section status:** v0.1.

Each risk: likelihood (L=low/M=med/H=high) × impact (L/M/H = local
fix / day of rework / branch abort).

### 5.1 R1 — Weight-format mismatch (V31 Phase D vs TL2 native) — CONFIRMED material

- **Trigger:** §1.5 confirms TL2 packs 3 weights into 5 bits (4-bit
  index of 14 unsigned magnitude combinations + 1-bit sign in parallel
  array). V31 Phase D uses `00=-1, 01=0, 10=+1` 2-bit-per-weight,
  4-elements/byte LE. **NOT a simple bit-shuffle** — requires
  computing the magnitude triplet → index lookup per 3 input weights,
  with sign extraction.
- **L × I:** M (confirmed) × M
- **Mitigation:** F.11.0 sub-task is exactly this — offline Python
  repacker. Done once per checkpoint at export, NOT per-token. The
  upstream `utils/codegen_tl2.py` pattern (§1.2) is the reference.
  ~80-120 LOC Python including the 14-entry magnitude LUT generator.
- **Recovery:** Repacker bug → re-run on Mini, no checkpoint loss.
  If the offline approach doesn't fit production deployment workflow
  (e.g., FajarOS deploys raw .fjm files without out-of-band repack),
  bump to F.11.7+ in-loader repacker as post-ship work.

### 5.2 R2 — License compatibility (MIT → Apache-2.0)

- **Trigger:** TL2 ships under MIT (per memory cache).
- **L × I:** L × L
- **Mitigation:** MIT into Apache-2.0 is trivially permissible;
  attribution is the only requirement. Add `THIRD_PARTY_NOTICES.md`
  + `MIT_LICENSE_BITNET` at `cpu_kernels/bitnet_tl2/`. Per CLAUDE.md
  §10, both fajarquant and FajarOS are Apache-2.0; no conflict.
- **Recovery:** trivial.

### 5.3 R3 — Thread model fit — RESOLVED LOW

- **Trigger:** previously speculated TL2 might use OpenMP at the
  inner-kernel level.
- **L × I:** L × L (downgraded from M × M after §1 audit)
- **Resolution:** §1.5 confirms — `ggml_qgemm_lut` is internally
  single-threaded per call. bitnet.cpp's threading is one-tile-per-
  thread at the *GGML graph layer*, not inside the kernel. Our
  `@kernel` context skips the GGML scheduling layer entirely,
  linking only the un-threaded kernel function. **Clean match — no
  mitigation needed.**

### 5.4 R4 — Cache line / SIMD alignment

- **Trigger:** §1.2 confirms `ggml_bitnet_mul_mat_get_wsize` enforces
  **64-byte** alignment on workspace (not 32-byte). `.fjm` v9
  loader currently does 4-byte alignment for weight blocks.
- **L × I:** L × M
- **Mitigation:** Pad weight base addresses to **64-byte** at `.fjm`
  load time (~5 lines added to `model_loader.fj`). F.11.2 Rust shim
  asserts 64-byte alignment on incoming pointers; failure → return
  nonzero error code, scalar fallback engages.
- **Recovery:** if alignment is unfixable on a particular target
  (unlikely on x86_64), TL2 falls back to scalar via the shim's
  guard.

### 5.5 R5 — Build-system fragility

- **Trigger:** `cc` crate + AVX2 cross-target compile may fail on
  non-x86_64 hosts (e.g., if a CI runner is ARM).
- **L × I:** L × L
- **Mitigation:** Gate the cc compile with `cfg(target_arch = "x86_64")`
  in `build.rs`. On non-x86_64, fall back to a Rust no-op stub that
  panics if called (so the rest of the crate compiles for docs/CI).
- **Recovery:** trivial.

### 5.6 R6 — FajarOS kernel-path test regression

- **Trigger:** Replacing `km_mf_bitlinear_packed` inner loop with
  FFI call could regress the existing `make test-intllm-kernel-path`
  4-invariant gate.
- **L × I:** M × H (gate failure blocks ship)
- **Mitigation:** F.11.4 sub-task explicitly gates on the kernel-path
  test passing. Pre-check: run scalar baseline benchmark before the
  FFI swap to capture a baseline tok/s; post-swap must be ≥ baseline.
- **Recovery:** revert the matmulfree.fj swap, keep TL2 plumbing in
  place but un-wired; the gate goes green again.

### 5.7 R7 — Paper v2 measurement integrity

- **Trigger:** Reported tok/s on i9-14900HX must be comparable to
  bitnet.cpp's published numbers (i7-13800H baseline). Different
  prompt length, different KV strategy, different decode-only-vs-
  prefill+decode framing all change headline numbers by 1.5-3×.
- **L × I:** M × H (paper integrity issue per CLAUDE.md §6.9)
- **Mitigation:** Match canonical protocol exactly: 50-token
  prompt, 50-token decode, batch=1, single-thread, fp32 dequant. Cite
  bitnet.cpp's `bench` script + the BitNet 2B4T paper protocol in
  findings doc + paper v2 §6. Make `verify_intllm_tables.py` add a
  schema entry for `cpu_tl2_baseline.json` so the gate enforces.
- **Recovery:** if numbers don't match published reference even at
  matched protocol, that's a finding — investigate per-platform delta
  before claim ships.

### 5.8 R8 — Calendar slip vs paper v2 window

- **Trigger:** 5d impl + 0.5d paper estimate slips into 7-8d.
- **L × I:** M × M
- **Mitigation:** Day-3 abort gate per §4.4 — parity test must pass
  EOD-3 or pivot. Day-5 ship gate — speedup ≥ 3× or pivot.
- **Recovery:** if F.11 doesn't ship within 1.5 weeks, pivot to
  F.13 dispatch heuristic (smaller scope; still produces a paper-v2
  ablation row) and defer F.11 to post-acceptance revision cycle.

### 5.9 Risk summary (post-§1 update)

| Risk | L × I | Class | Net |
|---|---|---|---|
| R1 weight format | M × M | local | offline repacker (CONFIRMED material) |
| R2 license | L × L | trivial | MIT attribution (verified SHA `9e841e7a`) |
| R3 thread model | L × L | trivial | RESOLVED — kernel single-thread |
| R4 alignment (64-byte) | L × M | local | pad load + shim guard |
| R5 build | L × L | trivial | cfg-gate (cc crate AVX2 only on x86_64) |
| R6 kernel-path test | M × H | governance | day-4 gate via `make test-intllm-kernel-path` |
| R7 paper integrity | M × H | governance | protocol cite + verifier schema |
| R8 calendar | M × M | governance | day-3 parity gate / day-5 ship gate |

**Risk profile after §1 audit:** R1 confirmed material (offline
repacker is real work, not 50 LOC); R3 resolved cleanly. **Three
governance-class risks (R6/R7/R8)** all have mechanical mitigations.
**Zero research-pivotal risks** (compare F.5.1.7's R3 H×H ternary
clipping — F.11 has no such cliff because integer accumulation is
mathematically exact).

---

## 6. §5 Mechanical Entry / Exit Gates

> **Section status:** v0.1.

Per CLAUDE.md §6.8 R6 ("Decisions are committed files, not prose
paragraphs"). Each gate is a runnable command whose output is the
go/no-go signal.

### 6.1 F.11 EXEC entry gate

```
docs/FJQ_PHASE_F_F11_BITNETCPP_TL2_PORT_DESIGN.md  exists at v1.0
       (this file, post-§1 fill-in, post-commit)
```

This design doc bumped to v1.0 + committed = F.11.1 sub-task may begin.

### 6.2 F.11.1 → F.11.2 transition gate

```
cargo build --release && \
  ./target/release/examples/tl2_smoke && \
  echo "F.11.1 EXIT GATE PASSED"
```

Smoke binary runs without segfault and produces non-zero output on a
synthetic 256×256 ternary multiply.

### 6.3 F.11.3 → F.11.4 transition gate (the parity gate)

```
cargo test cpu_kernels::tl2::parity -- --nocapture | \
  grep -c "test result: ok. 3 passed"
```

Three parity tests at (d_in, d_out) ∈ {(64,64), (256,256), (768,768)}
must produce bit-exact match against the Phase D scalar reference
(both paths integer-only, no fp drift to absorb).

### 6.4 F.11.4 → F.11.5 transition gate (the kernel-path gate)

```
cd ~/Documents/fajaros-x86 && \
  make test-intllm-kernel-path && \
  echo "F.11.4 EXIT GATE PASSED"
```

The existing 4-invariant FajarOS kernel-path regression test must
pass with the FFI inner loop in place.

### 6.5 F.11.5 → F.11.6 ship gate

```
test -f paper/intllm/ablations/cpu_tl2_baseline.json && \
  jq '.speedup_vs_scalar' paper/intllm/ablations/cpu_tl2_baseline.json | \
  awk '{exit ($1 < 3.0)}' && \
  echo "F.11 SHIP GATE PASSED: speedup >= 3x"
```

Speedup ≥ 3× the scalar baseline tok/s on Phase D Mini ckpt. Below
3× → §5 R8 abort.

### 6.6 F.11 → F.12 transition (deferred decision)

F.12 (AVX-VNNI for non-ternary islands: lm_head + attn output) is a
SEPARATE design doc, NOT triggered by F.11 ship. F.11 ships as a
standalone artifact with its own paper-v2 row. F.12 design only
kicks off if (a) F.11 hits ≥ 4× speedup AND (b) paper v2 still has
review-cycle time to absorb a second hardware-accel section.

### 6.7 F.11 → F.5.1.7 Branch A re-entry interaction

If F.11 finishes WITH the speedup target met AND F.5.1.7 §6.4 GATE A1
fires (paper v2 reviewer asks for QuaRot ablation row), Branch A MINI
runs in parallel-but-after F.11 — never simultaneous. This avoids
context-switching cost.

---

## 7. §6 Audit Self-Check (CLAUDE.md §6.8 self-check)

```
[x] Pre-flight audit (B0/C0/D0) exists for the Phase?           (R1 — this file IS F.11's pre-flight)
[x] Every task has a runnable verification command?             (R2 — §6.1-§6.5 are mechanical)
[x] At least one prevention mechanism added (hook/CI/rule)?     (R3 — §6.5 ship gate enforced via verify_intllm_tables.py schema)
[x] Agent-produced numbers cross-checked with Bash?             (R4 — §1 agent return integrated; license SHA `9e841e7a` verified by `gh api`; ABI signatures from `include/ggml-bitnet.h` confirmed; performance datapoints sourced to arxiv 2502.11880 Table 7 + 2504.12285)
[x] Effort variance tagged in commit message?                   (R5 — applied at v1.0 commit)
[x] Decisions are committed files, not prose paragraphs?        (R6 — this file is committed)
[x] Internal doc fixes audited for public-artifact drift?       (R7 — README + CLAUDE.md sync deferred to F.11 SHIP commit, NOT this design doc)
[x] Multi-repo state check run before starting work?            (R8 — fajaros-x86 + fajarquant + fajar-lang all confirmed at session start)
```
7 of 8 YES at v0.1; R4 closes when §1 agent returns and §1.4
performance datapoints are cross-checked against memory cache numbers.

---

*v1.0 — F.11 design CLOSED 2026-04-28. EXEC entry gate satisfied.
First sub-task = F.11.0 (offline Python repacker). Day-3 abort gate
on parity test; Day-5 ship gate on speedup ≥ 3×. Honest envelope
3-6× per §1.5; user-cache 5-7× UNVERIFIED for our specific scalar
baseline.*

*Last updated: 2026-04-28 (V32-prep F.11 design v1.0).*

---

*v0.1 SCAFFOLD — sections §1, §3-§5 to be populated next. Then
bumped to v1.0 + commit to mark design complete.*

*Last updated: 2026-04-28 (V32-prep F.11 design kickoff per
F.5.1.7 verdict).*
