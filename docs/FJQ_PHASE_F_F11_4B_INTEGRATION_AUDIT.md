# Phase F F.11.4(b) — Branch Y Integration Audit (v1.0)

> **Status:** F.11.4(b) ENTRY-CONDITIONS MAPPED. All three audited
> dependencies (Fajar Lang FFI, fajaros-x86 build chain,
> vendored TL2 libc) are tractable; integration is feasible per
> the F.11.4 design plan.
> **Origin:** F.11.4(a') BLOCKED (encoder not in vendorable upstream).
> Selected Branch Y per `FJQ_PHASE_F_F11_4_TL2_ENCODER_SEARCH_FINDINGS.md`
> §6: skip bit-exact parity, ship FFI binding, rely on
> `make test-intllm-kernel-path` 4-invariant gate as safety net.

---

## 1. Fajar Lang `@ffi` / `extern fn` capability — CONFIRMED PRESENT

**Lexer** (`Fajar Lang/src/lexer/token.rs`):
  - Token `AtFfi` for `@ffi` annotation (lines 263, 585, 786)
  - Annotation list documents `@kernel @device @safe @unsafe @ffi`
    (line 1436)

**Parser** (`Fajar Lang/src/parser/`):
  - AST node `Item::ExternFn` (ast.rs:69)
  - `ExternFn` struct supports optional annotation (typically `@ffi`)
  - Test fixture (parser/mod.rs:2168-2169):
    ```fj
    @ffi extern("C") fn malloc(size: u64) -> u64
    ```
    Round-trips through the parser.

**Analyzer** (`Fajar Lang/src/analyzer/type_check/`):
  - `register_extern_fn` registers the symbol (register.rs:2379)
  - Type-check tests verify FFI-safe params/returns (mod.rs:2973+):
    `extern_fn_with_ffi_safe_types_passes`,
    `extern_fn_registered_in_symbol_table`,
    `extern_fn_rejects_string_param`,
    `extern_fn_rejects_string_return`,
    `extern_fn_multiple_ffi_safe_params`

**Codegen** (`Fajar Lang/src/codegen/llvm/mod.rs`):
  - `declare_external_fn(name, params, ret)` — wraps LLVM
    `declare` for runtime/external symbols (line 693)
  - Already used for `fj_rt_println_int`, `fj_rt_print_str`, etc.
    (the runtime ABI). Pattern is established.

**Verdict:** Fajar Lang FFI v2 is present and production-quality.
F.11.4(b) can write:

```fj
@ffi extern("C") fn fjq_tl2_qgemm_lut(
    bs: i32, m: i32, k: i32, bk: i32,
    a: u64, sign: u64, lut: u64,
    scales: u64, lut_scales: u64, c: u64,
)
```

`@kernel` + `@ffi` co-application: compiler currently mutex's
`@no_vectorize` with `@kernel` (per matmulfree.fj:202 comment from
V31.B.P2). Need to verify `@ffi` doesn't have the same constraint;
if it does, the call site uses an indirection (a `@kernel` wrapper
function calls a `@unsafe @ffi extern fn` via raw pointer).

## 2. fajaros-x86 build chain — CAN LINK EXTERNAL `.o` OBJECTS

`fajaros-x86/Makefile` LLVM backend already links external C objects
into the kernel ELF:

```makefile
@# Step 1: compile FJ to .o + capture via ld wrapper
@$(FJ) build --no-std --backend llvm \
    --target-features="$(LLVM_FEATURES)" \
    --linker-script $(LINKER_LD) \
    --code-model kernel \
    --reloc static \
    --extra-objects $(RUNTIME_O) \         # runtime symbols
    --linker scripts/ld-wrapper.sh \
    $(COMBINED) -o /dev/null

@# Step 2: relink with C vecmat (no gc-sections)
@ld -T $(LINKER_LD) -nostdlib \
    $(BUILD_DIR)/combined.start.o.saved \
    $(VECMAT_O) \                          # gcc-compiled C — precedent!
    $(BUILD_DIR)/combined.o.saved \
    $(RUNTIME_O) \
    -o $(KERNEL_LLVM)
```

**Precedent:** `$(VECMAT_O)` is the V30 Track 3 gcc-compiled C
helper used to bypass an LLVM O2 miscompile on packed-2-bit
vecmat. It's compiled with custom flags (likely
`gcc -c -ffreestanding -nostdlib -mavx2 -O3 ...`) and linked
directly into the kernel ELF via the second-step `ld` invocation.

**For F.11.4(b)**: vendor a similar build pipeline for
`fajarquant_bitnet_tl2.o`:

```makefile
TL2_O = $(BUILD_DIR)/fajarquant_bitnet_tl2.o
TL2_SRC = ../fajarquant/cpu_kernels/bitnet_tl2/wrapper.cpp

$(TL2_O): $(TL2_SRC)
    g++ -c -ffreestanding -nostdlib -fno-exceptions -fno-rtti \
        -mavx2 -O3 -std=c++17 -DGGML_BITNET_X86_TL2 \
        -I ../fajarquant/cpu_kernels/bitnet_tl2 \
        $(TL2_SRC) -o $(TL2_O)
```

Then add `$(TL2_O)` to the `ld` step's input list.

## 3. Vendored TL2 libc dependencies — TRACTABLE

`grep -n posix_memalign|free|aligned_malloc` on
`cpu_kernels/bitnet_tl2/bitnet-lut-kernels-tl2.h`:

| Line | Symbol | Caller | Reachable from preprocessor/qgemm_lut? |
|---|---|---|---|
| 9-15 | `aligned_malloc` (posix_memalign branch) | `ggml_bitnet_transform_tensor` | NO — dead code |
| 19-24 | `aligned_free` (free branch) | `ggml_bitnet_transform_tensor` | NO — dead code |
| 1442 | `aligned_malloc` (call site) | `ggml_bitnet_transform_tensor` body | NO — dead code |

**Reachability check** (manually traced from FFI entry points):

```
fjq_tl2_qgemm_lut → ggml_qgemm_lut → three_qgemm_lut_<m>_<k>
                                    + two_qgemm_lut_<m>_<k>
  three_qgemm_lut_*: alignas(32) uint32_t CBits[...] (STACK alloc)
                    + memset (libc — but freestanding-OK)
                    + three_tbl_impl_*<>(int32_t*, int8_t*, uint8_t*, uint8_t*)
  three_tbl_impl_*: pure AVX2 intrinsics, no allocations

fjq_tl2_preprocessor → ggml_preprocessor
  ggml_preprocessor: per_tensor_quant + three_lut_ctor + two_lut_ctor
  All: pure AVX2 intrinsics + scalar arithmetic, no allocations

fjq_tl2_self_test: bitnet_float_type b[4096] (STACK)
                  + per_tensor_quant
```

**Conclusion:** the FFI-reachable hot path is allocation-free.
`aligned_malloc` / `aligned_free` / `free` are reachable only
through `ggml_bitnet_transform_tensor`, which we never call from
the FFI surface.

**Mitigation options** (in order of preference):

1. **Wrap `ggml_bitnet_transform_tensor` in `#if 0`** during the
   freestanding build. One-line patch to wrapper.cpp:
   ```cpp
   #define GGML_BITNET_X86_TL2 1
   #define BITNET_OMIT_TRANSFORM 1  // F.11.4(b) freestanding build
   #include ...
   ```
   Add `#if !defined(BITNET_OMIT_TRANSFORM)` around lines
   1124-1166 in the vendored header. Trivial.

2. **Provide weak stubs** for `posix_memalign` + `free` in our
   wrapper.cpp:
   ```cpp
   extern "C" __attribute__((weak)) int posix_memalign(...) {
       return -1;  // unreachable in our build
   }
   extern "C" __attribute__((weak)) void free(...) { /* no-op */ }
   ```
   No linker errors even if symbols accidentally appear in the
   call graph.

3. **Memset / memcpy / memmove**: the kernel inner loops use
   `memset(CBits, 0, ...)`. These are libc names but
   `-ffreestanding` compilers typically intrinsify them. If the
   linker complains, add `__builtin_memset` calls or self-roll a
   trivial loop. Not expected to be needed.

`memset` is the only legitimate libc-namespaced symbol in the hot
path. Modern gcc with `-ffreestanding` emits `__builtin_memset`
inline; freestanding is ergonomic here.

## 4. Concrete F.11.4(b) plan

**Effort estimate: ~0.5-1d.** 14h F.11 chain budget headroom available.

### 4.1 Sub-tasks

| Sub-step | Effort | Verification |
|---|---|---|
| F.11.4(b).1 — patch `wrapper.cpp` to `#if 0` `ggml_bitnet_transform_tensor` (or use the BITNET_OMIT_TRANSFORM macro per §3 mitigation 1) | 0.25h | `gcc -c -ffreestanding -nostdlib wrapper.cpp` builds without unresolved symbols |
| F.11.4(b).2 — add Makefile rule in fajaros-x86 to build `fajarquant_bitnet_tl2.o` from the vendored wrapper.cpp | 0.5h | `make $(TL2_O)` produces a valid x86_64 ELF .o |
| F.11.4(b).3 — add `$(TL2_O)` to fajaros-x86 LLVM-backend kernel link line (modify Makefile step 2 `ld` invocation) | 0.5h | `make build-llvm` produces a kernel ELF without unresolved-symbol errors |
| F.11.4(b).4 — write `@ffi extern("C")` declarations in `kernel/compute/matmulfree.fj` for `fjq_tl2_qgemm_lut` + `fjq_tl2_preprocessor` (and any helpers needed) | 1h | Fajar Lang `fj check` passes; the kernel ELF still links |
| F.11.4(b).5 — add a NEW `@kernel fn km_mf_bitlinear_packed_tl2(...)` that calls into the FFI extern fns. KEEP the existing scalar `km_mf_bitlinear_packed` as fallback | 1.5h | Both functions coexist; no Fajar Lang or LLVM errors |
| F.11.4(b).6 — gate dispatch: at the call-site (e.g., `tfm_matmulfree.fj`), add a runtime check `if cpu_has_avx2() { call_tl2 } else { call_scalar }`. Initial impl can hard-code "always TL2" for the test pass; runtime CPUID check is F.13 territory | 0.5h | Smoke build runs in QEMU; no instant crash |
| F.11.4(b).7 — `make test-intllm-kernel-path` regression. PASS = ship F.11.4(b); FAIL = revert dispatch hard-coding to scalar, file finding, escalate to Branch X | 0.5-1h | Test passes 4/4 invariants OR finding doc captures regression |

**Totals**: 4.75–5.25h impl + 0.5-1h test = 5.25–6.25h. Round to ~0.5-1d.

### 4.2 Day-1 abort condition

If by EOD-1 the kernel builds but `test-intllm-kernel-path` fails
with non-trivial output regression (e.g., wrong tokens, garbled
inference), revert to scalar dispatch and pivot to Branch X
(port encoder from llama.cpp fork).

If the kernel doesn't build by EOD-1 (linker errors,
freestanding-incompat issues), defer F.11.4(b) entirely and pivot
to Branch Z (F.13 dispatch heuristic).

### 4.3 What this audit COULD NOT predict

- Whether `@kernel @ffi` is mutex'd at the lexer level (similar
  to `@no_vectorize` per V31.B.P2 finding). If yes, need a
  `@unsafe @ffi extern` wrapper called from a thin `@kernel` shim.
  Test plan: `fj check` a minimal example combining the two; if
  it errors, work around per §1 last paragraph.

- Whether the kernel-path test's safety-net coverage is strong
  enough to catch a SUBTLE bit-exact regression in TL2 vs scalar
  output. The test asserts output magnitude + sentinel tokens +
  no-crash; a 1-2% logit drift wouldn't trip it. This is the
  Branch X vs Y trade-off documented in the encoder-search doc.

---

## 5. Decision-tree summary

```
F.11.4(a') BLOCKED on tile-permutation encoder
   │
   ├─ Branch X: port encoder from llama.cpp fork (~1-1.5d)
   │  → strongest validation, biggest single-commit
   │
   ├─ Branch Y: skip parity, ship F.11.4(b) FFI binding   ← THIS DOC
   │  → ~0.5-1d, audit shows all 3 entry conditions tractable:
   │     §1 Fajar Lang FFI v2 confirmed present
   │     §2 fajaros-x86 build chain links external .o (precedent: VECMAT_O)
   │     §3 Vendored TL2 libc deps reachable only via dead code; mitigations in §3
   │  → safety net: `make test-intllm-kernel-path` 4-inv gate
   │
   └─ Branch Z: STOP F.11 at (a) PARTIAL, pivot to F.13
      → clean rabbit-hole exit
```

## 6. Recommendation (still Branch Y)

Audit confirms the audit's own preconditions are met. F.11.4(b) is
a tractable 0.5-1d task with mechanical entry conditions and a
clear day-1 abort gate. Recommend executing as the next "lanjutkan"
increment.

If the user prefers algorithmic certainty over empirical: pivot to
Branch X (1-1.5d). Otherwise Branch Y is the path forward.

---

*v1.0 — F.11.4(b) entry conditions MAPPED 2026-04-29; ready to
execute pending user nod or autonomous "lanjutkan".*

*Last updated: 2026-04-29 (V32-prep F.11.4(b) audit).*
