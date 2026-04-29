// fajarquant V32-prep F.11.1 — Thin C++ wrapper exposing a stable
// C ABI over microsoft/BitNet's vendored TL2 preset kernel.
//
// Apache-2.0 (fajarquant). Includes the upstream MIT-licensed
// preset header (see MIT_LICENSE_BITNET in this directory).
//
// Why this wrapper exists: the upstream `bitnet-lut-kernels-tl2.h`
// is a header-only kernel that compiles cleanly under
// `-DGGML_BITNET_X86_TL2 -mavx2`, but its public entry points
// (`ggml_qgemm_lut`, `ggml_preprocessor`) are C++-mangled and
// reference the (stubbed) GGML types at the type-decl level. This
// .cpp file:
//   1. Includes the preset header in a single TU
//   2. Re-exports two `extern "C"` entry points with stable names
//      (`fjq_tl2_qgemm_lut`, `fjq_tl2_preprocessor`) so cargo's
//      Rust FFI layer doesn't have to grapple with C++ name mangling.
//   3. Provides a runtime "self-test" entry point
//      (`fjq_tl2_self_test`) that the F.11.1 smoke binary uses to
//      exercise an actual AVX2 inner loop without needing a
//      shape-matched LUT.
//
// Smoke contract: `fjq_tl2_self_test` runs `per_tensor_quant` on a
// synthetic 256-float input and returns a non-NaN scale.

#define GGML_BITNET_X86_TL2 1

// F.11.3.5: kernel header is now codegen-generated for Mini shapes
// (256×256, 1536×256, 256×768, 32768×256). The freshly-generated
// header includes `<cstring>` + `<immintrin.h>` correctly at the
// top.
//
// F.11.4(b).1: keep includes freestanding-safe so the same TU
// compiles for both hosted Cargo builds AND `-ffreestanding
// -nostdlib` FajarOS Nova kernel builds (under
// BITNET_OMIT_TRANSFORM):
//   - `<cstdint>` / `<cstring>` are freestanding-safe (libstdc++
//     `requires_hosted.h` allows them; `memset` is intrinsified
//     by `-ffreestanding` so it doesn't pull in libc).
//   - `<immintrin.h>` is freestanding-safe (it's just an AVX2
//     intrinsics header).
//   - `<cmath>` and `<cstdlib>` are HOSTED-ONLY in libstdc++;
//     replaced below by a tiny constexpr NAN. We use `NAN` only
//     in `fjq_tl2_self_test` for contract-violation returns.
#include <cstdint>
#include <cstring>
#include <immintrin.h>

// Freestanding-safe NAN constant (avoid <cmath> / <cstdlib>).
// `__builtin_nanf("")` is a compiler intrinsic available in both
// gcc and clang; it expands to a quiet-NaN bit pattern at compile
// time without pulling in libm.
#ifndef NAN
#define NAN __builtin_nanf("")
#endif

#include "bitnet-lut-kernels-tl2.h"

extern "C" {

// Forward to the upstream C++ entry points (single TU).
// Signatures mirror upstream/include/ggml-bitnet.h X86_TL2 branch.
void fjq_tl2_qgemm_lut(int bs, int m, int k, int BK,
                       void* A, void* sign, void* LUT,
                       void* Scales, void* LUT_Scales, void* C) {
    ggml_qgemm_lut(bs, m, k, BK, A, sign, LUT, Scales, LUT_Scales, C);
}

void fjq_tl2_preprocessor(int bs, int m, int three_k, int two_k,
                          void* B, void* LUT_Scales,
                          void* Three_QLUT, void* Two_QLUT) {
    ggml_preprocessor(bs, m, three_k, two_k, B, LUT_Scales,
                      Three_QLUT, Two_QLUT);
}

// Self-test for F.11.1 smoke: builds a random 256-float vector,
// calls upstream `per_tensor_quant`, returns the absmax-derived
// quantization scale. Non-NaN finite return = AVX2 path was
// reachable from this binary.
//
// Per upstream convention `per_tensor_quant` writes one float into
// `lut_scales`; we forward that as the return value.
float fjq_tl2_self_test(int n_floats) {
    if (n_floats <= 0 || (n_floats % 8) != 0) {
        return NAN;  // upstream's AVX2 loop requires k % 8 == 0
    }
    bitnet_float_type b[4096];   // upper-bounded; smoke uses n=256
    bitnet_float_type lut_scales = 0.0f;
    if (n_floats > (int) (sizeof(b) / sizeof(b[0]))) {
        return NAN;
    }
    // Deterministic synthetic input: triangle wave so absmax > 0
    for (int i = 0; i < n_floats; ++i) {
        b[i] = (bitnet_float_type) ((i % 17) - 8);
    }
    int rc = per_tensor_quant(n_floats, &lut_scales, b);
    (void) rc;  // upstream returns 0 unconditionally
    return lut_scales;
}

}  // extern "C"
