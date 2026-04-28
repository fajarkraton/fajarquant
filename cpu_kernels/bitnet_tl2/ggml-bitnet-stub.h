// fajarquant V32-prep F.11.1 — Minimal stub replacing microsoft/BitNet's
// `ggml-bitnet.h` for the bitnet.cpp TL2 vendor port.
//
// Apache-2.0 (fajarquant). Vendors NO upstream code; provides minimal
// type declarations so the upstream `bitnet-lut-kernels-tl2.h`
// compiles WITHOUT pulling in the full GGML header surface
// (`ggml.h` ~10K LOC, `ggml-backend.h` ~200 LOC, etc.).
//
// Why a stub: the upstream preset header references 4 GGML symbols
// in 8 sites, but the MAIN entry points (`ggml_qgemm_lut`,
// `ggml_preprocessor`) are pure-void* / int signatures and don't need
// the GGML graph machinery. Stubbing the 4 symbols at TYPE-DECL level
// lets the file compile cleanly while the real call surface stays
// minimal. Per F.11 design v1.0 §3.1 vendor-strategy choice (b).
//
// Sites stubbed:
//   - `bitnet_float_type` (1) — typedef float
//   - `bitnet_tensor_extra` (4) — struct (used in 3 statics + 1 fn arg)
//   - `enum ggml_type` (1) — used by `is_type_supported`
//   - `struct ggml_tensor` (1) — used by `ggml_bitnet_transform_tensor`
//
// `ggml_bitnet_transform_tensor` is NOT called by our smoke binary
// or by F.11.x sub-tasks; it remains as dead code so the upstream
// file compiles without modification.

#ifndef FAJARQUANT_GGML_BITNET_STUB_H
#define FAJARQUANT_GGML_BITNET_STUB_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef float bitnet_float_type;

struct bitnet_tensor_extra {
    int lut_scales_size;
    int BK;
    int n_tile_num;
    uint8_t * qweights;
    bitnet_float_type * scales;
};

// Minimal GGML types — referenced but not USED by `ggml_qgemm_lut`
// / `ggml_preprocessor`. Provided to satisfy C++ name lookup for
// `is_type_supported` and `ggml_bitnet_transform_tensor`. Layout
// here does NOT match upstream GGML; that's OK because we never
// instantiate or pass a real `ggml_tensor` through `ggml_bitnet_*`
// at runtime — the smoke binary calls only `ggml_qgemm_lut`
// (pure-void* signature).
enum ggml_type {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_TL2 = 35,
};

enum ggml_backend_type {
    GGML_BACKEND_TYPE_CPU = 0,
};

struct ggml_tensor {
    enum ggml_type type;
    enum ggml_backend_type backend;
    int64_t ne[4];
    void * data;
    void * extra;
};

#ifdef __cplusplus
}
#endif

#endif  // FAJARQUANT_GGML_BITNET_STUB_H
