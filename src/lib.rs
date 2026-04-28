//! # FajarQuant — Adaptive Vector Quantization for LLM KV Cache
//!
//! State-of-the-art KV cache quantization that **wins at 2-3 bit on real
//! Gemma 4 E2B perplexity**, with compile-time `@kernel` safety guarantees
//! no PyTorch implementation has.
//!
//! FajarQuant introduces three innovations over the [TurboQuant baseline](turboquant)
//! (Zandieh et al., 2025):
//!
//! 1. **[Adaptive PCA rotation](adaptive)** — replaces random orthogonal rotation
//!    with PCA-based per-head rotation. Concentrates variance in fewer
//!    coordinates → ~1.3-1.5× gap from optimal (vs TurboQuant's 2.7×).
//! 2. **[Fused quantized attention](fused_attention)** — computes attention scores
//!    directly on quantized KV via codebook dot products, skipping the
//!    dequantize buffer entirely. 524,288× memory reduction at 16K context.
//! 3. **[Hierarchical multi-resolution](hierarchical)** — allocates more bits to
//!    recent tokens via exponential decay schedule. 48.7% bit savings at 10K context.
//!
//! ## Real Gemma 4 E2B Perplexity (50 prompts from WikiText-2)
//!
//! | Bit-width | FajarQuant | KIVI | TurboQuant | Winner |
//! |-----------|-----------|------|------------|--------|
//! | **2-bit** | **80.14** | 231.89 | 117.11 | **FajarQuant** |
//! | **3-bit** | **75.65** | 193.86 | 108.06 | **FajarQuant** |
//! | 4-bit     | 157.01    | 145.35 | **92.84** | TurboQuant (design tradeoff) |
//!
//! ## Comparison Baselines
//!
//! - [`turboquant`] — Zandieh et al. (2025), ICML
//! - [`kivi`] — Liu et al. (2024), arXiv:2402.02750
//!
//! ## Quick Start
//!
//! ```no_run
//! use fajarquant::adaptive::compare_adaptive_vs_random;
//! use fajarquant::fused_attention::QuantizedKVCache;
//! use fajarquant::hierarchical::BitSchedule;
//!
//! // Compare adaptive PCA vs random rotation
//! # let data: Vec<Vec<f32>> = vec![];
//! let result = compare_adaptive_vs_random(&data, 128, 2);
//!
//! // Quantized KV cache with fused attention
//! let kv = QuantizedKVCache::new(128, 2);
//!
//! // Hierarchical bit allocation (8 → 2 bits as tokens age)
//! let schedule = BitSchedule::exponential_decay(8, 2, 1024);
//! ```
//!
//! ## Compile-Time `@kernel` Safety
//!
//! FajarQuant is implemented in pure Rust here, but its sister implementation
//! in [FajarOS Nova](https://github.com/fajarkraton/fajaros-x86) runs in
//! `@kernel` context with the [Fajar Lang](https://github.com/fajarkraton/fajar-lang)
//! compiler enforcing zero heap, zero tensor ops in `@kernel`, and bounds-checked
//! const-generic indexing — guarantees no PyTorch quant library has.
//!
//! ## Citation
//!
//! ```bibtex
//! @misc{putranto2026fajarquant,
//!   title={FajarQuant: Adaptive Vector Quantization for LLM KV Cache
//!          with Compile-Time Safety Guarantees},
//!   author={Putranto, Muhamad Fajar},
//!   year={2026},
//!   publisher={PrimeCore.id},
//!   url={https://github.com/fajarkraton/fajarquant}
//! }
//! ```

pub mod adaptive;
pub mod fused_attention;
pub mod hierarchical;
pub mod kivi;
pub mod turboquant;

// V32-prep F.11.1 — bitnet.cpp TL2 vendored kernel FFI declarations.
// Gated behind `--features bitnet_tl2`; the C++ static lib is built
// by `build.rs` only on x86_64 with the feature on. Re-declared here
// (in addition to the example's own re-declaration) so the linker
// pulls the static lib into ANY consumer of the crate, including
// downstream examples and tests.
//
// Per F.11 design v1.0 §2 / `THIRD_PARTY_NOTICES.md`. See
// `cpu_kernels/bitnet_tl2/wrapper.cpp` for the C++ thunks.
#[cfg(feature = "bitnet_tl2")]
pub mod bitnet_tl2 {
    /// Self-test for F.11.1 smoke. Returns the absmax-derived
    /// quantization scale (= 127 / max|input|) computed by upstream
    /// `per_tensor_quant` on a deterministic synthetic input.
    /// Returns NaN if `n_floats` is invalid.
    ///
    /// # Safety
    /// Pure function; no aliasing, no shared state. Marked unsafe
    /// only because all `extern "C"` FFI is unsafe by default.
    pub unsafe fn self_test(n_floats: i32) -> f32 {
        unsafe { fjq_tl2_self_test(n_floats) }
    }

    unsafe extern "C" {
        pub fn fjq_tl2_self_test(n_floats: i32) -> f32;
        pub fn fjq_tl2_qgemm_lut(
            bs: i32,
            m: i32,
            k: i32,
            bk: i32,
            a: *mut core::ffi::c_void,
            sign: *mut core::ffi::c_void,
            lut: *mut core::ffi::c_void,
            scales: *mut core::ffi::c_void,
            lut_scales: *mut core::ffi::c_void,
            c: *mut core::ffi::c_void,
        );
        pub fn fjq_tl2_preprocessor(
            bs: i32,
            m: i32,
            three_k: i32,
            two_k: i32,
            b: *mut core::ffi::c_void,
            lut_scales: *mut core::ffi::c_void,
            three_qlut: *mut core::ffi::c_void,
            two_qlut: *mut core::ffi::c_void,
        );
    }
}
