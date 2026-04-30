//! V32-prep F.11.2 — Rust shim over the vendored microsoft/BitNet
//! TL2 x86 AVX2 kernel.
//!
//! This module:
//!   1. Declares the `extern "C"` FFI surface for the three C++
//!      thunks built by `build.rs` (`fjq_tl2_self_test`,
//!      `fjq_tl2_qgemm_lut`, `fjq_tl2_preprocessor`).
//!   2. Wraps the unsafe FFI in safe Rust functions that validate
//!      input pointers' alignment and shape preconditions BEFORE
//!      calling into the C++ kernel. Pre-flight validation
//!      surfaces UB-class bugs as `Err` returns instead of as
//!      segfaults.
//!   3. Exposes `self_test` — a pure-Rust safe wrapper used by
//!      `examples/tl2_smoke.rs` (the F.11.1 gate) and by the unit
//!      tests below.
//!
//! **Alignment contract.** Per F.11 design §1.2 the upstream
//! `ggml_bitnet_mul_mat_get_wsize` enforces 64-byte alignment on
//! the workspace; vpshufb on AVX2 wants 32-byte at minimum. We
//! adopt **64-byte alignment** uniformly so callers don't have to
//! reason per-pointer about which threshold applies.
//!
//! **Shape support.** F.11.3.5 ships the `fajarquant_mini` codegen
//! output via `utils/codegen_tl2.py`. Supported shapes are the
//! Phase D Mini ckpt's BitLinears: (256, 256) for the four attention
//! projections, (1536, 256) for `mlp.gate_proj` (output is
//! `2 × intermediate` per HGRNBitMLP swiglu chunking), (256, 768) for
//! `mlp.down_proj`, and (32768, 256) for `lm_head`. Other `(m, k)`
//! pairs return `ShimError::UnsupportedShape`. To support new
//! shapes, add them to `ModelShapeDict["fajarquant_mini"]` in
//! `cpu_kernels/bitnet_tl2/codegen/codegen_tl2.py` and re-run
//! codegen (see that directory's README at the top of the script).

use core::ffi::c_void;

// ---------------------------------------------------------------------------
// §1. extern "C" FFI surface
// ---------------------------------------------------------------------------

unsafe extern "C" {
    /// Smoke entry: runs `per_tensor_quant` on a deterministic
    /// triangle-wave synthetic input of length `n_floats` and
    /// returns the absmax-derived quantization scale (= 127 / max|b|).
    /// Returns NaN on contract violation (n <= 0, n % 8 != 0,
    /// n > 4096).
    pub fn fjq_tl2_self_test(n_floats: i32) -> f32;

    /// TL2 quantized GEMM via lookup-table. Shape-locked to the
    /// vendored preset; see §1 caveat above.
    pub fn fjq_tl2_qgemm_lut(
        bs: i32,
        m: i32,
        k: i32,
        bk: i32,
        a: *mut c_void,
        sign: *mut c_void,
        lut: *mut c_void,
        scales: *mut c_void,
        lut_scales: *mut c_void,
        c: *mut c_void,
    );

    /// Activation preprocessor: builds the `Three_QLUT` (3-weight
    /// canonical) and `Two_QLUT` (K mod 3 tail) lookup tables for a
    /// given activation block.
    pub fn fjq_tl2_preprocessor(
        bs: i32,
        m: i32,
        three_k: i32,
        two_k: i32,
        b: *mut c_void,
        lut_scales: *mut c_void,
        three_qlut: *mut c_void,
        two_qlut: *mut c_void,
    );
}

// ---------------------------------------------------------------------------
// §2. Alignment + shape validation
// ---------------------------------------------------------------------------

/// Required pointer alignment for all TL2 buffer arguments. Adopting
/// the strictest threshold (workspace alignment from upstream
/// `ggml_bitnet_mul_mat_get_wsize`) so callers don't have to track
/// per-pointer requirements.
pub const TL2_ALIGNMENT: usize = 64;

/// Vendored preset shapes. F.11.3.5 ships the
/// `fajarquant_mini` codegen output: BitLinear shapes for the
/// Phase D Mini ckpt (hidden=256, intermediate=768, vocab=32768).
/// Any (m, k) pair not in this set returns
/// `ShimError::UnsupportedShape` from `qgemm_lut`.
pub const TL2_SUPPORTED_SHAPES: &[(i32, i32)] = &[
    (256, 256),   // attn.{i,f,g,o}_proj × 4 sites
    (1536, 256),  // mlp.gate_proj (out = 2 × intermediate)
    (256, 768),   // mlp.down_proj
    (32768, 256), // lm_head
];

/// Errors surfaced by the safe shim BEFORE calling into FFI. By
/// validating up-front, we turn UB-class bugs (mis-aligned ptr,
/// out-of-spec shape) into ordinary `Result::Err` returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShimError {
    /// One of the input pointers is not aligned to [`TL2_ALIGNMENT`].
    NotAligned {
        which: &'static str,
        ptr_addr: usize,
    },
    /// `(m, k)` is not in [`TL2_SUPPORTED_SHAPES`]. F.11.3 will lift
    /// this constraint by generating per-shape preset headers OR by
    /// shipping a Rust fallback path.
    UnsupportedShape { m: i32, k: i32 },
    /// `n_floats` failed the per_tensor_quant contract (must be
    /// > 0, ≤ 4096, divisible by 8).
    BadSelfTestN { n: i32 },
}

/// Return `true` iff `ptr` is aligned to [`TL2_ALIGNMENT`].
#[inline]
pub fn is_tl2_aligned<T>(ptr: *const T) -> bool {
    (ptr as usize).is_multiple_of(TL2_ALIGNMENT)
}

// ---------------------------------------------------------------------------
// §3. Safe wrappers
// ---------------------------------------------------------------------------

/// Safe-Rust front for `fjq_tl2_self_test`. Validates the F.11.1
/// smoke contract (n > 0, n ≤ 4096, n % 8 == 0) up-front and
/// returns a plain `Result<f32, ShimError>` instead of NaN-on-fail.
///
/// `Ok(scale)` carries the absmax-derived quantization scale
/// computed by upstream `per_tensor_quant`. Used by the F.11.1
/// smoke binary and by `tests::smoke_call_success`.
pub fn self_test(n_floats: i32) -> Result<f32, ShimError> {
    if n_floats <= 0 || n_floats > 4096 || n_floats % 8 != 0 {
        return Err(ShimError::BadSelfTestN { n: n_floats });
    }
    // SAFETY: contract validated above; the FFI is a pure function
    // (no aliasing, no side effects beyond the return value).
    let scale = unsafe { fjq_tl2_self_test(n_floats) };
    Ok(scale)
}

/// Safe-Rust front for `fjq_tl2_qgemm_lut`. Validates pointer
/// alignment and `(m, k)` shape support BEFORE calling into the
/// C++ kernel.
///
/// # Safety
///
/// Even after the safe-Rust pre-validation, the caller must
/// guarantee that:
///   - All pointer arguments point to valid, mutable buffers of
///     the size required by the upstream kernel for the given
///     `(bs, m, k, bk)` tile parameters.
///   - The `a` and `sign` buffers contain TL2-encoded weights for
///     the corresponding `(m, k)` shape.
///   - No other thread is reading/writing those buffers
///     concurrently.
///
/// These are upstream ABI constraints we cannot mechanically
/// check. The function is therefore `unsafe` even though the
/// wrapper does some pre-validation.
#[allow(clippy::too_many_arguments)]
pub unsafe fn qgemm_lut(
    bs: i32,
    m: i32,
    k: i32,
    bk: i32,
    a: *mut c_void,
    sign: *mut c_void,
    lut: *mut c_void,
    scales: *mut c_void,
    lut_scales: *mut c_void,
    c: *mut c_void,
) -> Result<(), ShimError> {
    if !TL2_SUPPORTED_SHAPES.contains(&(m, k)) {
        return Err(ShimError::UnsupportedShape { m, k });
    }
    for (label, ptr) in [
        ("a", a),
        ("sign", sign),
        ("lut", lut),
        ("scales", scales),
        ("lut_scales", lut_scales),
        ("c", c),
    ] {
        if !is_tl2_aligned(ptr) {
            return Err(ShimError::NotAligned {
                which: label,
                ptr_addr: ptr as usize,
            });
        }
    }
    // SAFETY: alignment + shape validated; the caller's #Safety
    // guarantees cover the buffer sizes.
    unsafe { fjq_tl2_qgemm_lut(bs, m, k, bk, a, sign, lut, scales, lut_scales, c) };
    Ok(())
}

// ---------------------------------------------------------------------------
// §4. Unit tests (F.11.2 verification gate)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// alignment-OK: a freshly allocated `Box<[u8; N]>` happens to be
    /// at least 8-byte aligned by Rust's allocator; for a
    /// `repr(align(64))` wrapper, it's 64-byte. Verify the
    /// alignment helper agrees.
    #[test]
    fn alignment_ok_for_64byte_aligned_buffer() {
        #[repr(align(64))]
        struct Aligned([u8; 128]);
        let buf = Aligned([0u8; 128]);
        let ptr: *const u8 = buf.0.as_ptr();
        assert!(
            is_tl2_aligned(ptr),
            "repr(align(64)) buffer should be 64-byte aligned, but ptr {ptr:?} is not"
        );
    }

    /// alignment-fail: an unaligned pointer (constructed by
    /// offsetting a 64-byte-aligned base by 1 byte) is rejected.
    #[test]
    fn alignment_fail_for_offset_pointer() {
        #[repr(align(64))]
        struct Aligned([u8; 128]);
        let buf = Aligned([0u8; 128]);
        // SAFETY: still inside the 128-byte allocation
        let unaligned = unsafe { buf.0.as_ptr().add(1) };
        assert!(
            !is_tl2_aligned(unaligned),
            "ptr+1 from a 64-byte-aligned base must NOT be 64-byte aligned"
        );

        // Demonstrate the safe wrapper's ShimError surface: pass a
        // SUPPORTED Mini shape with an unaligned pointer. The shape
        // check fires FIRST (before alignment), so we MUST use a
        // current Mini shape here — (256, 256) is in the
        // F.11.3.5 supported set.
        let bogus_ptr = 1usize as *mut c_void; // unaligned by construction
        assert!(!is_tl2_aligned(bogus_ptr));
        let result = unsafe {
            qgemm_lut(
                1, 256, 256, 128, bogus_ptr, bogus_ptr, bogus_ptr, bogus_ptr, bogus_ptr, bogus_ptr,
            )
        };
        assert!(matches!(result, Err(ShimError::NotAligned { .. })));
    }

    /// smoke-call-success: end-to-end FFI round-trip. Mirrors the
    /// F.11.1 smoke binary's assertion that
    /// `per_tensor_quant(triangle-wave) = 127/8 = 15.875`.
    #[test]
    fn smoke_call_success() {
        let n: i32 = 256;
        let scale = self_test(n).expect("self_test on n=256 should succeed");
        assert!(scale.is_finite(), "scale must be finite");
        assert!(scale > 0.0, "scale must be positive");
        let expected = 127.0_f32 / 8.0;
        let rel_err = (scale - expected).abs() / expected;
        assert!(
            rel_err < 1e-3,
            "rel_err {rel_err:.3e} > 1e-3; AVX2 path produced wrong value (got {scale}, expected {expected})"
        );
    }

    /// return-zero-on-empty: `n_floats <= 0` is caught BEFORE the
    /// FFI call (which would have returned NaN). Result is a
    /// `BadSelfTestN` error, not a panic and not a NaN.
    #[test]
    fn return_zero_on_empty() {
        for bad_n in [0, -1, 7, 4097] {
            let result = self_test(bad_n);
            assert!(
                matches!(result, Err(ShimError::BadSelfTestN { n }) if n == bad_n),
                "bad_n={bad_n} should map to BadSelfTestN, got {result:?}"
            );
        }
        // Sanity: the smallest valid n (8) still works.
        assert!(self_test(8).is_ok());
    }

    /// Bonus: shape-locking is enforced. Calling `qgemm_lut` with a
    /// (m, k) outside the preset's three supported shapes returns
    /// `UnsupportedShape` instead of segfaulting.
    #[test]
    fn unsupported_shape_rejected() {
        // Use a 64-byte-aligned bogus pointer so the shape check
        // (which runs FIRST) is what trips, not the alignment check.
        #[repr(align(64))]
        struct Aligned([u8; 64]);
        let buf = Aligned([0u8; 64]);
        let p = buf.0.as_ptr() as *mut c_void;

        // (768, 1024) is NOT in TL2_SUPPORTED_SHAPES even after the
        // F.11.3.5 Mini codegen swap (Mini covers 256 / 1536 / 32768
        // dims, not 768 outputs).
        let result = unsafe { qgemm_lut(1, 768, 1024, 96, p, p, p, p, p, p) };
        assert!(
            matches!(result, Err(ShimError::UnsupportedShape { m: 768, k: 1024 })),
            "(768, 1024) should be rejected, got {result:?}"
        );
    }

    /// F.11.4(a) PARTIAL: end-to-end exercise the `fjq_tl2_preprocessor`
    /// FFI surface. The preprocessor takes only activation buffers
    /// (B, LUT_Scales, Three_QLUT, Two_QLUT) — NOT the A/sign weight
    /// buffers. This means we can validate preprocessor's correctness
    /// WITHOUT needing the upstream Python convert.py path that lays
    /// out tile-organized A/sign (which is the F.11.4(b) prerequisite).
    ///
    /// What this test verifies:
    ///   1. The preprocessor entry point links and runs without
    ///      segfaulting on properly-aligned, properly-sized buffers.
    ///   2. LUT_Scales matches `per_tensor_quant`'s formula
    ///      (127 / max|b|), already validated by `self_test`.
    ///   3. Three_QLUT gets meaningfully non-zero entries (at least
    ///      one byte != 0), confirming `three_lut_ctor` actually
    ///      ran on AVX2 inside the preprocessor.
    #[test]
    fn preprocessor_smoke_at_mini_256_256() {
        // Allocate 64-byte-aligned buffers using a #[repr(align(64))]
        // struct wrapper. Sizes per upstream codegen:
        //   B: k floats = 256 floats = 1024 bytes (need to round up)
        //   LUT_Scales: bs floats = 1 float
        //   Three_QLUT: bs * three_k / 3 * 32 bytes per upstream
        //               codegen. For three_k=256, bs=1: 256/3 = 85,
        //               * 32 = 2720 bytes. Round up to 64-byte
        //               aligned: 2752 bytes.
        //   Two_QLUT: bs * two_k / 2 * 32 = 0 for two_k=0; pass a
        //             non-null aligned pointer regardless to keep
        //             the FFI guard happy.
        const K: usize = 256;
        const M: i32 = 256;
        const THREE_K: i32 = 256;
        const TWO_K: i32 = 0;
        const BS: i32 = 1;

        #[repr(align(64))]
        struct AlignedF32<const N: usize>([f32; N]);
        #[repr(align(64))]
        struct AlignedI8<const N: usize>([i8; N]);

        // Triangle-wave activations (max|b| = 8 → LUT_Scales = 127/8 = 15.875)
        let mut b_buf = AlignedF32([0.0_f32; K]);
        for (i, slot) in b_buf.0.iter_mut().enumerate() {
            *slot = ((i % 17) as i32 - 8) as f32;
        }
        let mut lut_scales_buf = AlignedF32::<8>([0.0_f32; 8]); // 32B aligned to 64B
        let mut three_qlut_buf = AlignedI8::<2752>([0_i8; 2752]);
        let mut two_qlut_buf = AlignedI8::<64>([0_i8; 64]); // unused, but FFI wants ptr

        // SAFETY: all pointers are 64-byte aligned (repr(align(64)))
        // and exclusively borrowed; FFI runs single-threaded.
        unsafe {
            super::fjq_tl2_preprocessor(
                BS,
                M,
                THREE_K,
                TWO_K,
                b_buf.0.as_mut_ptr() as *mut c_void,
                lut_scales_buf.0.as_mut_ptr() as *mut c_void,
                three_qlut_buf.0.as_mut_ptr() as *mut c_void,
                two_qlut_buf.0.as_mut_ptr() as *mut c_void,
            );
        }

        // Verify LUT_Scales[0] matches the self_test formula
        // (127 / max|b|). Triangle wave max|b| = 8 → expected 15.875.
        let scale = lut_scales_buf.0[0];
        let expected = 127.0_f32 / 8.0;
        let rel_err = (scale - expected).abs() / expected;
        assert!(
            rel_err < 1e-3,
            "LUT_Scales[0] = {scale:.6} should match per_tensor_quant 127/max|b| = {expected:.6} (rel_err {rel_err:.4e})"
        );

        // Verify Three_QLUT has meaningful content (not still all
        // zeros). At least 16 bytes should be non-zero — `three_lut_ctor`
        // emits 14-entry magnitude LUTs at fixed offsets.
        let nonzero = three_qlut_buf.0.iter().filter(|&&b| b != 0).count();
        assert!(
            nonzero > 16,
            "Three_QLUT should be populated; only {nonzero} non-zero bytes out of 2720 expected"
        );
    }

    /// F.11.4 Branch X-real iteration 5 — parity test against scalar
    /// baseline. Builds ONE TILE (BM=64 rows, K=256) of ternary
    /// weights, encodes V31 + TL2 tile-organized, runs both paths,
    /// compares i32 accumulator outputs.
    ///
    /// **Expected status: this test MAY FAIL on first run** —
    /// iteration 4's sign-bit position is a guess. Failure mode +
    /// diff guide iteration 6's debug.
    ///
    /// Strategy: keep the test atomic (one tile, deterministic
    /// inputs) so failures are debuggable. Generalize to all 4
    /// tiles once parity passes.
    #[test]
    #[ignore = "iteration 5 — known to potentially fail on sign-bit position; \
                run with `cargo test --release --features bitnet_tl2 parity -- --ignored --nocapture` \
                to capture diff for iteration 6 debug"]
    fn parity_real_mlp_one_tile_at_mini_256_256() {
        use crate::cpu_kernels::scalar_baseline::{absmax_quantize_i8, bitlinear_packed_scalar};
        use crate::cpu_kernels::tl2_encoder::pack_tile_organized;

        // Test geometry: ONE tile (BM=64 rows) of (256, 256) shape.
        const M_FULL: i32 = 256; // shape selector for the kernel dispatch
        const K: i32 = 256;
        const BM: usize = 64;
        const BBK: usize = 128;
        const BS: i32 = 1;

        // Deterministic synthetic ternary weights. Pattern: cycle
        // through (0, 0, 0), (1, 0, 0), (0, 1, 0), ..., to exercise
        // the magnitude index LUT at multiple values.
        let mut weights = vec![0_i8; BM * K as usize];
        for j in 0..BM {
            for i in 0..K as usize {
                weights[j * K as usize + i] = match (j + i) % 3 {
                    0 => 0,
                    1 => 1,
                    _ => -1,
                };
            }
        }

        // Float activations: triangle wave so absmax = 8.
        let activations: Vec<f32> = (0..K as usize)
            .map(|i| ((i % 17) as i32 - 8) as f32)
            .collect();
        let (act_i8, _gamma_x) = absmax_quantize_i8(&activations);

        // -------- SCALAR PATH --------
        // Pack V31 row-major: 4 weights/byte, 2 bits/weight LE.
        let mut v31_bytes = vec![0_u8; BM * K as usize / 4];
        for (i, &w) in weights.iter().enumerate() {
            let code: u8 = match w {
                -1 => 0b00,
                0 => 0b01,
                1 => 0b10,
                _ => unreachable!(),
            };
            v31_bytes[i / 4] |= code << ((i % 4) * 2);
        }
        let scalar_y = bitlinear_packed_scalar(&v31_bytes, &act_i8, BM, K as usize)
            .expect("scalar bitlinear should succeed");

        // -------- TL2 PATH --------
        let layout = pack_tile_organized(&weights, BM, K as usize, BM, BBK);

        // Allocate aligned buffers. Sizes for BS=1, M=256-shape,
        // three_k=256, two_k=0:
        //   B float: K = 256 floats
        //   LUT_Scales: BS = 1 float (round up to 64B align)
        //   Three_QLUT: BS * three_k/3 * 32 = 256/3*32 = 2720 bytes
        //   Two_QLUT: 0 bytes (two_k=0); pass non-null aligned scratch
        //   A: 1 tile worth = a_per_tile bytes (= 2 * 1344 = 2688)
        //   sign: 1 tile worth = s_per_tile bytes (= 2 * 320 = 640)
        //   Scales: 1 float
        //   C: BM = 64 i32 = 256 bytes

        #[repr(align(64))]
        struct AlignedF32<const N: usize>([f32; N]);
        #[repr(align(64))]
        struct AlignedI8<const N: usize>([i8; N]);
        #[repr(align(64))]
        struct AlignedU8<const N: usize>([u8; N]);
        #[repr(align(64))]
        struct AlignedI32<const N: usize>([i32; N]);

        let mut b_buf = AlignedF32([0.0_f32; 256]);
        for (i, slot) in b_buf.0.iter_mut().enumerate() {
            *slot = activations[i];
        }
        let mut lut_scales_buf = AlignedF32::<8>([0.0_f32; 8]);
        let mut three_qlut_buf = AlignedI8::<2752>([0_i8; 2752]);
        let mut two_qlut_buf = AlignedI8::<64>([0_i8; 64]);
        let mut scales_buf = AlignedF32::<8>([1.0_f32; 8]); // β scale
        let mut c_buf = AlignedI32::<64>([0_i32; 64]);

        // Magnitude buffer must be 64-byte aligned + sized for ONE tile.
        let mut a_buf = AlignedU8::<2752>([0_u8; 2752]); // pad 2688→2752
        a_buf.0[..layout.magnitudes.len() / layout.n_tiles]
            .copy_from_slice(&layout.magnitudes[..layout.magnitudes.len() / layout.n_tiles]);
        let mut sign_buf = AlignedU8::<704>([0_u8; 704]); // pad 640→704
        sign_buf.0[..layout.signs.len() / layout.n_tiles]
            .copy_from_slice(&layout.signs[..layout.signs.len() / layout.n_tiles]);

        // Run preprocessor + qgemm_lut three-path
        unsafe {
            super::fjq_tl2_preprocessor(
                BS,
                M_FULL,
                K,
                0,
                b_buf.0.as_mut_ptr() as *mut c_void,
                lut_scales_buf.0.as_mut_ptr() as *mut c_void,
                three_qlut_buf.0.as_mut_ptr() as *mut c_void,
                two_qlut_buf.0.as_mut_ptr() as *mut c_void,
            );
            super::fjq_tl2_qgemm_lut(
                BS,
                M_FULL,
                K,
                256, // BK=256 dispatches three-path per kernel header line 1303
                a_buf.0.as_mut_ptr() as *mut c_void,
                sign_buf.0.as_mut_ptr() as *mut c_void,
                three_qlut_buf.0.as_mut_ptr() as *mut c_void,
                scales_buf.0.as_mut_ptr() as *mut c_void,
                lut_scales_buf.0.as_mut_ptr() as *mut c_void,
                c_buf.0.as_mut_ptr() as *mut c_void,
            );
        }

        // -------- COMPARE --------
        // Scalar y is i64; TL2 C is i32. Cast for comparison.
        let mut diffs = Vec::new();
        for i in 0..BM {
            let scalar_val = scalar_y[i] as i32;
            let tl2_val = c_buf.0[i];
            if scalar_val != tl2_val {
                diffs.push((i, scalar_val, tl2_val));
            }
        }

        if !diffs.is_empty() {
            eprintln!(
                "PARITY DIFF — {} of {} outputs mismatch (first 5):",
                diffs.len(),
                BM
            );
            for &(i, sv, tv) in diffs.iter().take(20) {
                eprintln!("  row {i}: scalar = {sv}, tl2 = {tv}, diff = {}", tv - sv);
            }
            eprintln!(
                "Likely cause: iteration 4 sign-bit position guess is wrong. \
                 Iteration 6 debug: try alternative bit-in-byte permutations \
                 (interleaved hi/lo, 4-bit-per-byte, etc.)"
            );
        }

        assert!(
            diffs.is_empty(),
            "{} of {} outputs differ — see eprintln above for diff sample",
            diffs.len(),
            BM
        );
    }

    /// F.11.4 Branch X-real Path B step 2 — extended derivation
    /// covering row 0 triplet 1, row 1 triplet 0, row 16 triplet 0,
    /// row 0 triplet 2 (different ai). Same probe pattern as
    /// step 1: set magnitude byte for the target (row, triplet)
    /// to mag=13, all signs zero baseline, then sweep all bit
    /// positions in the relevant byte pair.
    #[test]
    #[ignore = "Path B step 2 — extended derivation across multiple positions"]
    fn derive_sign_bit_position_extended() {
        use crate::cpu_kernels::scalar_baseline::absmax_quantize_i8;

        const M_FULL: i32 = 256;
        const K: i32 = 256;
        const BS: i32 = 1;

        let activations: Vec<f32> = (0..K as usize)
            .map(|i| ((i % 17) as i32 - 8) as f32)
            .collect();
        let _ = absmax_quantize_i8(&activations);

        #[repr(align(64))]
        struct AlignedF32<const N: usize>([f32; N]);
        #[repr(align(64))]
        struct AlignedI8<const N: usize>([i8; N]);
        #[repr(align(64))]
        struct AlignedU8<const N: usize>([u8; N]);
        #[repr(align(64))]
        struct AlignedI32<const N: usize>([i32; N]);

        let test_cases: &[(&str, usize, usize, u8, usize, usize)] = &[
            // (label, row, mag_byte_offset, mag_byte_value, sign_byte_pair_lo, output_idx_to_check)
            ("row=0  triplet=0 (ai=0 nib=0)", 0, 0, 0xD0, 0, 0),
            ("row=0  triplet=1 (ai=0 nib=1)", 0, 0, 0x0D, 0, 0),
            ("row=0  triplet=2 (ai=1 nib=0)", 0, 32, 0xD0, 0, 0),
            ("row=1  triplet=0 (ai=0 nib=0)", 1, 1, 0xD0, 2, 1),
            ("row=15 triplet=0 (ai=0 nib=0)", 15, 15, 0xD0, 30, 15),
            ("row=16 triplet=0 (ai=0 nib=0)", 16, 672, 0xD0, 0, 16),
            ("row=17 triplet=0 (ai=0 nib=0)", 17, 673, 0xD0, 2, 17),
        ];

        for &(label, row, mag_off, mag_val, sign_pair_lo, out_idx) in test_cases {
            // Run with all signs zero
            let baseline = run_kernel(mag_off, mag_val, &[0_u8; 704], out_idx);
            eprintln!("\n{label}: row={row}, baseline (signs=0) C[{out_idx}] = {baseline}");

            for byte_idx in sign_pair_lo..sign_pair_lo + 2 {
                for bit in 0..8_u8 {
                    let mut signs = [0_u8; 704];
                    signs[byte_idx] = 1 << bit;
                    let result = run_kernel(mag_off, mag_val, &signs, out_idx);
                    let bit_pos_in_lane = (byte_idx - sign_pair_lo) * 8 + bit as usize;
                    let flipped = result.signum() != baseline.signum() && baseline != 0;
                    if flipped || result == -baseline {
                        eprintln!(
                            "  byte {byte_idx} bit {bit} (lane bit {bit_pos_in_lane:2}): \
                             C[{out_idx}] = {result:6}  {}",
                            if result == -baseline {
                                "← EXACT NEG"
                            } else {
                                "← FLIPS SIGN"
                            }
                        );
                    }
                }
            }
        }

        fn run_kernel(
            mag_byte_off: usize,
            mag_byte_val: u8,
            sign_bytes: &[u8; 704],
            out_idx: usize,
        ) -> i32 {
            const BS: i32 = 1;
            const M_FULL: i32 = 256;
            const K: i32 = 256;

            #[repr(align(64))]
            struct AlignedF32<const N: usize>([f32; N]);
            #[repr(align(64))]
            struct AlignedI8<const N: usize>([i8; N]);
            #[repr(align(64))]
            struct AlignedU8<const N: usize>([u8; N]);
            #[repr(align(64))]
            struct AlignedI32<const N: usize>([i32; N]);

            let mut a_buf = AlignedU8::<2752>([0_u8; 2752]);
            a_buf.0[mag_byte_off] = mag_byte_val;

            let mut sign_buf = AlignedU8::<704>([0_u8; 704]);
            sign_buf.0.copy_from_slice(sign_bytes);

            let mut b_buf = AlignedF32([0.0_f32; 256]);
            for (i, slot) in b_buf.0.iter_mut().enumerate() {
                *slot = ((i % 17) as i32 - 8) as f32;
            }
            let mut lut_scales_buf = AlignedF32::<8>([0.0_f32; 8]);
            let mut three_qlut_buf = AlignedI8::<2752>([0_i8; 2752]);
            let mut two_qlut_buf = AlignedI8::<64>([0_i8; 64]);
            let mut scales_buf = AlignedF32::<8>([1.0_f32; 8]);
            let mut c_buf = AlignedI32::<64>([0_i32; 64]);

            unsafe {
                super::fjq_tl2_preprocessor(
                    BS,
                    M_FULL,
                    K,
                    0,
                    b_buf.0.as_mut_ptr() as *mut c_void,
                    lut_scales_buf.0.as_mut_ptr() as *mut c_void,
                    three_qlut_buf.0.as_mut_ptr() as *mut c_void,
                    two_qlut_buf.0.as_mut_ptr() as *mut c_void,
                );
                super::fjq_tl2_qgemm_lut(
                    BS,
                    M_FULL,
                    K,
                    256, // BK=256 dispatches three-path per kernel header line 1303
                    a_buf.0.as_mut_ptr() as *mut c_void,
                    sign_buf.0.as_mut_ptr() as *mut c_void,
                    three_qlut_buf.0.as_mut_ptr() as *mut c_void,
                    scales_buf.0.as_mut_ptr() as *mut c_void,
                    lut_scales_buf.0.as_mut_ptr() as *mut c_void,
                    c_buf.0.as_mut_ptr() as *mut c_void,
                );
            }
            c_buf.0[out_idx]
        }
    }

    /// F.11.4 Branch X-real Path B step 1 — derive correct sign-bit
    /// position empirically by trial-and-error against the live
    /// AVX2 kernel.
    ///
    /// Setup: row 0 triplet 0 = (-1,-1,-1) → magnitude index 13,
    /// sign should be 1 (negative). All other triplets zero.
    /// Activations: triangle wave (`(i % 17) - 8`).
    ///
    /// Expected behavior:
    ///   - Sign all-zero baseline: kernel computes Σ q_x[0..2]
    ///     using LUT-stored canonical magnitude (positive value)
    ///   - Sign with correct bit set: result sign-flips
    ///
    /// Probe ALL 16 bit positions of vec_signs[0] lane 0
    /// (= bytes 0..1 of sign buffer). Whichever position flips
    /// the C[0] output sign is THE answer for triplet 0 / row 0.
    ///
    /// Run: `cargo test --release --features bitnet_tl2
    ///   derive_sign -- --ignored --nocapture`
    #[test]
    #[ignore = "Path B step 1 — diagnostic harness, prints \
                bit-position map; not a pass/fail test"]
    fn derive_sign_bit_position_for_first_triplet() {
        use crate::cpu_kernels::scalar_baseline::absmax_quantize_i8;

        const M_FULL: i32 = 256;
        const K: i32 = 256;
        const BS: i32 = 1;

        let activations: Vec<f32> = (0..K as usize)
            .map(|i| ((i % 17) as i32 - 8) as f32)
            .collect();
        let (_, _gamma_x) = absmax_quantize_i8(&activations);

        #[repr(align(64))]
        struct AlignedF32<const N: usize>([f32; N]);
        #[repr(align(64))]
        struct AlignedI8<const N: usize>([i8; N]);
        #[repr(align(64))]
        struct AlignedU8<const N: usize>([u8; N]);
        #[repr(align(64))]
        struct AlignedI32<const N: usize>([i32; N]);

        // Helper closure: run qgemm_lut with given sign byte values,
        // return C[0].
        let mut run_with_sign = |sign_bytes: &[u8; 704]| -> i32 {
            // Magnitude buffer: byte 0 = 0xD0 (upper nibble idx=13
            // = canonical (1,1,1); lower nibble = 0). All other
            // bytes 0.
            let mut a_buf = AlignedU8::<2752>([0_u8; 2752]);
            a_buf.0[0] = 0xD0;

            let mut sign_buf = AlignedU8::<704>([0_u8; 704]);
            sign_buf.0.copy_from_slice(sign_bytes);

            let mut b_buf = AlignedF32([0.0_f32; 256]);
            for (i, slot) in b_buf.0.iter_mut().enumerate() {
                *slot = activations[i];
            }
            let mut lut_scales_buf = AlignedF32::<8>([0.0_f32; 8]);
            let mut three_qlut_buf = AlignedI8::<2752>([0_i8; 2752]);
            let mut two_qlut_buf = AlignedI8::<64>([0_i8; 64]);
            let mut scales_buf = AlignedF32::<8>([1.0_f32; 8]);
            let mut c_buf = AlignedI32::<64>([0_i32; 64]);

            unsafe {
                super::fjq_tl2_preprocessor(
                    BS,
                    M_FULL,
                    K,
                    0,
                    b_buf.0.as_mut_ptr() as *mut c_void,
                    lut_scales_buf.0.as_mut_ptr() as *mut c_void,
                    three_qlut_buf.0.as_mut_ptr() as *mut c_void,
                    two_qlut_buf.0.as_mut_ptr() as *mut c_void,
                );
                super::fjq_tl2_qgemm_lut(
                    BS,
                    M_FULL,
                    K,
                    256, // BK=256 dispatches three-path per dispatcher line 1303 (BK=128 was the no-op)
                    a_buf.0.as_mut_ptr() as *mut c_void,
                    sign_buf.0.as_mut_ptr() as *mut c_void,
                    three_qlut_buf.0.as_mut_ptr() as *mut c_void,
                    scales_buf.0.as_mut_ptr() as *mut c_void,
                    lut_scales_buf.0.as_mut_ptr() as *mut c_void,
                    c_buf.0.as_mut_ptr() as *mut c_void,
                );
            }
            c_buf.0[0]
        };

        // Baseline: all-zero signs
        let zero_signs = [0_u8; 704];
        let baseline = run_with_sign(&zero_signs);
        eprintln!("baseline (signs=0): C[0] = {baseline}");
        eprintln!(
            "Probing bit positions of vec_signs[0] lane 0 (bytes 0..1):\n\
             For each bit_pos in 0..16, set ONLY that bit, observe C[0]:"
        );

        for byte_idx in 0..2_usize {
            for bit in 0..8_u8 {
                let mut signs = zero_signs;
                signs[byte_idx] = 1 << bit;
                let result = run_with_sign(&signs);
                let bit_pos_in_lane = byte_idx * 8 + bit as usize;
                let flipped = result.signum() != baseline.signum() && baseline != 0;
                eprintln!(
                    "  byte {byte_idx} bit {bit} (lane bit {bit_pos_in_lane:2}): \
                     C[0] = {result:6}  {} {}",
                    if flipped { "← FLIPS SIGN" } else { "" },
                    if result == -baseline {
                        "← EXACT NEG"
                    } else {
                        ""
                    },
                );
            }
        }

        eprintln!(
            "\nIf any bit_pos flipped sign, that's THE answer for \
             row 0 / triplet 0. Iter 6 predicted bit_pos=15 \
             (byte 1, bit 7). Check above for actual."
        );
    }

    /// F.11.4 Branch X-real Path B step 4 — byte→row mapping probe.
    ///
    /// For each magnitude byte position `bp` in vec_as[0] (bytes
    /// 0..32 of A buffer), set ONLY a_buf[bp]=0xDD (canonical (1,1,1)
    /// for both nibbles), signs=0, run kernel, observe which C[r] is
    /// non-zero. This reveals the byte→row mapping inside vec_as[0]
    /// definitively. Then probe vec_as[1] (bytes 32..64), vec_as[2]
    /// (bytes 64..96), … to see if k-block offset shifts the active
    /// row.
    ///
    /// What we learn:
    ///   - Whether byte b in vec_as[0] maps to row b (linear) or
    ///     some shuffled mapping per AVX2 vpshufb + unpackhi/unpacklo.
    ///   - Whether different vec_as[ai] hit the same rows (k-stride
    ///     accumulation) or different rows (row-stride layout).
    ///   - Whether vec_as[20] is genuinely dropped (resume-protocol
    ///     iter 7 finding from FJQ_PHASE_F_F11_X7_HYPOTHESIS_FINDINGS).
    #[test]
    #[ignore = "Path B step 4 — byte→row mapping probe; prints map, not pass/fail"]
    fn derive_byte_to_row_mapping() {
        const M_FULL: i32 = 256;
        const K: i32 = 256;
        const BS: i32 = 1;

        #[repr(align(64))]
        struct AlignedF32<const N: usize>([f32; N]);
        #[repr(align(64))]
        struct AlignedI8<const N: usize>([i8; N]);
        #[repr(align(64))]
        struct AlignedU8<const N: usize>([u8; N]);
        #[repr(align(64))]
        struct AlignedI32<const N: usize>([i32; N]);

        let run_with_mag = |mag_byte_off: usize, mag_byte_val: u8| -> [i32; 64] {
            let mut a_buf = AlignedU8::<2752>([0_u8; 2752]);
            a_buf.0[mag_byte_off] = mag_byte_val;
            let mut sign_buf = AlignedU8::<704>([0_u8; 704]);
            let mut b_buf = AlignedF32([0.0_f32; 256]);
            for (i, slot) in b_buf.0.iter_mut().enumerate() {
                *slot = ((i % 17) as i32 - 8) as f32;
            }
            let mut lut_scales_buf = AlignedF32::<8>([0.0_f32; 8]);
            let mut three_qlut_buf = AlignedI8::<2752>([0_i8; 2752]);
            let mut two_qlut_buf = AlignedI8::<64>([0_i8; 64]);
            let mut scales_buf = AlignedF32::<8>([1.0_f32; 8]);
            let mut c_buf = AlignedI32::<64>([0_i32; 64]);

            unsafe {
                super::fjq_tl2_preprocessor(
                    BS,
                    M_FULL,
                    K,
                    0,
                    b_buf.0.as_mut_ptr() as *mut c_void,
                    lut_scales_buf.0.as_mut_ptr() as *mut c_void,
                    three_qlut_buf.0.as_mut_ptr() as *mut c_void,
                    two_qlut_buf.0.as_mut_ptr() as *mut c_void,
                );
                super::fjq_tl2_qgemm_lut(
                    BS,
                    M_FULL,
                    K,
                    256,
                    a_buf.0.as_mut_ptr() as *mut c_void,
                    sign_buf.0.as_mut_ptr() as *mut c_void,
                    three_qlut_buf.0.as_mut_ptr() as *mut c_void,
                    scales_buf.0.as_mut_ptr() as *mut c_void,
                    lut_scales_buf.0.as_mut_ptr() as *mut c_void,
                    c_buf.0.as_mut_ptr() as *mut c_void,
                );
            }
            c_buf.0
        };

        let baseline = run_with_mag(0, 0); // all-zero magnitude
        eprintln!("Baseline (all zeros): C[0..8] = {:?}", &baseline[0..8]);
        let nonzero_baseline: Vec<usize> = baseline
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != 0)
            .map(|(i, _)| i)
            .collect();
        eprintln!("Baseline non-zero rows: {:?}\n", nonzero_baseline);

        eprintln!(
            "Sweep 1 — vary magnitude byte offset bp in 0..32 (vec_as[0]):\n\
             For each bp, set a_buf[bp]=0xDD (idx=13 both nibbles), \
             signs=0, observe non-zero C rows."
        );
        for bp in 0..32_usize {
            let c = run_with_mag(bp, 0xDD);
            let diffs: Vec<(usize, i32)> = c
                .iter()
                .zip(baseline.iter())
                .enumerate()
                .filter(|(_, (cv, bv))| **cv != **bv)
                .map(|(i, (cv, bv))| (i, *cv - *bv))
                .collect();
            if diffs.is_empty() {
                eprintln!("  bp={bp:2}: NO C[r] changed (byte ignored?)");
            } else {
                let summary: Vec<String> = diffs
                    .iter()
                    .take(8)
                    .map(|(r, d)| format!("C[{r}]+={d}"))
                    .collect();
                let total = diffs.len();
                eprintln!(
                    "  bp={bp:2}: {total} rows changed: {} {}",
                    summary.join(", "),
                    if total > 8 { "..." } else { "" }
                );
            }
        }

        eprintln!(
            "\nSweep 2 — vary ai-block (byte 0 of each vec_as[ai] for \
             ai=0..21):\n\
             For each ai, set a_buf[ai*32]=0xDD, signs=0, observe \
             non-zero C rows. If kernel uses ai=0..19 only and drops \
             ai=20, ai=20 should produce no change.\n\
             Per kernel inner-loop bound (KK/8=5 outer × 4 inner = 20 \
             ai-vectors used), ai=20 should be the dropped one."
        );
        for ai in 0..21_usize {
            let bp = ai * 32;
            let c = run_with_mag(bp, 0xDD);
            let diffs: Vec<(usize, i32)> = c
                .iter()
                .zip(baseline.iter())
                .enumerate()
                .filter(|(_, (cv, bv))| **cv != **bv)
                .map(|(i, (cv, bv))| (i, *cv - *bv))
                .collect();
            if diffs.is_empty() {
                eprintln!("  ai={ai:2} (bp={bp:3}): NO C[r] changed");
            } else {
                let summary: Vec<String> = diffs
                    .iter()
                    .take(4)
                    .map(|(r, d)| format!("C[{r}]+={d}"))
                    .collect();
                let total = diffs.len();
                eprintln!(
                    "  ai={ai:2} (bp={bp:3}): {total} rows changed: {} {}",
                    summary.join(", "),
                    if total > 4 { "..." } else { "" }
                );
            }
        }
    }
}
