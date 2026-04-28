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
//! **Shape-locking caveat.** The vendored preset header is
//! shape-locked to `bitnet_b1_58-3B` ({3200×8640, 3200×3200,
//! 8640×3200}). `fjq_tl2_qgemm_lut` ONLY produces meaningful
//! output for those `(m, k)` pairs; calling it with anything else
//! is undefined. F.11.3 will either generate a Mini-shape preset
//! via `utils/codegen_tl2.py` or implement a shape-generic Rust
//! fallback. For F.11.2 we expose `qgemm_lut` with the shape
//! constraint enforced as a returned error, NOT silently.

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

/// Vendored preset shapes (shape-locked to bitnet_b1_58-3B per §1).
/// Any (m, k) pair not in this set returns
/// `ShimError::UnsupportedShape` from `qgemm_lut`.
pub const TL2_SUPPORTED_SHAPES: &[(i32, i32)] = &[(3200, 8640), (3200, 3200), (8640, 3200)];

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

        // Demonstrate the safe wrapper's ShimError surface using a
        // shape-violation signal (which doesn't require constructing
        // valid buffers); the alignment branch is symmetric.
        let bogus_ptr = 1usize as *mut c_void; // unaligned by construction
        assert!(!is_tl2_aligned(bogus_ptr));
        let result = unsafe {
            qgemm_lut(
                1, 3200, 8640, 96, bogus_ptr, bogus_ptr, bogus_ptr, bogus_ptr, bogus_ptr, bogus_ptr,
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

        // (768, 1024) is the Mini gate_proj-ish shape; not in the
        // 3B preset's supported set.
        let result = unsafe { qgemm_lut(1, 768, 1024, 96, p, p, p, p, p, p) };
        assert!(
            matches!(result, Err(ShimError::UnsupportedShape { m: 768, k: 1024 })),
            "Mini-shape (768, 1024) should be rejected, got {result:?}"
        );
    }
}
