//! CPU-targeted vendored kernels.
//!
//! Currently houses one sub-module:
//!   - [`tl2`] — microsoft/BitNet TL2 x86 AVX2 ternary GEMV kernel,
//!     vendored under MIT (see `THIRD_PARTY_NOTICES.md`). Gated
//!     behind the `bitnet_tl2` feature; only available on
//!     `target_arch = "x86_64"`.
//!
//! Per V32-prep F.11 design v1.0. See
//! `docs/FJQ_PHASE_F_F11_BITNETCPP_TL2_PORT_DESIGN.md`.

/// Phase D scalar BitLinear baseline. Pure-integer ternary
/// signed-add over 2-bit packed V31 weights. Always available
/// (no feature gate, no platform restriction) — provides a
/// reference implementation for any future TL2/TL1/codegen path
/// to validate against. Ported from FajarOS Nova
/// `kernel/compute/matmulfree.fj` Op 3.
pub mod scalar_baseline;

/// V32-prep F.11.4 Branch X-real iteration 3 — Rust port of the
/// TL2 weight encoder (magnitude-side complete; sign packing in
/// iteration 4). Mirrors `python/phase_d/scripts/repack_to_tl2.py`
/// §7. Always available — needed by the iteration-4 parity test
/// regardless of `bitnet_tl2` feature gate.
pub mod tl2_encoder;

#[cfg(feature = "bitnet_tl2")]
pub mod tl2;
