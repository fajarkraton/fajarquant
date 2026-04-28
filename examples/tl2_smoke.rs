//! V32-prep F.11.1 — TL2 vendor smoke binary.
//!
//! Build + run gate for the vendored microsoft/BitNet TL2 AVX2 kernel.
//! Per F.11 design v1.0 §6.2: this example MUST exit 0 with non-trivial
//! output before F.11.2 (Rust shim with parity tests) begins.
//!
//! Run: `cargo run --release --example tl2_smoke --features bitnet_tl2`
//!
//! What it verifies:
//!   1. The vendored C++ TL2 source compiles under `cc` + `-mavx2 -O3`
//!      with the stub `ggml-bitnet-stub.h` standing in for upstream
//!      GGML.
//!   2. The compiled object is link-reachable from Rust via the
//!      `extern "C"` thunks in `wrapper.cpp` (`fjq_tl2_*`).
//!   3. The AVX2 inner loop in `per_tensor_quant` actually executes
//!      and returns a finite, non-zero quantization scale on synthetic
//!      input.
//!
//! What it does NOT verify (deferred to F.11.3):
//!   - Bit-exact parity vs the V31 Phase D scalar baseline.
//!   - Real Mini-shape (768×1024 etc.) end-to-end matmul.
//!   - The `fjq_tl2_qgemm_lut` entry point (preset header is shape-locked
//!     to 3200×8640 / 3200×3200 / 8640×3200; calling it with non-matched
//!     shapes is undefined and out of scope for F.11.1).

use std::process::ExitCode;

use fajarquant::cpu_kernels::tl2;

fn main() -> ExitCode {
    let n: i32 = 256;
    // F.11.2: safe wrapper validates the contract (n > 0, n ≤ 4096,
    // n % 8 == 0) up-front and returns Result instead of NaN-on-fail.
    let scale = match tl2::self_test(n) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("FAIL: self_test rejected n={n}: {e:?}");
            return ExitCode::from(4);
        }
    };

    println!("F.11.1 TL2 smoke");
    println!(
        "  upstream: microsoft/BitNet preset_kernels/bitnet_b1_58-3B/bitnet-lut-kernels-tl2.h"
    );
    println!("  vendored: cpu_kernels/bitnet_tl2/ (MIT, see MIT_LICENSE_BITNET)");
    println!("  call:     fjq_tl2_self_test(n_floats={n}) → per_tensor_quant");
    println!("  result:   lut_scales = {scale:.6}");

    if !scale.is_finite() {
        eprintln!("FAIL: scale not finite ({scale}); AVX2 path likely not reached");
        return ExitCode::from(1);
    }
    if scale <= 0.0 {
        eprintln!(
            "FAIL: scale <= 0 ({scale}); per_tensor_quant should produce positive scale on triangle-wave input"
        );
        return ExitCode::from(2);
    }

    // Sanity check: per_tensor_quant computes 127 / max(|b|). With
    // synthetic triangle-wave input b[i] = (i % 17) - 8, max(|b|) is
    // 8 over the first 16 samples and stays 8. So the expected scale
    // is 127 / 8 = 15.875.
    let expected = 127.0_f32 / 8.0;
    let rel_err = (scale - expected).abs() / expected;
    println!("  expected: {expected:.6} (= 127 / max|b|)");
    println!("  rel_err:  {rel_err:.4e}");
    if rel_err > 1e-3 {
        eprintln!("FAIL: rel_err {rel_err:.4e} > 1e-3; AVX2 quant path produced wrong value");
        return ExitCode::from(3);
    }

    println!("F.11.1 TL2 smoke PASSED");
    ExitCode::SUCCESS
}
