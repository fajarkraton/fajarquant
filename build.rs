// fajarquant V32-prep F.11.1 — Build script that compiles the
// vendored microsoft/BitNet TL2 AVX2 kernel into a static library
// linked into fajarquant. Gated behind `--features bitnet_tl2` and
// `cfg(target_arch = "x86_64")`.
//
// Per F.11 design v1.0 §3.1 vendor-strategy choice (b): vendored C
// + cc crate. Avoids re-implementing TL2 in Rust intrinsics; keeps
// upstream parity automatic.

fn main() {
    println!("cargo:rerun-if-changed=cpu_kernels/bitnet_tl2/wrapper.cpp");
    println!("cargo:rerun-if-changed=cpu_kernels/bitnet_tl2/bitnet-lut-kernels-tl2.h");
    println!("cargo:rerun-if-changed=cpu_kernels/bitnet_tl2/ggml-bitnet-stub.h");

    // Only build the TL2 stub when the feature is on AND we're on
    // x86_64. On other architectures (arm64 CI runners, the
    // FajarOS Surya ARM target), this is a no-op so the rest of
    // the crate continues to build.
    if std::env::var("CARGO_FEATURE_BITNET_TL2").is_err() {
        return;
    }
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_arch != "x86_64" {
        println!(
            "cargo:warning=feature `bitnet_tl2` enabled but target_arch={target_arch} != x86_64; \
             skipping TL2 build (use a stub-only path or rebuild on x86_64)"
        );
        return;
    }

    cc::Build::new()
        .cpp(true)
        .file("cpu_kernels/bitnet_tl2/wrapper.cpp")
        .include("cpu_kernels/bitnet_tl2")
        .define("GGML_BITNET_X86_TL2", None)
        // -mavx2 is required by upstream's `__AVX2__` ifdef
        // gating; -O3 matches the published bitnet.cpp build.
        .flag_if_supported("-mavx2")
        .flag_if_supported("-O3")
        // Upstream uses C++17 features (designated initializers in
        // structs, `nullptr`); set the standard explicitly so older
        // toolchains don't fall back to C++14.
        .flag_if_supported("-std=c++17")
        // Suppress noisy warnings from vendored upstream code; we
        // don't own those sources, and turning warnings into errors
        // would make tracking upstream brittle.
        .flag_if_supported("-Wno-unused-function")
        .flag_if_supported("-Wno-unused-variable")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-sign-compare")
        .flag_if_supported("-Wno-unused-but-set-variable")
        .compile("fajarquant_bitnet_tl2");
}
