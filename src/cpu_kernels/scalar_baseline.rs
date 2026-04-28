//! V32-prep F.11.3 — Phase D scalar BitLinear baseline ported to
//! Rust.
//!
//! This is a faithful port of FajarOS Nova's
//! `kernel/compute/matmulfree.fj` Op 3 `km_mf_bitlinear_packed`:
//! pure-integer signed-add over 2-bit packed ternary weights
//! against an i8-quantized activation vector. Produces an i64
//! accumulator per output channel.
//!
//! **Why ported here.** The TL2 LUT path (vendored under
//! `cpu_kernels/bitnet_tl2/`) must produce bit-exact-equal output
//! to this scalar baseline. F.11.3 needs the scalar side as a
//! callable Rust function so a future TL2-side parity test
//! (F.11.4 prerequisite) can compare them in-process.
//!
//! **Encoding contract.** Per
//! `docs/FJQ_PHASE_D_FJM_V9_SPEC.md` §5: V31 Phase D ternary
//! weights are packed at 2 bits per element, 4 elements per byte
//! (little-endian within byte), encoded as `00 = -1, 01 = 0,
//! 10 = +1, 11 = reserved (treat as 0)`. Weights are stored
//! row-major: `(out_features, in_features)` with `in_features`
//! the inner stride. Identical to the Python decoder in
//! `python/phase_d/scripts/repack_to_tl2.py` §1.
//!
//! **Activation contract.** Activations are pre-quantized to i8
//! by upstream `km_mf_absmax_quant_i8`. Per FajarOS reference,
//! the scale is `gamma_x = 127 / max(|x|)`; the quantized values
//! are `q[i] = round(x[i] * gamma_x).clamp(-127, 127)`. This
//! function does NOT do the quantization — it consumes the
//! already-i8 stream so callers can share a single quantized
//! buffer between the scalar and TL2 paths during parity checks.

/// Decode a single packed byte into 4 ternary weights `{-1, 0, +1}`.
/// Mirrors the V31 wire format precisely: bits 1-0 = w[0], bits 3-2
/// = w[1], bits 5-4 = w[2], bits 7-6 = w[3]. `pub` so external
/// consumers / future shape-generic Rust paths can use the canonical
/// V31 unpacker without re-deriving it.
#[inline]
pub fn decode_ternary_byte(b: u8) -> [i8; 4] {
    let map = |code: u8| -> i8 {
        match code & 0b11 {
            0b00 => -1,
            0b01 => 0,
            0b10 => 1,
            _ => 0, // 0b11 reserved per spec
        }
    };
    [map(b), map(b >> 2), map(b >> 4), map(b >> 6)]
}

/// Compute the per-output-channel integer accumulator for one
/// BitLinear matmul: `y[j] = Σ_i W[j, i] * x[i]` with ternary `W`
/// (V31-packed) and i8 `x`.
///
/// Output shape: `[out_features]`, dtype `i64` (room for the
/// worst-case sum 127 × in_features without overflow up to
/// in_features ≈ 7 × 10^16).
///
/// # Arguments
///
/// - `weights_packed`: V31 2-bit packed ternary, shape `(out, in)`
///   row-major, length `ceil(out * in / 4)` bytes.
/// - `activations_i8`: pre-quantized activations, length `in_features`.
/// - `out_features`, `in_features`: matrix dimensions.
///
/// # Errors
///
/// Returns `None` if the packed buffer is shorter than required.
pub fn bitlinear_packed_scalar(
    weights_packed: &[u8],
    activations_i8: &[i8],
    out_features: usize,
    in_features: usize,
) -> Option<Vec<i64>> {
    if activations_i8.len() != in_features {
        return None;
    }
    let total_weights = out_features * in_features;
    let required_bytes = total_weights.div_ceil(4);
    if weights_packed.len() < required_bytes {
        return None;
    }

    let mut out = vec![0_i64; out_features];
    for (j, out_slot) in out.iter_mut().enumerate() {
        let mut accum: i64 = 0;
        let row_start = j * in_features;
        for (i, &x) in activations_i8.iter().enumerate() {
            let weight_idx = row_start + i;
            let byte = weights_packed[weight_idx / 4];
            let lane = weight_idx % 4;
            let code = (byte >> (lane * 2)) & 0b11;
            let w: i8 = match code {
                0b00 => -1,
                0b01 => 0,
                0b10 => 1,
                _ => 0,
            };
            // Cast to i64 BEFORE multiplication — accumulate in i64
            // to match upstream's y_addr: i64×x1000 contract.
            accum += (w as i64) * (x as i64);
        }
        *out_slot = accum;
    }
    Some(out)
}

/// Quantize float activations to i8 using the absmax convention
/// shared by Phase D (`km_mf_absmax_quant_i8`) and TL2
/// (`per_tensor_quant`):
///
/// ```text
/// gamma_x = 127.0 / max(|x[i]|)
/// q[i]    = round(x[i] * gamma_x).clamp(-127, 127)
/// ```
///
/// Returns `(quantized, gamma_x)` where `gamma_x` is the scale used
/// (1.0 if all activations are zero, to avoid div-by-zero).
pub fn absmax_quantize_i8(activations: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = activations.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    let gamma_x = if max_abs > 0.0 { 127.0 / max_abs } else { 1.0 };
    let q: Vec<i8> = activations
        .iter()
        .map(|x| {
            let scaled = x * gamma_x;
            let rounded = scaled.round() as i32;
            rounded.clamp(-127, 127) as i8
        })
        .collect();
    (q, gamma_x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_v31(weights: &[i8]) -> Vec<u8> {
        let mut out = vec![0u8; weights.len().div_ceil(4)];
        for (i, &w) in weights.iter().enumerate() {
            let code: u8 = match w {
                -1 => 0b00,
                0 => 0b01,
                1 => 0b10,
                _ => panic!("non-ternary weight: {w}"),
            };
            out[i / 4] |= code << ((i % 4) * 2);
        }
        out
    }

    #[test]
    fn decode_ternary_byte_known_pattern() {
        // V31 spec example: byte = 0b01_10_01_00 = 0x64 → (-1, 0, 1, 0)
        let result = decode_ternary_byte(0b01_10_01_00);
        assert_eq!(result, [-1, 0, 1, 0]);
    }

    #[test]
    fn bitlinear_identity_weight_returns_input() {
        // 1x4 identity-ish: W = [1, 0, 0, 0] → y[0] = x[0]
        let w = [1_i8, 0, 0, 0];
        let x = [42_i8, 17, -3, 99];
        let packed = pack_v31(&w);
        let result = bitlinear_packed_scalar(&packed, &x, 1, 4).unwrap();
        assert_eq!(result, vec![42_i64]);
    }

    #[test]
    fn bitlinear_all_negative_one_negates_sum() {
        // 1x4: W = [-1, -1, -1, -1] → y[0] = -(x[0] + x[1] + x[2] + x[3])
        let w = [-1_i8, -1, -1, -1];
        let x = [10_i8, 20, 30, 40];
        let packed = pack_v31(&w);
        let result = bitlinear_packed_scalar(&packed, &x, 1, 4).unwrap();
        assert_eq!(result, vec![-100_i64]);
    }

    #[test]
    fn bitlinear_2x4_distinct_outputs() {
        // 2 outputs, 4 inputs:
        // W[0] = [1, 0, 1, 0], y[0] = x[0] + x[2]
        // W[1] = [0, -1, 0, 1], y[1] = -x[1] + x[3]
        let w = [1_i8, 0, 1, 0, 0, -1, 0, 1];
        let x = [5_i8, 10, 7, 13];
        let packed = pack_v31(&w);
        let result = bitlinear_packed_scalar(&packed, &x, 2, 4).unwrap();
        assert_eq!(result, vec![12_i64, 3_i64]);
    }

    #[test]
    fn bitlinear_rejects_short_weights_buffer() {
        let x = [0_i8; 16];
        // Need ceil(16 / 4) = 4 bytes for 1 output of 16 inputs;
        // only pass 1 byte
        let result = bitlinear_packed_scalar(&[0_u8], &x, 1, 16);
        assert!(result.is_none());
    }

    #[test]
    fn bitlinear_rejects_activation_size_mismatch() {
        let packed = vec![0u8; 4];
        let x = [0_i8; 8]; // wrong length: 8 != in_features=16
        let result = bitlinear_packed_scalar(&packed, &x, 1, 16);
        assert!(result.is_none());
    }

    #[test]
    fn absmax_quantize_known_input() {
        // x = [-8.0, 0.0, 8.0, 4.0] → max_abs = 8 → gamma_x = 127/8 = 15.875
        // q = round(x * 15.875) = [-127, 0, 127, 64]  (4 * 15.875 = 63.5, rounds to 64)
        let x = vec![-8.0_f32, 0.0, 8.0, 4.0];
        let (q, gamma_x) = absmax_quantize_i8(&x);
        assert_eq!(gamma_x, 127.0 / 8.0);
        assert_eq!(q, vec![-127_i8, 0, 127, 64]);
    }

    #[test]
    fn absmax_quantize_all_zero_safe() {
        // All-zero activations should not div-by-zero
        let x = vec![0.0_f32; 8];
        let (q, gamma_x) = absmax_quantize_i8(&x);
        assert_eq!(gamma_x, 1.0);
        assert_eq!(q, vec![0_i8; 8]);
    }

    #[test]
    fn absmax_quantize_clamps_at_minus_127() {
        // Construct input where rounding would land at -128 without
        // clamping: x[i] = -max_abs and rounding nudges. With
        // gamma_x = 127/max_abs, exact value is exactly -127, so no
        // clamping is strictly required; but verify no off-by-one.
        let x = vec![-1.0_f32, 1.0];
        let (q, _) = absmax_quantize_i8(&x);
        assert_eq!(q, vec![-127_i8, 127]);
    }

    /// End-to-end: f32 activations → quantize → BitLinear scalar →
    /// integer accumulator. Verifies the FULL Phase D scalar
    /// pipeline produces a deterministic, hand-computable result.
    #[test]
    fn end_to_end_known_input() {
        // 1×3 BitLinear: W = [+1, -1, 0], so y_int = x_q[0] - x_q[1]
        let w = [1_i8, -1, 0];
        // Activations chosen so quantization is exact: max_abs = 4,
        // gamma_x = 127/4 = 31.75
        // x = [4.0, -2.0, 3.0]
        // q[0] = round(4 * 31.75) = round(127) = 127
        // q[1] = round(-2 * 31.75) = round(-63.5) = clamps to nearest:
        //        Rust f32::round() rounds half-to-even? Actually
        //        f32::round documents "Returns the nearest integer
        //        to self. If a value is half-way between two
        //        integers, round away from 0.0." So -63.5 → -64.
        // q[2] = round(3 * 31.75) = round(95.25) = 95
        // y[0] = 127 - (-64) + 0 = 191
        let x = vec![4.0_f32, -2.0, 3.0];
        let (q, _gamma_x) = absmax_quantize_i8(&x);
        assert_eq!(q, vec![127_i8, -64, 95]);

        let packed = pack_v31(&w);
        let result = bitlinear_packed_scalar(&packed, &q, 1, 3).unwrap();
        assert_eq!(result, vec![191_i64]);
    }
}
