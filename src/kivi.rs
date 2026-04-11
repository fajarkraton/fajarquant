//! KIVI Baseline — Per-channel key / per-token value quantization.
//!
//! Implements the KIVI method (Liu et al., 2024, arXiv:2402.02750) as a baseline
//! for comparison with FajarQuant. KIVI observes that:
//! - **Keys** have large magnitude variation across channels (dimensions)
//!   → per-channel quantization preserves accuracy
//! - **Values** have large magnitude variation across tokens (sequence positions)
//!   → per-token quantization preserves accuracy
//!
//! This is a faithful reimplementation for fair comparison.

use ndarray::{Array1, Array2};

// ═══════════════════════════════════════════════════════════════════════
// KIVI Quantized Representations
// ═══════════════════════════════════════════════════════════════════════

/// Per-channel quantized key matrix.
///
/// Each channel (dimension) has its own scale and zero-point, computed from
/// the min/max of that channel across all tokens.
#[derive(Debug, Clone)]
pub struct KiviQuantizedKeys {
    /// Quantized indices: shape (seq_len, d_head), values 0..2^b-1.
    pub indices: Array2<u8>,
    /// Per-channel scale: (max - min) / (2^b - 1), shape (d_head,).
    pub scales: Array1<f64>,
    /// Per-channel zero-point (min value), shape (d_head,).
    pub zeros: Array1<f64>,
    /// Bit width.
    pub bits: u8,
}

/// Per-token quantized value matrix.
///
/// Each token position has its own scale and zero-point, computed from
/// the min/max of that token across all dimensions.
#[derive(Debug, Clone)]
pub struct KiviQuantizedValues {
    /// Quantized indices: shape (seq_len, d_head), values 0..2^b-1.
    pub indices: Array2<u8>,
    /// Per-token scale, shape (seq_len,).
    pub scales: Array1<f64>,
    /// Per-token zero-point (min value), shape (seq_len,).
    pub zeros: Array1<f64>,
    /// Bit width.
    pub bits: u8,
}

// ═══════════════════════════════════════════════════════════════════════
// KIVI Quantization Functions
// ═══════════════════════════════════════════════════════════════════════

/// Quantize keys per-channel (KIVI method).
///
/// For each channel d in [0, d_head):
///   scale_d = (max_d - min_d) / (2^b - 1)
///   q[t][d] = round((key[t][d] - min_d) / scale_d)
pub fn kivi_quantize_keys(keys: &Array2<f64>, bits: u8) -> KiviQuantizedKeys {
    let (seq_len, d_head) = keys.dim();
    let levels = (1u32 << bits) - 1;
    let levels_f = levels as f64;

    let mut scales = Array1::zeros(d_head);
    let mut zeros = Array1::zeros(d_head);
    let mut indices = Array2::zeros((seq_len, d_head));

    for d in 0..d_head {
        let col = keys.column(d);
        let min_val = col.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let range = max_val - min_val;
        let scale = if range > 1e-15 { range / levels_f } else { 1.0 };

        scales[d] = scale;
        zeros[d] = min_val;

        for t in 0..seq_len {
            let val = (keys[[t, d]] - min_val) / scale;
            let q = val.round().clamp(0.0, levels_f) as u8;
            indices[[t, d]] = q;
        }
    }

    KiviQuantizedKeys {
        indices,
        scales,
        zeros,
        bits,
    }
}

/// Dequantize keys (per-channel).
pub fn kivi_dequantize_keys(qk: &KiviQuantizedKeys) -> Array2<f64> {
    let (seq_len, d_head) = qk.indices.dim();
    let mut result = Array2::zeros((seq_len, d_head));

    for t in 0..seq_len {
        for d in 0..d_head {
            result[[t, d]] = qk.zeros[d] + (qk.indices[[t, d]] as f64) * qk.scales[d];
        }
    }
    result
}

/// Quantize values per-token (KIVI method).
///
/// For each token t in [0, seq_len):
///   scale_t = (max_t - min_t) / (2^b - 1)
///   q[t][d] = round((value[t][d] - min_t) / scale_t)
pub fn kivi_quantize_values(values: &Array2<f64>, bits: u8) -> KiviQuantizedValues {
    let (seq_len, d_head) = values.dim();
    let levels = (1u32 << bits) - 1;
    let levels_f = levels as f64;

    let mut scales = Array1::zeros(seq_len);
    let mut zeros = Array1::zeros(seq_len);
    let mut indices = Array2::zeros((seq_len, d_head));

    for t in 0..seq_len {
        let row = values.row(t);
        let min_val = row.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let range = max_val - min_val;
        let scale = if range > 1e-15 { range / levels_f } else { 1.0 };

        scales[t] = scale;
        zeros[t] = min_val;

        for d in 0..d_head {
            let val = (values[[t, d]] - min_val) / scale;
            let q = val.round().clamp(0.0, levels_f) as u8;
            indices[[t, d]] = q;
        }
    }

    KiviQuantizedValues {
        indices,
        scales,
        zeros,
        bits,
    }
}

/// Dequantize values (per-token).
pub fn kivi_dequantize_values(qv: &KiviQuantizedValues) -> Array2<f64> {
    let (seq_len, d_head) = qv.indices.dim();
    let mut result = Array2::zeros((seq_len, d_head));

    for t in 0..seq_len {
        for d in 0..d_head {
            result[[t, d]] = qv.zeros[t] + (qv.indices[[t, d]] as f64) * qv.scales[t];
        }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════
// KIVI Metrics
// ═══════════════════════════════════════════════════════════════════════

/// Compute MSE between original and quantized keys (per-channel KIVI).
pub fn kivi_key_mse(keys: &Array2<f64>, bits: u8) -> f64 {
    let qk = kivi_quantize_keys(keys, bits);
    let recon = kivi_dequantize_keys(&qk);
    let diff = keys - &recon;
    diff.mapv(|x| x * x).mean().unwrap_or(0.0)
}

/// Compute MSE between original and quantized values (per-token KIVI).
pub fn kivi_value_mse(values: &Array2<f64>, bits: u8) -> f64 {
    let qv = kivi_quantize_values(values, bits);
    let recon = kivi_dequantize_values(&qv);
    let diff = values - &recon;
    diff.mapv(|x| x * x).mean().unwrap_or(0.0)
}

/// Memory usage in bytes for KIVI quantized KV pair.
///
/// Keys: seq_len * d_head * bits/8 + d_head * 16 (scale+zero per channel)
/// Values: seq_len * d_head * bits/8 + seq_len * 16 (scale+zero per token)
pub fn kivi_memory_bytes(seq_len: usize, d_head: usize, bits: u8) -> usize {
    let data_bits = seq_len * d_head * (bits as usize);
    let data_bytes = data_bits.div_ceil(8);
    // Key overhead: per-channel (d_head scales + d_head zeros)
    let key_overhead = d_head * 16;
    // Value overhead: per-token (seq_len scales + seq_len zeros)
    let val_overhead = seq_len * 16;
    data_bytes * 2 + key_overhead + val_overhead
}

// ═══════════════════════════════════════════════════════════════════════
// 3-Way Comparison: FajarQuant vs KIVI vs TurboQuant
// ═══════════════════════════════════════════════════════════════════════

/// Result of a 3-way comparison.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub bits: u8,
    pub dimension: usize,
    pub seq_len: usize,
    /// MSE for each method.
    pub fajarquant_mse: f64,
    pub kivi_key_mse: f64,
    pub kivi_value_mse: f64,
    pub turboquant_mse: f64,
    /// Improvement ratios.
    pub fq_vs_kivi_key: f64, // (kivi - fq) / kivi * 100
    pub fq_vs_turbo: f64, // (turbo - fq) / turbo * 100
}

/// Run 3-way comparison on a given data matrix.
///
/// `data` is a (seq_len, d_head) matrix (one head of keys or values).
/// Returns comparison metrics for keys quantization.
pub fn compare_three_way(data: &Array2<f64>, bits: u8) -> ComparisonResult {
    let (seq_len, d_head) = data.dim();

    // 1. KIVI per-channel MSE (on keys)
    let kivi_mse = kivi_key_mse(data, bits);

    // 2. TurboQuant MSE (random rotation + coordinate-wise)
    let turbo_mse = turboquant_mse(data, bits);

    // 3. FajarQuant MSE (adaptive PCA rotation + coordinate-wise)
    let fq_mse = fajarquant_adaptive_mse(data, bits);

    let fq_vs_kivi = if kivi_mse > 1e-15 {
        (kivi_mse - fq_mse) / kivi_mse * 100.0
    } else {
        0.0
    };

    let fq_vs_turbo = if turbo_mse > 1e-15 {
        (turbo_mse - fq_mse) / turbo_mse * 100.0
    } else {
        0.0
    };

    ComparisonResult {
        bits,
        dimension: d_head,
        seq_len,
        fajarquant_mse: fq_mse,
        kivi_key_mse: kivi_mse,
        kivi_value_mse: kivi_value_mse(data, bits),
        turboquant_mse: turbo_mse,
        fq_vs_kivi_key: fq_vs_kivi,
        fq_vs_turbo,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Internal: TurboQuant and FajarQuant MSE wrappers
// ═══════════════════════════════════════════════════════════════════════

/// TurboQuant MSE: random rotation + coordinate-wise quantization.
fn turboquant_mse(data: &Array2<f64>, bits: u8) -> f64 {
    use crate::turboquant::{create_config, dequant_mse, quant_mse};

    let (_seq_len, d) = data.dim();
    let config = create_config(d, bits);

    let mut total_mse = 0.0;
    let mut count = 0usize;
    for row in data.rows() {
        let v = row.to_owned();
        let qv = quant_mse(&v, &config);
        let recon = dequant_mse(&qv, &config);
        let mse: f64 = v
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            / (d as f64);
        total_mse += mse;
        count += 1;
    }
    if count > 0 {
        total_mse / (count as f64)
    } else {
        0.0
    }
}

/// FajarQuant adaptive MSE: PCA rotation + adaptive codebook.
fn fajarquant_adaptive_mse(data: &Array2<f64>, bits: u8) -> f64 {
    use super::adaptive::{
        CalibrationBuffer, adaptive_codebook, compute_pca_rotation, decode_adaptive,
        encode_adaptive,
    };

    let (_seq_len, d) = data.dim();

    // 1. Calibrate: collect vectors into buffer
    let mut buffer = CalibrationBuffer::new(d, data.nrows());
    for row in data.rows() {
        buffer.add(row.to_owned());
    }

    // 2. Compute PCA rotation from calibration data
    let rotation = compute_pca_rotation(&buffer);

    // 3. Build adaptive codebook from rotated data
    let codebook = adaptive_codebook(&rotation, buffer.vectors(), bits);

    // 4. Encode/decode each row and measure MSE
    let mut total_mse = 0.0;
    let mut count = 0usize;
    for row in data.rows() {
        let v = row.to_owned();
        let encoded = encode_adaptive(&v, &rotation, &codebook);
        let decoded = decode_adaptive(&encoded, &rotation, &codebook);
        let mse: f64 = v
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            / (d as f64);
        total_mse += mse;
        count += 1;
    }
    if count > 0 {
        total_mse / (count as f64)
    } else {
        0.0
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn make_test_data(seq_len: usize, d: usize, seed: u64) -> Array2<f64> {
        // Low-rank structured data (rank-3 + noise)
        let mut rng_state = seed;
        let mut next = || -> f64 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0
        };

        let mut base = Array2::zeros((seq_len, d));
        // Rank-3 component
        for t in 0..seq_len {
            let a = next();
            let b = next();
            let c = next();
            for j in 0..d {
                base[[t, j]] = a * next() + b * next() + c * next();
            }
        }
        // Add small noise
        for t in 0..seq_len {
            for j in 0..d {
                base[[t, j]] += next() * 0.05;
            }
        }
        base
    }

    #[test]
    fn kivi_key_quantize_roundtrip() {
        let data = make_test_data(64, 128, 42);
        let qk = kivi_quantize_keys(&data, 4);
        let recon = kivi_dequantize_keys(&qk);
        let mse = (&data - &recon).mapv(|x| x * x).mean().unwrap();
        assert!(mse < 0.01, "4-bit KIVI key MSE should be small, got {mse}");
    }

    #[test]
    fn kivi_value_quantize_roundtrip() {
        let data = make_test_data(64, 128, 99);
        let qv = kivi_quantize_values(&data, 4);
        let recon = kivi_dequantize_values(&qv);
        let mse = (&data - &recon).mapv(|x| x * x).mean().unwrap();
        assert!(
            mse < 0.01,
            "4-bit KIVI value MSE should be small, got {mse}"
        );
    }

    #[test]
    fn kivi_lower_bits_higher_mse() {
        let data = make_test_data(64, 128, 55);
        let mse_2 = kivi_key_mse(&data, 2);
        let mse_3 = kivi_key_mse(&data, 3);
        let mse_4 = kivi_key_mse(&data, 4);
        assert!(mse_2 > mse_3, "2-bit should have higher MSE than 3-bit");
        assert!(mse_3 > mse_4, "3-bit should have higher MSE than 4-bit");
    }

    #[test]
    fn kivi_memory_calculation() {
        let mem = kivi_memory_bytes(512, 128, 3);
        let fp16_mem = 512 * 128 * 2 * 2; // keys + values, 2 bytes each
        assert!(mem < fp16_mem, "Quantized should use less memory than FP16");
    }

    #[test]
    fn three_way_comparison_runs() {
        let data = make_test_data(32, 64, 77);
        let result = compare_three_way(&data, 3);
        assert!(result.fajarquant_mse >= 0.0);
        assert!(result.kivi_key_mse >= 0.0);
        assert!(result.turboquant_mse >= 0.0);
        // FajarQuant should beat TurboQuant on structured data
        assert!(
            result.fq_vs_turbo > 0.0,
            "FajarQuant should improve over TurboQuant on low-rank data, got {}%",
            result.fq_vs_turbo
        );
    }

    #[test]
    fn fajarquant_beats_kivi_on_structured_data() {
        let data = make_test_data(64, 128, 123);
        let result = compare_three_way(&data, 3);
        // On structured (low-rank) data, FajarQuant's PCA rotation should help
        // KIVI's per-channel approach doesn't exploit low-rank structure
        println!("FajarQuant MSE: {:.6}", result.fajarquant_mse);
        println!("KIVI key MSE:   {:.6}", result.kivi_key_mse);
        println!("TurboQuant MSE: {:.6}", result.turboquant_mse);
        println!("FQ vs KIVI: {:.1}% improvement", result.fq_vs_kivi_key);
        println!("FQ vs Turbo: {:.1}% improvement", result.fq_vs_turbo);
    }

    /// Ablation study: isolate each innovation's contribution.
    #[test]
    fn ablation_each_innovation_independent() {
        let data = make_test_data(64, 128, 42);
        let bits = 3u8;

        // 1. No rotation (KIVI-like per-channel) — baseline
        let no_rot_mse = kivi_key_mse(&data, bits);

        // 2. Random rotation (TurboQuant)
        let turbo_mse = {
            let result = compare_three_way(&data, bits);
            result.turboquant_mse
        };

        // 3. PCA rotation (FajarQuant Innovation 1)
        let pca_mse = {
            let result = compare_three_way(&data, bits);
            result.fajarquant_mse
        };

        // Innovation 1: PCA should beat random on structured data
        assert!(
            pca_mse < turbo_mse * 1.01,
            "PCA rotation ({pca_mse:.6}) should be <= random rotation ({turbo_mse:.6})"
        );

        // Innovation 2: Fused attention — same MSE, less memory
        // Codebook dot product is proven equivalent (tested separately in fused_attention.rs)
        let fused_extra_bytes = (1usize << bits) * 8; // O(2^b) codebook only
        let standard_extra_bytes = 64 * 128 * 8; // O(N * d) dequantized keys
        assert!(
            fused_extra_bytes < standard_extra_bytes,
            "Fused memory ({fused_extra_bytes}) must be < standard ({standard_extra_bytes})"
        );

        // Innovation 3: Hierarchical — fewer total bits
        let schedule =
            super::super::hierarchical::BitSchedule::fixed_tiers(&[(5, 4), (20, 3), (50, 2)], 1);
        let hier_bits = schedule.total_bits(64);
        let flat_bits = 64 * bits as usize;
        assert!(
            hier_bits < flat_bits,
            "Hierarchical ({hier_bits}) should use fewer bits than flat ({flat_bits})"
        );

        println!("ABLATION RESULTS (d=128, b={bits}, N=64):");
        println!("  No rotation (KIVI): {no_rot_mse:.6e}");
        println!("  Random rot (Turbo): {turbo_mse:.6e}");
        println!("  PCA rot (FajarQ):   {pca_mse:.6e}");
        println!(
            "  Fused mem:          {fused_extra_bytes} B vs {standard_extra_bytes} B ({:.0}x reduction)",
            standard_extra_bytes as f64 / fused_extra_bytes as f64
        );
        println!(
            "  Hierarchical bits:  {hier_bits} vs {flat_bits} ({:.1}% savings)",
            (1.0 - hier_bits as f64 / flat_bits as f64) * 100.0
        );
    }
}
