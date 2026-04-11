//! FajarQuant Innovation 2: Fused Quantized Attention.
//!
//! Compute attention scores directly on quantized KV cache WITHOUT full
//! dequantization. The key insight: q^T * dequant(k) = sum_j q[j] * c[idx[j]]
//! — we never allocate the dequantized vector, saving O(N*d) memory.

use ndarray::Array1;
use std::collections::HashMap;

use crate::turboquant::{self, Codebook, QuantizedVector};

// ═══════════════════════════════════════════════════════════════════════
// Quantized KV Cache
// ═══════════════════════════════════════════════════════════════════════

/// A quantized key-value pair for a single token.
#[derive(Debug, Clone)]
pub struct QuantizedKV {
    /// Quantized key vector.
    pub k: QuantizedVector,
    /// Quantized value vector.
    pub v: QuantizedVector,
}

/// Quantized KV cache for a single (layer, head).
#[derive(Debug, Clone)]
pub struct QuantizedKVCache {
    /// Stored KV pairs in order of token position.
    entries: Vec<QuantizedKV>,
    /// Vector dimension.
    dim: usize,
    /// Bit width used for quantization.
    pub bit_width: u8,
}

impl QuantizedKVCache {
    /// Create a new empty cache.
    pub fn new(dim: usize, bit_width: u8) -> Self {
        Self {
            entries: Vec::new(),
            dim,
            bit_width,
        }
    }

    /// Append a new token's K and V (already quantized).
    pub fn append(&mut self, k: QuantizedVector, v: QuantizedVector) {
        self.entries.push(QuantizedKV { k, v });
    }

    /// Number of tokens in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Memory usage in bytes (quantized).
    pub fn quantized_bytes(&self) -> usize {
        // Each entry: dim indices for K + dim indices for V
        self.entries.len() * self.dim * 2
    }

    /// Full-precision equivalent memory in bytes.
    pub fn full_precision_bytes(&self) -> usize {
        // Each entry: dim * 8 bytes (f64) for K + dim * 8 for V
        self.entries.len() * self.dim * 2 * 8
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Fused Codebook Operations
// ═══════════════════════════════════════════════════════════════════════

/// Compute dot product between a query vector and a quantized key vector
/// WITHOUT dequantizing: score = sum_j q[j] * codebook.centroids[k_idx[j]].
pub fn codebook_dot_product(query: &Array1<f64>, k_indices: &[u8], codebook: &Codebook) -> f64 {
    query
        .iter()
        .zip(k_indices.iter())
        .map(|(&qj, &idx)| qj * codebook.centroids.get(idx as usize).copied().unwrap_or(0.0))
        .sum()
}

/// Compute weighted sum of quantized value vectors WITHOUT dequantizing:
/// output[j] = sum_i weights[i] * codebook.centroids[v_indices[i][j]].
pub fn codebook_weighted_sum(
    weights: &[f64],
    v_entries: &[&[u8]],
    codebook: &Codebook,
    dim: usize,
) -> Array1<f64> {
    let mut output = Array1::zeros(dim);
    for (i, &w) in weights.iter().enumerate() {
        if let Some(v_idx) = v_entries.get(i) {
            for j in 0..dim.min(v_idx.len()) {
                output[j] += w * codebook
                    .centroids
                    .get(v_idx[j] as usize)
                    .copied()
                    .unwrap_or(0.0);
            }
        }
    }
    output
}

/// Full fused quantized attention on a single query against a quantized KV cache.
///
/// The query should be in the **rotated** space (Pi * q) since keys are stored
/// as quantized rotated coordinates. The output is also in rotated space;
/// apply Pi^T to get back to the original space.
///
/// Steps:
/// 1. Compute attention scores: score[i] = q_rot * c[k_idx[i]] via codebook lookup
/// 2. Softmax: attn[i] = exp(score[i]) / sum(exp(scores))
/// 3. Weighted sum: output = sum(attn[i] * c[v_idx[i]]) via codebook lookup
///
/// Zero extra memory allocation for dequantized keys/values.
pub fn fused_quantized_attention(
    query_rotated: &Array1<f64>,
    cache: &QuantizedKVCache,
    codebook: &Codebook,
) -> Array1<f64> {
    if cache.is_empty() {
        return Array1::zeros(cache.dim);
    }

    // Step 1: Compute attention scores via codebook dot product
    let scores: Vec<f64> = cache
        .entries
        .iter()
        .map(|entry| codebook_dot_product(query_rotated, &entry.k.indices, codebook))
        .collect();

    // Step 2: Softmax (with numerical stability)
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp: f64 = exp_scores.iter().sum();
    let attn_weights: Vec<f64> = exp_scores.iter().map(|&e| e / sum_exp).collect();

    // Step 3: Weighted sum of values via codebook lookup
    let v_indices: Vec<&[u8]> = cache
        .entries
        .iter()
        .map(|e| e.v.indices.as_slice())
        .collect();
    codebook_weighted_sum(&attn_weights, &v_indices, codebook, cache.dim)
}

// ═══════════════════════════════════════════════════════════════════════
// KV Cache Manager (multi-layer, multi-head)
// ═══════════════════════════════════════════════════════════════════════

/// Manages quantized KV caches across all layers and heads.
#[derive(Debug, Clone)]
pub struct KVCacheManager {
    /// Caches indexed by (layer, head).
    caches: HashMap<(usize, usize), QuantizedKVCache>,
    /// Shared configuration.
    dim: usize,
    bit_width: u8,
}

impl KVCacheManager {
    /// Create a new cache manager.
    pub fn new(dim: usize, bit_width: u8) -> Self {
        Self {
            caches: HashMap::new(),
            dim,
            bit_width,
        }
    }

    /// Append a token's K and V to a specific (layer, head).
    pub fn append(&mut self, layer: usize, head: usize, k_vec: &Array1<f64>, v_vec: &Array1<f64>) {
        let config = turboquant::create_config(self.dim, self.bit_width);
        let k_quant = turboquant::quant_mse(k_vec, &config);
        let v_quant = turboquant::quant_mse(v_vec, &config);

        let cache = self
            .caches
            .entry((layer, head))
            .or_insert_with(|| QuantizedKVCache::new(self.dim, self.bit_width));
        cache.append(k_quant, v_quant);
    }

    /// Get the cache for a specific (layer, head).
    pub fn get(&self, layer: usize, head: usize) -> Option<&QuantizedKVCache> {
        self.caches.get(&(layer, head))
    }

    /// Total memory usage across all caches.
    pub fn total_quantized_bytes(&self) -> usize {
        self.caches.values().map(|c| c.quantized_bytes()).sum()
    }

    /// Total full-precision equivalent memory.
    pub fn total_full_precision_bytes(&self) -> usize {
        self.caches.values().map(|c| c.full_precision_bytes()).sum()
    }

    /// Compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        let fp = self.total_full_precision_bytes() as f64;
        let q = self.total_quantized_bytes() as f64;
        if q > 0.0 { fp / q } else { 0.0 }
    }

    /// Total number of tokens across all caches.
    pub fn total_tokens(&self) -> usize {
        self.caches.values().map(|c| c.len()).sum()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(dim: usize, bits: u8) -> turboquant::TurboQuantConfig {
        turboquant::create_config(dim, bits)
    }

    #[test]
    fn codebook_dot_matches_dequant_dot() {
        let dim = 8;
        let config = make_config(dim, 3);

        // Create a test vector and quantize it
        let mut rng = 42u64;
        let k: Array1<f64> =
            Array1::from_shape_fn(dim, |_| turboquant::lcg_next_f64(&mut rng) - 0.5);
        let q: Array1<f64> =
            Array1::from_shape_fn(dim, |_| turboquant::lcg_next_f64(&mut rng) - 0.5);

        let k_quant = turboquant::quant_mse(&k, &config);

        // Rotate the query (fused attention works in rotated space)
        let q_rotated = config.rotation.dot(&q);

        // Method 1: Fused (codebook lookup with rotated query)
        let fused_score = codebook_dot_product(&q_rotated, &k_quant.indices, &config.codebook);

        // Method 2: Dequant (includes inverse rotation) + dot with original query
        let k_dequant = turboquant::dequant_mse(&k_quant, &config);
        let standard_score: f64 = q.iter().zip(k_dequant.iter()).map(|(a, b)| a * b).sum();

        // Both compute q^T * Pi^T * c[idx] — same result, different order
        assert!(
            (fused_score - standard_score).abs() < 1e-10,
            "fused={fused_score}, standard={standard_score}"
        );
    }

    #[test]
    fn fused_attention_produces_valid_output() {
        let dim = 8;
        let bits = 2;
        let config = make_config(dim, bits);
        let mut cache = QuantizedKVCache::new(dim, bits);

        // Add some tokens
        let mut rng = 99u64;
        for _ in 0..5 {
            let k: Array1<f64> =
                Array1::from_shape_fn(dim, |_| turboquant::lcg_next_f64(&mut rng) - 0.5);
            let v: Array1<f64> =
                Array1::from_shape_fn(dim, |_| turboquant::lcg_next_f64(&mut rng) - 0.5);
            let k_q = turboquant::quant_mse(&k, &config);
            let v_q = turboquant::quant_mse(&v, &config);
            cache.append(k_q, v_q);
        }

        let query: Array1<f64> =
            Array1::from_shape_fn(dim, |_| turboquant::lcg_next_f64(&mut rng) - 0.5);
        let output = fused_quantized_attention(&query, &cache, &config.codebook);

        assert_eq!(output.len(), dim);
        assert!(
            output.iter().all(|v| v.is_finite()),
            "output should be finite"
        );
    }

    #[test]
    fn kv_cache_manager_tracks_memory() {
        let dim = 16;
        let bits = 3;
        let mut mgr = KVCacheManager::new(dim, bits);

        let mut rng = 77u64;
        for _ in 0..10 {
            let k = Array1::from_shape_fn(dim, |_| turboquant::lcg_next_f64(&mut rng) - 0.5);
            let v = Array1::from_shape_fn(dim, |_| turboquant::lcg_next_f64(&mut rng) - 0.5);
            mgr.append(0, 0, &k, &v);
        }

        assert_eq!(mgr.total_tokens(), 10);
        assert!(mgr.compression_ratio() > 1.0, "should compress");
        assert_eq!(mgr.total_quantized_bytes(), 10 * 16 * 2); // 10 tokens * 16 dim * 2 (k+v)
        assert_eq!(mgr.total_full_precision_bytes(), 10 * 16 * 2 * 8); // * 8 bytes per f64
    }

    #[test]
    fn empty_cache_attention_returns_zeros() {
        let dim = 8;
        let config = make_config(dim, 2);
        let cache = QuantizedKVCache::new(dim, 2);
        let query = Array1::zeros(dim);
        let output = fused_quantized_attention(&query, &cache, &config.codebook);
        assert!(output.iter().all(|&v| v == 0.0));
    }
}
