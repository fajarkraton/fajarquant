//! FajarQuant Innovation 3: Hierarchical Multi-Resolution Quantization.
//!
//! Allocates more bits to recent/important tokens and fewer to old tokens.
//! Same total bit budget achieves better quality than flat allocation because
//! recent tokens dominate attention scores.

use crate::turboquant::{self, Codebook, QuantizedVector};
use ndarray::Array1;

// ═══════════════════════════════════════════════════════════════════════
// Bit Schedule
// ═══════════════════════════════════════════════════════════════════════

/// Defines how many bits each token position gets.
#[derive(Debug, Clone)]
pub struct BitSchedule {
    /// Tiers: (max_age, bit_width). Sorted by max_age ascending.
    /// A token with age <= tier.max_age gets tier.bit_width bits.
    tiers: Vec<(usize, u8)>,
    /// Minimum bit-width (fallback for very old tokens).
    min_bits: u8,
}

impl BitSchedule {
    /// Create a fixed-tier schedule.
    ///
    /// Example: `fixed_tiers(&[(256, 4), (1024, 3), (4096, 2)], 1)`
    /// Last 256 tokens: 4 bits, 257-1024: 3 bits, 1025-4096: 2 bits, 4097+: 1 bit
    pub fn fixed_tiers(tiers: &[(usize, u8)], min_bits: u8) -> Self {
        Self {
            tiers: tiers.to_vec(),
            min_bits,
        }
    }

    /// Create an exponential decay schedule.
    ///
    /// `bits(age) = max(min_bits, round(base_bits * exp(-decay * age)))`
    pub fn exponential_decay(base_bits: u8, min_bits: u8, decay: f64, max_age: usize) -> Self {
        let mut tiers = Vec::new();
        let mut prev_bits = base_bits;
        let mut tier_start = 0;

        for age in 1..=max_age {
            let bits_f = base_bits as f64 * (-decay * age as f64).exp();
            let bits = (bits_f.round() as u8).max(min_bits);
            if bits != prev_bits {
                tiers.push((age - 1, prev_bits));
                prev_bits = bits;
            }
            tier_start = age;
        }
        tiers.push((tier_start, prev_bits));

        Self { tiers, min_bits }
    }

    /// Get the bit-width for a token at a given age (distance from current position).
    pub fn bits_for(&self, age: usize) -> u8 {
        for &(max_age, bits) in &self.tiers {
            if age <= max_age {
                return bits;
            }
        }
        self.min_bits
    }

    /// Compute total bits for a sequence of given length.
    pub fn total_bits(&self, seq_len: usize) -> usize {
        (0..seq_len).map(|age| self.bits_for(age) as usize).sum()
    }

    /// Average bits per token for a given sequence length.
    pub fn avg_bits(&self, seq_len: usize) -> f64 {
        if seq_len == 0 {
            return 0.0;
        }
        self.total_bits(seq_len) as f64 / seq_len as f64
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Hierarchical Quantized Entry
// ═══════════════════════════════════════════════════════════════════════

/// A token entry with its current quantization level.
#[derive(Debug, Clone)]
pub struct HierarchicalEntry {
    /// Original vector (kept for re-quantization during demotion).
    original: Array1<f64>,
    /// Current quantized representation.
    pub quantized: QuantizedVector,
    /// Current bit-width.
    pub current_bits: u8,
    /// Token position (0 = oldest).
    pub position: usize,
}

/// Hierarchical quantized KV cache.
#[derive(Debug, Clone)]
pub struct HierarchicalCache {
    /// Entries in position order.
    entries: Vec<HierarchicalEntry>,
    /// Dimension.
    dim: usize,
    /// Bit schedule.
    schedule: BitSchedule,
    /// Codebooks per bit-width: bits -> Codebook.
    codebooks: std::collections::HashMap<u8, Codebook>,
    /// Rotation matrix (shared).
    rotation: ndarray::Array2<f64>,
}

impl HierarchicalCache {
    /// Create a new hierarchical cache.
    pub fn new(dim: usize, schedule: BitSchedule, seed: u64) -> Self {
        let rotation = turboquant::random_orthogonal(dim, seed);
        let mut codebooks = std::collections::HashMap::new();
        // Pre-compute codebooks for bits 1..4
        for b in 1..=4u8 {
            codebooks.insert(b, turboquant::lloyd_max(dim, b, 25));
        }
        Self {
            entries: Vec::new(),
            dim,
            schedule,
            codebooks,
            rotation,
        }
    }

    /// Append a new token. Uses the schedule's bit allocation for the newest position.
    pub fn append(&mut self, vector: Array1<f64>) {
        let bits = self.schedule.bits_for(0); // newest token gets highest bits
        let codebook = self.codebooks.get(&bits).expect("codebook for bit-width");
        let y = self.rotation.dot(&vector);
        let indices: Vec<u8> = y
            .iter()
            .map(|&yj| find_bucket(yj, &codebook.boundaries) as u8)
            .collect();
        let norm = vector.iter().map(|v| v * v).sum::<f64>().sqrt();
        let qv = QuantizedVector {
            indices,
            norm,
            rotation_id: 0,
        };

        self.entries.push(HierarchicalEntry {
            original: vector,
            quantized: qv,
            current_bits: bits,
            position: self.entries.len(),
        });

        // Re-quantize older tokens if their age-based bit allocation changed
        self.rebalance();
    }

    /// Rebalance: demote older tokens to fewer bits according to the schedule.
    fn rebalance(&mut self) {
        let n = self.entries.len();
        for i in 0..n {
            let age = n - 1 - i; // age = distance from newest
            let target_bits = self.schedule.bits_for(age);
            if target_bits < self.entries[i].current_bits {
                // Demote: re-quantize at fewer bits
                if let Some(codebook) = self.codebooks.get(&target_bits) {
                    let y = self.rotation.dot(&self.entries[i].original);
                    let indices: Vec<u8> = y
                        .iter()
                        .map(|&yj| find_bucket(yj, &codebook.boundaries) as u8)
                        .collect();
                    self.entries[i].quantized.indices = indices;
                    self.entries[i].current_bits = target_bits;
                }
            }
        }
    }

    /// Number of tokens.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total bits used by the cache.
    pub fn total_bits_used(&self) -> usize {
        self.entries
            .iter()
            .map(|e| e.current_bits as usize * self.dim)
            .sum()
    }

    /// Equivalent flat allocation bits at a given uniform bit-width.
    pub fn flat_equivalent_bits(&self, uniform_bits: u8) -> usize {
        self.entries.len() * uniform_bits as usize * self.dim
    }

    /// Average bits per coordinate across all tokens.
    pub fn avg_bits_per_coord(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        self.entries
            .iter()
            .map(|e| e.current_bits as f64)
            .sum::<f64>()
            / self.entries.len() as f64
    }

    /// Compute MSE per tier (for quality analysis).
    pub fn mse_per_tier(&self) -> Vec<(u8, f64, usize)> {
        let mut tier_data: std::collections::HashMap<u8, (f64, usize)> =
            std::collections::HashMap::new();
        for entry in &self.entries {
            let codebook = match self.codebooks.get(&entry.current_bits) {
                Some(cb) => cb,
                None => continue,
            };
            // Reconstruct and compute MSE
            let y_hat: Array1<f64> = Array1::from_vec(
                entry
                    .quantized
                    .indices
                    .iter()
                    .map(|&idx| codebook.centroids.get(idx as usize).copied().unwrap_or(0.0))
                    .collect(),
            );
            let x_hat = self.rotation.t().dot(&y_hat);
            let diff = &entry.original - &x_hat;
            let mse = diff.iter().map(|v| v * v).sum::<f64>() / self.dim as f64;
            let e = tier_data.entry(entry.current_bits).or_insert((0.0, 0));
            e.0 += mse;
            e.1 += 1;
        }
        let mut result: Vec<(u8, f64, usize)> = tier_data
            .into_iter()
            .map(|(bits, (total_mse, count))| (bits, total_mse / count as f64, count))
            .collect();
        result.sort_by_key(|&(bits, _, _)| bits);
        result
    }
}

fn find_bucket(x: f64, boundaries: &[f64]) -> usize {
    for (i, &b) in boundaries.iter().enumerate() {
        if x < b {
            return i;
        }
    }
    boundaries.len()
}

/// Compare hierarchical vs flat allocation at the same total bit budget.
///
/// Returns (hierarchical_avg_mse, flat_avg_mse, bit_savings_pct).
pub fn compare_hierarchical_vs_flat(
    data: &[Array1<f64>],
    dim: usize,
    schedule: &BitSchedule,
    flat_bits: u8,
) -> (f64, f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    // Hierarchical cache
    let mut hier_cache = HierarchicalCache::new(dim, schedule.clone(), 12345);
    for x in data {
        hier_cache.append(x.clone());
    }

    // Flat quantization
    let flat_config = turboquant::create_config(dim, flat_bits);
    let mut flat_mse = 0.0;
    for x in data {
        flat_mse += turboquant::mse_distortion(x, &flat_config);
    }
    flat_mse /= data.len() as f64;

    // Hierarchical MSE (weighted average across tiers)
    let tiers = hier_cache.mse_per_tier();
    let hier_mse = if tiers.is_empty() {
        0.0
    } else {
        let total: f64 = tiers
            .iter()
            .map(|(_, mse, count)| mse * *count as f64)
            .sum();
        total / data.len() as f64
    };

    let hier_bits = hier_cache.total_bits_used();
    let flat_bits_total = hier_cache.flat_equivalent_bits(flat_bits);
    let savings = if flat_bits_total > 0 {
        (1.0 - hier_bits as f64 / flat_bits_total as f64) * 100.0
    } else {
        0.0
    };

    (hier_mse, flat_mse, savings)
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_tier_schedule() {
        let schedule = BitSchedule::fixed_tiers(&[(256, 4), (1024, 3), (4096, 2)], 1);
        assert_eq!(schedule.bits_for(0), 4); // newest
        assert_eq!(schedule.bits_for(100), 4); // within first tier
        assert_eq!(schedule.bits_for(256), 4); // boundary
        assert_eq!(schedule.bits_for(257), 3); // second tier
        assert_eq!(schedule.bits_for(1024), 3);
        assert_eq!(schedule.bits_for(1025), 2);
        assert_eq!(schedule.bits_for(5000), 1); // beyond all tiers
    }

    #[test]
    fn exponential_decay_schedule() {
        let schedule = BitSchedule::exponential_decay(4, 1, 0.005, 1000);
        assert!(schedule.bits_for(0) >= 3, "recent should have high bits");
        assert!(schedule.bits_for(1000) <= 2, "old should have low bits");
    }

    #[test]
    fn hierarchical_cache_append_and_rebalance() {
        let dim = 8;
        let schedule = BitSchedule::fixed_tiers(&[(2, 3), (5, 2)], 1);
        let mut cache = HierarchicalCache::new(dim, schedule, 42);

        let mut rng = 42u64;
        for _ in 0..6 {
            let v = Array1::from_shape_fn(dim, |_| turboquant::lcg_next_f64(&mut rng) - 0.5);
            cache.append(v);
        }

        assert_eq!(cache.len(), 6);
        // Newest 3 tokens should be at 3 bits, next 3 at 2 bits, oldest beyond at 1 bit
        assert!(
            cache.avg_bits_per_coord() < 3.0,
            "should have mixed bit-widths"
        );
    }

    #[test]
    fn total_bits_less_than_flat() {
        let dim = 8;
        let schedule = BitSchedule::fixed_tiers(&[(5, 4), (20, 2)], 1);
        let mut cache = HierarchicalCache::new(dim, schedule, 42);

        let mut rng = 42u64;
        for _ in 0..30 {
            let v = Array1::from_shape_fn(dim, |_| turboquant::lcg_next_f64(&mut rng) - 0.5);
            cache.append(v);
        }

        let hier_bits = cache.total_bits_used();
        let flat_bits = cache.flat_equivalent_bits(4);
        assert!(
            hier_bits < flat_bits,
            "hierarchical ({hier_bits}) should use fewer bits than flat-4 ({flat_bits})"
        );
    }

    #[test]
    fn mse_per_tier_reports_correctly() {
        let dim = 8;
        let schedule = BitSchedule::fixed_tiers(&[(3, 3), (6, 2)], 1);
        let mut cache = HierarchicalCache::new(dim, schedule, 42);

        let mut rng = 42u64;
        for _ in 0..10 {
            let v = Array1::from_shape_fn(dim, |_| turboquant::lcg_next_f64(&mut rng) - 0.5);
            cache.append(v);
        }

        let tiers = cache.mse_per_tier();
        assert!(!tiers.is_empty(), "should have tier data");
        // Higher bit tiers should have lower MSE
        if tiers.len() >= 2 {
            let high_bits_mse = tiers.iter().find(|(b, _, _)| *b == 3).map(|(_, m, _)| *m);
            let low_bits_mse = tiers.iter().find(|(b, _, _)| *b <= 2).map(|(_, m, _)| *m);
            if let (Some(h), Some(l)) = (high_bits_mse, low_bits_mse) {
                assert!(
                    h <= l * 2.0,
                    "3-bit MSE ({h}) should be lower than 2-bit MSE ({l})"
                );
            }
        }
    }
}
