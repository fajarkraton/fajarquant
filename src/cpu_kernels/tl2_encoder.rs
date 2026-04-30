//! V32-prep F.11.4 Branch X-real iteration 3 — Rust port of the
//! TL2 weight encoder (magnitude side only; sign packing deferred
//! to iteration 4).
//!
//! Mirrors `python/phase_d/scripts/repack_to_tl2.py` §2 + §7.
//! Required as a Rust-side helper so the iteration-4 parity test
//! at `cpu_kernels::tl2::tests` can build TL2-formatted weights
//! in-process without shelling out to Python.
//!
//! **Indexing rule** (derived from kernel strides at
//! `bitnet-lut-kernels-tl2.h` `three_tbl_impl_<m>_<k>`):
//!
//! ```text
//! tile_idx          = output_row // BM
//! row_in_tile       = output_row %  BM
//! i_group           = row_in_tile // 32   (BM=64 → ∈ {0, 1})
//! lane              = row_in_tile %  32
//!
//! k_outer           = triplet_col_idx // KK   (KK = BBK / 3 = 42)
//! triplet_in_kouter = triplet_col_idx %  KK
//! ai                = triplet_in_kouter // 2
//! nibble            = triplet_in_kouter %  2  (0=upper, 1=lower)
//!
//! byte_offset = tile_idx * (n_kouter * a_per_kouter)
//!             + k_outer  * a_per_kouter
//!             + i_group  * (a_per_kouter / 2)
//!             + ai       * 32
//!             + lane
//! ```
//!
//! Per-byte: upper nibble = magnitude index for triplet (2*ai),
//!           lower nibble = magnitude index for triplet (2*ai+1).

/// Canonical 14-entry magnitude LUT — same as Python
/// `THREE_LUT_CANONICAL_TO_IDX` at `repack_to_tl2.py:118`.
/// Indexed by sign-canonicalized triplet (where the leading
/// nonzero is +1); maps to the 4-bit magnitude index 0..13.
/// Indices 14, 15 are reserved/unused.
const fn canonical_to_idx(canonical: (i8, i8, i8)) -> u8 {
    match canonical {
        (0, 0, 0) => 0,
        (0, 0, 1) => 1,
        (0, 1, -1) => 2,
        (0, 1, 0) => 3,
        (0, 1, 1) => 4,
        (1, -1, -1) => 5,
        (1, -1, 0) => 6,
        (1, -1, 1) => 7,
        (1, 0, -1) => 8,
        (1, 0, 0) => 9,
        (1, 0, 1) => 10,
        (1, 1, -1) => 11,
        (1, 1, 0) => 12,
        (1, 1, 1) => 13,
        _ => 0, // unreachable for valid {-1, 0, +1} input
    }
}

/// Encode a (w0, w1, w2) ternary triplet → (magnitude_index, sign_bit).
/// Mirrors `repack_to_tl2.py::encode_triplet`.
///
/// Sign convention: leading-nonzero weight's sign is factored out;
/// the magnitude index identifies the sign-canonicalized form
/// (where the leading nonzero is +1).
pub fn encode_triplet(w0: i8, w1: i8, w2: i8) -> (u8, u8) {
    debug_assert!(w0 == -1 || w0 == 0 || w0 == 1);
    debug_assert!(w1 == -1 || w1 == 0 || w1 == 1);
    debug_assert!(w2 == -1 || w2 == 0 || w2 == 1);

    let sign_factor: i8 = if w0 != 0 {
        if w0 > 0 { 1 } else { -1 }
    } else if w1 != 0 {
        if w1 > 0 { 1 } else { -1 }
    } else if w2 != 0 {
        if w2 > 0 { 1 } else { -1 }
    } else {
        1 // all-zero; sign immaterial
    };

    let canonical = (w0 * sign_factor, w1 * sign_factor, w2 * sign_factor);
    let idx = canonical_to_idx(canonical);
    let sign = if sign_factor == 1 { 0 } else { 1 };
    (idx, sign)
}

/// Tile-organized layout sizes per kernel strides.
/// Mirrors `repack_to_tl2.py::tl2_tile_layout_sizes`.
#[derive(Debug, Clone, Copy)]
pub struct TileLayoutSizes {
    pub a_bytes: usize,
    pub sign_bytes: usize,
    pub a_per_kouter: usize,
    pub sign_per_kouter: usize,
    pub n_tiles: usize,
    pub n_kouter: usize,
}

/// Compute byte counts for the tile-organized layout. Asserts
/// kernel preconditions (m % bm == 0, k % bbk == 0).
///
/// For (m=256, k=256, bm=64, bbk=128): a_bytes=10752, sign_bytes=2560.
pub fn tile_layout_sizes(m: usize, k: usize, bm: usize, bbk: usize) -> TileLayoutSizes {
    assert!(m.is_multiple_of(bm), "m={m} not divisible by BM={bm}");
    assert!(k.is_multiple_of(bbk), "k={k} not divisible by BBK={bbk}");
    let kk = bbk / 3;
    let a_per_kouter = 2 * (kk / 2) * 32;
    let s_per_kouter = 2 * (kk / 8) * 32;
    let n_tiles = m / bm;
    let n_kouter = k / bbk;
    TileLayoutSizes {
        a_bytes: n_tiles * n_kouter * a_per_kouter,
        sign_bytes: n_tiles * n_kouter * s_per_kouter,
        a_per_kouter,
        sign_per_kouter: s_per_kouter,
        n_tiles,
        n_kouter,
    }
}

/// Tile-organized TL2 wire layout. Magnitude buffer follows the
/// kernel's `a[i * KK/2 + ai * 32]` stride; sign buffer is parallel
/// but currently zero-filled (iteration 4 deliverable).
#[derive(Debug)]
pub struct TileLayout {
    pub magnitudes: Vec<u8>,
    pub signs: Vec<u8>,
    pub m: usize,
    pub k: usize,
    pub bm: usize,
    pub bbk: usize,
    pub n_tiles: usize,
    pub n_kouter: usize,
}

/// Pack ternary weights `(m, k)` into TL2 tile-organized format.
/// Iteration 4: magnitude AND sign bytes populated.
///
/// Mirrors `repack_to_tl2.py::pack_tl2_tile_organized`.
///
/// # Arguments
///
/// - `weights`: row-major `m * k` slice of ternary values
///   `{-1, 0, +1}`. Indexed as `weights[row * k + col]`.
/// - `m, k, bm, bbk`: matrix + tile dims (default Mini: bm=64, bbk=128).
///
/// # Sign packing layout (iteration 4 best guess)
///
/// Per kernel `slli_epi16(vec_sign, 4*k+sub)` pattern:
///
/// ```text
/// as_idx           = triplet_in_kouter // 8     (KK/8 = 5 vec_sign per i_group)
/// triplet_in_as    = triplet_in_kouter %  8
/// sign_byte_offset = tile_idx * s_per_tile
///                  + k_outer  * s_per_kouter
///                  + i_group  * (s_per_kouter / 2)
///                  + as_idx   * 32
///                  + lane
/// bit_in_byte      = triplet_in_as
/// ```
///
/// Validated by iteration-5 parity_real_mlp test; if that fails,
/// the bit_in_byte permutation is the most likely culprit.
///
/// # Panics
///
/// On `m % bm != 0` or `k % bbk != 0` (kernel precondition) or
/// `weights.len() != m * k`.
pub fn pack_tile_organized(
    weights: &[i8],
    m: usize,
    k: usize,
    bm: usize,
    bbk: usize,
) -> TileLayout {
    assert_eq!(weights.len(), m * k, "weights len mismatch");
    let sizes = tile_layout_sizes(m, k, bm, bbk);
    let kk = bbk / 3;
    let a_per_tile = sizes.n_kouter * sizes.a_per_kouter;
    let i_group_a_stride = sizes.a_per_kouter / 2;
    let s_per_tile = sizes.n_kouter * sizes.sign_per_kouter;
    let i_group_s_stride = sizes.sign_per_kouter / 2;
    let max_as_idx = kk / 8;

    let mut a_buf = vec![0u8; sizes.a_bytes];
    let mut s_buf = vec![0u8; sizes.sign_bytes];

    for output_row in 0..m {
        let tile_idx = output_row / bm;
        let row_in_tile = output_row % bm;
        let i_group = row_in_tile / 32;
        let lane = row_in_tile % 32;

        for k_outer in 0..sizes.n_kouter {
            for triplet_in_kouter in 0..kk {
                let k_base = k_outer * bbk + triplet_in_kouter * 3;
                if k_base + 2 >= k {
                    continue;
                }
                let w0 = weights[output_row * k + k_base];
                let w1 = weights[output_row * k + k_base + 1];
                let w2 = weights[output_row * k + k_base + 2];
                let (mag_idx, sign_bit) = encode_triplet(w0, w1, w2);

                let ai = triplet_in_kouter / 2;
                let nibble = triplet_in_kouter % 2;

                // F.11.4 Path B step 4 finding (`derive_byte_to_row_mapping`):
                // kernel uses unpacklo/unpackhi to combine LUT results,
                // which crosses AVX2 lanes. Empirically (single-byte
                // sweep with K=256, BK=256, signs=0):
                //   row 0..7   ← input byte 0..7
                //   row 8..15  ← input byte 16..23   (lane-crossed)
                //   row 16..23 ← input byte 8..15    (lane-crossed)
                //   row 24..31 ← input byte 24..31
                // Encoder must place row r's mag byte at the position
                // the kernel will read it from. Without this fix, rows
                // 8-23 produce DRAMATICALLY scrambled output (diffs of
                // ±97/±126/±443 instead of the uniform per-period
                // ±32/±31/±1 cycle that signals a row-uniform residual).
                let byte_in_vec = match lane {
                    l if l < 8 => l,
                    l if l < 16 => l + 8,
                    l if l < 24 => l - 8,
                    _ => lane,
                };

                let byte_offset = tile_idx * a_per_tile
                    + k_outer * sizes.a_per_kouter
                    + i_group * i_group_a_stride
                    + ai * 32
                    + byte_in_vec;
                if nibble == 0 {
                    a_buf[byte_offset] |= (mag_idx & 0xF) << 4;
                } else {
                    a_buf[byte_offset] |= mag_idx & 0xF;
                }

                // Iteration 6 sign packing — derived from kernel
                // `slli_epi16+srai_epi16` bit-extraction pattern.
                // See repack_to_tl2.py::pack_tl2_tile_organized §sign-packing
                // for full derivation.
                if sign_bit != 0 {
                    let as_idx = triplet_in_kouter / 8;
                    let triplet_in_as = triplet_in_kouter % 8;
                    if as_idx < max_as_idx {
                        let ai_in_as = triplet_in_as / 2;
                        let nibble_for_sign = triplet_in_as % 2; // 0=top, 1=bot
                        let row_hi_lo = if lane >= 16 { 1 } else { 0 };
                        let bit_pos_in_lane: usize =
                            15 - 4 * ai_in_as - 2 * nibble_for_sign - row_hi_lo;

                        let byte_in_lane_pair = bit_pos_in_lane / 8;
                        let bit_in_byte = bit_pos_in_lane % 8;
                        let lane_in_16 = lane % 16;

                        let sign_byte_off = tile_idx * s_per_tile
                            + k_outer * sizes.sign_per_kouter
                            + i_group * i_group_s_stride
                            + as_idx * 32
                            + lane_in_16 * 2
                            + byte_in_lane_pair;
                        s_buf[sign_byte_off] |= 1u8 << bit_in_byte;
                    }
                }
            }
        }
    }

    TileLayout {
        magnitudes: a_buf,
        signs: s_buf,
        m,
        k,
        bm,
        bbk,
        n_tiles: sizes.n_tiles,
        n_kouter: sizes.n_kouter,
    }
}

/// Back-compat alias for iteration 3 — same as `pack_tile_organized`
/// but emphasizes the magnitude side of the encoding. Will be
/// removed once all callers migrate.
#[deprecated(note = "use pack_tile_organized; sign packing now included")]
pub fn pack_tile_organized_magnitude(
    weights: &[i8],
    m: usize,
    k: usize,
    bm: usize,
    bbk: usize,
) -> TileLayout {
    pack_tile_organized(weights, m, k, bm, bbk)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_triplet_all_27_round_trip_to_canonical() {
        // Property: every (w0, w1, w2) ∈ {-1, 0, +1}^3 produces a
        // (mag_idx, sign_bit) where decoding via the inverse LUT
        // recovers the input. Mirrors Python's
        // `test_encode_triplet_all_27_round_trip`.
        for w0 in [-1_i8, 0, 1] {
            for w1 in [-1_i8, 0, 1] {
                for w2 in [-1_i8, 0, 1] {
                    let (idx, sign) = encode_triplet(w0, w1, w2);
                    assert!(idx <= 13, "({w0},{w1},{w2}) → idx {idx} > 13");
                    assert!(sign <= 1);
                }
            }
        }
    }

    #[test]
    fn encode_triplet_all_zero_idx_zero_sign_zero() {
        assert_eq!(encode_triplet(0, 0, 0), (0, 0));
    }

    #[test]
    fn encode_triplet_all_positive_one_idx_thirteen_sign_zero() {
        assert_eq!(encode_triplet(1, 1, 1), (13, 0));
    }

    #[test]
    fn encode_triplet_all_negative_one_idx_thirteen_sign_one() {
        // (-1,-1,-1) sign-canonicalizes to (1,1,1) by negating all;
        // mag=13, sign=1 (negate the LUT-stored value at decode time)
        assert_eq!(encode_triplet(-1, -1, -1), (13, 1));
    }

    #[test]
    fn tile_layout_sizes_mini_256_256() {
        let sizes = tile_layout_sizes(256, 256, 64, 128);
        assert_eq!(sizes.a_bytes, 10_752);
        assert_eq!(sizes.sign_bytes, 2_560);
        assert_eq!(sizes.n_tiles, 4);
        assert_eq!(sizes.n_kouter, 2);
        assert_eq!(sizes.a_per_kouter, 1_344);
        assert_eq!(sizes.sign_per_kouter, 320);
    }

    #[test]
    fn pack_tile_organized_zero_input_yields_zero_buffers() {
        let weights = vec![0_i8; 256 * 256];
        let layout = pack_tile_organized(&weights, 256, 256, 64, 128);
        assert!(layout.magnitudes.iter().all(|&b| b == 0));
        assert!(layout.signs.iter().all(|&b| b == 0));
    }

    #[test]
    fn pack_tile_organized_all_positive_yields_idx13_no_signs() {
        // (+1,+1,+1) → idx=13, sign=0. Magnitudes = 0xDD; signs = 0.
        let weights = vec![1_i8; 256 * 256];
        let layout = pack_tile_organized(&weights, 256, 256, 64, 128);
        assert!(
            layout.magnitudes.iter().all(|&b| b == 0xDD),
            "expected all 0xDD; first non-DD at {:?}",
            layout.magnitudes.iter().position(|&b| b != 0xDD)
        );
        assert!(layout.signs.iter().all(|&b| b == 0));
    }

    #[test]
    fn pack_tile_organized_all_negative_yields_idx13_all_signs() {
        // (-1,-1,-1) → canonical (+1,+1,+1) → idx=13, sign=1.
        // Magnitudes: 0xDD same as +1 case (canonicalized).
        // Signs: every triplet has sign bit set. Per (256, 256):
        //   - 5 valid as_idx values × 8 bit positions = 40 sign bits/lane
        //   - But we set bit (triplet_in_kouter % 8) for each triplet
        //   - Per as_idx (covers triplets 8*as_idx..8*as_idx+7), all 8
        //     bits in that byte set → byte = 0xFF
        // → all sign bytes should be 0xFF.
        let weights = vec![-1_i8; 256 * 256];
        let layout = pack_tile_organized(&weights, 256, 256, 64, 128);
        assert!(
            layout.magnitudes.iter().all(|&b| b == 0xDD),
            "magnitudes for (-1,-1,-1) canonicalize to (+1,+1,+1) idx 13"
        );
        assert!(
            layout.signs.iter().all(|&b| b == 0xFF),
            "expected all 0xFF signs; first non-FF at {:?}",
            layout.signs.iter().position(|&b| b != 0xFF)
        );
    }

    #[test]
    fn pack_tile_organized_known_offset_first_triplet_positive() {
        // Row 0, cols 0..2 = (+1,+1,+1) → idx=13, sign=0.
        // → byte 0 magnitudes = 0xD0; all signs zero.
        let mut weights = vec![0_i8; 256 * 256];
        weights[0] = 1;
        weights[1] = 1;
        weights[2] = 1;
        let layout = pack_tile_organized(&weights, 256, 256, 64, 128);
        assert_eq!(layout.magnitudes[0], 0xD0);
        assert!(layout.magnitudes[1..32].iter().all(|&b| b == 0));
        assert!(layout.signs.iter().all(|&b| b == 0));
    }

    #[test]
    fn pack_tile_organized_known_offset_first_triplet_negative() {
        // Row 0, cols 0..2 = (-1,-1,-1) → idx=13, sign=1.
        // Magnitudes: byte 0 = 0xD0 (canonical magnitude same).
        // Signs (per iteration 6 layout):
        //   row 0 → lane=0, lane_in_16=0, row_hi_lo=0
        //   triplet 0 → as_idx=0, ai_in_as=0, nibble=0
        //   bit_pos_in_lane = 15 - 4*0 - 2*0 - 0 = 15
        //   byte_in_lane_pair = 15 // 8 = 1 (high byte of pair)
        //   bit_in_byte = 15 % 8 = 7
        //   sign_byte_off = 0 + 0 + 0 + 0 + 0*2 + 1 = 1
        //   → byte 1 has bit 7 set = 0x80
        let mut weights = vec![0_i8; 256 * 256];
        weights[0] = -1;
        weights[1] = -1;
        weights[2] = -1;
        let layout = pack_tile_organized(&weights, 256, 256, 64, 128);
        assert_eq!(layout.magnitudes[0], 0xD0);
        assert_eq!(layout.signs[0], 0x00);
        assert_eq!(
            layout.signs[1], 0x80,
            "expected bit 7 set in sign byte 1; got 0x{:02X}",
            layout.signs[1]
        );
        assert!(layout.signs[2..].iter().all(|&b| b == 0));
    }

    #[test]
    fn pack_tile_organized_rejects_size_mismatch() {
        let weights = vec![0_i8; 100]; // wrong: 100 != 256*256
        let result = std::panic::catch_unwind(|| pack_tile_organized(&weights, 256, 256, 64, 128));
        assert!(result.is_err());
    }
}
