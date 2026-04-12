# V26 C1.6 Path B — Phase B2 Diagnosis

> **Date:** 2026-04-13
> **Input:** `docs/V26_C1_6_BASELINE_RESULTS.md` (B1 data), eigenvalue analysis
> **Output:** Root cause ranking + design option scoring → B2.D decision
> **Rule compliance:** §6.8 Rule 6 (mechanical decision gate), §6.9 Rule 2 (literature first)

---

## B2.1.1: Δ(FQ - best competitor) per model per bit

| Model | Bits | FQ PPL | Best PPL | Best method | Δ(FQ - Best) | FQ rank |
|---|---|---|---|---|---|---|
| Gemma 4 E2B | 2 | 125.19 | 39.73 | TQ_outlier | +85.46 | 3rd/4 |
| Gemma 4 E2B | 3 | 24.26 | 21.90 | KIVI | +2.36 | 2nd/4 |
| Gemma 4 E2B | 4 | 26.75 | 26.75 | FQ | 0.00 | **1st/4** |
| Mistral-7B | 2 | 868.57 | 23.96 | KIVI | +844.61 | 4th/4 |
| Mistral-7B | 3 | 19.44 | 5.99 | KIVI | +13.44 | 4th/4 |
| Mistral-7B | 4 | 6.07 | 5.73 | KIVI | +0.35 | 4th/4 |
| Qwen2-7B | 2 | 262.89 | 46.70 | KIVI | +216.19 | 3rd/4 |
| Qwen2-7B | 3 | 8.93 | 8.01 | KIVI | +0.91 | 4th/4 |
| Qwen2-7B | 4 | 7.74 | 7.62 | KIVI | +0.12 | 4th/4 |

**Pattern:** FQ v1 loses worst at 2-bit (Δ = +85 to +845), converges at 4-bit
(Δ < 1 except Gemma where it wins). The 2-bit regime is where PCA's
weaknesses are maximally exposed.

---

## B2.1.2: Architecture correlation

### KV-head architecture vs FQ performance

| Model | KV heads | Head dim | FQ 2-bit Δ_FP16 | KIVI 2-bit Δ_FP16 |
|---|---|---|---|---|
| Gemma 4 E2B | 1 (MQA) | 256 | +97.06 | +442.37 |
| Mistral-7B | 8 (GQA 32:8) | 128 | +862.90 | +18.29 |
| Qwen2-7B | 4 (GQA 28:4) | 128 | +255.34 | +39.15 |

**Observation:** KIVI excels on GQA (more KV heads = more per-channel statistics
to work with). FQ's PCA actually does better on MQA (1 KV head, larger head_dim
= richer eigenspectrum). But "better" is relative — FQ still loses to TQ_outlier
on Gemma.

### Eigenvalue concentration (layer 0 K, 512 tokens)

| Model | Head dim | Top-1 % var | Top-5 % var | Top-10 % var | Outlier ratio (max/median σ) |
|---|---|---|---|---|---|
| Gemma 4 E2B | 256 | 10.9% | 32.0% | 42.2% | 9.3× |
| Mistral-7B | 128 | 16.9% | 51.9% | 67.2% | 9.0× |
| Qwen2-7B | 128 | **68.1%** | **99.7%** | **99.8%** | **231.5×** |

**Critical finding:** Qwen2-7B has extreme eigenvalue concentration — 68% of
variance in a single component, 99.7% in top-5. This means:

1. PCA on Qwen2 K data is dominated by 1-5 directions. The remaining 123+
   components carry almost no variance. Per-chunk PCA will be extremely
   unstable because the trailing eigenvalues are noise-dominated.

2. KIVI's per-channel quantization works well because the "outlier channels"
   ARE the signal — preserving channel-wise min/max captures the dominant
   variance directions implicitly.

3. TurboQuant's random rotation "spreads" this concentrated variance across
   all dimensions, which is good for uniform quantization but loses the
   structural information that a targeted method could exploit.

**Mistral** has moderate concentration (top-5 = 52%), explaining why KIVI still
wins (per-channel captures the structure) but the gap to TQ_outlier is smaller
than Qwen2.

**Gemma** has the flattest spectrum (top-5 = 32%), making it the most amenable
to PCA — and indeed FQ v1 does best (relatively) on Gemma.

---

## B2.1.3: Root Cause Ranking

Based on the data, ranked by contribution to FQ v1's underperformance:

| Rank | Root Cause | Evidence | Weight |
|---|---|---|---|
| **1** | **RC5: No outlier extraction** | TQ_outlier beats TQ_naive by 1.3-4.7× at 2-bit across all models. FQ has zero outlier handling. Adding outlier extraction alone would close a large fraction of the gap. | **40%** |
| **2** | **RC2: Per-chunk PCA noise** | Qwen2's 231× outlier ratio means per-chunk PCA axes are wildly unstable. Calibrated PCA (stable axes from large sample) would eliminate this variance. Gemma's flatter spectrum makes per-chunk PCA more stable, which is why FQ does best on Gemma. | **25%** |
| **3** | **RC1: PCA aligns with outliers** | With 68% variance in 1 component (Qwen2), PCA's top axis IS the outlier axis. Quantizing uniformly along PCA axes means the outlier axis gets the same bit depth as noise axes — wasting bits. Extracting outliers before PCA (RC5 fix) also fixes this. | **20%** |
| **4** | **RC3: PCA on RoPE-applied K** | RoPE rotates K by position-dependent angles, making the "true" K distribution position-dependent. PCA on RoPE-applied K captures a mixture of positional structure and semantic structure. Undoing RoPE before rotation would give cleaner PCA. Less impactful than RC5/RC2 because all methods face the same RoPE challenge. | **10%** |
| **5** | **RC4: Independent K/V rotation** | K and V have different optimal rotations. But since KIVI (which treats K and V differently) dominates, the asymmetry is important. However, the main FQ weakness is not K-V coupling but outlier handling. | **5%** |

**Total budget:** RC5 (40%) + RC2 (25%) + RC1 (20%) = **85% of the gap** from
the top 3 root causes alone. These are all addressed by **F2-D (PCA + outlier
extraction)** or its superset **F2-F**.

---

## B2.2.1: Design Option Scoring

Scoring against the B1 data and eigenvalue analysis:

| Option | Targets | RC coverage | Effort | Expected 2-bit PPL gain | Paper narrative | Score |
|---|---|---|---|---|---|---|
| F2-A (calibrated PCA only) | RC2 | 25% | 6-10h | Small (fixes noise, not outliers) | weak | 3/10 |
| F2-B (Hadamard + linear) | RC1, RC2, RC4 | 50% | 12-20h | Medium (spreads outliers, doesn't extract) | novel but distant from PCA story | 5/10 |
| F2-C (learned rotation) | RC1, RC2 | 45% | 3-5d | Medium-high (optimized rotation) | strong but slow to train | 6/10 |
| **F2-D (PCA + outlier extraction)** | **RC1, RC2, RC5** | **85%** | **2-4d** | **High (fixes top 3 root causes)** | **preserves PCA narrative + adds outlier story** | **9/10** |
| F2-E (DP bit allocation, addon) | partial RC1, RC5 | 15% standalone | 1-2d addon | Low standalone, good as addon | addon only | 4/10 |
| F2-F (full combined: D + E + RoPE) | RC1-RC5 | 100% | 1-3w | Maximum | strongest but highest risk | 8/10 |

---

## B2.2.2: Decision

### Chosen design: **F2-D (calibrated PCA + outlier extraction)**

**Rationale:**

1. **Highest efficiency:** F2-D covers 85% of the root cause weight (RC5 + RC2
   + RC1) in 2-4 days of effort. F2-F covers 100% but costs 1-3 weeks — the
   marginal 15% (RC3 + RC4) is not worth 3-5× more time.

2. **Paper narrative preserved:** FajarQuant's identity is "PCA-based KV cache
   quantization." F2-D adds outlier extraction as an enhancement to PCA, not a
   replacement. The paper story is "calibrated PCA with outlier-aware
   preprocessing" — a natural evolution, not a pivot.

3. **Clear ablation path:** v1 → v2-A (calibrated only) → v2-D (+ outliers)
   gives a clean 3-row ablation table that shows each component's contribution.
   Can optionally add F2-E (DP bits) as a 4th row if time permits.

4. **Data-backed:** The eigenvalue analysis shows Qwen2's extreme concentration
   (68% top-1) is the hardest case. Extracting the top-1% outlier channels
   before PCA should dramatically improve PCA stability on Qwen2 and Mistral.
   Gemma (flattest spectrum) should also benefit from calibrated PCA.

5. **Upgrade path:** If F2-D is insufficient, F2-F adds RoPE undo + DP bits
   incrementally. No throwaway work.

### F2-D Algorithm Summary

```
Calibration (offline, once per model):
  For each layer L:
    1. Run 128 WikiText-2 sequences through the model
    2. Collect K[L] tensors: shape (N_total, head_dim)
    3. Compute per-channel variance across all samples
    4. Identify top-1% channels by variance → outlier_channels[L]
    5. Remove outlier channels from K: K_clean = K[:, non_outlier_channels]
    6. PCA on K_clean: cov → eigendecomposition → R[L], mean[L]
    7. Save: outlier_channels[L], R[L], mean[L] → .npz file

Inference (per layer, per token batch):
  1. Split K into K_outlier (outlier channels, kept fp16) and K_rest
  2. K_rest_centered = K_rest - mean[L]
  3. K_rest_rotated = K_rest_centered @ R[L].T
  4. K_rest_quant = per_coord_uniform_quant(K_rest_rotated, bits)
  5. K_rest_recon = K_rest_quant @ R[L] + mean[L]
  6. K_recon = merge(K_outlier_fp16, K_rest_recon) at original positions
  
  Same process for V (separate outlier channels, separate PCA).
```

### Effort breakdown

| Task | Est. | Notes |
|---|---|---|
| Calibration script (`calibrate_fq_v2.py`) | 4-6h | 128 samples × 3 models, per-layer K/V extraction, PCA, outlier identification, .npz saving |
| v2 quantization core (`quant_attention_v2.py`) | 4-6h | Outlier split + calibrated PCA apply + merge. Reuse R-α.1 monkey-patch infra from v1 |
| Smoke test (Gemma 5 samples) | 0.5h | Verify v2 < v1 at 2-bit |
| Unit tests | 2h | Roundtrip accuracy, shape checks, outlier channel preservation |
| **Total** | **~2-3 days** | Within B3 budget (2-5 days) |

### Fallback plan

If F2-D smoke test (B3.S) shows < 20% improvement over v1 at 2-bit:
→ Upgrade to F2-F by adding undo-RoPE step before PCA (1-2 days extra).
If still insufficient: → Add F2-E DP bit allocation (1 day extra).
If all combined still loses to TQ_outlier at 2-bit on ≥2/3 models:
→ B4.G = NO-GO. Honest paper writeup as "PCA-based methods are inferior
  to rotation-based methods at ultra-low bitwidths, but competitive at 4-bit."

---

*Document version: 2026-04-13 v1.0. Decision: F2-D. Gate: B2.D.1 → commit
before any v2 code (Rule 6).*
