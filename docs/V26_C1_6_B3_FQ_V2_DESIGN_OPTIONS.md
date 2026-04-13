# FajarQuant v2.12 — Design Options (Reference for Phase B2)

> **Purpose:** Pre-B2 reference document cataloging each v2 design option
> with enough detail that B2.2 can score them against the B1 failure data.
> Created during B1 execution to parallelize prep work.
>
> **Usage:** after B1.G baseline results are signed off, read this doc
> alongside `V26_C1_6_BASELINE_RESULTS.md` (not yet created) and score
> each option in §"Design Option Scoring" of `V26_C1_6_DIAGNOSIS.md`.
> The best-scoring option becomes the B2.D decision.

---

## Design space overview

FajarQuant v1 has 5 diagnosed root causes (RC1-RC5). Each design option
targets a subset. The "ideal" design would address all 5, but complexity
and effort scale with the number addressed.

| RC | Root cause | Most impactful fix |
|---|---|---|
| RC1 | PCA aligns with outliers | Hadamard / learned rotation / regularized PCA |
| RC2 | Per-chunk PCA recomputation = noise | Calibrated global PCA per layer |
| RC3 | PCA on RoPE-applied K = wrong basis | Undo RoPE before rotation |
| RC4 | Independent K and V rotations | Joint K-V rotation or asymmetric treatment |
| RC5 | No outlier extraction | Top-K channel preservation in fp16 |

---

## F2-A: Calibrated Global PCA (KVTC-inspired)

### Concept
Replace per-chunk PCA with a single PCA basis per layer, computed once
on a calibration dataset (128 WikiText samples). Store the basis as a
`.npz` file, load at inference time, reuse for all chunks.

### Algorithm
```
Calibration (offline, once per model per layer):
  1. Run model on 128 WikiText sequences, collect per-layer K tensors
  2. Concatenate: K_all[layer] = concat(K[0], K[1], ..., K[127])  shape (N, D)
  3. PCA: mean = K_all.mean(0), cov = (K_centered.T @ K_centered) / (N-1)
  4. eigh(cov) → eigenvalues, eigenvectors sorted desc
  5. R[layer] = eigenvectors.T  (rotation matrix, shape D×D)
  6. Save R[layer] + mean[layer] to data/calibration/fq_v2_{model}.npz

Inference (per layer, per token batch):
  1. Load R[layer], mean[layer] from calibration file
  2. centered = K - mean
  3. rotated = centered @ R.T
  4. quantized = per_coord_uniform_quant(rotated, bits)
  5. K_recon = quantized @ R + mean
```

### Targets
- RC2 ✅ (calibrated = stable across chunks)
- RC1 ❌ (PCA still aligns with outliers)
- RC3 ❌ (still PCA on RoPE-applied data)
- RC4 ❌ (K and V still independent)
- RC5 ❌ (no outlier extraction)

### Effort: ~6-10 hours
- Calibration script: 3-5h
- Loader + integration into quant_attention.py: 2-3h
- Testing: 1-2h

### Expected PPL improvement
Moderate: eliminates per-chunk noise but keeps outlier alignment problem.
KVTC (which does this + adaptive quant + entropy coding) gets "within 1
score point of vanilla" — but KVTC also has RC5 via DP bit allocation.
F2-A alone probably gets ~50% of the way there.

---

## F2-B: Hadamard + Linear Correction (KVLinC-inspired)

### Concept
Replace PCA with data-independent Walsh-Hadamard Transform (WHT) for
values; use per-channel quantization for keys (per KVLinC finding that
"keys contain channel-wise outliers, values do not"). Add a small
learnable linear correction adapter per layer to compensate for
residual error.

### Algorithm
```
For values:
  1. V_had = hadamard_transform(V)  // O(S * D * log(D))
  2. V_quant = per_coord_quant(V_had, bits)
  3. V_recon = inverse_hadamard(V_quant)
  4. V_corrected = V_recon + linear_correction[layer](V_recon)

For keys:
  1. K_quant = per_channel_quant(K, bits)  // min/max per (B,H,D) over S
  2. K_corrected = K_quant + linear_correction_k[layer](K_quant)
```

### Targets
- RC1 ✅ (Hadamard anti-concentrates outliers)
- RC2 ✅ (WHT is data-independent = no recomputation)
- RC3 ❌ (still applied after RoPE)
- RC4 ✅ (K and V treated differently by design)
- RC5 partial (Hadamard spreads but doesn't extract)

### Effort: ~12-20 hours
- Hadamard implementation: 3-5h (FFT butterfly, or use scipy/torch)
- Linear correction adapter training: 4-8h (calibration loop)
- Integration: 3-4h
- Testing: 2-3h

### Expected PPL improvement
High: Hadamard is proven to work for outlier handling (QuaRot, SpinQuant,
RotateKV, KVLinC all use variants). Linear correction closes the remaining
gap. KVLinC reports "strong results on Llama, Qwen2.5, Qwen3".

---

## F2-C: Learned Rotation via Cayley SGD (SpinQuant-inspired)

### Concept
Initialize rotation matrix R as Hadamard. Optimize R via Cayley
parameterization (constrains R to SO(d)) on a small calibration set.
Loss = downstream attention output error (not raw MSE, but actual
model quality impact).

### Algorithm
```
Calibration:
  1. Initialize R[layer] = Hadamard(D)
  2. For 100 iterations:
     a. Sample batch from WikiText-2 calibration set
     b. Forward pass with R[layer]-quantized attention
     c. Compute loss = attention_output_mse(quantized, fp16)
     d. Cayley gradient step: R ← exp(η * skew(∇R)) @ R
  3. Save R[layer] to calibration file

Inference:
  1. Load R[layer]
  2. rotated = (K - mean) @ R.T
  3. quantized = per_coord_quant(rotated, bits)
  4. K_recon = quantized @ R + mean
```

### Targets
- RC1 ✅ (learned rotation optimized away from outlier alignment)
- RC2 ✅ (calibrated once per layer)
- RC3 partial (could add undo-RoPE step before rotation)
- RC4 partial (could learn shared K-V rotation)
- RC5 ❌ (no outlier extraction)

### Effort: ~3-5 days
- Cayley parameterization: 1 day
- Training loop with attention loss: 1-2 days
- Integration + testing: 1-2 days

### Expected PPL improvement
Highest: SpinQuant narrows accuracy gap to 2.9 points on Llama-2 7B
at W4A4KV4. FlatQuant beats SpinQuant by 7.5% with affine transforms.
This is the closest to state-of-the-art individual method.

---

## F2-D: PCA + Outlier Extraction Hybrid (KVQuant + FajarQuant fusion)

### Concept
Keep PCA narrative (data-driven). Fix the outlier problem by extracting
top-1% high-magnitude channels before PCA. PCA operates on the remaining
99% channels which are outlier-free → PCA works much better.

### Algorithm
```
Calibration (offline):
  1. Collect per-layer K/V statistics over 128 WikiText sequences
  2. Identify top-1% channels by per-layer cross-batch variance
  3. Save: outlier_channels[layer], PCA_R[layer], mean[layer]

Inference:
  1. Split K into K_outlier (1% channels, fp16) and K_rest (99%)
  2. K_rest_centered = K_rest - mean[layer]
  3. K_rest_rotated = K_rest_centered @ PCA_R[layer].T
  4. K_rest_quant = per_coord_quant(K_rest_rotated, bits)
  5. K_rest_recon = K_rest_quant @ PCA_R[layer] + mean[layer]
  6. K_recon = merge(K_outlier, K_rest_recon)  // scatter back to orig positions
```

### Targets
- RC1 ✅ (outliers removed before PCA → PCA basis clean)
- RC2 ✅ (calibrated PCA, not per-chunk)
- RC3 partial (could add undo-RoPE)
- RC4 ❌ (K and V still independent, but could share outlier channels)
- RC5 ✅ (explicit outlier extraction)

### Effort: ~2-4 days
- Outlier identification + dual storage: 1-2 days
- Calibrated PCA integration: 0.5-1 day
- Merge/scatter logic: 0.5 day
- Testing: 0.5-1 day

### Expected PPL improvement
High: KVQuant shows "removing only 1% of outliers → < 0.1 PPL degradation
at 3-bit." Combined with calibrated PCA, this should fix both the outlier
and noise problems. Paper narrative is preserved (FajarQuant = PCA-based)
while addressing the core weakness.

---

## F2-E: DP-Optimal Bit Allocation (KVTC-inspired addon)

### Concept
After any rotation (PCA, Hadamard, or learned), allocate bits per
coordinate non-uniformly using dynamic programming. High-variance
coordinates get more bits; trailing components get 0 bits (= free
dimensionality reduction).

### Algorithm
```
Calibration:
  1. After rotation, compute per-coordinate variance
  2. DP: minimize sum of per-coord MSE subject to total bit budget B
     - States: (coord_idx, remaining_bits)
     - Transitions: try 0, 1, 2, ..., 8 bits per coord
     - Each transition's cost = MSE for that coord at that bit depth
  3. Save bit_allocation[layer] = array of per-coord bit widths

Inference:
  1. After rotation, quantize each coordinate with its allocated bits
  2. Coords with 0 bits are dropped (free dim reduction)
  3. Dequantize with variable bits per coord
```

### Targets
- RC1 partial (assigns fewer bits to outlier axes — compensates)
- RC2 ❌ (orthogonal to chunk vs calibrated)
- RC3 ❌ (orthogonal to RoPE)
- RC4 ❌ (orthogonal to K-V coupling)
- RC5 partial (0-bit allocation = implicit extraction)

### Effort: ~1-2 days (as addon to F2-A, F2-C, or F2-D)
- DP algorithm: 0.5 day
- Variable-bit packing/unpacking: 0.5-1 day
- Integration: 0.5 day

### Expected PPL improvement
KVTC gets 20× compression from DP bit allocation. As a standalone fix
for FajarQuant, the gain depends on what rotation is used. Best combined
with F2-A or F2-D.

---

## F2-F: Full Combined (A + D + E + undo-RoPE)

### Concept
Maximum-effort combination: undo RoPE before PCA, extract outliers,
calibrated global PCA on clean data, DP bit allocation on rotated data.
This is the "KVTC-equivalent built in FajarQuant's framework."

### Algorithm
```
Calibration (offline):
  1. Collect K/V with RoPE UNDONE (inverse-rotate by position)
  2. Identify outlier channels (top 1%)
  3. On non-outlier channels: compute global PCA per layer
  4. On rotated data: compute per-coord variance → DP bit allocation
  5. Save: outlier_channels, PCA_R, mean, bit_allocation per layer

Inference:
  1. Receive K after model's RoPE application
  2. Undo RoPE: K_raw = inverse_rope(K, position_ids)
  3. Split: K_outlier (fp16), K_rest
  4. Rotate: K_rest_rotated = (K_rest - mean) @ PCA_R.T
  5. DP-quantize: apply per-coord bit allocation
  6. Inverse rotate + add mean
  7. Merge outliers back
  8. Re-apply RoPE: K_quant = apply_rope(K_recon, position_ids)
```

### Targets
- RC1 ✅ (outliers extracted before PCA → clean basis)
- RC2 ✅ (calibrated global PCA)
- RC3 ✅ (undo RoPE before PCA)
- RC4 partial (could share outlier channels between K and V)
- RC5 ✅ (explicit outlier extraction + DP assigns 0 bits to more)

### Effort: ~1-3 weeks
- RoPE undo implementation: 1-2 days
- Outlier extraction: 1-2 days
- Calibrated PCA: 1-2 days
- DP bit allocation: 1-2 days
- Integration + per-architecture forward patches: 2-3 days
- Testing + debugging: 2-3 days

### Expected PPL improvement
Maximum: addresses all 5 root causes. Should be competitive with KVTC
(which does PCA + adaptive quant + entropy coding) and potentially beat
TurboQuant at all bit widths. This is the version that justifies the
"FajarQuant v2.12" name and the paper narrative.

---

## Comparison matrix

| Option | RC1 | RC2 | RC3 | RC4 | RC5 | Effort | Expected PPL | Paper narrative |
|---|---|---|---|---|---|---|---|---|
| F2-A | ❌ | ✅ | ❌ | ❌ | ❌ | 6-10h | Moderate | "calibrated PCA" |
| F2-B | ✅ | ✅ | ❌ | ✅ | partial | 12-20h | High | "Hadamard + linear correction" |
| F2-C | ✅ | ✅ | partial | partial | ❌ | 3-5d | Highest (single) | "learned rotation" |
| F2-D | ✅ | ✅ | partial | ❌ | ✅ | 2-4d | High | "PCA + outlier extraction" |
| F2-E | partial | — | — | — | partial | 1-2d (addon) | Depends on base | "adaptive bit allocation" |
| **F2-F** | **✅** | **✅** | **✅** | **partial** | **✅** | **1-3w** | **Maximum** | **"outlier-aware calibrated PCA with adaptive bit allocation"** |

## Recommended default (subject to B1 data)

**F2-D** if the user wants moderate-effort results within days, with
option to upgrade to F2-F later. **F2-F** if the user commits to
maximum-effort full research scope (3-5 weeks total Path B estimate).

The user chose **"full effort"** in the Path B decision, so F2-F is the
recommended default unless B1 data shows a simpler fix suffices.

---

## Ablation roadmap (for the paper)

Regardless of which option is chosen, the paper §FajarQuant v2 section
should include an ablation table:

| Ablation | What changes | Purpose |
|---|---|---|
| v1 (baseline) | Per-chunk PCA, no outlier handling | Historical baseline |
| v2-A: calibrated only | Calibrated PCA, no outlier | Isolate RC2 fix |
| v2-D: + outlier extraction | + top-1% fp16 | Isolate RC5 fix |
| v2-D+RoPE: + undo-RoPE | + inverse RoPE step | Isolate RC3 fix |
| v2-F: + DP bit allocation | + adaptive bits | Isolate RC-E fix |
| v2-F-full | All combined | Full v2.12 |

Each row in the ablation table is one eval run (3 models × 3 bits ×
canonical protocol). Total ablation data: 6 rows × 9 cells = 54 numbers.
GPU cost: ~9 hours total (6 × 1.5h). This is the paper's strongest
argument: systematic improvement from each design component.

---

*Document version: 2026-04-12 v1.0. Will be updated in B2 with scoring
against B1 data. The chosen design is committed as a separate file
(V26_C1_6_V2_DESIGN.md) per Rule 6.*
