# FajarQuant v2.12 — Design Decision File

> **Decision:** F2-D (calibrated PCA + outlier extraction)
> **Date:** 2026-04-13
> **Author:** Fajar + Claude Opus 4.6
> **Gate:** B2.D (CLAUDE.md §6.8 Rule 6 — mechanical decision gate)
> **Input:** `docs/V26_C1_6_DIAGNOSIS.md`, `docs/V26_C1_6_BASELINE_RESULTS.md`
> **Blocked until this file is committed:** B3 (v2 implementation)

---

## Decision: F2-D

**Calibrated PCA + outlier extraction.** Covers 85% of root cause weight
(RC5 40% + RC2 25% + RC1 20%) in 2-3 days effort.

### Why F2-D over alternatives

- F2-A (calibrated only): fixes RC2 but not RC5 — outlier handling is 40% of gap
- F2-B (Hadamard): loses PCA paper narrative, 12-20h for partial RC5 coverage
- F2-C (learned rotation): 3-5 days for 45% coverage — worse ROI than F2-D
- F2-F (full combined): 1-3 weeks for marginal 15% gain over F2-D — upgrade later if needed

### Algorithm specification

```
=== CALIBRATION (offline, once per model per layer) ===

Input: model M, calibration dataset D (128 WikiText-2 sequences)
Output: per-layer files in data/calibration/fq_v2_{model}.npz

For each attention layer L in M:
  1. Forward D through M, collect K[L] and V[L]
     K_all shape: (N_total_tokens, num_kv_heads, head_dim)
     Flatten to: (N_total * num_kv_heads, head_dim)

  2. Compute per-channel variance:
     var_per_channel = K_all.var(dim=0)  # shape: (head_dim,)

  3. Identify outlier channels:
     threshold = percentile(var_per_channel, 99)  # top 1%
     outlier_mask = var_per_channel >= threshold
     n_outlier = outlier_mask.sum()  # typically 1-3 channels for head_dim=128

  4. Remove outlier channels:
     K_clean = K_all[:, ~outlier_mask]  # shape: (N, head_dim - n_outlier)

  5. PCA on clean data:
     mean = K_clean.mean(dim=0)
     K_centered = K_clean - mean
     cov = (K_centered.T @ K_centered) / (N - 1)
     eigenvalues, eigenvectors = torch.linalg.eigh(cov)
     # Sort descending
     idx = eigenvalues.argsort(descending=True)
     R = eigenvectors[:, idx].T  # rotation matrix, shape (D', D') where D' = head_dim - n_outlier

  6. Repeat steps 2-5 for V[L] (separate outlier channels, separate PCA)

  7. Save per-layer:
     - outlier_channels_k[L]: bool mask, shape (head_dim,)
     - outlier_channels_v[L]: bool mask, shape (head_dim,)
     - pca_rotation_k[L]: R matrix, shape (D'_k, D'_k)
     - pca_mean_k[L]: mean vector, shape (D'_k,)
     - pca_rotation_v[L]: R matrix, shape (D'_v, D'_v)
     - pca_mean_v[L]: mean vector, shape (D'_v,)


=== INFERENCE (per layer, per token batch) ===

Input: K tensor shape (B, num_kv_heads, S, head_dim), bits b
       Calibration data for this layer

For K:
  1. SPLIT: K_outlier = K[:, :, :, outlier_mask_k]   # (B, H, S, n_outlier) — kept in fp16
            K_rest = K[:, :, :, ~outlier_mask_k]      # (B, H, S, D')

  2. CENTER: K_rest -= mean_k                         # broadcast subtract

  3. ROTATE: K_rotated = K_rest @ R_k.T               # (B, H, S, D')

  4. QUANTIZE: K_quant = per_coord_uniform(K_rotated, b)
     Per-coordinate min/max over (B, H, S) dimension:
       min_val = K_rotated.amin(dim=(0,1,2))          # (D',)
       max_val = K_rotated.amax(dim=(0,1,2))          # (D',)
       scale = (max_val - min_val) / (2^b - 1)
       K_int = round((K_rotated - min_val) / scale)   # clamp to [0, 2^b-1]
       K_deq = K_int * scale + min_val

  5. INVERSE ROTATE: K_recon_rest = K_deq @ R_k + mean_k

  6. MERGE: K_recon = scatter(K_recon_rest, K_outlier_fp16, outlier_mask_k)
     # outlier channels get fp16 originals; rest get quantized+reconstructed

Same process for V with V-specific calibration data.

Return: K_recon, V_recon (same shape as input, mixed precision)
```

### Ablation roadmap

| Row | Config | What changes vs previous | Purpose |
|---|---|---|---|
| 1 | v1 (baseline) | — | Historical FQ v1 |
| 2 | v2-A (calibrated PCA only) | Calibrated global PCA replaces per-chunk PCA | Isolate RC2 (per-chunk noise) |
| 3 | v2-D (calibrated PCA + outlier) | + top-1% outlier extraction in fp16 before PCA | Isolate RC5 (outlier handling) |
| 4 | (optional) v2-D+E (+ DP bits) | + DP bit allocation on rotated coordinates | Isolate F2-E contribution |

Each ablation row = 3 models × 3 bits = 9 eval cells × 30 samples.

### Success criteria

**B3.S smoke test:** v2-D at 2-bit on Gemma < v1 at 2-bit on Gemma (125.19 PPL).
**B4.G go/no-go:** v2-D beats TQ_outlier at 2-bit on ≥2 of 3 models.

### Fallback plan

1. If v2-D smoke fails (< 20% improvement): add undo-RoPE (RC3) → F2-D+RoPE
2. If still insufficient: add DP bit allocation (F2-E) → F2-D+E
3. If full F2-F still loses to TQ_outlier on ≥2/3 models: B4.G = NO-GO
4. NO-GO paper: "PCA methods competitive at 4-bit, inferior at 2-bit;
   outlier-aware rotation methods (TurboQuant, QuaRot) dominate ultra-low."

---

*This file MUST be committed before any v2 implementation code (Rule 6).*
*Decision is final unless B3.S smoke test triggers fallback escalation.*
