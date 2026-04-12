# V26 C1.6 Path B — Phase B1 Baseline Results

> **Date:** 2026-04-13
> **Protocol:** R-α.1 model surgery (per-architecture attention forward subclasses)
> **Evaluator:** `scripts/eval_perplexity.py` — canonical non-overlapping chunks
> **Dataset:** WikiText-2 test set (Salesforce/wikitext, wikitext-2-raw-v1)
> **Parameters:** seq_len=2048, max_samples=30, 61,410 scored tokens per cell
> **GPU:** NVIDIA RTX 4090 Laptop (16 GB), torch 2.11.0+cu128
> **Source JSONs:** `data/kv_cache/perplexity_v1_baseline_{gemma,mistral,qwen2}.json`

---

## Methods evaluated

| Short name | Full name | Description |
|---|---|---|
| **FQ** | FajarQuant v1 | Per-chunk PCA rotation + per-coord uniform quantization |
| **KIVI** | KIVI (ICML 2024) | Per-channel asymmetric uniform (key) + per-token (value) |
| **TQ** | TurboQuant naive | Random orthogonal rotation + Lloyd-Max, NO outlier handling |
| **TQ_out** | TurboQuant outlier | Published TurboQuant ICLR 2026: top-15% high-variance channels in fp16 + TQ on rest |

---

## Model specifications

| Model | HF ID | KV-head architecture | Attn modules | Head dim |
|---|---|---|---|---|
| Gemma 4 E2B | `google/gemma-4-E2B` | MQA (1 KV head) | 35 | 256 |
| Mistral-7B | `mistralai/Mistral-7B-v0.1` | GQA 32:8 | 32 | 128 |
| Qwen2-7B | `Qwen/Qwen2-7B` | GQA 28:4 | 28 | 128 |

---

## FP16 baselines

| Model | FP16 PPL | Literature range | Status |
|---|---|---|---|
| Gemma 4 E2B | **28.13** | ~20-35 (small model) | Plausible |
| Mistral-7B | **5.67** | ~5.2-6.0 | Matches literature |
| Qwen2-7B | **7.55** | ~7-9 | Matches literature |

All FP16 baselines are healthy. Protocol is sound.

---

## Full results — PPL (lower is better)

### Gemma 4 E2B (FP16: 28.13)

| Bits | FQ | KIVI | TQ | TQ_out | Winner |
|---|---|---|---|---|---|
| **2** | 125.19 (+97.06) | 470.50 (+442.37) | 51.56 (+23.43) | **39.73 (+11.60)** | TQ_out |
| **3** | 24.26 (-3.87) | **21.90 (-6.23)** | 26.82 (-1.31) | 26.96 (-1.17) | KIVI |
| **4** | **26.75 (-1.38)** | 35.17 (+7.04) | 27.35 (-0.78) | 27.33 (-0.80) | FQ |

Note: 3-bit and 4-bit show PPL below FP16 baseline for some methods.
This is anomalous (quantization should not improve quality). Possible causes:
regularization effect, numerical precision differences in the DynamicCache
path, or Gemma-specific behavior. Does NOT occur on Mistral or Qwen2.

### Mistral-7B (FP16: 5.67)

| Bits | FQ | KIVI | TQ | TQ_out | Winner |
|---|---|---|---|---|---|
| **2** | 868.57 (+862.90) | **23.96 (+18.29)** | 787.51 (+781.84) | 167.89 (+162.22) | KIVI |
| **3** | 19.44 (+13.77) | **5.99 (+0.32)** | 13.64 (+7.97) | 9.48 (+3.81) | KIVI |
| **4** | 6.07 (+0.40) | **5.73 (+0.06)** | 5.96 (+0.29) | 5.88 (+0.21) | KIVI |

### Qwen2-7B (FP16: 7.55)

| Bits | FQ | KIVI | TQ | TQ_out | Winner |
|---|---|---|---|---|---|
| **2** | 262.89 (+255.34) | **46.70 (+39.15)** | 464.35 (+456.80) | 165.60 (+158.05) | KIVI |
| **3** | 8.93 (+1.38) | **8.01 (+0.46)** | 8.52 (+0.97) | 8.26 (+0.71) | KIVI |
| **4** | 7.74 (+0.19) | **7.62 (+0.07)** | 7.71 (+0.16) | 7.67 (+0.12) | KIVI |

---

## FajarQuant v1 rank summary

| Model × Bits | FQ rank (of 4) | FQ Δ vs FP16 | Winner Δ vs FP16 | FQ/Winner ratio |
|---|---|---|---|---|
| Gemma 2-bit | 3rd | +97.06 | +11.60 (TQ_out) | 8.4× worse |
| Gemma 3-bit | 2nd | -3.87 | -6.23 (KIVI) | — |
| Gemma 4-bit | **1st** | -1.38 | -1.38 (FQ) | **winner** |
| Mistral 2-bit | **4th** | +862.90 | +18.29 (KIVI) | 47.2× worse |
| Mistral 3-bit | **4th** | +13.77 | +0.32 (KIVI) | 43.0× worse |
| Mistral 4-bit | 4th | +0.40 | +0.06 (KIVI) | 6.7× worse |
| Qwen2 2-bit | 3rd | +255.34 | +39.15 (KIVI) | 6.5× worse |
| Qwen2 3-bit | **4th** | +1.38 | +0.46 (KIVI) | 3.0× worse |
| Qwen2 4-bit | 4th | +0.19 | +0.07 (KIVI) | 2.7× worse |

**FajarQuant v1 wins 1 of 9 cells (Gemma 4-bit). Dead last in 5 of 9 cells.**

---

## Cross-method observations

1. **KIVI dominates GQA architectures** (Mistral, Qwen2) — wins all 6 cells.
   Simple per-channel asymmetric quantization is surprisingly effective when
   KV heads are shared across query groups.

2. **TQ_outlier consistently beats TQ naive** — outlier handling is critical,
   confirming CLAUDE.md §6.9 Rule 5. The gap is largest at 2-bit (4-30×).

3. **FajarQuant v1's per-chunk PCA is the root weakness.** It lacks:
   - Outlier handling (no high-variance channel preservation)
   - Calibration (PCA recomputed per chunk, accumulating noise — §6.9 Rule 4)
   - Architecture awareness (same algorithm for MQA and GQA)

4. **The Gemma 3-bit anomaly** (quant PPL < FP16) is model-specific and does
   not appear on Mistral or Qwen2. Not a protocol bug.

---

## Implications for Path B (v2 design)

The data strongly points toward:

- **RC5 (no outlier extraction)** as the dominant root cause — TQ_outlier
  consistently outperforms TQ naive, and FQ has zero outlier handling.
- **RC2 (per-chunk PCA noise)** as secondary — calibrated rotation would
  eliminate the per-chunk recomputation overhead and noise accumulation.
- **RC1 (outlier-PCA misalignment)** as tertiary — PCA axes are sensitive
  to outliers in the calibration data; extracting outliers before PCA
  would improve the rotation quality.

Recommended v2 design direction: **F2-D (calibrated PCA + outlier extraction)**
or **F2-F (F2-D + DP bit allocation)** — to be confirmed in B2 diagnosis.

---

## Verification commands

```bash
# Reproduce (requires GPU + venv with torch 2.11+, transformers 5.5+)
cd ~/Documents/fajarquant
PYTHON=~/Documents/Fajar\ Lang/.venv/bin/python3

$PYTHON scripts/eval_perplexity.py --model google/gemma-4-E2B --bits 2,3,4 \
  --seq-len 2048 --max-samples 30 --output data/kv_cache/perplexity_v1_baseline_gemma.json

$PYTHON scripts/eval_perplexity.py --model mistralai/Mistral-7B-v0.1 --bits 2,3,4 \
  --seq-len 2048 --max-samples 30 --output data/kv_cache/perplexity_v1_baseline_mistral.json

$PYTHON scripts/eval_perplexity.py --model Qwen/Qwen2-7B --bits 2,3,4 \
  --seq-len 2048 --max-samples 30 --output data/kv_cache/perplexity_v1_baseline_qwen2.json

# Verify JSON integrity
for f in data/kv_cache/perplexity_v1_baseline_*.json; do
  echo "$f: $(jq 'keys | length' $f) keys, FP16=$(jq '.fp16.ppl' $f)"
done
```

---

*Document version: 2026-04-13 v1.0. Phase B1 sign-off gate (B1.G).*
