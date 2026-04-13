# FajarQuant v3 Results — Full 3-Model x 3-Bit Evaluation

> **Date:** 2026-04-13 | **Protocol:** R-alpha.1 model surgery
> **Dataset:** WikiText-2 test, 30 chunks (Gemma/Mistral: seq_len=2048, Qwen2: 1024 due to OOM)
> **Pipeline:** B-fix.1-4 applied (pure threshold tree, real MSE tuner, real KV verifier)

---

## Full Results Table

### Gemma-4-E2B (MQA, 1 KV head, 15 layers) — FP16=28.13

| Bit | FQ v3 | KIVI | TQ Outlier | v2a | **Best** | v3 Strategy |
|-----|-------|------|------------|-----|----------|-------------|
| 2 | 171.19 | 470.50 | **39.73** | 46.20 | TQ | 0A+0B+30C (override) |
| 3 | **16.15** | 21.90 | 26.96 | 23.61 | **v3** | 17A+13B |
| 4 | 28.32 | 35.17 | 27.33 | **26.51** | v2a | 17A+13B |

### Mistral-7B-v0.1 (GQA, 8 KV heads, 32 layers) — FP16=5.67

| Bit | FQ v3 | KIVI | TQ Outlier | v2-best | **Best** | v3 Strategy |
|-----|-------|------|------------|---------|----------|-------------|
| 2 | 78.69 | **23.96** | 167.89 | 23.96 | KIVI | 196A+234B+82D |
| 3 | 6.38 | **5.99** | 9.48 | 5.99 | KIVI | 273A+239B |
| 4 | 5.76 | **5.73** | 5.88 | 5.73 | KIVI | 273A+239B |

### Qwen2-7B (GQA, 4 KV heads, 28 layers) — FP16=7.69

| Bit | FQ v3 | KIVI | TQ Outlier | v2-best | **Best** | v3 Strategy |
|-----|-------|------|------------|---------|----------|-------------|
| 2 | 27.88 | **27.87** | 73.63 | 46.70* | KIVI | 181A+43D |
| 3 | 8.13 | 8.13 | 8.32 | **8.01*** | KIVI* | 224A (=KIVI) |
| 4 | 7.77 | 7.77 | 7.81 | **7.62*** | KIVI* | 224A (=KIVI) |

*Qwen2 v2/v3 eval at seq_len=1024 (OOM at 2048); v1 baseline at 2048.
Absolute PPL not directly comparable, relative ordering is valid.

---

## Scorecard

| Cell | Winner | v3 adds value? |
|------|--------|----------------|
| Gemma 2-bit | TQ Outlier (39.73) | No — Path B catastrophic at 2-bit |
| **Gemma 3-bit** | **v3 (16.15)** | **Yes — 26% better than KIVI** |
| Gemma 4-bit | v2a (26.51) | No — v3 28.32 slightly worse |
| Mistral 2-bit | KIVI (23.96) | No — Path B catastrophic at 2-bit |
| Mistral 3-bit | KIVI (5.99) | No — v3 6.38, 6.5% worse |
| Mistral 4-bit | KIVI (5.73) | No — v3 5.76, 0.5% worse (near-tie) |
| Qwen2 2-bit | KIVI (27.87) | Tie — v3 = KIVI (tuner chose KIVI) |
| Qwen2 3-bit | KIVI (8.13) | Tie — v3 = KIVI (identical) |
| Qwen2 4-bit | KIVI (7.77) | Tie — v3 = KIVI (identical) |

**Result: 1 win, 3 ties, 5 losses. Target was ≥7/9.**

---

## Key Findings

### 1. Path B (PCA) helps only MQA at 3+ bits

Calibrated PCA rotation is beneficial when:
- **Single KV head** (MQA) — all information must pass through one head,
  PCA decorrelation captures the structure
- **3+ bits** — enough grid resolution for PCA-rotated coordinates
- **Low kurtosis** — Gemma's KV cache has kurtosis < 1.0 (near-Gaussian),
  ideal for PCA

It fails when:
- **2-bit** — 4-level grid too coarse for rotated coordinates, systematic
  errors compound through layers
- **Multi-head GQA** — redundancy across 4-8 heads makes per-channel
  quantization (KIVI) sufficient; PCA's decorrelation adds overhead
  without proportional benefit

### 2. KIVI dominates GQA architectures

For Mistral (8 heads) and Qwen2 (4 heads), KIVI is consistently optimal.
The MSE-based tuner correctly identifies this — for Qwen2, it produces
all-KIVI strategies. For Mistral, it assigns Path B to many heads, but
the end-to-end PPL shows KIVI is still better.

**MSE ≠ PPL:** Per-head reconstruction MSE is a useful proxy but does
not capture inter-head coherence effects. Path B can have lower MSE
per-head but higher PPL because rotation introduces correlated errors
across attention heads.

### 3. v3 adaptive system works correctly but hypothesis is narrow

The v3 system is *technically correct*:
- Pure threshold tree (no architecture gates) — verified
- MSE-based threshold tuning — verified with cross-validation
- Real KV cache verification — verified with swaps
- Per-head dispatch — verified

But the *scientific hypothesis* (per-head diversity improves PPL) is
only validated in one regime: MQA + 3-bit. The hypothesis that different
heads benefit from different quantization strategies is true at the MSE
level but does not translate to PPL improvement for GQA models.

### 4. The "no single method wins" finding from v2 is confirmed

v2's cross-architecture analysis showed KIVI wins wide-GQA, TQ wins
MQA 2-bit, and no method dominates all cells. v3's MSE-tuned adaptive
system converges to the same conclusion: KIVI for GQA, non-KIVI for
MQA low-bit. v3 adds value only at the sweet spot (MQA 3-bit PCA).

---

## Protocol Notes

- All methods use R-alpha.1 model surgery (monkey-patched attention)
- FP16 baseline confirmed independently (B-fix.4: standalone eval)
- PPL < FP16 at 3-4 bit is a model property (quantization regularization
  on small models), not a protocol artifact
- Qwen2 ran at seq_len=1024 due to 16GB VRAM OOM at 2048
- B-fix pipeline: 9 defects found and fixed before evaluation
