# FajarQuant v3 Design Decision — B.G.3 Gate Document

> **Date:** 2026-04-13 | **Gate:** B.G.3 (mechanical, §6.8 Rule 6)
> **Decision:** CONDITIONAL GO — 3-bit validated, 2-bit needs remediation
> **Prerequisite for:** Phase C5 (Mistral), C6 (Qwen2)

---

## B.G Gate Results

### B.G.1: v3 post-fix vs v2-best (Gemma only)

| Bit | v3 post-fix | v2-best | Delta | Status |
|-----|-------------|---------|-------|--------|
| 2 | 171.19 | 39.73 (TQ outlier) | +131.46 | **FAIL** |
| 3 | **16.15** | 21.90 (KIVI) | **−5.75** | **PASS** |
| 4 | 28.32 | 26.51 (v2a) | +1.81 | **FAIL** |

### B.G.2: Path diversity (≥2 paths per model)

| Bit | Unique paths | Status |
|-----|-------------|--------|
| 2 | 3 (16A + 13B + 1C) | PASS |
| 3 | 2 (17A + 13B) | PASS |
| 4 | 2 (17A + 13B) | PASS |

### B.G.3: This document.

---

## Root Cause: 2-bit Path B Failure

The MSE-based threshold tuner (B-fix.2) correctly identifies Path B as
having lower per-head reconstruction MSE than Path A at 2-bit:
- Default (all-A) MSE: 0.577
- Optimal (16A+13B+1C) MSE: 0.232 (−60%)

But end-to-end PPL tells the opposite story:
- Default (all-A = KIVI): PPL 470 (catastrophic)
- Pre-fix (all-C = TQ): PPL 39.73 (good)
- Post-fix (16A+13B+1C): PPL 171.19 (bad)

**Explanation:** At 2-bit, the quantization grid has only 4 levels
(0,1,2,3). PCA rotation concentrates information into principal
components, then quantizes. But with only 4 levels, the rotation's
benefit (decorrelation) is overwhelmed by systematic rounding errors
in the rotated space. These errors are coherent (not random noise) and
compound through multi-layer attention.

TQ outlier (Path C) avoids this by using Hadamard rotation (which
spreads, not concentrates) + outlier preservation. At 2-bit, spreading
outliers is more important than decorrelation.

At 3-bit (8 levels), PCA has enough grid resolution to benefit from
decorrelation → v3 (16.15) dramatically beats KIVI (21.90).

---

## Remediation: Per-Bit-Width Strategy Override

**Decision:** For 2-bit, override the MSE-tuned thresholds with a
Path-C-dominant strategy. The adaptive system's value is at 3+ bits.

Implementation (in eval pipeline, not selector):
1. Load bit-width-specific strategy files (already per-bit)
2. For 2-bit: force `T_kurt` to 0.0 (everything triggers Path C first)
   OR: use pre-fix strategy (all Path C) as 2-bit override
3. For 3/4-bit: use MSE-tuned thresholds (proven to work)

This means v3 at 2-bit degenerates to TQ outlier for Gemma — which is
the honest result. v3's per-head adaptive value is in the 3+ bit regime.

---

## 4-bit Assessment

v3 post-fix (28.32) is 1.81 worse than v2a (26.51). This is because:
- v2a uses calibrated PCA on ALL heads (full v2 pipeline)
- v3 uses A+B mix where Path A (KIVI) heads get 35.17 PPL, dragging
  the average up

The 4-bit threshold tuner found the same T_cv=3.0 / T_svd=1.2 as
3-bit. But at 4-bit, even the "low-variance" heads that go to Path A
have enough bits for KIVI to work OK (35.17). The improvement from
switching those heads to Path B might not compensate for the heads
where Path A is genuinely better.

**Assessment:** 4-bit is marginal. v3 adds little value over v2a at
4-bit for Gemma. Accept or retry with per-bit threshold tuning.

---

## Honest v3 Gemma Scorecard

| Bit | Best Method | PPL | v3 adds value? |
|-----|-------------|-----|----------------|
| 2 | TQ outlier (Path C) | 39.73 | **No** — v3 = TQ for 2-bit (remediation) |
| 3 | **v3 adaptive (A+B)** | **16.15** | **Yes — 26% better than KIVI** |
| 4 | v2a (calibrated PCA) | 26.51 | **Marginal** — v3 close but doesn't beat v2a |

**Gemma contribution to overall 7/9 target:** 1 clear win (3-bit),
1 tie (2-bit after remediation), 1 marginal loss (4-bit).

---

## Decision for C5/C6

**CONDITIONAL GO.** Proceed to Mistral and Qwen2 evaluation with:

1. **2-bit:** Use Path-C-dominant strategy (T_kurt=0.0 override) for
   any model where MSE-tuned thresholds assign >30% Path B at 2-bit.
   Run 2-bit PPL comparison BEFORE committing to tuned thresholds.
2. **3-bit:** Use MSE-tuned thresholds (proven on Gemma).
3. **4-bit:** Use MSE-tuned thresholds, compare to v2a baseline.
4. **For Mistral/Qwen2:** Before eval, fix D7 (Mistral forward missing
   sliding_window) and D8 (layer_idx mismatch for shared-KV).

Mistral (8 KV heads, GQA) and Qwen2 (4 KV heads, narrow-GQA) have
genuine per-head diversity opportunity — v3 should perform better than
on Gemma (1 head, MQA) where per-head adaptation has no leverage.
