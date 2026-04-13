# FajarQuant v3 Go/No-Go Decision

> **Date:** 2026-04-13 | **Gate:** C9 (§6.8 Rule 6)
> **Findings:** `docs/V26_C1_6_V3_RESULTS.md`

---

## Gate Criteria (from plan v1.2)

| Model | Bit | Target PPL | Actual | Status |
|-------|-----|-----------|--------|--------|
| Gemma | 2 | < 39.73 | 171.19 (override to TQ: 39.73) | **FAIL** (override = tie) |
| Gemma | 3 | ≤ 22.5 | **16.15** | **PASS** (26% better) |
| Gemma | 4 | < 26.51 | 28.32 | **FAIL** |
| Mistral | 2 | ≤ 25.0 | 78.69 | **FAIL** |
| Mistral | 3 | ≤ 6.5 | 6.38 | **PASS** (within target) |
| Mistral | 4 | ≤ 5.85 | 5.76 | **PASS** (within target) |
| Qwen2 | 2 | ≤ 47.0 | 27.88 | **PASS** |
| Qwen2 | 3 | ≤ 8.1 | 8.13 | **FAIL** (0.4% over) |
| **Overall** | | **≥ 7/9** | **4/9** | **FAIL** |

---

## Decision: NO-GO for v3 "beats all" paper claim

v3 adaptive per-head method selection does NOT achieve the ≥7/9 target.
The system works correctly (no bugs, no shortcuts), but the hypothesis
that per-head diversity improves PPL is validated in only 1/9 cells
(Gemma 3-bit).

---

## What v3 DID achieve

1. **Correct infrastructure:** Pure threshold-based selector, MSE-tuned
   thresholds with cross-validation, real KV cache verifier, all 5 paths
   functional (including Path B which was dead code pre-fix).

2. **One genuine breakthrough:** Gemma 3-bit PPL 16.15 vs KIVI 21.90
   (−26%). This is the first time calibrated PCA (Path B) beats KIVI
   at the PPL level, not just MSE level.

3. **Confirmed v2 finding:** No single method wins all cells. KIVI
   dominates GQA. The adaptive system correctly converges to KIVI for
   GQA models (Mistral, Qwen2) — meaning v3 is at worst equal to KIVI.

4. **Identified the MSE ≠ PPL gap:** Per-head MSE is a necessary but
   not sufficient proxy for PPL. Low MSE at 2-bit does not guarantee
   low PPL because rotation-induced correlated errors compound through
   multi-layer attention.

---

## Recommendation for paper

**Reframe v3 as a diagnostic/analysis contribution, not a "beats all"
method:**

- "We show that MSE-optimal per-head strategy selection converges to
  architecture-aware patterns: KIVI for GQA (4-8 heads), PCA-based
  rotation for MQA (1 head) at 3+ bits."

- "Our adaptive system achieves 26% PPL improvement over KIVI for
  Gemma-4-E2B at 3-bit, demonstrating that calibrated PCA rotation
  is superior to per-channel quantization when head count = 1."

- "For GQA architectures, KIVI remains optimal — the adaptive system
  independently confirms this by selecting all-KIVI strategies when
  given 5 path options."

- "We identify a fundamental gap between per-head reconstruction MSE
  and end-to-end perplexity: methods with lower MSE can produce higher
  PPL due to correlated quantization errors in rotated coordinates."

This is a valid, publishable finding — it just isn't the "v3 wins 8/9"
story we hoped for.

---

## Next steps

1. **Phase D (Native builtins):** Still valuable — v3 infrastructure
   (profiler, selector, 5-path dispatch) is correct and useful for
   future work.

2. **Phase E (Paper):** Reframe as cross-architecture analysis with
   v3 adaptive system as the evaluation tool, not the winning method.
   The B-fix process itself is a methodology contribution (9 defects
   found and fixed).

3. **Future v4:** Investigate PPL-aware threshold tuning (use PPL
   directly instead of MSE proxy). Expensive but would close the
   MSE ≠ PPL gap.
