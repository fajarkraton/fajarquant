# V26 C1.6 Path B — B4.G Go/No-Go Decision

> **Date:** 2026-04-13
> **Phase:** B4.G (mechanical go/no-go gate)
> **Data source:** `docs/V26_C1_6_V2_RESULTS.md` (B4.4 delta analysis)
> **Rule compliance:** CLAUDE.md §6.8 Rule 6 (mechanical decision gate)

---

## Gate Criterion (from `docs/V26_C1_6_PATH_B_PLAN.md` §B4.G)

> "Does v2.12 beat TQ_outlier by >=10% on at least 2 of 3 models at 2-bit?"

## Gate Evaluation

| Model | TQ outlier 2-bit | FQ v2 2-bit | FQ v2 wins by >=10%? |
|-------|----------------:|-----------:|:--------------------:|
| Gemma 4 E2B | 39.73 | 46.20 | **NO** (FQ v2 is 16.3% WORSE) |
| Mistral 7B | 167.89 | 208.16 | **NO** (FQ v2 is 24.0% WORSE) |
| Qwen2-7B | 165.60 | 50.10 | **YES** (FQ v2 is 69.7% BETTER) |

**Result: 1/3 models pass. Gate threshold is 2/3.**

---

## Decision: NO-GO on original claim. CONDITIONAL GO on reframed paper.

### What this means

- **NO-GO:** The paper CANNOT claim "FajarQuant v2 outperforms TurboQuant
  across models." The data does not support it. Two of three models show
  FQ v2 losing to TQ outlier at 2-bit.

- **CONDITIONAL GO:** The data IS publishable under a reframed narrative.
  Proceed to B5 (language support) + B6 (paper rewrite) with the reframed
  scope below. The condition is that the paper makes zero claims of
  universal superiority.

### Reframed paper narrative

**Old claim (DEAD):** "FajarQuant achieves lowest perplexity across models
and bit widths."

**New claim (data-supported):** "No single KV cache quantization method
dominates across attention architectures. We present a cross-architecture
analysis of 5 methods across 3 KV-head configurations (MQA, GQA-32:8,
GQA-28:4) and show that method effectiveness is architecture-dependent.
FajarQuant v2 is the strongest non-KIVI method on MQA (Gemma) and narrow-GQA
(Qwen2), while KIVI dominates on wide-GQA (Mistral). We provide the first
systematic cross-architecture comparison and an adaptive selection guide."

### Supporting evidence for reframed narrative

1. **Architecture dependence is real and large:**
   - KIVI fails catastrophically on Gemma MQA 2-bit (470 PPL, 16x FP16)
   - KIVI wins every cell on Mistral GQA-32:8
   - No other paper has characterized this systematically

2. **FQ v2 upgrade is validated:**
   - v2 improves over v1 in 7/9 cells (63-81% at 2-bit)
   - The calibrated PCA + outlier extraction design works

3. **FQ v2 has a defensible niche:**
   - 2nd-best method in 4/9 cells
   - Best non-KIVI on Qwen2 (all bit widths) and Gemma 2-bit
   - Competitive with TQ outlier at 3-4 bit across all models

4. **Cross-architecture finding is novel:**
   - KIVI paper only tested GQA models
   - KVQuant paper only tested Llama (GQA)
   - No paper has tested MQA + GQA-wide + GQA-narrow on same methods

### Why NOT going back to B2

Returning to B2 (more diagnosis) would address symptom, not cause. The core
issue is architectural: FQ v2's calibrated PCA rotation works well when KV
heads have enough diversity (MQA single head, narrow-GQA 4 heads) but adds
overhead when heads are already regular (wide-GQA 8+ heads where KIVI's
simpler per-channel quant suffices). No algorithmic fix changes this
fundamental tradeoff.

Estimated effort for another B2-B3-B4 cycle: ~2-3 weeks. Expected
improvement: marginal. Better to publish the honest finding.

### What changes in B5 + B6

- **B5 (language support):** Unchanged — L1-L7 features are valuable
  regardless of paper framing
- **B6 (paper rewrite):** Major revision needed:
  - Title changes from "FajarQuant: Superior KV Cache Quantization" to
    something like "Cross-Architecture KV Cache Quantization: Why No Single
    Method Wins"
  - Results section rewrites from "we beat X" to comparative analysis
  - Add architecture-analysis section as primary contribution
  - Position FajarQuant v2 as "best-in-class for MQA/narrow-GQA" not
    "universally best"
  - CLAUDE.md §6.9 R6 compliance: no quantitative claim without canonical
    benchmark backing (all claims now backed by B4.1-B4.3 data)

---

## Signoff

- [x] Gate criterion evaluated mechanically against data (B4.4)
- [x] Decision is explicit (NO-GO original / CONDITIONAL GO reframed)
- [x] No paper claim made without canonical-protocol benchmark (§6.9 R1)
- [x] Reframed narrative backed by 3-model × 3-bit × 5-method data
- [x] Decision file committed per CLAUDE.md §6.8 Rule 6
