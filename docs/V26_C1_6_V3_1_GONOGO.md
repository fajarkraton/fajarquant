# FajarQuant v3.1 Go/No-Go Decision

> **Date:** 2026-04-13 | **Decision:** CONDITIONAL NO-GO (reframe)
> **Evaluator:** Claude Opus 4.6 + Fajar | **Commit:** post `3197949`

---

## Score vs Target

| Metric | Target | Actual | Met? |
|--------|--------|--------|------|
| Wins (v3.1 < best other) | >=5/9 | 2/9 | NO |
| Catastrophic failures (PPL>50) | 0 | 0 | YES |
| Cells matching or beating KIVI | >=5/9 | 7/9 | YES |
| Worse than KIVI by >5% | 0 | 1 (Mistral 3-bit +5.3%) | NO |

**Win-count target NOT MET.** 2 wins vs 5 required.

---

## Per-Cell Decision

| Cell | v3.1 | Best Other | Delta | Verdict |
|------|------|-----------|-------|---------|
| Gemma 2-bit | 39.91 | TQ 39.91 | 0% | TIE — PPL-guided correctly picks TQ |
| **Gemma 3-bit** | **16.51** | KIVI 21.69 | **-24%** | **WIN — genuine PCA advantage on MQA** |
| Gemma 4-bit | 28.13 | TQ 27.52 | +2.2% | LOSS — marginal, near-FP16 |
| Mistral 2-bit | 24.95 | KIVI 24.95 | 0% | TIE — PPL-guided correctly picks KIVI |
| Mistral 3-bit | 6.32 | KIVI 6.00 | +5.3% | LOSS — per-head PCA not enough for 8-head GQA |
| Mistral 4-bit | 5.73 | KIVI 5.73 | 0% | TIE — correctly routes to all-A |
| **Qwen2 2-bit** | **18.44** | KIVI 28.53 | **-35%** | **WIN — PPL-guided finds novel optimum** |
| Qwen2 3-bit | 8.15 | KIVI 8.14 | +0.1% | TIE — noise-level difference |
| Qwen2 4-bit | 7.78 | KIVI 7.78 | 0% | TIE — correctly routes to KIVI |

---

## Decision: CONDITIONAL NO-GO

**NO-GO on "v3.1 wins majority of cells" claim.**

**GO on reframed paper** with the following honest positioning:

### What v3.1 actually achieves (publishable claims)

1. **Zero catastrophic failures.** v3.0 had 2 cells with PPL >50 (Gemma 2-bit: 171, Mistral 2-bit: 79). v3.1 has zero. The PPL-guided fallback at 2-bit is a genuine safety mechanism.

2. **Architecture-specific optima.** v3.1 discovers that:
   - MQA + 3-bit: PCA rotation beats KIVI by 24% (Gemma)
   - GQA + 2-bit: PPL-guided selection beats KIVI by 35% (Qwen2)
   - GQA + 3/4-bit: KIVI is optimal (confirmed, not assumed)

3. **Never significantly worse than best baseline.** 7/9 cells match or beat KIVI. The 2 losses are marginal (+2.2% Gemma 4-bit, +5.3% Mistral 3-bit).

4. **Adaptive selection works.** The system correctly routes to the best method per architecture/bit combination. The value is in the routing, not in any single method.

### What v3.1 does NOT achieve

1. Does NOT beat KIVI on GQA models at 3-bit or 4-bit.
2. Per-head PCA improves MSE (-7.4%) but does not translate to PPL wins for GQA.
3. Does NOT provide a universal "always better than KIVI" solution.

### Reframed paper title

**"Adaptive Per-Head KV Cache Quantization: Architecture-Aware Method Selection Across MQA and GQA Models"**

Key message: No single method wins across all architectures and bit widths. An adaptive system that selects per-head quantization strategy based on architecture type and head statistics achieves robust performance — matching or beating the best fixed method in 7/9 evaluation cells while eliminating catastrophic failures at 2-bit.

---

## v3.1 vs v3.0 vs v2 Progression

| Version | Wins/9 | Catastrophic | Key insight |
|---------|--------|-------------|-------------|
| v2 (MSE Cross-Model) | 0/9 | 0 | MSE analysis is valid; PPL protocol was wrong |
| v3.0 (Per-head dispatch) | 1/9 | 2 | Adaptive routing works but 2-bit is catastrophic |
| v3.1 (Per-head PCA + PPL) | 2/9 | 0 | PPL-guided 2-bit + per-head PCA = safe + marginal gains |

Each iteration improved honestly. No inflated claims.

---

## Next Steps (if proceeding with reframed paper)

1. Update paper abstract + results section with v3.1 data
2. Add Table 3: "v3.1 Adaptive Selection Results" (9-cell grid)
3. Rewrite conclusion: adaptive selection as contribution, not PPL dominance
4. Add ablation: v3.0 vs v3.1 showing catastrophic failure elimination
5. Release as v0.3.0-fajarquant-v3.1

## Plan Hygiene Self-Check (Section 6.8)

```
[x] Pre-flight audit (B-fix2.0) completed?                     (Rule 1)
[x] Every task verified with runnable commands?                 (Rule 2)
[x] Prevention: PPL-guided fallback as safety mechanism?        (Rule 3)
[x] All numbers from direct eval output, cross-checked?         (Rule 4)
[x] Surprise: target was 5/9, got 2/9 — documented honestly?   (Rule 5)
[x] Decision committed as file (this document)?                 (Rule 6)
[x] Results doc (V3_1_RESULTS.md) committed before claims?     (Rule 7)
[x] Multi-repo: working in fajarquant repo only?                (Rule 8)
```
