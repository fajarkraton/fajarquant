# FajarQuant v3 Phase B-fix Findings

> **Date:** 2026-04-13 | **Triggered by:** C4 Gemma results audit
> **Rule compliance:** CLAUDE.md §6.8 Rule 1 (pre-flight audit), §6.9 R1 (canonical protocol)
> **Status:** 5 defects found, 0 fixed

---

## Summary

C4 Gemma v3 perplexity evaluation revealed that v3's "adaptive per-head
method selection" produces **zero improvement** over running TQ outlier
uniformly. Root cause: the entire B2 pipeline (selector, auto-tuner,
verifier) contains circular reasoning, synthetic proxies, and hardcoded
architecture gates that prevent genuine per-head adaptation.

This is the **same class of error** as FajarQuant v1→v2 (§6.9 post-mortem):
custom "convenience" protocols that bypass actual data produce misleading
results. The B.G gate that should have caught this was skipped.

---

## C4 Gemma Results (for reference)

| Bit | FQ v3 | KIVI | TQ Outlier | FP16 |
|-----|-------|------|------------|------|
| 2 | 39.73 | 470.50 | 39.73 | 28.13 |
| 3 | 25.17 | 21.90 | 26.96 | 28.13 |
| 4 | 27.31 | 35.17 | 27.33 | 28.13 |

**Observation:** v3 2-bit = TQ 2-bit exactly (0.0000 difference). v3 4-bit
vs TQ 4-bit = 0.016 (noise). v3 3-bit worse than KIVI by 3.27 PPL.

---

## Defect 1: `strategy_selector.py` — Hardcoded Architecture Gates

**File:** `scripts/strategy_selector.py:59-78`
**Plan requirement:** Threshold-based per-head decision tree (5 paths)
**Actual behavior:** Architecture-aware gates bypass the decision tree entirely

```python
# MQA: never KIVI, prefer rotation-based methods
if is_mqa:                          # n_kv_heads <= 1
    if kurt > t["T_kurt"] * 0.5:   # threshold halved → nearly always True
        return "C"
    if svd > t["T_svd"] * 0.5:
        return "B"
    if bits == 2:
        return "C"
    return "C"                      # ALL fallbacks → C
```

**Impact:** For Gemma (MQA, 1 head), **every head is forced to Path C
regardless of statistics**. The threshold halving (`* 0.5`) makes it
trivially easy to trigger C (kurtosis > 3.0 for T_kurt=6.0), and even if
it doesn't, all fallbacks return C. Similarly, wide-GQA forces Path A.

**Evidence:** Gemma 2-bit: 30/30 heads = Path C. Gemma kurtosis range:
−0.13 to 0.97 (all BELOW T_kurt=6.0 and even below T_kurt*0.5=3.0 for
keys). Yet all assigned C because the MQA gate catches them before the
threshold tree.

Only narrow-GQA (2-4 heads, i.e., Qwen2) actually uses the plan's
threshold-based decision tree.

**Root cause:** Prior empirical observation (KIVI PPL=470 on Gemma 2-bit)
was hardcoded as an architectural rule, eliminating adaptive behavior.
This makes v3 a **lookup table**, not an adaptive system.

---

## Defect 2: `tune_thresholds.py` — Circular Scoring Function

**File:** `scripts/tune_thresholds.py:56-83`
**Plan requirement (B2.T):** "Grid search over thresholds on **calibration
data** with **80/20 cross-val**"
**Actual behavior:** Scoring uses path-count heuristics, no calibration data,
no cross-validation

```python
if n_kv_heads <= 1:
    # MQA: want Path C → score = %C
    return path_counts["C"] / total * 100
elif n_kv_heads >= 8:
    # Wide-GQA: want Path A → score = %A
    return path_counts["A"] / total * 100
```

**Impact:** Auto-tuner **defines** the optimal answer instead of measuring it.
For Gemma: optimal = 100% Path C (score=100.0). Searched 625 grid
combinations, all produced the same result because the scoring function
rewards maximizing C regardless of actual quantization quality.

**Evidence:**
- `thresholds_gemma_2bit.json`: default_score = optimal_score = 100.0
- optimal_thresholds = default_thresholds (grid search found nothing better)

**What plan required:** MSE or PPL on real calibration data with train/test
split. Score = reconstruction quality, not path distribution.

---

## Defect 3: `strategy_verifier.py` — Synthetic Data Instead of Real KV Cache

**File:** `scripts/strategy_verifier.py:113-115`
**Plan requirement (B2.V):** "Mini-batch PPL check per head, swap if better
strategy found" with pseudocode `best = argmin([mse(path, head_data)])`
**Actual behavior:** Uses random Gaussian noise as proxy for head data

```python
synthetic = torch.randn(1, 1, 64, head_dim) * mean_abs
```

**Impact:** Random Gaussian data has no outliers, no inter-channel
correlations, no asymmetric distributions — the exact properties that
differentiate paths A-E. MSE on Gaussian data is nearly identical for
all paths. Result: **0 swaps** out of 30 heads.

**Evidence:**
- `strategy_gemma_2bit_verified.json`: swaps=0/30, swap_rate=0.0

**What plan required:** Real KV cache chunks from calibration data (the
`.npz` files contain this). Path B (PCA) is also skipped in verifier
(`paths_to_test = ["A", "C", "D", "E"]  # Skip B`).

---

## Defect 4: B.G Gate Skipped

**Plan requirement:** Three gates before C-phase validation:
1. B.G.1: v3 PPL < v2 PPL on same args (smoke test)
2. B.G.2: ≥2 different paths used across all models
3. B.G.3: Decision file `docs/V26_C1_6_V3_DESIGN.md` committed

**Actual:** All three skipped. C4 was run directly after B8 commit.

**Evidence:**
- No `perplexity_v3_*` files existed before C4 run → B.G.1 never ran
- Gemma 2-bit uses 1 path only → B.G.2 would have FAILED
- `docs/V26_C1_6_V3_DESIGN.md` does not exist → B.G.3 never created

**Impact:** The gate would have caught Defect 1 (Gemma single-path) before
spending GPU hours on full evaluation.

---

## Defect 5: PPL < FP16 Anomaly (Pre-existing, Uninvestigated)

**Observation:** Across v1, v2, and v3, multiple quantized methods produce
PPL **below** FP16 baseline on Gemma:

| Version | Methods below FP16 (28.13) | Bit widths |
|---------|---------------------------|------------|
| v1 | 7 methods (KIVI 3-bit=21.90, FQ 3-bit=24.26, ...) | 3-bit, 4-bit |
| v2 | 9 methods (FQ-v2a 3-bit=23.61, ...) | 3-bit, 4-bit |
| v3 | 5 methods (v3 3-bit=25.17, KIVI 3-bit=21.90, ...) | 3-bit, 4-bit |

**Impact:** Quantized KV cache producing lower PPL than full-precision is
physically anomalous — lossy compression should not improve quality.
Possible explanations (unverified):
1. DynamicCache code path differs between patched/unpatched model
2. Quantization acts as regularization on this small model (unlikely)
3. Evaluation protocol artifact (all methods share same anomaly)

This has been present since v1 and was never investigated. It does not
invalidate relative comparisons (all methods share the same baseline), but
it means absolute PPL values cannot be compared to published numbers from
other papers.

**Required:** Standalone FP16 evaluation (separate script, no
patch/unpatch machinery) to verify baseline. If anomaly persists, document
as known limitation.

---

## Severity Assessment

| Defect | Severity | Impact on v3 Claim |
|--------|----------|-------------------|
| D1: Architecture gate | **CRITICAL** | v3 = lookup table, not adaptive system |
| D2: Circular tuner | **HIGH** | Auto-tuning produces no signal |
| D3: Synthetic verifier | **HIGH** | Verifier cannot detect wrong assignments |
| D4: Gate skipped | **MEDIUM** | Would have caught D1 before GPU spend |
| D5: PPL < FP16 | **LOW** | Pre-existing; doesn't affect relative comparisons |

---

## Required Fixes (to be detailed in plan update)

1. **D1 fix:** Remove architecture gates from `strategy_selector.py`. Use
   ONLY the threshold-based decision tree. Let real data (via fixed D2/D3)
   determine which path wins for each architecture.

2. **D2 fix:** Replace scoring function with real reconstruction MSE on
   KV cache data from calibration `.npz`. Implement actual 80/20
   cross-validation as plan requires.

3. **D3 fix:** Replace `torch.randn` with real KV cache chunks from
   calibration `.npz`. Re-enable Path B in the test matrix.

4. **D4 fix:** Execute B.G gate mechanically before C5/C6. Commit
   `docs/V26_C1_6_V3_DESIGN.md` with honest Gemma C4 results and
   decision rationale.

5. **D5 fix:** Run standalone FP16 baseline script. Document finding.
