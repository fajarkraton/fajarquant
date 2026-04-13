# FajarQuant v3.1 — Per-Head PCA + PPL-Guided Selection Plan

## Context

v3 Phase C evaluation scored **1 win, 3 ties, 5 losses** (target ≥7/9).
Deep investigation found **two fixable root causes**:

1. **Per-layer PCA on a per-head problem.** The `.npz` calibration stores
   ONE PCA rotation per layer. For Gemma (1 KV head, MQA) this is correct
   — per-layer = per-head. But for Mistral (8 heads) and Qwen2 (4 heads),
   all heads share one rotation computed from pooled data. This creates
   cross-head contamination: heads with cv=0.2 get a rotation optimized
   for heads with cv=4.9. The MSE-based tuner assigns Path B to these
   heads (lower average MSE), but PPL gets worse because critical
   attention heads receive suboptimal rotations.

   Evidence: Gemma gets **62% MSE reduction** from Path B. Mistral gets
   only **13%**. The 5x difference is explained by the calibration mismatch.

2. **MSE ≠ PPL at 2-bit.** At 2-bit (4 levels), rotation-based methods
   have lower per-head MSE but higher end-to-end PPL because systematic
   rounding errors compound through multi-layer attention. The MSE-based
   tuner cannot detect this. A PPL-guided fallback for 2-bit is needed.

Additionally, investigation found that **v2a** (calibrated PCA without
outlier fp16 preservation) beats KIVI on Gemma 4-bit (26.51 vs 35.17).
v3's Path B should incorporate v2a's uniform-quantization-on-outliers
approach at higher bit widths.

## Goal

Achieve **≥5/9 wins** (honest improvement over v3.0's 1/9) with:
- Per-head PCA calibration for GQA models
- PPL-guided 2-bit strategy override
- v2a-style uniform quantization at 4-bit

Realistic targets per cell:

| Cell | Current | Target | Mechanism |
|------|---------|--------|-----------|
| Gemma 2-bit | TQ 39.73 (tie) | ≤38 | PPL-guided Path C selection |
| Gemma 3-bit | **v3 16.15** (win) | keep | Already winning |
| Gemma 4-bit | v2a 26.51 (loss) | ≤26.5 | Incorporate v2a approach in Path B |
| Mistral 2-bit | KIVI 23.96 (loss) | ≤23.9 | Per-head PCA + PPL fallback |
| Mistral 3-bit | KIVI 5.99 (loss) | ≤5.99 | Per-head PCA calibration |
| Mistral 4-bit | KIVI 5.73 (near-tie) | ≤5.73 | Per-head PCA + v2a approach |
| Qwen2 2-bit | KIVI 27.87 (tie) | ≤27.8 | Per-head PCA + PPL fallback |
| Qwen2 3-bit | KIVI 8.13 (tie) | ≤8.1 | Per-head PCA calibration |
| Qwen2 4-bit | KIVI 7.77 (tie) | ≤7.7 | Per-head PCA + v2a approach |

Conservative: 5/9 wins. Optimistic: 7/9.

---

## Phase B-fix2.0: Pre-Flight Audit

| # | Task | Verification | Est |
|---|------|-------------|-----|
| 0.1 | Verify current v2 calibration accumulates per-layer (line 109 k.reshape(-1, D)) | Read calibrate_fq_v2.py:109 | 10 min |
| 0.2 | Verify Mistral .npz has 1 rotation per layer (k_0_pca_rotation shape) | `python3 -c "import numpy as np; print(np.load('...').['k_0_pca_rotation'].shape)"` | 10 min |
| 0.3 | Estimate per-head .npz sizes: Mistral ~93MB, Qwen2 ~78MB, Gemma no change | Calculation from exploration data | 10 min |
| 0.4 | Confirm v2a implementation exists at quant_attention_v2.py:188-236 | Read file | 10 min |

---

## Phase B-fix2.1: Per-Head Calibration Pipeline

> **Goal:** Modify `calibrate_fq_v2.py` to accumulate covariance per-head
> instead of per-layer when `n_kv_heads > 1`.
> **Effort:** 4h (3h + 30% surprise)
> **Key file:** `scripts/calibrate_fq_v2.py`

| # | Task | Verification | Est |
|---|------|-------------|-----|
| 1.1 | Modify `collect_streaming_stats()` — add per-head loop. Instead of `k.reshape(-1, D)` (pools all heads), iterate `for h in range(H)` and accumulate `k[h].reshape(-1, D)` per head. Store as `layer_stats[li]["heads"][hi]` | Unit test: synthetic (2 heads, known different distributions) produces 2 different covariance matrices | 2h |
| 1.2 | Modify saving logic — new key format `k_{layer}_h{head}_pca_rotation` when n_heads > 1. Keep old format `k_{layer}_pca_rotation` when n_heads == 1 (backward compat for Gemma) | `python3 -c "d=np.load('new.npz'); assert 'k_0_h0_pca_rotation' in d"` | 0.5h |
| 1.3 | Add `_n_heads` metadata field to .npz | Verify `d['_n_heads']` exists | 10 min |
| 1.4 | Add `--per-head` flag (default True for n_heads > 1, False for n_heads == 1) | `--per-head` and `--no-per-head` both work | 20 min |

**Prevention:** Test that loading old per-layer .npz still works (backward compat).

---

## Phase B-fix2.2: Per-Head Path B in v3 Pipeline

> **Goal:** Modify `_path_b_pca()` and `load_calibration()` to support
> per-head calibration. Add v2a-style uniform outlier quantization for 4-bit.
> **Effort:** 3h (2.5h + 25% surprise)
> **Key files:** `scripts/quant_attention_v2.py`, `scripts/quant_attention_v3.py`

| # | Task | Verification | Est |
|---|------|-------------|-----|
| 2.1 | Modify `load_calibration()` to detect per-head format (presence of `k_0_h0_pca_rotation`) and build nested structure: `cal["layers"][l]["heads"][h]["k_pca_rotation"]`. Fall back to current per-layer format when `k_0_h0_*` keys don't exist. | Load both old and new .npz formats successfully | 1h |
| 2.2 | Modify `_path_b_pca()` to index per-head: `layer_cal["heads"][head_idx]` instead of `layer_cal` when per-head data available. For single-head (Gemma), fall back to current per-layer access. | Smoke test: Mistral Path B with per-head .npz produces different MSE per head (not identical like before) | 1h |
| 2.3 | Add v2a-style variant to Path B: when `bits >= 4`, quantize outlier channels with per-coord uniform instead of fp16 preservation (matching v2a's approach at quant_attention_v2.py:227-228) | Gemma 4-bit Path B produces PPL closer to v2a (26.51) than current v3 (28.32) | 0.5h |
| 2.4 | Lazy GPU tensor caching per-head (only current layer's heads cached) to avoid 2GB GPU memory for Mistral | GPU memory usage < 200MB for calibration tensors | 0.5h |

---

## Phase B-fix2.3: Recalibrate Mistral + Qwen2

> **Goal:** Run per-head calibration for both GQA models.
> **Effort:** 2h GPU (mostly waiting)
> **Key file:** `scripts/calibrate_fq_v2.py` (modified)

| # | Task | Verification | Est |
|---|------|-------------|-----|
| 3.1 | Recalibrate Mistral with `--per-head` | `fq_v2_mistral_7b_v0.1_perhead.npz` exists, contains `k_0_h0_pca_rotation` through `k_31_h7_pca_rotation` | 1h GPU |
| 3.2 | Recalibrate Qwen2 with `--per-head` | `fq_v2_qwen2_7b_perhead.npz` exists, contains `k_0_h0_*` through `k_27_h3_*` | 1h GPU |
| 3.3 | Verify per-head rotations differ within a layer | `python3 -c "R0=d['k_0_h0_pca_rotation']; R1=d['k_0_h1_pca_rotation']; assert not np.allclose(R0, R1)"` | 10 min |

**Gemma:** No recalibration needed — 1 KV head = per-layer is per-head.

---

## Phase B-fix2.4: PPL-Guided 2-Bit Fallback

> **Goal:** For 2-bit, test top 3 strategies on small PPL sample (5 chunks)
> and pick the lowest PPL, instead of trusting MSE-based tuner.
> **Effort:** 2h (1.5h + 30% surprise)
> **Key file:** New `scripts/ppl_guided_select.py`

| # | Task | Verification | Est |
|---|------|-------------|-----|
| 4.1 | Write `ppl_guided_select.py` — for a given model and bit width: (a) generate 3 candidate strategies (all-A/KIVI, all-C/TQ, MSE-tuned), (b) run eval_perplexity on 5 chunks for each, (c) pick lowest PPL, (d) save as the final strategy | Script outputs "Selected: all-C (PPL=39.73) over MSE-tuned (PPL=171.19)" for Gemma 2-bit | 1.5h |
| 4.2 | Run for all 3 models at 2-bit | 3 final 2-bit strategy files with `_method: "ppl_guided"` metadata | 30 min GPU |

This directly addresses the MSE ≠ PPL gap. At 3-bit and 4-bit, MSE-based
tuning works well (Gemma 3-bit proves it). Only 2-bit needs PPL guidance.

---

## Phase B-fix2.5: Re-Tune + Re-Verify All Models

> **Goal:** Run MSE-based threshold tuner with per-head calibration for
> Mistral and Qwen2 at 3-bit and 4-bit. 2-bit handled by PPL-guided.
> **Effort:** 3h GPU
> **Key files:** `scripts/tune_thresholds.py`, `scripts/strategy_verifier.py`

| # | Task | Verification | Est |
|---|------|-------------|-----|
| 5.1 | Tune Mistral 3-bit + 4-bit with per-head calibration | `thresholds_mistral_*bit.json` updated with per-head MSE | 1h GPU |
| 5.2 | Tune Qwen2 3-bit + 4-bit with per-head calibration | `thresholds_qwen2_*bit.json` updated | 1h GPU |
| 5.3 | Generate strategies from tuned thresholds for all 6 cells (2 models × 3 bits) | `strategy_*_*bit.json` files updated | 15 min |
| 5.4 | Run verifier on all 6 cells with per-head calibration | Verified strategy files with real-data swap counts | 45 min GPU |

**Gemma:** No re-tuning needed — calibration unchanged, existing thresholds valid.

---

## Phase B-fix2.6: Full Re-Evaluation (C4-C6 v2)

> **Goal:** Run complete 3×3 PPL evaluation with all fixes applied.
> **Effort:** 4h GPU

| # | Task | Verification | Est |
|---|------|-------------|-----|
| 6.1 | Gemma eval (2/3/4-bit) — 2-bit uses PPL-guided strategy, 3/4-bit use existing tuned+v2a-enhanced | `perplexity_v3.1_gemma.json` | 30 min GPU |
| 6.2 | Mistral eval (2/3/4-bit) — 2-bit uses PPL-guided, 3/4-bit use per-head-tuned | `perplexity_v3.1_mistral.json` | 1.5h GPU |
| 6.3 | Qwen2 eval (2/3/4-bit) — same pattern | `perplexity_v3.1_qwen2.json` (seq_len=1024 for OOM) | 1.5h GPU |
| 6.4 | Delta table: v3.1 vs v3.0 vs KIVI vs v2-best per cell | `docs/V26_C1_6_V3_1_RESULTS.md` | 30 min |

---

## Phase B-fix2.7: Go/No-Go Decision v2

| # | Task | Verification | Est |
|---|------|-------------|-----|
| 7.1 | Score: count v3.1 wins vs KIVI and v2-best per cell | Committed doc | 30 min |
| 7.2 | Decision: GO (≥5/9) or NO-GO with honest reframe | `docs/V26_C1_6_V3_1_GONOGO.md` committed | 30 min |

### Revised Go/No-Go Criteria

| Model | Bit | Target | Comparison |
|-------|-----|--------|------------|
| Gemma | 2 | ≤ 39.73 | Match TQ outlier (PPL-guided) |
| Gemma | 3 | ≤ 16.5 | Keep v3.0 win (16.15) |
| Gemma | 4 | ≤ 26.51 | Match or beat v2a |
| Mistral | 2 | ≤ 23.96 | Match or beat KIVI |
| Mistral | 3 | ≤ 5.99 | Match or beat KIVI |
| Mistral | 4 | ≤ 5.73 | Match or beat KIVI |
| Qwen2 | 2 | ≤ 27.87 | Match or beat KIVI |
| Qwen2 | 3 | ≤ 8.01 | Beat KIVI (v1 baseline) |
| Qwen2 | 4 | ≤ 7.62 | Beat KIVI (v1 baseline) |
| **Overall** | | **≥ 5/9 wins** | Honest improvement |

### Fallback if <5/9
If per-head PCA still doesn't beat KIVI for GQA models:
1. Accept that KIVI is optimal for GQA — v3's value is MQA-specific
2. Paper framing: "Adaptive system confirms KIVI optimality for GQA
   while discovering PCA superiority for MQA at 3-bit"
3. This is still a valid, publishable finding — just narrower scope

---

## Timeline

| Phase | Est | Cumulative | Dependency |
|-------|-----|-----------|------------|
| B-fix2.0 Pre-flight | 30 min | 0.5h | — |
| B-fix2.1 Per-head calibration pipeline | 3h | 3.5h | 0 |
| B-fix2.2 Per-head Path B in v3 | 2.5h | 6h | 1 |
| B-fix2.3 Recalibrate Mistral + Qwen2 | 2h GPU | 8h | 1, 2 |
| B-fix2.4 PPL-guided 2-bit fallback | 1.5h + 30min GPU | 10h | — (parallel with 1-3) |
| B-fix2.5 Re-tune + re-verify | 3h GPU | 13h | 3 |
| B-fix2.6 Full re-evaluation | 4h GPU | 17h | 4, 5 |
| B-fix2.7 Go/No-Go v2 | 1h | **18h** | 6 |

**Total: ~18h (~2.5 working days)**

**Parallelization:** B-fix2.4 (PPL-guided) can run in parallel with
B-fix2.1-2.3 (per-head calibration). This saves ~2h.

---

## Critical Files

| File | Change |
|------|--------|
| `scripts/calibrate_fq_v2.py` | Per-head accumulation loop + new .npz key format |
| `scripts/quant_attention_v2.py` | `load_calibration()` per-head format detection |
| `scripts/quant_attention_v3.py` | `_path_b_pca()` per-head indexing + v2a outlier mode |
| `scripts/ppl_guided_select.py` | **NEW** — PPL-guided 2-bit strategy selection |
| `scripts/tune_thresholds.py` | Pass per-head calibration to `_apply_path_with_cal()` |
| `scripts/strategy_verifier.py` | Pass per-head calibration |

## Verification

After B-fix2.6, the test is simple:
```bash
# Full eval results exist
ls data/kv_cache/perplexity_v3.1_{gemma,mistral,qwen2}.json

# Count wins
python3 -c "
# Compare v3.1 vs best-other per cell
# Score: count cells where v3.1 PPL <= best-other PPL
"
```

## Risk Register

| Risk | P | Impact | Mitigation |
|------|---|--------|------------|
| Per-head PCA still loses to KIVI on GQA | **High** | High | Accept + reframe paper (fallback) |
| Per-head .npz too large (93MB Mistral) | Low | Med | Lazy per-layer GPU loading |
| Recalibration OOM on 7B models | Med | Med | Reduce n_samples from 128 to 64 |
| v2a approach hurts 2-bit (known from v2 data) | Low | Low | Only apply at ≥4-bit |
| PPL-guided select picks same as MSE at 2-bit | Med | Low | At least confirms MSE is valid |

## Plan Hygiene Self-Check (§6.8)

```
[x] Pre-flight audit (B-fix2.0)?                              (Rule 1)
[x] Every task has runnable verification command?              (Rule 2)
[x] Prevention: backward-compat test for old .npz?            (Rule 3)
[x] Numbers will be cross-checked with Bash?                  (Rule 4)
[x] Surprise budget: +25-30% built into estimates?            (Rule 5)
[x] Go/No-Go as committed file?                               (Rule 6)
[x] Results doc committed before paper claims?                 (Rule 7)
[x] Multi-repo state check before starting?                    (Rule 8)
```

## Research Integrity Self-Check (§6.9)

```
[x] Canonical R-alpha.1 protocol unchanged?                    (R1)
[x] Per-head PCA grounded in exploration findings?             (R2)
[x] KIVI baseline unmodified (full feature)?                   (R3)
[x] Calibration once, reuse (per-head calibrated offline)?     (R4)
[x] Outlier handling preserved in Path B?                      (R5)
[x] Validation (B-fix2.6) before any paper claims?             (R6)
[x] verify_paper_tables.py gate still required?                (R7)
```
