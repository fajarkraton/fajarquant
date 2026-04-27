# Phase E E2.4 — Bilingual Calibration Decision (v1.0)

> **Status:** E2.4.C.4 decision FINAL. Phase E2.4 exits with a HONEST NEGATIVE RESULT — `balanced_calib` is NOT adopted as a Phase E2 ablation feature.
>
> **Plan reference:** `FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.8 §3 PHASE E2.4 + `FJQ_PHASE_E_E2_4_FINDINGS.md` v1.4 §6.3-§6.5.
>
> **Last updated:** 2026-04-27 (E2.4.C.4 decision committed alongside this v1.0 issue).

---

## 1. Decision (one sentence)

**`balanced_calib` is DEMOTED to Phase D infrastructure-diagnostic status; it is NOT adopted as a Phase E2 ablation feature, and it is NOT included in any V31-cycle paper main-results table.**

---

## 2. Evidence

E2.4.C.3 Mini-scale ablation, 24K steps × 8K tok/step bilingual at 60:40, 43.4 min on RTX 4090. Spec `FJQ_PHASE_E_E2_4_C_METRIC_SPEC.md` v1.0 § 5.3 + § 6 gate:

| Metric | Result | Gate | Verdict |
|---|---|---|---|
| `outlier_global_reduction` | **−82.13** | ≥ 0.10 | **FAIL** by 82× over the threshold (in the wrong direction) |
| `global_mean_reduction` | **−148.27** | (diagnostic) | Calibrated quantizer is 149× WORSE than baseline overall |
| `val_loss(EN)` (50 batches, seed 999) | 4.83 (PPL 125.7) | Q5 baseline 4.73 | Within seed-variance — sampler doesn't perturb training |
| Training wall | 43.4 min | (ref Q5 = 42 min) | Bilingual stream throughput matches EN-only |
| BitLinear sites | 37 | (ref 6 layers × 6 + 1 lm_head) | Q1 closure prediction confirmed |

Artifacts: `paper/intllm/ablations/mini_balanced_calib.json` (ablation row), `mini_balanced_calib_maps.pt` (37-layer calibration), `mini_balanced_calib_quant_error.json` (metric output).

---

## 3. Why it failed (3-cause analysis)

The metric correctly identified that `q_calibrated` (per-channel calibrated scale derived from `running_max` accumulated across 24K training steps + per-channel adaptive bits 8/10) is fundamentally inferior to upstream `q_baseline` (per-token `activation_quant`) for BitLinear inputs in this model. Three converging causes:

### 3.1 All-time-max running_max captures non-stationary peaks

`BitLinearStatTracker.running_max` (`intllm/qat.py:88-101`) is `torch.maximum` over every forward call across all 24K training steps. Early training has random-init activation peaks 10-100× larger than steady-state. The accumulator captures these early peaks and never decays, so the calibrated scale is locked to a worst case the late-training model rarely produces. Per-layer evidence from the E2.4.C.3 maps:

| Layer family | running_max range | typical late-training |x| | calibrated scale = 127/running_max |
|---|---|---|---|
| `attn.o_proj` | 11 – 20 | ~0.1 – 0.5 (post-RMSNorm) | 6 – 12 |
| `mlp.down_proj` | 19 – 41 | ~0.1 – 0.5 | 3 – 7 |
| `attn.{i,f,g}_proj` | 5 – 6 | ~0.1 – 0.5 | 21 – 25 |

For `attn.o_proj` the scale is ~7 (far too coarse for 0.5-magnitude inputs); baseline gets `scale_t = 127/0.5 = 254` → near-perfect quantization. Hence the 112× MSE gap on that layer family.

### 3.2 RMSNorm-bounded inputs reward per-token adaptation, not per-channel calibration

Upstream `BitLinear.forward` (`_upstream/mmfreelm/ops/fusedbitnet.py:560-583`) runs `RMSNorm(x)` BEFORE `activation_quant`. RMSNorm bounds inputs adaptively per-token:

```
x_norm[t, :] = x[t, :] / sqrt(mean(x[t, :]^2) + eps)  # per-token RMS scaling
```

After RMSNorm, the per-token absmax of `x_norm` is similar across rows but its absolute magnitude shifts as the model trains. `q_baseline = activation_quant(x_norm)` benefits from this because `scale_t = 127 / max(|x_norm[t, :]|)` IS large for the typical small-magnitude rows that emerge late in training, granting fine-grained 8-bit quantization. `q_calibrated` cannot adapt — it locks to `running_max` from a peak observed during training, missing the steady-state distribution by 10-50×.

This is the key insight that distinguishes our setting from SmoothQuant/GPTQ/AWQ: those methods calibrate on **post-training, no-RMSNorm** activations where per-channel scale beats per-token. For RMSNorm-ed BitLinear inputs, per-token wins.

### 3.3 Outlier-bit-allocation is the wrong lever when scale is wrong

The 5% top-K → 10-bit promotion gives outlier channels MORE bits, but only addresses GRANULARITY error within the channel's chosen scale. It does not fix the LOCKED-COARSE-SCALE problem in §3.1+§3.2. With scale_c too coarse, even fine 10-bit resolution lands inside a quantization grid that no longer matches the actual data distribution. The ablation map's adaptive bit allocation is, in this regime, unhelpful at best and noise-amplifying at worst.

This matches a pattern in the literature: AWQ explicitly notes that per-channel quant-mass reallocation only helps when the underlying scale is correctly calibrated; GPTQ uses an iterative weight-update procedure to compensate for scale errors. Our calibration procedure has no such compensation path.

---

## 4. What stays vs what gets demoted

### 4.1 KEPT — these landed clean and are reused elsewhere

| Asset | Why we keep it |
|---|---|
| `intllm.data.bilingual_stream` (Q5 + E2.4.A) | Used by Q5 baseline + every future E2.x ablation that wants 60:40 ID:EN training data. Independent of any quantization decision. |
| `intllm.qat.attach_stat_trackers` running in production training (E2.4.A) | Phase D infrastructure that was previously test-only; now wired for any future calibration scheme. Hooks have zero training-trajectory impact (verified by `test_qat_hooks_do_not_break_training_convergence`). |
| `intllm.qat.save_calibration_maps` (E2.4.A.2) | Atomic `.pt` artifact format + 37-site layer count = stable surface for any future calibration scheme to consume. |
| `intllm.quant.activation_quant_per_channel` (E2.4.C.2) | Standalone math helper. Independent of which calibration scheme produced the inputs. Reusable. |
| `intllm.eval.compute_quant_error_per_channel` (E2.4.C.2) | The metric itself is sound — it correctly produced a NEGATIVE result here, which is a successful test of the methodology. Reusable for any future calibration-scheme proposal. |
| `--balanced-calib` flag in `train_mini_ablation.py` | Continues to exist; produces the maps for diagnostic / debugging use. NOT a feature flag any paper claim depends on. |
| `train_mini_ablation.py` default `n_steps = 24K` (matches Q5) | Future E2.x ablations get like-for-like training budget against Q5 baseline by default. Bug discovered + fixed during E2.4.C.3 launch. |

### 4.2 DEMOTED — these do not appear in V31 paper claims

| Asset | New status |
|---|---|
| "balanced_calib" as an E2 ablation winner | NOT in main results. NOT in adoption decision. The −82.13 metric value can appear in an appendix as a methodology lesson if a paper wants it; otherwise omit. |
| `compute_bit_allocation` + `compute_channel_permutation` integration into forward-time inference | DEFERRED. The all-time-max problem applies to any forward-time consumption of these maps, not just the offline metric. Not a V31 deliverable. |
| Plan v1.8 §3 PHASE E2.4 adoption-rate language ("≥+0.05 nat improvement") | OBSOLETE per Q9 = Option C re-scope (already handled in findings v1.1+) AND now reinforced by §3 above — even the re-scoped MSE metric fails on this calibration scheme. |

### 4.3 REPRESERVED FOR FUTURE WORK — not started

| Item | Future scope | Pointer |
|---|---|---|
| Post-training PTQ calibration on held-out data (SmoothQuant pattern) | Calibrate scales on a fixed held-out batch AFTER training completes, not via running-max during training. Standard literature approach; should match or beat upstream `q_baseline` if implemented correctly. | V32 cycle or Phase F. Reference: SmoothQuant arxiv 2211.10438 Algorithm 1. |
| Exponential moving average for running_max | Replace `torch.maximum` accumulator with EMA (e.g., `ema_alpha = 0.99`) so late-training stability dominates over early-training peaks. Lower-effort than full PTQ; partial mitigation only. | V32; ~1 day implementation + 1 ablation. |
| Skipping calibration during warmup | Only accumulate `running_max` after `warmup_steps`. Cheaper still; addresses the early-peak issue most directly without changing the accumulator type. | V32; ~½ day. Worth trying first. |
| Option B (`IntLLMBitLinear` wrapper applying maps at forward-time) | Deferred per Q10. Independent of the calibration source — the wrapper would consume whatever maps are saved. Re-evaluate once one of the three above ships and demonstrates the calibrated path can beat baseline. | Phase F. |

---

## 5. Why this is a successful Phase E2 outcome

It would have been easy to ship a paper claim that `balanced_calib` improves quantization quality based on the Phase D `BitLinearStatTracker` infrastructure being "shipped." The Option C metric pre-flight + the failed gate prevented that. Per CLAUDE §6.6 R3 (no inflated statistics) and §6.9 R1+R6 (canonical-protocol benchmarks; no paper claim without one):

- We produced a runnable, reproducible adoption gate (`outlier_global_reduction ≥ 0.10`).
- We ran the gate end-to-end on a real model.
- The gate failed.
- We documented WHY it failed (§3 above) at a level of specificity another team can replicate.
- We DEMOTED the feature instead of pretending it works or blaming the metric.
- We catalogued (§4.3) the next bets that would address each cause.

This is the methodology working as intended. A negative result published honestly is more valuable than a positive result that doesn't replicate — both for the paper and for the team's ability to make the next call correctly. The 10 hours human + 85 min GPU spent on E2.4 produces a clear "don't do that" signal that saves multiples of that effort downstream.

---

## 6. Stakeholder sign-off

| Role | Name | Sign-off |
|---|---|---|
| Author | Muhamad Fajar Putranto (PrimeCore.id) | 2026-04-27 |
| Engineering | Claude Opus 4.7 (1M context) | 2026-04-27 |
| Reviewers | (paper team, when v3.x submission is drafted) | TBD |

---

## 7. References

- Predecessor: `FJQ_PHASE_E_E2_4_FINDINGS.md` v1.4 §6.3-§6.5
- Spec: `FJQ_PHASE_E_E2_4_C_METRIC_SPEC.md` v1.0
- Plan: `FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.8 §3 PHASE E2.4
- Q5 baseline: `paper/intllm/ablations/q5_bilingual_baseline.json` (commit `1074883`)
- This ablation: `paper/intllm/ablations/mini_balanced_calib.json` + `_maps.pt` + `_quant_error.json`
- Code: `intllm/{qat.py, quant.py, eval.py, data.py}`, `scripts/train_mini_ablation.py`
- Tests: 53/53 PASS across `test_quant.py` + `test_qat.py` + `test_data.py` + `test_eval_quant_error.py`
- Literature: SmoothQuant (arxiv 2211.10438), GPTQ (arxiv 2210.17323), AWQ (arxiv 2306.00978), QuaRot (arxiv 2404.00456), SpinQuant (arxiv 2405.16406)

*Document version: 1.0 (E2.4.C.4 final decision). Author: Claude Opus 4.7 + Fajar (PrimeCore.id).*
*Phase E v1.8 Tier 1+2 scope. Predecessor: E2.4.0 closed 2026-04-27.*
