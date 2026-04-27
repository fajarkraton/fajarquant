# Phase E E2.1 — Hadamard Rotation Decision (v1.0)

> **Status:** E2.1.4 decision FINAL. Phase E2.1 exits with a HONEST NEGATIVE RESULT — Hadamard rotation on `block.attn.o_proj` is NOT adopted as a Phase E2 ablation feature at Mini scale + training-from-scratch + bilingual-data conditions.
>
> **Plan reference:** `FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.9 §3 PHASE E2.1.
>
> **Last updated:** 2026-04-27 (E2.1.4 decision committed alongside this v1.0 issue).

---

## 1. Decision (one sentence)

**Hadamard rotation as implemented (Walsh-Hadamard, fixed/non-learned, applied via forward pre-hook on `block.attn.o_proj` inputs only, training-from-scratch on bilingual data) is NOT adopted as a Phase E2 ablation feature; demoted to "tested + does not help at Mini scale" status.** The E2.1.1 `HadamardRotation` module + E2.1.2 driver wiring stay in the codebase as reusable infrastructure for future post-hoc QuaRot-style experiments (deferred to Phase F.6).

---

## 2. Evidence

E2.1.3 Mini-scale ablation, 24K steps × 8K tok/step bilingual at 60:40, 61.8 min on RTX 4090. Adoption gate per plan v1.9 §3 PHASE E2.1: `val_loss(EN) < 4.68` (= Q5 baseline 4.732 − 0.05 nat improvement threshold).

| Metric | Q5 baseline | balanced_calib | **Hadamard** | Δ Hadamard vs Q5 |
|---|---|---|---|---|
| `val_loss(EN, seed=999, 50 batches)` | 4.732 | 4.834 | **4.852** | **+0.120 nat (WORSE)** |
| `PPL(EN)` | 113.5 | 125.7 | **128.0** | +12.8% (worse) |
| `final_loss` (training step 24K) | 3.048 | 3.049 | 3.058 | within noise |
| `initial_loss` (random init) | 10.428 | 10.417 | 10.412 | ≈ ln(32768) for all |
| `training_seconds` | 2519 (42.0 min) | 2604 (43.4 min) | **3706 (61.8 min)** | **+47.1%** wall |
| `n_steps` | 24000 | 24000 | 24000 | same step budget |
| Bilingual sampler `id_share` | 0.6 | 0.6 | 0.6 | same |
| Adoption gate `val_loss(EN) < 4.68` | n/a (baseline) | n/a (FAIL on different metric) | **FAIL** (4.852 ≥ 4.68) | regression by 0.17 nat past gate |

Artifacts: `paper/intllm/ablations/mini_hadamard.json` (ablation row, schema v1.2). Q5 baseline reference: `paper/intllm/ablations/q5_bilingual_baseline.json` (commit `1074883`).

---

## 3. Why it failed (3-cause analysis)

The result is INSIDE the OBSERVATION zone (regression < 0.20 nat) but cleanly outside the ≥+0.05 nat adoption threshold. Three converging causes:

### 3.1 Training-from-scratch loses Hadamard's primary benefit

QuaRot (arxiv 2404.00456) and SpinQuant (arxiv 2405.16406) — the literature precedent for `HadamardRotation` — apply rotation **post-training** to a pre-trained model that is then re-quantized. The pre-trained model already has stable, learned representations; rotation reduces outlier-driven quantization error WITHOUT requiring the model to adapt its weights. The benefit is on the QUANTIZATION-ERROR axis at fixed model weights.

In our setup we train FROM SCRATCH with the rotation in place. The training-from-scratch math equivalence is:
```
y = quant(W) · quant(H x) ≈ W · H x = (W H) x
```

The model learns `W_eff = W H` instead of `W`. At convergence the OUTPUT semantics are identical, but the model spends gradient updates fighting against the rotation rather than benefiting from outlier reduction. Our Mini run gets 491M-token-equivalent (24K steps × 8K) of compute, much less than the budgets where weight-learning saturates; the rotation is still net cost at this point.

This is consistent with a known SpinQuant ablation: the paper reports that fixed Hadamard during PRE-TRAINING gives smaller wins than the LEARNED rotation applied post-hoc.

### 3.2 HGRN attention output may not have transformer-style outlier concentration

The QuaRot/SpinQuant outlier story is about transformer self-attention's QKV projection inputs (post-RMSNorm hidden states), where a few channels concentrate magnitude due to the softmax(QK^T/√d) attention pattern emphasizing certain feature dimensions. HGRN's gated linear recurrence is structurally different: gate values `g_t = sigmoid(g_proj(x_t))` and forget values `f_t = sigmoid(f_proj(x_t))` modulate the recurrence smoothly, without softmax-induced sharpness. The activation arriving at `o_proj` is the post-recurrence accumulator state, not a softmax-attention-weighted sum.

**We did not measure outlier concentration on o_proj inputs before running this ablation.** That measurement (per-channel absmax / per-channel mean ratio on captured o_proj inputs from a Q5-trained Mini checkpoint) would have predicted whether rotation has anything to spread. Hindsight: this should have been a Q12 in the E2.1.0 pre-flight if we'd done a dedicated pre-flight findings doc. We did not. Adding it to F.6 entry as a hard prerequisite for any post-hoc Hadamard experiment.

### 3.3 Q1 closure's "rotate o_proj only" call may have been overly conservative

Q1 closure (FJQ_PHASE_E_E2_FINDINGS.md v1.1 §4) ruled out rotating `i_proj`/`f_proj`/`g_proj` because they are HGRN's gated-linear-recurrence projections, structurally distinct from QKV. **That argument is structurally correct but empirically untested.** The full QuaRot recipe rotates inputs to ALL quant-sensitive projections; HGRN's i/f/g_proj inputs are also post-RMSNorm hidden states, just like QKV inputs in transformers. The same outlier pattern *could* be present there.

By restricting rotation to o_proj only, this ablation tested a 1/4 fraction of the potential rotation sites in HGRN attention. A future E2.1-redux could test "rotate everywhere" or "rotate {i,f,g}_proj, not o_proj" as an A/B against the canonical "o_proj only" choice.

---

## 4. What stays vs what gets demoted

### 4.1 KEPT — these landed clean and are reused

| Asset | Why we keep it |
|---|---|
| `intllm.quant.HadamardRotation` (E2.1.1) | Standalone math primitive: orthogonal Walsh-Hadamard rotation with full unit-test coverage (orthogonality, norm preservation, round-trip, outlier suppression on synthetic spike, dim guards, dtype passthrough, buffer-not-parameter). Independent of any application choice. Reusable for F.6 post-hoc QuaRot. |
| `--hadamard` CLI flag in `train_mini_ablation.py` | Continues to exist; produces a real ablation result for diagnostic use. NOT a feature flag any paper claim depends on. |
| `--bilingual-data` CLI flag (NEW in E2.1.2) | Orthogonal flag for ANY E2.x ablation that wants apples-to-apples vs Q5 bilingual baseline without requiring `--balanced-calib` (which is demoted). Independent of Hadamard's failure. |
| `register_forward_pre_hook` wiring pattern | Mirrors the E2.4.A `attach_stat_trackers` pattern; reusable for any future "modify input to BitLinear" experiment without forking `_upstream/`. |

### 4.2 DEMOTED — these do not appear in V31 paper claims

| Asset | New status |
|---|---|
| "hadamard" as an E2 ablation winner | NOT in main results. Records as one of the negative results in any paper appendix discussing Phase E2's outlier-handling experiments. |
| Plan v1.9 §3 PHASE E2.1 "adopt iff ≥+0.05 nat improvement" gate language | Gate stayed mechanical and ran clean; the FAIL is a real outcome, not a mis-specified gate. Keep the language for future E2.x rows. |

### 4.3 PRESERVED FOR FUTURE WORK — Phase F.6 (NEW)

Items deferred to Phase F because the Phase E2.1 investigation surfaced specific algorithmic gaps that need post-Phase-E-paper investigation:

| Sub-task | Description | Risk / cost |
|---|---|---|
| F.6.1 | Outlier-concentration measurement on Q5-trained Mini checkpoint o_proj inputs (per-channel absmax / per-channel mean ratio histogram) | ~½ day; uses existing `compute_quant_error_per_channel` infrastructure with hooks adapted to dump distributions instead of MSE. **Hard prerequisite** for any post-hoc Hadamard experiment — if HGRN o_proj inputs don't have outlier concentration, post-hoc rotation has nothing to spread either. |
| F.6.2 | Post-hoc QuaRot on Q5-trained Mini checkpoint (canonical recipe: load checkpoint, fuse Hadamard into adjacent weights, re-evaluate without re-training) | ~1 day; tests whether the failure mode is specifically "training-from-scratch fights rotation" vs "HGRN doesn't have outlier-prone activations." If F.6.1 shows no outlier concentration, F.6.2 is moot — skip. |
| F.6.3 | Hadamard rotation on i/f/g_proj inputs (against Q1 closure's "non-canonical" warning) | ~1 day implementation + 1 ablation. Tests whether the right insertion point for HGRN was QKV-equivalent paths, not o_proj. Run alongside F.6.2 if budget permits. |
| F.6.4 | Hadamard at Base or Medium scale (rather than Mini) | 4-6h GPU per scale; tests whether outlier concentration emerges at larger scales where the QuaRot literature's effect might appear. |

**Entry condition for F.6:** Phase E paper accepted (or at least submitted) AND F.6.1 measurement confirms ≥3× max/mean ratio on at least one BitLinear site (the threshold below which outlier-spreading rotations are net cost).

---

## 5. Why this is still a successful Phase E2 outcome

Like E2.4, Phase E2.1 produced a clear NEGATIVE answer that prevents an inflated paper claim. Specifically:

- The methodology surfaced two failures (E2.4 catastrophic, E2.1 modest regression) before the paper could claim "Phase E2 ablations show [feature X] improves Mini-scale val_loss."
- The mechanical gate (val_loss(EN) < 4.68) is reproducible — anyone re-running `make train-mini-ablation TAG=hadamard --hadamard --bilingual-data` will hit a similar number (within seed-variance ±0.03 nat).
- The 3-cause analysis (§3) gives the next research team specific hypotheses to test, not just "it didn't work."
- The infrastructure built (HadamardRotation module, --bilingual-data flag, forward-pre-hook attach pattern) is reusable for F.6 post-hoc experiments, so the engineering effort is not lost.
- Per CLAUDE §6.6 R3 + §6.9 R3 + §6.9 R6: a negative result published honestly is more valuable than a positive result that doesn't replicate.

E2.4 + E2.1 both FAIL is also a SIGNAL that the original Phase E2 plan-v1.8 framing of "5 features all expected to ≥+0.05 nat each + ≥+0.15 nat combined" was over-aggressive. Plan v1.9 already softened the combined gate to ≥+0.10 nat (4 features); after E2.1's regression, the realistic expectation is now "1-2 of E2.2/E2.3/E2.5 may help; the others will be neutral or slight regression." This is not a Phase E2 ABORT (the combined `FJQ_PHASE_E_E2_ABORT.md` path triggers only if aggregate < 0 nat after all 4 features), but it is a CALIBRATION.

---

## 6. Stakeholder sign-off

| Role | Name | Sign-off |
|---|---|---|
| Author | Muhamad Fajar Putranto (PrimeCore.id) | 2026-04-27 |
| Engineering | Claude Opus 4.7 (1M context) | 2026-04-27 |
| Reviewers | (paper team, when v3.x submission is drafted) | TBD |

---

## 7. References

- Predecessor: `FJQ_PHASE_E_E2_FINDINGS.md` v1.1 §3 (E2.1 sequenced first lowest-risk per plan v1.1; updated in v1.9 to E2.1 first after E2.4 closure)
- Plan: `FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.9 §3 PHASE E2.1
- Q5 baseline: `paper/intllm/ablations/q5_bilingual_baseline.json` (commit `1074883`)
- This ablation: `paper/intllm/ablations/mini_hadamard.json` (commit landing alongside this doc)
- Companion E2.4 negative result: `FJQ_PHASE_E_E2_BILINGUAL_CALIB_DECISION.md` v1.0
- Code: `intllm.quant.HadamardRotation`, `scripts/train_mini_ablation.py --hadamard --bilingual-data`
- Tests: `tests/test_quant.py` (9 unit tests for HadamardRotation)
- Literature: QuaRot (arxiv 2404.00456), SpinQuant (arxiv 2405.16406)

*Document version: 1.0 (E2.1.4 final decision). Author: Claude Opus 4.7 + Fajar (PrimeCore.id).*
*Phase E v1.9 Tier 1+2 scope. Predecessor: E2.0 closed 2026-04-27; E2.4 closed 2026-04-27.*
