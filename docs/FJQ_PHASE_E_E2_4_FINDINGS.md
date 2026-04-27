# Phase E E2.4 — BilingualCalibrationSampler Pre-flight Audit (Findings v1.2)

> **Status:** E2.4.0 CLOSED + E2.4.A CLOSED + **E2.4.C.1 spec CLOSED**. Q9 = Option C; Q10 = N/A; Q11 closed in spec doc. E2.4 implementation in flight under Option C; metric spec frozen.
>
> **Plan reference:** `FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.8 §3 PHASE E2.4 + `FJQ_PHASE_E_E2_FINDINGS.md` v1.1 §3 (E2.4 sequenced as priority #1, lowest-risk).
>
> **Predecessor:** Phase E2.0 fully closed 2026-04-27 (`FJQ_PHASE_E_E2_FINDINGS.md` v1.1 sign-off, commit `1074883`).
>
> **Companion spec:** `FJQ_PHASE_E_E2_4_C_METRIC_SPEC.md` v1.0 (E2.4.C.1 deliverable, this session). Defines eval set, quantizer formulas, aggregation rules, adoption gate, and the implementation contract for E2.4.C.2.
>
> **Last updated:** 2026-04-27 (E2.4.0 v1.1 → v1.2 promotion: E2.4.C.1 spec doc complete; Q11 closed in spec doc §2).

---

## 1. Headline finding

**The plan v1.8's framing of E2.4 as "lowest-risk + no arch changes + ≥+0.05 nat improvement at Mini scale" is internally inconsistent.** A *sampler* alone (= Phase D `intllm.qat.attach_stat_trackers` fed by `intllm.data.bilingual_stream`) does not change training loss by itself, because the existing upstream `BitLinear.forward` (`_upstream/mmfreelm/ops/fusedbitnet.py:560-583`) hard-codes uniform 8-bit `activation_quant` and ignores any per-channel bit map. Without an arch-level mod that *applies* the bit allocation map at forward time, the Mini ablation `train-mini-ablation TAG=balanced_calib --balanced-calib` will produce val_loss within noise of the Q5 baseline (val_loss(ID)=2.68, val_loss(EN)=4.73).

This is a real surface-area gap, not a claim of plan error — pre-flight surfaces it before implementation starts (§6.8 R1).

**Decision required (user-level) before E2.4.1 commits any code:**
- **Option A** — E2.4 = A-only ("calibration plumbing"). Build the sampler + stat-tracker calibration loop + persist bit allocation map; accept that the ablation row reports val_loss within ±0.02 nat of Q5 baseline (no signal, by construction). Adopt as Phase D infrastructure prereq, NOT as an E2 ablation winner.
- **Option B** — E2.4 = A+B (full feature). Add the calibration plumbing AND a custom `activation_quant_per_channel` in `BitLinear.forward` that consumes the saved bit allocation map. This is an upstream-fork or weight-wrapper arch change. Reclassifies E2.4 risk as medium, scope ~5-7 days (matching E2.1/E2.2).
- **Option C** — Re-scope E2.4 ablation gate: drop the val_loss target and replace with a quantization-error-reduction metric (e.g., FP32 vs ternary-with-bit-map activation MSE on a fixed eval set). Run the sampler + map computation; report MSE delta. No arch change needed; gate becomes a quality-of-the-bit-map measurement, not a training improvement claim.

Recommendation (preliminary): **Option C** if Phase E2 wants to ship E2.4 quickly without forking upstream; **Option B** if Phase E2 wants E2.4 to be a real E2 ablation row. Option A by itself does not justify an ablation row in the paper.

The findings below provide the evidence base for whichever option the user picks.

---

## 2. Existing surface area (what's actually in the tree)

### 2.1 `intllm.qat` (209 LOC, `intllm/qat.py`)

Three pieces shipped by Phase D Track B step 1 (commit chain ending in `950a379`):

| Symbol | Lines | Purpose | Wired in production? |
|---|---|---|---|
| `BitLinearStatTracker` | 67-102 | Forward-hook helper; accumulates per-channel `running_max` of `\|x\|` | **No** — only used in `tests/test_qat.py` |
| `attach_stat_trackers(model)` | 111-130 | Walk model, install one tracker per BitLinear-family module | **No** — only `tests/test_qat.py` calls it |
| `compute_bit_allocation(tracker, ...)` | 149-177 | Top-K outlier channels → 10-bit, rest → 8-bit; returns int32 tensor `(in_features,)` | **No** — only `tests/test_qat.py` calls it |
| `compute_channel_permutation(tracker)` | 184-199 | Permutation that sorts channels by descending running_max | **No** — only `tests/test_qat.py` calls it |
| `QATConfig` | 43-60 | Toggles for `enable_adaptive_bits`, `enable_channel_permutation`, `enable_periodic_recal` | Has no consumer in production code |

**Status:** All four primitives are documented and unit-tested (`tests/test_qat.py` 14/14 PASS) but never called from a training driver. The module docstring is honest about this — line 24-27: *"This module ships the **measurement + transformation** layer. The **training-loop integration** ... lives in `intllm.train` extensions and the C.P4 training driver."* The C.P4 training driver does not in fact integrate them; Phase D shipped infrastructure, not integration.

### 2.2 `intllm.train.train_loop` (`intllm/train.py:272-end`)

Pure Phase D BitNet training: cosine-schedule SGD over a stream, no calibration phase, no QAT toggles. Direct path:
1. Build optimizer + scheduler.
2. Iterate batches.
3. `loss = model(input_ids=ids, labels=ids).loss`; `.backward()`; `step()`.
4. Periodic checkpoint via `_save_checkpoint`.

No call to `attach_stat_trackers`, no consumption of `QATConfig`. Adding calibration is a clean train-loop extension (pre-loop hook attach; post-loop or periodic compute_bit_allocation), not a refactor.

### 2.3 `_upstream/mmfreelm/ops/fusedbitnet.py:541-583` (BitLinear.forward)

```python
def forward(self, x):
    w = self.weight
    x_norm = self.norm(x)                                                    # RMSNorm
    x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()          # 8-bit STE
    w_quant = w + (weight_quant(w) - w).detach()                             # ternary STE
    y = F.linear(x_quant, w_quant)
    return y
```

**Critical observation:** `activation_quant` (line 514 in same file) takes only `x` and returns 8-bit-quantized output. There's no per-channel bit-width parameter. To use the `bits` tensor from `compute_bit_allocation`, the forward would need:

```python
# hypothetical Option-B forward
def forward(self, x):
    x_norm = self.norm(x)
    if self.bits_per_channel is not None:                                    # adaptive path
        x_quant = x_norm + (activation_quant_per_channel(
            x_norm, self.bits_per_channel) - x_norm).detach()
    else:                                                                     # baseline path
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
    w_quant = w + (weight_quant(w) - w).detach()
    return F.linear(x_quant, w_quant)
```

`activation_quant_per_channel` doesn't exist. Implementing it is straightforward (per-channel max → per-channel scale → quantize-cast-dequantize; see e.g. SmoothQuant arxiv 2211.10438 for the data flow), but it's an arch-level change. Whether to fork `_upstream` or wrap externally is itself a design decision.

### 2.4 `intllm.data.bilingual_stream` (THIS SESSION, commit `1074883`)

Already shipped as part of Q5. Signature exactly matches what a "BilingualCalibrationSampler" needs:
```python
bilingual_stream(
    *, tokenizer, seq_len, batch_size, id_share=0.6, device,
    seed, en_repo, id_corpus_root, id_sources,
) -> Iterator[torch.Tensor]
```

Per-batch coin-flip RNG (seeded), distinct per-source seeds, `intllm_en.BILINGUAL_RATIO_DEFAULT` as the natural default. Long-run ratio converges to `id_share` within ±2% at 1000+ batches (verified in `tests/test_data.py::test_bilingual_stream_ratio_converges_to_target`).

**Implication:** the "BilingualCalibrationSampler" of plan v1.8 §3 PHASE E2.4 IS `bilingual_stream`. There is no need to define a separate class. Wrapping it in `intllm.qat.BilingualCalibrationSampler` would be naming-only; behavior is identical. Decision: keep `bilingual_stream` as the canonical name, add a thin alias `intllm.qat.BilingualCalibrationSampler = bilingual_stream` if the plan-level name is desired for paper/decision-doc cross-reference.

---

## 3. Q6 closure (in-flight Phase E2 question)

> **Q6.** Does the existing BitLinearStatTracker harness see ID + EN data correctly when bilingual sampler is plugged in? E2.4 must NOT regress the existing per-coord adaptive bit allocation (Track B step 1 already shipped this).

**Answer: YES, by inspection — no smoke run needed.**

`BitLinearStatTracker.__call__` (`intllm/qat.py:90-102`) is a forward hook on the BitLinear module. It runs once per forward call, irrespective of which data source produced the input batch. The accumulator `torch.maximum(self.running_max, per_channel, out=self.running_max)` is element-wise and order-independent. Therefore:

- A bilingual training run feeding `bilingual_stream(id_share=0.6)` produces ~60% of forward calls on ID-derived activations and ~40% on EN-derived activations.
- The accumulator captures the *union* of outlier channels — channels that are high-magnitude in EITHER language.
- `compute_bit_allocation` then selects top-K from this union, allocating 10-bit precision to channels that are outlier-prone for either language.
- `compute_channel_permutation` sorts by descending union absmax.

**The bilingual sampler does not regress the per-coord adaptive bit allocation; it generalizes it to bilingual outlier detection.** This is exactly the property the plan wants. No tracker change needed.

**Edge case worth noting:** if an outlier channel exists in only one language (e.g. an Indonesian Wikipedia template that activates an embedding row hard, while no EN doc activates it), the union accumulator captures that channel even though only ~60% of calls would have triggered it. The 5% top-K bit allocation may therefore over-spend on language-specific outliers compared to a unified-corpus baseline. This is a *feature* if cross-lingual deployment matters; a *cost* if measuring against an EN-only baseline. For E2 ablations the cost is small (≤0.5% extra channels at high precision).

---

## 4. Implementation plan (conditional on user's option choice)

### 4.1 If Option A (calibration plumbing only)

Sub-tasks (each ~½ day; ~3 days total):

1. **E2.4.A.1** Add `attach_stat_trackers` invocation to `train_mini_ablation.py` when `--balanced-calib` is set. Track all hooks for the duration of training.
2. **E2.4.A.2** After training loop ends, call `compute_bit_allocation` and `compute_channel_permutation` per BitLinear; serialize results to `paper/intllm/ablations/mini_balanced_calib_maps.pt`.
3. **E2.4.A.3** Wire `--balanced-calib` to use `bilingual_stream` (currently `slimpajama_stream` only); reuse `id_share=BILINGUAL_RATIO_DEFAULT`.
4. **E2.4.A.4** Smoke test: assert `len(trackers) == n_layers × 6 + 1` (6 BitLinears per HGRN block + lm_head, per Q1 closure); assert all trackers have `n_calls > 0` post-training.
5. **E2.4.A.5** Mini ablation: `make train-mini-ablation TAG=balanced_calib --balanced-calib` (~1.5h GPU at Mini scale, matches Q5 wall ≈ 42 min). Acknowledged: val_loss within ±0.02 nat of Q5 baseline; ablation row reports the maps' delta against a hypothetical `compute_bit_allocation(EN-only)` reference, NOT a training-loss delta.
6. **E2.4.A.6** Decision doc `FJQ_PHASE_E_E2_BILINGUAL_CALIB_DECISION.md`: documents map shape, top-K channel overlap between EN-only and bilingual calibration, recommendation to keep bilingual calib as default for Phase E onward.

**No upstream fork.** Pure additive surface in `intllm.qat` + `train_mini_ablation.py`.

### 4.2 If Option B (full A+B feature)

Sub-tasks A.1-A.6 above, plus:

7. **E2.4.B.1** Implement `activation_quant_per_channel(x, bits_per_channel)` in `intllm.quant`. Per-channel scale via `(2**(bits_per_channel-1)-1) / running_max`; quantize-cast-int-dequantize. Preserve STE via `.detach()` trick. Unit tests for forward/backward.
8. **E2.4.B.2** Subclass or monkey-patch `BitLinear.forward` to consume `bits_per_channel` (loaded from saved map). Decide between (i) upstream fork in `_upstream/mmfreelm/ops/fusedbitnet.py` (touches frozen-dep snapshot — see `_upstream/UPSTREAM_PIN.md`), or (ii) external wrapper class `IntLLMBitLinear(BitLinear)` that overrides forward. Preferred: (ii), keeps upstream pinning intact.
9. **E2.4.B.3** Train Mini with `--balanced-calib --apply-bit-allocation` two-pass: pass 1 trains baseline + computes maps (= A.1-A.5); pass 2 reloads + applies maps + trains a few more steps OR re-trains from scratch with maps active. Decide which is the canonical "ablation" — paper reports both.
10. **E2.4.B.4** Mini ablation gate: ≥+0.05 nat val_loss improvement vs Q5 baseline. If pass-1-then-pass-2 from-scratch shows ≥+0.05 nat → adopt; if not → revert to Option A and document.

Calendar adjustment if Option B: Phase E v1.8 §3 PHASE E2.4 estimate of 3-5 days becomes 5-7 days; total Phase E2 budget +2-3 days.

### 4.3 If Option C (re-scope ablation gate)

Sub-tasks A.1-A.4 above, plus:

7. **E2.4.C.1** Define a quantization-error metric: pick a fixed held-out activation tensor (e.g. concatenated forward-pass activations on 100 batches of bilingual eval data, layer-by-layer); compute MSE between FP32 and ternary+8-bit-uniform vs ternary+per-channel-bit-map; report ratio.
8. **E2.4.C.2** Implement the metric in `intllm.eval` as `compute_quant_error_per_channel(model, batches, bit_map)` (no model change; just measures `(activation_quant(x) - x).pow(2).mean()` with and without map).
9. **E2.4.C.3** Mini ablation: `make train-mini-ablation TAG=balanced_calib --balanced-calib` produces map; run `compute_quant_error_per_channel` post-training; report MSE-reduction delta. Adoption gate: ≥10% MSE reduction on outlier channels.

No upstream fork; lower scope than Option B; provides paper-defensible adoption signal that Option A lacks. Calendar matches Option A (~3 days).

---

## 5. Open questions (must close before E2.4.1 implementation)

| # | Question | Why it matters | Owner / next step |
|---|---|---|---|
| ~~Q9~~ ✅ **CLOSED v1.1** | Which option (A / B / C) does the user pick for E2.4 scope? | Determines whether E2.4 is plumbing-only (A), full feature (B), or signal-redefined (C). | **Answer: Option C** (re-scoped to MSE quantization-error metric on bilingual eval set). User nod 2026-04-27. Calibration plumbing landed this session; metric + ablation deferred to E2.4.C.1+. |
| ~~Q10~~ ✅ **N/A under Option C** | If Option B: does the user accept an external `IntLLMBitLinear` wrapper class (not a `_upstream/` fork), even if it duplicates ~30 LOC of forward logic? | Forking `_upstream/` breaks Phase D Medium training reproducibility. | **N/A under Option C** — no BitLinear modification needed; `_upstream/` pin preserved. Revisit if a future session upgrades from C → B. |
| ~~Q11~~ ✅ **CLOSED v1.2 (in spec doc)** | If Option C: which held-out activation set defines "the" quantization-error metric? Phase D bench-canonical EN-only? Phase E1 corpus 1000-batch bilingual sample? Both? | The metric is the ablation gate; choosing the eval set is choosing what "improvement" means. | **Answer:** Q5-style 1000-batch bilingual sample, `bilingual_stream(id_share=0.6, seed=42)`, batch_size=8, seq_len=1024, 8.19M tokens total. Distinct seeds from Q5 (train=0, val_id=999, val_en=998); seed=42 reserved for E2.4.C metric. EN-only and "both" alternatives explicitly rejected. Full justification: `FJQ_PHASE_E_E2_4_C_METRIC_SPEC.md` v1.0 §2. |

---

## 6. E2.4.0 ship criteria (what makes this pre-flight CLOSED)

- [x] §2 surface-area audit (5 modules: qat, quant, train, BitLinear forward, bilingual_stream)
- [x] §3 Q6 closure (BitLinearStatTracker is data-agnostic; bilingual_stream feeds it cleanly)
- [x] §4 implementation plan for all three user options (A / B / C) with sub-task breakdowns
- [x] §5 open questions that block E2.4.1 commit (Q9 mandatory; Q10/Q11 conditional)
- [x] **User decision on Q9: Option C** — re-scoped ablation gate to MSE quantization-error metric on bilingual eval set. Calibration plumbing (A.1+A.2+A.3+A.4) lands this session as the prerequisite for the C.1+C.2 metric work.

### 6.1 Q9 closure — Option C selected

**Decision (user nod 2026-04-27, commit landed alongside this v1.1 promotion):** Phase E E2.4 proceeds under **Option C** — sampler + calibration plumbing + MSE-error ablation gate. Rationale per §1: avoids forking the `_upstream/` pinned snapshot (preserves Phase D Medium training reproducibility), lower scope than Option B (~3 days vs ~5-7 days), provides a paper-defensible adoption signal that Option A alone lacks.

Q10 (wrapper subclass for Option B) **does not apply** under Option C — no BitLinear modification needed. Recorded as "N/A under Option C" in §5; revisit if the project later upgrades from C → B.

Q11 (held-out activation set for the MSE metric) is **deferred to E2.4.C.1** — first sub-task under Option C is to define the metric, which inherently picks the eval set. Preliminary recommendation per §4.3: Q5-style 1000-batch bilingual sample (matches Q5 val protocol and is more honest for Phase E goals).

### 6.2 What landed this session

Calibration plumbing shipped alongside this v1.1 promotion (single commit, this session):

- **`intllm.qat.save_calibration_maps(trackers, out_path, ...)`** — atomic-write helper that runs `compute_bit_allocation` + `compute_channel_permutation` per tracker and dumps a single `.pt` artifact with one entry per BitLinear plus a `_meta` block. 5 new unit tests in `tests/test_qat.py` (all PASS, including atomic-write + zero-observations error path).

- **`scripts/train_mini_ablation.py` `--balanced-calib` wired to real impl:**
  - Stream selector: `bilingual_stream(id_share=BILINGUAL_RATIO_DEFAULT)` when flag set; `slimpajama_stream` otherwise (baseline path unchanged).
  - `attach_stat_trackers(model)` called pre-loop when flag set.
  - Post-loop: `save_calibration_maps` writes `paper/intllm/ablations/mini_<TAG>_maps.pt`; `detach_stat_trackers` cleans up.
  - `features_active` now records `balanced_calib`; the [WARN] for unimplemented features is suppressed via the new `E2_REAL_FEATURES` set.
  - JSON schema bumped to v1.1 with new `tracker_maps_path` field.

- **Smoke test passed:** 200-step POL on RTX 4090, 44s wall, **37 BitLinear sites covered** (= 6 layers × 6 attention/MLP projections + 1 lm_head — matches Q1 closure prediction exactly). Map .pt artifact contains `running_max`, `bits`, `permutation`, `in_features` per layer; bits tensor shows the expected ~5% top-K at 10-bit (13/256 = 5.08% on a layer 0 attn.i_proj sample).

Module-level test coverage post-session: `tests/test_qat.py` 15 tests / `tests/test_data.py` 14 tests = 29/29 PASS.

### 6.3 Remaining E2.4 sub-tasks (deferred to next sessions)

- [x] **E2.4.C.1** define MSE quantization-error metric — DONE this session, `FJQ_PHASE_E_E2_4_C_METRIC_SPEC.md` v1.0
- [ ] **E2.4.C.2** implement `compute_quant_error_per_channel(model, batches, bit_map_path, ...)` in `intllm.eval` + `activation_quant_per_channel(x, running_max, bits_per_channel)` standalone math helper in `intllm.quant`. Unit tests including the spec §7 cross-reference invariant (q_calibrated with bits=8 everywhere ≡ q_baseline). ~½ day human, no GPU.
- [ ] **E2.4.C.3** Mini ablation: train w/ `--balanced-calib` 24K steps, run metric on the 1000-batch eval set per spec §2, gate on `outlier_global_reduction ≥ 0.10` per spec §6. Writes `paper/intllm/ablations/mini_balanced_calib_quant_error.json` per spec §7 schema. ~1.5h GPU + ~5 min metric pass.
- [ ] **E2.4.C.4** Decision doc `FJQ_PHASE_E_E2_BILINGUAL_CALIB_DECISION.md` documenting adoption call — PASS / OBSERVATION / FAIL path per spec §6.

Estimated remaining effort: ~1.5 days human + ~1.5h GPU (was ~2 days at v1.1; -25% because C.1 finished in this session was budgeted at ½ day).

---

## 7. References

- Plan: `docs/FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.8 §3 PHASE E2.4
- Predecessor findings: `docs/FJQ_PHASE_E_E2_FINDINGS.md` v1.1 §3 (E2.4 sequenced first), §4 (Q6 was open)
- Q5 plumbing (this session): `python/phase_d/intllm/data.py::bilingual_stream`, `scripts/q5_bilingual_baseline.py`
- QAT infrastructure: `python/phase_d/intllm/qat.py` (209 LOC, Track B step 1)
- Upstream BitLinear: `python/phase_d/_upstream/mmfreelm/ops/fusedbitnet.py:541-583`
- Tests: `python/phase_d/tests/test_qat.py` 14/14 PASS; `tests/test_data.py` 14/14 PASS (post-Q5)
- Literature: SmoothQuant (arxiv 2211.10438) for `activation_quant_per_channel` data flow; Calibrating Beyond English (arxiv 2601.18306) for bilingual calibration motivation; QuaRot (arxiv 2404.00456) + SpinQuant (arxiv 2405.16406) as cross-feature anchors

---

**Sign-off (v1.0):** Phase E2.4.0 pre-flight AUDIT complete. Surface area mapped (qat / quant / train / BitLinear / bilingual_stream); Q6 closed by inspection (sampler is data-agnostic; bilingual_stream feeds tracker cleanly); core scope-clarification surfaced — plan-v1.8 framing of "lowest risk + ≥+0.05 nat improvement" is internally inconsistent without an arch-level change at BitLinear.forward. **Status: v1.0 (audit + plan + scope question only) — promotion to v1.1 (E2.4.0 CLOSED) requires user decision on Q9 (option A / B / C). No code committed; no GPU run.**

**Sign-off (v1.1, 2026-04-27):** Phase E2.4.0 pre-flight CLOSED. Q9 = Option C (re-scoped ablation gate to MSE quantization-error metric). Calibration plumbing shipped this session: `intllm.qat.save_calibration_maps` + `scripts/train_mini_ablation.py --balanced-calib` real impl + 5 unit tests + 200-step smoke (44s wall, 37 BitLinear sites covered, ~5% top-K at 10-bit verified). Total test coverage post-session: 29/29 PASS across `test_qat.py` + `test_data.py`. No upstream fork; `_upstream/` pin preserved per Q10 (which is N/A under C).

**Sign-off (v1.2, 2026-04-27):** E2.4.C.1 metric specification COMPLETE. Q11 closed (eval set = `bilingual_stream(id_share=0.6, seed=42)`, 1000 batches × 8 × 1024 = 8.19M tokens). Companion spec doc `FJQ_PHASE_E_E2_4_C_METRIC_SPEC.md` v1.0 freezes the implementation contract for E2.4.C.2 (`compute_quant_error_per_channel` + `activation_quant_per_channel` standalone helper). Adoption gate = `outlier_global_reduction ≥ 0.10` (10% MSE reduction restricted to channels promoted to 10-bit by the saved map). No code shipped this session — spec only.

Next session natural first step: E2.4.C.2 implementation — `activation_quant_per_channel` math helper in `intllm.quant` + `compute_quant_error_per_channel` driver in `intllm.eval`, with unit tests including the spec §7 cross-reference invariant. Pure CPU work; ~½ day. Then E2.4.C.3 Mini ablation (~1.5h GPU); then E2.4.C.4 decision doc.

*Document version: 1.2 (E2.4.C.1 metric spec frozen). Author: Claude Opus 4.7 + Fajar (PrimeCore.id).*
*Phase E v1.8 Tier 1+2 scope. Predecessor: E2.0 fully closed 2026-04-27.*
