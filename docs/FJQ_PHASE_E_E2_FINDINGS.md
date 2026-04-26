# Phase E E2 — Algorithm Catch-up Pre-flight Audit (Findings v1.0)

> **Status:** E2.0 PRE-FLIGHT only — no code changes, no GPU runs. Audit + plan deliverable per CLAUDE §6.8 R1 (every Phase starts with a pre-flight audit).
>
> **Plan reference:** `FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.8 §3 PHASE E2 (Algorithm catch-up, Weeks 5-12, ~25h human + ~30-60h GPU).
>
> **Predecessor:** Phase E1 FULLY CLOSED 2026-04-27 (`FJQ_PHASE_E_E1_FINDINGS.md` v1.3 sign-off). Bilingual corpus v1.0 ready: 25.67 B tokens at 60:40 ID:EN.
>
> **Last updated:** 2026-04-27 (Phase E v1.8, E2.0 audit + Q1 + Q2 + Q3 + Q4 closures — 4/5 blockers down; only Q5 ~3-5h GPU remaining).

---

## 1. Phase E2 deliverable summary (per plan v1.8 §3)

5 algorithm features evaluated at Mini scale via individual ablations + 1 combined ablation:

| Sub-phase | Feature | Surface area | Decision doc |
|---|---|---|---|
| E2.1 | Hadamard rotation (QuaRot/SpinQuant) | `intllm.quant.HadamardRotation` (NEW); applied to attn output + LM head | `FJQ_PHASE_E_E2_HADAMARD_DECISION.md` |
| E2.2 | Mixed-precision FP8 (E4M3) | `lm_head` + `attn.W_o` cast to FP8; rest stays ternary | `FJQ_PHASE_E_E2_FP8_DECISION.md` |
| E2.3 | FP16 distillation during QAT | `intllm.qat` extended with teacher-student loss (KL on logits + MSE on hidden states) | `FJQ_PHASE_E_E2_DISTILL_DECISION.md` |
| E2.4 | Bilingual calibration sampler | `intllm.qat.BilingualCalibrationSampler` (NEW); default 60:40 matches `intllm_en.BILINGUAL_RATIO_DEFAULT` | `FJQ_PHASE_E_E2_BILINGUAL_CALIB_DECISION.md` |
| E2.5 | Language-conditioned design | Pick (a) per-lang LayerNorm OR (b) per-lang LoRA OR (c) language-aware routing; classes in `intllm.arch.LanguageConditioned*` | `FJQ_PHASE_E_E2_LANG_COND_DECISION.md` |
| E2.6 | Combined ablation | All 5 features ON vs Phase D Mini baseline | (no separate doc; `_E2_FINDINGS.md` v1.x final closure) |

**Adoption thresholds:**
- Per-feature: ≥+0.05 nat improvement at Mini scale (E2.x.4 decision doc gates)
- Combined (E2.6): ≥+0.15 nat aggregate AND coherence ratio ≤1.5×
- Gate E2 PASS: ≥+0.10 nat aggregate AND coherence ≤1.5×; FAIL → fall back to Phase D config + bilingual calib only

---

## 2. Phase D code surface audit (where features hook in)

### 2.1 `intllm.model` (architecture shim, 42 LOC)

Re-exports upstream `mmfreelm.models.{HGRNBitConfig, HGRNBitForCausalLM, HGRNBitModel}` from vendored `python/phase_d/_upstream/`. Every BitLinear hook in HGRN-Bit:

| Layer | Module | BitLinears |
|---|---|---|
| `HGRNBitBlock.attn_norm` | `RMSNorm` | — |
| `HGRNBitBlock.attn` | `HGRNBitAttention` | needs surface check (see §4 open Q1) |
| `HGRNBitBlock.mlp_norm` | `RMSNorm` | — |
| `HGRNBitBlock.mlp.gate_proj` | `BitLinear` | (hidden → 2× intermediate, gated) |
| `HGRNBitBlock.mlp.down_proj` | `BitLinear` | (intermediate → hidden) |
| `HGRNBitForCausalLM.lm_head` | `BitLinear` | (hidden → vocab; FP8 candidate per E2.2) |

Forking the architecture for E2.5 Language-Conditioned design means subclassing `HGRNBitBlock` (or pre-/post-norm hooking). The `intllm.model` shim docstring already anticipates this: *"Later C.P2.2 work will add IntLLM-specific subclasses here: IntBitLinear, IntRMSNorm, QATHook"* — same pattern applies for E2.5.

### 2.2 `intllm.quant` (267 LOC)

Existing public API:

| Symbol | Purpose | E2 relevance |
|---|---|---|
| `SigmoidLUT` (autograd) | LUT-approximated sigmoid for kernel-context | unchanged in E2 |
| `SiLULUT` | LUT-approximated SiLU | unchanged |
| `IntRMSNorm` | γ-less RMSNorm with fixed-point scale | E2.5 candidate (per-lang γ would re-introduce per-lang LayerNorm if option (a) chosen) |
| `fake_quantize_absmax_ste(x, n_bits=8)` | activation quant via STE | unchanged |
| `_FakeQuantizeAbsmaxSTE` | autograd fn | unchanged |
| `_next_power_of_two` | utility | unchanged |

**E2.1 Hadamard rotation insertion point:** new symbol `HadamardRotation` (autograd or plain `nn.Module`). Apply pre-quant to attn output + LM head per QuaRot/SpinQuant. Does NOT touch existing symbols.

**E2.2 FP8 mixed-precision:** new helpers — likely `cast_module_to_fp8(module: BitLinear) -> nn.Module` swap. Or PyTorch native `torch.float8_e4m3fn` if recent enough version. Surface: 2 functions + dtype check helper. Does NOT touch existing ternary path.

### 2.3 `intllm.qat` (209 LOC)

Existing public API:

| Symbol | Purpose | E2 relevance |
|---|---|---|
| `QATConfig` (dataclass) | bit allocation + channel permutation knobs | extend for E2.3 distillation params + E2.4 sampler config |
| `BitLinearStatTracker` | activation stats for adaptive bit allocation | unchanged |
| `_is_bitlinear` | duck typing | unchanged |
| `attach_stat_trackers(model)` / `detach_stat_trackers(model)` | tracker lifecycle | unchanged |
| `compute_bit_allocation(tracker, ...)` | top-K outlier protection | unchanged |
| `compute_channel_permutation(tracker)` | per §6.9 R5 | unchanged |

**E2.3 distillation insertion point:** new function `distillation_loss(student_logits, teacher_logits, student_hidden, teacher_hidden, alpha=0.5, beta=0.5, temperature=2.0)`. Compose with existing CE loss in `intllm.train`. Surface: 1 function + 1 config field (`teacher_path: Path | None`).

**E2.4 BilingualCalibrationSampler insertion point:** new class. Wraps the calibration data iterator (currently EN-only via `slimpajama_stream`); adds ID-side stream from `data/phase_e/corpus_id_dedup_exact/*/shard_*.parquet` mixed at `ratio_id_share = intllm_en.BILINGUAL_RATIO_DEFAULT`. Imports `intllm_en` cross-package (Phase E shim).

### 2.4 `intllm.train` (384 LOC)

Already hardened by Track B 6-layer interruption-safety (V31.C.P6.1-P6.5):
- `ckpt_every` + atomic write + rotation
- `--resume` / `--resume-auto` with bit-exact state restoration
- `StepWatchdog` (1800s default, single-shot SIGTERM)
- HF streaming retry (`_retry_iter` + 60s timeouts)
- LambdaLR scheduler (warmup + cosine + min_lr_ratio)

**E2 hooks:**
- E2.3 distillation: `TrainConfig.teacher_ckpt_path` field; teacher loaded lazily, `distillation_loss` added to step's loss accumulation
- E2.4 sampler: `TrainConfig.calibration_sampler` callable swap (default = current `slimpajama_stream`; bilingual = `BilingualCalibrationSampler`)
- E2.6 combined ablation: `make train-mini-ablation TAG=combined` driver invokes train_mini.py with all 5 features ON

### 2.5 Test coverage gap

Existing tests (`python/phase_d/tests/`):
- `test_quant.py` — covers SigmoidLUT, SiLULUT, IntRMSNorm, fake_quantize_absmax_ste
- `test_qat.py` — covers BitLinearStatTracker, attach/detach, compute_bit_allocation, compute_channel_permutation
- `test_train.py` — covers TrainConfig, LambdaLR, ckpt_every, --resume, StepWatchdog, _retry_iter
- (others: data, export, fp16_parity, model_shim, tokenizer, configs)

**E2 needs (NEW test files):**
- `test_hadamard.py` (E2.1.1) — verify rotation matrix orthogonality + outlier suppression
- `test_mixed_precision.py` (E2.2.1) — param dtype check + numerical stability vs all-ternary baseline
- `test_distillation.py` (E2.3.1) — teacher-student loss components + KL/MSE shapes
- `test_bilingual_sampler.py` (E2.4.x) — ratio invariance over N batches; ID/EN stream interleaving
- `test_lang_conditioned.py` (E2.5.x) — depends on chosen option; param-count delta + forward shape sanity

---

## 3. Recommended implementation order (low-risk → high-risk)

Per CLAUDE §6.9 R3 (port full features, no strawman) + §6.8 R5 (surprise budget) — implement in increasing risk order so blockers surface early:

1. **E2.4 BilingualCalibrationSampler** (~3-5 days, lowest risk)
   - No arch changes; just a sampler that interleaves ID + EN streams at fixed ratio.
   - Imports `intllm_en.BILINGUAL_RATIO_DEFAULT` — Phase E1 deliverable already lands the constant.
   - Smoke test: 10K samples, expect 6,000 ± 100 ID and 4,000 ± 100 EN.
   - Mini ablation: train-mini-ablation TAG=balanced_calib (~3h GPU).
   - **Why first:** literature evidence strongest (Calibrating Beyond English, arxiv 2601.18306), no arch-level coupling, reuses Phase D pipeline 100%.

2. **E2.1 Hadamard rotation** (~5-7 days, low-medium risk)
   - New autograd module `HadamardRotation` in `intllm.quant`.
   - Apply to LM head + attn output projection (need to confirm exact attribute path in HGRNBitAttention — see §4 Q1).
   - Reference: QuaRot (arxiv 2404.00456) + SpinQuant (arxiv 2405.16406).
   - Smoke: orthogonality check + outlier reduction on Mini checkpoint.
   - Mini ablation: ~3h GPU.
   - **Why second:** literature consensus + clean forward-pass insertion; no QAT-loop changes.

3. **E2.2 FP8 mixed-precision** (~5-7 days, medium risk)
   - Cast lm_head + attn.W_o to torch.float8_e4m3fn.
   - Risk: PyTorch FP8 support varies by version; may need bf16 fallback path.
   - Kernel binary size budget check (≤ 16 MB per plan §1 item 12) — FP8 weights cost 2× ternary.
   - Mini ablation: ~3h GPU.
   - **Why third:** FP8 ecosystem maturity uneven; want clean surface before risking dtype edge cases.

4. **E2.3 FP16 distillation** (~7-10 days, medium-high risk)
   - Need a teacher checkpoint. Options:
     - **Option A:** Phase D Medium c.1 (140M params, val_loss 3.72) — IS available on disk per memory. Caveat: Phase D was English-only; teacher will not have ID prior. Bilingual student distilling from EN-only teacher = expected partial signal.
     - **Option B:** BitNet-2B4T (HF available, 2B params). Larger + bilingual-pretrained but fundamentally a different arch (BitNet ≠ HGRN-Bit). Cross-arch distillation risky.
     - **Option C:** Train fresh FP16 teacher on bilingual corpus (~30-60h GPU additional, may double E2 budget).
   - **Recommendation:** Option A first (sunk-cost teacher), validate distill loss shape + decreases, then evaluate signal on bilingual eval. If ID coverage poor, escalate to Option C as Phase E5/F follow-up.
   - Mini ablation: ~3-5h GPU (extra inference passes for teacher).
   - **Why fourth:** highest infra dep (need teacher loaded + extra forward passes); risk of negative transfer from EN-only teacher.

5. **E2.5 Language-conditioned design** (~10-14 days, highest risk)
   - Requires V28.5 forensic review FIRST to pick option (a/b/c).
   - Option (a) per-language RMSNorm γ: smallest param overhead (~0.1%); simplest to implement (subclass `IntRMSNorm` with `language_id` argument).
   - Option (b) per-language LoRA adapter: more expressive but adds inference cost; conflicts with kernel-context goal.
   - Option (c) language-aware routing (MoE-light): largest change; route gate selects FFN expert.
   - Mini ablation: ~3-5h GPU.
   - **Why last:** arch fork; coherence gate (val_loss(ID)/val_loss(EN) within 1.5×) is paper-headline-load-bearing; want all other features stable before adding routing complexity.

**E2.0 entry condition:** reproduce Phase D Mini v2 baseline on bilingual corpus 10% sample (~2.6 B token mix at 60:40, sample to ~200M tokens for fast iteration). Confirms the bilingual data pipeline + sampler work end-to-end BEFORE feature work begins. ~1-2h GPU.

**E2.1.0 prerequisite (NEW from Q4 closure):** scaffold `make train-mini-ablation TAG=X` parametric target + `python/phase_d/scripts/train_mini_ablation.py` driver before any feature work. Mirror `bench-canonical-real TAG=X` pattern. ~30-45 min, no GPU, doc-only-style scaffold. Without this harness, NO E2.x.3 ablation row produces a JSON artifact for `verify_paper_tables.py` to check (§6.9 R7).

---

## 4. Open questions (must close before E2.1 implementation)

| # | Question | Why it matters | Owner / next step |
|---|---|---|---|
| ~~Q1~~ ✅ **CLOSED v1.0+Q1** | Where exactly does `HGRNBitAttention` expose its output projection? Need attribute path for E2.1 + E2.2 hooks. | E2.1.2 verification ("Apply to attention.W_o") and E2.2.1 ("FP8 for attn.W_o") both need a stable attribute name. | **Answer:** `HGRNBitAttention` exposes 4 BitLinear instances per block (`mmfreelm/layers/hgrn_bit.py:56-70`): `i_proj` (input), `f_proj` (forget-gate), `g_proj` (gating), and **`o_proj` (output, line 70)**. The "attn.W_o" target in plan §3 PHASE E2 = `block.attn.o_proj`. **Per-block BitLinear total: 6** (4 attention + `mlp.gate_proj` + `mlp.down_proj`), plus 1 top-level `lm_head` BitLinear. Matches `export.py` "6 BitLinears per layer" spec-correction note. **Implication for E2.1:** Hadamard rotation per QuaRot/SpinQuant rotates only `o_proj` (input/forget/gate are HGRN's gated-linear-recurrence projections, structurally distinct from QKV; rotating them is not part of canonical recipe). **Implication for E2.2:** FP8 mass = `num_layers × 1 (per-block o_proj) + 1 (lm_head)`; do NOT FP8-cast i/f/g_proj or MLP projections. |
| ~~Q2~~ ✅ **CLOSED v1.0+Q2** | Is V28.5 forensic data still available, and what does it say about ID-EN coherence drift? | E2.5 option selection (a/b/c) requires this evidence. | **Critical finding — re-frames plan §3 PHASE E2.5 framing:** V28.5 was a FajarOS *kernel-side numerical* test (Gemma 3 inference inside `@kernel`), NOT a model-level multilingual training stability test. Forensic chain: commit `5670b4e` (initial V28.5 success — ~50 multilingual tokens) was contradicted by V29.P1.P4.3 retest (`fb26769`, 2026-04-16) which found all 64 output tokens decoded to 0x00 pad byte once `@noinline` actually compiled (V29.P1 lexer fix exposed the latent bug). Two interpretations consistent with data: (a) original sample was outlier, steady-state is pad=0; (b) `@noinline` active changes numerical computation enough to collapse to pad=0. **Implication for E2.5 option selection:** V28.5 evidence does NOT actually demand heavy architectural fix — it's a kernel-numerical issue, not a bilingual-training issue. Plan §3 PHASE E2.5's "V28.5 multilingual coherence gap proves naive multilingual ternary fails" is **misleading framing**; the real evidence base for E2.5 should be **literature** (Calibrating Beyond English arxiv 2601.18306) + **empirical Mini-scale signal post-E2.1-2.4 ablations**, not V28.5 forensics. **Preliminary E2.5 recommendation: Option (a) per-language RMSNorm γ** — smallest overhead (~0.1%), simplest, kernel-context-friendly (Tier 1 inference cost penalizes options b+c); final selection deferred until E2.1-2.4 land (their ablations show whether bilingual interference is empirically present at Mini scale). Forensic source: `fajaros-x86` git log (`fb26769`, `4bd308c`, `4100092`). |
| ~~Q3~~ ✅ **CLOSED v1.0+Q3** | Is Phase D Medium c.1 checkpoint (`medium_final.pt`) FP16 or already-quantized? | E2.3 distillation needs an FP16 teacher. If Medium c.1 is post-QAT ternary, Option A is moot. | **Answer:** `python/phase_d/checkpoints/medium/medium_final.pt` is **FP32** master weights — 185/185 params `torch.float32` (top-level keys: `arch`, `train`, `state_dict`, `result`). Standard QAT pattern: master weights stored FP32; ternary representation only emerges via `intllm.export.export_fjm_v9` to `.fjm` byte format. Sample params: `model.embeddings.weight` (32768×512 fp32), `model.layers.0.attn.i_proj.weight` (512×512 fp32). **Implication for E2.3:** Option A (Medium c.1 as teacher) is **viable** — load via `torch.load(...)['state_dict']` then `model.half()` for FP16 teacher inference. ~140M params / 2 = ~280 MB FP16 in VRAM during student training. **Caveat:** Phase D Medium was English-only training, so teacher has no ID prior → distill loss on ID-side samples expected to provide weak/zero signal (negative transfer risk avoided by sampling-aware distill weight: full distill weight on EN samples, down-weighted or zeroed on ID samples). E2.3 sub-task design must address this; otherwise Option C (train fresh bilingual FP16 teacher, ~30-60 h GPU) becomes the fallback. |
| ~~Q4~~ ✅ **CLOSED v1.0+Q4** | Does the Phase D Mini ablation harness (`make train-mini-ablation`) exist as a Makefile target? | E2.x.3 verification rows all reference this. | **Answer:** Target NOT present. `grep "train-mini-ablation" Makefile` exit 1 (no match). Plan v1.8 §3 PHASE E2.1-2.5 references the target ~5× as `make train-mini-ablation TAG={hadamard,fp8_lmhead,distill,balanced_calib,lang_cond}` producing `paper/intllm/ablations/mini_<TAG>.json`. **Scaffolding plan (becomes E2.1.0 sub-task, NOT a feature):** Mirror existing `bench-canonical-real TAG=X` pattern (Makefile line 158) — parametric target invoking a new driver script `python/phase_d/scripts/train_mini_ablation.py` that takes `--tag <name>` + boolean flags for the 5 features (`--hadamard`, `--fp8-lmhead`, `--distill`, `--balanced-calib`, `--lang-cond`). Driver is thin shim around `intllm.train.train_mini` with feature toggles. JSON schema: `{tag, val_loss, ppl, gate_pass, features_active, training_seconds, ...}` matching `paper/intllm/results/training_intllm-mini.json` shape. Estimated scaffold effort: ~30-45 min (driver + Makefile target + smoke test). **Implication:** add E2.1.0 to E2 sub-task list AHEAD of E2.1+ feature work; without this harness, no E2.x.3 ablation row produces its JSON artifact, which then blocks `verify_paper_tables.py` E2 row checks (§6.9 R7 prevention layer). |
| **Q5** | What's the bilingual coherence baseline (ratio val_loss(ID)/val_loss(EN)) for an UNTRAINED bilingual Mini? | E2.5.4 gate is "ratio ≤1.5× at Mini scale". Need to know the no-feature baseline ratio to interpret deltas. | Add to E2.0 reproduction-Mini-on-bilingual sub-task: report both losses + ratio in `_E2_FINDINGS.md` v1.1. |
| Q6 | Does the existing BitLinearStatTracker harness see ID + EN data correctly when bilingual sampler is plugged in? | E2.4 must NOT regress the existing per-coord adaptive bit allocation (Track B step 1 already shipped this). | Smoke test E2.4.1 (sampler) before any quant changes — verify trackers see expected mix. |
| Q7 | FP8 (E4M3) PyTorch availability on RTX 4090 Laptop + the production `torch` version pinned in fajarquant? | E2.2 may need a compatibility shim. RTX 4090 supports FP8 in hardware (Hopper/Ada gen) but PyTorch ops vary. | `python -c "import torch; print(torch.float8_e4m3fn, torch.cuda.get_device_capability())"` and verify support. |
| Q8 | What's the kernel binary size budget delta if FP8 LM head is adopted? | Plan §1 item 12: ≤ 16 MB total. Current Nova v3.9.0 + IntLLM = 14 MB; ≤ 2 MB headroom. FP8 lm_head = 2× ternary lm_head bytes; for vocab 32K × hidden 768 = 24 MB FP8 vs 12 MB ternary → potentially 12 MB delta. | Compute exact bytes from `HGRNBitConfig` defaults; commit to E2.2 decision doc as binding constraint. |

Q1–Q5 are blocker-class (must close before E2.1 commit). Q6–Q8 can close in-flight as their respective sub-tasks (E2.4 / E2.2) start.

---

## 5. E2.0 ship criteria (what makes this pre-flight CLOSED)

This findings doc is v1.0 (audit + plan only). Promotion to v1.1 (E2.0 CLOSED) requires:

- [x] Q1 attribute mapping for `HGRNBitAttention` output projection committed (closed in §4 above; `block.attn.o_proj`)
- [x] Q2 V28.5 forensic summary + E2.5 option (a/b/c) preliminary recommendation (closed in §4 above; **Option (a) per-lang RMSNorm γ** preliminary; final defer to post-E2.1-2.4 empirical signal)
- [x] Q3 Phase D Medium checkpoint dtype confirmed (FP32 master weights → FP16 teacher viable via `.half()`; English-only caveat noted in §4)
- [x] Q4 `make train-mini-ablation TAG=...` target presence audited (NOT present; E2.1.0 scaffold sub-task added — driver + Makefile target via `bench-canonical-real` pattern; ~30-45 min)
- [ ] Q5 bilingual coherence baseline measured: val_loss(ID), val_loss(EN), ratio at Mini scale on 10% sample (~200M tokens at 60:40)
- [ ] One paragraph "go/no-go" decision per CLAUDE §6.8 R6 — proceed to E2.1 or escalate to user

**Estimated effort to v1.1:** ~2 days human (Q1–Q4 are doc-only) + ~3-5h GPU (Q5 baseline reproduction).

**No GPU is needed yet for v1.0** — current findings closes the audit-and-plan step. v1.0 sign-off below means the audit is complete; the listed open questions are next.

---

## 6. References

- Plan: `docs/FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.8 §3 PHASE E2 (lines ~395-465)
- Predecessor findings: `docs/FJQ_PHASE_E_E1_FINDINGS.md` v1.3 (Phase E1 closure)
- Bilingual corpus spec: `docs/FJQ_PHASE_E_BILINGUAL_CORPUS_V1.md` v1.0
- EN config shim: `python/phase_e/intllm_en.py`
- Quant module: `python/phase_d/intllm/quant.py`
- QAT module: `python/phase_d/intllm/qat.py`
- Train driver: `python/phase_d/intllm/train.py`
- Architecture shim: `python/phase_d/intllm/model.py` + upstream at `python/phase_d/_upstream/mmfreelm/`
- E2 paper anchors: QuaRot (arxiv 2404.00456), SpinQuant (arxiv 2405.16406), Calibrating Beyond English (arxiv 2601.18306), MMfreeLM (Zhu+ 2024)

---

**Sign-off (v1.0):** Phase E2.0 pre-flight AUDIT complete. Plan + surface area + risk-ordered implementation sequence + 8 open questions captured. **Status: v1.0 (audit + plan only) — promotion to v1.1 (E2.0 CLOSED) requires Q1–Q5 closures + 10% sample baseline reproduction, estimated ~2 days human + 3-5h GPU.**

Next session natural first step: close Q1 (HGRNBitAttention attribute path read) — pure code-read, no GPU. Then Q2 (V28.5 forensics) — memory/git-log read. Q3-Q5 follow.

*Document version: 1.0 (E2.0 first audit). Author: Claude Opus 4.7 + Fajar (PrimeCore.id).*
*Phase E v1.8 Tier 1+2 scope. Predecessor: E1 fully closed 2026-04-27.*
