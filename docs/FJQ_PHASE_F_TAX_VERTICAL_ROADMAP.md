# Phase F — Tax-Vertical Fine-Tune Roadmap (Deferred from Phase E v1.8)

> **Status:** ROADMAP. No commitment to start. Activation gated on Phase E v1.8 success + founder bandwidth re-check + TaxPrime archive access cleared + Phase E paper accepted/arXiv-posted.
>
> **Origin:** Created 2026-04-26 alongside `FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.7 → v1.8 surgical revision (Tier 3 deferral pivot).
>
> **Predecessor plan:** `FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.8 §F (Phase F roadmap section).
>
> **Authoritative dataset spec:** `FJQ_PHASE_E_TAXPRIME_DATASET_SPEC.md` v1.2 (header updated 2026-04-26 to mark Phase F deferral; body unchanged).

---

## 0. Why this document exists

Tier 3 (tax/legal Indonesian vertical AI with audit-trail-proof provenance) was the third blue-ocean wedge identified in Phase E §0. It comprised:

- **E1.3** Vertical fine-tune corpus (TaxPrime archives, public tax law PDFs, court rulings, glossary)
- **E4** entire phase (E4.0 pre-flight + E4.1 SFT + E4.2 bilingual instruct)
- **E5.2** Tax-vertical kernel-path
- **§11** Tax-vertical eval methodology spec
- **§14.1 Lapis 3a** Tax-data policy (TaxPrime real data only)

**v1.8 deferral rationale** (full audit trail in plan v1.8 changelog block + §F.0):

1. **Founder bandwidth honesty** — solo execution + day-job + V31.E1 in flight cannot sustainably absorb 190-270 person-hours of Tier 3 work concurrent with Phase E.
2. **Cleaner first paper** — bilingual base + kernel-context contributions stand alone for MLSys 2027; Tier 3 wedge becomes follow-up paper.
3. **De-risk first artifact** — E1.3 archive ingest was founder-blocked on TaxPrime NDA review; E4+E5.2 downstream of E1.3.

This document tracks what Phase F will deliver when activated, and where to lift content from. It is NOT a commitment to start.

---

## 1. Phase F prerequisites (entry conditions)

Activation requires ALL of:

1. ✅ **Phase E v1.8 base model shipped.** Tier 1+2 paper artifact + `verify_phase_e_tables.py --strict` exit 0. Bilingual Medium checkpoint exists as SFT base.
2. ✅ **Founder bandwidth re-check committed.** `docs/FJQ_PHASE_F_BANDWIDTH_CHECK.md` (NEW per plan v1.8 §F.2) explicit yes/no on whether 70 person-hours over 13 weeks is sustainable given concurrent commitments.
3. ✅ **TaxPrime archive access cleared.** Phase E E0.0.2 deferred sub-task reactivated — `docs/FJQ_PHASE_E_TAXPRIME_DATA_REVIEW.md` committed before E1.3-equivalent ingest begins.
4. ✅ **Phase E paper accepted or arXiv-posted.** De-risks Phase F as standalone follow-up paper rather than holding co-publication dependency on Phase E review cycles.

If ANY prerequisite fails, Phase F holds at "ROADMAP" status; no abort doc required (this is opt-in, not gated by FAIL paths).

---

## 2. Phase F sub-phase summary

Lifted from Phase E v1.8 §3 PHASE E4 + §3 PHASE E5.2 (which contain DEFERRED summary blocks pointing here). Detailed sub-task tables to be reproduced verbatim in a future `FJQ_PHASE_F_PRODUCTION_PLAN.md` v1.0 when Phase F activates.

### F.0 Pre-flight audit

Lifted from Phase E v1.8 §3 PHASE E4 § E4.0:

| Task | Verification |
|---|---|
| F.0.1 Dataset acceptance audit | `python scripts/verify_taxprime_dataset.py --version v1.0 --strict` exit 0; checks all §11 acceptance items |
| F.0.2 PII residual scan | `scripts/check_pii_redaction.py --dataset data/tax_id_corpus_v1/ --strict` exit 0 |
| F.0.3 Cross-contamination check | `scripts/check_train_eval_overlap.py` — 0 examples appear in both train and eval (id + content hash) |
| F.0.4 Baseline pass@1 (no FT) | bilingual Medium evaluated on tax eval set; result calibrates pass@1 threshold |

### F.1 Tax/legal Indonesian fine-tune

Lifted from Phase E v1.8 §3 PHASE E4 § E4.1 (TaxPrime real data only — Lapis 3a policy):

| Task | Verification |
|---|---|
| F.1.1 SFT recipe (LoRA r=64 + full-tune comparison) | `make finetune-tax-lora` and `make finetune-tax-full` produce checkpoints |
| F.1.2 Eval against TaxPrime 100-200-prompt eval set | `make eval-tax-vertical` outputs `paper/intllm/results/tax_eval.json` |
| F.1.3 Pass@1 ≥ 65% on tax-rule retrieval | gate per Phase E v1.8 §1 item 4 (DEFERRED there; reactivated here) |
| F.1.4 Hallucination check | manual review of 50 random outputs by founder (single-rater per spec v1.2) |
| F.1.5 Citation-format compliance | auto-check: model output must contain valid regulatory citation in `UU/PMK/PER-DJP/SE` format for any factual claim; pass rate ≥ 90% |

### F.2 Optional: bilingual-instruct version (general-purpose ID+EN chat)

Lifted from Phase E v1.8 §3 PHASE E4 § E4.2:

| Task | Verification |
|---|---|
| F.2.1 Instruct dataset assembly (Indonesian Alpaca-style) | corpus checked in |
| F.2.2 SFT run | checkpoint produced |
| F.2.3 Eval on IndoBench instruction-following | results committed |

### F.3 Tax-vertical kernel-path

Lifted from Phase E v1.8 §3 PHASE E5 § E5.2 (DEFERRED there):

| Task | Verification |
|---|---|
| F.3.1 Port tax-vertical FT checkpoint to `.fjm` | export script ran |
| F.3.2 `make test-tax-vertical-kernel-path` (5 invariants incl. tax-pass@1) | green in CI |
| F.3.3 Demo shell command in Nova: `nova> tax-query "Apa tarif PPh badan tahun 2026?"` returns coherent answer | manual smoke test |

### F.4 Phase F paper + IP extension

| Task | Verification |
|---|---|
| F.4.1 Phase F paper draft | extends Phase E paper with vertical fine-tune section + tax eval methodology (lifted from Phase E v1.8 §11) + tax-vertical kernel-path serving numbers |
| F.4.2 Patent extension filing (if Phase E patent already filed) | tax-vertical claims as continuation-in-part |
| F.4.3 Submission target | venue TBD — likely follow-up to Phase E venue, or domain-specific (LegalTech / FinTech AI) |

---

## 3. Estimated Phase F effort (informational)

From Phase E v1.8 §F.3:

| Sub-phase | Calendar (solo, ~10h/wk) | Human (h) | GPU (h) | Cost |
|---|---|---|---|---|
| F.0 Pre-flight | 1 wk | 5 | 0 | $0 |
| F.1 Vertical corpus assembly (E1.3 lift) | 2 wk | 10 | ~1 | $0 |
| F.2 SFT + eval (E4.0 + E4.1 lift) | 4 wk | 25 | ~30-60 | $0 |
| F.2.x Bilingual instruct (E4.2 lift) | 2 wk | 10 | ~10-20 | $0 |
| F.3 Tax-vertical kernel-path (E5.2 lift) | 1 wk | 5 | ~1 | $0 |
| F.4 Phase F paper + IP | 3 wk | 15 | 0 | $0 |
| **Total Phase F solo** | **~13 weeks (~3 months)** | **~70 h** | **~42-82 h** | **$0** |

---

## 4. Companion docs (preserved verbatim for Phase F lift)

- `docs/FJQ_PHASE_E_TAXPRIME_DATASET_SPEC.md` v1.2 — authoritative dataset spec (header marks Phase F deferral; body unchanged)
- `docs/FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.9 §11 — eval methodology spec (DEFERRED header; body preserved)
- `docs/FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.9 §F — Phase F roadmap section (this doc's parent)

When Phase F activates, the natural sequence is:
1. Cut `FJQ_PHASE_F_PRODUCTION_PLAN.md` v1.0 — lift §F + §11 + this doc's §2 verbatim
2. Bump TaxPrime spec to v1.3 with "Phase F active" header
3. Start F.0.1 (dataset acceptance audit)

---

## 4.1 Additional Phase F future-work entries (from Phase E negative results)

Items deferred to Phase F because the Phase E investigation surfaced specific algorithmic gaps that need post-Phase-E-paper investigation:

### F.5 Post-training PTQ-style activation calibration on held-out data

**Origin:** Phase E E2.4 closure 2026-04-27 (`FJQ_PHASE_E_E2_BILINGUAL_CALIB_DECISION.md` v1.0 §4.3). The all-time-max `running_max` accumulator for `BitLinearStatTracker` produced an 83× WORSE quantization error on outlier channels vs upstream baseline — calibration scale locked to early-training peaks 10-100× larger than steady-state activations.

**What to investigate in Phase F:**

| Sub-task | Description | Risk / cost |
|---|---|---|
| F.5.1 | Post-training PTQ calibration on a fixed held-out batch (SmoothQuant arxiv 2211.10438 Algorithm 1 pattern) | Standard literature approach; should match or beat upstream `q_baseline` if implemented correctly. ~1 week solo + 1 Mini ablation. |
| F.5.2 | Exponential moving average for `running_max` (e.g. `ema_alpha = 0.99`) | Late-training stability dominates over early-training peaks; partial mitigation only. ~1 day implementation + 1 ablation. |
| F.5.3 | Skip calibration during warmup (only accumulate `running_max` after `warmup_steps`) | Cheapest mitigation; addresses early-peak issue most directly. ~½ day. **WORTH TRYING FIRST** before F.5.1. |
| F.5.4 | Option B (`IntLLMBitLinear` wrapper applying maps at forward-time) — was deferred per E2.4.0 Q10 | Independent of calibration source — wrapper consumes whatever maps are saved. Re-evaluate once F.5.3 demonstrates the calibrated path can beat baseline. |

**Entry condition for F.5:** Phase E paper accepted (or at least submitted) AND at least one of F.5.1–F.5.3 shows ≥+10% MSE reduction on outlier channels in a sub-day-effort smoke run. This filters out the case where the all-time-max issue isn't the dominant cause of E2.4's failure — if the cheap fixes don't help, the problem is deeper than calibration scheme and Option C decision stands.

**Reusable infrastructure already shipped** (does NOT need to be re-built in Phase F):
- `intllm.qat.{BitLinearStatTracker, attach_stat_trackers, save_calibration_maps, compute_bit_allocation, compute_channel_permutation}` — production-integrated since E2.4.A
- `intllm.quant.activation_quant_per_channel` — standalone math helper, dtype-agnostic
- `intllm.eval.compute_quant_error_per_channel` — driver, atomic JSON output, gate-checking logic
- `bilingual_stream` for calibration data
- `train_mini_ablation.py --balanced-calib` flag for smoke runs

Phase F.5 work is purely about replacing `running_max` accumulator semantics and re-running the metric.

### F.6 Post-hoc QuaRot-style Hadamard rotation on pre-trained checkpoint

**Origin:** Phase E E2.1 closure 2026-04-27 (`FJQ_PHASE_E_E2_HADAMARD_DECISION.md` v1.0 §4.3). Training-from-scratch with `HadamardRotation` on `block.attn.o_proj` produced a +0.12 nat REGRESSION vs Q5 baseline (gate required ≥+0.05 nat IMPROVEMENT). The literature precedent (QuaRot arxiv 2404.00456, SpinQuant arxiv 2405.16406) applies rotation POST-hoc to pre-trained models; training-from-scratch loses the primary benefit.

**What to investigate in Phase F:**

| Sub-task | Description | Risk / cost |
|---|---|---|
| F.6.1 | Outlier-concentration measurement on trained Mini checkpoint o_proj inputs (per-channel absmax / per-channel mean ratio histogram via forward-pre-hook capture). | ✅ **CLOSED 2026-04-27** via `python/phase_d/scripts/measure_outlier_concentration.py` against `python/phase_d/checkpoints/mini/mini_final.pt`; 100 batches of bilingual_stream. **Result: STRONG concentration** — o_proj mean ratio $51.6\times$, max $421\times$ (well above $\geq 3\times$ threshold). mlp.down_proj also very concentrated ($40\times$ / $250\times$). i/f/g_proj moderate ($5.3\times$ / $7.8\times$). F.6.2 is GO; F.6.3 also strongly recommended given mlp.down_proj's concentration. Artifact: `paper/intllm/ablations/outlier_concentration_mini.json`. |
| F.6.2 | Post-hoc QuaRot on Q5-trained Mini checkpoint (canonical recipe: load checkpoint, fuse Hadamard into adjacent weights via algebraic identity, re-evaluate val_loss without re-training) | ~1 day; tests whether E2.1 failure was specifically "training-from-scratch fights rotation" vs "HGRN architecture doesn't have outlier-prone activations." Must be preceded by F.6.1. |
| F.6.3 | Hadamard rotation on i/f/g_proj inputs (against Q1 closure's "non-canonical" warning) | ~1 day implementation + 1 ablation. Tests whether the right insertion point for HGRN was QKV-equivalent paths, not o_proj. Run alongside F.6.2 if budget permits. |
| F.6.4 | Hadamard at Base or Medium scale (rather than Mini) | 4-6h GPU per scale; tests whether outlier concentration emerges at larger scales. SpinQuant paper shows benefit grows with model size. |

**Entry condition for F.6:** Phase E paper accepted (or at least submitted) AND F.6.1 measurement confirms ≥3× max/mean ratio on at least one BitLinear site (the threshold below which outlier-spreading rotations are net cost).

**Reusable infrastructure already shipped** (does NOT need to be re-built in Phase F):
- `intllm.quant.HadamardRotation` — orthogonal Walsh-Hadamard module, 9 unit tests, dtype-agnostic
- `--hadamard` + `--bilingual-data` flags in `train_mini_ablation.py` — diagnostic ablation harness ready
- `register_forward_pre_hook` attach pattern in driver — mirrors E2.4.A; reusable for any "modify input to BitLinear" experiment

Phase F.6 work is about WHEN/WHERE to apply the rotation, not the rotation itself.

### F.7 FP8 (E4M3) mixed-precision for `lm_head` + `attn.W_o`

**Origin:** Path A v1.10 cut decision 2026-04-27 — Phase E2.2 deferred to here. Was originally planned as Mini-scale ablation but never executed; Path A stops Phase E2 after 2 negative results (E2.4 + E2.1) and pivots to paper.

**What to investigate in Phase F:**

| Sub-task | Description | Risk / cost |
|---|---|---|
| F.7.0 | Pre-flight: verify `torch.float8_e4m3fn` availability on RTX 4090 Laptop + production `torch` version pin (was Q7 in E2.0 findings, deferred). `python -c "import torch; print(torch.float8_e4m3fn, torch.cuda.get_device_capability())"`. | ~½h. Determines whether F.7 is buildable on this hardware at all. |
| F.7.1 | Implement FP8 (E4M3) cast for `lm_head` and `attn.W_o` weights only (not full BitLinear); document binary-size delta vs ternary. | ~2 days human; may need PyTorch upgrade if current pinned version lacks fp8. |
| F.7.2 | Mini-scale ablation `make train-mini-ablation TAG=fp8_lmhead --fp8-lmhead --bilingual-data` (~50 min GPU). | Standard E2-style run; like-for-like vs Q5 baseline. |
| F.7.3 | Decision doc `FJQ_PHASE_E_E2_FP8_DECISION.md` — adopt iff ≥+0.05 nat val_loss AND kernel binary size ≤ 16 MB. |
| F.7.4 | If PASS: extend to Base / Medium scale; consider as candidate for paper update / second paper. |

**Entry condition for F.7:** Path A paper accepted (or at least submitted) AND F.7.0 confirms PyTorch FP8 is buildable on this hardware. If F.7.0 fails (e.g., `torch.float8_e4m3fn` not available), defer further until PyTorch / Triton upgrade.

**Risk specific to F.7:** unlike F.5/F.6 which adapt existing primitives, F.7 may require a PyTorch upgrade that destabilizes the rest of the stack. Should be tested on a separate branch.

### F.8 FP16 distillation during QAT (teacher-student)

**Origin:** Path A v1.10 cut decision 2026-04-27 — Phase E2.3 deferred to here.

**What to investigate in Phase F:**

| Sub-task | Description | Risk / cost |
|---|---|---|
| F.8.1 | Add teacher-student loss to `intllm.qat` (KL on logits + MSE on hidden states). Teacher = Phase D Medium c.1 FP32 master (per E2.0 Q3 closure: viable via `model.half()`; ~280 MB FP16 in VRAM during student training). | ~3 days human. |
| F.8.2 | Address negative-transfer risk per Q3 caveat: teacher is English-only Phase D Medium; bilingual student distilling from EN-only teacher needs sampling-aware distill weight (full weight on EN samples, down-weighted on ID samples). | Adds complexity; needs ablation of weighting scheme itself. |
| F.8.3 | Mini-scale ablation `make train-mini-ablation TAG=distill --distill --bilingual-data`. | Standard E2-style run. |
| F.8.4 | Decision doc `FJQ_PHASE_E_E2_DISTILL_DECISION.md`. |
| F.8.5 | If F.8.1-F.8.4 negative-result-on-EN-only-teacher: pivot to Option C — train a fresh bilingual FP16 teacher (~30-60 h GPU additional). High cost; only pursue if F.8 FAIL is clearly attributable to teacher-language-mismatch and the architecture itself shows promise. |

**Entry condition for F.8:** Path A paper accepted AND F.7 outcome documented (FP8 has lower implementation risk and may inform whether mixed-precision path is even worth pursuing).

### F.9 Language-conditioned design (per-lang RMSNorm γ / LoRA / MoE-light)

**Origin:** Path A v1.10 cut decision 2026-04-27 — Phase E2.5 deferred to here.

**What to investigate in Phase F:**

| Sub-task | Description | Risk / cost |
|---|---|---|
| F.9.1 | Re-evaluate option (a/b/c) selection given F.5/F.6/F.7/F.8 outcomes. Original preliminary recommendation (Option (a) per-lang RMSNorm γ) was contingent on post-E2.1-2.4 empirical signal which was 0/2 FAIL — preliminary may need revision. | ~½ day; literature + decision doc. |
| F.9.2 | Implement chosen option as `intllm.arch.LanguageConditioned*` classes with unit tests. | ~3-5 days human depending on option (a easiest; c hardest). |
| F.9.3 | Mini-scale ablation `make train-mini-ablation TAG=lang_cond --lang-cond X --bilingual-data`. | Standard E2-style run. |
| F.9.4 | Decision doc — adopt iff coherence ratio improvement ≥+0.10× toward 1.5× absolute (Q5 baseline 1.77×; gate 1.67× achievable). |
| F.9.5 | If PASS at Mini, optionally extend to Base scale to confirm scaling-friendliness. |

**Entry condition for F.9:** Path A paper accepted AND at least one of F.5/F.6/F.7/F.8 has demonstrated that ANY Phase E-class feature can help under our regime (post-hoc or otherwise). If all of F.5-F.8 fail, F.9 is unlikely to flip the trend; demote to "experimental research" without paper-submission target.

---

## 4.2 Additional Phase F future-work entries (from V32-prep hardware-acceleration research, 2026-04-28)

Items deferred to Phase F because the V32-prep autonomous research surfaced specific hardware-acceleration gaps that need post-Phase-E-paper investigation. These are **input-only** at V31 — no V31 work blocked or re-opened on their account; they stack onto the §4.1 entries when Phase F kicks off.

### F.10 GPU 2:4 structured sparsity via Sparse-BitNet recipe (NVIDIA Ada Lovelace)

**Origin:** V32-prep research 2026-04-28 (this doc's parallel turn). User asked whether RTX 4090's 4th-gen Tensor Core 2:4 structured sparsity (165 TFLOPS dense FP16 → 330 TFLOPS sparse) is being exploited by IntLLM. Audit confirmed: NO — Phase D inference path runs `F.linear()` cuBLAS dense FP16 via upstream `mmfreelm/ops/fusedbitnet.py:460` with zero `torch.sparse.SparseSemiStructuredTensor` / cuSPARSElt references in the codebase. Trained Mini checkpoint stores FP32 dense master weights (0% exact zeros at rest; ternary discretization only at forward time via STE).

**What to investigate in Phase F:**

| Sub-task | Description | Risk / cost |
|---|---|---|
| F.10.1 | Reproduce **Sparse-BitNet** (Microsoft Research, arXiv 2603.05168, github.com/AAzdi/Sparse-BitNet, MIT) at Mini scale. Sparse-BitNet's empirical claim: 1.58-bit BitNet weights post-training are ~42% naturally zero with tri-modal "quantization-valley" histogram; under 2:4 structured sparsity, ternary BitNet degrades 3.3× less than FP16 baseline. Their recipe uses **sparse-from-scratch QAT** with Dual-STE + trainable mask. | ~5h GPU + 3 days human. Triton kernels are public, no wgmma → RTX 4090 compatible. Single cleanest "validate the 2× compute headroom is real for IntLLM" experiment. |
| F.10.2 | If F.10.1 passes ≥+0.05 nat val_loss AND ≥1.10× wall-clock prefill speedup at Mini: extend to Base / Medium scale via re-train-from-scratch. Speedup grows with model size per Sparse-BitNet results (1.30× prefill at 65K seq on A100/B200). | ~10h GPU + 1 week human across 2 scales. |
| F.10.3 | Stack 2:4 onto FP8/FP16 islands (lm_head + attn output projection) when F.7 FP8 path is active. Ternary BitLinear core stays packed-int2 / Triton; 2:4 only at the precision edges. | Depends on F.7 outcome; ~3 days. |

**Entry condition for F.10:** Path A paper accepted AND F.10.1 passes both gates (val_loss + wall-clock). The wall-clock gate is critical because at Mini–Medium scale + batch=1 the model is memory-bandwidth bound; theoretical 2× compute headroom does NOT automatically translate to wall-clock wins. Sparse-BitNet itself measured 1.30× prefill / 1.18× decode at batch=128, so realistic IntLLM expectation is **1.10–1.20× end-to-end**, not 2×.

**Reusable infrastructure already shipped** (does NOT need to be re-built):
- Phase D training pipeline (re-target with Sparse-BitNet's mask-creator kernel)
- `scripts/train_mini_ablation.py --hadamard --bilingual-data` pattern for ablation rows
- `verify_intllm_tables.py` R7 audit gate (extends naturally to a `mini_sparse_bitnet.json` row)

**Known dead-end (do NOT pursue):** Post-hoc 2:4 prune of an existing Phase D checkpoint (`mini_final.pt`, `medium_final.pt`). Sparse-BitNet Table 4 explicitly shows dense-to-sparse late-switch is meaningfully worse than sparse-from-scratch (PPL 27.48 vs 26.31). Combined with our V31.E2 pattern (Mini-scale + training-from-scratch hostile to features ported from post-hoc literature: E2.1 Hadamard FAIL +0.12 nat, E2.4 balanced_calib FAIL −82.13), retro-fit will almost certainly fail gates.

---

### F.11 CPU bitnet.cpp TL2 kernel port to Fajar Lang `@kernel` context

**Status (2026-05-01): DEMOTED to PERMANENT-DEFERRED / future-work.** F.11
chain consumed ~31 cumulative hours (F.11.0-3 setup, F.11.4 parity
iterations 1-7 + Path B, Tier 2E kernel disassembly, Branch X-real
attempted port). Vendored kernel + FFI binding + scalar baseline + magnitude
encoder all bit-exact and shipping. **Sign-byte parity gap unresolved:**
row-uniform `+32, -31, -1` cycle on parity test attributable to multiple
layout bugs. Branch X-real `verbatim port` attempt blocked by upstream
encoder structural gap (raises `NotImplementedError` for K=256). Per F.13.3
dispatch verdict (decision-doc §11), FajarOS Nova's deployment workload is
batch=1, ≤100M params, where the F.11.3 scalar Rust path (50-100 tok/s) is
adequate for the kernel-resident embedded-ML use case. TL2 acceleration
(projected 200-400 tok/s) would be 3-5× faster but is **not critical-path**
for any planned FajarOS deployment. Re-entry condition documented at end
of this section.

**Origin:** V32-prep research 2026-04-28. User asked "bagaimana dengan teknologi yang ada di CPU yang kita pakai?" (Indonesian: what about the CPU technology we use?). Detected hardware: Intel Core i9-14900HX (Raptor Lake Refresh-HX, 24C/32T, AVX-VNNI present, AVX-512 + AMX absent, DDR5-5600 dual-channel ~70 GB/s sustained). Codebase audit confirmed: ZERO CPU-targeted SIMD code in either `fajarquant/python/phase_d/` or `Fajar Lang/src/`. FajarOS Nova kernel-path runs ternary inference **pure scalar** inside `@kernel` context — leaves all CPU SIMD performance on the table.

**Critical insight from research:** AVX-VNNI's `VPDPBUSD` accelerates **INT8×INT8** — NOT {-1,0,+1}×INT8. Both bitnet.cpp's TL2 (x86) and llama.cpp's TQ2_0 use `vpshufb-LUT + vpmaddubsw + vpaddw`, NOT VPDPBUSD. Having AVX-VNNI does NOT automatically accelerate the dominant ternary kernel.

**What to investigate in Phase F:**

| Sub-task | Description | Risk / cost |
|---|---|---|
| F.11.1 | Wrap **microsoft/BitNet** TL2 x86 kernel (~3000 LOC C, MIT, AVX2-only — no AVX-512 dependency, boots on Atom-class x86) via Fajar Lang FFI v2 from `@kernel` context. Bind to BitLinear forward in the FajarOS Nova kernel-path inference loop. | ~3-5 days human. Engineering risk near-zero (Microsoft maintains it). HGRN gated-recurrence projections (i_proj, f_proj, g_proj) aren't standard BitLinear-shaped — measure whether they fall through to scalar gracefully. |
| F.11.2 | Replace pure-scalar `runtime/ml/stack_tensor.rs:209-238` matmul (triple-nested f64 loop, no Cranelift SIMD hint) with TL2-equivalent path emitted by Fajar Lang's `@simd` token + `target_feature("+avx2")` Cranelift backend. Token + builtin already exist (`hw_simd_width()` in `interpreter/eval/builtins.rs:2724`) but no actual AVX2 emission today. | ~1 week human. Higher risk because Fajar Lang Cranelift backend hasn't emitted target-feature intrinsics before. |
| F.11.3 | Validate against published bitnet.cpp benchmark: BitNet b1.58 2B4T = 29 ms/tok decode on i7-13800H Surface Laptop Studio 2 (8 threads). Our 14900HX (24C/32T, +400 MHz boost, +50% L3) target: 40-50 tok/s on equivalent 2B model. | ~½ day; existing `paper/intllm/results/bench_baselines_microsoft__bitnet-b1_58-2B-4T.json` will move from N/A → measured. |

**Entry condition for F.11:** Path A paper accepted AND FajarOS Nova kernel-path tokens-per-second target (currently ≥10 tok/s for Medium per E2_BILINGUAL_KERNEL_PRODUCTION_PLAN.md:208) needs uplift OR a 2B-class production deployment is on the roadmap. This is the **single biggest CPU lift available** and aligns directly with the FajarOS embedded-ML vision; expected speedup vs current scalar Nova kernel-path = **5–7× decode tok/s**.

**Reusable infrastructure** (NEW, does not exist yet):
- microsoft/BitNet TL2 kernel — vendor via Cargo external crate or git submodule
- Fajar Lang FFI v2 `extern "C"` bridge for kernel-context calls
- FajarOS Nova `km_mf_*` symbol table extension (matmulfree.fj path)

**Re-entry conditions for F.11 (any one triggers reactivation):**
1. Paper v2 reviewer specifically requests bit-exact TL2 parity on
   fajarquant infrastructure claim.
2. FajarOS deployment ask shifts to a workload where current scalar tok/s
   (50-100 on Mini) is insufficient AND production blocker is the kernel
   path (not training/quality).
3. Microsoft/BitNet upstream releases an encoder that supports custom
   shapes (M, K) including (256, 256) — would unblock verbatim port.
4. A new contributor with reverse-engineering / kernel disassembly
   bandwidth volunteers the work (printf-instrument the kernel for
   sign-byte read trace, ~3-4h). Documented strategic Option B in
   `docs/FJQ_PHASE_F_F11_BRANCH_X_REAL_FINDINGS.md` §5.

Until any of those conditions: F.11 stays in this PERMANENT-DEFERRED state.
Vendored kernel + FFI infrastructure remain in the codebase as a
shipping-ready foundation; F.11.3 scalar baseline is the production path.

---

### F.12 AVX-VNNI for non-ternary islands (lm_head + attention output projection)

**Origin:** Same research turn as F.11. Insight: AVX-VNNI accelerates INT8×INT8 dot products via `VPDPBUSD ymm` (32 MAC per uop, throughput 1/cycle on Golden Cove P-core). For ternary BitLinear weights this is the WRONG instruction (see F.11). But for the FP8/INT8 islands that Phase F.7 will introduce (lm_head + attention output projection), VNNI wins ~2× over plain AVX2.

**What to investigate in Phase F:**

| Sub-task | Description | Risk / cost |
|---|---|---|
| F.12.1 | Pre-flight: confirm Fajar Lang Cranelift backend can emit `vpdpbusd ymm` via raw inline-assembly intrinsic OR via `target_feature("+avxvnni")` attribute. If neither path works, escalate to LLVM backend (which has well-documented `@llvm.x86.avx_vnni.vpdpbusd_256` intrinsic). | ~½ day. May be already-working — depends on Cranelift's AVX-VNNI support state, which has been improving. |
| F.12.2 | Implement INT8 GEMM kernel for lm_head + attn.W_o using VPDPBUSD on i9-14900HX. Measure throughput vs AVX2-only baseline on those layers. Realistic gain: ~2× on those layers individually. | ~3 days human. |
| F.12.3 | End-to-end measurement: lm_head is ~10–15% of decode time on small models, so end-to-end gain ~1.3–1.5× when stacked on top of F.11's TL2 kernel for the ternary BitLinear core. **Hybrid pays for itself; full-VNNI port for the ternary core does NOT** (see F.11 critical insight). | ~½ day measurement. |

**Entry condition for F.12:** Phase F.7 (FP8/INT8 lm_head + attn output) has shipped AND F.11.1 has shipped (TL2 kernel for ternary core). F.12 is a layered optimization on top of those two — without F.11 the dominant matmul stays unaccelerated and VNNI's 1.3–1.5× end-to-end gain shrinks to 5–10%.

**Reusable infrastructure:**
- F.7 FP8/INT8 path (when shipped)
- F.11 TL2 ternary kernel (when shipped)
- Fajar Lang `@simd` annotation token + `hw_simd_width()` builtin (already exists)

---

### F.13 CPU-vs-GPU dispatch heuristic in FajarOS Nova kernel-path runtime

**Status (2026-04-30):** F.13.0 + F.13.3 + F.13.5 SHIPPED in Branch Z-narrow scope. F.13.1 + F.13.2 DEFERRED. Verdict: static rule "CPU default; GPU optional for batch ≥ 8." See `FJQ_PHASE_F_F13_DISPATCH_DECISION.md` (paper v2 §6 narrative ready). Verify gate: `make verify-f13-decision` (19/19 PASS).

**Origin:** Same research turn as F.11/F.12. Insight: at batch=1 + small model (≤100M params), CPU + bitnet.cpp TL2 kernel can **beat the GPU** because PCIe + cudaLaunch overhead dominates over the actual GEMM cost. Concrete numbers: Mini (22M, ~5 MB packed) is L2-resident → "hundreds of tok/s on CPU"; GPU at batch=1 lands ~80–120 tok/s for the same size due to dispatch overhead. This crossover is the ACTUAL differentiator vs llama.cpp / transformers which assume "GPU always wins."

**What to investigate in Phase F:**

| Sub-task | Description | Risk / cost |
|---|---|---|
| F.13.0 | Pre-flight audit. **CLOSED 2026-04-30** — see `FJQ_PHASE_F_F13_FINDINGS.md`. Surfaced GPU driver blocker that forked Branch Z into Z-narrow (no live measurements). | ~1h actual. |
| F.13.1 | Build cost model: (model size in bytes, batch size, sequence length, KV cache size) → predicted CPU tok/s vs GPU tok/s. Calibrate on i9-14900HX + RTX 4090 Laptop measurements across {Mini, Base, Medium, 2B} × {batch=1, 8, 32}. **DEFERRED** — needs nvidia driver + F.11 parity for the TL2 datapoint. Z-narrow projects from BitNet 2B4T anchor instead. | ~1 week human (when both prereqs return). |
| F.13.2 | Implement runtime dispatch in FajarOS Nova kernel-path: at inference start, evaluate cost model + select CPU or GPU path. Both paths must be available (F.11 + GPU export). **DEFERRED** — F.13.3 verdict says static rule supersedes runtime dispatch for FajarOS workload. Re-entry gates F.13.2-A and F.13.2-B documented in decision-doc §5.2. | ~3 days. |
| F.13.3 | Decision-doc mode: if cost model says crossover happens beyond what FajarOS realistically deploys (i.e., FajarOS embedded use case is always small-model batch=1), simplify to "CPU is the default; GPU is optional for batch ≥ N" rather than runtime dispatch. Likely outcome given embedded-ML vision. **CLOSED 2026-04-30** — verdict 3/3 G1+G2+G3 PASS, N=8. Static rule confirmed. See `FJQ_PHASE_F_F13_DISPATCH_DECISION.md`. | ~½ day est, ~3h actual. |
| F.13.5 | Prevention layer: pinned-anchors fixture (`tests/fixtures/f13_dispatch_anchors.toml`) + verify script (`scripts/verify_f13_dispatch.py`) + Makefile gate (`make verify-f13-decision`) + pre-commit hook layer 5 (conditional on staged F.13 docs/fixture/script). **CLOSED 2026-04-30** — 19/19 PASS. | ~1.5h actual. |

**Entry condition for F.13:** F.11 has shipped (need a real CPU path to compare GPU against). F.13 is the natural completion of the F.11 + F.12 stack; without it, FajarOS Nova kernel-path doesn't exploit the CPU-wins-batch-1 crossover that is unique vs commodity LLM inference frameworks.

**Reusable infrastructure:**
- F.11 (mandatory prereq)
- F.12 (optional but tightens the crossover)
- FajarOS Nova kernel-path runtime entry point (already exists, currently CPU-only by design per `FJQ_PHASE_D_PRODUCTION_PLAN.md` line 72-83)

---

**Cross-cutting note for §4.2 (F.10–F.13):** these four entries are NOT prerequisite for any §4.1 entry (F.5–F.9). They live in a parallel hardware-acceleration track. Natural ordering when Phase F activates:
1. §4.1 first (F.5/F.6 = algorithmic mitigations of E2.1 + E2.4 negative results — paper-defining work)
2. §4.2 second (F.10–F.13 = hardware-acceleration uplift on top of whatever §4.1 produced)

Premature focus on §4.2 risks shipping a fast version of a wrong algorithm. Precondition: §4.1 must have at least one positive ablation row before §4.2 work justifies investment.

---

## 5. Decision artifact (when Phase F kicks off OR is decided NOT to)

Per Phase E v1.8 §F.5: do NOT leave §F as ambiguous "someday" forever. Schedule a yes/no review at **Phase E paper acceptance + 2 weeks**.

When the decision is made:
- **YES:** create `docs/FJQ_PHASE_F_KICKOFF_DECISION.md` with rationale + activation date + initial sub-phase (F.0); cut `FJQ_PHASE_F_PRODUCTION_PLAN.md` v1.0 from this roadmap.
- **NO / postpone again:** create `docs/FJQ_PHASE_F_KICKOFF_DECISION.md` with rationale; record next review checkpoint (e.g., +6 months) so the decision doesn't drift indefinitely.

Either way, the decision is committed as code per CLAUDE §6.8 R6 (mechanical decision gate).

---

*Document version: 1.3 (Path A absorption — F.5+F.6+F.7+F.8+F.9 entries). Author: Claude Opus 4.7 + Fajar (PrimeCore.id).*
*v1.0→v1.1 (2026-04-27): added §4.1 F.5 sub-tasks for post-training PTQ calibration as future-work entry from E2.4 negative result.*
*v1.1→v1.2 (2026-04-27): added §4.1 F.6 sub-tasks for post-hoc QuaRot-style Hadamard rotation as future-work entry from E2.1 negative result.*
*v1.2→v1.3 (2026-04-27): added §4.1 F.7 (FP8 mixed-precision, deferred E2.2) + F.8 (FP16 distillation, deferred E2.3) + F.9 (language-conditioned design, deferred E2.5) as Path A scope-cut absorption.*
*Origin: created alongside Phase E v1.7 → v1.8 surgical revision (2026-04-26) to preserve Tier 3 work as Phase F roadmap rather than delete.*
