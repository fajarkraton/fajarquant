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
| F.6.1 | Outlier-concentration measurement on Q5-trained Mini checkpoint o_proj inputs (per-channel absmax / per-channel mean ratio histogram via forward-pre-hook capture; reuse `compute_quant_error_per_channel` infrastructure adapted for distribution dump) | ~½ day; CPU-bound. **Hard prerequisite** — if HGRN o_proj inputs don't have outlier concentration (≥3× max/mean ratio), post-hoc rotation has nothing to spread either; F.6.2-6.4 are moot. |
| F.6.2 | Post-hoc QuaRot on Q5-trained Mini checkpoint (canonical recipe: load checkpoint, fuse Hadamard into adjacent weights via algebraic identity, re-evaluate val_loss without re-training) | ~1 day; tests whether E2.1 failure was specifically "training-from-scratch fights rotation" vs "HGRN architecture doesn't have outlier-prone activations." Must be preceded by F.6.1. |
| F.6.3 | Hadamard rotation on i/f/g_proj inputs (against Q1 closure's "non-canonical" warning) | ~1 day implementation + 1 ablation. Tests whether the right insertion point for HGRN was QKV-equivalent paths, not o_proj. Run alongside F.6.2 if budget permits. |
| F.6.4 | Hadamard at Base or Medium scale (rather than Mini) | 4-6h GPU per scale; tests whether outlier concentration emerges at larger scales. SpinQuant paper shows benefit grows with model size. |

**Entry condition for F.6:** Phase E paper accepted (or at least submitted) AND F.6.1 measurement confirms ≥3× max/mean ratio on at least one BitLinear site (the threshold below which outlier-spreading rotations are net cost).

**Reusable infrastructure already shipped** (does NOT need to be re-built in Phase F):
- `intllm.quant.HadamardRotation` — orthogonal Walsh-Hadamard module, 9 unit tests, dtype-agnostic
- `--hadamard` + `--bilingual-data` flags in `train_mini_ablation.py` — diagnostic ablation harness ready
- `register_forward_pre_hook` attach pattern in driver — mirrors E2.4.A; reusable for any "modify input to BitLinear" experiment

Phase F.6 work is about WHEN/WHERE to apply the rotation, not the rotation itself.

---

## 5. Decision artifact (when Phase F kicks off OR is decided NOT to)

Per Phase E v1.8 §F.5: do NOT leave §F as ambiguous "someday" forever. Schedule a yes/no review at **Phase E paper acceptance + 2 weeks**.

When the decision is made:
- **YES:** create `docs/FJQ_PHASE_F_KICKOFF_DECISION.md` with rationale + activation date + initial sub-phase (F.0); cut `FJQ_PHASE_F_PRODUCTION_PLAN.md` v1.0 from this roadmap.
- **NO / postpone again:** create `docs/FJQ_PHASE_F_KICKOFF_DECISION.md` with rationale; record next review checkpoint (e.g., +6 months) so the decision doesn't drift indefinitely.

Either way, the decision is committed as code per CLAUDE §6.8 R6 (mechanical decision gate).

---

*Document version: 1.2 (post-E2.4 + E2.1 sync — F.5 + F.6 entries added). Author: Claude Opus 4.7 + Fajar (PrimeCore.id).*
*v1.0→v1.1 (2026-04-27): added §4.1 F.5 sub-tasks for post-training PTQ calibration as future-work entry from E2.4 negative result.*
*v1.1→v1.2 (2026-04-27): added §4.1 F.6 sub-tasks for post-hoc QuaRot-style Hadamard rotation as future-work entry from E2.1 negative result.*
*Origin: created alongside Phase E v1.7 → v1.8 surgical revision (2026-04-26) to preserve Tier 3 work as Phase F roadmap rather than delete.*
