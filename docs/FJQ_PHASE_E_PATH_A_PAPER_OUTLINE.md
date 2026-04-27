# Phase E Path A — Paper Outline (v1.0, MLSys 2027 target)

> **Status:** Path A scope decision committed 2026-04-27 — STOP Phase E2 after 2 honest negative results (E2.4 + E2.1); CUT E2.2 / E2.3 / E2.5 (deferred to Phase F.7-F.9); REMOVE E2.6 combined ablation; pivot directly to Phase E3 (bilingual scale-up) + Phase E4 (paper LaTeX).
>
> **Replaces:** the original Phase E v1.0-v1.9 paper plan that assumed 5 outlier-handling features each ≥+0.05 nat at Mini scale.
>
> **Driver:** 0/2 attempted features helped under our Mini-scale + training-from-scratch + bilingual regime. Continuing E2.2/E2.3/E2.5 is high-cost (3 × ~50 min GPU each + 2-3 weeks human total) for low expected paper yield (likely 0-2 of 3 help, by §3 meta-pattern reasoning). Re-frame is honest, defensible, faster.

---

## 1. Headline Paper Title

**Working title (MLSys 2027 — 10 page submission):**

> **IntLLM: Compiler-Verified 1.58-Bit Ternary Language Models for Operating-System Kernel Inference**

**Tagline (one-sentence):**

> A scaling-law-validated 1.58-bit ternary LM family (22M-74M params) trained end-to-end and deployed inside the operating-system kernel, with honest negative results on outlier-handling features at this small-scale training-from-scratch regime.

---

## 2. Five Headline Contributions

The paper claims exactly these — no more, no less. Each is backed by a runnable artifact.

### C1. Scaling chain at training-time-quant validates Chinchilla-style behavior at small scale

**Claim:** First systematic 3-data-point scaling-law validation for 1.58-bit ternary training-from-scratch at the 22M-74M parameter range. Mini → Base → Medium gates PASS with monotonically widening margins (val_loss 4.38 → 3.99 → 3.72; gate margins 0.12 → 0.21 → 0.28 nat). Verified by `paper/intllm/results/training_intllm-{mini,base,medium}.json`.

**Why it matters:** BitNet b1.58 paper (Microsoft, 2024) reports scaling at ≥700M only; we close the gap from 22M up. Useful for embedded / kernel-context deployments where 700M+ is infeasible.

**Tables/figures:** Table 1 (per-config gate margins), Figure 1 (val_loss vs n_params curve), Table 2 (wikitext word-PPL + bits-per-byte; lambada_openai PPL + acc; arc_easy + openbookqa).

### C2. Bilingual extension produces clean ternary multilingual baseline

**Claim:** Phase E1 corpus v1.0 = 25.67 B tokens at 60:40 ID:EN, 0% synthetic, 0.0254% exact-hash dedup; reproducible from `data/phase_e/corpus_id_dedup_exact/_manifest.json`. Q5 Mini-scale bilingual baseline: val_loss(ID) = 2.68, val_loss(EN) = 4.73, ratio = 1.77×. First bilingual ternary LM in this size class.

**Why it matters:** Indonesia-specific / multilingual ternary modeling is unaddressed in literature; standard datasets (SlimPajama, etc.) are English-dominant. Our corpus is the first reusable bilingual ID+EN corpus released for this purpose.

**Tables/figures:** Table 5 (Q5 baseline numbers), Figure 4 (corpus composition pie chart: Wikipedia + FineWeb-2 + SlimPajama + dedup losses).

### C3. In-kernel deployment via FajarOS Nova v3.9.0 IntLLM kernel-path

**Claim:** First demonstrated kernel-resident 1.58-bit LLM inference path. Medium-scale ternary checkpoint (74M params) runs entirely inside `@kernel` context (no heap, no syscalls during inference, deterministic). Binary footprint ≤16 MB; 4-invariant kernel-path regression gate (`make test-intllm-kernel-path`) green per FajarOS Nova v3.9.0 release notes.

**Why it matters:** Existing in-kernel ML solutions (eBPF AI offloading) are inference-shim only — model weights stay in userspace. We put the entire inference path inside the kernel address space. Unique to FajarQuant + FajarOS + Fajar Lang stack.

**Tables/figures:** Table 4 (FajarOS Nova kernel-path metrics: binary size, tok/s, watts per size — placeholders for real measurement at paper time), Figure 2 (kernel binary size budget breakdown).

### C4. Honest negative results on outlier-handling features at our regime

**Claim:** Two QuaRot/SmoothQuant-inspired outlier-handling features were ablated at Mini scale and BOTH FAILED:

- **balanced_calib (E2.4)**: per-channel calibrated quantization scales from running-max accumulator → `outlier_global_reduction = −82.13` (gate ≥ 0.10), 83× WORSE than upstream baseline. See `FJQ_PHASE_E_E2_BILINGUAL_CALIB_DECISION.md` v1.0.
- **Hadamard (E2.1)**: orthogonal Walsh-Hadamard rotation on `block.attn.o_proj` inputs → `val_loss(EN) = 4.852` vs Q5 baseline 4.732 (regression +0.12 nat vs gate ≥+0.05 nat improvement). See `FJQ_PHASE_E_E2_HADAMARD_DECISION.md` v1.0.

**Common attribution (3 causes per decision docs):** (1) post-hoc-quantization literature precedents (QuaRot, SpinQuant, SmoothQuant, GPTQ, AWQ) all apply rotation/calibration to PRE-TRAINED models being re-quantized; training-from-scratch with these features in place makes the model fight the modification rather than benefit from it. (2) HGRN-Bit's gated-linear-recurrence attention may not have transformer-style outlier concentration that these features assume. (3) Mini scale (22M params) is below the threshold where outlier concentration emerges per SpinQuant ablations.

**Why it matters:** Negative results published honestly prevent inflated claims (per CLAUDE §6.6 R3). The 3-cause attribution gives the next research team specific hypotheses to test rather than "it didn't work." This is the kind of empirical fence-post that small-scale ternary literature currently lacks.

**Tables/figures:** Table 5 (extends Q5 baseline with E2.4 + E2.1 ablation rows), Table 6 (3-cause attribution summary), Figure 3 (Hadamard outlier-suppression on synthetic spike — DOES work mathematically; supports F.6 future-work pitch).

### C5. Compiler-verified safety via `@kernel` / `@device` / `@noinline` annotations

**Claim:** The Fajar Lang compiler enforces kernel-context safety at compile time: `@kernel` functions cannot heap-allocate, cannot use Tensor type, cannot call `@device` functions (and conversely). Type system validates kernel-LLM call paths at compile time, eliminating a class of runtime safety failures that conventional ML deployment cannot rule out.

**Why it matters:** First language with type-system-level guarantees for kernel-LLM coexistence. Differentiator from "Python+CUDA in userspace" status quo.

**Tables/figures:** Table 7 (compile-time error examples from `@kernel` enforcement; e.g. `String::new()` → KE001, `zeros(3,4)` → KE002), Figure 5 (compilation pipeline with `@kernel`/`@device` analyzer pass).

---

## 3. Paper Section Map (10 pages MLSys + 11 pages arXiv)

| § | Title | Page budget | Backed by |
|---|---|---|---|
| 1 | Introduction | 1.0 | C1+C2+C3 motivation |
| 2 | Background | 1.0 | BitNet b1.58, MMfreeLM, kernel-context constraints |
| 3 | IntLLM architecture | 1.0 | HGRN-Bit + Mistral v3 BPE + RMSNorm + ternary activation_quant |
| 4 | Training methodology + Track B | 1.0 | Calibrated gates + interruption-safety per CLAUDE §6.11 |
| 5 | Phase D scaling-chain results (C1) | 1.5 | Tables 1+2, Figure 1 |
| 6 | Bilingual extension + Q5 baseline (C2) | 1.0 | Table 5 partial, Figure 4 |
| 7 | Honest negative results on outlier-handling (C4) | 1.5 | Tables 5+6, Figure 3, decision docs |
| 8 | In-kernel deployment (C3) | 1.0 | Table 4, Figure 2, FajarOS Nova kernel-path gate |
| 9 | Compiler-verified safety (C5) | 1.0 | Table 7, Figure 5 |
| 10 | Discussion + Future Work | 1.0 | Phase F.5/F.6/F.7/F.8/F.9 future-work pointers |

**arXiv-only extras (1 extra page):** detailed reproducibility appendix, full hyperparameter table, expanded ablation negative-result attribution.

---

## 4. Tables / Figures Inventory

### Tables

| # | Title | Status | Source artifact |
|---|---|---|---|
| 1 | Per-config Phase D gate margins | ✅ Real data | `paper/intllm/results/training_intllm-*.json` |
| 2 | Phase D wikitext + zero-shot benchmarks | ⚠️ 12/13 PASS, 1 PEND (kernel E2E tok/s) | `paper/intllm/results/bench_canonical_*.json` + `verify_intllm_tables.py` |
| 3 | fp16-vs-ternary parity gate | ❌ NOT IMPLEMENTED | Phase D §3.5 — needs `--fp16-reference` flag in train driver + `make test-intllm-fp16-parity` |
| 4 | FajarOS Nova kernel-path E2E metrics | ⚠️ Kernel-path runs but binary-size + tok/s + watts not measured at paper-time freshness | FajarOS Nova v3.9.0 release artifacts |
| 5 | Q5 baseline + 2 negative-result ablation rows | ✅ Real data | `q5_bilingual_baseline.json` + `mini_balanced_calib_quant_error.json` + `mini_hadamard.json` |
| 6 | 3-cause attribution summary | ✅ Documented | Decision docs §3 |
| 7 | Compile-time `@kernel` enforcement examples | ✅ Existing | fajar-lang test suite + `docs/ERROR_CODES.md` |

### Figures

| # | Title | Status | Source |
|---|---|---|---|
| 1 | Scaling-law curve (val_loss vs n_params) | TODO — needs matplotlib script | Real Phase D numbers exist |
| 2 | Kernel binary size budget breakdown | TODO — needs measurement + diagram | FajarOS Nova v3.9.0 build artifacts |
| 3 | Hadamard outlier-suppression on synthetic | TODO — easy plot | `intllm.quant.HadamardRotation` + outlier-suppression unit test reused |
| 4 | Bilingual corpus composition pie chart | TODO — easy plot | `_manifest.json` per corpus_id_dedup_exact source |
| 5 | Compilation pipeline with @kernel analyzer pass | TODO — diagram | fajar-lang docs/ARCHITECTURE.md |

---

## 5. Outstanding work for paper publication (effort estimate)

### MUST-HAVE before submission

| Task | Status | Effort | Owner |
|---|---|---|---|
| Phase D §4 LaTeX writeup (10-page MLSys + 11-page arXiv) | DEFERRED | ~3-5 days | Founder |
| Related-work doc `INTLLM_RELATED_WORK.md` (≥8 papers cited per Phase D §4.1) | NOT STARTED | ~2-3 days | Founder |
| Table 3 fp16-vs-ternary parity gate impl + measurement | NOT STARTED | ~2 days human + ~2h GPU | Founder |
| `bench-baselines-real-all` activation (7 repos × 3 metrics = 21 placeholder claims) | DEFERRED | ~13h GPU + 4h human | Founder |
| Figure 1 / 2 / 3 / 4 / 5 generation | NOT STARTED | ~1-2 days total | Founder |
| Citations + ORCID + Zenodo DOI wire-up | DEFERRED | ~½ day | Founder |
| `verify_paper_tables.py --strict` exit-0 check (paper-claim verification) | INFRA EXISTS | ~½ day | Founder |
| Co-author review + revision | NOT STARTED | ~2-3 days iteration | Founder + reviewers |

**Total: ~10-14 days human + ~13-15h GPU.**

### NICE-TO-HAVE (skip if time pressure)

| Task | Why nice-to-have | Effort if done |
|---|---|---|
| Phase D Stretch training (15B tokens, ≥1 extra scaling-table row) | 4th data point strengthens C1 scaling-law claim | ~17 days GPU continuous (DEFERRED in plan v1.9; recommend SKIP for Path A timing) |
| BitNet 2B4T fair comparison row in Table 2 | Apples-to-apples vs literature | Already scaffolded in `bench_baselines.py`; ~13h GPU bundle with the must-have sweep |
| Multilingual eval beyond val_loss (e.g. xnli for ID, hellaswag for EN) | Stronger C2 claim | ~1 day human + ~3h GPU |

### EXPLICITLY OUT-OF-SCOPE for Path A paper

| Item | Why excluded |
|---|---|
| E2.2 FP8 mixed-precision ablation | CUT under Path A; deferred to Phase F.7 |
| E2.3 FP16 distillation ablation | CUT under Path A; deferred to Phase F.8 |
| E2.5 Language-conditioned design ablation | CUT under Path A; deferred to Phase F.9 |
| E2.6 Combined ablation | REMOVED under Path A |
| Phase E3 bilingual scale-up (Base/Medium ternary on bilingual corpus) | Phase E3 territory; paper covers Q5 Mini baseline only |
| Tier 3 tax-vertical TaxPrime | Phase F.1-F.4; gated on this paper's acceptance |
| Phase F.5 PTQ post-training calibration | Phase F future-work; pointer only in §10 |
| Phase F.6 post-hoc QuaRot on pretrained checkpoint | Phase F future-work; pointer only in §10 |

---

## 6. Risk register for Path A execution

| Risk | Mitigation | Severity |
|---|---|---|
| LaTeX writeup takes longer than 5 days | Outline first (this doc); write section-by-section using existing decision docs as content; reuse v3.1 paper LaTeX as scaffold | MEDIUM |
| Table 4 kernel E2E metrics need fresh measurement that conflicts with FajarOS Nova v3.9.0 freeze | Measure WITHOUT modifying FajarOS code; ship measurement script alongside paper artifact | LOW |
| Reviewer pushback on "0/2 ablations" being light evidence | Frame as "honest negative results with 3-cause attribution" rather than "we tried and failed"; cite §6.6 R3 / §6.9 R3 as methodology | MEDIUM |
| Phase D §3.5 fp16-vs-ternary gate slips (Table 3 incomplete) | Mark as "future work" if not done; paper still has Tables 1+2+4+5+6+7 = enough for 10 pages | LOW |
| Stretch training pressure to include 4th scaling-law point | Explicit decision in §10 future work; cite the 17-day GPU budget rationale | LOW |
| BitNet 2B4T baseline row in Table 2 missing | Footnote: "BitNet 2B4T comparison deferred — different scale + post-hoc protocol; see Phase F" | MEDIUM |

---

## 7. Calendar (Path A target: paper submission within 4 weeks)

Week 1 (now → 2026-05-04):
- Section §3 architecture writeup (1 day)
- Section §4 methodology writeup (1 day)
- Related-work doc `INTLLM_RELATED_WORK.md` (2-3 days)
- bench-baselines-real-all real sweep launch + verify_intllm_tables activation (1 day human + 13h GPU)

Week 2 (2026-05-04 → 2026-05-11):
- Section §5 results writeup (Tables 1+2 + Figure 1) (1.5 days)
- Section §6 bilingual extension writeup (Table 5 partial + Figure 4) (1 day)
- Section §7 negative results writeup (Tables 5 full + 6 + Figure 3) (1.5 days)
- Section §1 intro draft (½ day)

Week 3 (2026-05-11 → 2026-05-18):
- Section §8 in-kernel deployment writeup (Table 4 + Figure 2) (1 day; needs fresh measurement)
- Section §9 compiler safety writeup (Table 7 + Figure 5) (1 day)
- Section §10 discussion + future work (1 day)
- Section §2 background writeup (½ day)
- Table 3 fp16-parity gate impl + measurement (2 days human + 2h GPU)

Week 4 (2026-05-18 → 2026-05-25):
- Full draft revision pass (1-2 days)
- `verify_paper_tables.py --strict` check + fix discrepancies (½ day)
- ORCID + Zenodo DOI + final submission package (1 day)
- BUFFER (1-2 days for unexpected slip)

**Submission target: 2026-05-25** (~4 weeks from Path A decision).

MLSys 2027 submission cycle: typical fall deadline (October-ish); arXiv preprint can land any time. Path A is paced for arXiv first, MLSys submission whenever the next cycle opens. SOSP / OSDI / EuroSys are also viable venues given the kernel-deployment angle.

---

## 8. Sign-off

| Role | Name | Sign-off |
|---|---|---|
| Author | Muhamad Fajar Putranto (PrimeCore.id) | 2026-04-27 |
| Engineering | Claude Opus 4.7 (1M context) | 2026-04-27 |
| Reviewers | (paper team) | TBD pre-submission |

---

## 9. References

- Predecessors:
  - `FJQ_PHASE_D_PRODUCTION_PLAN.md` v1.2 (Phase D plan; §2.3 Stretch + §3.4 bench + §4 paper LaTeX all DEFERRED to V32 = this paper effort)
  - `FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.9 (Phase E plan; §3 PHASE E2 closure; v1.10 will add Path A scope decision)
  - `FJQ_PHASE_E_E2_FINDINGS.md` v1.1 (E2.0 closure)
  - `FJQ_PHASE_E_E2_4_FINDINGS.md` v1.4 (E2.4 closure)
  - `FJQ_PHASE_E_E2_BILINGUAL_CALIB_DECISION.md` v1.0 (E2.4 demote decision)
  - `FJQ_PHASE_E_E2_HADAMARD_DECISION.md` v1.0 (E2.1 demote decision)
- Companions:
  - `FJQ_PHASE_F_TAX_VERTICAL_ROADMAP.md` v1.2 → v1.3 (will absorb F.7/F.8/F.9 post-Path-A)
- Tooling:
  - `scripts/verify_intllm_tables.py` (verify_paper_tables.py equivalent for Phase D)
  - `paper/intllm/` directory (target for paper LaTeX + tables)
- Literature anchors:
  - BitNet b1.58 2B4T (Microsoft, 2024)
  - MatMul-Free LLM (Zhu+ 2024)
  - QuaRot (arxiv 2404.00456) — cited as future-work source for Phase F.6
  - SmoothQuant (arxiv 2211.10438) — cited as future-work source for Phase F.5
  - Calibrating Beyond English (arxiv 2601.18306) — cited for C2 motivation
  - SmolLM2, Pythia, TeLLMe, TerEffic, SLM-Bench, seL4 — Phase D §4.1 related-work list

*Document version: 1.0 (Path A scope decision committed). Author: Claude Opus 4.7 + Fajar (PrimeCore.id).*
*Phase E v1.9 → v1.10 transition; paper-track pivot.*
