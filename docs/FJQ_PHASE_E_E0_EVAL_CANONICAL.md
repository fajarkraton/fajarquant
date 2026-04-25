# Phase E E0.0.5 — Canonical Evaluation Suite Selection

> **Status:** PASS — canonical eval set chosen + lm-eval SHA pinned.
> **Decision date:** 2026-04-25
> **Plan reference:** `FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.2 §1 items 1+2 + §3 E0.0.5
> **Companion files:** `eval/HARNESS_SHA` (pinned commit `27988a2...`)
> **Empirical basis:** `paper/intllm/results/phase_e/e0_id_corpora_inventory.json` (E0.0.1 audit) + lm-eval v0.4.11 task registry probe (94 strict-ID tasks confirmed)

---

## 1. lm-eval-harness pinning (full 40-char SHA)

```
repo: EleutherAI/lm-evaluation-harness
ref:  27988a293647d5853e48edea291640b4af54740c
tag:  v0.4.11
date: 2026-02-13T20:21:47Z
```

**Install command** (paper §4.1 reproducibility footnote):
```bash
pip install "lm-evaluation-harness @ git+https://github.com/EleutherAI/lm-evaluation-harness.git@27988a293647d5853e48edea291640b4af54740c"
```

**Drift detection (CI):** `scripts/check_harness_pin.py` reads `eval/HARNESS_SHA`, computes SHA of installed `lm_eval` package via `pip show + git ls-remote`, fails if mismatch. Wired into `.github/workflows/phase_e_eval.yml` (planned).

**Rationale (CLAUDE §6.9 R1):** lm-eval-harness has known 0.3→0.4 ARC-E normalization drift (see `FJQ_PHASE_D_P2_1_GATE.md` for evidence on `ridger/MMfreeLM-370M`). Pinning eliminates this class of silent benchmark disagreement between Phase E IntLLM numbers and replicated baselines.

---

## 2. Canonical Indonesian evaluation set (4 core + 33 extended)

### 2.1 Core Indonesian set (4 tasks — REQUIRED for every Phase E checkpoint)

| Task | Description | Few-shot | Expected runtime per Medium ckpt |
|---|---|---|---|
| `arc_id` | Indonesian translation of AI2 ARC (challenge + easy) science questions | 0-shot | ~5 min |
| `belebele_ind_Latn` | Belebele reading comprehension Indonesian (Latin script) — multiple choice | 0-shot | ~3 min |
| `global_mmlu_id` | Indonesian Global MMLU (curated translation) — 6 subject groups | 5-shot | ~15 min |
| `m_mmlu_id` | Multilingual MMLU Indonesian (Okapi) — broad knowledge | 5-shot | ~12 min |

**Aggregate runtime:** ~35 min per checkpoint per ID core eval.

### 2.2 Extended Indonesian set (33 subtasks — paper Table 5 detail)

#### MMMLU Indonesian (Okapi-style multilingual MMLU breakdown)

```
mmmlu_id_id_abstract_algebra
mmmlu_id_id_anatomy
mmmlu_id_id_astronomy
mmmlu_id_id_business_ethics
mmmlu_id_id_clinical_knowledge
mmmlu_id_id_college_biology
mmmlu_id_id_college_chemistry
mmmlu_id_id_college_computer_science
mmmlu_id_id_college_mathematics
mmmlu_id_id_college_medicine
... (full 57-subject MMLU translation; total 33+ available in lm-eval v0.4.11)
```

**Reference:** the parent `mmmlu_id_id` aggregator runs all subjects.

#### INCLUDE Indonesian (cultural / professional knowledge benchmark)

```
include_base_44_indonesian                       (aggregator)
include_base_44_indonesian_applied_science
include_base_44_indonesian_arts_humanities
include_base_44_indonesian_professional_certification
include_base_44_indonesian_social_science
include_base_44_indonesian_stem
```

Plus `_few_shot_en_*` variants (English-context) and `_few_shot_og_*` variants (original-language) — useful for measuring cross-lingual transfer.

#### Global MMLU Indonesian breakdown

```
global_mmlu_id_business
global_mmlu_id_humanities
global_mmlu_id_medical
global_mmlu_id_other
global_mmlu_id_social_sciences
global_mmlu_id_stem
```

### 2.3 Tax-vertical Indonesian (Phase E E4.1 + §11)

NOT in lm-eval registry — built by TaxPrime team per `FJQ_PHASE_E_TAXPRIME_DATASET_SPEC.md` v1.0. 500-prompt eval set, double-rater + Cohen κ ≥ 0.7 + bias audit. Runs via custom `make eval-tax-vertical` (see plan §3 E4.1.2).

---

## 3. Canonical English evaluation set (10 tasks — apples-to-apples with Phase D baselines)

Selected to match Phase D §3.4 baseline sweep exactly, ensuring direct comparability with BitNet 2B4T, MMfreeLM-370M (and pending 1.3B/2.7B), SmolLM2, Pythia.

| Task | Description | Few-shot |
|---|---|---|
| `wikitext` | Word + byte perplexity on Wikitext-103 | 0-shot |
| `lambada_openai` | Last-token prediction (long context) | 0-shot |
| `hellaswag` | Commonsense sentence completion | 0-shot |
| `piqa` | Physical commonsense | 0-shot |
| `winogrande` | Coreference resolution | 0-shot |
| `arc_easy` | Grade-school science (easy) | 0-shot |
| `arc_challenge` | Grade-school science (challenge) | 0-shot |
| `openbookqa` | Open-book elementary science | 0-shot |
| `mmlu` | Massive multitask language understanding | 5-shot |
| `triviaqa` | Reading comprehension trivia | 5-shot |
| `boolq` | Yes/no questions | 0-shot |

**11 tasks total** matching Phase D `BASELINE_REGISTRY` num_fewshot_map. Aggregate runtime ~30-60 min per Medium checkpoint per EN core eval.

---

## 4. Bilingual coherence test set (Phase E specific, NEW v1.2)

To validate §1 item 3 bilingual coherence gate (ID-PPL and EN-PPL within 1.3× of each other):

| Task | Setup |
|---|---|
| Bilingual perplexity test | Sample 100 ID + 100 EN articles from held-out Wikipedia subset, compute PPL on both, compare ratio |
| Cross-lingual translation test | flores200_indo_eng + flores200_eng_indo from lm-eval (if registered); else custom |

Result reported in `paper/intllm/results/phase_e/bilingual_coherence_<scale>.json` per E3 gate.

---

## 5. Per-paper-table mapping

| Paper Table | Eval set | Used by |
|---|---|---|
| Table 1 — Indonesian core | §2.1 (4 core tasks) | Phase D Mini/Base/Medium + Phase E bilingual scales + baselines |
| Table 2 — English canonical | §3 (11 tasks) | apples-to-apples baselines (BitNet/MMfreeLM/SmolLM2/Pythia) — partial sweep already done |
| Table 3 — Indonesian extended | §2.2 (33 subtasks) | Phase E Medium + Stretch only (cost reasons) |
| Table 4 — kernel-context serving (NEW v1.1) | §13 specs (tok/s, TTFT, RSS, power) | Phase E E5.1.6-9 |
| Table 5 — Bilingual coherence (NEW v1.2) | §4 above | Phase E E3 every scale |
| Table 6 — Tax vertical (NEW v1.1) | TaxPrime 500-prompt set per §11 | Phase E E4 only |
| Table 7 — Ablation (NEW v1.1) | §2.1 + §3 mix | Phase E E2 mini-scale ablations |

---

## 6. Aggregate eval runtime budget

Per Medium-scale checkpoint full evaluation (all tables):

| Component | Runtime estimate |
|---|---|
| ID core (§2.1) | ~35 min |
| ID extended (§2.2) | ~120 min |
| EN canonical (§3) | ~60 min |
| Bilingual coherence (§4) | ~10 min |
| Tax vertical (§5 Table 6) | ~30 min |
| **Total per Medium ckpt** | **~4–5 hours GPU** |

Phase E plan §4 budgets ~10–20h GPU for E3.5 canonical benchmarks at Medium + Stretch. Consistent with this runtime.

For Mini and Base scales: subset to §2.1 + §3 + §4 only (skip §2.2 extended + §5 tax) → ~1.5h per checkpoint.

---

## 7. Reproducibility commitment

Every paper claim derived from these eval sets traces to:
1. lm-eval-harness commit `27988a2...` (this doc + `eval/HARNESS_SHA`)
2. Task names (verbatim above)
3. Few-shot configurations (verbatim above)
4. Phase E IntLLM checkpoints (committed to S3/HF Datasets per E0.0.8)
5. Per-checkpoint result JSON (`paper/intllm/results/phase_e/<scale>_<table>.json`)

`scripts/verify_phase_e_tables.py --strict` (E5.3 deliverable) cross-references all paper tables against these JSONs, fails CI if drift.

---

## 8. Decision: PASS

All §1 item 1–2 + §3 E0.0.5 plan requirements met:

- [x] lm-eval v0.4.11 SHA pinned (full 40-char commit `27988a293647d5853e48edea291640b4af54740c`)
- [x] 4 core ID tasks selected
- [x] 33+ extended ID subtasks selected
- [x] 11 canonical EN tasks selected (apples-to-apples with Phase D baselines)
- [x] Bilingual coherence test set spec'd
- [x] Per-paper-table mapping documented
- [x] Aggregate runtime budget estimated
- [x] Reproducibility commitment recorded

Phase E plan §3 E0.0.5 gate: **PASS** (mechanical decision file landed per §6.8 R6).

---

*Decision committed 2026-04-25. Author: Claude Opus 4.7 + Fajar.*
*Companion: `eval/HARNESS_SHA` (full SHA pinned).*
*Reproducibility: every Phase E paper claim must trace to the lm-eval ref above + task names + checkpoint hash.*
