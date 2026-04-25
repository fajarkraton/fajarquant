# TaxPrime Dataset Specification — Phase E Tier 3 Vertical Wedge

> **Spec version:** 1.0 (2026-04-25)
> **Audience:** TaxPrime knowledge team + domain experts (Sr. tax consultants)
> **Authoring lead:** Fajar (TaxPrime founder + FajarQuant author)
> **Implementation lead:** TBD per TaxPrime team
> **Plan reference:** `FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.1 §3 E4.1 + §11
> **Repo location for final dataset:** `~/Documents/fajarquant/data/tax_id_corpus_v1/` (DVC- or HF-Datasets-pinned)

---

## 0. Why this spec exists

Phase E Tier 3 (vertical wedge) hinges on a high-quality, privacy-clean, audit-trail-proof Indonesian tax/legal dataset built BY TaxPrime team, not synthesized via foreign models.

Two distinct artifacts must be produced from the same TaxPrime knowledge base:

1. **Training corpus** (E4.1 — for vertical fine-tune) — coverage-oriented, 5K–50K examples
2. **Evaluation set** (§11 — for `pass@1` measurement) — quality-oriented, 500 gold-labeled prompts

These two have **different methodologies, different acceptance criteria, different cost structures.** Conflating them produces either over-engineered training data or under-validated evaluation.

This spec is the contract between TaxPrime knowledge team (data producers) and FajarQuant Phase E plan (data consumer). Once the dataset meets §13 acceptance criteria, it unblocks E4.1 SFT training and §11 eval gating.

---

## 1. Two datasets, two methodologies (the most important distinction)

| Attribute | Training corpus (E4.1) | Evaluation set (§11) |
|---|---|---|
| **Volume target** | 5,000–50,000 examples | exactly 500 examples |
| **Annotator count per example** | 1 (single-annotator OK) | 2 (double-rater MANDATORY) |
| **IRR requirement** | none | Cohen's κ ≥ 0.7 |
| **Quality target** | ≥80% factually correct (5% sample) | 100% (after IRR + adjudication) |
| **Conflict resolution** | informal — discard ambiguous | formal 3rd-rater adjudication, logged |
| **Difficulty mix** | natural distribution | enforced 30% easy / 50% medium / 20% hard |
| **PII sanitization** | mandatory (§6) | mandatory (§6) |
| **Bias audit** | optional | required (light external auditor OK) |
| **Annotator seniority** | mid+ tax consultant | senior tax consultant (3+ yrs) |
| **Cost (rough estimate)** | 50–80 person-hours | 30–40 person-hours |
| **Reusability** | training only — never used for eval | eval only — never used for training |
| **Acceptance gate** | §13.1 | §13.2 |

**Hard rule:** No example appears in both training and eval sets. Cross-contamination invalidates Phase E paper claims (§6.9 R6 violation).

---

## 2. JSONL schema (canonical format)

Each line in `train.jsonl` / `val.jsonl` / `eval.jsonl` is one JSON object:

```json
{
  "id": "tax-2026-001234",
  "prompt": "Bagaimana cara menghitung PPh Pasal 21 untuk karyawan tetap dengan tantiem?",
  "response": "PPh Pasal 21 untuk karyawan tetap dihitung dengan...",
  "citations": [
    {"type": "UU", "ref": "UU 7/2021 (HPP)", "section": "Pasal 17"},
    {"type": "PMK", "ref": "PMK-168/PMK.03/2023", "section": "Pasal 5"},
    {"type": "PER-DJP", "ref": "PER-16/PJ/2016", "section": "Pasal 14"}
  ],
  "category": "PPh-21",
  "subcategory": "karyawan-tetap-dengan-bonus",
  "source": {
    "type": "internal-memo-sanitized",
    "memo_id": "[INT-XXXX]",
    "client_industry_redacted": true
  },
  "language": "id",
  "difficulty": 2,
  "tax_period_relevance": "2024-2026",
  "pii_status": "clean",
  "annotator_id": "anno-A12",
  "annotated_at": "2026-05-01T10:00:00+07:00",
  "reviewed_by": "senior-B7",
  "reviewed_at": "2026-05-03T15:00:00+07:00",
  "split": "train"
}
```

**Eval-set-only additional fields:**
```json
{
  "raters": ["anno-A12", "anno-B7"],
  "rater_agreement": "match",
  "cohen_kappa_global": 0.74,
  "adjudicator_id": null,
  "bias_audit_flag": "none"
}
```

### Field reference

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | string | yes | Format: `tax-YYYY-NNNNNN` (year-incremented) |
| `prompt` | string | yes | The question / instruction (max 4096 tokens) |
| `response` | string | yes | The gold answer (max 8192 tokens) |
| `citations` | array | yes (≥1 for any factual claim) | List of regulatory references; auto-validated by `scripts/check_citations.py` |
| `category` | string | yes | One of taxonomy in §4 |
| `subcategory` | string | optional | Free-form sub-classification |
| `source` | object | yes | Provenance per §5 |
| `language` | string | yes | `id` / `en` / `mixed` |
| `difficulty` | int | yes | 1–5 per §9 rubric |
| `tax_period_relevance` | string | optional | Year range when this answer is valid (regulations change!) |
| `pii_status` | string | yes | `clean` / `sanitized` / `blocked` |
| `annotator_id` | string | yes | Pseudonymous; map to real identity in TaxPrime internal log |
| `annotated_at` | ISO 8601 | yes | Timezone +07:00 (WIB) |
| `reviewed_by` | string | training: optional / eval: required | |
| `reviewed_at` | ISO 8601 | training: optional / eval: required | |
| `split` | string | yes | `train` / `val` / `eval` |

---

## 3. Topic taxonomy (12 main + sub-categories)

### Primary categories

| Code | Name (Indonesian) | English | Coverage target % |
|---|---|---|---|
| `PPh-Badan` | Pajak Penghasilan Badan | Corporate income tax | 15% |
| `PPh-OP` | PPh Orang Pribadi | Personal income tax | 10% |
| `PPh-21` | PPh Pasal 21 | Withholding tax — employee compensation | 10% |
| `PPh-22-23-26` | PPh Pasal 22/23/26 | Withholding tax — vendors / dividends / cross-border | 10% |
| `PPh-4(2)` | PPh Pasal 4 ayat 2 | Final income tax (deposits, property, construction) | 5% |
| `PPN-PPnBM` | PPN dan PPnBM | VAT and luxury tax | 12% |
| `PBB-BPHTB` | PBB dan BPHTB | Land/building tax + transfer | 5% |
| `Bea-Cukai` | Bea Cukai | Customs and excise | 5% |
| `Transfer-Pricing` | Transfer pricing (TP) | TP (incl. BEPS, country-by-country) | 8% |
| `Pengadilan-Pajak` | Putusan Pengadilan Pajak | Tax court cases | 8% |
| `Pajak-Daerah` | Pajak Daerah dan Retribusi | Regional taxes | 3% |
| `Pajak-Internasional` | Tax treaty + transparansi | International tax | 4% |
| `Pajak-Digital` | Pajak digital + e-commerce | Digital tax + e-commerce | 3% |
| `Procedure-KUP` | KUP, PPSP, sengketa | Tax procedure, dispute resolution | 2% |

Total: 100%. Annotators should track per-batch coverage and flag gaps.

### Industry overlay (orthogonal)

For categories above, also tag relevant industry where applicable:
- `industry-banking`, `industry-oilgas`, `industry-mining`, `industry-realestate`, `industry-manufaktur`, `industry-konsumer`, `industry-jasa`, `industry-startup`, `industry-bumn`, `industry-umkm`, `industry-tidakspesifik`

---

## 4. Source provenance taxonomy

Every example must declare source via `source.type`:

| `source.type` | Description | License notes | Volume target % |
|---|---|---|---|
| `regulation-public` | UU, PMK, PER-DJP, SE — extracted directly | public domain | 25% |
| `court-public` | Pengadilan Pajak, MA decisions | public | 10% |
| `taxprime-memo-sanitized` | TaxPrime internal advisories, sanitized of client identifiers | internal — proprietary, dataset-only license | 35% |
| `taxprime-clientqa-sanitized` | Real client questions, sanitized | internal — proprietary | 20% |
| `treaty-public` | Indonesia tax treaties, OECD model docs | public | 5% |
| `external-publication` | Indonesian tax journal articles, IFA proceedings (with attribution) | per-source — check before include | 5% |

**Critical rule for `taxprime-*` sources:** must pass `scripts/check_pii_redaction.py` strict mode AND TaxPrime senior partner sign-off per batch (50 examples). No example ships without this.

---

## 5. PII sanitization rules (NON-NEGOTIABLE)

### 5.1 Mandatory redactions

| PII type | Replace with | Examples |
|---|---|---|
| NPWP (15-digit tax ID) | `[NPWP-XX]` | `01.234.567.8-901.000` → `[NPWP-XX]` |
| Personal name | `[CLIENT]`, `[KONSULTAN]`, `[PETUGAS]` | "Bapak Hartono Wijaya" → `[CLIENT]` |
| Company name (private) | `[PERUSAHAAN-X]` | "PT Maju Sejahtera Abadi" → `[PERUSAHAAN-A]` |
| Address (specific) | `[ALAMAT]` | "Jl. Sudirman No. 123, RT 04 RW 02" → `[ALAMAT]` |
| Specific Rp amount | `[Rp-XX]` (keep magnitude) | "Rp 1.234.567.890" → "ratusan juta rupiah" or `[Rp-XX]` |
| Letter/case number | `[NOMOR-XX]` | "S-1234/PJ.01/2025" → `[NOMOR-XX]` |
| Specific date | year/month OK; specific date redact | "15 Maret 2025" → "Maret 2025" |
| Email | `[EMAIL]` | `name@taxprime.net` → `[EMAIL]` |
| Phone | `[PHONE]` | `+62-21-1234567` → `[PHONE]` |
| Bank account | `[ACCOUNT]` | `1234567890` → `[ACCOUNT]` |

### 5.2 Keep-as-is (NOT redacted)

- Public official names (Menteri Keuangan, Dirjen Pajak) when in regulatory context
- Public company names (Tbk listed) when in regulatory or court-case context
- Regulation numbers (UU, PMK, PER-DJP — these ARE the citations)
- Year ranges, tax periods
- Generic industry references

### 5.3 Verification gates

- **Automated:** `scripts/check_pii_redaction.py` runs regex + named-entity-recognition + LLM-as-judge over each example. ZERO tolerance for residual NPWP / phone / email patterns.
- **Manual:** TaxPrime senior partner reviews every 50 examples in batch sample. Sign-off recorded in `data/tax_id_corpus_v1/AUDIT_LOG.md`.
- **External (optional):** independent reviewer (e.g., privacy lawyer) spot-checks 100 random examples pre-finalization.

---

## 6. Difficulty rubric (1–5)

For consistent labeling — annotators should reference these examples:

| Level | Label | Description | Example prompt |
|---|---|---|---|
| 1 | `lookup` | Single-rule direct lookup | *"Berapa tarif PPh badan tahun 2026?"* |
| 2 | `synthesis` | 2–3 rules synthesis, single domain | *"Bagaimana hitung PPh 21 karyawan dengan tantiem dan tunjangan kehadiran?"* |
| 3 | `edge-case` | Edge case or gray-area within single domain | *"Apakah biaya entertainment client masuk biaya yang dapat dikurangkan?"* |
| 4 | `court-derivation` | Apply court precedent + analogize | *"Berdasar Putusan PP No. XXX/2024, bagaimana sengketa transfer pricing intercompany services 2025?"* |
| 5 | `multi-jurisdiction` | Tax treaty + domestic + cross-border | *"PT-A Indonesia royalti ke parent BV Belanda; bagaimana withholding tax + treaty + BEPS Pillar 2 Indonesia?"* |

**Eval set distribution (enforced):** 30% level 1–2 (easy), 50% level 3 (medium), 20% level 4–5 (hard).

**Training corpus distribution:** natural — whatever TaxPrime knowledge base produces, but track and flag if extreme imbalance (>70% any single level).

---

## 7. Bilingual handling

Plan §1 item 2 specifies bilingual coverage. Within tax dataset:

| Mix | % target | Examples |
|---|---|---|
| ID-only prompt + ID-only response | 60% | Most domestic-tax questions |
| ID prompt + response with EN regulatory citations | 20% | Common practice — quote OECD, BEPS, treaty articles in English |
| Mixed ID + EN prompt (e.g., professional shorthand) | 10% | "Bagaimana withholding tax atas service fees ke parent company?" |
| EN-only | 10% | Tax treaty interpretation, IFRS-tax bridges, foreign-investor inquiries |

---

## 8. Quality control

### 8.1 Training corpus QC

- **5% random re-review** by senior partner per batch. Spot-check for: factual correctness, citation accuracy, PII residual, format compliance.
- **Automated checks:** schema validation (`jsonschema`), citation format regex, language detection accuracy.
- **Annotator calibration:** before each annotator's first batch, they label 20 calibration examples vs gold standard. Pass threshold: 16/20 correct. Fail → mentor pairing for next batch.
- **Reject criteria:** any example failing schema, citation format, PII redaction, OR with >2 calibration errors → reject + retrain annotator.

### 8.2 Eval set QC (stricter)

- **100% double-rated.** Every example MUST have 2 independent rater labels.
- **Cohen's κ ≥ 0.7 threshold per batch of 50.** If a batch falls below, full re-annotate with refined guideline.
- **Conflict resolution:** any disagreement → 3rd rater (senior partner) adjudicates. Adjudication recorded with reasoning.
- **Bias audit:** category coverage, industry coverage, regional coverage. No category >20% or <2%. No single industry >25%.
- **External audit (light):** ~50 random eval examples reviewed by external Indonesian tax academic (1 day engagement, ~1500 USD typical).

---

## 9. Privacy / compliance checklist

Before any commit to `data/tax_id_corpus_v*/`:

- [ ] NDA review per source category (TaxPrime in-house counsel)
- [ ] Data residency confirmed: all data stays in Indonesia per UU PDP (Personal Data Protection Law 27/2022)
- [ ] Retention policy: dataset version-tagged, NEVER deleted but versioned forward
- [ ] Access control: TaxPrime team only during creation; final committed dataset → read-only after sign-off
- [ ] Audit log: every who-touched-what entry in `AUDIT_LOG.md`
- [ ] External privacy counsel review final pre-commit
- [ ] PII automated check: `scripts/check_pii_redaction.py --strict` exit 0
- [ ] Sample 100-example manual privacy review by senior partner
- [ ] Public-release decision: by default, dataset is **NOT public**. Only summary statistics + sample examples (≤20) appear in paper. Full dataset retained internally + deposited with neutral 3rd party (e.g., academic repository under embargo) for paper-reproducibility verification only.

---

## 10. Deliverable format & versioning

### 10.1 Folder structure

```
data/tax_id_corpus_v1/
├── README.md                 # version description, stats, contributors
├── LICENSE.md                # internal-use license; restrictions on redistribution
├── AUDIT_LOG.md              # who-touched-what
├── BIAS_AUDIT.md             # category/industry/regional coverage report
├── CHANGELOG.md              # v1.0 → v1.1 changes
├── metadata.json             # token counts, example counts, schema version, pin SHA
├── train.jsonl               # training corpus
├── val.jsonl                 # validation set (held-out from train, ~5%)
├── eval.jsonl                # 500-prompt gold eval set
├── eval_irr_log.json         # per-example IRR data (raters, agreement)
├── annotators_pseudonyms.csv # encrypted; anno-A12 → real identity
└── checksums.sha256          # checksums for all files above
```

### 10.2 Versioning

- Semver: `v1.0`, `v1.1`, `v2.0`
- Bump `v1.x → v1.y` for additions / refinements (backward-compatible)
- Bump `v1.x → v2.0` for schema breaks or major topic-taxonomy changes
- Each version is **immutable** post-commit (additions become new version)
- Phase E paper claims pin to a specific version (e.g., `v1.0` for paper Table 5)

### 10.3 Storage strategy

- **Public-safe artifacts (paper sample, schema):** committed to git directly under `docs/`
- **Full dataset (proprietary):** DVC-tracked OR HF Datasets private OR S3+manifest. Decided in Phase E plan E0.0.8.
- **NEVER commit full dataset to public git.** Use DVC pointer + S3 backend with TaxPrime-controlled access.

---

## 11. Acceptance criteria (gate for Phase E E4.1 + §11)

### 11.1 Training corpus acceptance

- [ ] Schema validation passes for 100% of `train.jsonl` examples
- [ ] PII automated check `--strict` exit 0 across all examples
- [ ] Total examples: ≥ 5,000
- [ ] Total token count: ≥ 5M (estimated via Mistral-v3 32k tokenizer)
- [ ] Coverage: every primary category (§3) has ≥ 50 examples; max single category ≤ 25%
- [ ] Calibration: 5% random sample re-reviewed; ≥ 80% pass on factual correctness
- [ ] Annotator calibration: every annotator passed 16/20 calibration before first batch
- [ ] Privacy: senior partner sign-off in `AUDIT_LOG.md`
- [ ] Source mix: matches §4 target percentages within ±5pp

### 11.2 Evaluation set acceptance (stricter)

- [ ] Total examples: exactly 500 ±5
- [ ] Schema validation passes for 100%
- [ ] PII automated check `--strict` exit 0
- [ ] Double-rated: 100% of examples have 2 rater labels
- [ ] Cohen's κ overall ≥ 0.7
- [ ] Per-category κ ≥ 0.6 (some flexibility)
- [ ] All conflicts adjudicated; adjudication log committed
- [ ] Difficulty distribution: 30% L1-2, 50% L3, 20% L4-5 (within ±3pp)
- [ ] Bilingual mix matches §7 (within ±5pp)
- [ ] Bias audit: `BIAS_AUDIT.md` committed; no category >20% or <2%
- [ ] External light audit: 50 random examples reviewed; <5 issues flagged
- [ ] Cross-contamination: 0 examples appear in both `train.jsonl` and `eval.jsonl` (auto-checked via id + content hash)

---

## 12. Estimated effort

### 12.1 Person-hours

| Task | Hours | Owner |
|---|---|---|
| Spec walkthrough + annotator training | 8h | TaxPrime methodology lead + 2-3 annotators |
| Calibration pass (each annotator) | 4h × N annotators | annotators |
| Source curation (training corpus) | 30–50h | TaxPrime knowledge team |
| Sanitization (training corpus) | 20–30h | sanitization specialist |
| Annotation (training, ~10K examples) | 50–80h | annotators (mid+ tax consultants) |
| Annotation (eval, 500 examples × 2 raters) | 30–40h | senior tax consultants |
| IRR computation + conflict adjudication | 10h | methodology lead + senior partner |
| QC + privacy review | 20–30h | senior partner + in-house counsel |
| Bias audit + external review | 10h | methodology lead + external auditor |
| Final commit + version-tag + handoff | 5h | technical lead |
| **Total** | **~190–270h** | TaxPrime team |

### 12.2 Calendar time

- Realistic part-time pace: 6–10 weeks
- Crunch full-time (3–4 dedicated people): 3–4 weeks
- **Phase E plan §4 budgeted E0.0.2 + E1.3 + E4.0 combined ≈ 4 weeks for TaxPrime data path.** Fits the 6–10 week realistic envelope if started early in Phase E0.

### 12.3 Cost estimates (rough, per-Indonesian-market rates)

- Senior tax consultant time: ~Rp 500K–1M / hour (consulting rate); but for internal effort, opportunity cost ~Rp 200K–400K / hour
- External privacy counsel: ~Rp 5–15M one-shot review
- External light auditor (academic): ~Rp 15–25M one-shot
- **Total external cost: ~Rp 20–40M (~ USD 1300–2600)**
- **Internal opportunity cost: ~190h × Rp 300K/h = ~Rp 57M (~USD 3700)**

---

## 13. Handoff to FajarQuant Phase E

Once §11.1 and §11.2 acceptance criteria met:

1. TaxPrime team commits dataset to S3 / DVC / HF Datasets (private) at version `v1.0`
2. `metadata.json` checksum recorded in `docs/FJQ_PHASE_E_E1_FINDINGS.md`
3. Phase E plan §3 E1.3 marked closed
4. Phase E E4.0 (vertical FT pre-flight) starts, loads dataset, runs sanity training pass
5. Phase E E4.1 (full SFT) and §11 (eval) proceed

**Future re-versioning:** if domain experts identify systematic gaps post-paper, version bump to v1.1 / v2.0 with changelog. Paper-reproducibility version remains pinned to v1.0 forever.

---

## 14. FAQ for TaxPrime annotators

**Q: Should I include very recent regulations (2026 PMK)?**
A: Yes, but tag `tax_period_relevance: "2026-..."`. Recent regulations = high signal but high churn — flag if you're uncertain about settled interpretation.

**Q: Some client memos cover multiple categories. How to label?**
A: Pick the PRIMARY category (most-of-content). Use `subcategory` for secondary. If genuinely 50/50, split into 2 examples.

**Q: Court cases — should I include the full ruling text?**
A: No — too long. Extract: facts (sanitized), legal question, court reasoning, holding, applicability. Aim for 500–1500 tokens response.

**Q: Bilingual examples — when does ID-with-EN-citations apply?**
A: When the answer cites OECD docs, treaty articles in English, or IFRS-tax provisions. Common in TP, international, and IFRS-bridge questions.

**Q: I'm unsure about an answer's correctness. What to do?**
A: Mark `pii_status: "blocked"` and don't include. Better to skip than include uncertain. Flag for senior review.

**Q: Can I use ChatGPT / Claude to draft answers and then edit?**
A: **NO.** Per Phase E synthetic-data policy + Anthropic ToS concerns, do NOT use third-party LLM tools to draft training data. Use only your own expertise + reference materials. (Exception: Claude / GPT may be used as DICTIONARY / TRANSLATION reference, not as drafter.)

**Q: How do I handle outdated regulations (UU 36/2008 superseded by UU 7/2021 HPP)?**
A: Include both old and new where relevant. Use `tax_period_relevance` to mark validity windows. The model needs to know how rules evolved (court cases referencing old rules + new rules).

---

*Spec version: 1.0 (2026-04-25). Author: Claude Opus 4.7 + Fajar (TaxPrime / PrimeCore.id).*
*Dataset deliverable: `data/tax_id_corpus_v1/` to be produced by TaxPrime knowledge team.*
*Phase E gate dependency: E1.3 + E4.1 + §11. Dataset acceptance unblocks vertical FT training and pass@1 evaluation.*
*This spec is the contract between TaxPrime data producers and FajarQuant Phase E plan. Modifications require version bump (v1.x → v1.y or v2.0) and re-acceptance per §11.*
