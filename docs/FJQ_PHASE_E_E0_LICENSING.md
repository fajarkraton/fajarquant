# Phase E E0.0.7 — Per-Source Data Licensing Review

> **Status:** PASS — paper-publication path universally clean; commercial deployment requires source-mix caveats.
> **Decision date:** 2026-04-25
> **Plan reference:** `FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.3 §3 E0.0.7
> **Companion docs:** `FJQ_PHASE_E_E0_SYNTHETIC_LEGAL.md` (E0.0.10 — generator licensing); `FJQ_PHASE_E_E0_DATA_VERSIONING.md` (E0.0.8 — storage)

---

## 1. Per-source license matrix (verified 2026-04-25)

### Indonesian corpus sources (E1.1)

| Source | License | Paper publication | Commercial deployment | Phase E status |
|---|---|---|---|---|
| **CC-100 Indonesian** | "no IP claims on prep"; underlying Common Crawl under CC BY 4.0 with web-content copyright caveat | ✅ fair use (Bartz precedent) | ⚠️ gray — Common Crawl fair-use defense, contested for commercial. Many production LLMs (BLOOM, mGPT) ship despite | conditional include |
| **OSCAR Indonesian v23.01** | **CC0-1.0** (full public domain) | ✅ unrestricted | ✅ unrestricted | ✅ include |
| **Wikipedia Indonesian** | **CC-BY-SA 4.0** | ✅ fair use + attribution | ⚠️ share-alike clause contested for LLM training; Wikimedia Foundation accepts ML training as fair use | ✅ include with attribution |
| **CulturaX Indonesian** | **ODC-By 1.0** (Open Data Commons Attribution) | ✅ unrestricted with attribution | ✅ permissive commercial | ✅ include — primary path |
| **MADLAD-400 Indonesian** | **ODC-By 1.0** + Common Crawl ToU bound | ✅ unrestricted with attribution | ✅ permissive (CC inheritance caveat) | ✅ include |
| **FineWeb-2 Indonesian** | **ODC-By 1.0** | ✅ unrestricted with attribution | ✅ "intended as research artifact" but license permits commercial | ✅ include — primary path |
| **Indonesian news crawls (Detik/Kompas/Tempo)** | per-source varies, generally non-commercial copyright | ✅ fair use research | ❌ **commercial blocked** for most outlets without licensing | ⚠️ paper-only / RESEARCH path; DROP for commercial |
| **Indonesian books OCR** (gutenberg-id, archive.org-id public domain subset) | **public domain** (70 yrs post-author death per UU Hak Cipta 28/2014) | ✅ unrestricted | ✅ unrestricted | ✅ include (verify per-book PD status) |
| **Indonesian academic/research** (theses, ITB/UI/UGM repos) | per-institution varies; generally academic-fair-use | ✅ fair use research | ⚠️ requires per-institution permission for commercial | ⚠️ research path; per-source consent for commercial |

### English corpus sources (E1.2)

| Source | License | Paper publication | Commercial deployment | Phase E status |
|---|---|---|---|---|
| **SlimPajama subset** (already used in Phase D) | mixed permissive (RedPajama provenance) | ✅ | ✅ permissive | ✅ retain as-is |
| **FineWeb-Edu** (optional EN add-on) | ODC-By 1.0 | ✅ | ✅ | ✅ optional |

### Synthetic generator outputs (E1.5, per E0.0.10)

| Generator | License pass-through to FajarQuant trained model | Status |
|---|---|---|
| Gemma 4 (Apache 2.0) | derivative permitted, commercial OK | ✅ PRIMARY |
| Mixtral-8x7B (Apache 2.0) | derivative permitted, commercial OK | ✅ secondary |
| Qwen2.5 / Qwen3 | output license <100M MAU restriction | ❌ blocked for FajarQuant |
| Llama 3.x | output license <700M MAU restriction | ❌ blocked |
| Claude / GPT API | ToS prohibits competing-model training | ❌ blocked |

### Tax-vertical corpus (E4.1, TaxPrime-owned)

| Source | License | Status |
|---|---|---|
| TaxPrime internal memos (sanitized) | TaxPrime-owned proprietary; founder authority | ✅ proprietary, internal license-grant for Phase E |
| Indonesian tax law (UU/PMK/PER-DJP/SE) | Indonesian government public regulation = freely usable | ✅ public |
| Pengadilan Pajak public decisions | public court records | ✅ public |
| Bilingual tax glossary (TaxPrime-curated) | TaxPrime-owned | ✅ proprietary |

## 2. Two-tier deployment scoping (CRITICAL DECISION)

Phase E deployment splits into **two scopes** with different licensing constraints:

### Tier A: PAPER + RESEARCH ARTIFACT (universally clean)

- All Phase E corpus sources usable for research paper publication (Bartz v. Anthropic 2025: LLM training on copyrighted works = "quintessentially transformative" fair use)
- Includes: news crawls, academic, all the gray-zone sources
- Paper Tables 1-7 fully populated; full reproducibility chain
- **No legal blocker for paper publication**

### Tier B: COMMERCIAL DEPLOYMENT (kernel-context LLM as product)

Two options for commercial deployment scope:

**Option B1 — Permissive-only commercial mix:**
- INCLUDE: OSCAR (CC0), CulturaX (ODC-By), MADLAD-400 (ODC-By), FineWeb-2 (ODC-By), Wikipedia (CC-BY-SA, with attribution), Indonesian books OCR (PD), TaxPrime proprietary
- EXCLUDE: CC-100, news crawls, academic-restricted
- Token impact: drop ~10-15 B raw ID (CC-100 + news + academic) from initial 25-54 B post-dedup → still ~15-30 B usable
- Verdict: **VIABLE — commercial-clean ID corpus achievable**

**Option B2 — Same dataset as paper, commercial under fair-use claim:**
- Use full corpus including gray-zone sources
- Rely on Bartz v. Anthropic fair-use precedent for commercial deployment
- Risk: new lawsuit could establish precedent against; reputational risk if Anthropic-style accusation
- Verdict: **DEFENSIBLE but riskier** — would require legal opinion before commercial launch

**v1.3 plan recommendation: prepare BOTH paper-set (Tier A) AND commercial-clean-set (Tier B1).** Cost: ~+10% data curation effort; benefit: commercial launch readiness without re-train.

### Tier C: THE INDONESIAN-MARKET COMMERCIAL CARVE-OUT

For commercial deployment **specifically in Indonesian market**:
- UU Hak Cipta 28/2014 has fair-use-like provisions (Pasal 43-45) for educational/research/commentary use
- Indonesian tax/legal vertical = arguably falls under "public interest" (Pasal 43 ayat 2 — *"penggunaan ciptaan untuk keperluan pendidikan, penelitian, kepentingan umum"*)
- Indonesian commercial deployment of tax-vertical IntLLM has stronger fair-use defense than generic commercial product
- **Specific Tier 3 (tax-vertical) commercial deployment: defensible even with Tier B2 mix**

## 3. Citation + attribution requirements

For paper §4 + product attribution:

```
The Indonesian + English bilingual corpus used for Phase E training includes:
  - CulturaX (Nguyen et al. 2024) - ODC-By 1.0
  - MADLAD-400 (Kudugunta et al. 2023) - ODC-By 1.0
  - FineWeb-2 (Penedo et al. 2024) - ODC-By 1.0
  - Wikipedia (Wikimedia Foundation, CC-BY-SA 4.0)
  - OSCAR-2301 (Suárez et al. 2022) - CC0-1.0
  - CC-100 (Conneau et al. 2020) - data.statmt.org/cc-100 terms
  - Indonesian news public archives (research-only subset)
  - SlimPajama (Cerebras 2023) - RedPajama provenance
We acknowledge the foundational data work of these sources.
```

## 4. Indonesian legal context

**UU Hak Cipta 28/2014 (Indonesian Copyright Law):**
- Pasal 43: penggunaan terbatas untuk pendidikan, penelitian, kritik, ulasan = fair-use-like
- Pasal 44 (ayat 1): tidak dianggap pelanggaran cipta jika penggunaan untuk kepentingan umum + sumber disebut
- Pasal 9 ayat 2: copyright duration = lifetime + 70 years post-author death

**UU PDP 27/2022 (Personal Data Protection):**
- TaxPrime data sanitization mandatory per `FJQ_PHASE_E_TAXPRIME_DATASET_SPEC.md` §5
- Data residency Art. 56: Indonesian personal data should be processed in Indonesia
- Consent / legitimate interest required for processing

**No specific Indonesian case law on AI training data yet.** Conservative interpretation: Indonesian courts likely follow US fair-use logic for transformative use. Indonesian fair-use clause (Pasal 43-44) supports research + educational + public-interest use.

## 5. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| CC-100 commercial-use challenge | Low (BLOOM/mGPT precedent) | Medium (requires Tier B1 fallback) | Tier B1 dual-corpus prep |
| Wikipedia CC-BY-SA share-alike challenge | Low (Wikimedia accepts ML fair use) | Low | Attribution + share-alike-equivalent of model release docs |
| News crawl copyright suit | Low–Medium | Medium-High | Drop from commercial; research paper path only |
| Academic source consent | Low | Low | Per-source provenance log; drop ambiguous sources |
| TaxPrime data PII residual | Very low (spec §5 mandatory) | Catastrophic | Automated PII check + senior partner sign-off + external counsel review |
| Bartz-style piracy contamination | Very low | Catastrophic ($1.5B precedent) | Source-provenance audit; only legitimately-acquired data |
| Anthropic ToS challenge for trained model | Very low (we don't use Claude as generator) | High | Plan v1.3 §14 R2 + E0.0.10 generator manifest enforcement |

## 6. Acceptance criteria

- [x] All 9 ID corpus sources licensed-mapped
- [x] EN corpus retention from Phase D verified
- [x] Synthetic generator pass-through documented (cross-ref E0.0.10)
- [x] TaxPrime corpus license source captured
- [x] Two-tier deployment scoping (paper vs commercial) defined
- [x] Indonesian commercial carve-out (Tier C) identified
- [x] Citation/attribution boilerplate written for paper §4
- [x] Indonesian legal context (UU Hak Cipta + UU PDP) applied
- [x] Risk register populated (7 risks scored + mitigations)

**Phase E E0.0.7 gate: PASS** (mechanical decision file landed per §6.8 R6).

**🎉 ALL 10 OF 10 E0 SUB-TASKS NOW CLOSED.** Phase E E0 (pre-flight) complete.

## 7. Plan v1.4 patch scope (next commit)

When v1.4 plan patch is written (bundle E0.0.4 + E0.0.7 updates):

- §1 NEW item: Tier B1 commercial-clean corpus subset deliverable (E1.1.1 sub-task)
- §1 NEW item: Tier B2 vs Tier C commercial scope decision recorded
- §2 Q4 + Q7 closed: laptop-only confirmed; licensing per-source captured
- §3 E0.0.4 + E0.0.7: PASS status reflected
- §3 E1.1.1 NEW: Tier B1 corpus subset assembly (light fork of E1.1)
- §3 E3.4 Stretch: option (C) reduced documented
- §4 calendar: 28→34 weeks (or 28 if Medium-only)
- §6 risk register: add commercial-licensing rows; resolve cloud-cost row

---

*Decision committed 2026-04-25. Author: Claude Opus 4.7 + Fajar.*
*Companion: `FJQ_PHASE_E_E0_SYNTHETIC_LEGAL.md` (generator licenses) + `FJQ_PHASE_E_E0_DATA_VERSIONING.md` (storage) + `FJQ_PHASE_E_TAXPRIME_DATASET_SPEC.md` (tax-corpus methodology).*
*Phase E E0 (pre-flight) phase: 10/10 sub-tasks closed. Ready to proceed to E1 (bilingual data curation).*
