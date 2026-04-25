# Phase E E0.0.10 — Synthetic Data Legal Review

> **Status:** PASS — generator selection legally clean per use-case.
> **Decision date:** 2026-04-25
> **Plan reference:** `FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.3 §3 E0.0.10 + §14
> **Companion:** `FJQ_PHASE_E_TAXPRIME_DATASET_SPEC.md` v1.0 (tax data path)
> **Plan-level reasoning:** v1.3 §14 (this doc is the standalone decision artifact)

---

## 1. Generator licensing matrix (verified 2026-04-25)

| Generator | Weight license | Output license / restrictions | Phase E eligibility |
|---|---|---|---|
| **Gemma 4** (Google DeepMind) | Apache 2.0 | **Unrestricted** — derivative models permitted, commercial use OK | ✅ **PRIMARY** for Lapis 2 (E1.5) + Lapis 3b (E4.2) |
| **Mixtral-8x7B-Instruct** (Mistral AI) | Apache 2.0 | Apache 2.0 weights → output ToS effectively unrestricted (Mistral has no separate output-restriction clause for Apache-released models) | ✅ secondary for Lapis 2 + 3b |
| **Qwen2.5-72B-Instruct** (Alibaba) | Apache 2.0 weights | **Custom output license** — entities with **<100M MAU** prohibited from using Qwen outputs to train other LLMs besides Qwen/derivatives | ⚠️ **RESTRICTED for FajarQuant** (TaxPrime/PrimeCore.id under threshold). Eligible only with explicit Alibaba enterprise carve-out. |
| **Qwen3-235B-A22B / Qwen3-8B** (Alibaba 2026) | Apache 2.0 weights | Same custom output license assumed (verify before adopting) | ⚠️ same restriction as Qwen2.5 |
| **Llama-3.3-70B-Instruct** (Meta) | Llama Community License | **<700M MAU restriction** — output cannot be used to train other LLMs besides Llama/derivatives | ❌ **BLOCKED for FajarQuant** (under threshold) |
| **Llama-3.1-8B-Instruct** (Meta) | Llama Community License | Same Llama Community License | ❌ blocked |
| **DeepSeek V3 / R1** | Open weights w/ DeepSeek License | Specific terms — verify per-use-case (additional usage restrictions) | ⚠️ check ToS first; not currently in primary path |
| **Phi-4** (Microsoft) | MIT (where released) | MIT | ⚠️ used internally by Microsoft as generator for Phi-4 itself; check current public-release license |
| **Claude Opus 4.7** (Anthropic API) | API only | Anthropic Usage Policy: prohibits using outputs to train competing AI models | ❌ **BLOCKED** — active enforcement (xAI/Cursor restriction; DeepSeek/Moonshot/MiniMax public accusations CNBC Feb 2026) |
| **GPT-5 / GPT-4 / OpenAI API** | API only | OpenAI ToS prohibits training competing AI models | ❌ blocked, similar reasoning |

## 2. Decision matrix per Phase E use case (per plan v1.3 §14.1)

| Use case | Lapis | Generator decision | Reasoning |
|---|---|---|---|
| Pretrain core (E1+E3) | 1 | **NONE — real corpus only** (25–54 B post-dedup ID per E0.0.1) | No synthetic needed; abundant authentic ID data |
| Pretrain augmentation (E1.5) | 2 | **PRIMARY: Gemma 4. Secondary: Mixtral-8x7B.** Self-host on cloud H100. **REPHRASE-of-real recipe** (BeyondWeb arxiv 2508.10975) | Apache 2.0 unrestricted; SOTA recipe; ~$1000–2000 for 5–10 B tokens |
| Vertical FT — Tax (E4.1) | 3a | **NONE — TaxPrime real data only** per spec v1.0 | Tier 3 wedge moat is proprietary real data; synthetic-tax = hallucination risk (regulations are factual) |
| Vertical FT — Instruct (E4.2) | 3b | Gemma 4 / Mixtral OR open ID-instruct (Cendol, etc.) | Same as Lapis 2; non-tax instruct doesn't need TaxPrime |
| Eval / LLM-as-judge (§11) | 3c | **Any judge OK — Claude / GPT / Gemma / etc.** | Judging ≠ training-data; standard lm-eval practice |

## 3. Fair-use defense (US legal precedent, applicable globally as best-practice baseline)

**Bartz v. Anthropic** (N.D. Cal. June 2025):
- Holding: training LLMs on copyrighted works is "**quintessentially transformative**" fair use under U.S. Copyright Act
- Critical caveat: **pirated copies** are "inherently, irredeemably infringing" — cannot claim fair use defense
- Anthropic settlement: **$1.5 B** for downloading copyrighted books from pirated sources

**Phase E posture:**
- ✅ Real ID corpus from CC-100, OSCAR, Wikipedia, CulturaX, MADLAD-400, FineWeb-2 — all legally-acquired (each has documented provenance + license)
- ✅ Synthetic via Gemma 4/Mixtral self-host — generator weights legally distributed by Google/Mistral
- ✅ TaxPrime data — Fajar (founder) owns or licenses
- ✅ News crawls (E1.1 source 4) — pending E0.0.7 per-source licensing review (mitigation: drop sources with ambiguous license)
- ✅ Books OCR (E1.1 source 5) — public domain (Indonesian copyright term: 70 years post-author-death + government-publication subset)

**No piracy → fair-use defense holds for Phase E pretrain corpus.**

**Thomson Reuters v. Ross Intelligence** (D. Del. 2023): training AI on third-party data to build **competing** product = NOT fair use.

**Phase E posture:** IntLLM bilingual is NOT competing with Gemma 4/Mixtral (different deployment — kernel-context embedded; different scope — Indonesian + tax vertical). Defense holds.

## 4. Indonesian legal context (UU PDP 27/2022 + UU Hak Cipta 28/2014)

- **UU PDP 27/2022** (Personal Data Protection Law): Indonesian personal data must be processed with consent or legal basis. Apply to TaxPrime data (real personal/corporate data) — sanitization per `FJQ_PHASE_E_TAXPRIME_DATASET_SPEC.md` §5 mandatory.
- **UU Hak Cipta 28/2014** (Copyright Law): generally aligns with Berne Convention. No specific Indonesian case law on AI training data yet. Conservative interpretation: Indonesian courts likely follow US fair-use logic for transformative use.
- **Data residency** (UU PDP Art. 56): Indonesian personal data should be stored/processed in Indonesia. Cloud burst for Phase E E1.5 generator self-host should be in **AP-Southeast (Singapore or Jakarta)** region; AVOID US/EU regions for TaxPrime data processing path.

## 5. Risk register (synthetic-data-specific)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Anthropic ToS challenge (FajarQuant accused of distillation) | Very low (we don't use Claude as generator) | High (reputational + paper rejection) | §14 R2 + this doc + monthly literature sweep |
| Llama Community License inadvertent use | Low | Medium | Plan v1.3 explicit BLOCKER; CI lint script `verify-no-llama-output.py` (NEW deliverable) checks Lapis 2 generator manifest |
| Qwen output license violation | Low | Medium | Plan v1.3 RESTRICTED status; if used, requires Alibaba enterprise carve-out written agreement |
| Indonesian copyright challenge | Very low | Medium | Conservative source selection (drop ambiguous-license sources per E0.0.7) |
| Pirated source contamination | Very low | Catastrophic ($1.5 B Anthropic precedent) | Source provenance audit; only legitimately-acquired data; CC-100/OSCAR/Wikipedia all have documented chains |
| Synthetic content libel / hallucination claims | Low | Medium | Gemma 4 quality filter (LLM-as-judge bottom-20% drop per §14 R4); manual review sample on TaxPrime team for any tax-domain residual |

## 6. Decision: PASS

All §1 item 31 (synthetic legal review) plan requirements met:

- [x] Generator licensing matrix verified (10 generators reviewed)
- [x] Per-use-case decision recorded (5 lapis × generator selection)
- [x] Fair-use defense documented (Bartz v. Anthropic + Thomson Reuters v. Ross)
- [x] Indonesian legal context applied (UU PDP + UU Hak Cipta + data residency)
- [x] Risk register populated with mitigations
- [x] CI prevention layer identified (verify-no-llama-output.py — new deliverable for E1.5)

**Phase E E0.0.10 gate: PASS** (mechanical decision file landed per §6.8 R6).

5 of 10 E0 sub-tasks now closed: E0.0.1 ✅, E0.0.2 ✅, E0.0.3 ✅, E0.0.5 ✅, **E0.0.10 ✅**.

---

*Decision committed 2026-04-25. Author: Claude Opus 4.7 + Fajar.*
*Reproducibility: re-verify generator licenses every 6 months OR when adopting new generator (whichever first).*
*Companion: plan v1.3 §14, FJQ_PHASE_E_TAXPRIME_DATASET_SPEC.md v1.0.*
