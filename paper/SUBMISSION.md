---
venue: arXiv (immediate) → MLSys 2027 (primary) → NeurIPS 2026 ML Systems Workshop (fallback)
strategy: hybrid
deadline: 2026-04-25
deadline_internal_v26_c3_2: 2026-04-25
deadline_arxiv: none
deadline_mlsys_2027: TBC — typically Oct/Nov of preceding year, reconfirm against official CFP before C3.3
deadline_neurips_2026_workshop: TBC — typically Aug/Sept, reconfirm before fallback activation
venue_primary: MLSys 2027
venue_fallback: NeurIPS 2026 ML Systems Workshop
venue_immediate: arXiv preprint
author: Muhamad Fajar Putranto, SE., SH., MH.
affiliation: TaxPrime / PrimeCore.id, Jakarta, Indonesia
orcid: deferred — to be registered at https://orcid.org before any venue submission
doi_code: deferred — plan Zenodo via GitHub release integration on next fajarquant tag
doi_data: deferred — plan separate Zenodo DOI for data/kv_cache/ extracted Gemma 4 E2B traces
plan_reference: fajar-lang/docs/V26_PRODUCTION_PLAN.md §C3.2
hygiene_rule: Plan Hygiene Rule 6 — decisions must be committed files
signed_by: Muhamad Fajar Putranto
signed_at: 2026-04-11
---

# FajarQuant Paper Submission Decision

**Status:** committed 2026-04-11 (V26 Phase C3.2, Plan Hygiene Rule 6 mechanical gate satisfied — 14 days ahead of the 2026-04-25 internal deadline).

## Decision

**Hybrid strategy.** arXiv preprint immediately, MLSys 2027 as the primary peer-reviewed venue, NeurIPS 2026 ML Systems workshop as the fallback if MLSys rejects.

## Justification

1. **arXiv preprint first** establishes priority for the three FajarQuant innovations (PCA-based adaptive rotation, fused quantized attention, hierarchical multi-resolution allocation) before any external review window opens. It protects against scoop risk during the long MLSys review cycle and gives reviewers a stable citable artifact regardless of subsequent venue outcome. This matches the V26 plan §6 risk register directive: *"Workshop track as backup; arXiv preprint regardless."*

2. **MLSys 2027 is the best venue fit.** The paper's core contribution — hardware-aware quantization for embedded inference targets — is exactly the systems-meets-ML niche MLSys was created for. The 2-3 bit perplexity wins over KIVI and TurboQuant on real Gemma 4 E2B data (PPL 80.1 at 2-bit vs TurboQuant 117.1, KIVI 231.9) are competitive with prior MLSys-accepted KV cache quantization work. The V26 plan §C explicitly identifies MLSys 2027 as the highest-leverage acceptance target.

3. **NeurIPS 2026 ML Systems workshop is the fallback** because the paper is already 5 pages (workshop length without extension), workshop turnaround is faster (~Dec 2026 vs ~March 2027), the NeurIPS umbrella still provides citation visibility, and pre-authorising the fallback in this committed file removes future decision overhead if MLSys rejects.

## Submission Timeline

| Stage | Action | Target window |
|---|---|---|
| **0** | arXiv preprint upload (current 5-page version, light polish) | Within 2 weeks of C3.7 proofread completion |
| **1** | MLSys 2027 full submission (extended to venue page limit) | Oct/Nov 2026 (TBC against official CFP) |
| **2** | If MLSys rejects → NeurIPS 2026 workshop submission | Aug/Sept 2026 if rejection arrives early; otherwise NeurIPS 2027 workshop |
| **3** | If both reject → arXiv-only is the steady state | n/a |

**External deadline TBC actions:** before C3.3 (venue formatting) commits, reconfirm the MLSys 2027 CFP date and NeurIPS 2026 ML Systems workshop deadline from official sources and update this file in place.

## Author Identification

- **Name:** Muhamad Fajar Putranto, SE., SH., MH.
- **Affiliation:** TaxPrime / PrimeCore.id, Jakarta, Indonesia
- **ORCID:** *deferred* — to be registered at https://orcid.org before any venue submission (free, ~2 minutes; required by most ML venues for corresponding authors)
- **Code DOI:** *deferred* — plan: mint via Zenodo-GitHub integration on tagging fajarquant v0.1.0 (or the next release) before venue submission
- **Data DOI:** *deferred* — plan: separate Zenodo DOI for `data/kv_cache/` extracted Gemma 4 E2B traces before venue submission

ORCID and DOI deferral is acceptable for the C3.2 internal decision file but **must be resolved before C3.6 closes** and cannot block C3.3-C3.7.

## Pre-Submission Checklist (gates the v26-c3 phase)

- [x] **C3.2** Venue strategy committed (this file)
- [ ] **C3.1** Honest split paragraph: Rust runtime FajarQuant vs FajarOS kernel FajarQuant
- [ ] **C3.3** Format for MLSys 2027 template (column width, font, citation style)
- [ ] **C3.4** Supplementary materials (full reproduction commands, dataset checksums, model provenance)
- [ ] **C3.5** Broader impact statement (interpretability cost of aggressive quantization)
- [ ] **C3.6** ORCID registered + Zenodo DOI minted + both wired into `paper/fajarquant.tex`
- [ ] **C3.7** Proofread (3 passes: technical, grammar, clarity)
- [ ] External deadlines reconfirmed before any submission attempt

## Mechanical Gate

A commit-msg hook in this repo (`scripts/install-git-hooks.sh` writes it to `.git/hooks/commit-msg`) blocks any commit whose message contains the `v26-c3` scope token if `paper/SUBMISSION.md` is missing on disk on or after 2026-04-25. The gate is dormant while this file exists; deleting this file re-arms the gate.

## Sign-Off

Decision committed by **Muhamad Fajar Putranto** on **2026-04-11**, satisfying:

- V26 Phase C3.2 task (`fajar-lang/docs/V26_PRODUCTION_PLAN.md` §C3.2)
- Plan Hygiene Rule 6 (decisions must be committed files, not prose paragraphs)
- 14 days ahead of the internal 2026-04-25 deadline
