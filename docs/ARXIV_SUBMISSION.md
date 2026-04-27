# IntLLM Paper — arXiv Submission Checklist (v1.0)

> **Status:** Path A Week 4 polish complete; submission tarball
> built and verified. Founder actions remain before public arXiv
> upload.
>
> **Companion docs:**
> - `docs/FJQ_PHASE_E_PATH_A_PAPER_OUTLINE.md` v1.0 (paper structure)
> - `docs/INTLLM_RELATED_WORK.md` v1.0 (18-paper synthesis backing the citations)
> - `paper/intllm/intllm.tex` (manuscript source, 2400+ LOC)
> - `paper/intllm/intllm.pdf` (compiled PDF, 20 pages, 787 KB)
> - `paper/intllm/intllm-arxiv.tar.gz` (the submission tarball, 77 KB)

---

## 0. Submission readiness summary

| Item | Status | Owner |
|---|---|---|
| Manuscript drafted (§1-§10 + abstract) | ✅ DONE | Path A Weeks 1-4 |
| All 5 figures generated + integrated | ✅ DONE | `scripts/generate_paper_figures.py` |
| `refs.bib` with 15 verified citations | ✅ DONE | `paper/intllm/refs.bib` |
| `verify_intllm_tables.py --strict` 32/32 PASS | ✅ DONE | R7 audit gate |
| Standalone tarball builds cleanly | ✅ DONE | `scripts/build_intllm_arxiv_tarball.sh` |
| 20-page compiled PDF (arXiv-quality) | ✅ DONE | `paper/intllm/intllm.pdf` |
| Real ORCID iD activated | ⏳ PENDING | **Founder action** (§1) |
| Real Zenodo DOI minted | ⏳ PENDING | **Founder action** (§2) |
| arxiv.org account + endorsement | ⏳ PENDING | **Founder action** (§3) |
| First-draft end-to-end review pass | ⏳ PENDING | **Founder action** (§4) |
| arXiv upload + announcement | ⏳ PENDING | **Founder action** (§5) |

---

## 1. ORCID iD activation

The author block currently uses placeholder `0000-0000-0000-0000`. Real
ORCID needs to be registered:

1. Go to https://orcid.org/register and create an account
   (free; takes ~5 minutes; uses TaxPrime/PrimeCore.id email).
2. Once registered, copy the 16-digit ORCID iD from the profile.
3. Edit `paper/intllm/intllm.tex` line ~46 (the `\thanks{}` block):

   ```latex
   \author{Muhamad Fajar Putranto\thanks{ORCID: <real-iD>. ...}}
   ```

4. Re-build via `bash scripts/build_intllm_arxiv_tarball.sh` to
   regenerate the tarball.

ORCID activation is reversible (the placeholder commit can be
amended), but worth doing before arXiv upload so the v1.0 announcement
links to a real iD from day one.

---

## 2. Zenodo DOI for corpus + code release

The author block also references
`DOI~10.5281/zenodo.PLACEHOLDER`. To activate:

1. Go to https://zenodo.org and sign in (free; can link to ORCID).
2. Create a new upload:
   - **Title:** "IntLLM v1.0 — Code, Phase E1 bilingual corpus, and trained checkpoints"
   - **Authors:** Muhamad Fajar Putranto (PrimeCore.id)
   - **License:** Apache-2.0 for code + corpus (matches repo license)
   - **Files:** upload three artifacts:
     a. `fajarquant-v0.4.0-intllm.tar.gz` (code release; produced by `git archive --format=tar.gz HEAD > fajarquant-intllm.tar.gz`)
     b. `phase-e1-bilingual-corpus-v1.0.tar.gz` (corpus; produced from `data/phase_e/corpus_id_dedup_exact/` after running `make verify-bilingual-corpus`)
     c. `intllm-checkpoints-mini-base-medium.tar.gz` (Phase D trained checkpoints from `python/phase_d/checkpoints/`)
3. Reserve the DOI before publishing (Zenodo offers "reserve DOI" so
   the same DOI can be referenced in the paper before the artifacts
   are uploaded; useful for paper version-of-record consistency).
4. Edit `paper/intllm/intllm.tex` line ~46 with the real DOI.
5. Re-build the tarball.

Zenodo deposits are versioned: v1.0 of the artifact set goes with v1.0
of the paper. Future updates (Phase E2 follow-up via Phase F) will
mint v2.0 against an updated paper version.

---

## 3. arxiv.org account + endorsement

arXiv requires submitter endorsement for first-time authors in a
subject area. Steps:

1. Go to https://arxiv.org/user/register and create an account.
2. The IntLLM paper is best classified under **cs.LG** (Machine Learning)
   primary + **cs.OS** (Operating Systems) secondary. cs.LG typically
   requires no endorsement for established researchers; if the
   founder's first arXiv submission, an endorsement code from a
   prior cs.LG arXiv author is needed (typical sources: a co-author,
   a research-collaborator, or a paper reviewer who has prior cs.LG
   submissions).
3. The MLSys-2027-bound v3.1 KV cache paper by the same author
   (`paper/fajarquant.pdf`) may already establish authorship in
   cs.LG; if so, no separate endorsement needed for v1.0.

If endorsement is the binding constraint and external endorsement is
slow to secure, the paper can be uploaded to alternative venues first
(e.g., openreview.net, hugging-face papers, github releases) while
arXiv endorsement processes.

---

## 4. First-draft end-to-end review pass

Before public release, founder reads the manuscript end-to-end and
edits for:

1. **Personal voice + framing.** The current draft was written by an
   AI co-author (Claude Opus 4.7) following the methodology rules of
   `CLAUDE.md`; founder's editorial voice should pass through every
   section to ensure attribution accuracy and tonal consistency.

2. **Honesty cross-check.** Every "we did X" claim should match
   founder's actual experience of the work. Per CLAUDE.md §6.6 R3,
   any inflated claim caught in this pass should be downgraded;
   per §6.9 R3, any unmentioned-but-relevant baseline-comparison
   caveat should be added.

3. **Personal-experience details.** The methodology sections (§4 +
   §10.5) reference specific incidents (laptop suspend on 2026-04-22,
   8.5-hour hang, manual shutdown on 2026-04-23). Founder confirms
   these are accurate as written.

4. **Acknowledgements.** Add an Acknowledgements section before
   References if there are co-authors, reviewers, or hardware
   sponsors who should be credited (currently absent).

5. **Submission-version pinning.** Once review-pass complete, tag
   the submission state in git: `git tag intllm-arxiv-v1.0` so the
   paper can be reproduced from a specific commit later.

---

## 5. arXiv upload + announcement

Once §1-§4 are complete:

1. Go to https://arxiv.org/submit and start a new submission.
2. Primary subject: **cs.LG** (Machine Learning).
3. Secondary subjects: **cs.OS** (Operating Systems), optionally **cs.PL**
   (Programming Languages) if reviewers prefer the C5 angle highlighted.
4. Upload `paper/intllm/intllm-arxiv.tar.gz` (verified by
   `scripts/build_intllm_arxiv_tarball.sh`).
5. arXiv compiles the source; review the auto-generated PDF for any
   discrepancy vs `paper/intllm/intllm.pdf`.
6. Confirm metadata: title, abstract (paste from §abstract), authors,
   ORCID iD, license (CC BY-SA 4.0 recommended for paper text;
   matches the corpus attribution chain).
7. Submit; arXiv typically processes within 1 business day for
   established cs.LG authors. The paper appears on arXiv next-day-
   announcement window.

After announcement:
- Update `paper/intllm/README.md` with the arXiv ID and BibTeX entry.
- Update fajarquant repo description on GitHub with the arXiv link.
- Notify the MLSys 2027 program committee (when CFP opens, typically
  Aug-Oct 2027) by submitting the same `intllm-arxiv.tar.gz` adapted
  to the MLSys conference template.
- Tweet/announce on professional channels (LinkedIn, Twitter/X, etc.)
  per the founder's preferred outreach pattern.

---

## 6. Reproducibility re-build verification

Anyone receiving the tarball can verify standalone-build:

```bash
tar -xzf intllm-arxiv.tar.gz
cd intllm-arxiv
pdflatex intllm.tex
bibtex intllm
pdflatex intllm.tex
pdflatex intllm.tex
ls -la intllm.pdf  # should be ~787 KB / 20 pages
```

Or, if the builder has access to the fajarquant repo:

```bash
bash scripts/build_intllm_arxiv_tarball.sh
```

The script does the full extract + 4-pass build cycle in
`/tmp/intllm-arxiv-verify-*/` and confirms zero undefined-reference
warnings on the final pass.

---

*Document version: 1.0. Author: Claude Opus 4.7 + Fajar (PrimeCore.id).*
*Path A Week 4 polish — submission tarball verification complete; founder action items remain before public arXiv upload.*
