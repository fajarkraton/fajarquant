# arXiv Upload Guide — FajarQuant v3.1

This document is the step-by-step checklist for uploading `paper/fajarquant.tex`
to arXiv as a preprint. Per `paper/SUBMISSION.md` §Stage 0, the preprint goes
up before any peer-reviewed venue submission (MLSys 2027, NeurIPS workshop).

## Prerequisites

| Item | Action | Status |
|------|--------|--------|
| arXiv account | Register at https://arxiv.org/user/ | ☐ user action |
| Endorsement (first-time submitter to cs.LG) | Request from an existing arXiv cs.LG author, OR submit to cs.PF first (Performance — often auto-endorses) | ☐ user action if needed |
| ORCID ID | Required in arXiv metadata | ☐ user action (see `SUBMISSION.md` C3.6) |

**Note on endorsement:** if this is Fajar's first cs.LG submission, arXiv
may require endorsement from a prior cs.LG author. The fastest workaround
is submitting to **cs.PF (Performance)** or **cs.DC (Distributed Computing)**
as primary category — these usually allow first-time submissions without
explicit endorsement. FajarQuant fits both categories well.

## Build the Tarball

The repository already ships a reproducible Makefile target:

```bash
cd paper
make clean           # reset build state
make                 # build fajarquant.pdf (verify paper still compiles)
make arxiv           # package arxiv_v3.1/ → fajarquant_arxiv_v3.1.tar.gz
make arxiv-verify    # isolated compile check (confirms arXiv will succeed)
```

**Expected output:**

```
arXiv tarball ready: fajarquant_arxiv_v3.1.tar.gz
Contents:
-rw-rw-r-- user/user 49689 ... fajarquant.tex
-rw-rw-r-- user/user  4624 ... fajarquant.bbl
-rw-rw-r-- user/user  5413 ... references.bib
```

**Tarball size:** ~20 KB compressed, ~60 KB uncompressed (3 source files
only, no figures, no custom .sty dependencies beyond the standard TeXLive
packages listed in `\usepackage` lines of `fajarquant.tex`).

## Submission Steps

1. Go to https://arxiv.org/submit

2. **Start a new submission:**
   - License: pick `CC BY 4.0` (most permissive, recommended for ML)
   - Primary category: `cs.LG` (Machine Learning), or `cs.PF` if endorsement blocks
   - Secondary categories: `cs.PF` (Performance), `cs.DC` (Distributed Computing)

3. **Upload the tarball** `paper/fajarquant_arxiv_v3.1.tar.gz`
   - arXiv auto-detects `fajarquant.tex` as the main file
   - arXiv runs `pdflatex` twice to resolve references (we ship the .bbl
     so no bibtex run is needed)
   - Expected: 12 pages, 608 KB PDF output (matches local `make` result)

4. **Metadata:**
   - **Title:** *FajarQuant v3.1: Adaptive Per-Head KV Cache Quantization
     with Compile-Time Safety Guarantees*
   - **Author:** `Muhamad Fajar Putranto`
   - **ORCID:** *(fill in once registered — see `SUBMISSION.md` C3.6)*
   - **Affiliation:** `TaxPrime / PrimeCore.id, Jakarta, Indonesia`
   - **Abstract:** copy from `fajarquant.tex` lines 38–70 (the `\begin{abstract} ... \end{abstract}` block, stripping LaTeX commands)
   - **Comments field:** `12 pages, no figures. Extended MLSys 2027 submission in fajarquant_mlsys.tex (10 pages).`
   - **MSC / ACM classes:** leave blank (cs.LG doesn't use them)
   - **DOI:** *(fill in Zenodo code DOI once minted — see `SUBMISSION.md` C3.6)*

5. **Preview the compiled PDF on arXiv** before confirming. Check:
   - Abstract renders correctly
   - Tables 1–6 all render (multirow, makecell)
   - Bibliography resolves (natbib citations work)
   - Page count is ~12 (matches local build)

6. **Announce (confirm submission):**
   - Select "Announce now" — paper becomes public at next arXiv
     distribution cycle (Mon–Fri, typically within 24h of confirmation)
   - Note the arXiv ID (format: `2604.XXXXX`) — this is the citable
     identifier

## Post-Submission Tasks

After the arXiv ID is assigned:

1. **Update README.md** — add arXiv badge and link:
   ```markdown
   [![arXiv](https://img.shields.io/badge/arXiv-2604.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2604.XXXXX)
   ```

2. **Update `paper/SUBMISSION.md`:**
   - Flip the arXiv line in Stage 0 table to `[x]` with date and arXiv ID
   - Add `arxiv_id: 2604.XXXXX` to the frontmatter

3. **Update citation in README.md** — replace the github URL with
   `https://arxiv.org/abs/2604.XXXXX` in the BibTeX block.

4. **Announce the preprint** (optional):
   - Twitter/X, LinkedIn, personal website
   - Any ML Systems mailing list relevant to MLSys 2027 audience

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| arXiv refuses tarball | .bbl not included | Re-run `make clean && make && make arxiv` |
| Compile fails on arXiv | Missing package | Check `fajarquant.tex` `\usepackage` lines — all are in standard TeXLive |
| "Endorsement required" error | First-time cs.LG submission | Submit to `cs.PF` primary instead, OR request endorsement from an arXiv author at https://arxiv.org/auth/request-endorsement.php |
| PDF >10 MB | Shouldn't happen (no figures) | Investigate — v3.1 PDF is ~600 KB |
| References don't resolve | .bbl path issue | Verify tarball contains `fajarquant.bbl` not just `references.bib` |

## Reference Documents

- `paper/SUBMISSION.md` — venue strategy and full C3 checklist
- `paper/fajarquant.tex` — source of truth for paper content
- `paper/Makefile` — `arxiv` and `arxiv-verify` targets
- `reproduce.sh` — companion reproducibility script for reviewers

---

*Guide version: 1.0 (2026-04-16). Update when arXiv workflow or paper
version changes.*
