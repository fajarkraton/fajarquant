# IntLLM Paper — Accompanying Artifacts

This directory contains the manuscript, citations, figures, evaluation
artifacts, and the arXiv submission tarball for the paper:

> **IntLLM: Compiler-Verified 1.58-Bit Ternary Language Models for
> Operating-System Kernel Inference**
> Muhamad Fajar Putranto (TaxPrime / PrimeCore.id)

The paper is part of the FajarQuant umbrella research repository and
is the V31-cycle deliverable for the bilingual ternary kernel-LLM line
of work (Arms~A+C in the [umbrella README](../../README.md)).

---

## 1. Directory Layout

| Path | Purpose |
|---|---|
| `intllm.tex` | Manuscript LaTeX source (~2400 LOC, 10 numbered sections + abstract) |
| `refs.bib` | BibTeX entries (15 verified citations; see [`docs/INTLLM_RELATED_WORK.md`](../../docs/INTLLM_RELATED_WORK.md) for the 18-paper synthesis backing them) |
| `intllm.pdf` | Compiled PDF (~21 pages, ~810 KB) |
| `intllm-arxiv.tar.gz` | Standalone arXiv submission tarball (~78 KB; manuscript + bib + figures only, no build intermediates) |
| `figures/*.pdf` | Five paper figures generated from JSON artifacts via `scripts/generate_paper_figures.py` |
| `ablations/*.json` | Phase E2 ablation result JSONs (Q5 baseline, balanced_calib, hadamard, F.6.1 outlier-concentration measurements per checkpoint) |
| `ablations/*.pt` | Calibration-map artifacts (saved `running_max` per BitLinear from training) |
| `results/*.json` | Phase D scaling-chain training-result JSONs (Mini, Base, Medium gates + canonical lm-eval benchmarks) |
| `tables/README.md` | Authoritative-source markdown tables (Phase D side; Path A paper tables are inline in `intllm.tex` and verified mechanically against the JSONs in `ablations/` + `results/` rather than rendered to markdown first) |

---

## 2. Rebuilding the PDF Locally

### Quick path (single LaTeX cycle)

```bash
cd paper/intllm
pdflatex intllm.tex
bibtex intllm
pdflatex intllm.tex
pdflatex intllm.tex
```

This produces `intllm.pdf` — should be 21 pages, ~810 KB, with zero
undefined-citation warnings on the final pass.

### Verified path (extract-to-tmp + standalone build)

```bash
cd <repo root>
make arxiv-tarball   # wraps scripts/build_intllm_arxiv_tarball.sh
```

The `arxiv-tarball` target packages the source files, extracts to a
temp directory, runs the full pdflatex+bibtex+pdflatex×2 cycle on the
extracted copy (NOT the source tree), and aborts with `exit 1` if any
pass fails or any undefined-reference warning appears on the final
pass. The same script verifies the tarball that goes to arXiv.

---

## 3. Mechanical Verification of Paper Claims

Per CLAUDE.md §6.9 R7 (pre-publication audit gate), every numeric
literal in `intllm.tex` tables must map to a registered claim backed
by a JSON artifact:

```bash
make verify-intllm-tables
# or directly:
python3 scripts/verify_intllm_tables.py --strict
```

Expected output: `OK — 40 activated claims verified within tolerance;
22 placeholder(s) pending Phase 2/3 results.` Any drift between the
paper text and the underlying artifacts produces `exit 2`.

The 40 activated claims cover:
- 21 Phase D scaling-chain results (Tables 1+2 / `bench_canonical_*.json` + `training_intllm-*.json`)
- 11 Path A Phase E claims (Q5 baseline + 2 negative-result rows in §6+§7)
- 4 F.6.1 outlier-concentration claims (post-V31 bonus measurements)
- 4 F.5.0 cross-comparison claims (training-time vs steady-state running_max)

22 placeholder claims await the deferred `bench-baselines-real-all`
GPU sweep (BitNet 2B4T, MMfreeLM 370M/1.3B/2.7B, SmolLM2 135M/360M,
Pythia-160m × wikitext + hellaswag + mmlu); see
[`docs/FJQ_PHASE_D_PRODUCTION_PLAN.md`](../../docs/FJQ_PHASE_D_PRODUCTION_PLAN.md)
v1.10 §3.4 for the deferred-to-V32 status.

---

## 4. Regenerating Figures

```bash
make paper-figures
# or directly:
python3 scripts/generate_paper_figures.py
```

The script reads `results/training_intllm-*.json`,
`ablations/q5_bilingual_baseline.json`, and the corpus manifests
under `data/phase_e/corpus_id_dedup_exact/`, then produces five
matplotlib-rendered PDFs in `figures/`:

- `scaling_curve.pdf` (Fig 1, §5) — Mini/Base/Medium val_loss vs n_params
- `corpus_pie.pdf` (Fig 4, §6) — Phase E1 corpus token composition
- `hadamard_spike.pdf` (Fig 3, §7) — Walsh-Hadamard rotation on synthetic spike
- `kernel_budget.pdf` (Fig 2, §8) — FajarOS Nova binary footprint vs ≤16 MB budget
- `compiler_pipeline.pdf` (Fig 5, §9) — Fajar Lang compilation pipeline

The script is idempotent and deterministic (modulo creation timestamp
in the matplotlib PDF backend), so re-running produces functionally
identical PDFs and a `git diff` flags only timestamp deltas.

---

## 5. Bonus Empirical Evidence (V31 Post-Path-A)

After Path A Week 4 closure, three additional measurement scripts
strengthened the paper's negative-results attribution
(§7.2 cause-1, §7.3 causes-2+3) without re-running any GPU training:

| Script | Output | Purpose |
|---|---|---|
| `python/phase_d/scripts/measure_outlier_concentration.py` | `ablations/outlier_concentration_{mini,balanced_calib,hadamard}.json` | F.6.1 per-channel max/mean ratio measurement on three trained Mini variants |
| `python/phase_d/scripts/compare_running_max_train_vs_steady.py` | `ablations/running_max_train_vs_steady{,_within_model}.json` | Cross-comparison of training-time running_max vs steady-state channel_max |

Run them via:

```bash
cd python/phase_d

# F.6.1 measurement on a checkpoint
PYTHONPATH=. ../../.venv/bin/python scripts/measure_outlier_concentration.py \
    --checkpoint <path-to-ckpt.pt> \
    --out <output.json> \
    --n-batches 100

# Train-vs-steady comparison
PYTHONPATH=. ../../.venv/bin/python scripts/compare_running_max_train_vs_steady.py \
    --e24-maps  ../../paper/intllm/ablations/mini_balanced_calib_maps.pt \
    --f61-json ../../paper/intllm/ablations/outlier_concentration_balanced_calib.json \
    --out      ../../paper/intllm/ablations/running_max_train_vs_steady_within_model.json \
    --label    within-model
```

Outputs are R7-gated by `verify_intllm_tables.py` so paper claims that
quote these numbers cannot drift from the source artifacts.

---

## 6. arXiv Submission Workflow

The submission tarball at `intllm-arxiv.tar.gz` contains everything
arXiv needs to compile the paper standalone. Founder actions
remain before public upload — see
[`docs/ARXIV_SUBMISSION.md`](../../docs/ARXIV_SUBMISSION.md) for the
full checklist:

1. ORCID iD activation (orcid.org register → edit `intllm.tex` line ~46)
2. Zenodo DOI mint (3 artifacts: code release, Phase E1 corpus, Phase D checkpoints)
3. arxiv.org account + endorsement (cs.LG primary)
4. First-draft end-to-end review pass
5. Upload `intllm-arxiv.tar.gz` to arxiv.org/submit

After step 5, this README's link to `intllm-arxiv.tar.gz` should be
updated to also reference the published arXiv ID + BibTeX entry.

---

## 7. Reproducibility Re-Build

Anyone receiving the source tarball can verify standalone-build:

```bash
tar -xzf intllm-arxiv.tar.gz
cd intllm-arxiv
pdflatex intllm.tex
bibtex intllm
pdflatex intllm.tex
pdflatex intllm.tex
ls -la intllm.pdf  # should be ~810 KB / 21 pages
```

Or, if the builder has access to the fajarquant repo:

```bash
make arxiv-tarball    # extract + 4-pass verify cycle
make verify-intllm-tables  # cross-check 40 R7-gated claims
make paper-figures    # regenerate the 5 figure PDFs
```

All three targets are in the umbrella `Makefile`.

---

## 8. Companion Documents

For the broader Path A scope decision and supporting docs:

- [`docs/FJQ_PHASE_E_PATH_A_PAPER_OUTLINE.md`](../../docs/FJQ_PHASE_E_PATH_A_PAPER_OUTLINE.md) v1.0 — paper structure
- [`docs/INTLLM_RELATED_WORK.md`](../../docs/INTLLM_RELATED_WORK.md) — 18-paper literature synthesis (R2 audit gate)
- [`docs/FJQ_PHASE_E_E2_BILINGUAL_CALIB_DECISION.md`](../../docs/FJQ_PHASE_E_E2_BILINGUAL_CALIB_DECISION.md) v1.0 — E2.4 negative-result decision
- [`docs/FJQ_PHASE_E_E2_HADAMARD_DECISION.md`](../../docs/FJQ_PHASE_E_E2_HADAMARD_DECISION.md) v1.0 — E2.1 negative-result decision
- [`docs/FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md`](../../docs/FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md) v1.10 — overall plan with Path A scope decision
- [`docs/FJQ_PHASE_F_TAX_VERTICAL_ROADMAP.md`](../../docs/FJQ_PHASE_F_TAX_VERTICAL_ROADMAP.md) v1.3 — V32 follow-up entries (F.5/F.6/F.7/F.8/F.9)
- [`docs/ARXIV_SUBMISSION.md`](../../docs/ARXIV_SUBMISSION.md) — founder-action checklist
