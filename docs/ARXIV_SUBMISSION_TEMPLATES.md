---
phase: arXiv submission v1 — copy-paste templates for founder
status: ACTIVE 2026-05-01
prereq: ARXIV_SUBMISSION.md v1.1 (5-step founder action checklist)
purpose: Ready-to-paste content for ORCID, Zenodo, arXiv submission forms
---

# arXiv v1 Submission — Copy-Paste Templates

> **Purpose.** Founder external actions (Steps 1-5 of `ARXIV_SUBMISSION.md`)
> require typing/pasting content into orcid.org, zenodo.org, and arxiv.org
> forms. This doc collects all that content in one place so founder can
> copy-paste rather than compose from scratch.
>
> **Pre-flight verified 2026-05-01:**
> - `make verify-intllm-tables` → 40/40 activated PASS (+ 22 deferred placeholders)
> - `make arxiv-tarball` → standalone build PASS (21 pages, 78 KB tarball)
> - `paper/intllm/intllm-arxiv.tar.gz` ready for upload

## Step 1 — ORCID iD activation

**Site:** https://orcid.org/register

**Form fields:**

| Field | Value |
|---|---|
| First name | Muhamad |
| Last name | Putranto |
| Primary email | fajar@taxprime.net (or other TaxPrime/PrimeCore email) |
| Password | (your choice) |
| Visibility | Public (recommended for academic publishing) |
| Default visibility for activities | Public |

**After ORCID iD generated** (format `0000-XXXX-XXXX-XXXX`):

Edit `paper/intllm/intllm.tex` line 55. Replace placeholder with real iD:

```latex
\author{Muhamad Fajar Putranto\thanks{ORCID: <YOUR-REAL-ORCID-iD>. Code + corpus archived at Zenodo: DOI~<from Step 2>.}\\TaxPrime / PrimeCore.id, Jakarta, Indonesia}
```

Then re-run: `bash scripts/build_intllm_arxiv_tarball.sh`

## Step 2 — Zenodo DOI mint

**Site:** https://zenodo.org → Sign in (link with ORCID from Step 1)

**New Upload form fields:**

| Field | Value |
|---|---|
| Resource type | Software (or Dataset, your call — Software for code+corpus is closer) |
| Title | IntLLM v1.0 — Code, Phase E1 Bilingual Corpus, and Trained Checkpoints |
| Authors | Muhamad Fajar Putranto, ORCID: (from Step 1), Affiliation: TaxPrime / PrimeCore.id |
| Description | (paste below) |
| License | Apache License 2.0 |
| Communities | None required (or "Open Science" if you want indexing) |
| Funding | None (or specify if there are grant codes) |

**Description (paste verbatim):**

```
Companion artifact archive for the paper "IntLLM: Compiler-Verified
1.58-Bit Ternary Language Models for Operating-System Kernel Inference"
by Muhamad Fajar Putranto (TaxPrime / PrimeCore.id, Jakarta, Indonesia).

Contents (3 tarballs, all reproducible from the public fajarkraton/fajarquant
repository at the v0.4.0-phase-d release tag):

  1. fajarquant-v0.4.0-intllm.tar.gz
     Source archive of the fajarquant codebase at the IntLLM paper
     submission revision. Includes Phase D training pipeline, Phase E1
     bilingual corpus tooling, FajarQuant v3.1 KV cache quantization
     (Arm B), and verify-intllm-tables R7 audit gate.
     Generated via: git archive --format=tar.gz HEAD

  2. phase-e1-bilingual-corpus-v1.0.tar.gz
     The 25.67 B-token bilingual Indonesian + English training corpus
     described in paper §6:
       - 15.40 B Indonesian tokens (CulturaY ID + FineWeb-2 ID + Wikipedia ID)
       - 10.27 B English tokens (DKYoon/SlimPajama-6B sliced + reduced)
     60:40 ID:EN mix, 0% synthetic, 0.0254% exact-hash dedup rate.
     License attribution per NOTICE_BILINGUAL_CORPUS_V1 included.

  3. intllm-checkpoints-mini-base-medium.tar.gz
     Trained model checkpoints for Mini (22 M), Base (46 M), and Medium
     (74 M) variants. Each is the final-step checkpoint from the
     scaling-chain training runs reported in paper Table 1. PyTorch
     state_dict format; loadable via intllm.model.HGRNBitForCausalLM.

License: Apache 2.0 for code + Apache 2.0 + per-source attribution for
corpus (see NOTICE_BILINGUAL_CORPUS_V1) + Apache 2.0 for checkpoints.

Reproducibility: every numeric claim in the companion paper is backed
by an artifact in this archive + verified by `make verify-intllm-tables
--strict` in the source repo (40/40 activated claims pass at submission
time).

Companion paper: arXiv:<TBD after Step 5 upload>
```

**Reserve DOI** (click button before uploading files — DOI gets reserved
so you can paste it in the paper before files are public).

**After DOI reserved** (format `10.5281/zenodo.<NUMBER>`):

Edit `paper/intllm/intllm.tex` line 55. Replace placeholder DOI:

```latex
\author{Muhamad Fajar Putranto\thanks{ORCID: <from Step 1>. Code + corpus archived at Zenodo: DOI~10.5281/zenodo.<YOUR-DOI-NUMBER>.}\\TaxPrime / PrimeCore.id, Jakarta, Indonesia}
```

Re-run: `bash scripts/build_intllm_arxiv_tarball.sh`

**Building the 3 tarballs to upload to Zenodo:**

```bash
# 1. Source archive
cd ~/Documents/fajarquant
git archive --format=tar.gz --prefix=fajarquant-v0.4.0/ HEAD \
  -o /tmp/fajarquant-v0.4.0-intllm.tar.gz

# 2. Bilingual corpus archive (after make verify-bilingual-corpus 8/8 PASS)
make verify-bilingual-corpus
tar -czf /tmp/phase-e1-bilingual-corpus-v1.0.tar.gz \
  --transform 's,^data/phase_e/,phase-e1-corpus-v1.0/,' \
  data/phase_e/corpus_id_dedup_exact/ \
  NOTICE_BILINGUAL_CORPUS_V1

# 3. Checkpoint archive (NB: ~few GB, may take 5-10 minutes)
tar -czf /tmp/intllm-checkpoints-mini-base-medium.tar.gz \
  --transform 's,^python/phase_d/,intllm-v1.0/,' \
  python/phase_d/checkpoints/mini/mini_final.pt \
  python/phase_d/checkpoints/base/base_final.pt \
  python/phase_d/checkpoints/medium/medium_final.pt

# Verify sizes
ls -lh /tmp/{fajarquant,phase-e1,intllm-checkpoints}*.tar.gz
```

Upload all 3 to Zenodo. Click Publish (or schedule publication).

## Step 3 — arxiv.org account + endorsement

**Site:** https://arxiv.org/user/register

**Form fields:**

| Field | Value |
|---|---|
| First name | Muhamad |
| Last name | Putranto |
| Email | fajar@taxprime.net (or other primary) |
| Password | (your choice) |
| ORCID iD | (from Step 1) |
| Default category | cs.LG |

**Endorsement (if first-time cs.LG submitter):**

Visit https://arxiv.org/auth/need-endorse after registration. arXiv
will issue a unique endorsement code.

**Who can endorse you:**
- Anyone with cs.LG submissions in the last 5 years (auto-eligible)
- Need 1 endorser

**Candidate endorser pools (your network):**

1. **IKANAS STAN alumni** with academic publishing history (you are
   Ketua Umum 2025-2028, ~80K alumni)
2. **TaxPrime + PrimeGlobal** professional network with academic crossover
3. **Indonesian ML research community:**
   - Universitas Indonesia (UI) Faculty of Computer Science
   - Institut Teknologi Bandung (ITB) STEI
   - ITS / UGM / Binus AI labs
4. **International collaborators** from prior tax-tech work that
   crosses into computational research

**Endorsement request email template (paste into your communication
with potential endorser):**

```
Subject: arXiv endorsement request — IntLLM (cs.LG)

Dear [endorser],

Saya mau request endorsement untuk submission arXiv pertama saya di
kategori cs.LG. Paper-nya: "IntLLM: Compiler-Verified 1.58-Bit Ternary
Language Models for Operating-System Kernel Inference" — joint work
TaxPrime / PrimeCore.id, 21 halaman, lima kontribusi termasuk dua
honest negative results.

Endorsement code dari arxiv.org: <PASTE CODE FROM ENDORSEMENT REQUEST PAGE>
Endorsement URL: https://arxiv.org/auth/endorse?x=<CODE>

PDF preprint untuk review (kalau perlu konteks): <Zenodo or repo link>

Terima kasih banyak atas waktunya.

Salam,
Muhamad Fajar Putranto
TaxPrime / PrimeCore.id
ORCID: <from Step 1>
```

Endorsement biasanya 1-3 hari; arXiv kirim email confirmation.

## Step 4 — Editorial review pass

Per `ARXIV_SUBMISSION.md` v1.1 §4, founder reads `paper/intllm/intllm.pdf`
end-to-end (~21 pages, 1-2 hours focused).

**Checklist (mark each as you review):**

```
[ ] §abstract — every C1-C5 claim accurate from your perspective
[ ] §1 introduction — voice/framing reads like Anda's writing
[ ] §4 + §10.5 — interruption-safety incident details accurate
    (laptop suspend 2026-04-22, 8.5h hang, manual shutdown 2026-04-23)
[ ] §5 scaling-chain table — numbers match your memory
[ ] §6 bilingual Q5 baseline — 196M token/42-min framing accurate
[ ] §7 negatives section — both E2.4 and E2.1 attributions match
    your understanding (especially the 3-cause structure)
[ ] §8 FajarOS Nova section — kernel-path narrative accurate
    (16 MB budget claim, Mini/Base fit but Medium overruns marginally)
[ ] §9 Fajar Lang type system — 4-context model description accurate
[ ] §10.5 research-integrity rules — §6.x rule references current
[ ] No typos in title or abstract (these go directly into arXiv metadata)
```

After review: optionally add Acknowledgements section before References.

```latex
\section*{Acknowledgements}

[Add anyone who deserves credit: co-authors, reviewers, hardware
sponsors, family, IKANAS STAN colleagues, TaxPrime / PrimeCore team
members. Empty section is acceptable.]
```

Then tag git state:

```bash
cd ~/Documents/fajarquant
git tag intllm-arxiv-v1.0
git push origin intllm-arxiv-v1.0
```

## Step 5 — arXiv upload

**Site:** https://arxiv.org/submit (after endorsement received)

**Submission metadata (copy-paste these fields):**

**Title:**
```
IntLLM: Compiler-Verified 1.58-Bit Ternary Language Models for Operating-System Kernel Inference
```

**Abstract:**
```
We present IntLLM, a 1.58-bit ternary language model family (22-74 M parameters) trained from scratch and deployed inside the FajarOS Nova v3.9.0 operating-system kernel. Five contributions: (C1) a three-point scaling chain with monotonically widening calibrated-gate margins (0.118 -> 0.279 nat over a ~3.5x parameter range) extends BitNet-style ternary training validation downward from the literature's 2 B+ regime into the kernel-context-relevant 22-74 M range; (C2) a 25.67 B-token bilingual Indonesian + English corpus (60:40 mix, 0% synthetic) and a Q5 Mini-scale baseline establish the first fully-public bilingual ternary LM resources at this scale, reproducible in ~42 minutes on commodity laptop hardware; (C3) deployment of the Phase D Medium checkpoint via three Fajar Lang @kernel modules under a 4-invariant CI regression gate, with binary footprint within the <=16 MB design budget at the Mini and Base scales; (C4) two outlier-handling features ported from the post-hoc-quantization literature (per-channel calibration after SmoothQuant; Hadamard rotation after QuaRot) FAIL their mechanical adoption gates by -82.13 and -0.12 nat respectively, with three-cause attributions identifying a common pattern that we hypothesize generalizes: post-hoc-literature precedents do not transfer cleanly to training-from-scratch ternary at small scale; (C5) a four-context Fajar Lang type system enforces the C3 kernel-context invariants at compile time via seven domain-specific error codes. The combination of these five - none individually a breakthrough - is the contribution; we do not claim frontier benchmark performance, FP16 parity at the same parameter count, or formal verification at the seL4 level. The five-entry Phase F research agenda (post-training PTQ, post-hoc QuaRot, and three more) informs by the C4 negative-results and constitutes a more well-posed follow-up than the original five-feature Phase E plan that the empirical evidence has refuted.
```

(Note: arXiv strips LaTeX in abstract field. The version above is plain-text — `→` replaced with `->`, `≤` replaced with `<=`, `\textbf{}` removed, `~` retained where appropriate.)

**Authors:**
```
Muhamad Fajar Putranto
```

**Affiliation:**
```
TaxPrime / PrimeCore.id, Jakarta, Indonesia
```

**ORCID iD:** (from Step 1)

**Primary subject category:** `cs.LG` (Machine Learning)

**Cross-list / secondary subjects:**
- `cs.OS` (Operating Systems) — for the kernel-path contribution C3
- `cs.PL` (Programming Languages) — for the Fajar Lang type system C5

**License selection:**
- Recommended: `arXiv.org perpetual, non-exclusive license to distribute this article` + `CC BY-SA 4.0` (Creative Commons Attribution-ShareAlike 4.0 International)
- Matches the corpus attribution chain (per NOTICE_BILINGUAL_CORPUS_V1)

**Comments field (optional, helps reviewers):**
```
21 pages. Companion artifacts (code + corpus + checkpoints) at Zenodo DOI <from Step 2>. Source repo: https://github.com/fajarkraton/fajarquant
```

**File upload:**
- Upload `paper/intllm/intllm-arxiv.tar.gz` (78 KB)
- arXiv will compile source automatically
- Review the auto-generated PDF — it should match `paper/intllm/intllm.pdf`
  byte-for-byte modulo arxiv's standard cover page

**Submit. arXiv processes within 1 business day.** You'll receive
confirmation email; paper appears in next-day-announcement window
(typically 4 PM Eastern US time).

## Step 6 — Post-announcement updates

After arXiv assigns paper ID (format `2605.XXXXX`):

```bash
cd ~/Documents/fajarquant

# 1. Update paper README with arXiv ID + BibTeX
# (Edit paper/intllm/README.md — replace "<TBD>" placeholders with real ID)

# 2. Update GitHub repo description
gh repo edit fajarkraton/fajarquant --description "FajarQuant — kernel-resident 1.58-bit ternary LM. arXiv:2605.XXXXX"

# 3. Tag the published version
git tag intllm-arxiv-published-v1.0
git push origin intllm-arxiv-published-v1.0

# 4. Optional: add citation entry to a CITATIONS.md file
```

**BibTeX entry template (paste into ~/Documents/fajarquant/CITATIONS.md):**

```bibtex
@misc{putranto2026intllm,
  title={IntLLM: Compiler-Verified 1.58-Bit Ternary Language Models for Operating-System Kernel Inference},
  author={Muhamad Fajar Putranto},
  year={2026},
  eprint={2605.XXXXX},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  doi={<from Step 2 Zenodo DOI>},
  note={Code, corpus, and checkpoints at Zenodo. Companion repo: https://github.com/fajarkraton/fajarquant}
}
```

## Tracking sheet (for founder use during the multi-day flow)

```
Step 1 — ORCID activation
  [ ] Account created at orcid.org
  [ ] ORCID iD: __________________________
  [ ] paper/intllm/intllm.tex line 55 updated
  [ ] make arxiv-tarball re-run (verifies PDF builds with real iD)
  Date completed: __________

Step 2 — Zenodo DOI
  [ ] Account linked with ORCID
  [ ] 3 tarballs prepared (source / corpus / checkpoints)
  [ ] DOI reserved: 10.5281/zenodo._______
  [ ] paper/intllm/intllm.tex line 55 updated
  [ ] make arxiv-tarball re-run
  [ ] Files uploaded + Published
  Date completed: __________

Step 3 — arxiv.org account
  [ ] Account created at arxiv.org
  [ ] Endorsement requested
  [ ] Endorser identified: ______________
  [ ] Endorsement received
  Date completed: __________

Step 4 — Editorial review
  [ ] All sections reviewed
  [ ] Acknowledgements added (or confirmed empty OK)
  [ ] git tag intllm-arxiv-v1.0 pushed
  Date completed: __________

Step 5 — arXiv upload
  [ ] Submission form filled with metadata
  [ ] Tarball uploaded
  [ ] arXiv compile preview matches local intllm.pdf
  [ ] Submitted
  [ ] Confirmation email received
  [ ] arXiv ID assigned: 2605.________
  Date completed: __________

Step 6 — Post-announcement
  [ ] paper/intllm/README.md updated with arXiv ID
  [ ] GitHub repo description updated
  [ ] git tag intllm-arxiv-published-v1.0 pushed
  Date completed: __________
```

---

*Templates v1.0 — 2026-05-01. Companion to ARXIV_SUBMISSION.md v1.1
5-step founder action checklist.*
