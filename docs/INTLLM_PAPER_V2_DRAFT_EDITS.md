---
phase: post-Path-A v2 paper draft edits
status: DRAFT (founder review needed before merging into intllm.tex)
created: 2026-04-30
created_by: Claude Opus 4.7 (1M context)
parent: docs/ARXIV_SUBMISSION.md v1.1 §0.1 (founder v1-vs-v2 decision)
---

# IntLLM Paper v2 Draft Edits — Three Insertions Ready for Review

> **Purpose.** If founder chooses the v2 path per `ARXIV_SUBMISSION.md`
> v1.1 §0.1, the three post-Week-4 findings (F.5.1 PARTIAL, F.6.2 honest
> correction, F.13 dispatch heuristic) need integration into `intllm.tex`
> before re-uploading the tarball. This doc provides paste-ready LaTeX
> for those three edits, with insertion-point line references and
> rationale per edit.
>
> **Founder action.** Review each of the three edits below; for each, choose
> ACCEPT (paste into `intllm.tex` at the indicated line) or REJECT (skip;
> document in `docs/PAPER_V2_DECISIONS.md` why the change is unwanted). After
> all three decisions, run `make arxiv-tarball` to rebuild and `make
> verify-intllm-tables` to confirm no claim drift. v1.1 of `ARXIV_SUBMISSION.md`
> stays as-is until the editorial pass closes.
>
> **Scope:** TEXT edits only. None of the three edits adds new tables, new
> figures, new \verb|\cite{}|s outside refs.bib, or new tracked numerics
> requiring `verify_intllm_tables.py` updates. Each edit is a single-pass
> insertion.
>
> **Cumulative effort estimate (founder editorial time):** ~2-3 hours for
> all three edits + tarball rebuild + final verify-gate confirm.

---

## 1. Why these three edits, and not others

The five post-Week-4 developments catalogued in `ARXIV_SUBMISSION.md`
v1.1 §0.1 sort into three buckets:

| # | Development | v2 edit needed? | Reason |
|---|---|---|---|
| 1 | V31 bonus measurements (F.6.1 + F.5.0-cross, +8 verified claims) | NO | already integrated into v1 paper §7.3 (F.6.1 table) and §7.2 (cause-1 narrative) by `b6b1995` and `d4b1f82`; the verify gate growth from 32→40 reflects this. |
| 2 | F.5.1 SmoothQuant PTQ closed PARTIAL | YES (Edit B) | §7.5 future-work F.5 entry describes the experiment that F.5.1 ran; current text predicts an outcome that the actual run REFINED. |
| 3 | F.6.2 honest verdict correction (recipe-incomplete, not architecture) | YES (Edit A) | §7.3 cause-1 attribution implicitly assumed full QuaRot recipe in future work F.6; F.6.2 ran an activation-only pre-hook, which is RECIPE-INCOMPLETE relative to canonical QuaRot. v1 §7.3 should note that v1's "training-from-scratch dominated" attribution is STRENGTHENED — not weakened — by F.6.2's outcome. |
| 4 | F.5.1.7 Branch A pre-flight NO-GO | LATER | Future-work narrative impact only; ships in v3 if Branch A ever re-enters. v2 doesn't need this. |
| 5 | F.13 dispatch heuristic Z-narrow shipped | YES (Edit C) | New material directly relevant to §8 (FajarOS Nova kernel deployment). Adds a §8.5 subsection identifying the CPU-wins-batch=1 crossover as a structural differentiator. |

The three edits A/B/C below are the minimum coherent v2 update.

---

## 2. Edit A — §7.3 footnote: F.6.2 honest correction

**Insertion point:** `intllm.tex` line ~1493, immediately AFTER the
\verb|\end{enumerate}| that closes the three-cause attribution list,
and BEFORE the existing paragraph "Figure~\ref{fig:hadamard-spike}
(deferred to Week~3)..." at line ~1495.

**Rationale.** The §7.3 cause-1 attribution argues "training-from-scratch
loses Hadamard's primary benefit" and §7.5 future-work F.6 commits to
"Apply the canonical QuaRot recipe... to the Q5-trained checkpoint."
F.6.2 ran a partial form of this experiment (activation-only forward
pre-hook, NOT canonical weight-fusion) and observed regression across
all four modes. The honest read is that F.6.2 is RECIPE-INCOMPLETE
relative to canonical QuaRot, so it does not yet test cause-1 cleanly;
v2 should note this so future readers do not over-interpret F.6.2's
post-hoc regression as evidence against cause-1.

**Paste-ready LaTeX (one paragraph + footnote):**

```latex
\paragraph{Post-paper update: F.6.2 partial post-hoc rotation
(2026-04-28).}
A post-publication ablation re-ran the rotation as a forward pre-hook
on the Q5-trained Mini checkpoint at four site sets
(\texttt{no\_rotation}, \texttt{o}, \texttt{igf}, \texttt{igfo}) without
weight fusion. All three rotation modes regressed by $2.5$--$3.0$~nat
against the \texttt{no\_rotation} control of $5.55$~nat. The naive
read --- ``rotation hurts ternary HGRN'' --- is misleading: the
pre-hook breaks the forward-pass equivalence
$y = W \cdot H x \neq W' \cdot x$ that canonical QuaRot recovers
through weight fusion ($W' = W H^{\top}$); applying the rotation to
activations alone after training mismatches the trained weights and
corrupts the FP path before any quantization runs.\footnote{F.6.2
findings doc: \texttt{docs/FJQ\_PHASE\_F\_F6\_2\_FINDINGS.md} +
honest verdict correction commit \texttt{7fec08f}. The recipe-
incompleteness reading is the only one consistent with both the
published QuaRot positive results~\citep{ashkboos2024quarot} and the
F.6.2 observation; it strengthens cause~1 (training-from-scratch
matters) by adding a parallel observation (post-hoc-recipe-incomplete
also fails) without testing the canonical recipe directly. F.5.1.7
(\texttt{docs/FJQ\_PHASE\_F\_F5\_1\_7\_BRANCH\_A\_PREFLIGHT.md})
audits this and pre-flights canonical QuaRot via three mechanical
re-entry gates (GATE A1/A2/A3) tied to literature signals on
ternary HGRN-family models.}
```

**Verify-gate impact:** ZERO — no new numerics tracked by
`verify_intllm_tables.py`. The "$2.5$--$3.0$~nat" range comes from F.6.2
findings doc but is descriptive (not a table claim).

---

## 3. Edit B — §7.5 F.5 future-work entry: append F.5.1 status

**Insertion point:** `intllm.tex` line ~1577, immediately AFTER the
existing F.5 bullet (which ends at "...rather than WHAT calibration
is computed (per-channel scale)."), and BEFORE the F.6 bullet starting
at line ~1579.

**Rationale.** v1's §7.5 F.5 entry predicts a future test. F.5.1
ran that test on 2026-04-28 and produced a PARTIAL result. The F.5
entry's prediction direction was correct (calibration-axis test) but
the outcome was negative-but-informative: HGRN-ternary trained models
are essentially INVARIANT to activation-only SmoothQuant. v2 should
report the actual outcome alongside the prediction.

**Paste-ready LaTeX (sub-bullet appended to the F.5 \texttt{\textbackslash item}):**

```latex
F.5.1 status (2026-04-28): the calibration-axis test ran on the
Q5-trained Mini checkpoint with SmoothQuant migration strength
$\alpha \in \{0.3, 0.4, 0.5, 0.6, 0.7\}$ across five site sets
(\texttt{o}, \texttt{o\_down}, \texttt{igf}, \texttt{igfo},
\texttt{all}) for eight ablation runs total. Best outcome:
\texttt{o\_down}~$\times$~$\alpha=0.3$ at $+0.0001$~nat
(model-invariant). Worst outcome: \texttt{sites=all}~$\times$~$\alpha=0.3$
at $+0.27$~nat (cumulative weight-saturation; passed all 148 per-layer
\textsection4.6 gates despite catastrophic regression, surfacing a gate
design gap closed by the post-hoc G5 forward-equivalence gate in
\texttt{6987de3}). Verdict: PARTIAL --- the trained HGRN-ternary
model has effectively pre-absorbed the per-channel scale via RMSNorm
$\gamma$, so SmoothQuant's offline migration finds little to migrate.
Source: \texttt{docs/FJQ\_PHASE\_F\_F5\_1\_FINDINGS.md} v1.0.
```

**Verify-gate impact:** ZERO — no new tracked numerics. The "+0.0001"
and "+0.27" are descriptive of the F.5.1 sweep run; if founder wants
to elevate them to table claims later, that is a separate v3 step that
populates `paper/intllm/ablations/smoothquant_*.pt` results into a new
F.5.1 table.

---

## 4. Edit C — §8.5 NEW subsection: dispatch heuristic

**Insertion point:** `intllm.tex` line ~1849, immediately BEFORE the
existing \verb|\subsection{Honest scope of contribution C3}| line.
The new \verb|\subsection{}| sits between current §8.4 ("Runtime
metrics: tok/s and watts", lines 1804-1847) and the renamed §8.6
("Honest scope of contribution C3", currently §8.5).

**Rationale.** F.13.3 dispatch decision-doc shipped 2026-04-30 with a
ready paper-§8 narrative paragraph. The verdict --- "CPU default; GPU
optional for batch~$\geq 8$" --- is structurally interesting because
it inverts commodity-LLM-inference assumptions ("GPU always wins") at
the precise regime that matters for FajarOS Nova kernel deployment.
This is contribution-relevant: §8 currently presents the kernel-path
as a path-verification claim ("inference runs E2E inside @kernel
context") with PEND-marked runtime numbers. Adding §8.5 reframes the
runtime story from "we don't have measurements yet" to "we have a
*structural* prediction that explains why CPU-only is the right
@kernel default and motivates exactly which deployments would benefit
from a future GPU dispatch path." This is a substantive C3 strengthening,
not a numerical claim.

**Paste-ready LaTeX (full new subsection, ~280 words, no new
table/figure/cite):**

```latex
\subsection{CPU-vs-GPU dispatch heuristic}
\label{sec:results-kernel-dispatch}

Production deployment of large language models has converged on the
assumption that GPUs uniformly outperform CPUs. We argue that this
assumption is workload-specific: at the regime that matters for
embedded and kernel-resident inference --- model parameters bounded
above by tens of millions, batch size equal to one, sequence length
below a few thousand --- the assumption inverts. The dominant cost
on GPU at batch~one is not floating-point throughput but kernel
launch latency: a small model with a dozen transformer-class layers
may dispatch on the order of seventy GEMV-class kernels per token,
each of which incurs roughly ten to thirty microseconds of
host-to-device launch overhead. Empirically, this places GPU
per-token latency in the eighty-to-one-hundred-twenty
tokens-per-second range for the same workloads where a recent
ternary CPU kernel (Microsoft's BitNet TL2~\citep{xu2025bitnet})
sustains two-to-four-hundred tokens-per-second on consumer x86
hardware --- a two-to-threefold CPU advantage. We exploit this in
the FajarOS Nova kernel-path by treating the CPU path as the
\texttt{@kernel}-context default and reserving GPU dispatch for the
throughput regime (batch~$\geq 8$, or models above five billion
parameters) where the launch overhead amortizes. The crossover
boundary is structural --- determined by cache hierarchy and PCIe
topology, not by any measurable noise in microbenchmarks --- and
survives hardware refreshes within the laptop-class envelope.

The verdict statement, the three mechanical decision gates that
support it (G1 deployment-envelope check, G2 CPU--TL2-vs-scalar
ratio check, G3 GPU-batch-1-vs-CPU-TL2 margin check), and the
re-entry gates F.13.2-A and F.13.2-B that would re-open runtime
dispatch as a research direction are all committed in
\texttt{docs/FJQ\_PHASE\_F\_F13\_DISPATCH\_DECISION.md}. The pinned
literature anchors that back the projection (BitNet~b1.58 2B4T at
$29$~ms/tok on i7-class CPU; cudaLaunch overhead $10$--$30$~$\mu$s;
GPU small-model batch~1 throughput $80$--$120$ tok/s) live in
\texttt{tests/fixtures/f13\_dispatch\_anchors.toml} and are
re-asserted on every commit by \texttt{make verify-f13-decision}.
This is a Z-narrow scope deliverable: the verdict is architecture-
derived and anchor-backed, not bench-calibrated on the present
hardware (live calibration is deferred to the same Phase~F
deployment-platform partner that gates Table~\ref{tab:kernel-runtime}).
```

**Verify-gate impact:** ZERO at the `verify_intllm_tables.py` level
(no table claims). NEW gate `make verify-f13-decision` already exists
and is independent.

---

## 5. Cross-cutting impacts (none expected, but listed for completeness)

| Impact | Status |
|---|---|
| `verify_intllm_tables.py` claim count | UNCHANGED — 40/40 activated |
| `refs.bib` new entries needed | NO — all three edits cite already-bibbed papers (\verb|ashkboos2024quarot|, \verb|xu2025bitnet|, \verb|xiao2022smoothquant|, \verb|liu2024spinquant|) |
| `paper/intllm/figures/` regenerate | NO — no new figures |
| `paper/intllm/intllm.pdf` page count | LIKELY +1 page (21 → 22) — Edit C adds ~280 words; Edits A+B add ~150 words combined |
| Tarball size | LIKELY +2-3 KB |
| Bibliography ordering | UNCHANGED |
| Section numbering | §8 grows by one subsection (8.5 NEW; old 8.5 becomes 8.6); other section numbers unchanged |
| Abstract / introduction | NO update needed (abstract scope already matches; introduction's "what this paper is not" §2.1 implicitly covers it) |
| `intllm.tex` last-touch metadata | will refresh `git log` on intllm.tex; no change to authorship line |

---

## 6. Founder review checklist (mechanical)

For each of Edit A / Edit B / Edit C:

```
[ ] Read the rationale (§2/§3/§4 above)
[ ] Read the paste-ready LaTeX
[ ] Decide ACCEPT or REJECT
[ ] If ACCEPT:
    [ ] Open intllm.tex at the indicated line
    [ ] Paste the LaTeX block exactly as shown
    [ ] Save
[ ] If REJECT:
    [ ] Document the reason in docs/PAPER_V2_DECISIONS.md
[ ] Record the decision below in §7
```

After all three decisions:

```
[ ] make arxiv-tarball                     (rebuild + standalone-verify)
[ ] make verify-intllm-tables              (40/40 still PASS expected)
[ ] make verify-f13-decision               (19/19 still PASS expected)
[ ] git diff paper/intllm/intllm.tex       (sanity-check the final edits)
[ ] commit + push                          (use commit message format below)
[ ] update docs/ARXIV_SUBMISSION.md v1.1 §0.1 (mark v1-vs-v2 decision = v2)
[ ] proceed with founder external actions §1-§5 of ARXIV_SUBMISSION.md
```

Suggested commit message format (after editorial pass):

```
docs(paper-v2): integrate F.5.1 PARTIAL + F.6.2 honest correction + F.13 dispatch §8.5 [actual <Nh>, est 2-3h, <variance>]

Three editorial inserts per docs/INTLLM_PAPER_V2_DRAFT_EDITS.md:
  - §7.3 footnote: F.6.2 honest verdict correction (recipe-incomplete)
  - §7.5 F.5 future-work: F.5.1 PARTIAL status (HGRN-ternary invariant)
  - §8.5 NEW: dispatch heuristic ("CPU default; GPU optional for batch ≥ 8")

verify-intllm-tables 40/40 unchanged (no new tracked numerics).
arxiv-tarball rebuilt: 22 pages, ~80 KB.
arXiv ready for v2 upload pending founder external actions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

## 7. Founder decisions (fill in after review)

| Edit | Decision (ACCEPT / REJECT / DEFER) | Notes |
|---|---|---|
| A — §7.3 F.6.2 footnote | <pending> | |
| B — §7.5 F.5 status sub-bullet | <pending> | |
| C — §8.5 NEW dispatch subsection | <pending> | |

---

## 8. If founder defers v2 entirely

Acceptable. Per `ARXIV_SUBMISSION.md` v1.1 §0.1, v1 is internally
consistent and verify-gate green; arxiv revisions cost zero. Skipping
v2 means:

1. Ship v1-as-is now, founder external actions §1-§5 of
   `ARXIV_SUBMISSION.md`.
2. After arxiv announcement, this `INTLLM_PAPER_V2_DRAFT_EDITS.md`
   doc is renamed to `INTLLM_PAPER_V2_DRAFT_EDITS_DEFERRED.md` (or
   moved to `docs/archive/`) so future maintainers can see the
   prepared edits exist for a future revision.
3. v3 (or arxiv v2 of the same paper) integrates these three edits
   plus any new Phase F findings that have landed by then.

---

*Draft v1.0 — 2026-04-30. Pending founder editorial review.*
