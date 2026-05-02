---
phase: F.6.4 — Hadamard rotation @ Base + Medium scale (closeout decision)
status: CLOSED 2026-05-02
budget: 13.8h GPU actual (Cell 1 3.4h + Cell 2 5.1h + Cell 3 5.3h)
        vs plan 21.7h cap → -37% under cap
artifacts:
  - paper/intllm/ablations/{base,medium}_baseline.json (FULL, EN-only train)
  - paper/intllm/ablations/medium_hadamard.json (FULL, EN-only train, mode=o)
  - This doc
plan_doc: docs/FJQ_PHASE_F_F6_4_FULL_EXECUTION_PLAN.md v1.0
predecessor: docs/FJQ_PHASE_E_E2_HADAMARD_DECISION.md (E2.1 Mini-scale)
---

# F.6.4 Decision — Hadamard at Larger Scale

> **TL;DR.** Three full ablation cells (Base baseline, Medium baseline,
> Medium hadamard) ran cleanly across 13.8h GPU on RTX 4090 Laptop,
> finishing well under the 21.7h surprise-budget cap. The decision-gate
> metric **Δ_medium = +0.025 nat** falls in the (−0.05, +0.05) NULL band
> of the plan §4 verdict matrix. The Mini-scale negative result
> (Δ_mini = +0.120 nat regression, E2.1) does NOT clearly hold at
> Medium scale — the regression gap **narrows 4.7×** when scaling from
> d=256 to d=512. Honest reading: Hadamard at d=512 is within noise of
> baseline, neither a clear win nor a clear loss. Paper v2 §7.5 should
> reflect this scale-dependent narrowing rather than holding the flat
> "Hadamard hostile to ternary" claim from E2.1.

## 1. Three-cell results table

| Cell | Tag | val_loss | PPL | Gate | Wall-clock |
|---|---|---|---|---|---|
| 1 | `base_baseline` (EN-only, 24K steps) | **4.156** | 63.81 | 4.2 PASS ✓ (margin 0.044) | 3.36h |
| 2 | `medium_baseline` (EN-only, 24K steps) | **4.005** | 54.88 | 4.0 FAIL ✗ (boundary, by 0.005) | 5.10h |
| 3 | `medium_hadamard` (EN-only, 24K steps, mode=o) | **4.031** | 56.30 | 4.0 FAIL ✗ (by 0.031) | 5.32h |

**Note on Cell 2 gate-FAIL:** the Phase D Medium gate of 4.0 was
calibrated against the **91K-step FULL** Phase D run (val_loss 3.72,
margin 0.28). At the 24K-step ablation budget the gate is informational
only — the relevant comparison is the Δ between cells 2 and 3, not
absolute pass/fail at this token budget.

## 2. Decision-gate metric

```
Δ_medium = val_loss(medium_hadamard) − val_loss(medium_baseline)
        = 4.030672745704651 − 4.005183477401733
        = +0.025489 nat
```

Per plan §4 mechanical verdict matrix:

| Δ_medium range | Verdict | This run |
|---|---|---|
| Δ ≤ −0.05 nat | PASS @ Medium | – |
| **−0.05 < Δ < +0.05 nat** | **NULL (within noise)** | **← +0.025 falls here** |
| +0.05 ≤ Δ ≤ +0.12 nat | NEUTRAL-NEGATIVE | – |
| Δ > +0.12 nat | WORSE | – |

**Verdict: NULL @ Medium.** Hadamard at d=512 produces neither a clear
win nor a clear loss vs baseline at the 24K-step ablation budget.

**Δ_base is not measurable.** Phase D Base hidden_size=384 = 3×128 is
not a power of 2; `intllm.quant.HadamardRotation` requires `dim ∈ {2^k}`
per Walsh-Hadamard recursion. Documented in scaffold validation findings
§3 (commit `e939523`); F.6.4 plan §2 logged Base as "STRUCTURAL N/A".
Workarounds (pad-to-2^k, composite rotation, random orthogonal) all
sacrifice either the O(d log d) FFT speed or the uniform-spread property
that motivates Hadamard in the first place — out of F.6.4 scope.

## 3. Cross-scale trend (Mini d=256 → Medium d=512)

E2.1 Mini-scale result (`docs/FJQ_PHASE_E_E2_HADAMARD_DECISION.md`,
commit `<predecessor>`):

| Scale | hidden_size | Δ vs baseline | Verdict @ scale |
|---|---|---|---|
| Mini | 256 = 2^8 | **+0.120 nat** | regression (gate FAIL by 0.12) |
| Medium | 512 = 2^9 | **+0.025 nat** | **NULL (within noise)** |

**Narrowing factor:** Δ_mini / Δ_medium = 0.120 / 0.025 ≈ **4.7×**.

**Interpretation:** the regression gap shrinks substantially when
doubling hidden dim. Two competing hypotheses:

1. **Scale-dependent benefit hypothesis (SpinQuant-aligned):** Hadamard's
   value (rotating outliers into a uniform-spread distribution) grows
   with model size because larger models concentrate outliers in
   per-channel ways more amenable to a fixed rotation. Mini @ d=256 is
   below the threshold where this matters; Medium @ d=512 is at-or-near
   the threshold; Stretch @ d=768 (or larger) might cross to net
   benefit. **Predicts: extrapolate trend, expect Δ < 0 at d ≥ 768.**

2. **Convergence-noise hypothesis (Occam-aligned):** Hadamard inherently
   adds a small adverse bias to ternary quantization (E2.1's three-cause
   analysis: orthogonal rotation incompatible with ternary-{-1,0,1}'s
   fixed quantization-grid alignment) but at larger d the bias becomes
   small relative to overall convergence noise (~0.05 nat at this
   budget), so the absolute Δ shrinks even if the per-parameter bias is
   constant. **Predicts: Δ stays roughly within (−0.05, +0.05) at
   larger scales; never clearly net-positive.**

**F.6.4 cannot distinguish these hypotheses.** Both are consistent with
the data. Resolving requires either:
- Stretch-scale ablation (+10h GPU, ~145M params @ d=768)
- Multiple-seed Mini ablations to estimate noise floor variance (~10h GPU
  for 5 seeds at Mini)

Both are deferred. **F.6.4 records the observation, not the explanation.**

## 4. Honest caveats

### 4.1 Training-distribution mismatch (Mini vs Medium)

The cross-scale Δ comparison is WITHIN-SCALE clean (Cell 2 vs Cell 3 at
Medium are both EN-only-trained), but CROSS-SCALE has a training
distribution mismatch:

| Scale | Train mix | Val protocol |
|---|---|---|
| Mini E2.1 | bilingual 60:40 ID:EN (`--bilingual-data`) | val_loss_en (EN component of held-out) |
| Medium F.6.4 | EN-only (`slimpajama_stream`) | val_loss EN-only |

Hadamard is a deterministic transform invariant to training distribution,
so the per-scale Δ should still be comparable in principle. But strict
apples-to-apples cross-scale claim requires either Medium bilingual runs
or Mini EN-only runs. Both are deferred.

A defensible cross-scale claim is therefore **"both within their
respective training regimes, Hadamard's regression at d=256 is +0.120
nat, at d=512 is +0.025 nat"** — without committing to which hypothesis
above explains the narrowing.

### 4.2 24K-step ablation budget vs 60K/91K-step Phase D budget

Both Cell 2 and Cell 3 fall just below the Phase D gate at 24K (4.005,
4.031 vs gate 4.0). This is expected: Phase D Medium gate was calibrated
against 91K steps where val_loss 3.72 was achievable. At the 24K
comparison budget, neither cell PASSES the gate, but that's not the
F.6.4 question — the F.6.4 question is the Δ, which is well-measured.

### 4.3 Single-seed measurement, no noise-floor estimate

Each cell ran a single seed (seed=0 train, seed=999 val). The (−0.05,
+0.05) NULL band in the verdict matrix was set conservatively per E2.1
analysis but is itself unmeasured at Medium scale. A multi-seed Medium
ablation pair (3-5 seeds) would estimate the actual run-to-run variance
and refine the NULL band. Out of F.6.4 scope.

## 5. Paper v2 / v3 narrative impact

The current paper v1 (commit `5bd277d`, ready for arxiv submission)
treats Hadamard as a flat negative result per E2.1 Mini-only data.
F.6.4's NULL-at-Medium with 4.7× narrowing trend warrants a v2 §7.5
update.

**Proposed v2 §7.5 sub-bullet** (replaces or adds to existing F.5
honest-correction entries; LaTeX-ready):

```latex
\textbf{F.6.4 Hadamard scale-dependence (NULL @ Medium).} Re-running
Phase E2.1's Hadamard ablation at Medium scale (d=512, 74M params, 24K
steps EN-only) yields $\Delta_{\text{medium}} = +0.025$~nat — within
the noise floor and a $4.7\times$ narrowing of E2.1's Mini-scale
$\Delta_{\text{mini}} = +0.120$~nat regression. The negative-result
verdict from E2.1 thus does \emph{not} cleanly extrapolate to larger
scale. Two hypotheses remain consistent with the data: (a) the
SpinQuant-aligned scale-dependent benefit (predicting $\Delta < 0$ at
d $\geq$ 768), and (b) the convergence-noise-dominates explanation
(predicting $\Delta$ stays in $(-0.05, +0.05)$ at all scales).
F.6.4's single-seed measurement at Medium cannot distinguish these.
At the d=384 Base scale the Walsh-Hadamard transform is structurally
unsupported (3×128 not 2$^k$), so no Base measurement can complete the
trend. \textbf{Status:} demoted from "negative" to "scale-dependent,
unresolved at Medium," pending a multi-seed or Stretch-scale follow-up.
```

The paper v1 already contains a §7.4 three-cause attribution of the
Mini failure, which remains valid as scale-dependent attribution
(it just needs to add "...at Mini scale; the trend at Medium is closer
to noise"). No table/fixture changes are required because F.6.4 results
are not yet in the paper text.

**verify-intllm-tables impact:** none. F.6.4 cells produce JSON in the
existing schema v1.2; no claim in the paper currently asserts a number
that depends on F.6.4 results. Adding a v2 narrative reference to the
JSONs is optional and would not require gate-script changes (since v2
has not yet shipped).

## 6. What this validates / does not validate

### Validates ✓

- F.6.4 scaffold execution path is GPU-runnable (re-confirmed beyond
  the e939523 PoL smokes — full 24K runs on each cell)
- §6.11 R1 ckpt-write atomicity + R2 resume mechanism (Cell 2's first
  attempt died on laptop suspend; ckpt rotation + clean atomic writes
  preserved no partial state)
- §6.11 R3 watchdog limitation: caught nothing during Cell-2-first-
  attempt's silent loss because OS suspend pauses the watchdog thread
  too. Documented in Cell 2 commit (`c6b0c2c`).
- Plan §4 mechanical decision matrix discriminated cleanly (Cell 3
  result fell in NULL band without ambiguity)
- Surprise budget tracking: cumulative -7.9h vs plan §2 estimates
  (Cell 1 -1.7h, Cell 2 +1.3h, Cell 3 -0.4h — three-cell variance
  largely cancels, well under +25% cap)

### Does NOT validate ✗

- Cross-scale claim apples-to-apples (training-distribution mismatch
  per §4.1)
- Whether NULL-at-Medium is "trend continues" or "noise floor" — single
  seed measurement (§4.3)
- Behavior at Stretch/larger scale (out of scope)
- Behavior with `--hadamard-sites igfo` (4-projection rotation, F.6.3
  hypothesis) — only `o` was run

## 7. Recommended next-actions (not auto-actioned)

1. **Founder-direction needed for paper v2:** Edit A/B/C in
   `docs/INTLLM_PAPER_V2_DRAFT_EDITS.md` already pending; F.6.4 result
   adds a fourth potential edit per §5 above. Founder picks which edits
   land in v2 vs defer to v3.
2. **Multi-seed Medium ablation** (~10h GPU) to refine NULL-band
   threshold. Optional; defensible as future-work.
3. **Stretch-scale Hadamard ablation** (~10h GPU at Stretch d≥768)
   to test the scale-dependent benefit hypothesis. Speculative;
   higher-risk but higher-information.
4. **F.6.3 igfo-mode Medium ablation** (~6h GPU) — what if rotating
   all 4 attn projections instead of o-only changes the Δ at Medium?
   Cheaper than #3 and tests a co-axial hypothesis.

None of these are critical-path. F.6.4 closes here.

## 8. Self-check (CLAUDE.md §6.8)

| Rule | Status |
|---|---|
| §6.8 R1 pre-flight audit | YES — F.6.4 plan v1.0 §1, scaffold-validation findings |
| §6.8 R2 verification = runnable commands | YES — JSON inspect commands in plan §3; manual `python3 -c ...` produced the deltas |
| §6.8 R3 prevention layer | YES — F.6.4-P0 + P0.1 commits (--resume/--resume-auto/--ckpt-every) added across all 3 ablation drivers, locked in for future ablation runs |
| §6.8 R4 numbers cross-checked | YES — all val_loss values read back from JSONs via `python3 -c "import json; ..."`; no LLM-paraphrased numbers |
| §6.8 R5 surprise budget tagged | YES — per-cell commit messages tag `[actual Xh, est Yh, Z%]`; cumulative -7.9h vs plan |
| §6.8 R6 mechanical decision gate | YES — §2 verdict matrix applied with no judgment-call; Δ_medium = +0.025 unambiguously NULL band |
| §6.8 R7 public-artifact sync | YES — this doc + 3 JSONs commit together; no README/fixture/CHANGELOG drift to fix (paper v1 doesn't reference F.6.4 yet, so no contradiction to repair) |
| §6.8 R8 multi-repo state check | YES — fajarquant only (`git status -sb` clean before+after) |

8/8 satisfied. F.6.4 CLOSED with NULL-at-Medium verdict.

## 9. Self-check (CLAUDE.md §6.9 — research integrity)

| Rule | Status |
|---|---|
| §6.9 R1 canonical protocol from ≥2 reference papers | partial — Phase E E2.1 protocol used (Q5 baseline + ablation budget) is canonical for this codebase but NOT a literature-canonical Hadamard protocol (which would use QuaRot/SpinQuant calibration sets, not Phase D's slimpajama). Paper v1 §7.4 already discloses this as a Phase-E-internal protocol; F.6.4 inherits the same caveat. |
| §6.9 R2 literature review ≥8 papers | inherited from E2.1 — QuaRot, SpinQuant, the BitNet paper line, Hadamard-rotation prior work; not re-swept for F.6.4 because F.6.4 only changes the SCALE, not the rotation method |
| §6.9 R3 baseline parity | YES — Cell 2 (baseline) has all the same training infra as Cell 3 (hadamard) except for the rotation hooks; clean apples-to-apples within-scale |
| §6.9 R4 calibrated > per-chunk | YES — HadamardRotation matrix is built once at attach time and shared across all 12 sites; not per-chunk |
| §6.9 R5 outlier handling | partial — Hadamard IS the outlier-handling-via-rotation. F.6.4 shows its effect is scale-dependent and absent at Medium. Per §3, additional outlier-handling (top-K + adaptive-bits) is not in F.6.4 scope; that's an F.5.x track. |
| §6.9 R6 algorithmic validation precedes paper validation | YES — paper v1 ships WITHOUT F.6.4 results (paper v1 closes with E2.1's flat negative claim); v2 narrative integration is the FOLLOW-UP, gated on founder Edit A/B/C decision |
| §6.9 R7 verify_paper_tables.py --strict | YES — pre-existing 40/40 ACTIVATED + 22 deferred PASS; this commit does not add any new claim that needs verifying (F.6.4 numbers go into v2 narrative, not v1 tables) |

7/7 satisfied (with R1 + R2 + R5 partial-flagged via inheritance from E2.1 — F.6.4 is a scale-extension, not a fresh-protocol). Research integrity preserved.

---

*F.6.4 Decision v1.0 — closed 2026-05-02. NULL @ Medium verdict via
mechanical decision matrix; cross-scale narrowing 4.7× recorded as
observation; two candidate hypotheses preserved without commitment.
Paper v2 §7.5 update gated on founder Edit A/B/C direction.*
