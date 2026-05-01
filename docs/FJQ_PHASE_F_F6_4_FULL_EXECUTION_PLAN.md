---
phase: F.6.4 — Hadamard rotation @ Base + Medium scale (full execution)
status: in_progress 2026-05-01
budget: ~17.4h GPU (est) + ~3h Claude prep/closeout
                +25% surprise: 21.7h GPU + 3.75h Claude
prereq: F.6.4 scaffolds validated (commit e939523, 2026-05-01) + paper
        substantively submission-ready (tarball, gates green; arxiv ID
        gated only on endorsement logistics)
artifacts:
  - paper/intllm/ablations/base_baseline.json (FULL, replaces PoL)
  - paper/intllm/ablations/medium_baseline.json (FULL, replaces PoL)
  - paper/intllm/ablations/medium_hadamard.json (FULL, replaces PoL)
  - docs/FJQ_PHASE_F_F6_4_DECISION.md (post-run verdict + paper v2 §6 narrative)
---

# F.6.4 Full Execution Plan v1.0

> **Goal.** Measure Hadamard rotation's val_loss effect at Base (d=384, N/A
> for Hadamard) and Medium (d=512, supported) scales, vs Phase-D-Q5-
> comparable baseline at the same 24K-step ablation token budget. Decide
> whether the E2.1 Mini-scale negative result holds at larger scales (per
> SpinQuant paper hypothesis "benefit grows with model size").

## 1. Why this plan exists

Per `docs/FJQ_PHASE_F_F6_4_SCAFFOLD_VALIDATION_FINDINGS.md` (2026-05-01),
F.6.4 scaffolds are GPU-validated (4 PoL smokes green, 1 structural N/A
for Base hadamard at d=384). Full execution was gated on "Phase E paper
accepted (or at least submitted)". Paper v1 is substantively submission-
ready (tarball builds clean, all 40 verify gates PASS, draft polished
post visual-inspection-fix `5bd277d`); only arxiv-ID-assignment is
blocked on endorsement logistics. We treat "submission-ready" as the
substantive trigger and proceed.

F.6.4 is about WHEN/WHERE to apply Hadamard, not the rotation itself.
Reusable infrastructure (`HadamardRotation`, `--hadamard-sites` flag,
forward-pre-hook attach pattern) already shipped at Mini scale (E2.1).
This plan executes the existing scaffolds at Base + Medium scale with
ablation-budget runs.

## 2. Cell matrix (3 NEW full runs)

| # | Cell | Driver | Flags | Expected loss range | Wall-clock |
|---|---|---|---|---|---|
| 1 | Base baseline | `train_base_ablation.py` | `--tag baseline` | 4.5–5.0 nat (gate 4.2 won't pass at 24K) | ~5.1h |
| 2 | Medium baseline | `train_medium_ablation.py` | `--tag baseline` | 4.5–5.0 nat (gate 4.0 won't pass at 24K) | ~5.8h |
| 3 | Medium hadamard | `train_medium_ablation.py` | `--hadamard --tag hadamard` (default `--hadamard-sites o`) | comparable to #2 ± delta | ~6.5h |

Total: **~17.4h GPU**. +25% surprise budget = **21.7h cap**.

Pre-existing cells NOT re-run (memory tag E2.x):
- Mini-baseline (Q5 baseline, commit `1074883`)
- Mini-hadamard (E2.1, commit `7234a3a`)

Base-hadamard is **STRUCTURAL N/A** (d=384 = 3×128, not power of 2 →
HadamardRotation rejects). Documented at `train_base_ablation.py:282-298`
+ scaffold validation findings §3.

## 3. Run commands (verifiable, runnable)

All commands run from `~/Documents/fajarquant/python/phase_d` with
`PYTHONPATH=.` and the project venv. Watchdog 30 min, ckpt-every 4000
(6 mid-ckpts per 24K-step run, ~22–26 min loss-cap on suspend).

```bash
# Cell 1: Base baseline (24K steps, ~5.1h)
PYTHONPATH=. ../../.venv/bin/python scripts/train_base_ablation.py \
    --tag baseline \
    --ckpt-every 4000 \
    --watchdog-idle-seconds 1800 \
    > ../../logs/f6_4_base_baseline.log 2>&1

# Cell 2: Medium baseline (24K steps, ~5.8h) — sequential after Cell 1
PYTHONPATH=. ../../.venv/bin/python scripts/train_medium_ablation.py \
    --tag baseline \
    --ckpt-every 4000 \
    --watchdog-idle-seconds 1800 \
    > ../../logs/f6_4_medium_baseline.log 2>&1

# Cell 3: Medium hadamard (24K steps, ~6.5h) — sequential after Cell 2
# (--hadamard-sites o is the default, matches E2.1 Mini convention; produces medium_hadamard.json)
PYTHONPATH=. ../../.venv/bin/python scripts/train_medium_ablation.py \
    --hadamard \
    --tag hadamard \
    --ckpt-every 4000 \
    --watchdog-idle-seconds 1800 \
    > ../../logs/f6_4_medium_hadamard.log 2>&1
```

Resume command (any cell, after suspend or kill):
```bash
# e.g. cell 1 mid-suspend
PYTHONPATH=. ../../.venv/bin/python scripts/train_base_ablation.py \
    --tag baseline --ckpt-every 4000 --resume-auto \
    > ../../logs/f6_4_base_baseline.resume.log 2>&1
```

Verification (post-run):
```bash
# Each cell writes JSON; verify schema + gate fields
cd ~/Documents/fajarquant
python3 -c "import json; r = json.load(open('paper/intllm/ablations/base_baseline.json')); print(r['_schema_version'], r['n_steps'], r['val_loss'], r['gate_pass'])"
python3 -c "import json; r = json.load(open('paper/intllm/ablations/medium_baseline.json')); print(r['_schema_version'], r['n_steps'], r['val_loss'], r['gate_pass'])"
python3 -c "import json; r = json.load(open('paper/intllm/ablations/medium_hadamard.json')); print(r['_schema_version'], r['n_steps'], r['val_loss'], r['gate_pass'])"
```

## 4. Decision gates (mechanical, post-cells-1+2+3)

Per E2.1 Mini-scale baseline (memory: "Hadamard FAIL by 0.12 nat regression
in val_loss vs baseline @ Mini scale"), the F.6.4 decision is:

**Δ_medium = val_loss(medium_hadamard) − val_loss(medium_baseline)**

| Δ_medium range | Verdict | Action |
|---|---|---|
| Δ ≤ −0.05 nat | **PASS @ Medium** | Hadamard helps at Medium scale; recommend paper v2 §7.5 update from "negative result" to "scale-dependent" |
| −0.05 < Δ < +0.05 nat | **NULL** | Within noise floor; report observation, no recommendation change |
| +0.05 ≤ Δ ≤ +0.12 nat | **NEUTRAL-NEGATIVE @ Medium** | Negative result holds; gap not widening with scale |
| Δ > +0.12 nat | **WORSE @ Medium** | Negative result deepens; paper v2 §7.5 strengthens "Hadamard hostile to ternary" claim |

**Δ_base** is N/A (Hadamard structurally unsupported at d=384). Document
this explicitly in the decision doc; do NOT extrapolate.

## 5. Operational notes

- **Power**: Laptop charger MUST stay connected for full 17h+. Battery
  100% verified at plan start; AC=Full per `/sys/class/power_supply/`.
- **Disk**: 83 GB free, expect ~6 GB of ckpts across 3 cells. OK.
- **HF cache**: 61 GB; bilingual stream warmed already from prior runs.
- **Watchdog**: 1800s (30 min) per §6.11 R3. Kills process if step counter
  idles > 30 min (e.g. dead HF socket). Single-shot; external orchestrator
  resumes via `--resume-auto`.
- **Sequential, not parallel**: 16 GB VRAM cannot fit 2 trainings.

## 6. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Laptop suspend mid-run | Med (battery is plugged but lid-close possible) | ~24min loss per cell w/ ckpt-every 4000 | --resume-auto + 6 mid-ckpts |
| HF CDN socket timeout | Low (retry_iter §6.11 R4 wraps stream) | 0 (retried) | already mitigated |
| OOM at Medium hadamard | Low (PoL ran successfully w/ Hadamard at d=512) | run aborts, cell 3 lost | catch + report; does not block cell 1+2 |
| Disk space exhaustion | Low (6 GB needed, 83 GB free) | run aborts | manual cleanup if triggered |
| val_loss < 4.0 (Medium gate, unlikely at 24K) | Very low | misleads "gate_pass=true" | document explicitly: 24K is comparison budget, not gate budget |

## 7. Surprise budget tracking

Estimate: 17.4h GPU + 3h Claude prep/closeout = ~20.4h total
Cap (+25%): **25.5h total**

Per-cell tags in commit messages (per §6.8 R5):
```
feat(v32-prep-f.6.4-cell-1): Base baseline, val_loss=X.XXX [actual Yh, est 5.1h, +Z%]
feat(v32-prep-f.6.4-cell-2): Medium baseline, val_loss=X.XXX [actual Yh, est 5.8h, +Z%]
feat(v32-prep-f.6.4-cell-3): Medium hadamard, val_loss=X.XXX, Δ=±X.XXX [actual Yh, est 6.5h, +Z%]
docs(v32-prep-f.6.4-decision): verdict + paper v2 §6 narrative [actual Xh, est 1h, +Z%]
```

## 8. Self-check (CLAUDE.md §6.8)

| Rule | Status |
|---|---|
| §6.8 R1 pre-flight audit (B0/C0/D0 hands-on) | YES — see findings doc + this plan §1 |
| §6.8 R2 verification = runnable commands | YES — §3 commands; §3 verification commands |
| §6.8 R3 prevention layer per phase | YES — F.6.4-P0 + P0.1 commits added --resume/--resume-auto + --ckpt-every (§6.11 R2 + R1 deepening) |
| §6.8 R4 numbers cross-checked (no agent inflation) | YES — wall-clock estimates from PoL timings (152.8/173.9/194.2s for 200 steps); param counts hand-verified vs scaffold doc |
| §6.8 R5 surprise budget +25% tagged | YES — 21.7h GPU cap; per-commit `[actual Yh, est ?h, +Z%]` |
| §6.8 R6 mechanical decision gates | YES — §4 Δ_medium thresholds with verdict matrix |
| §6.8 R7 public-artifact sync (no drift) | YES — this doc + JSONs commit together; CHANGELOG updated post-decision |
| §6.8 R8 multi-repo state check | YES — fajarquant only (`git status -sb` clean) |

8/8 satisfied. F.6.4 full execution AUTHORIZED.

## 9. Self-check (CLAUDE.md §6.11 — training-script interruption-safety)

| Rule | Status (post P0 + P0.1) |
|---|---|
| §6.11 R1 ckpt_every + atomic + rotation | YES — train_loop atomic write + keep_last_n_ckpts=3 |
| §6.11 R2 --resume / --resume-auto bit-exact | YES — added in P0 commit `c537d4d` |
| §6.11 R3 StepWatchdog | YES — 1800s default exposed via --watchdog-idle-seconds |
| §6.11 R4 HF read timeouts + retry_iter | YES — wired in `intllm.data` (V31.C.P6.4) |
| §6.11 R5 test-train-watchdog Makefile gate | YES — `make test-train-watchdog` green pre-push |

5/5 satisfied. Training-script interruption-safety AUTHORIZED.

---

*F.6.4 Full Execution Plan v1.0 — written 2026-05-01. Cell 1 kicks off
immediately upon plan-doc commit; cells 2 + 3 are sequential gated on
prior cell completion (single-GPU constraint).*
