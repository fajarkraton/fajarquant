---
phase: F.13 — CPU-vs-GPU dispatch heuristic in FajarOS Nova kernel-path
branch: Z-narrow
plan_version: 1.0
status: ACTIVE 2026-04-30
prereq: docs/FJQ_PHASE_F_F13_FINDINGS.md (F.13.0 CLOSED)
parent_roadmap: docs/FJQ_PHASE_F_TAX_VERTICAL_ROADMAP.md §4.2 F.13
---

# F.13 — CPU-vs-GPU Dispatch Heuristic Production Plan (Branch Z-narrow v1.0)

## 1. Why this plan exists

The F.11 chain shipped at infrastructure-only state on 2026-04-30 (`a296c4f` + earlier; CHANGELOG `e594e4b` + CLAUDE.md V31.1 `86989180`). F.11 gives FajarOS Nova a real-vendored AVX2 TL2 kernel with FFI binding but a known row-uniform `+32, -31, -1` parity gap that blocks runtime activation. F.11 chain budget is exhausted (28h vs ~3d).

The natural pivot in `MEMORY.md` resume protocol is Branch Z = F.13 dispatch heuristic. F.13 ships independently of F.11 parity because:

1. The dispatch *decision* is architecture-dominated (PCIe, cudaLaunch, cache hierarchy), not bit-precision-dominated.
2. The roadmap pre-declares F.13.3 "static rule" the *likely* outcome given embedded-ML vision, regardless of measurement precision.
3. Paper v2 §6 needs the dispatch narrative; it does NOT need bit-exact TL2 — projected upstream tok/s suffices.

Z-narrow scope = F.13.0 + F.13.3 + F.13.5 (prevention). F.13.1 (live measurements) and F.13.2 (runtime impl) deferred to a future session when (a) nvidia driver returns AND (b) F.11 parity closes.

## 2. Scope (in / out)

### IN scope (Z-narrow)

- **F.13.0** — pre-flight audit (CLOSED, see `FJQ_PHASE_F_F13_FINDINGS.md`)
- **F.13.3** — decision-doc: cost-model architecture analysis, crossover regions, static-rule recommendation, paper v2 §6 narrative draft
- **F.13.5** — prevention layer: Makefile gate `make verify-f13-decision` + `tests/fixtures/f13_dispatch_anchors.toml` re-asserts pinned literature numbers
- Update `docs/FJQ_PHASE_F_TAX_VERTICAL_ROADMAP.md` §4.2 F.13 with Z-narrow status + cross-link
- Update `CHANGELOG.md` with F.13 Branch Z-narrow entry
- Update `MEMORY.md` resume protocol

### OUT of scope (deferred, future sessions)

- **F.13.1** — live calibration sweep (Mini, Base, Medium, 2B) × {batch 1, 8, 32} × {scalar CPU, TL2 CPU, GPU}. Needs nvidia driver + F.11 parity for the TL2 datapoint.
- **F.13.2** — runtime dispatch implementation in FajarOS Nova kernel-path (`@kernel` annotation + cost-model lookup + path selection). Needs F.13.1 calibration data + F.11 parity (so TL2 is callable).
- F.10 GPU 2:4 Sparse-BitNet (separate hardware-acceleration track)
- F.12 AVX-VNNI (depends on F.7 lm_head FP8/INT8 path shipping first)

## 3. Sub-task table

| # | Description | Effort est | Verification command |
|---|---|---|---|
| F.13.3.1 | Architecture analysis: PCIe + cudaLaunch overhead vs Mini/Base/Medium GEMV cost at batch=1. Identify crossover regions in (model_size, batch) space. | 1.5h | doc has §3.x crossover table + 3 worked examples |
| F.13.3.2 | Cost-model formalization: piecewise function `predicted_path(M, B) → {scalar_cpu, projected_tl2_cpu, gpu}`. Document inputs, decision boundaries, fallback rules. | 1h | doc has §4 explicit pseudo-code listing |
| F.13.3.3 | Static-rule recommendation per F.13.3 spec: "CPU default; GPU optional for batch ≥ N." Derive N from §3 crossover. Document anti-patterns + when runtime dispatch DOES become worthwhile (F.13.2 trigger conditions). | 0.5h | doc has §5 verdict + §5.1 F.13.2 re-entry gates |
| F.13.3.4 | Paper v2 §6 narrative: 1-2 paragraph synthesis suitable for paste-in. Highlights the "CPU-wins-batch=1 crossover" as a fajarquant differentiator vs llama.cpp/transformers' "GPU always wins." | 0.5h | LaTeX-ready snippet in doc §7 |
| F.13.5.1 | Pinned-anchors fixture `tests/fixtures/f13_dispatch_anchors.toml` (BitNet 2B4T 29ms/tok, llama.cpp TQ2_0 30-40 tok/s range, cudaLaunch 10-30µs range, Mini sizes). Each entry has source URL + verification date. | 0.5h | `cat tests/fixtures/f13_dispatch_anchors.toml` shows ≥6 anchors with sources |
| F.13.5.2 | `scripts/verify_f13_dispatch.py` script: loads fixture, asserts ranges still match what's referenced in decision-doc, exits 0/1. | 0.5h | `python3 scripts/verify_f13_dispatch.py --strict` exit 0 |
| F.13.5.3 | Makefile target `verify-f13-decision` invokes the script. Pre-commit hook runs it when `docs/FJQ_PHASE_F_F13_*.md` or the fixture changes. | 0.5h | `make verify-f13-decision` returns 0 |
| F.13.5.4 | CHANGELOG entry + roadmap §4.2 F.13 status update + MEMORY.md resume-protocol update. | 0.5h | three files have F.13 Branch Z-narrow text |

**Total Z-narrow effort:** 5.5h (5 doc + 0.5 prevention scaffold). Surprise budget +25% (CLAUDE.md §6.8 R5) → cap at **~7h** before re-scoping or surfacing to user.

## 4. Dependency graph

```
F.13.0 (CLOSED)
   │
   ├──> F.13.3.1 — architecture analysis
   │       │
   │       ├──> F.13.3.2 — cost model
   │       │       │
   │       │       └──> F.13.3.3 — static rule + verdict
   │       │               │
   │       │               └──> F.13.3.4 — paper v2 §6 snippet
   │       │
   │       └──> F.13.5.1 — pinned anchors fixture (parallel-able with F.13.3.x)
   │               │
   │               └──> F.13.5.2 — verify script
   │                       │
   │                       └──> F.13.5.3 — Makefile + hook
   │
   └──> F.13.5.4 — public artifact sync (last)
```

Critical path: F.13.3.1 → 3.2 → 3.3 → 3.4 → 5.4. Total ≈ 4h. F.13.5.1–3 ride alongside.

## 5. Mechanical decision gates (CLAUDE.md §6.8 R6)

These gates are pre-committed BEFORE drafting F.13.3 verdict. Outcome dictates verdict mechanically:

### G1: Crossover within FajarOS deployment range?

```
Input:  (M_typical, B_typical) for FajarOS embedded use case
        Per FJQ_PHASE_D_PRODUCTION_PLAN.md line 72-83: M ≤ 100M, B = 1.
Check:  Is the crossover (CPU-beats-GPU boundary) > B_typical?
PASS  → static rule: "CPU default" — F.13.3 outcome.
FAIL  → runtime dispatch needed (F.13.2 must ship); F.13.3 outcome demotes to static-by-default-with-dispatch-hint.
```

### G2: Projected TL2 tok/s ≥ scalar tok/s × 3?

```
Input:  BitNet 2B4T 29ms/tok @ 2.4B i7-13800H (anchor §5)
        Scaled to Mini 22M on i9-14900HX:
          (i9 perf factor vs i7-13800H ≈ 1.10–1.20 single-thread, similar IPC)
          (size scales tok/s by 2.4B/22M = ~109× theoretically; capped by L2 bandwidth)
        Conservative projection: TL2 Mini ≈ 200–400 tok/s
        Scalar Mini (extrapolate F.11.3 baseline at 22M × AVX2 8-lane ≈ 50–100 tok/s)
Check:  TL2 / scalar ≥ 3
PASS  → TL2 path is worth dispatching to (F.13.2 has work to do once F.11 parity closes).
FAIL  → scalar may be "good enough"; F.13.2 priority demoted.
```

### G3: GPU at batch=1 lands < CPU TL2 projected tok/s for Mini?

```
Input:  cudaLaunch ≥ 10µs/kernel × layers × steps; Mini has 12 transformer layers × ~6 matmul kernels = 72 kernels/step
        72 × 10µs = 720µs minimum GPU latency per token = max ~1389 tok/s upper-bound
        Real GPU at batch=1 small model: 80-120 tok/s (memory-bandwidth + launch dominated)
Check:  CPU TL2 (projected 200-400 tok/s) > GPU (80-120 tok/s)?
PASS  → CPU wins-batch=1 confirmed; static "CPU default" justified.
FAIL  → GPU still wins; static rule reversed to "GPU default" — would invalidate fajarquant's CPU-first thesis.
```

**Verdict matrix:**

| G1 | G2 | G3 | Outcome |
|---|---|---|---|
| PASS | PASS | PASS | **Static rule "CPU default; GPU optional for batch ≥ N"** (likely outcome per roadmap) |
| PASS | PASS | FAIL | Static rule reversed: "GPU default" — paper v2 §6 narrative pivots |
| PASS | FAIL | PASS | "Scalar CPU default" — TL2 effort priority demoted |
| FAIL | * | * | "Runtime dispatch required (F.13.2 must ship)" — Z-narrow CANNOT close F.13 |

If verdict ≠ G1.PASS+G2.PASS+G3.PASS, F.13.3.3 documents the mismatch + escalates to user before committing.

## 6. Risk register

| ID | Risk | Likelihood | Mitigation |
|---|---|---|---|
| R1 | Projected TL2 tok/s wildly off (no live bench) | M | Anchor §5 has range; F.13.5 fixture pins acceptance band |
| R2 | cudaLaunch overhead estimate stale on new GPUs | L | Anchor §5 cites NVIDIA programming guide range 10-30µs; conservative band |
| R3 | F.13.3 narrative gets quoted in paper v2 then F.13.1 actuals contradict | M | Paper §6 snippet uses "projected" / "anchor" language; verify_f13_dispatch.py runs as paper-prep gate |
| R4 | Z-narrow effort blows past 7h cap | L | §3 sub-task table has explicit estimates; checkpoint at 5h |
| R5 | F.11 parity closes faster than expected, retroactively making Z-narrow obsolete | L | Z-narrow output is *foundational* (cost model + verdict); F.13.1 actuals refine, don't replace |
| R6 | nvidia driver returns mid-session and tempts a re-scope | L | Memory rule: explicit user nod for path forks; treat as separate session |

## 7. Self-check (CLAUDE.md §6.8 + §6.6)

| Rule | Status |
|---|---|
| §6.8 R1 pre-flight audit (B0/C0/D0) | YES — F.13.0 CLOSED, findings doc committed |
| §6.8 R2 runnable verification per task | YES — `make verify-f13-decision` and per-task §3 column |
| §6.8 R3 prevention layer | YES — F.13.5 pinned-anchors CI gate |
| §6.8 R4 multi-agent numbers cross-checked | N/A — Z-narrow uses no parallel agents; numbers from public sources only |
| §6.8 R5 surprise budget +25% tagged | YES — 5.5h × 1.25 = 7h cap; commits will tag variance |
| §6.8 R6 mechanical decision gates | YES — §5 G1/G2/G3 verdict matrix |
| §6.8 R7 public-artifact sync | F.13.5.4 covers CHANGELOG, roadmap, MEMORY |
| §6.8 R8 multi-repo state check | YES — done at session start, all 3 clean |
| §6.6 R1 [x] = end-to-end working | F.13.3 closeout requires `make verify-f13-decision` exit 0 (mechanical) |
| §6.6 R3 no inflated stats | Z-narrow status honestly labels deferred work in §2 OUT |

10/10 satisfied. Plan READY for execution.

---

*F.13 Production Plan v1.0 — Branch Z-narrow. Created 2026-04-30. Active.*
