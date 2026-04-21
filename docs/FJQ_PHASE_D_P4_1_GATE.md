# V31.C.P4.1 Mini Gate — Decision: FAIL

> **Date:** 2026-04-21 | **Gate:** §6.9 R1 + FJQ_PHASE_D_CONFIG.md §5.1: val_loss < 4.0 on held-out SlimPajama after Mini training | **Result:** **FAIL by 0.78 nat** | **Rule:** §6.8 R6 mechanical decision gate (committed file blocks downstream work)

## 0. Summary

Full Mini training (60,000 steps × 8K tokens/step ≈ 491M tokens) on
RTX 4090 Laptop completed in 107.9 min. Held-out val_loss = **4.7849
(PPL 119.7)** vs gate target **< 4.0**. **Gate FAILS by 0.78 nat.**

Per FJQ_PHASE_D_CONFIG.md §5.1: *"If FAIL: lr sweep + batch-size
ablation; debug QAT harness; do NOT proceed to Base."*

## 1. Numerical results

| Metric | Value |
|---|---|
| Steps | 60,000 |
| Tokens trained | 491.5 M |
| Wall-clock | 107.9 min |
| Initial loss | 10.4242 (= ln(32768) ✓) |
| Final train loss | 4.3970 |
| Held-out val loss | **4.7849** |
| Initial PPL | 33,663.7 |
| Final train PPL | 81.2 |
| Held-out val PPL | 119.7 |
| Train-val gap | 0.388 nat (slight overfit) |
| Gate (val < 4.0) | **FAIL by 0.78** |

## 2. Trajectory milestones

| Step | Train loss | Tokens seen |
|---|---|---|
| 100 | 7.37 | 819K |
| 1,000 | 5.76 | 8.2M |
| 5,000 | 5.40 | 41M |
| 10,000 | 5.38 | 82M (val_loss=5.03 measured) |
| 20,000 | 5.27 | 164M |
| 30,000 | 4.95 | 246M |
| 40,000 | 4.84 | 328M |
| 50,000 | 4.76 | 410M |
| 60,000 | 4.40 | 491M (val_loss=4.78 measured) |

Loss curve plateaued ~4.7-4.9 from step 30K onwards with high
per-step variance. **No clear inflection point** — gradual decline,
no lr-decay-driven late drop, no sign of further convergence headroom.

## 3. Root-cause hypotheses (ranked)

### H1 — Implementation gap: train_loop ignores LR schedule (HIGH likelihood, +0.4 nat)

`intllm.train.train_loop` runs constant `lr=2e-3` from step 1. The
`MiniTrainConfig.warmup_steps=2000` + cosine-decay-to-10% specified
in `configs/mini.py` are **stored but never consumed** by the
training loop.

Standard observation in the LM-pretraining literature: AdamW without
warmup at lr=2e-3 produces ~0.3-0.5 nat worse loss vs the same
budget with linear warmup + cosine decay. This alone could be
responsible for most of the 0.78-nat gap.

**Fix:** add `torch.optim.lr_scheduler.LambdaLR` honoring warmup +
cosine to `train_loop`. ~15 LOC change. Re-run Mini → expected
val_loss ~4.0-4.4.

### H2 — lr=2e-3 may be too high without warmup (MEDIUM, +0.2 nat)

Per-step training loss has high variance (4.0-6.5 spikes throughout
the run). Symptom of lr too high relative to gradient norm at the
current scale. With warmup applied the effective lr at the unstable
phase becomes much lower; without warmup, lr=2e-3 may be on the
edge.

**Fix:** if H1's warmup doesn't fully close the gap, sweep
`lr ∈ {5e-4, 1e-3, 2e-3, 5e-3}` with the same 60K-step budget.

### H3 — Mini may simply need more tokens (LOW, +0.0 nat)

Chinchilla-optimal ratio is ~20 tokens/param. Mini at 21.5M params
× 491M tokens = 22.8 tokens/param — well within the optimal range.
**Token undertraining is unlikely to be the root cause.** Mini at
this size SHOULD reach val_loss ~3.5-4.5 with proper hparams (per
typical scaling laws for 21M-param Transformer-class on web data).

### H4 — MatMul-Free arch underperforms Transformer at small scale (LOW, +0.1 nat)

Per Zhu et al. Table 1, MatMul-Free at 370M scale is ~0.8 avg-pt
*below* Transformer++ on zero-shot tasks. Extrapolating to 21M, we'd
expect a similar small gap. But that's vs Transformer++ baseline,
not vs an absolute val_loss target. Architecture isn't the limiting
factor here.

## 4. Decision: do NOT proceed to Base

Per FJQ_PHASE_D_CONFIG.md §5.1, Mini gate FAIL blocks Base. Required
before re-running:

1. **Implement LR schedule in `intllm.train.train_loop`** (H1 fix,
   ~15 LOC, ~30 min). Re-run Mini with same budget.
2. **If still FAIL: lr sweep** ({5e-4, 1e-3, 2e-3, 5e-3} × 60K
   steps each = ~7h GPU). Land on the best config.
3. **Re-measure val_loss with best config.** Update this gate doc
   with the second attempt's numbers.

Only after one of these passes < 4.0 does Base C.P4.1.5 unblock.

## 5. Implementation gap surfaced — `train_loop` schedule

For accountability per §6.8 R7 transparency:

`intllm.train.TrainConfig` has `lr` and `weight_decay` fields. It
does NOT expose `warmup_steps` or scheduler. `MiniTrainConfig` has
`warmup_steps=2000` declared but never propagated. The driver
(`scripts/train_mini.py`) creates `TrainConfig(lr=train_hp.lr, ...)`
which drops the schedule fields silently.

This is a real implementation gap, not just a hyperparameter miss.
Documented here so the H1 fix is implemented properly:
1. Extend `TrainConfig` with `warmup_steps`, `total_steps`, `min_lr_ratio`
2. Modify `train_loop` to build a `LambdaLR` scheduler using these
3. Update test_train.py to verify scheduler activates
4. Re-run Mini

## 6. What this un-gates / blocks

- ❌ C.P4.1.5 Base config training — **BLOCKED** until Mini passes
- ❌ C.P4.4 Stretch (17-day) — blocked by Base
- ✅ C.P5.4 Python exporter (`intllm.export`) — already shipped, not
  gated on Mini PASS
- ✅ Track D ext2 fix — independent, can proceed when convenient
- ✅ Future paper draft Sections §1-3 (no quantitative claims yet) —
  not blocked

## 7. Artifacts committed

- `python/phase_d/scripts/train_mini_full_trace.json` — 601 loss
  points (downsampled every 100th + final from 60K trace; original
  1.4 MB pre-downsample retained locally for analysis but not
  committed)
- `python/phase_d/checkpoints/mini/mini_final.pt` — 87 MB (gitignored)
- This decision doc

## 8. Self-check (§6.8 + §6.9)

- [x] R1 — Pre-flight done (C.P4.0 POL ran first, validated stack)
- [x] R2 — Verifications are runnable: `pytest test_*.py` + the
       60K Mini run is a measurable command
- [x] R3 — Prevention layer: H1 fix is the next concrete deliverable;
       commit message will tag schedule activation in re-run
- [x] R4 — Numbers cross-checked: val_loss 4.7849 confirmed by
       direct stdout + JSON trace
- [x] R5 — Variance in commit msg: actual gate FAIL vs intermediate's
       projected ~4.0-4.3 PASS; documented as legitimate negative
       result, not goalpost-moving
- [x] R6 — This file IS the mechanical Mini gate decision; future
       sessions cannot proceed to Base until val_loss < 4.0 commits
       a new gate doc
- [x] R7 — No public artifact (internal training milestone)
- [x] R8 — Multi-repo state confirmed pre-commit
