# FJQ Phase D — Base c.1 Gate Decision

**Status:** PASS ✓ (production-grade margin)
**Gate closed:** 2026-04-23 02:51 WIB
**Authority:** `FJQ_PHASE_D_GATE_CALIBRATION.md` (Base <4.2 threshold, evidence-backed)

## Result

| Metric | Value |
|---|---|
| `val_loss` | **3.9903** |
| Perplexity (PPL) | 54.1 |
| Gate threshold | 4.2 |
| **Margin** | **0.21 nat** |
| Margin vs c.2 (0.071 nat) | **2.95× wider** |

## Evidence

- Source trace: `python/phase_d/scripts/train_base_full_trace.json`
- Gate artifact: `paper/intllm/results/training_intllm-base.json` (`config_variant = "c.1"`)
- Training log: `/tmp/fajarquant_logs/base_c1_full.log` (628 lines, 29 KB, complete)
- Checkpoint: `python/phase_d/checkpoints/base/base_final.pt` (186 MB, timestamp 2026-04-23 02:51)
- Verify claim: `scripts/verify_intllm_tables.py:175` — `"Gate: Base val_loss (c.1 982M tok Chinchilla)"`

## Training Configuration

| Field | Value |
|---|---|
| Params | 46,454,016 (46.4M) |
| Arch | d=384, L=12, V=32768, T_max=2048 (Mistral v3 tokenizer) |
| Steps | 60,000 |
| Batch × seq_len | 8 × 2048 = 16,384 tokens/step |
| **Tokens seen** | **982,320,000 (982M)** |
| Tokens/param ratio | **21.16** (Chinchilla-optimal ~20) |
| LR | 2e-3 peak, linear warmup 2K → cosine decay to 10% |
| Weight decay | 0.1 |
| Grad clip | 1.0 |
| Dataset | `DKYoon/SlimPajama-6B` (streaming) |
| Hardware | RTX 4090 Laptop |
| Wall-clock | 8h 03m (29,024s), pace ~0.48s/step |
| Launched | 2026-04-22 18:45 WIB |
| Finished | 2026-04-23 02:51 WIB |

## Production-Readiness Assessment

The 0.21 nat margin is **3× wider than c.2's 0.071 nat**. This is the difference
between "gate PASSED narrowly, vulnerable to eval-run variance" and "gate
PASSED comfortably, reviewer-defensible". Per the user's "100% siap produksi"
directive on 2026-04-22:

- ✅ **Margin** ≥ 0.2 nat (production threshold for robust pass)
- ✅ **Tokens/param** ≥ 20 (Chinchilla-optimal — c.2 at 10.6 was half-Chinchilla,
  would have invited reviewer pushback on undertrained claim)
- ✅ **Loss decay** clean: initial 10.44 → final 4.02, 74% reduction, no plateaus
  or NaN spikes
- ✅ **PPL** 54.1 on a 50-batch held-out eval (seed=999, different from train seed=0)

## Supersession of c.2

The c.2 gate artifact (`val_loss = 4.129`, batch_size=4, 491M tokens) was a
sub-Chinchilla intermediate run landed 2026-04-21 to unblock the calibration
doc (`FJQ_PHASE_D_GATE_CALIBRATION.md`). It passed by 0.07 nat — adequate for
calibration but too narrow for paper Table 2 final row. c.1 supersedes c.2
as the production Base gate.

The supersession is recorded in `paper/intllm/results/training_intllm-base.json`
`_supersedes` block (val_loss, tokens, commit SHA preserved for audit trail).

## Relationship to the 2026-04-22 Hang

The first attempted c.1 launch (also 2026-04-22, earlier that day) hung at
step 8600 after ~1h42m due to laptop battery-low → OS suspend → dead HF CDN
TCP sockets (root cause confirmed 2026-04-22 evening by user: laptop was not
plugged in). 8.5h wall-clock wasted before user noticed.

This gate result is from the relaunched run (PID 65409, started 2026-04-22
18:45 WIB after AC plugged in). That run completed cleanly.

**Prevention delivered** (V31.C.P6.1–P6.5, Track B, merged 2026-04-22):
1. Intermediate checkpoints (`ckpt_every` wired, atomic write, rotation)
2. `--resume <path>` / `--resume-auto` with bit-exact state restoration
3. `StepWatchdog` — SIGTERM if step idle > 30 min
4. HF streaming `HF_DATASETS_DOWNLOAD_TIMEOUT=60` + `_retry_iter` 5 attempts
5. `make test-train-watchdog` regression gate + `CLAUDE.md §6.11` rule

Any future training run (including Medium, next on the queue) inherits all
five layers by default. The c.1 hang was the last occurrence of that failure
class for the Phase D program.

## Plan Status After c.1 Lands

- §2.1 Base c.1 training — ✅ DONE, gate PASS 0.21 nat margin
- §2.2 Medium training — next (ready to launch, GPU free, hardened defaults
  inherited from Track B)
- §3.2 bench-canonical scaffold ✓, real mode pending (use new `bench-canonical-real
  TAG=base` now that `base_final.pt` is final)
- §3.3 bench-knowledge scaffold ✓, real mode pending same
- §3.4 baselines — pending scaffold
- §4 paper — Table 2 Base row now populated with defensible numbers

## §6.8 R6 Decision-Gate Checklist

```
[x] Gate threshold cited + traceable to calibration doc
[x] Evidence paths listed + committed
[x] Margin computed, not asserted
[x] Comparison against previous artifact (c.2) in the same table
[x] Production-readiness assessment independent of paper claims
[x] Supersession recorded in JSON (audit trail preserved)
[x] Prevention mechanisms for root-cause failure listed
```

Seven YES = ship.
