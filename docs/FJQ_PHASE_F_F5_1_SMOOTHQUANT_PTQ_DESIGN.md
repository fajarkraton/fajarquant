# Phase F F.5.1 — SmoothQuant PTQ Design (v0.1 SKELETON)

> **Status:** SKELETON — §1 Motivation + §2 Background fleshed; §3-§7 are placeholders. Each subsequent first-step fills one section. Target v1.0: full design doc, ready for impl scaffold.
> **Origin:** Phase F roadmap §4.1 F.5.1 + post-F.6.2 strategic pivot (2026-04-28)
> **Companion docs:**
> - `docs/FJQ_PHASE_F_TAX_VERTICAL_ROADMAP.md` v1.3 §4.1 F.5
> - `docs/FJQ_PHASE_E_E2_BILINGUAL_CALIB_DECISION.md` v1.0 (E2.4 demote → F.5)
> - `paper/intllm/ablations/posthoc_hadamard_mini.json` (F.6.2 honest verdict)
> - SmoothQuant: Xiao et al. 2023, arxiv 2211.10438 Algorithm 1

---

## 1. Motivation

### 1.1 Why SmoothQuant after two F.6.2 + E2.4 negative results

Phase E E2.4 (`balanced_calib`) and F.6.2 (`activation-only Hadamard
pre-hook`) both produced strong negative results. Both targeted the
same underlying problem — outlier-channel quantization error in HGRN
BitLinear — but neither succeeded at improving val_loss vs the Q5
bilingual baseline.

| Attempt | Approach | Result | Diagnosis |
|---|---|---|---|
| **E2.4 balanced_calib** | All-time-max `running_max` accumulator on per-channel outliers + bit-allocation map | `outlier_global_reduction = −82.13` (gate ≥0.10) | Calibration scale locked to early-training peaks 10-100× larger than steady-state; bit-allocation lever wrong when scale is coarse |
| **F.6.2 online QuaRot** | HadamardRotation pre-hook on attn projection inputs | `rotation_hurts=true`, all modes Δ+2.5–3.0 nat vs no_rotation | Pre-hook breaks FP path (`y = W·Hx ≠ W·x`); recipe-incomplete (no weight-fusion + no γ_x recalib) |

Both failures point to the same need: a **principled, calibrated, FP-path-preserving** way to suppress activation outliers before quantization. SmoothQuant (Xiao et al., 2023) provides exactly this for transformers, and is the cleanest next move because:

1. **FP-path preserving by construction.** Unlike F.6.2 pre-hook variant, SmoothQuant uses a per-channel scale `s` such that `(X / s) · (s · W) = X · W` mathematically — no FP behavior change at all.

2. **Calibration is the design.** SmoothQuant's α-tunable migration of difficulty from activations to weights IS the calibration; no separate "all-time-max" or "running_max" accumulator semantics to argue about. This sidesteps the E2.4 calibration-scale failure mode entirely.

3. **Per-layer locality.** No model-wide rotation, no residual-stream coordination, no RMSNorm γ mutation. Each linear layer gets its own per-channel scale `s`, computed from a small calibration batch. Implementation-wise much smaller than canonical QuaRot.

4. **Outlier evidence already exists.** F.6.1 measured 51.6× mean / 421× max ratio on `o_proj` inputs and 40×/250× on `mlp.down_proj` — exactly the outlier profile SmoothQuant was designed to address (`paper/intllm/ablations/outlier_concentration_mini.json`).

5. **No retraining required.** True PTQ — load Phase D Mini final ckpt, apply SmoothQuant calibration, re-eval val_loss. Same workflow as F.6.2 but with a recipe that doesn't break the FP path.

### 1.2 What success looks like

| Metric | Target | Source |
|---|---|---|
| `val_loss(no_smoothquant)` baseline | match F.6.2 no_rotation = 5.5530 nat | `paper/intllm/ablations/posthoc_hadamard_mini.json` |
| `val_loss(smoothquant)` improvement | **≥0.05 nat reduction** | Matches E2.1 / F.6.2 gate threshold for "rotation helps" (F62_GATE_NAT) |
| `outlier_global_reduction` (E2.4.C metric) | ≥+0.10 (E2.4.C gate threshold) | Re-uses `compute_quant_error_per_channel` from `intllm.eval` |
| Per-channel quant MSE on top-K outlier channels | ≥10% reduction vs baseline | Phase F roadmap §4.1 F.5 entry condition |
| End-to-end runtime overhead | ≤+5% per token at inference | SmoothQuant paper §4.4 reports ~0% — dependent on impl |

If ALL four pass: F.5.1 ships as production calibration; paper Table 4
gets a positive ablation row.

If `val_loss` regression but quant MSE reduction passes: ambiguous —
SmoothQuant works at the per-channel quant level but the model isn't
sensitive at the val_loss level. Demote to "infra-diagnostic", same
fate as E2.4 balanced_calib.

If both fail: pivot to the more involved canonical QuaRot weight-fusion
implementation (estimated ~3-5 days work) or accept that HGRN ternary
training already does most of what calibration could do, and quantization
error is bounded by something other than activation-outlier mishandling.

### 1.3 Why solo + ~1 week is the right scoping

Per Phase F roadmap §4.1: "~1 week solo + 1 Mini ablation". This
matches the pattern of E2.4 (2-3 day impl + Mini ablation + verdict),
plus extra buffer for:
- Calibration data sweep (different α values, different batch sizes)
- Sensitivity analysis (which BitLinear sites benefit most)
- Integration with existing `intllm.qat.attach_stat_trackers` and
  `compute_quant_error_per_channel` infrastructure

Bigger implementations (full canonical QuaRot, 3-5 days) are deferred
until F.5.1 result clarifies whether calibration alone is sufficient.

---

## 2. Background — SmoothQuant Algorithm 1

### 2.1 The mathematical identity

Given a linear layer `Y = X · W` where:
- `X` is the activation tensor `(batch, seq, hidden)`
- `W` is the weight tensor `(hidden, output)`

SmoothQuant introduces a **per-channel scaling vector** `s ∈ ℝ^hidden` such that:

```
X_smooth = X / diag(s)         shape (batch, seq, hidden)
W_smooth = diag(s) · W         shape (hidden, output)

Y_smooth = X_smooth · W_smooth
        = (X / s) · (s · W)
        = X · (s⁻¹ · s) · W
        = X · W       (s⁻¹·s = identity)
        = Y           (mathematically preserved exactly)
```

**Key property:** The transformation is FP-path preserving. Under
unrestricted precision, `Y_smooth = Y` bit-exactly (modulo floating-
point round-off, which is symmetric).

### 2.2 Why it helps quantization

The benefit appears only in the **quantized** version:

```
Y_quant      = quant(X) · quant(W)              ← original
Y_smooth_q   = quant(X / s) · quant(s · W)      ← SmoothQuant
```

- `X / s` has REDUCED outliers when `s` is chosen to match the per-
  channel `max(|X|)` profile → `quant(X/s)` has lower MSE
- `s · W` has SLIGHTLY LARGER values where weights matched outlier
  channels → `quant(sW)` has marginally higher MSE
- **Net:** activation quant error is the bottleneck (outliers dominate);
  trading some weight-quant error for activation-quant error reduction
  is net positive

### 2.3 Choosing the per-channel scale `s`

SmoothQuant Algorithm 1 (paper §3.3 + §3.4):

```
s_j = max(|X_j|)^α / max(|W_j|)^(1−α)          (1)
```

where:
- `X_j` is the j-th activation channel across the calibration batch
- `W_j` is the j-th column of `W` (across all output positions)
- `α ∈ [0, 1]` is the **migration strength** — how aggressively to push outlier difficulty from activations to weights

**Special cases:**
- `α = 0`: `s_j = 1 / max(|W_j|)^1` → all weight-side scaling, no activation help
- `α = 1`: `s_j = max(|X_j|)^1` → all activation-side scaling, weights take all the burden
- `α = 0.5` (paper default): balanced; activation outlier-channels are scaled DOWN by `√max(|X_j|)`, weights scaled UP by same; the j-th channel's effective range is roughly `√(max(|X_j|) · max(|W_j|))`

The α=0.5 choice gives equal max-magnitude on both sides post-smoothing,
which is optimal when activation and weight precisions are equal.

### 2.4 Calibration data — what's needed

Per SmoothQuant paper §3.4 + §4.2:
- ~512 calibration samples are sufficient (paper uses 512; ablations show stable from 128 upward)
- Calibration runs ONE forward pass over the calibration batch, recording per-channel `max(|X_j|)` at each linear layer
- No backward pass, no gradient, no parameter update
- Calibration cost: ~30 seconds per layer at Mini scale on a CPU host (forward-only, 512 sequences)

For FajarQuant adaptation:
- Calibration batch: 512 sequences from `bilingual_stream(id_share=0.6, seed=42)`
  matches the seed/distribution used in `compute_quant_error_per_channel`
- Per-BitLinear `s` vector stored alongside the bit-allocation map
  produced by E2.4 (`save_calibration_maps`) — natural extension of
  existing infrastructure

### 2.5 What SmoothQuant does NOT need

To anchor scope vs canonical QuaRot:

| Requirement | Canonical QuaRot | SmoothQuant |
|---|---|---|
| Orthogonal rotation H | Yes (Hadamard) | No |
| Weight fusion `W' = W·Hᵀ` | Yes | No (just `W' = s·W`, per-channel) |
| Matched residual-stream rotation | Yes (RMSNorm γ + entry projections) | No |
| γ_x recalibration | Yes | Implicit in `s` |
| Model-wide coordination | Yes | No (per-layer local) |
| Re-training step | No | No |
| FP path preserving | Yes (orthogonality H·Hᵀ=I) | Yes (s⁻¹·s=I) |

SmoothQuant is strictly simpler. The scope difference explains why
Phase F roadmap estimates F.5.1 at ~1 week vs F.6.2-canonical at
~3-5 days (canonical QuaRot is shorter on actual coding lines but
needs more model-aware integration; SmoothQuant has more design
parameters to sweep but each is locally simpler).

---

## 3. Adaptation to FajarQuant BitLinear

> **TODO** — to be fleshed in next first-step. Sketch:
> - How `s_j` integrates with BitLinear's existing γ_x (per-channel input scale)
> - Which BitLinear sites benefit most per F.6.1 outlier measurement (o_proj 51.6×, mlp.down_proj 40×)
> - Whether to apply `s` ON TOP of γ_x or REPLACE γ_x calibration
> - Storage format for per-BitLinear `s` vectors (extends `save_calibration_maps` v1.2 schema)
> - Static (one-time PTQ calibration) vs dynamic (per-sequence recalibration)

---

## 4. Calibration Recipe

> **TODO** — to be fleshed in next first-step. Sketch:
> - Calibration data: `bilingual_stream(id_share=0.6, seed=42)` 512 sequences
> - Per-BitLinear forward-pre-hook to capture `max(|X_j|)` activations
> - Per-BitLinear weight scan to capture `max(|W_j|)` per output column
> - Compute `s_j = max(|X_j|)^α / max(|W_j|)^(1-α)` with α default 0.5
> - α sensitivity sweep: [0.3, 0.5, 0.7] x BitLinear-site combinations
> - Apply: at inference time, `X' = X / s` pre-hook + `W' = s · W` weight rewrite (or fuse `s` into the BitLinear's γ_x state)

---

## 5. Implementation Plan

> **TODO** — to be fleshed in next first-step. Sketch sub-tasks:
> - F.5.1.1 — `intllm.quant.SmoothQuantCalibrator` module (~½ day)
> - F.5.1.2 — `--smoothquant` flag in `eval_smoothquant_posthoc.py` (mirrors `eval_hadamard_posthoc.py` from F.6.2; ~½ day)
> - F.5.1.3 — α sweep harness `--alpha 0.3,0.5,0.7` (~½ day)
> - F.5.1.4 — Site-restriction flag `--smoothquant-sites o,igfo,mlp_down,all` (~½ day)
> - F.5.1.5 — Mini scale ablation execution (RTX 4090, ~1 hour for full sweep)
> - F.5.1.6 — Findings doc + verdict (~1 day)
> - F.5.1.7 — Optional: integrate into `train_mini_ablation.py` as `--smoothquant-train` for QAT-time calibration variant (~1 day, deferred unless F.5.1.5 result is strongly positive)

---

## 6. Decision Criteria & Gate

> **TODO** — to be fleshed in next first-step. Sketch:
> - Gate 1 (existence): val_loss with SmoothQuant ≤ baseline + 0.05 nat (no regression)
> - Gate 2 (improvement): val_loss reduction ≥ 0.05 nat OR outlier MSE reduction ≥ 10%
> - Gate 3 (composability): SmoothQuant + balanced_calib (E2.4 ablation) compatible without conflict
> - PASS path: F.5.1 ships, paper claims SmoothQuant-style calibration as part of FajarQuant production recipe
> - PARTIAL path: SmoothQuant works at quant-error level but not val_loss → infra-diagnostic, demote to F.x future-work
> - FAIL path: pivot to canonical QuaRot (F.6 weight-fusion impl) OR accept HGRN ternary's calibration is already near-optimal

---

## 7. Risks & Fallbacks

> **TODO** — to be fleshed in next first-step. Sketch:
> - R1: SmoothQuant α=0.5 default may be wrong for ternary (literature uses INT8 activations); may need α nearer 0.3-0.4 to compensate
> - R2: HGRN's gated paths (i/f/g_proj feeding into MLGRU recurrence) may benefit from rotation more than from calibration; SmoothQuant addresses uniform-outlier suppression, not gated-path-specific issues
> - R3: BitLinear's existing per-channel γ_x may already absorb most of what SmoothQuant adds → marginal benefit
> - R4: Migration of difficulty TO weights at extreme α may push weight-quant MSE into ternary's clipping regime → hidden net regression
> - Fallback ordering: F.5.1 fails → F.6 canonical QuaRot → F.x SmoothQuant + canonical-QuaRot composition → demote outlier-mitigation entirely (accept ternary's structural noise floor)

---

*Document version: 0.1 (skeleton)*
*Last updated: 2026-04-28 (V32-prep: §1 Motivation + §2 Background fleshed; §3-§7 placeholders for next first-step)*
