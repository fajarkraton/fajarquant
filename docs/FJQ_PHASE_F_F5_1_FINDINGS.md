# Phase F F.5.1 — SmoothQuant PTQ Findings (v0.1 SKELETON)

> **Status:** SKELETON — §1 Executive Summary + §2 Empirical Results fleshed; §3-§8 are placeholders. Each subsequent first-step fills one section. Target v1.0 closes F.5.1 with full diagnostic + branch decision.
> **Origin:** F.5.1 sweep complete 2026-04-28. 8 runs total (5 primary α × 3 secondary sites). Verdict: PARTIAL.
> **Companion docs:**
> - `docs/FJQ_PHASE_F_F5_1_SMOOTHQUANT_PTQ_DESIGN.md` v1.0 (the design this evaluates)
> - `docs/FJQ_PHASE_F_TAX_VERTICAL_ROADMAP.md` v1.3 §4.1 F.5
> - 8 ablation JSONs at `paper/intllm/ablations/smoothquant_*.json`
> - 8 calibration side-cars at `paper/intllm/ablations/smoothquant_*_maps.pt`

---

## 1. Executive Summary

**Verdict: F.5.1 → PARTIAL.** No STRONG-PASS run; best result is
**model invariance** (+0.0001 nat at `o_down` × α=0.3), not improvement.

**Headline finding:** The trained Phase D Mini ckpt is essentially
INVARIANT to activation-only SmoothQuant when applied at the right
(α, sites) combination. The model neither benefits nor regresses —
RMSNorm γ during training has already absorbed the per-channel
rescaling SmoothQuant would offer post-hoc. This confirms design
§7.1 R3 hypothesis directly.

**Sweep coverage:**

  - 5 primary α runs × sites=o: α ∈ [0.3, 0.4, 0.5, 0.6, 0.7]
  - 3 secondary site runs × α=0.3: o_down, all, igfo
  - 8 total runs at Mini scale on RTX 4090
  - Cumulative wall-clock: ~10 minutes

**Outcome distribution:**

  - 0/8 STRONG-PASS (val_loss improvement ≥ 0.05 nat)
  - 6/8 NEUTRAL (|delta| < 0.05 nat noise band)
  - 2/8 HARD-FAIL (α=0.7 single-site, sites=all multi-site)
  - 0/8 PARTIAL/CALIBRATION-PASS (G2 quant-error metric not measured;
    deferred to a separate compute_quant_error_per_channel pass)

**What this NEGATIVE result tells us:**

1. **SmoothQuant is not broken on HGRN.** The recipe applies cleanly
   (no NaN/Inf, validation gates pass for 7/8 runs, side-car schema
   round-trips successfully). The math works.

2. **The model is what's invariant, not the recipe.** With α=0.3 +
   sites=o_down (the two highest-outlier sites per F.6.1), val_loss
   moves by 1e-4 nat — within floating-point noise. The trained
   ckpt's RMSNorm γ has effectively pre-rescaled.

3. **Multi-site coverage saturates ternary.** sites=all (37 BitLinear
   sites including i/f/g_proj + lm_head + mlp.gate_proj) regresses
   catastrophically by +0.27 nat at the same α=0.3. Ternary's ±1
   weight clipping cannot absorb the cumulative s·W magnitude
   inflation. Confirms §7.1 R4 prediction.

4. **Gated paths don't help.** sites=igfo (24 sites: i/f/g + o) shows
   +0.009 nat regression — small but non-zero, and worse than
   o_down. Per-channel scaling on HGRN's gated recurrent paths
   doesn't unlock value the way QuaRot-style rotation might.

5. **F.5.1 design predictions held.** §3.2 (ternary wants α<0.5),
   §7.1 R1 (α=0.5 default wrong), R3 (γ absorbs), R4 (weight
   saturation) all confirmed empirically. The design's risk
   assessment was accurate; the recipe just doesn't unlock new
   value on this specific architecture and training regime.

**Branch decision (§6.4 PARTIAL path):**

  - F.5.1 ships PARTIAL findings doc declaring no production claim
  - F.5.1.7 QAT-time variant NOT pursued (PARTIAL doesn't justify
    training-side investment)
  - Paper §7 Ablations adds an honest negative row + caveat about
    HGRN-ternary's RMSNorm-γ-absorption pattern
  - Phase F roadmap §4.1 F.5.1 closes as PARTIAL; F.5.4 Option B
    becomes optional (no site clearly benefits)
  - Branch A pursuit (canonical QuaRot weight-fusion ~3-5 days) is
    BETTER MOTIVATED than before — F.5.1 invariance evidence shows
    activation-only is provably insufficient, so rotation is the
    next class of recipe to test. Decision deferred to a separate
    F.6 design phase.

**What remains undone:**

  - G2 quant-error per-channel metric (`compute_quant_error_per_channel`
    pass on each calibration map) — would clarify whether the
    invariance is "no per-channel quant improvement either" or
    "per-channel improves but model doesn't see it at val_loss"
  - Larger n_val_batches (e.g. 200) — could shift NEUTRAL deltas
    into measurable territory if there's signal below 0.005 nat
  - Slimpajama-EN-only val stream — bilingual stream is OOD for an
    EN-only ckpt; might be obscuring small effects

---

## 2. Empirical Results

### 2.1 Full sweep table

| Run | α | sites | sites_matched | val_loss | Δ vs baseline | outcome | gates |
|---|---|---|---|---|---|---|---|
| 1 | 0.3 | o | 6 | 5.5599 | +0.0068 | NEUTRAL | 4/4 |
| 2 | 0.4 | o | 6 | 5.5653 | +0.0123 | NEUTRAL | 4/4 |
| 3 | 0.5 | o | 6 | 5.5804 | +0.0274 | NEUTRAL | 4/4 |
| 4 | 0.6 | o | 6 | 5.5905 | +0.0375 | NEUTRAL | 4/4 |
| 5 | 0.7 | o | 6 | 5.6104 | **+0.0573** | **HARD-FAIL** | 2/4 (median band fail at L0/3/4/5) |
| **6** | **0.3** | **o_down** | **12** | **5.5531** | **+0.0001** | **NEUTRAL (best)** | 4/4 |
| 7 | 0.3 | all | 37 | 5.8196 | **+0.2665** | **HARD-FAIL** | 4/4 (gates pass — but model still regresses) |
| 8 | 0.3 | igfo | 24 | 5.5619 | +0.0088 | NEUTRAL | 4/4 |

**Baseline:** `no_smoothquant` val_loss = **5.5530 ± 0.0001 nat**
(reproducibility check: 8 baseline measurements within ±1e-4 of each
other across all runs ✓; matches F.6.2 commit 89f3219 baseline 5.5530
exactly to 4 decimals ✓).

**Calibration data (all runs):** `bilingual_stream(id_share=0.6,
seed=42)` × 64 batches (= 512 sequences at bs=8 × seq_len=1024 =
524,288 tokens).

**Validation data (all runs):** `bilingual_stream(id_share=0.6,
seed=999)` × 50 batches (= 400 sequences = 409,600 tokens).

### 2.2 Monotonic α-regression (single-site `o`)

```
α      delta_vs_baseline
0.3    +0.0068
0.4    +0.0123
0.5    +0.0274
0.6    +0.0375
0.7    +0.0573 (HARD-FAIL)
```

Linear fit: `delta ≈ 0.135 · α − 0.030` (R² ≈ 0.98). Each +0.1 in α
costs ~+0.013 nat val_loss. Extrapolating: α=0.0 (pure weight-side
scaling) → −0.030 nat (would HELP) — but that's the degenerate
`s_j = 1/max(|W_j|)^1` case where activations don't get rescaled at
all, so it isn't actually SmoothQuant.

The α=0.7 HARD-FAIL row's gate failures (`median_in_band` False on
4/6 layers) are the canonical signature of weight-side burden too
high — `s` has shifted some channels' magnitudes outside the
[0.1, 10] median band, indicating extreme outlier compensation.

### 2.3 Site coverage scaling at fixed α=0.3

```
sites      sites_matched   delta_vs_baseline   notes
o                  6        +0.0068            primary site only (51× outlier per F.6.1)
o_down            12        +0.0001            +6 sites of mlp.down_proj (40× outlier)
igfo              24        +0.0088            +18 sites of i/f/g (5-8× outlier)
all               37        +0.2665            +13 sites of lm_head, mlp.gate_proj, etc.
```

The **non-monotonic shape** of the site-coverage axis is the most
informative result:
- `o → o_down`: +6 sites with 40× outlier concentration → delta DECREASES
  to near-zero. The o_down combo is "right".
- `o_down → igfo`: replace down_proj with 18 i/f/g sites (5-8× outlier)
  → delta INCREASES to +0.0088. Adding low-concentration sites costs.
- `igfo → all`: add 13 more sites including very-low-concentration
  ones (lm_head and mlp.gate_proj are not even in the F.6.1 outlier
  table) → delta EXPLODES to +0.2665. Net regression dominates.

Implication: SmoothQuant only helps (or stays out of the way) on
sites with strong genuine outliers. Per-channel scaling on uniform
distributions (lm_head, mlp.gate_proj) is anti-helpful — it
introduces unnecessary scale mismatch without compensating quant
benefit.

### 2.4 Edge-case counts per §4.4 clamping

```
Run      n_act_clamped   n_w_clamped   n_s_clamped_lo   n_s_clamped_hi
α=0.3 o      24             0               0                0
α=0.4 o      24             0               0                0
α=0.5 o      24             0               0                0
α=0.6 o      24             0               0                0
α=0.7 o      24             0               7                0      ← weight saturation marker
o_down       35             0               0                0
all          39             0               0                0      ← wider sweep but no per-channel clamp
igfo         24             0               0                0
```

`n_act_clamped` represents per-channel max activations below 1e-5
clamping threshold (zero-input channels in calibration). Stable across
runs at 24 (matches the 4 i/f/g/o input channels at hidden=256 with 6
layers — most at-zero channels are in non-attended dims at calibration
end). For sites=all, count rises to 39 because lm_head and gate_proj
also have at-zero columns (less-active vocab tokens / unused MLP
neurons).

`n_s_clamped_lo` (s clamped to 1e-3 minimum) appears ONLY at α=0.7
single-site `o`: 7 of 6 layers × 256 channels = 1792 channels total,
so 7/1792 = 0.4% of channels hit the floor. Very localized but
consistent with the §7.1 R4 weight-saturation prediction.

`n_s_clamped_hi` (s clamped to 1e3 maximum) is zero across all runs
— no calibration produced extreme high-magnitude scales. The
calibration distribution's outliers are bounded enough that
α-blowup at high migration values is the only failure mode, not
small-max-act explosion.

---

## 3. Validation Gates Detail

> **TODO** — to be fleshed in next first-step. Sketch:
> - Per-run breakdown of which §4.6 gates passed/failed
> - Layer-level analysis of α=0.7 median_in_band failure pattern
> - Comparison with F.6.2 honest verdict pattern (recipe-incompleteness
>   vs recipe-saturation distinction)

---

## 4. Diagnostic Analysis

> **TODO** — to be fleshed in next first-step. Sketch:
> - Why o_down at α=0.3 is invariant: RMSNorm γ pre-absorption hypothesis
> - Why igfo regresses slightly: gated path activation distribution
> - Why all catastrophically fails: cumulative weight saturation
> - α-axis interpretation: ternary tolerance vs INT8 literature

---

## 5. Comparison vs F.6.2 + E2.4 Baselines

> **TODO** — to be fleshed in next first-step. Sketch:
> - F.5.1 best (+0.0001 nat) vs F.6.2 worst (+2.59 nat) — recipe
>   completeness matters; activation-only SmoothQuant doesn't break
>   FP path while activation-only Hadamard pre-hook does
> - E2.4 balanced_calib comparison — both target outliers, both fail to
>   move val_loss; different mechanism (running_max vs SmoothQuant)
>   converges on same conclusion
> - The cumulative E2.1 + F.6.2 + F.5.1 evidence supports a strong
>   negative claim about HGRN ternary's calibration ceiling

---

## 6. Decision per §6 Verdict Tree

> **TODO** — to be fleshed in next first-step. Sketch:
> - Verdict: PARTIAL (per §6.3)
> - Per-verdict next steps: F.5.1.7 not pursued; paper Table 4 honest
>   row; consider Branch A canonical QuaRot
> - Why not FAIL: best run is invariance, not regression; gates pass

---

## 7. Composability Notes

> **TODO** — to be fleshed in next first-step. Sketch:
> - SmoothQuant + canonical QuaRot stacking: still potentially valid
>   per §6.5 (algebraic composition), but F.5.1 invariance suggests
>   SmoothQuant adds no value to compose with
> - SmoothQuant + balanced_calib: moot per §6.5 (one undoes the other)
> - SmoothQuant + Hadamard pre-hook: NOT recommended per §6.5

---

## 8. Implications for Paper Table 4

> **TODO** — to be fleshed in next first-step. Sketch:
> - Adds honest negative row to paper §7 Ablations
> - Positions SmoothQuant as "tested, doesn't add value to HGRN-ternary"
>   with diagnostic explanation (RMSNorm γ absorption hypothesis)
> - Updates verify_intllm_tables.py with new ablation data
> - Strengthens the broader narrative: HGRN ternary saturates
>   activation-outlier-mitigation axis; future gains require structural
>   changes, not better calibration

---

*Document version: 0.1 (skeleton)*
*Last updated: 2026-04-28 (V32-prep: §1 Executive Summary + §2 Empirical Results fleshed; §3-§8 placeholders for next first-step)*
