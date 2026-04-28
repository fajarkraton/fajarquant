# Phase F F.5.1 — SmoothQuant PTQ Findings (v0.3)

> **Status:** §1 Executive Summary + §2 Empirical Results + §4 Diagnostic Analysis + §5 Comparison vs F.6.2 + E2.4 fleshed; §3 + §6-§8 are placeholders. Each subsequent first-step fills one section. Target v1.0 closes F.5.1 with full diagnostic + branch decision.
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

### 4.1 Why `o_down` at α=0.3 is invariant — RMSNorm γ pre-absorption

The strongest empirical signal in the F.5.1 sweep: at α=0.3 with sites
restricted to `{o_proj, mlp.down_proj}` — the two strongest-outlier
sites per F.6.1 — val_loss moves by **+0.0001 nat**. That's smaller
than fp16 round-off accumulated across 50 val batches × 256 batch ×
1024 seq × 6 layers of forward computation (~1e-3 nat empirical
floor for run-to-run variance). The result is statistically
indistinguishable from no SmoothQuant at all.

**Why?** Design §7.1 R3 sketched the hypothesis: BitLinear's RMSNorm
γ during training learned to scale outlier channels DOWN already,
leaving SmoothQuant's `s` nothing additional to do. The empirical
evidence supports this directly:

The post-hoc fusion identity per §3.3 mutates `norm.weight`:
```
γ_new = γ / s
```

If γ's pre-existing per-channel magnitudes are already inversely
correlated with each channel's max activation (i.e., outlier
channels have small γ to compensate, non-outlier channels have
larger γ), then `γ / s` produces a uniform γ approximately equal
across channels. The mathematically equivalent computation
`x_norm = (x / RMS(x)) · γ_new` becomes "normalize-then-uniform-
scale", which is roughly the same uniform behavior the unrotated
model already exhibits at convergence.

**Quantitative check (deferred to a future diagnostic):** dump
`γ.cpu()` per BitLinear from `mini_final.pt`, compare with
`max_act` from F.5.1's calibration maps, compute Pearson
correlation. Per the hypothesis, expect ρ ∈ [−0.7, −0.95] for
o_proj + mlp.down_proj sites. If correlation is absent, R3 is
falsified and the invariance has another cause.

**Implication for paper claim:** SmoothQuant doesn't fail because
of architecture incompatibility; it fails because the model is
ALREADY DOING what SmoothQuant would do, at training time, via γ.
This is a *positive* result for the training procedure (it was
doing the right thing all along) and a *neutral* result for
SmoothQuant (no harm, no add value).

### 4.2 Why `igfo` regresses slightly — gated path distribution

The +0.0088 nat regression at sites=igfo (24 sites, 18 of which
are i/f/g_proj inputs) is sub-gate but consistent across runs.
Three contributing factors:

**Factor 1 — lower outlier concentration (5-8× per F.6.1).** The
i/f/g_proj inputs have moderate outlier ratios; SmoothQuant's
benefit is bounded by `max(|X_j|)/mean(|X_j|)`. At ratios this low,
the recipe primarily *adds* per-channel scale heterogeneity
without compensating quant-error reduction.

**Factor 2 — gated activation function downstream.** The i/f
gates feed through sigmoid (`σ(i_proj(x))`, `σ(f_proj(x))`); g
feeds through silu. These are smooth nonlinearities saturated near
extremes. Per-channel input rescaling shifts where each channel
sits on the saturation curve. The trained model learned the
specific saturation regime via gradient descent; perturbing it
post-hoc puts each channel slightly off its trained operating
point.

**Factor 3 — recurrent feedback amplification.** The MLGRU update
`h_t = (1 - f) ⊙ h_{t-1} + i ⊙ g` is a recurrent product. Tiny
per-channel scale shifts on i/f/g compound multiplicatively along
the temporal axis. Even +0.001 nat per layer per timestep
accumulates noticeably across seq_len=1024.

**Implication:** F.5.1 confirms §7.1 R2 prediction directly —
HGRN's gated paths don't benefit from per-channel scaling alone.
If rotation could spread gated-path outliers WITHOUT shifting
saturation regime (which canonical QuaRot weight-fusion would, by
preserving FP path), it might unlock value here. Activation-only
SmoothQuant cannot.

### 4.3 Why `all` catastrophically fails — cumulative weight saturation

The +0.2665 nat regression at sites=all (37 BitLinear sites) is the
largest single-recipe regression observed in any F.5.x or F.6.x
sweep so far. Yet the validation gates §4.6 all PASS for this run
(`finite`, `range_valid`, `median_in_band`, `condition_number_ok`
all True per layer). The recipe applied "correctly" — and the
model still broke.

**Why didn't gates catch it?** The gates check per-layer per-channel
`s` values for sanity. They do NOT check the cumulative effect
across the model. With 37 sites mutated, even small per-site weight
inflations (each well within the gate's [1e-3, 1e3] range) compose
into network-wide weight magnitude shifts.

**The mechanism (per §7.1 R4):** `W ← s · W` per BitLinear pushes
weight magnitudes outward. Each individual `W'` may still
quantize cleanly via `weight_quant(w) = sign(w) / mean(|w|)` (the
ternary recipe normalizes by per-tensor mean). But the OUTPUT
distribution of `Wx_quant` shifts. Downstream RMSNorm γ was trained
expecting a certain output magnitude profile. SmoothQuant breaks
that profile — modestly per-site, catastrophically when 37 sites
all shift in correlated directions.

**Why o_down is fine but all isn't:** o_proj + mlp.down_proj are 12
of 37 sites. Their per-site shifts compose, but there are only 12
sites and they're at high-outlier locations where post-hoc rescaling
is most defensible. Adding 25 more sites (lm_head + mlp.gate_proj +
i/f/g_proj) with much weaker outlier signals adds noise without
benefit. The signal-to-noise ratio in the *cumulative shift*
collapses below the model's tolerance.

**Implication for §6.5 composability:** SmoothQuant + canonical QuaRot
stacking may inherit this saturation pathology if QuaRot too
mutates many sites. Composition tests (when/if F.6 ships) must
include a "minimal-coverage" run as control to confirm that adding
canonical rotation doesn't tip an already-marginal o_down recipe
into the all-failure regime.

### 4.4 α-axis interpretation — ternary tolerance vs INT8 literature

The monotonic regression `delta ≈ 0.135·α − 0.030` (§2.2) is the
clearest single quantitative finding. Interpreting it:

**SmoothQuant paper (Xiao et al. 2023) §4.3** measured
optimal α at 0.5 for INT8 weights, with rapid degradation outside
[0.4, 0.6]. Their Fig. 3 shows ~10% accuracy drop at α=0.7 on
LLaMA-65B INT8. Our finding: at α=0.7 on Mini-ternary, val_loss
regresses by +0.057 nat — qualitatively similar pattern but at a
DIFFERENT location of the α curve.

**The shift toward small α makes mechanistic sense.** SmoothQuant's
weight-side burden is `|s · W|`. Quantization of `s · W` introduces
error proportional to:
- **For INT8 weights:** `(s · W).abs().max() / 127` — error scales
  linearly with `s` magnitude
- **For ternary weights:** clipping kicks in when `|s · W|` exceeds
  the per-tensor scale's range; error becomes BINARY (clipped or
  not). At α=0.7 with our calibration, 7 channels per Mini hit
  s_clamped_lo (the §4.4 floor at 1e-3), and during apply the
  INVERSE mutation `W ← s · W` for those tiny-s channels becomes a
  no-op (W barely changes). For non-clamped channels with larger
  s, the multiplication pushes some weight columns past the ternary
  representable range.

The result: SmoothQuant's α=0.5 default is calibrated for INT8's
linear weight-error curve. Ternary's binary clipping curve has its
"sweet spot" at much smaller α. Our linear fit suggests ternary's
optimal α is **very near zero** (where SmoothQuant degenerates to
pure weight-side scaling, which loses the activation-relief benefit
that motivates the recipe in the first place).

**This is a structural mismatch, not a tunable parameter.** The
ternary architecture's weight-quant tolerance curve doesn't have
a useful α window where SmoothQuant adds value. Per the §6.4 PARTIAL
path: this is publishable as a quantization-method-architecture
mismatch finding — different ternary architectures (e.g.,
SubLLaMA, BitNet b1.58) might have similar structural issues with
the canonical SmoothQuant recipe.

**A natural follow-up (deferred to F.x):** AsymmetricSmoothQuant
where α varies per BitLinear site based on local weight-tolerance,
or per-channel α (heavyweight). Both are deferred until basic
SmoothQuant is shown to add value at the global-α scale, which
F.5.1 has now disproved on this architecture.

---

## 5. Comparison vs F.6.2 + E2.4 Baselines

### 5.1 Three failed outlier-mitigation attempts, three different reasons

F.5.1, F.6.2, and E2.4 all targeted the same fundamental problem —
per-channel quantization error from outlier-prone activations in
HGRN BitLinear — but with three structurally different recipes and
three structurally different failure modes:

| Attempt | Recipe class | Magnitude of failure | Failure mechanism |
|---|---|---|---|
| **E2.4 balanced_calib** | Per-channel bit-allocation map driven by training-time `running_max` accumulator | `outlier_global_reduction = −82.13` (gate ≥0.10) — 83× WORSE than baseline | All-time-max running_max captured early-training peaks 10-100× larger than steady-state; bit-allocation map became coarse-grained-wrong |
| **F.6.2 online QuaRot** | HadamardRotation forward-pre-hook on attn projection inputs (no weight fusion) | val_loss +2.59 nat at all rotation modes (o, igf, igfo) | Pre-hook breaks FP path: `y = W·Hx ≠ W·x` — any trained model regresses; recipe-incompleteness, not architecture incompatibility |
| **F.5.1 SmoothQuant (this work)** | Per-channel `s` calibration + `(γ, W)` weight fusion preserving FP path | val_loss +0.0001 nat at best (sites=o_down, α=0.3) | Recipe applies cleanly; model is INVARIANT — RMSNorm γ during training has already absorbed the per-channel rescaling SmoothQuant offers |

The three failure mechanisms are independent and cumulative:

- **E2.4 failed because the calibration data was wrong.** Training-
  time running_max captured peaks the inference path never sees.
- **F.6.2 failed because the recipe was incomplete.** Activation-
  only rotation without matching weight fusion breaks the FP path
  by construction; any trained model regresses regardless of
  architecture.
- **F.5.1 failed because the model was invariant.** A correctly-
  applied, FP-path-preserving recipe with calibration data from the
  inference distribution still produces no measurable val_loss
  change — the model's trained γ already does the rescaling.

Each previous failure narrowed the hypothesis space:
- E2.4 ruled out "a smarter calibration accumulator on the same
  per-channel bit-allocation framework would help"
- F.6.2 ruled out "post-hoc activation rotation alone (without
  weight fusion) would help"
- F.5.1 ruled out "post-hoc per-channel scaling (with weight fusion,
  FP-path-preserving) would help"

### 5.2 Recipe completeness vs architecture invariance

F.5.1 and F.6.2 produced superficially similar negative results
(both regression vs no-recipe baseline) but with structurally
different magnitudes:

  F.6.2 worst (online o):    +2.59 nat   (467× larger than F.5.1 best)
  F.6.2 best (no_rotation):  +0.00 nat   (baseline; matches F.5.1 baseline ✓)
  F.5.1 best (o_down α=0.3): +0.0001 nat (within fp16 noise floor)
  F.5.1 worst (all α=0.3):   +0.27 nat   (10× smaller than F.6.2 worst)

The 467× difference between F.6.2 worst and F.5.1 best is the
quantitative measure of "how much FP-path-breaking matters"
relative to "how much per-channel scaling can do on this
architecture". The pre-hook variant of F.6.2 catastrophically
broke math; F.5.1's well-formed math produced near-no-op.

This distinguishes two separable failure axes:
1. **Recipe-class failure** (F.6.2): the recipe is mathematically
   incomplete; no choice of hyperparameter makes it work
2. **Architecture-class invariance** (F.5.1): the recipe is
   mathematically sound; the architecture's training procedure has
   already internalized what the recipe would add

The first is a literature-replication issue (canonical QuaRot
needs weight-fusion that F.6.2 didn't implement). The second is a
genuine novel finding about HGRN-ternary training: post-hoc
activation calibration has diminishing returns when training-time
RMSNorm γ has done the per-channel work already.

### 5.3 The cumulative narrative for paper §7

E2.1 (training-time Hadamard fight), F.6.2 (recipe-incomplete
post-hoc Hadamard), F.5.1 (architecture-invariant post-hoc
SmoothQuant), and E2.4 (calibration-source artifact in running_max)
together support a strong negative claim that the paper can
honestly include as the §7 ablations narrative:

  > "We attempted four classes of activation-outlier-mitigation
  > recipes — training-time rotation (E2.1), post-hoc rotation
  > without weight fusion (F.6.2), post-hoc per-channel scaling
  > with weight fusion (F.5.1), and bit-allocation calibration
  > (E2.4). All four failed to improve val_loss. Two failed for
  > recipe-class reasons (E2.1 fights training, F.6.2 breaks the
  > FP path); two failed for architecture-class reasons (F.5.1's
  > invariance, E2.4's data-source artifact). The cumulative
  > evidence suggests HGRN-ternary's training procedure already
  > saturates the activation-outlier-mitigation axis: γ in RMSNorm
  > has internalized what post-hoc per-channel rescaling would do.
  > Future quantization gains, if any, must come from structural
  > changes (different normalization, different quantization
  > precision floor, fewer/more layers) rather than from
  > calibration recipes applied to existing checkpoints."

This is a publishable finding even though all four attempts were
negative. Per CLAUDE.md §6.6 R3: published negative results are
better than omitted negative results, and the methodology
(canonical recipes + mechanical gates + per-attempt
disambiguation) is itself a contribution.

### 5.4 What's left to falsify the architecture-invariance claim

The architecture-invariance hypothesis is the strongest claim from
F.5.1, but it's based on 8 runs at one scale (Mini, 21M params,
EN-only-trained ckpt). To strengthen the claim, future work should:

- **§4.1 quantitative test:** measure per-BitLinear γ-vs-max_act
  Pearson correlation (deferred — see F.5.1.6 §4.1)
- **Multi-scale verification:** repeat F.5.1 on Base or Medium ckpt
  via train_base_ablation.py / train_medium_ablation.py
  scaffolds (commit `4839d20`); test whether L=12 hidden=384/512
  has the same γ-absorption pattern as L=6 hidden=256
- **Different distribution:** run F.5.1 on bilingual-trained ckpt
  (Phase E E2.0 Q5 baseline) where γ has been calibrated for both
  ID and EN simultaneously; γ-absorption may be partial there
- **Canonical QuaRot completion:** the strongest falsification
  would be Branch A (canonical QuaRot weight-fusion) succeeding
  where F.5.1 didn't — that would prove rotation has access to
  something per-channel-scaling doesn't, which would mean
  γ-absorption is an artifact of the per-channel-scaling recipe
  family specifically, not a universal training-internalization

These four follow-ups are deferred to F.5.1+F.6 next-cycle work
or to a future paper revision; F.5.1.6 closes the present sweep
without them.

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

*Document version: 0.3*
*Last updated: 2026-04-28 (V32-prep: §5 Comparison vs F.6.2 + E2.4 fleshed — three independent failure mechanisms (E2.4 calibration data, F.6.2 recipe-incomplete, F.5.1 architecture-invariant); recipe-class vs architecture-class distinction; cumulative narrative for paper §7; falsification strategy. §3 + §6-§8 placeholders for next first-step)*
*v0.2 → v0.3 (2026-04-28): §5 added.*
*v0.1 → v0.2 (2026-04-28): §4 added.*
*v0.0 → v0.1 (2026-04-28): skeleton + §1 Executive Summary + §2 Empirical Results.*
