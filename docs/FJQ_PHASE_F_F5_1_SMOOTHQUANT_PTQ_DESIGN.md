# Phase F F.5.1 — SmoothQuant PTQ Design (v0.4)

> **Status:** §1 Motivation + §2 Background + §3 Adaptation to FajarQuant + §4 Calibration Recipe + §5 Implementation Plan fleshed; §6-§7 are placeholders. Each subsequent first-step fills one section. Target v1.0: full design doc, ready for impl scaffold.
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

### 3.1 What BitLinear actually does (correction vs §1/§2 placeholder)

Reading the upstream `mmfreelm/ops/bitnet.py:64-87` shows BitLinear's
forward is:

```python
def forward(self, x):
    x_norm = self.norm(x)                                        # RMSNorm — has per-channel γ
    x_quant = activation_quant(x_norm)                           # per-TOKEN INT8, NOT per-channel
    w_quant = weight_quant(w)                                    # per-tensor ternary
    y = F.linear(x_quant, w_quant)
    return y
```

with:

```python
def activation_quant(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values     # dim=-1 ⇒ per-token max
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y
```

**Key correction:** BitLinear has **no per-channel γ_x quantization
scale**. The per-channel parameter is RMSNorm's γ (`self.norm.weight`,
shape `[in_features]`), which is a *normalization* scale — applied
BEFORE quantization, learnable during training. The actual
quantization uses a per-TOKEN max (one scalar per row of the
`[batch×seq, hidden]` tensor), computed dynamically at every forward
pass.

This is *more* favorable to SmoothQuant integration, not less:

1. There is no existing per-channel quantization-scale parameter to
   fight with. `s` slots in cleanly without conflicting against an
   already-calibrated γ_x.

2. RMSNorm's per-channel γ is the natural fusion target (§3.3 below).
   SmoothQuant becomes a *one-line edit* to two existing tensors per
   BitLinear site — no new state, no forward-time overhead.

3. The known weakness of per-token max — that a single outlier
   channel forces the entire token's scale to that outlier's
   magnitude, crushing all other channels — IS exactly the
   pathology SmoothQuant was designed to fix. Canonical use case.

### 3.2 Why SmoothQuant fits BitLinear better than transformer FP layers

In a standard transformer with FP linear layers + INT8 PTQ
post-training, SmoothQuant trades ~5-10% activation-quant MSE
reduction for ~1-2% weight-quant MSE increase. Net positive but
modest because FP activations have better baseline tolerance.

In FajarQuant BitLinear:
- Activations are 8-bit but per-TOKEN, so **outlier sensitivity is
  worse** — one bad channel per token destroys precision in all
  others.
- Weights are 1.58-bit (ternary), so **weight scaling tolerance is
  also worse** — pushing magnitude up via `s · W` may saturate the
  ternary clipping range earlier.

The α-tuning lever (§2.3) becomes more important here. Literature
default α=0.5 was tuned for INT8 weights; ternary weights probably
want α<0.5 (less migration to weights, more activation-side relief),
and this should be measured empirically (§4 calibration recipe will
specify the sweep).

### 3.3 The fusion point — RMSNorm γ (one-line PTQ edit)

The mathematical opportunity:

```
forward (current) :  x  →  RMSNorm(x) = (x / RMS(x)) · γ  →  quant  →  W·(...)  →  y
forward (smooth)  :  x  →  RMSNorm(x) = (x / RMS(x)) · (γ / s)  →  quant  →  (s·W)·(...)  →  y
```

Substituting:

```
y_smooth = (s · W) · activation_quant( (x / RMS(x)) · (γ / s) )
        ≈ (s · W) · activation_quant( x_norm / s )       (where x_norm has per-token large outliers)
        = (s · W) · (x_norm_quant / s)                   (per-token quant of x_norm/s)
        = (s · s⁻¹) · W · x_norm_quant
        = W · x_norm_quant     (mathematically preserved in FP; in quant, error reduced)
```

**Implementation as a PTQ edit** (assuming `s_j` per-channel scale already
computed per §4 calibration recipe):

```python
# At PTQ time, for each BitLinear instance b:
b.norm.weight.data /= s         # RMSNorm γ ← γ / s  (in-place, no new state)
b.weight.data *= s.unsqueeze(0) # W ← W · diag(s)    (broadcasted along output dim)
```

Two tensor mutations, no new module, no new parameter, no forward-time
overhead. After mutation, the model runs unchanged through normal
inference. The `s` vector itself doesn't need to be stored for inference
— it's been absorbed into the existing parameters.

### 3.4 Storage format — extension of E2.4 `save_calibration_maps`

For ablation reproducibility + α sweep + diagnostic re-evaluation,
`s` MUST be stored alongside the model. Three storage options:

| Option | Description | Pros | Cons |
|---|---|---|---|
| **A: Pure in-place mutation** | Mutate ckpt directly, save modified state_dict | Simplest deploy path | No way to revert, no audit trail of what was applied, breaks reproducibility if α changes |
| **B: Side-car file** | Save unmutated ckpt + `<ckpt_stem>_smoothquant.pt` containing `{layer_name: s}` dict | Reversible, supports α sweep, separate from model | Two files to ship together; requires loader awareness |
| **C: Extend E2.4 maps schema** | Add `smoothquant_s` field to `save_calibration_maps` v1.3 schema | Composes with E2.4 outlier maps; one calibration artifact | E2.4 demoted (F.5 future-work); reusing its schema may confuse readers |

**Recommendation: Option B** for V32-prep. Cleanest separation. If F.5.1
ships into production, the side-car can be optionally fused into the
ckpt via a one-time tool. E2.4 schema reuse (Option C) would couple
F.5.1 to a demoted feature, increasing tech-debt risk.

`smoothquant.pt` schema v1.0:
```python
{
  "_schema_version": "1.0",
  "ckpt_path": "<source ckpt>",
  "alpha": float,                          # migration strength used
  "n_calibration_sequences": int,          # e.g. 512
  "calibration_seed": int,                 # e.g. 42
  "calibration_stream": str,               # e.g. "bilingual(id_share=0.6)"
  "per_layer": {
    "model.layers.0.attn.o_proj": {
      "s": Tensor[hidden_size],            # per-channel scale
      "max_act": Tensor[hidden_size],      # diagnostic: max(|X_j|) seen
      "max_weight": Tensor[hidden_size],   # diagnostic: max(|W_j|) seen
    },
    ...
  },
  "timestamp": str (ISO 8601)
}
```

This format supports the four use cases:
1. **Apply** — load `s`, mutate model in-place per §3.3
2. **Sweep** — same calibration data, multiple α values → multiple side-car files
3. **Diagnose** — inspect per-channel max stats to understand why F.6.1 outlier shape responds (or not) to scaling
4. **Combine** — composing SmoothQuant + canonical QuaRot or SmoothQuant + balanced_calib could read both side-cars and reason about composition

### 3.5 Site-restriction strategy

Per F.6.1 outlier evidence (`paper/intllm/ablations/outlier_concentration_mini.json`):
- `o_proj`: 51.6× mean, 421× max ratio — **strongest concentration, primary target**
- `mlp.down_proj`: 40× mean, 250× max — strong, secondary target
- `i_proj`, `f_proj`, `g_proj`: 5.3-7.8× — moderate, tertiary
- `lm_head`: not yet measured (F.6.1 measured 37 BitLinear sites including head; need to confirm)

Recommended sweep ordering:
1. `--smoothquant-sites o` — test on highest-concentration site only (matches F.6.2 baseline mode)
2. `--smoothquant-sites o,down` — add MLP down_proj
3. `--smoothquant-sites all` — every BitLinear gets SmoothQuant
4. `--smoothquant-sites igfo` — test moderate-concentration attn projs (parallel to F.6.3 site mode)

Expected result if SmoothQuant works as intended: monotonic improvement
from `o` → `o,down` → `all`, with `igfo` showing modest standalone
benefit. Non-monotonic result indicates either α was wrong (compensation
overshoot at some sites) or composition failure (two sites' `s` vectors
disagree about residual stream's effective distribution).

### 3.6 Static vs dynamic calibration

**Static (one-time PTQ):** compute `s` once on a 512-sequence
calibration batch, fuse into `norm.weight` and `weight` per §3.3, save
side-car, ship. All inference uses the static `s`.

**Dynamic (per-batch recalibration):** compute `s` at every forward
pass from current activation distribution. No side-car needed; model
adapts to deployment-time data.

**Recommendation: Static.** Reasons:
1. SmoothQuant paper's experiments are all static — literature-aligned.
2. Dynamic recalibration requires additional forward-pass cost (max
   over hidden dim), which negates the "no overhead" win in §3.3.
3. Phase D Mini final checkpoint was trained on slimpajama EN — its
   activation distribution at inference on bilingual data is the same
   distribution used for our calibration batch (`bilingual_stream(seed=42)`),
   so static calibration is representative.
4. The side-car artifact (Option B in §3.4) is itself a useful
   inspection tool. Throwing it away with dynamic recalibration loses
   reproducibility leverage.

Dynamic could be revisited if F.5.1 ships and deployment-time
activation distributions differ significantly from calibration
distributions — but that's a §6 fallback consideration, not the
primary design.

---

## 4. Calibration Recipe

### 4.1 Calibration data specification

| Field | Value | Rationale |
|---|---|---|
| Stream | `bilingual_stream(id_share=0.6, seed=42)` | Matches `compute_quant_error_per_channel` (E2.4.C metric) + F.6.1 outlier measurement seed → composable artifacts |
| Sample size | **512 sequences** | SmoothQuant paper §4.2: stable from 128 upward; 512 is the conservative default. Mini scale: 64 batches at bs=8 = 512 sequences. |
| Sequence length | `train_hp.seq_len = 1024` | Mini config; matches inference distribution |
| Total tokens | 512 × 1024 = **524,288** | Sufficient for stable per-channel max statistics on hidden=256 dim |
| Wall-clock (RTX 4090) | ~30-45 sec at Mini scale | Forward-only, no gradient |
| Wall-clock (CPU) | ~5-8 min | Acceptable for offline PTQ |

The same stream + seed used by F.6.1 + F.6.2 + E2.4.C (all reference
`seed=42` for calibration tasks). Side-effect: F.5.1 calibration `s`
artifacts will be cross-verifiable against F.6.1 outlier maps without
re-computing.

### 4.2 Per-BitLinear forward-pre-hook setup

For each BitLinear instance, attach a forward-pre-hook that:

1. Captures the input tensor `x` (shape `[batch, seq, hidden]`)
2. Reshapes to `[batch*seq, hidden]` (flatten token dim)
3. Computes per-channel absmax along token dim:
   ```python
   x_max = x.abs().reshape(-1, hidden).max(dim=0)[0]   # shape [hidden]
   ```
4. Updates running max in tracker state:
   ```python
   tracker.max_act = torch.maximum(tracker.max_act, x_max)
   ```

Hook is non-mutating: `return None` so the original input flows through
unchanged. Calibration is observation-only.

After 64 batches (512 sequences), `tracker.max_act` per BitLinear
contains the per-channel max of activations across the entire
calibration set — this is the `max(|X_j|)` term in SmoothQuant
Algorithm 1.

**Reuse note:** the existing `attach_stat_trackers` from
`intllm.qat` (commit pre-existing; production-integrated since
E2.4.A) already implements this hook pattern with
`accumulator_mode="max"` semantics. SmoothQuant calibration can reuse
it verbatim — no new infrastructure needed for activation capture.

### 4.3 Per-weight scan (per-input-channel max)

Distinct from activation capture: per-weight scan is a one-time
inspection of the static weight tensor, not a forward-time hook.

Recall `nn.Linear` stores `W` with shape `[out_features, in_features]`.
SmoothQuant's `s_j` is indexed by INPUT channels (the dimension that
gets divided/multiplied), so:

```python
max_w_per_in_channel = W.abs().max(dim=0)[0]   # reduce over output dim
                                                # result shape [in_features]
```

This is per-BitLinear, computed once at calibration time. No tracker,
no hook — just `module.weight.abs().max(dim=0)[0]`.

For BitLinear specifically, `W` is the FP-shadow weight (the one
passed through `weight_quant` at training/inference). Use the FP
shadow, NOT the post-`weight_quant` ternary version, because:
- Post-quant weights are always ±1 or 0; per-channel max collapses to 1 or 0 depending on whether any non-zero exists in the column
- That's a degenerate signal — every channel either gets `s_j ≈ 1` or `s_j → ∞`
- Pre-quant FP shadow preserves the actual magnitude information
  SmoothQuant needs

### 4.4 Computation of `s_j`

Given per-channel `max_act_j` and `max_w_j`, compute:

```python
s_j = (max_act_j ** alpha) / (max_w_j ** (1 - alpha))
```

with edge cases:
- `max_act_j == 0` (channel never activated in calibration): set
  `s_j = 1.0` (no scaling, leave channel alone). Clamp before
  exponentiation: `max_act = max_act.clamp(min=1e-5)`.
- `max_w_j == 0` (channel has all-zero weight column — possible after
  ternary clipping): set `s_j = 1.0`. Same clamp.
- `s_j` overflow / underflow: clamp final `s` to a sensible range,
  e.g. `[1e-3, 1e3]` — extreme `s` values indicate calibration data
  was unrepresentative or there's an upstream NaN.
- After clamp + compute: log how many channels (per BitLinear) hit
  each edge case as part of the side-car file's diagnostic fields.

### 4.5 α sweep specification

Per §3.2 conclusion (ternary weights have worse magnitude tolerance
than INT8), the literature default α=0.5 is unlikely to be optimal
for FajarQuant.

**Primary α sweep:** `[0.3, 0.4, 0.5, 0.6, 0.7]` × `--smoothquant-sites
o` (single-site, fastest signal). 5 calibration runs, 5 evaluations,
~5 minutes total wall-clock at Mini scale. Picks the per-site optimal
α before scaling up to multi-site experiments.

**Secondary α sweep (after primary picks an α):** fix α at primary's
best, sweep `--smoothquant-sites` ∈ `{o, "o,down", all, igfo}`. 4
calibration runs. Tests whether SmoothQuant compounds across sites
or if there's a saturation point.

**Optional α-per-site sweep:** if primary shows α-dependency varies
significantly between sites (e.g. `o` likes α=0.4 but `down_proj`
likes α=0.6), allow per-site α via a flag `--alpha-per-site
o:0.4,down:0.6,igf:0.5`. This is overkill for the initial run; ship
only if simple uniform-α leaves clear performance on the table.

### 4.6 Validation & sanity checks

Before applying `s` to a model, validate:

| Check | Threshold | Action on fail |
|---|---|---|
| `s` finite (no NaN/Inf) | All values finite | Abort calibration, dump `max_act` / `max_w` distributions for diagnosis |
| `s` range | `[1e-3, 1e3]` per §4.4 | Clamp + log count of clamped channels as diagnostic |
| `s` median | Within [0.1, 10] of 1.0 | Calibration is sane; if not, investigate |
| Median condition number | `max(s) / min(s) ≤ 1e6` | Calibration produced highly uneven scales — likely bad calibration data |
| Pre-mutation forward | val_loss within ±0.01 nat of un-mutated baseline | Forward equivalence check (FP path preservation) |

The pre-mutation forward check is critical: BEFORE applying `γ ← γ/s`
+ `W ← s·W`, run a 1-batch val_loss on the mutated copy vs the un-
mutated original. If they disagree by more than ±0.01 nat, the
mathematical fusion is broken (typo in the elementwise broadcasting,
shape mismatch, etc.). Catches bugs before downstream gates.

### 4.7 Determinism & reproducibility

Required for paper-claim integrity per §6.6 R3:

- All `torch.manual_seed` / `numpy.random.seed` set at calibration entry
- `bilingual_stream(seed=42)` deterministic (already verified by E2.4.C
  metric reproducibility)
- Calibration order — process BitLinears in `model.named_modules()`
  iteration order (deterministic since Python 3.7 dict ordering)
- Side-car file `_schema_version: "1.0"` checked at apply-time;
  schema mismatch aborts apply
- α / sites / seed all recorded in side-car file — re-running the same
  calibration on same ckpt with same parameters MUST produce
  bit-identical `s` (up to floating-point determinism in CUDA reductions
  — verify via diff)

CUDA reduction determinism requires `torch.use_deterministic_algorithms(True)`
+ `CUBLAS_WORKSPACE_CONFIG=:4096:8` env var. Acceptable performance
penalty for offline PTQ calibration; would not be enabled for inference.

---

## 5. Implementation Plan

### 5.1 Sub-task overview

| ID | Sub-task | File(s) | Effort | Depends on | Critical path? |
|---|---|---|---|---|---|
| **F.5.1.1** | `SmoothQuantCalibrator` module | `python/phase_d/intllm/quant.py` (extend) | ~½ day | — | Yes |
| **F.5.1.2** | Eval harness scaffold + dry-run smoke | `python/phase_d/scripts/eval_smoothquant_posthoc.py` (new) | ~½ day | F.5.1.1 | Yes |
| **F.5.1.3** | α sweep wrapper | same file (CLI flag) | ~¼ day | F.5.1.2 | Yes |
| **F.5.1.4** | Site-restriction flag | same file (CLI flag) | ~¼ day | F.5.1.2 | Yes |
| **F.5.1.5** | Primary α sweep execution | RTX 4090 run | ~5 min wall + ~¼ day analysis | F.5.1.1-F.5.1.4 | Yes |
| **F.5.1.6** | Findings doc + verdict | `docs/FJQ_PHASE_F_F5_1_FINDINGS.md` (new) | ~½ day | F.5.1.5 | Yes |
| **F.5.1.7** | (Optional) QAT-time variant | `train_mini_ablation.py` (extend) | ~1 day | F.5.1.6 PASS | No (deferred) |

**Critical-path total:** ~3 days solo. **With F.5.1.7:** ~4-5 days.
Phase F roadmap §4.1 estimate "~1 week" matches with buffer for
α-per-site exploration (§4.5 optional sweep) + write-up polish.

### 5.2 F.5.1.1 — `SmoothQuantCalibrator` module

Extends `python/phase_d/intllm/quant.py` (existing module containing
`HadamardRotation` etc.).

```python
class SmoothQuantCalibrator:
    """Static-calibration SmoothQuant scale generator + apply utility.

    Per design doc §3-§4. Produces per-BitLinear `s` vectors from a
    fixed calibration batch, validates, and either:
      - returns side-car payload for `save_smoothquant_maps` (§3.4 Option B), or
      - mutates the model in-place per §3.3 fusion identity.
    """

    def __init__(self, model: nn.Module, alpha: float = 0.5,
                 sites: list[str] = None, device: str = "cuda"):
        ...

    def calibrate(self, batches, n_batches: int = 64) -> dict:
        """Run forward-pre-hook capture + weight scan, return per-layer
        {name: {'s': Tensor, 'max_act': Tensor, 'max_w': Tensor}}.
        """
        ...

    def validate(self, calibration_dict) -> dict:
        """Run §4.6 validation gates. Return {gate_name: passed: bool, ...}.
        Raises CalibrationError if any hard gate fails.
        """
        ...

    def apply(self, model, calibration_dict, in_place: bool = True) -> nn.Module:
        """Apply §3.3 fusion: γ ← γ/s and W ← s·W per BitLinear site.
        Returns mutated model (or copy if in_place=False).
        """
        ...

def save_smoothquant_maps(out_path, calibration_dict, alpha, sites, ...):
    """Side-car file writer per §3.4 Option B schema v1.0."""
    ...

def load_smoothquant_maps(path) -> dict:
    """Reverse of save_smoothquant_maps."""
    ...
```

Test coverage required (`tests/test_smoothquant_calib.py`):
- `s` shape correct per BitLinear input dim
- Forward equivalence pre/post fusion (±0.01 nat per §4.6 gate)
- `s` finite + range valid (§4.4 edge cases)
- Save/load round-trip preserves all keys + values
- Determinism: same seed + ckpt → bit-identical `s`

### 5.3 F.5.1.2 — Eval harness scaffold

New file: `python/phase_d/scripts/eval_smoothquant_posthoc.py`.
Mirrors `eval_hadamard_posthoc.py` (F.6.2 commit `b46453a`) closely
for interface consistency.

CLI:
```
PYTHONPATH=. python scripts/eval_smoothquant_posthoc.py
    [--ckpt PATH]                       # default mini_final.pt
    [--alpha A]                         # default 0.5
    [--sites {o,o+down,all,igfo}]       # default o
    [--n-calibration-batches N]         # default 64 (= 512 sequences at bs=8)
    [--n-val-batches N]                 # default 50
    [--device cpu|cuda]
    [--out PATH]                        # default paper/intllm/ablations/smoothquant_<sites>_a<α>.json
    [--out-maps PATH]                   # default same stem with _maps.pt
    [--dry-run]                         # build + calibrate + validate without final eval
    [--apply-mode {in_place,side_car}]  # default in_place
```

Output JSON schema (mirror F.6.2's `posthoc_hadamard_mini.json` shape):
```json
{
  "_schema_version": "1.0",
  "ckpt_path": "...",
  "alpha": 0.5,
  "sites": "o",
  "calibration": {
    "n_sequences": 512,
    "stream": "bilingual(id_share=0.6, seed=42)",
    "wall_clock_seconds": 35.2,
    "validation_gates": {...}
  },
  "modes": {
    "no_smoothquant": {"val_loss": F, "ppl": F},
    "smoothquant":    {"val_loss": F, "ppl": F, "delta_vs_baseline": F}
  },
  "verdict": {
    "rotation_outcome": "helps|hurts|neutral",
    "interpretation": "...",
    "gate_nat_threshold": 0.05
  },
  "timestamp": "..."
}
```

Dry-run smoke (CPU, ~1 min): build model + calibrator + run §4.6 hooks
on 1 batch + validate `s` schema. Skips final eval.

### 5.4 F.5.1.3 + F.5.1.4 — α sweep + site-restriction CLI

Both are flags on F.5.1.2's eval harness, not separate scripts. Sweep
is invoked via shell `for` loop (a Makefile target `make
test-smoothquant-sweep` will encapsulate this once F.5.1.6 ships):

```bash
# Primary α sweep (§4.5)
for alpha in 0.3 0.4 0.5 0.6 0.7; do
    python scripts/eval_smoothquant_posthoc.py \
        --alpha "$alpha" --sites o --tag "primary_a${alpha}"
done

# Secondary site sweep (§4.5)
ALPHA_BEST=0.4   # picked from primary
for sites in o "o,down" all igfo; do
    python scripts/eval_smoothquant_posthoc.py \
        --alpha "$ALPHA_BEST" --sites "$sites" --tag "secondary_${sites//,/_}"
done
```

Each invocation produces 1 JSON + 1 maps.pt. After all sweeps, the
9-12 JSONs feed into F.5.1.6 findings analysis.

### 5.5 F.5.1.5 — Primary α sweep execution

Runbook (after F.5.1.1-F.5.1.4 land):

1. **Pre-flight:** `nvidia-smi` confirms GPU idle. Verify F.6.2
   baseline `posthoc_hadamard_mini.json:no_rotation.val_loss` ≈ 5.5530
   reproduces (sanity check on val stream + ckpt).
2. **Run primary α sweep:** 5 runs × ~1 min/run = ~5 min wall.
3. **Pick best α:** lowest `delta_vs_baseline`. Expect α ∈ [0.3, 0.5]
   per §3.2 ternary tolerance argument.
4. **Run secondary site sweep:** 4 runs × ~1 min/run = ~4 min wall.
5. **Inspect results:** all 9 JSONs in `paper/intllm/ablations/`,
   verify validation gates (§4.6) all PASS, no NaN/Inf.
6. **Decision:** if any run shows `delta_vs_baseline ≤ −0.05 nat`,
   F.5.1 PASS gate hit; proceed to F.5.1.6 findings. Otherwise F.5.1
   PARTIAL or FAIL — see §6 decision criteria.

### 5.6 F.5.1.6 — Findings doc

New file: `docs/FJQ_PHASE_F_F5_1_FINDINGS.md`. Mirrors
`FJQ_PHASE_E_E2_4_FINDINGS.md` v1.4 structure (E2.4 closure precedent):

1. Executive summary (PASS / PARTIAL / FAIL)
2. Empirical results table (α × sites grid, 9-12 cells, val_loss + delta)
3. Validation gates summary (§4.6, per-run)
4. Diagnostic analysis (which sites helped, which α optimal, edge case
   counts from §4.4 clamping)
5. Comparison vs F.6.2 baseline + E2.4 baseline
6. Decision per §6 criteria
7. Composability notes (whether SmoothQuant is compatible with future
   F.5.x or F.6.x stacking)
8. Implications for paper Table 4 (PASS adds row; FAIL adds caveat)

### 5.7 F.5.1.7 — QAT-time variant (deferred)

If F.5.1.6 verdict is PASS with strong delta (≥0.10 nat val_loss
reduction), worth investigating whether SmoothQuant calibration during
QAT training compounds with PTQ application. Implementation:

- Add `--smoothquant-train` flag to `train_mini_ablation.py`
- Compute `s` from running per-channel max during calibration window
  (re-uses `BitLinearStatTracker` from E2.4)
- Apply `s` to weights at training time, similar to E2.4.A.2 schema
  but WITHOUT the bit-allocation map step
- Re-run Mini full training with `--smoothquant-train` and compare
  to PTQ-only baseline

Effort: ~1 day. Deferred until F.5.1.5/F.5.1.6 confirms PTQ-only is
worth the investment in a training-side variant.

### 5.8 Dependencies + parallelism

```
F.5.1.1  ←──── F.5.1.2  ←─┬── F.5.1.3 ──┐
                          │                ├── F.5.1.5 ── F.5.1.6 ── F.5.1.7?
                          └── F.5.1.4 ──┘
```

F.5.1.3 and F.5.1.4 can be developed in parallel (both are CLI-flag
extensions to the same script) but are typically batched into one
commit for review simplicity. F.5.1.5 is the critical hand-off from
implementation to evaluation. F.5.1.7 is optional and gated on
F.5.1.6 outcome.

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

*Document version: 0.4*
*Last updated: 2026-04-28 (V32-prep: §5 Implementation Plan fleshed — sub-task table F.5.1.1 through F.5.1.7 with concrete files + effort estimates + dependency graph. Critical-path total ~3 days solo, ~4-5 days with optional QAT-time variant. §6-§7 placeholders for next first-step)*
*v0.3 → v0.4 (2026-04-28): §5 added.*
*v0.2 → v0.3 (2026-04-28): §4 added.*
*v0.1 → v0.2 (2026-04-28): §3 added; §1/§2 γ_x correction.*
*v0.0 → v0.1 (2026-04-28): skeleton + §1 Motivation + §2 Background.*
