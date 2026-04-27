# Phase E E2.4.C.1 — MSE Quantization-Error Metric Specification (v1.0)

> **Status:** E2.4.C.1 design complete — pure spec, no code, no GPU. The contract that E2.4.C.2 (`compute_quant_error_per_channel` in `intllm.eval`) implements against.
>
> **Plan reference:** `FJQ_PHASE_E_E2_4_FINDINGS.md` v1.1 §4.3 (Option C selection) + §6.3 (E2.4.C deferred sub-tasks).
>
> **Predecessor:** E2.4.A calibration plumbing landed 2026-04-27 in commit `61832d5`. `paper/intllm/ablations/mini_<TAG>_maps.pt` artifacts are the input to this metric.
>
> **Last updated:** 2026-04-27 (E2.4.C.1 first issue).

---

## 1. Purpose

Replace the val_loss-based ablation gate (`≥+0.05 nat improvement at Mini scale`) — which is impossible for E2.4 to produce without an arch-level forward-time bit-map application (see findings v1.1 §1) — with a **forward-time-equivalent quantization-error metric** that:

1. Measures whether the saved per-channel bit allocation map (output of `intllm.qat.save_calibration_maps`) actually *would* reduce activation-quantization error if it were applied.
2. Runs purely as offline post-processing on captured activations — no model modification, no `BitLinear.forward` change, no upstream-pin break.
3. Gates on outlier channels specifically (the channels the map promotes to 10-bit), so a "neutral" map that doesn't actually exploit the higher precision fails the gate.

Adoption signal under this metric: ≥10% MSE reduction on outlier channels (per §6 below). FAIL → revert balanced_calib to Phase D infra-only status; PASS → mark balanced_calib as adopted in E2.4.C.4 decision doc.

---

## 2. Eval set (Q11 closure)

**Selected:** Q5-style 1000-batch bilingual sample.

| Parameter | Value | Rationale |
|---|---|---|
| Source | `intllm.data.bilingual_stream(id_share=0.6)` | Matches Q5 val protocol; bilingual is more honest for Phase E goals (vs EN-only literature comparison) |
| Batch size | 8 | Mini config (`MiniTrainConfig.batch_size`) |
| Sequence length | 1024 | Mini config (`MiniTrainConfig.seq_len`) |
| n_batches | 1000 | 8 × 1000 × 1024 = 8.19M tokens; ≥20× the Q5 val sample → stable per-channel MSE estimates |
| `seed` | 42 | Distinct from Q5 train (`seed=0`), Q5 val ID (`seed=999`), Q5 val EN (`seed=998`); reproducible across runs |
| Device | matches the model checkpoint loaded for evaluation | typically `cuda` on RTX 4090 Laptop |

The eval set is **fixed by seed** — every E2.4.C ablation run sees byte-identical batches given the same tokenizer + corpus state. Cross-run comparability is the entire point.

Q11 alternatives explicitly **rejected**:
- *Phase D bench-canonical EN-only:* would let SmoothQuant/GPTQ/AWQ literature comparison drive the metric, but defeats the bilingual focus of Phase E.
- *Both EN-only and bilingual:* doubles the metric surface (now have to pick which gate is binding); not worth the complexity at Mini scale.

---

## 3. Quantizer definitions

Two quantizers are defined; both take `x` of shape `(N, in_features)` (where `N = batch_size × seq_len` after a `reshape(-1, in_features)`) and return a quantized-then-dequantized tensor `y` of the same shape and dtype as `x`.

### 3.1 `q_baseline(x)` — upstream BitNet 8-bit per-token

Verbatim copy of `_upstream/mmfreelm/ops/fusedbitnet.py:15-29`:

```
scale_t   = 127 / max( max(|x[t, :]|), 1e-5 )       for each row t
y[t, c]   = round(x[t, c] * scale_t).clamp(-128, 127) / scale_t
```

- **Granularity:** per-token (per-row) scale, uniform across channels.
- **Bit width:** 8 bits across all channels.
- **No calibration input.** Scale is computed from the data itself per call. This is what production training currently uses and what the Q5 baseline reflects.

### 3.2 `q_calibrated(x, running_max, bits_per_channel)` — per-channel adaptive

```
scale_c   = (2^(bits_per_channel[c] - 1) - 1) / max(running_max[c], 1e-5)
clip_lo_c = -(2^(bits_per_channel[c] - 1))
clip_hi_c = (2^(bits_per_channel[c] - 1) - 1)
y[t, c]   = round(x[t, c] * scale_c).clamp(clip_lo_c, clip_hi_c) / scale_c
```

- **Granularity:** per-channel scale, per-channel bit width.
- **Bit width:** 8 or 10 (low_bits / high_bits per `intllm.qat.QATConfig` defaults), determined by the saved bit allocation map.
- **Calibration input:** `running_max` and `bits_per_channel` from the saved `mini_<TAG>_maps.pt` artifact. Calibration is data-driven (per CLAUDE §6.9 R4: calibrated > per-chunk for data-driven decompositions).

**Implementation note:** `bits_per_channel` is `int32`; `2^(bits-1)` should be computed via `1 << (bits.long() - 1)` to stay in integer arithmetic before the cast back to float. Beware Python's `**` operator silently producing `float` for tensor inputs.

### 3.3 What is *not* defined here

`q_calibrated` is a **standalone math function** consumed only by the metric. It is **not** wired into `BitLinear.forward`. Wiring it into forward is Option B work (see findings v1.1 §1) and is explicitly out of scope for E2.4 under Option C. The metric tells us what the gain *would be* if Option B were later adopted; it does not realize that gain at training time.

---

## 4. Metric formula

For each BitLinear site `L` (37 sites at Mini scale per Q1 closure: 6 layers × 6 + 1 lm_head):

1. Capture `x_L` — the **input** to `L` — via a forward pre-hook (`register_forward_pre_hook`). The hook reshapes the captured tensor to `(N, in_features)` for streaming accumulation.
2. Stream over the 1000 eval batches; for each batch:
   - Compute `q_baseline(x_L)` and `q_calibrated(x_L, running_max_L, bits_per_channel_L)`.
   - Accumulate per-channel sum-squared-error:
     - `SSE_baseline_L[c]   += sum_over_t( (x_L[t,c] - q_baseline(x_L)[t,c])^2 )`
     - `SSE_calibrated_L[c] += sum_over_t( (x_L[t,c] - q_calibrated(x_L,...)[t,c])^2 )`
   - Accumulate per-channel `count_L[c] += N` (= number of rows seen in this batch).
3. After the streaming pass:
   - `MSE_baseline_L[c]   = SSE_baseline_L[c] / count_L[c]`
   - `MSE_calibrated_L[c] = SSE_calibrated_L[c] / count_L[c]`
   - `reduction_L[c]      = (MSE_baseline_L[c] - MSE_calibrated_L[c]) / max(MSE_baseline_L[c], 1e-12)`

`reduction_L[c] > 0` means `q_calibrated` does better than `q_baseline` on channel `c` of layer `L`.

**Streaming is mandatory.** Storing all 8.19M activations × 37 layers in VRAM is infeasible (~30+ GB even with hidden=256). Use the streaming SSE pattern; per-channel state is `O(sum of in_features) ≈ O(L × max(in_features))` total — trivially fits.

---

## 5. Aggregation rules

Three aggregations are reported; the adoption gate (§6) is keyed only on aggregation 3.

### 5.1 Per-layer global mean

For each layer `L`:
- `mean_reduction_L = mean_over_c( reduction_L[c] )`

Reports whether the calibrated quantizer is uniformly better/worse on this layer. Used for diagnostic plots only — NOT a gate.

### 5.2 Global mean across all layers

`global_mean_reduction = mean_over_L( mean_reduction_L )`

Single-number summary for paper Tables 4. Diagnostic only — NOT a gate.

### 5.3 Outlier-channel-restricted mean (the gate)

For each layer `L`, define the outlier set `O_L = { c : bits_per_channel_L[c] == high_bits }` (≈ top 5% by running_max — exactly the channels the map *promotes* to higher precision):

- `outlier_reduction_L = mean_over_c_in_O_L( reduction_L[c] )`
- `outlier_global_reduction = mean_over_L( outlier_reduction_L )`

Why this is the gate: a calibrated quantizer that does not exploit its extra precision on the channels that NEED it provides no real value; conversely, even small uniform improvements on non-outlier channels could be an artifact of per-channel scaling itself rather than the bit-allocation feature. Restricting to `O_L` isolates the actual contribution of the adaptive bit map.

---

## 6. Adoption gate

| Threshold | Value | Justification |
|---|---|---|
| `outlier_global_reduction ≥ 0.10` | 10% | Matches plan-v1.8 §3 PHASE E2.4 adoption threshold and findings v1.1 §4.3 ("≥10% MSE reduction on outlier channels"); 10% is a meaningful effect size at activation-MSE noise floor (typical MSE residuals on Mini-scale BitLinear inputs are ~1e-2 to 1e-3 nat-equivalent). |

**PASS path:** record `outlier_global_reduction` in `paper/intllm/ablations/mini_balanced_calib_quant_error.json` (schema in §7); E2.4.C.4 decision doc records `balanced_calib` as adopted; `intllm_en.BILINGUAL_RATIO_DEFAULT = 0.6` becomes the binding default for any future calibration step.

**FAIL path:** record the negative result honestly; E2.4.C.4 decision doc states "balanced_calib does not provide a measurable quantization-error reduction over uniform 8-bit baseline at Mini scale; demoted to Phase D infrastructure-only status; not recommended for adoption in Phase E2 main results." Note this is a §6.6 R3 honesty event — negative results are paper-worthy and must be reported.

**Sensitivity:** at Mini scale (22M params), MSE on outlier channels has high variance batch-to-batch. The 1000-batch eval set should bring per-channel MSE standard error to ≤2% (rough Monte-Carlo estimate); a measured 10% reduction with that SE is significant. If the measured reduction lands in the 5-10% range, that is OBSERVATION territory — flag for re-evaluation at Base or Medium scale rather than auto-FAIL.

---

## 7. Implementation contract for E2.4.C.2

The function `intllm.eval.compute_quant_error_per_channel` must satisfy this signature and behavior:

```python
def compute_quant_error_per_channel(
    model: torch.nn.Module,
    *,
    batches: Iterator[torch.Tensor],
    n_batches: int,
    bit_map_path: str | Path,
    device: str | torch.device = "cuda",
    out_path: Path | None = None,
) -> dict:
    """Run `n_batches` of forward passes through `model`, capture the
    INPUT to each BitLinear via forward pre-hook, and compute per-channel
    MSE under both `q_baseline` and `q_calibrated` quantizers.

    `bit_map_path` points to the .pt artifact written by
    `intllm.qat.save_calibration_maps` (one entry per BitLinear with
    `running_max` and `bits` keys).

    Returns a dict with this schema (atomically written to `out_path`
    if provided):

        {
          "_schema_version": "1.0",
          "tag": str,                          # carried from bit_map_path._meta
          "n_batches": int,
          "n_layers": int,                     # = 37 at Mini
          "global_mean_reduction": float,      # §5.2
          "outlier_global_reduction": float,   # §5.3 — the gate
          "gate_threshold": 0.10,
          "gate_pass": bool,
          "per_layer": {
            "<layer_name>": {
              "mean_reduction": float,         # §5.1
              "outlier_mean_reduction": float,
              "n_outlier_channels": int,
              "n_total_channels": int,
              "mse_baseline_mean": float,      # diagnostic
              "mse_calibrated_mean": float,    # diagnostic
            },
            ...
          },
          "timestamp": str (ISO 8601),
        }

    The `q_calibrated` helper (a standalone math function, NOT a
    BitLinear modification) lives at `intllm.quant.activation_quant_per_channel`
    and is implemented as part of E2.4.C.2.
    """
```

**Memory contract:** per-layer state ≤ `2 × in_features × float64 + counts` bytes; total ≤ `2 × sum(in_features) × 8 bytes ≈ kilobytes`. No activation tensors held across batches.

**Determinism contract:** given a fixed `seed=42` + fixed model checkpoint + fixed bit_map, the output JSON must be byte-identical (modulo `timestamp`) across runs. No torch RNG calls outside the seeded `bilingual_stream`.

**Performance contract:** 1000-batch sweep on Mini at RTX 4090 ≤ 5 minutes wall (forward-only, batch_size=8, seq_len=1024 ≈ Q5 val rate × 20). Beyond 10 minutes → investigate (likely accidental tensor copy in the streaming accumulator).

**Math invariants (unit-test surface for E2.4.C.2.2):** the v1.0 draft of this section claimed `q_calibrated` with `bits=8` everywhere ≡ `q_baseline`. That is **mathematically incorrect** — `q_baseline` uses a per-token (per-row) scale derived from the row's own max, while `q_calibrated` uses a per-channel scale derived from the calibration `running_max`. The two only coincide in the degenerate case where every row's per-channel max equals the calibration `running_max` for every channel — which never holds for real activations. The corrected invariants the unit tests actually enforce:

  1. **Hand-computed reference:** for a tiny known `(x, running_max, bits)`, `q_calibrated` output matches a hand-derived expected tensor within 1e-4 (banker's rounding on tensor-half values).
  2. **Shape preservation:** input shape `(..., in_features)` is preserved across 2D and 3D input.
  3. **Clip behavior:** values with `|x| > running_max[c]` are clamped to `±qmax/scale_c`, never blown up.
  4. **Bit-width effect:** with input held inside the calibrated range (no clipping), 10-bit MSE is ≥4× smaller than 8-bit MSE — uniform-quantizer step² scaling, the property the adoption gate exploits.
  5. **Idempotence:** `q_calibrated(q_calibrated(x, ...), ...) ≈ q_calibrated(x, ...)` within 1e-5 — projection onto the per-channel grid.
  6. **Zero handling:** all-zero input + zero `running_max` returns all-zero (no NaN from divide-by-eps).
  7. **Channel-dim mismatch:** mismatched `in_features` between `x`, `running_max`, and `bits_per_channel` raises `ValueError` before any compute.

---

## 8. References

- Plan: `FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.8 §3 PHASE E2.4
- Predecessor findings: `FJQ_PHASE_E_E2_4_FINDINGS.md` v1.1 §4.3 + §6.1+§6.3
- E2.4.A artifact: `paper/intllm/ablations/mini_<TAG>_maps.pt` (schema in `intllm.qat.save_calibration_maps` docstring)
- Upstream `activation_quant`: `python/phase_d/_upstream/mmfreelm/ops/fusedbitnet.py:15-29`
- Q5 plumbing reused: `intllm.data.bilingual_stream` (commit `1074883`)
- E2.4.A wire-up: `scripts/train_mini_ablation.py` `--balanced-calib` (commit `61832d5`)
- Quantization theory: SmoothQuant (arxiv 2211.10438), GPTQ (arxiv 2210.17323), AWQ (arxiv 2306.00978) — all use per-channel calibrated scales as the standard pattern
- Outlier-channel bit-allocation precedent: BitNet b1.58 paper §3.3 framing of "few outlier-prone channels keep more dynamic range"

---

**Sign-off (v1.0):** E2.4.C.1 metric specification COMPLETE. Q11 closed (eval set = bilingual 1000-batch sample, seed 42). Implementation contract for E2.4.C.2 frozen: `compute_quant_error_per_channel` + `activation_quant_per_channel` standalone math helper. Adoption gate = ≥10% MSE reduction on outlier channels (the channels the map promotes to 10-bit).

**No code shipped this session — spec only.** Next session natural first step: E2.4.C.2 implementation (~½ day; pure CPU work; covered by unit tests in `tests/test_qat.py` extension).

*Document version: 1.0 (E2.4.C.1 first issue). Author: Claude Opus 4.7 + Fajar (PrimeCore.id).*
*Phase E v1.8 Tier 1+2 scope under Option C. Predecessor: E2.4.A landed 2026-04-27.*
