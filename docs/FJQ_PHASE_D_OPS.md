# FajarQuant Phase D — Integer Ops Inventory (C.P1.2)

> **Date:** 2026-04-21 | **Depends on:** `FJQ_PHASE_D_ARCH.md` (commit `8c5ce9b`) — Primary arch = MatMul-Free LLM | **Rule:** §6.9 R4 (calibration once, not per-chunk) + §6.9 R5 (outlier handling)

## 0. Convention

**Fixed-point scale:** x1000 throughout the kernel (matches V28/V29/V30
Gemma 3 convention in fajaros-x86 `kmatrix.fj`). A real value `v` is
stored as integer `v × 1000`, truncated toward zero. Multiplications
use i64 temporaries and divide by 1000 at the end.

**Activation quantization:** 8-bit absmax per the BitNet/MatMul-Free
convention (Zhu et al. §3.1). `Q_b = 127`. One `γ_x = ‖x‖_∞` scalar per
BitLinear input — computed once during QAT calibration, reused at
inference.

**Weight quantization:** ternary {−1, 0, +1} via absmean. `β = ‖W‖₁ / nm`.
One scalar `β` per weight matrix — computed once during QAT, baked into
the .fjm file.

**Greedy decode:** LM head produces i64 logits; argmax over `V` vocab
entries. **No softmax at inference**, matching V30.GEMMA3 kernel-path
pattern.

---

## 1. Op catalog

Ten ops total. Six are integer-native; three are bounded transcendentals
with LUT approximations; one is a lookup.

| # | Op | Kind | Per-block count | LOC est |
|---|---|---|---|---|
| 1 | Embedding lookup | Integer lookup | 1 (input) + 1 (LM head weight-tie) | ~30 |
| 2 | RMSNorm | Integer + isqrt | 2× BitLinear per block + 1× final | ~80 |
| 3 | BitLinear ⊛ (ternary MatMul-free) | Integer signed-add | 7 per block (4 MLGRU + 3 GLU) + 1 LM head | ~60 per variant, 3 variants |
| 4 | σ (sigmoid) | LUT transcendental | 2 per MLGRU block | ~40 (LUT + interp) |
| 5 | SiLU = x·σ(x) | Composed LUT | 1 per MLGRU + 1 per GLU | ~20 (wraps σ) |
| 6 | Hadamard ⊙ | Integer multiply | 4 per MLGRU + 1 per GLU | ~10 |
| 7 | Elementwise add | Integer add | 3 per block (residuals + MLGRU update) | ~5 |
| 8 | MLGRU state update | Composed (⊙, add, 1−f_t) | 1 per block | ~15 |
| 9 | Argmax over vocab | Integer reduce | 1 per generated token | ~20 |
| 10 | Activation absmax quant | Integer max-abs + scale + clamp + round | 1 per BitLinear input | ~30 |

Estimated total new Fajar Lang kernel LOC: ~400 (~1 week to write,
~2 weeks to debug to kernel-path cleanliness per V28.1+V28.2 pace).

---

## 2. Op-by-op specification

### Op 1 — Embedding lookup

**Input:** `token_id ∈ [0, V)`, `embed_table: i32[V, d]` (x1000 fixed-point)
**Output:** `x ∈ i32[d]` (x1000 fixed-point, copy of row)
**Dtype:** i32 storage; i32 output.
**Overflow:** None — pure memcpy.
**FajarOS mapping:** `km_embed_lookup(token_id, d)` — already exists
in FajarOS `kmatrix.fj` from V28 SmolLM work. **Reuse.**

---

### Op 2 — RMSNorm (integer)

**Formula** (BitNet/MatMul-Free variant — no learnable γ; scale absorbed
downstream into β/Q_b):
```
rms(x) = sqrt((1/d) × Σ x_i²)
y_i    = x_i / (rms(x) + ε)
```

**Input:** `x: i32[d]` (x1000 fixed-point), `ε ≈ 1e-6` = `1` in x10⁶
scale (scaled internally)
**Output:** `y: i32[d]` (x1000 fixed-point)

**Overflow analysis (d_max = 8192, the largest config):**
- `x_i` worst-case magnitude ≤ β × 127 × d / 1000 × 1000 = β × 127 × d
  (post-⊛ + rescale). For β ≈ 0.7 and d = 8192: |x_i| ≤ 729K. Store
  in i32 comfortably.
- `x_i²` as i64 temp: 729K² = 5.3 × 10¹¹. Safe in i64 (max 9.2 × 10¹⁸).
- `Σ x_i²` over d = 8192: 5.3 × 10¹¹ × 8192 = 4.4 × 10¹⁵. **Still
  safe in i64** — 3 orders of magnitude of headroom.
- `rms(x)` via `km_isqrt`: produces i32 reciprocal in x1000 scale.
- Final `x_i × rms_recip`: i64 temp (7.3 × 10⁵ × 1000 × 1000 = 7.3 ×
  10¹¹). Safe.

**FajarOS mapping:** Extend `km_rmsnorm` (from SmolLM/Gemma work) by
removing the learnable γ multiplication — this is the V30 post-norm γ
site that caused pad-collapse. New kernel fn: `km_rmsnorm_no_gamma`.

**Prevention vs V31.R3:** No per-element γ multiplication anywhere
along the normalization hot path. Only the post-⊛ scalar rescale
`β/Q_b` remains, which is O(0.005) — always shrinking.

---

### Op 3 — BitLinear `⊛` (ternary signed-add MatMul-free)

**Forward** (Zhu et al. eq. 8, Appendix A):
```
x̃ = clip(round(x × Q_b / γ_x), −Q_b+ε, Q_b−ε)       ∈ i8[d_in]
W̃_ij ∈ {−1, 0, +1}                                   ← from stored ternary packing
Y_i = Σ_{j: W̃_ij=+1} x̃_j  −  Σ_{j: W̃_ij=−1} x̃_j    ∈ i32
y_i = Y_i × β / Q_b                                    ∈ i32 (x1000 fixed-point)
```

**Input:** `x: i32[d_in]` (x1000), learned `γ_x`, packed ternary `W`,
learned `β`
**Output:** `y: i32[d_out]` (x1000)

**Overflow analysis:**
- `x̃` is i8, magnitude ≤ 127.
- Inner sum `Σ ±x̃_j` over `d_in` worst case: `127 × d_in`. For d_max =
  8192: 1,040,384 < 2²¹. **Fits in i32 without overflow concern.**
- `Y_i × β / Q_b`: β is stored in x1000 scale, so `Y_i × β_scaled` is
  i64 (1,040,384 × 700 ≈ 7.3 × 10⁸). Safe. Divide by `Q_b × 1` (since β
  already x1000, we output x1000 directly after `/Q_b`).

**FajarOS mapping (three variants):**
- `km_bitlinear_packed` — for the four MLGRU projections per block
- `km_bitlinear_glu` — for the three GLU projections
- `km_bitlinear_head` — for the final LM head to vocab (different
  output dim, possibly weight-tied with embedding)

Ternary packing: 4 entries per byte (2 bits each, `{00=−1, 01=0, 10=+1,
11=reserved}`). Matrix size reduction: ~16× vs f32. For the 2.7B repro
config, 2.7B × 2 bits = 675 MB of packed weights — fits in consumer
RAM.

**Prevention vs V31.R3:** No softmax accumulator hazard (this op
doesn't accumulate exp-of-anything). β is always shrinking, never
magnifying.

---

### Op 4 — σ (sigmoid LUT)

**Formula:** σ(x) = 1 / (1 + exp(−x))

**Input:** `x: i32` (x1000 fixed-point, input range clipped to [−8000, 8000])
**Output:** `y: i32` (x1000 fixed-point, y ∈ [0, 1000])

**Implementation:**
- 257-entry LUT over input range [−8, +8] (step = 1/16 = 0.0625)
- Linear interpolation between LUT points
- Input clamped to [−8000, +8000] (values outside saturate to 0 or 1000)

**LUT construction** (offline Python, baked into .fjm):
```python
import numpy as np
LUT_STEP = 1 / 16  # 0.0625
LUT = np.array([
    int(1 / (1 + np.exp(-(i - 128) * LUT_STEP)) * 1000 + 0.5)
    for i in range(257)
], dtype=np.int32)
```

**Max error:** LUT step 0.0625 + linear interp. Max derivative of σ is
0.25, so max interp error ≤ (0.0625)² × 0.25 / 8 ≈ 1.2 × 10⁻⁴. In
x1000 scale, error ≤ 1 LSB. **Sufficient for ternary weights where
signal per channel is O(100-1000 LSB).**

**Overflow:** None — LUT lookup + linear interp stays within i32.

**FajarOS mapping:** New kernel fn `km_sigmoid_lut`. LUT stored in
kernel `.rodata` section (257 × 4 bytes = 1028 bytes). No runtime
memory alloc.

---

### Op 5 — SiLU (composed)

**Formula:** SiLU(x) = x · σ(x)

**Input:** `x: i32` (x1000), **Output:** `y: i32` (x1000)

**Implementation:**
```
y = (x × km_sigmoid_lut(x)) / 1000
```

**Overflow:** `x × σ(x)` i64 temp: x worst case 2 × 10⁶ (after MLGRU
state), σ output ≤ 1000. Product ≤ 2 × 10⁹. Safe in i64.

**FajarOS mapping:** `km_silu_lut` (thin wrapper over σ).

---

### Op 6 — Hadamard ⊙

**Formula:** `z_i = x_i × y_i / 1000` (preserves x1000 scale)

**Input:** two i32 vectors, **Output:** i32 vector
**Overflow:** `x_i × y_i` i64 temp ≤ 2 × 10⁶ × 2 × 10⁶ = 4 × 10¹² ≪ i64 max.
**FajarOS mapping:** `km_hadamard` — **reuse from V28/V29**.

---

### Op 7 — Elementwise add

**Formula:** `z_i = x_i + y_i`
**Overflow:** i32 sum of i32 values with sign-check; promote to i64 if
guard predicate fires. In practice, residual-stream values bounded by
post-⊛ magnitudes per op 3.
**FajarOS mapping:** `km_vec_add` — **reuse from V28/V29**.

---

### Op 8 — MLGRU state update

**Formula:**
```
h_t_i = f_t_i × h_{t−1}_i + (1000 − f_t_i) × c_t_i
                                     ↑
                           (1 − f_t) in x1000 scale
```

**Input:** `f_t, h_prev, c_t: i32[d]` (x1000)
**Output:** `h_t: i32[d]` (x1000)

**Overflow analysis:**
- `f_t × h_{t−1}` i64 temp: 1000 × 2e6 = 2e9. Safe.
- `(1000 − f_t) × c_t` i64 temp: same bound.
- Sum: 4e9. Safe.
- Final divide by 1000 preserves scale.

**State boundedness proof** (critical for long contexts — addresses
V31.R3-adjacent concern):

Since `f_t ∈ [0, 1]` (scaled [0, 1000]) and the update is a convex
combination:
```
|h_t| ≤ max(|h_{t−1}|, |c_t|) per-channel
```
So ‖h_t‖_∞ ≤ max_s ‖c_s‖_∞ for s ≤ t. The state cannot grow
unboundedly. **No exp-accumulator hazard** (unlike Mamba-2's
`exp(segsum)`).

**FajarOS mapping:** New kernel fn `km_mlgru_update` — ~15 LOC.

---

### Op 9 — Argmax over vocab

**Input:** `logits: i32[V]`
**Output:** `token_id: i32` in `[0, V)`

**Implementation:** single pass, track max_val + max_idx.
**Overflow:** Pure comparison, no accumulation.
**FajarOS mapping:** `km_argmax_i32` — **reuse from V30 lmhead path**
(already written in C for V30 bypass; keep as Fajar Lang post-B.P2
once verified non-vectorized).

---

### Op 10 — Activation absmax quantize

**Formula:**
```
γ_x = max_i |x_i|
s = Q_b × 1000 / γ_x            # scale to [−127, 127], store in x1000 units
x̃_i = clip(round(x_i × s / 1000), −127, 127)
```

**Input:** `x: i32[d]` (x1000), **Output:** `x̃: i8[d]`, `γ_x: i32`
**Overflow:** `x_i × s` i64 temp ≤ 2e6 × 2e8 = 4e14. Safe in i64.
**FajarOS mapping:** New kernel fn `km_absmax_quant_i8`.

---

## 3. Per-block op count (Base config, d=384, L=12)

One decoder block:
- 5 RMSNorm (pre each BitLinear: 4 MLGRU + 3 GLU + 1 pre-first; paper
  shows 1 per BitLinear per Alg. 1)
- 7 BitLinear `⊛` (4 MLGRU: f, c, g, o; 3 GLU: g, u, d)
- 2 sigmoid LUT (f_t, g_t in MLGRU)
- 2 SiLU (c_t in MLGRU, gate in GLU)
- 4 Hadamard (f⊙h, (1-f)⊙c, g⊙h, gate⊙up)
- 3 elementwise add (MLGRU state sum, MLGRU residual, GLU residual)
- 7 activation absmax quant (one per BitLinear input)

Total per block: ~30 heavy ops. L=12 layers: ~360 ops per token.

---

## 4. Calibration-once table (§6.9 R4 compliance)

Quantities computed **once at .fjm load**, never recomputed per-forward:

| Constant | Scope | Count (Base config) | Size |
|---|---|---|---|
| β (per weight matrix) | Per BitLinear | 7 × 12 + 1 = 85 | 340 B |
| γ_x calibration (per BitLinear input) | Per BitLinear | 85 | 340 B |
| RMSNorm ε scale factor | Per RMSNorm site | 60 | 240 B |
| σ LUT | Global | 1 | 1028 B |
| Ternary weight storage (packed) | Per BitLinear | 85 | ~6 MB at Base, ~650 MB at 2.7B |

All constants baked into `.fjm` header or `.rodata`. Total memory
overhead < 2 KB for non-weight constants. Fits comfortably in kernel
BSS.

---

## 5. Outlier handling plan (§6.9 R5 compliance)

Three outlier-mitigation strategies implemented during QAT (C.P4):

1. **Per-coord adaptive activation bit allocation.** Track per-channel
   ‖x_·,j‖_∞ over calibration corpus. For top-5% channels by magnitude,
   allocate 10-bit activations instead of 8-bit (extra bit for sign +
   scale). 10-bit × 95% + 10-bit × 5% → blended 8.1-bit average.
2. **Ternary-aware channel reordering.** During QAT, permute output
   channels so that high-β rows cluster together — enables block-sparse
   compute-skip on the `W̃_ij = 0` entries.
3. **Per-block γ_x re-calibration every K iterations.** During QAT,
   γ_x is re-computed every 1000 training steps, not held fixed at
   initialization. After QAT freeze, γ_x is fixed for inference.

Novel vs published recipes:
- BitNet b1.58: static γ_x, no per-coord bit allocation, no permutation
- MatMul-Free LM: static γ_x, no outlier handling at all
- Bi-Mamba: learnable per-column α scales (different technique)

Phase D contribution: composable outlier handling on ternary LLM arch
(not published in this combination).

---

## 6. FajarOS kernel delivery layout

New files in `fajaros-x86/kernel/compute/`:
- `matmulfree.fj` — Ops 3, 8, 10 (BitLinear variants + MLGRU + absmax quant)
- `matmulfree_lut.fj` — Op 4 σ LUT + Op 5 SiLU (re-exports)
- Extends `kmatrix.fj` with `km_rmsnorm_no_gamma` (Op 2 variant)
- Extends `drivers/serial.fj` unchanged

New files in `fajaros-x86/kernel/model/`:
- `fjm_v9.fj` — Phase D `.fjm` parser (version bump from v8)
- `tfm_matmulfree.fj` — forward-pass orchestration (peer to
  `tfm_gemma3_stream.fj`)

Estimated total new kernel LOC: ~600 including error paths.

Per V31.B.P2: all matrix-dense kernels annotated `@no_vectorize` to
prevent LLVM O2 miscompile until V31.B.P1 root-cause fix lands.

---

## 7. Scale headroom summary

For each config (per FJQ_PHASE_D_CONFIG.md C.P1.4):

| Config | d | L | Max ‖x‖_∞ (post-⊛) | Max i64 temp in RMSNorm | Headroom |
|---|---|---|---|---|---|
| Mini | 256 | 6 | 23,000 | 1.4e12 | 10⁶× safety |
| Base | 384 | 12 | 34,000 | 4.4e12 | 10⁶× safety |
| Medium | 512 | 12 | 46,000 | 1.1e13 | 10⁵× safety |
| Stretch | 1024 | 24 | 92,000 | 8.7e13 | 10⁴× safety |

All configs have ≥4 orders of magnitude i64 headroom in every op.
No overflow-risk site identified. Calibration integration test
(planned for C.P5.5) will verify zero saturation events over 64-token
generation on all configs.

---

## 8. Gate for C.P2

C.P1.2 deliverable complete. C.P2 (PyTorch reference impl) can start
once all three C.P1 docs committed (this + outline + config matrix).

**C.P2.1 (3-4d) first step:** fork `ridgerchu/matmulfreellm`, verify
reproduction of published Table 1 numbers on SlimPajama-370M/100B
(sanity check before FajarQuant QAT modifications).
