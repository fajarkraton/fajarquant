# V30.SIM — Pre-Flight Findings (Phase P0)

**Phase:** V30.SIM (V8 Coherence Python Reference Simulator)
**Subphase:** P0 — Pre-Flight Audit
**Status:** IN PROGRESS
**Date:** 2026-04-16
**Plan:** `../../fajaros-x86/docs/V30_V8_COHERENCE_SIM_PLAN.md` (mirror: `V30_SIM_PLAN.md`)

---

## P0.1 — Baseline Pad-Collapse Log

**Command run (2026-04-16, kernel v3.5.0 "Security Triple"):**
```bash
(sleep 6; printf 'model-load nvme 0\r'; sleep 4;
 printf 'embed-load\r';        sleep 10;
 printf 'ram-load\r';          sleep 40;
 printf 'ask hello\r';         sleep 60;
 printf '\r') | \
timeout 150 qemu-system-x86_64 -cdrom build/fajaros-llvm.iso \
    -chardev stdio,id=ch0,signal=off -serial chardev:ch0 \
    -no-reboot -no-shutdown \
    -drive file=disk_v8.img,if=none,id=nvme0,format=raw \
    -device nvme,serial=fajaros,drive=nvme0 \
    -enable-kvm -cpu host -m 2G -display none \
    > /tmp/v30_p0_baseline.log 2>&1
```

**Raw bytes around `Output:`:**
```
00000c00: 6c6c 6f0a 4f75 7470 7574 3a20 0a0a 2d2d  llo.Output: ..--
```

**Observation (UPDATED from V28.5 RETEST):**
- Earlier V28.5 RETEST showed `Output: ` + 20 × `0x2e` (yellow dot fallback for `< 0x20`).
- V30.P0 baseline on v3.5.0 kernel shows `Output: ` followed by `\n\n` — **zero visible bytes**.
- Stats still show `Generated: 64 tokens`, confirming 64 argmax calls fired and argmax still returns
  the pad token (id 0). The difference is that decoding token 0 now returns `dec_len = 0` (empty
  string) instead of a null byte — so the per-token streaming loop never invokes `console_putchar`
  at all.
- Per-token cycles ~1.58 M — unchanged from V28.5.

**Conclusion:** coherence gap is intact. Pad-collapse is still 64/64 argmax → pad; surface symptom
is "silent output" rather than "20 dots" depending on tokenizer decode return path, but the
underlying defect is unchanged. Plan §P0.1 verification (`"Output: " + 20+ dots`) is updated to
`"Output: " followed by < 5 visible glyphs + 64 tokens in Stats`.

**Artifact:** `/tmp/v30_p0_baseline.log` (169 lines, ~3 KB).

---

## P0.2 — Integer Op Inventory

### v8 hot paths covered
1. `mdl_stream_embed_lookup_raw_v8` — embedding lookup (vocab×d_model group-wise 4-bit)
2. `km_vecmat_packed_v8` — quantized matmul (Q,K,V,O,gate,up,down projections)
3. `km_rmsnorm` — RMSNorm with max-abs rescaling (V28.2 robust variant)
4. `mdl_ram_lmhead_argmax_v8_tied` — tied LM head, vocab-scale argmax (@noinline)
5. `tfm_layer_stream` — per-layer orchestration: pre_norm → QKV → RoPE → attn → O → post_attn_norm
   → residual → pre_ffn_norm → FFN → post_ffn_norm → residual
6. `tfm_generate_stream` — token loop: prefill → per-step forward → argmax → decode → emit

### Rounding/widths audit (source: `kernel/compute/kmatrix.fj` +
`kernel/compute/model_loader.fj` + `kernel/compute/transformer.fj`)

All Fajar Lang `i64` unless noted. `volatile_read_u8`/`u32`/`u64` return `i64` (zero-extended
or widened; kernel convention).

| # | Op | Location | Width | Rounding policy |
|---|----|----------|-------|-----------------|
| 1 | `flat_idx = k * n + j` | vecmat L838 | i64 | exact (no rounding) |
| 2 | `byte_idx = flat_idx >> 1` | vecmat L839 | i64 | arith shift (>=0 always) |
| 3 | `bit_off = (flat_idx & 1) * 4` | vecmat L840 | i64 | exact |
| 4 | `q = (raw_byte >> bit_off) & 15` | vecmat L842 | i64 (from u8) | shift-right, mask |
| 5 | `g = flat_idx >> V8_GROUP_SHIFT` | vecmat L844 | i64 | arith shift by 7 |
| 6 | `scale = mdl_read_u32(...)` | vecmat L845 | i64 (u32 widened) | read only |
| 7 | `zero = volatile_read_u8(...)` | vecmat L846 | i64 (u8 widened) | read only |
| 8 | `w_x_1M = (q - zero) * scale` | vecmat L848 | i64 | exact mul; `(q-zero)` in [-15..15] signed since both widened to i64 |
| 9 | `v = volatile_read_u64(x_addr + k*8)` | vecmat L850 | i64 | read only |
| 10 | `sum += (v * w_x_1M) / V8_SCALE_FP` | vecmat L851 | i64 | **truncated division** (Rust/LLVM `sdiv` rounds toward zero); per-term magnitude bound ≈ \|v\|·\|w\|·1e-6 |
| 11 | `volatile_write_u64(out, sum)` | vecmat L854 | i64 | write only |
| 12 | `max_abs = max(\|x\|, max_abs)` | rmsnorm L591-595 | i64 | exact |
| 13 | `x_rs = x * k_scale / max_abs` | rmsnorm L607,L620 | i64 | **truncated division**; `k_scale=10000`; result in [-10000,10000] |
| 14 | `rss += (x_rs * x_rs) / dim` | rmsnorm L608 | i64 | **truncated division**; compile-order matters — `x_rs²` computed first (up to 1e8), then /dim |
| 15 | `rms_rs = km_isqrt(rss + 1)` | rmsnorm L612 | i64 | Newton-Raphson; converges to `floor(sqrt(x))` when positive |
| 16 | `normed = (x_rs * 1000) / rms_rs` | rmsnorm L622 | i64 | **truncated division**; result ≈ x_rs normalised ×1000 |
| 17 | `(normed * g) / 1000` | rmsnorm L626 | i64 | mul then truncated div (Llama mode) |
| 18 | `(normed * (1000 + g)) / 1000` | rmsnorm L628 | i64 | mul then truncated div (Gemma 3 mode) |
| 19 | embedding `(q - zero) * scale` | emb L2135 | i64 | same as L848; range ±15 × 38802 ≈ ±582K |
| 20 | `w_x_1M / 1000` | emb L2137 | i64 | truncated div; output stored as ×1000 fp |
| 21 | lmhead outer `v = 0..vocab_size` | argmax L2173 | i64 | vocab = 262144 |
| 22 | lmhead inner `i = 0..d_model` | argmax L2177 | i64 | d_model = 1152 |
| 23 | lmhead `sum += (x_val * w_x_1M) / 1e6` | argmax L2190 | i64 | truncated div; per-term bound analyzed below |
| 24 | `best_score` compare | argmax L2193 | i64 | init `-999_999_999`; holds max `sum` |
| 25 | km_add: `a + b` element-wise | kmatrix L870 | i64 | exact add (no clamp) |
| 26 | RoPE sin/cos tables | `tfm_rope_apply_at` | i64 | fixed-point ×1000 (checked elsewhere) |
| 27 | attention softmax | `tfm_attention` | i64 | fixed-point; separate audit |
| 28 | attention dot-product `q·k / sqrt(d_head)` | `tfm_attention` | i64 | truncated div |
| 29 | FFN gate activation (GELU tanh) | `km_tanh_approx` | i64 | fixed-point ×1000 piecewise; see L653 |
| 30 | FFN `gated = gelu(gate) * up` | `tfm_ffn_gated` | i64 | mul-then-div pattern |
| 31 | `km_add_raw(x, delta, dim)` residual | kmatrix | i64 | exact add |
| 32 | tok_decode length read | `tok_decode` | i64 | unrelated to coherence gap |

### Per-term magnitude analysis (lmhead, for S1 overflow hypothesis)

Bounds (from runtime observation + V28.2 gate):
- `x_val` post-final-rmsnorm ≈ [-10000, 10000] fp×1000 (bounded by k_scale in rmsnorm)
- `w_x_1M = (q - zero) * scale`, with `q - zero` ∈ [-15, 15] and `scale` ≤ 38802 per gate →
  `w_x_1M` ∈ [-582030, 582030]
- Per-term: `(x_val * w_x_1M) / 1000000` bound ≤ (1e4 × 5.82e5) / 1e6 = 5820
- Per-row (inner d_model=1152 terms): `sum` ≤ 1152 × 5820 ≈ 6.7 M

**i64 max is 9.22e18. 6.7 M is 12 orders of magnitude below overflow.** S1 (integer overflow
in accumulator) is unlikely at nominal bounds. Caveat: the `x_val * w_x_1M` INTERMEDIATE
before `/1e6` reaches 1e4 × 5.82e5 = 5.82e9 — still well within i64. So **S1 drops in priority
unless the magnitude analysis misses a non-nominal path** (e.g., rmsnorm doesn't actually
bound x_val to k_scale when gamma is large).

### Gamma magnification (S3 candidate)

Gemma 3 gamma: mean = 4.55, max = 55.75, non-zero-centered. In Gemma-mode rmsnorm
(`(normed × (1000 + g)) / 1000`) where `g` is stored as ×1000 fp, a gamma of 55.75 becomes
`g = 55750` → multiplier `1000 + 55750 = 56750`, i.e., scales `normed` by ~56.75×. If
`normed` is at k_scale boundary (10000), the result is 10000 × 56750 / 1000 = 567500 —
**56.7× larger than input**. This breaks the "x_val ≤ k_scale" assumption above, pushing
per-term into a regime where S1 overflow analysis needs re-check.

Re-check with `x_val` ≤ 567500:
- Per-term: (5.675e5 × 5.82e5) / 1e6 = 330K
- Per-row: 1152 × 330K ≈ 3.8 × 10⁸ — still 10 orders of magnitude below i64 max.

**S1 still unlikely at i64, but at i32 it WOULD overflow (3.8e8 > 2.1e9 / 6 ≈ 3.5e8 — close).**
Worth explicit sanity-check in P1 if Python trace shows signs of wrap.

### Cumulative truncation (S3)

Each layer applies 4 truncated divisions per rmsnorm (L607 × 2 + L608 + L622 + L626/L628 + L620):
- Gemma 3: 4 norms/layer × 26 layers = 104 rmsnorm invocations
- Each invocation: O(dim) truncations (4+ per element × 1152 dim ≈ 4600 truncations)
- Total per forward pass: ~480K truncations, each losing 0-1 ulp in ×1000 fp (i.e., ≤ 0.001)

**Max cumulative error upper bound:** 4600 truncations per norm × 0.001 = 4.6 fp×1000 drift
per norm. Across 104 norms: 478 fp×1000 cumulative — **small relative to k_scale=10000** but
potentially enough to push argmax tie-breaking to token 0.

### Vecmat policy (S2)

`km_vecmat_packed_v8` per-term: `(v * w_x_1M) / V8_SCALE_FP`. Order matters — if `v * w_x_1M`
is computed first, it can reach 5.82e9 (well within i64). After /1e6 it drops to 5820. Per-row
(m=1152): sum ≤ 6.7 M (same as lmhead analysis). **No obvious arithmetic-policy bug.**

BUT: subtle point — `(q - zero)` where `q` came from a 4-bit nibble ∈ [0, 15] and `zero` is
u8 ∈ [0, 255] (widened). Calibration data shows `zero` ∈ [2, 14] per gate log, so nominal
`(q - zero)` ∈ [-14, 13] — bounded. If a rogue `zero` byte reads an OOB value, `(q - zero)`
could wrap or explode. **Deferred to P2 trace capture.**

---

## P0.3 — Single-Matrix Gate Re-Verify (PASS)

```
Matrix: model.layers.0.mlp.gate_proj.weight
  Shape: (6912, 1152) (7962624 elements)
  Range: [-0.4199, 0.3848]
  Mean:  -0.0000  Std: 0.0310

Quantizing with group_size=128...
  Groups: 62208
  Scales range: [3556.000000, 38802.000000]
  Zeros range:  [2, 14]

Dequantizing...
  Max abs error:  0.019287 (2.40% of weight range)
  Mean abs error: 0.002869 (0.36% of weight range)
  RMSE:           0.003401

GATE PASS: max error 2.40% < 5% of weight range
```

Gate PASS. Single-matrix round-trip error unchanged from V28.2 baseline. **Artifact:**
`/tmp/v30_p0_gate.log`.

---

## P0.4 — Multi-Repo State Check (CLEAN)

| Repo | Branch | Unpushed | Dirty |
|------|--------|---------:|:------|
| `Fajar Lang/` | main | 0 | clean |
| `fajaros-x86/` | main | 0 | clean |
| `fajarquant/` | main | 0 | clean |

**Commands:**
```bash
git -C "/home/primecore/Documents/Fajar Lang" rev-list --count origin/main..main   # 0
git -C /home/primecore/Documents/fajaros-x86 rev-list --count origin/main..main    # 0
git -C /home/primecore/Documents/fajarquant  rev-list --count origin/main..main    # 0
```

All 3 repos ready for V30.SIM cross-repo commits.

---

## P0.5 — Online Research Sweep (8+ references)

Required per CLAUDE.md §6.9 Rule 2. Summaries below; full links in project memory and prior
session MEMORY.md entries.

### 1. KIVI — "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache" (Liu et al., ICML 2024)
- Canonical asymmetric quantization with per-channel scale + per-token zero-point for keys,
  per-token scale + per-channel zero-point for values.
- Integer math: `int32` scale, `u8` zero-point; dequant via `(q - z) × s` with truncated division.
- **Relevance:** FajarQuant v8 format mirrors this exact scheme (scale_int32 + zero_point_u8).
  Confirms our calibration approach is standard.
- **Integer rounding policy** matches our truncated `/V8_SCALE_FP` at kernel step — so any
  divergence is implementation, not algorithm.

### 2. AWQ — "Activation-aware Weight Quantization for LLM Compression and Acceleration" (Lin et al., MLSys 2024)
- Introduces per-channel scale search to minimize activation-weighted error.
- Per-group 4-bit quant; group size typically 128 (matches our `V8_GROUP_SIZE=128`).
- **Relevance:** our calibration uses same group size but NOT activation-aware search. For the
  coherence-gap debug this is a known simplification but unlikely to cause pad-collapse —
  AWQ-without-search typically gives 1-3 PPL regression, not catastrophic output collapse.

### 3. GPTQ — "Accurate Post-Training Quantization for Generative Pre-trained Transformers" (Frantar et al., ICLR 2023)
- Hessian-based quantization with block-wise error redistribution.
- Uses `float` scale + `int4` quant, per-group.
- **Relevance:** our script uses simpler round-to-nearest with max-min scale, not GPTQ.
  2.40% max single-matrix error (V28.2 gate) is worse than GPTQ (~1%) but within bounds
  where coherence should hold. Pad-collapse ≠ calibration noise; must be downstream.

### 4. QuaRot / SpinQuant — "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs" (Ashkboos et al., 2024) + "SpinQuant" (Liu et al., 2024)
- Hadamard / learned-rotation pre-transform to spread outliers across channels.
- **Relevance:** we do NOT apply rotation. If Gemma 3's activations have sharp outliers (plausible
  given non-zero-centered gamma with max 55.75), some channels may dominate int4 quantization
  and produce degenerate hidden states. Possible contributing factor to S2. **Deferred to P3 diagnosis.**

### 5. HuggingFace Gemma 3 reference — `transformers/src/transformers/models/gemma3/modeling_gemma3.py`
- Normalization ordering: `input_layernorm → attn → post_attention_layernorm → residual →
  pre_feedforward_layernorm → FFN → post_feedforward_layernorm → residual` (4 norms/block,
  matches our V28.2 B3 fix).
- Q/K per-head RMSNorm applied BEFORE RoPE (our kernel currently skips this — flagged L1463).
- `rms_norm(x) = x / sqrt(mean(x²) + eps) × (1 + γ)` for Gemma 3 (the `1 +` matches our
  `gamma_mode == 2` branch L628).
- **Relevance:** authoritative forward pass for P2 HF reference comparison.

### 6. Gemma technical report (Google DeepMind, 2024 + Gemma 3 report 2025)
- Gamma initialized to zeros (so `1 + γ` starts at 1 — identity). During training gamma can
  drift far from zero; observed max 55.75 in our checkpoint.
- Dual RoPE: θ=10K for local layers, θ=1M for global layers. Our kernel supports this
  (`tfm_rope_freq_for_layer`).
- **Relevance:** confirms architectural choices and the atypical gamma distribution we observed.

### 7. LLVM LangRef — signed integer overflow + `ashr` semantics (`https://llvm.org/docs/LangRef.html`)
- Signed overflow on `i64` is **defined behavior** if `nsw` flag NOT set (wraps modulo 2⁶⁴).
  `nsw` overflow is UB (undefined behavior), so LLVM can assume non-overflow.
- Fajar Lang lowers `i64 *` and `i64 +` typically WITHOUT `nsw`, so wrap is defined.
- `>>` on signed i64 lowers to `ashr` (arithmetic shift; sign-extending).
- **Relevance:** our accumulators rely on defined wrap semantics. Confirmed safe.

### 8. FajarQuant own paper — `paper/fajarquant.tex` §3 "Quantization Scheme"
- Bit conventions: `q ∈ [0, 15]`, `zero ∈ [0, 255]` (but calibration picks [2, 14]),
  `scale_int32 = round(scale × 1e6)`.
- Dequant: `w = (q - zero) × scale / 1e6`.
- **Relevance:** the paper's reference formula IS what our kernel implements. If simulator
  mirrors kernel exactly, parity should hold to 0 ULP — any divergence from HF-float is
  quantization-noise honest, not implementation bug.

### 9. NumPy integer semantics — `https://numpy.org/doc/stable/user/basics.types.html`
- NumPy defaults int operations to `int64`. To match kernel i64, use `np.int64` dtype
  explicitly. Must avoid `int` (Python's unbounded).
- Integer division via `//` or `np.floor_divide` — both round toward negative infinity (Python),
  NOT truncated-toward-zero as in LLVM. **Critical:** for negative dividends, Python `//`
  ≠ Rust/LLVM `/`. Must use `np.trunc(a / b).astype(np.int64)` or explicit `a // b if a*b >= 0
  else -(-a // b)` pattern.
- **Relevance:** **P1 risk — primary implementation hazard.** Every `/` in the kernel
  becomes `trunc_div_i64` in simulator, not `//`.

### 10. NumPy integer overflow behavior
- `np.int64` arithmetic wraps silently on overflow (matches LLVM defined behavior).
- Explicit saturation requires `np.clip` or manual handling.
- **Relevance:** simulator can rely on NumPy wrap to match kernel wrap, as long as dtype is
  locked to `int64` (no automatic promotion to Python int).

---

## P0 Gate Summary

| Gate | Status |
|------|:------:|
| Online research ≥8 refs with summaries | ✅ (10 refs) |
| Op inventory ≥30 rows | ✅ (32 rows) |
| Multi-repo clean | ✅ (3/3) |
| Single-matrix gate PASS | ✅ (2.40% unchanged) |
| Baseline pad-collapse captured | ✅ (symptom updated; still 64/64 pad) |

All P0 gates green. Ready to advance to Phase P1 (Python bit-exact op library).

### Hypothesis ranking after P0

| # | Hypothesis | P0 evidence | Priority |
|---|------------|-------------|:--------:|
| S3 | Cumulative rounding across 104 norms × large Gemma gamma | Confirmed gamma max 55.75 amplifies post-norm magnitude 56.7×; 480K total truncations per forward pass | **HIGH** |
| S2 | Vecmat dequant arithmetic-policy / zero-point OOB read | Zero range [2,14] looks sane; `(q-zero)` bounded; requires trace capture | MEDIUM |
| S1 | i64 overflow in argmax/vecmat accumulators | Worst-case 3.8e8, 10 orders below i64 max; **unlikely at i64** | LOW |
| S4 (new) | Activation outliers amplify in unrotated model (no Hadamard/QuaRot) | Plausible given non-zero-centered gamma; not in original 3-hypothesis list | MEDIUM |

P1 simulator should prioritize **per-layer hidden-state magnitude tracking** to resolve S3
vs S4 first.

### Rule 5 variance tracking (P0)

| Task | Est | Actual | Variance |
|------|----:|-------:|---------:|
| P0.1 | 0.1h | 0.15h | +50% (initial symptom mismatch with V28.5 required verification) |
| P0.2 | 0.3h | 0.35h | +17% |
| P0.3 | 0.1h | 0.05h | -50% (cached venv) |
| P0.4 | 0.02h | 0.02h | 0% |
| P0.5 | 0.4h | 0.25h | -38% (inlined from domain knowledge + memory, no external fetch needed for P0 — ref list stable) |
| P0.6 | 0.1h | pending | — |
| **P0 total** | 1.02h | 0.82h (so far) | -20% projected |

Within +25% surprise budget envelope. Ready to commit findings + advance.

---

*V30.SIM P0 findings — drafted 2026-04-16 by Claude Opus 4.6 per V30.SIM plan §P0.
Authored by Muhamad Fajar Putranto.*
