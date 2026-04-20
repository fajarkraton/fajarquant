# FajarQuant Phase D — Config Matrix (C.P1.4)

> **Date:** 2026-04-21 | **Depends on:** `FJQ_PHASE_D_ARCH.md` (MatMul-Free primary) + `FJQ_PHASE_D_OPS.md` (op inventory) | **Hardware:** RTX 4090 16 GB (training), consumer x86-64 + FajarOS Nova (inference)

## 0. Parameter accounting formula

Per V31 arch: MLGRU token mixer (4 BitLinear: f, c, g, o) + GLU
channel mixer (3 BitLinear: g, u, d with FFN ratio 8/3) + 2 RMSNorm
per layer (no learnable γ per V28.2/Phase-D variant) + untied embedding
and LM head (both BitLinear for consistency).

```
params_per_layer = 4·d² + 3·(d × 8d/3) = 4d² + 8d² = 12d²   (ternary)
params_total     = L · 12d² + 2·V·d                         (embed + LM head)
                 ≈ 12·L·d² + 2·V·d
```

**Precision accounting:** all `12d²` matrix params are 2-bit ternary
(packed 4-per-byte in `.fjm`). Embedding table + LM head tensor are
also ternary (MatMul-Free convention §3.1). Only 60 × L scalars per
network are FP (per-BitLinear β + γ_x + per-RMSNorm ε) — negligible
at any scale.

## 1. Tokenizer decision gate

Two candidates retained per `FJQ_PHASE_D_PAPER_OUTLINE.md`:

| Tokenizer | V | Rationale | Deferred to |
|---|---|---|---|
| Mistral v3 | 32,768 | Matches MatMul-Free LLM paper Table 1 for direct repro-check at Stretch config | — |
| SmolLM 49K | 49,152 | Already shipped in FajarOS Nova V26+; no .fjt export work needed | — |
| Custom BPE on SlimPajama | 16,384 / 8,192 | Smaller embedding footprint for Mini/Base | — |

**Default for this config matrix:** V = 32,768 (Mistral v3) — keeps
comparability with published MatMul-Free numbers. Decision locked in
C.P3.1 after a micro-trained tokenizer eval; this matrix uses V=32K
for param/FLOP computation. Swapping V later re-scales only the
embedding/LM-head term `2·V·d`, not the core `12·L·d²`.

## 2. Config matrix (final)

| Config | d | L | V | Params (ternary) | Params (FP overhead) | **Total params** | FFN hidden (8d/3) | Context | Batch × seq | Purpose |
|---|---|---|---|---|---|---|---|---|---|---|
| **Mini** | 256 | 6 | 32,768 | 4.72 M | 16.78 M (embed + head) | **21.5 M** | 683 | 1,024 | 256 × 1024 | Fast QAT sweeps (<1 day wall clock). Debugging ablations. |
| **Base** | 384 | 12 | 32,768 | 21.23 M | 25.17 M (embed + head) | **46.4 M** | 1,024 | 2,048 | 128 × 2048 | **Primary C.P4.1.** First FP-shadow vs QAT comparison. |
| **Medium** | 512 | 12 | 32,768 | 37.75 M | 33.55 M (embed + head) | **71.3 M** | 1,365 | 2,048 | 96 × 2048 | Scaling sanity + first vs-Transformer++ table-2 point. |
| **Stretch** | 1,024 | 24 | 32,768 | 301.99 M | 67.11 M (embed + head) | **369.1 M** | 2,731 | 2,048 | 32 × 2048 | Direct repro of Zhu et al. 370M row; 15B-token training; gated on Medium gate. |

**Ternary-heavy fraction:** Mini 22% / Base 46% / Medium 53% / Stretch
82%. At Stretch the matrix weights dominate, matching mainstream
published ternary LLM profiles.

**Non-ternary FP parameters at all configs:** constant `60·L ≈ 720`
scalars (well under 3 KB). Zero training/inference FLOP impact.

## 3. Compute budget

### 3.1 Forward FLOPs per token (approximate, dominant terms)

```
flops_per_token ≈ 24·L·d² + 2·V·d   (ternary signed-adds)
```

| Config | Ternary adds / token | LM head adds / token | **Total / token** |
|---|---|---|---|
| Mini | 2.36 M | 16.78 M | **19.1 M** |
| Base | 21.23 M | 25.17 M | **46.4 M** |
| Medium | 37.75 M | 33.55 M | **71.3 M** |
| Stretch | 301.99 M | 67.11 M | **369.1 M** |

Parameter count = forward ops/token (exact for dense ternary arch, as
expected — no attention quadratic in context).

### 3.2 Training token budgets + RTX 4090 wall-clock estimate

QAT training uses FP16 shadow weights + fake-quantize forward + STE
backward (standard ternary recipe). Real wall-clock on RTX 4090 scales
as 3-5× the theoretical FLOP estimate due to MFU ~20-30% on
irregular-structured ops + LUT non-compute overhead.

FLOP budget per training run: `6 × N × D` (industry rule-of-thumb for
causal LM) where N = total params, D = training tokens.

| Config | N | D (tokens) | Training FLOPs (6·N·D) | RTX 4090 (peak FP16 83 TFLOPs, MFU 25%) |
|---|---|---|---|---|
| Mini | 21.5 M | 500 M | 6.45 × 10¹³ | **~1 h** |
| Base | 46.4 M | 1 B | 2.78 × 10¹⁴ | **~4 h** |
| Medium | 71.3 M | 2 B | 8.56 × 10¹⁴ | **~11 h (~0.5 day)** |
| Stretch | 369.1 M | 15 B | 3.32 × 10¹⁶ | **~17 days** |

**Stretch is a commitment.** 17-day RTX 4090 run is the biggest
training budget in Phase D. Per §6.9 R6, Stretch training is gated on
Medium clearing the canonical PPL gate (§6.9 R1) — if Medium fails vs
Transformer++ by >3 PPL, abort Stretch and pivot to Candidate A
(RWKV-7) ablation per `FJQ_PHASE_D_ARCH.md` §3 triggers.

### 3.3 Inference memory footprint (FajarOS Nova)

Packed ternary weights + FP constants only:

| Config | Ternary pack | Embed FP | Per-token FP scratch | **Total RAM** |
|---|---|---|---|---|
| Mini | 1.18 MB | 16.78 MB | 4 KB | **18.0 MB** |
| Base | 5.31 MB | 25.17 MB | 12 KB | **30.5 MB** |
| Medium | 9.44 MB | 33.55 MB | 16 KB | **43.0 MB** |
| Stretch | 75.5 MB | 67.11 MB | 64 KB | **142.7 MB** |

Even Stretch fits in 256 MB RAM (the FajarOS Nova default). The
embedding table is FP16 (2 × V × d bytes = 50 MB at Stretch with
V=32K, d=1024 — counted above). LM head is ternary (same as body),
so weight-tied embed+head would halve the FP overhead.

### 3.4 Training memory budget (RTX 4090, 16 GB)

FP16 shadow weights + Adam optimizer (3× params for m, v states) +
activations:

| Config | Shadow (FP16) | Optimizer (FP32 m+v) | Activations (batch × seq) | **Peak VRAM** |
|---|---|---|---|---|
| Mini | 43 MB | 258 MB | ~2 GB | **~2.3 GB** |
| Base | 93 MB | 557 MB | ~6 GB | **~6.7 GB** |
| Medium | 143 MB | 856 MB | ~8 GB | **~9.0 GB** |
| Stretch | 738 MB | 4,429 MB | ~8 GB | **~13.2 GB** |

**Stretch fits** with batch=32 seq=2048 on 16 GB. Tight; would need
gradient checkpointing if seq extends to 4K. Medium is the
recommended upper-bound for parallel hyperparameter sweeps (leaves
~7 GB headroom for multiple concurrent runs).

## 4. Training schedule per config

All configs share the same lr schedule + optimizer, differ only in
total tokens + batch size.

- **Optimizer:** AdamW β₁=0.9, β₂=0.95, ε=1e-8, weight_decay=0.1
- **Warmup:** 2000 steps linear
- **Peak lr:** 2e-3 for Mini/Base, 1e-3 for Medium, 5e-4 for Stretch
  (paper §4.3 ceiling was 2e-2 — instability above that; we stay 10×
  safer)
- **Decay:** cosine to 10% of peak
- **Clip:** global norm 1.0
- **QAT schedule:** FP shadow training for 80% of tokens, then
  activation quantization enabled for remaining 20% (paper §3 default)
- **γ_x re-calibration:** every 1000 steps during QAT phase (§6.9 R5
  contribution per `FJQ_PHASE_D_OPS.md` §5)

## 5. Go/no-go gates per config (mechanical per §6.8 R6)

### 5.1 Mini gate (after ~1 h training)

- **Gate file:** `docs/FJQ_PHASE_D_MINI_GATE.md` (C.P4.1 deliverable)
- **PASS condition:** val loss < 4.0 on 1M held-out tokens from
  SlimPajama val split
- **If FAIL:** lr sweep + batch-size ablation; debug QAT harness;
  do NOT proceed to Base
- **Gate check:** `./scripts/eval_harness.py --config mini --ckpt
  ckpt/mini_last.pt --mode val_loss`

### 5.2 Base gate (after ~4 h training)

- **Gate file:** `docs/FJQ_PHASE_D_BASE_GATE.md`
- **PASS conditions (all three):**
  1. Wikitext-103 PPL ≤ 25 (comparison: Transformer++ ~24 at 46M)
  2. LAMBADA-o acc ≥ 38%
  3. No saturation events in kernel FJTRACE (§6.9 R5 verification)
- **If FAIL:** diagnose; either loop back on QAT recipe or trigger
  Candidate A ablation per `FJQ_PHASE_D_ARCH.md` §3

### 5.3 Medium gate (after ~11 h training)

- **Gate file:** `docs/FJQ_PHASE_D_MEDIUM_GATE.md`
- **PASS conditions (all three):**
  1. Wikitext-103 PPL gap vs Transformer++ at matched params ≤ 3
  2. MMLU 5-shot ≥ 22%
  3. FajarOS Nova E2E boot + ask yields coherent output (no pad-collapse)
     on 3 test prompts
- **If FAIL:** STOP. Do not proceed to Stretch. Paper becomes a
  research-artifact submission at Medium scale only; Stretch pivots to
  a Phase D+1 follow-up.

### 5.4 Stretch gate (after ~17 days training)

- **Gate file:** `docs/FJQ_PHASE_D_STRETCH_GATE.md`
- **PASS conditions (all three):**
  1. Match Zhu et al. Table 1 numbers at 370M within ±1 avg-point on
     6 zero-shot tasks (repro check — §6.9 R3 baseline parity)
  2. Wikitext-103 PPL (new metric not in original paper) within ±2
     of FP16 Transformer++ at matched params
  3. Kernel-path E2E on FajarOS Nova: no saturation, no collapse,
     tok/s ≥ 1 on consumer x86-64
- **If FAIL:** paper submits at Medium with Stretch marked as "future
  work".

## 6. Compute cost total

Optimistic path (all 4 configs, no retries, no Candidate A ablation):

| Config | Wall-clock | Cumulative |
|---|---|---|
| Mini | 1 h | 1 h |
| Base | 4 h | 5 h |
| Medium | 11 h | 16 h |
| Stretch | 17 days | **~17.7 days** |

Pessimistic path (includes Candidate A ablation triggered at Medium
gate fail, retries, hyperparameter searches):
- +3 lr/batch sweeps at Base (×2h each) = +6 h
- +Candidate A Stretch ablation = +17 days
- **Total pessimistic:** ~35 days RTX 4090 wall-clock

## 7. Electricity + budget sanity check

RTX 4090 draws ~450W under full load. At $0.15/kWh:
- Optimistic: 17.7 days × 24 h × 450 W × 1 kWh/1000W × $0.15 = **$28.70**
- Pessimistic: 35 days × same = **$56.70**

Solo research budget well within typical month-of-experimentation
envelope. No GPU rental required; all local RTX 4090 per V24
inventory.

## 8. Dependencies on C.P2-C.P5

All config matrix rows depend on:
- **C.P2** (PyTorch reference): training harness works at least at Mini
- **C.P3** (dataset pipeline): tokenized SlimPajama shards ≥15B tokens
  staged and streaming
- **C.P5** (kernel integration): `.fjm` v9 parser + `tfm_matmulfree.fj`
  for the §5.2/5.3/5.4 "FajarOS Nova E2E" gate checks

Parallel-startable: Mini configs can begin C.P4.1 training as soon as
C.P2 green-bars on synthetic corpus; Stretch training is sequenced
last and depends on all upstream gates.

## 9. Unblocks

With C.P1.2 + C.P1.3 + C.P1.4 now all committed, **Phase C.P1 is
COMPLETE**. The next phase (C.P2 PyTorch reference implementation)
can begin — expect a separate planning session to fork
`ridgerchu/matmulfreellm`, set up the FajarQuant QAT harness, and
define repro acceptance gates.
