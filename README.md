# FajarQuant — Quantization Research for Compiler-Verified LLM Systems

> **Two research arms, one umbrella.**
> **Phase D IntLLM** trains a 1.58-bit MatMul-Free LLM family (Mini/Base/Medium 21M-74M params), validates a 3-scale calibrated training-gate chain with monotonically widening margins (0.12 → 0.21 → 0.28 nat), and deploys end-to-end inside the OS kernel via FajarOS Nova IntLLM kernel-path. **v3.1 KV Cache Quant** (mature, paper artifact) profiles each KV head and routes to the optimal strategy, matching or beating the best fixed method in 7 of 9 evaluation cells with zero catastrophic failures.

[![Crates.io](https://img.shields.io/crates/v/fajarquant?color=blue)](https://crates.io/crates/fajarquant)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](LICENSE)
[![Compiler](https://img.shields.io/badge/compiler-Fajar_Lang_v31.0.0-blueviolet)](https://github.com/fajarkraton/fajar-lang)
[![Phase D](https://img.shields.io/badge/Phase_D-Mini%2BBase%2BMedium_PASS-success)](docs/FJQ_PHASE_D_PRODUCTION_PLAN.md)
[![Track B](https://img.shields.io/badge/Track_B-6--layer_interruption--safety-success)](python/phase_d/intllm/train.py)
[![Paper](https://img.shields.io/badge/paper-v3.1_KV_quant-success)](paper/fajarquant.pdf)
[![Verify](https://img.shields.io/badge/claims-12%2F13_verified-brightgreen)](scripts/verify_intllm_tables.py)
[![FajarOS](https://img.shields.io/badge/FajarOS-Nova_v3.9.0_IntLLM_kernel--path-orange)](https://github.com/fajarkraton/fajaros-x86)
[![Made in Indonesia](https://img.shields.io/badge/Made_in-Indonesia-red)]()

FajarQuant is a Rust + Python research repository housing **two distinct research lines** in LLM quantization, both with compile-time `@kernel`/`@device` safety guarantees through the Fajar Lang compiler. The repo started as a KV cache quantization library (v0.1.0–v0.3.0, paper v3.1) and expanded with the Phase D IntLLM training-quantization research line in v0.4.0 (this release).

---

## Two Research Arms

### Arm A — Phase D IntLLM (training-time quant, v0.4.0, primary going forward)

Train ternary {-1, 0, +1} weights end-to-end via MatMul-Free LLM architecture (HGRNBitForCausalLM), validate scaling chain across 3 calibrated gates, deploy entirely inside `@kernel` context with no heap allocation.

- **Models:** intllm-mini (21.5M), intllm-base (46.4M), intllm-medium (74.5M)
- **Training:** 491M / 982M / 1.819B tokens (Chinchilla 22.8 / 21.16 / 24.4 tok/p)
- **3 gates PASS** with monotonically widening margins:
  - Mini v2 val_loss 4.38 PPL 80.0 (gate < 4.5, margin 0.12 nat)
  - Base c.1 val_loss 3.99 PPL 54.1 (gate < 4.2, margin 0.21 nat)
  - Medium c.1 val_loss 3.72 PPL 41.3 (gate < 4.0, margin **0.28 nat**)
- **Track B 5+1-layer interruption-safety:** ckpt_every / --resume / StepWatchdog / HF timeout+retry / regression gate / nohup line-buffering. Validated end-to-end during a real laptop-shutdown event mid-Medium training.
- **In-kernel deployment** via [FajarOS Nova v3.9.0 IntLLM Kernel Path](https://github.com/fajarkraton/fajaros-x86/releases/tag/v3.9.0).

### Arm B — v3.1 KV Cache Quantization (mature, paper artifact)

Profile each KV head's statistical properties at calibration time, route to optimal quantization strategy. First systematic cross-architecture perplexity evaluation (3 models × 3 bit widths = 9 cells) using the canonical R-α.1 model-surgery protocol. Paper at `paper/fajarquant.pdf` (MLSys 2027 target).

See [Arm B section below](#v31-kv-cache-quant--headline-results-arm-b) for full results.

## Phase D IntLLM — Scaling Chain Results (Arm A)

3-row monotonic LM-modeling improvement on 8-task lm-eval v0.4.11 (real bench, no limit, RTX 4090 Laptop):

| Metric                      | Mini 21M | Base 46M | Medium 74M | Δ Mini→Med |
|-----------------------------|---------:|---------:|-----------:|-----------:|
| wikitext word_PPL           |   342.98 |   201.09 | **138.36** |       −60% |
| wikitext bits_per_byte      |    1.575 |    1.431 |  **1.330** |       −16% |
| lambada_openai PPL          |  51,121  |  16,729  | **5,277**  |       −90% |
| **lambada_openai acc**      |    0.001 |    0.007 |  **0.023** |    **16×** |
| arc_easy acc                |    0.306 |    0.319 |  **0.341** |     +0.035 |
| openbookqa acc              |    0.110 |    0.128 |  **0.130** |     +0.020 |

**Pure LM modeling (wikitext, lambada):** clean monotonic scaling. Lambada PPL drops by an order of magnitude per scale step. Lambada accuracy scales 16× from Mini to Medium.

**Multi-choice reasoning (hellaswag, piqa, winogrande, arc_*, openbookqa):** noisy at sub-100M scale, mostly within ±1-2 stderr. Per Chinchilla literature, sub-100M models cannot meaningfully beat random on these tasks; expectation-aligned. Phase D contribution is NOT "win on benchmark X" — model is too small for that.

### Phase D Contribution

The actual contribution of Phase D is three-fold:

1. **Compiler/kernel-path enabling LLM inference inside `@kernel` context** with no heap allocation (FajarOS Nova v3.9.0 IntLLM Kernel Path).
2. **Track B 5+1 layer interruption-safety hardening** validated end-to-end during a real laptop-shutdown event mid-Medium training (no progress lost beyond the worst-case 36-min checkpoint window).
3. **Calibrated training-gate methodology** (Mini < 4.5 / Base < 4.2 / Medium < 4.0 / Stretch < 3.7) that all three calibrated scales pass with monotonically widening margins (0.12 → 0.21 → 0.28 nat).

Bench numbers above verify the scaling validation but are not the headline claim.

### Phase D Reproducibility

```bash
make verify-intllm-tables          # 12/13 paper claims verified (--strict)
make bench-canonical-real TAG=mini # 8-task lm-eval on mini_final.pt (~10 min)
make bench-canonical-real TAG=base
make bench-canonical-real TAG=medium
make test-train-watchdog            # Track B 5+1 layer gate (24 tests + signal delivery)
make test-intllm-fp16-parity       # fp16-vs-ternary parity (37 hooks, IntLLM differentiator)
```

See `docs/FJQ_PHASE_D_PRODUCTION_PLAN.md` for the 9-week plan + `docs/FJQ_PHASE_D_GATE_CALIBRATION.md` for evidence-backed calibrated gate thresholds.

---

## v3.1 KV Cache Quant — Headline Results (Arm B)

| Model | Arch | Bits | FP16 | **FQ v3.1** | KIVI | TQ outlier | Strategy |
|-------|------|-----:|-----:|------------:|-----:|-----------:|----------|
| Gemma 4 E2B | MQA (1 KV head) | 2 | 28.11 | **39.91** | 480.66 | **39.91** | PPL-guided → Path C |
| Gemma 4 E2B | MQA (1 KV head) | 3 | 28.11 | **16.51** | 21.69 | 26.40 | 17A + 13B |
| Gemma 4 E2B | MQA (1 KV head) | 4 | 28.11 | 28.13 | 35.11 | **27.52** | 17A + 13B |
| Mistral 7B | GQA (8 KV heads) | 2 | 5.67 | **24.95** | **24.95** | 163.96 | PPL-guided → all-A |
| Mistral 7B | GQA (8 KV heads) | 3 | 5.67 | 6.32 | **6.00** | 9.44 | 278A + 234B |
| Mistral 7B | GQA (8 KV heads) | 4 | 5.67 | **5.73** | **5.73** | 5.88 | all-A |
| Qwen2-7B | GQA (4 KV heads) | 2 | 7.69 | **18.44** | 28.53 | 75.15 | PPL-guided |
| Qwen2-7B | GQA (4 KV heads) | 3 | 7.69 | 8.15 | **8.14** | 8.38 | 222A + 2B |
| Qwen2-7B | GQA (4 KV heads) | 4 | 7.69 | **7.78** | **7.78** | 7.82 | all-A |

**Score: 2 wins, 5 ties, 2 losses** (7 of 9 match-or-beat best fixed method).
Bold = best quantized method in that cell. Protocol: R-α.1 canonical
model surgery, WikiText-2 test set. See `paper/fajarquant.pdf` §6.

### Key Findings

- **Architecture dependence is the dominant signal.** KIVI wins 6/9 cells
  on its own but fails catastrophically on Gemma MQA at 2-bit
  (PPL 480 vs FP16 28). No fixed method is safe across architectures.
- **Two architecture-specific optima that no fixed method finds:**
  - **MQA at 3-bit:** PCA rotation beats KIVI by **−24%** on Gemma
    (PPL 16.51 vs 21.69). Single KV head concentrates information,
    making PCA's decorrelation highly effective.
  - **GQA at 2-bit:** PPL-guided selection beats KIVI by **−35%** on
    Qwen2 (PPL 18.44 vs 28.53) via novel per-head mixture.
- **Zero catastrophic failures.** v3.0's MSE-only selection produced
  2 failures (Gemma 2-bit PPL 171, Mistral 2-bit PPL 79). v3.1's
  PPL-guided fallback eliminates both.

FajarQuant is **part of the Fajar Lang ecosystem**: it ships as both a
standalone Rust crate (this repo) and a native implementation in
[Fajar Lang](https://github.com/fajarkraton/fajar-lang) with `@device`
context, SE023 type safety, and AVX2 SIMD (1.9× Hadamard, 1.6× fused
kernel, 5× vs Python reference).

---

## Four Innovations

### 1. Adaptive Per-Head Method Selection — NEW IN v3.1 (`adaptive.rs`)

Profiles each KV head's statistical properties (variance per channel,
kurtosis, SVD ratio) at calibration time and routes to the optimal
strategy: Path A (KIVI per-channel), Path B (per-head PCA rotation),
or Path C (outlier-aware TQ). At 2-bit, falls back to PPL-guided
selection when MSE and PPL disagree on the best strategy.

**Result:** Discovers two architecture-specific optima: PCA rotation
on MQA 3-bit (−24% vs KIVI), PPL-guided mixture on GQA 2-bit (−35% vs KIVI).

### 2. Outlier-Aware Calibrated PCA (`turboquant.rs` v2)

Replaces TurboQuant's random rotation with per-head PCA, calibrated
once on representative data. Concentrates variance in fewer dimensions
and handles outliers explicitly (top-K channel preservation).

**Result:** 63–81% MSE improvement over v1 at 2-bit; 4–6% over random
rotation on Gemma 4 E2B; peak 88% on synthetic d=128, b=3.

### 3. Fused Quantized Attention (`fused_attention.rs`)

Computes attention directly on quantized KV vectors via codebook dot
products, skipping the dequantize buffer entirely.

**Result:** 524,288× memory reduction at 16K context (33.5 GB → 64 B
per head). Zero allocation in the hot path.

### 4. Hierarchical Multi-Resolution Bit Allocation (`hierarchical.rs`)

Allocates more bits to recent tokens (which dominate attention scores)
and fewer to distant ones via exponential decay.

**Result:** 48.7% bit savings at 10K context, 55.7% at 16K, with
negligible perplexity loss.

---

## Quick Start

```rust
use fajarquant::adaptive::{select_strategy, StrategyPath};
use fajarquant::fused_attention::QuantizedKVCache;
use fajarquant::hierarchical::BitSchedule;

// v3.1 adaptive: profile head stats, select optimal path
let stats = profile_head(&kv_head_data);
let path: StrategyPath = select_strategy(stats, bits=3, n_kv_heads=8);
// path ∈ { A = KIVI per-channel, B = per-head PCA, C = TQ outlier }

// Quantized KV cache with fused attention
let mut kv_cache = QuantizedKVCache::new(128, 2);
// ... insert keys/values, attention computes on quantized form

// Hierarchical bit allocation (8 → 2 bits as tokens age)
let schedule = BitSchedule::exponential_decay(8, 2, 1024);
```

---

## Repo Layout

```
fajarquant/
├── src/
│   ├── lib.rs                 # Public API
│   ├── turboquant.rs          # v2 outlier-aware calibrated PCA baseline
│   ├── adaptive.rs            # v3.1 adaptive per-head selector
│   ├── fused_attention.rs     # Fused attention on quantized KV
│   ├── hierarchical.rs        # Multi-resolution bit allocation
│   └── kivi.rs                # KIVI baseline (for comparison)
├── benches/
│   ├── noise_floor.rs         # Noise floor measurement
│   └── quant_latency.rs       # Quantization latency benchmark
├── examples/                  # 6 .fj demos (run via Fajar Lang)
│   ├── adaptive_demo.fj
│   ├── benchmark.fj
│   ├── fused_demo.fj
│   ├── hierarchical_demo.fj
│   ├── kv_cache.fj
│   └── paper_benchmark.fj
├── paper/                     # MLSys 2027 paper artifacts
│   ├── fajarquant.tex         # arXiv version (11 pages)
│   ├── fajarquant.pdf
│   ├── fajarquant_mlsys.tex   # MLSys 2027 formatted (10 pages)
│   ├── fajarquant_mlsys.pdf
│   ├── SUBMISSION.md          # venue decision (arXiv → MLSys 2027)
│   ├── references.bib
│   └── Makefile
├── data/
│   └── kv_cache/              # 50 prompts from WikiText-2
│       ├── prompt_000/...prompt_049/
│       ├── perplexity_v3.1_*.json       # v3.1 primary results
│       ├── comparison_results.json
│       ├── ablation_results.json
│       └── metadata.json
├── scripts/                   # Reproducibility (Python)
│   ├── extract_kv_cache.py    # Extract Gemma 4 E2B / Mistral / Qwen2 KV cache
│   ├── eval_perplexity_v3.py  # WikiText-2 perplexity eval (R-α.1 protocol)
│   ├── calibrate_fq_v3.py     # Per-head PCA calibration
│   ├── ppl_guided_select.py   # PPL-guided strategy selection at 2-bit
│   ├── profile_kv_heads.py    # Statistical profiling of KV heads
│   └── strategy_selector.py   # Architecture-agnostic decision tree
└── reproduce.sh               # One-script reproduction (4 modes)
```

---

## Reproducing the Paper

**One script, four modes:**

```bash
./reproduce.sh --verify    # Quick verify (28 claims, ~2 minutes)
./reproduce.sh --smoke     # Smoke test (1 model, ~10 min on RTX 4090)
./reproduce.sh --full      # Full reproduction (3 models × 3 bits, ~3 hours)
./reproduce.sh --fallback  # Public-model fallback (SmolLM-135M, no HF login)
```

**Prerequisites:**

- Rust 1.87+
- Python 3.10+ with `transformers`, `torch`, `numpy`
- Hugging Face access to `google/gemma-4-e2b`, `mistralai/Mistral-7B-v0.1`,
  `Qwen/Qwen2-7B` (`--fallback` mode avoids this)
- ~16 GB GPU VRAM (RTX 4090 verified)

Results land in `data/kv_cache/perplexity_v3.1_*.json`. Paper tables are
auto-regenerated via `paper/Makefile`. CI verifies all 28 paper claims
on every push (`.github/workflows/paper-reproduce-smoke.yml`).

---

## Compile-Time `@kernel` Safety

FajarQuant is implemented in pure Rust, but its sister implementation in
[FajarOS Nova](https://github.com/fajarkraton/fajaros-x86) runs in
`@kernel` context with the
[Fajar Lang](https://github.com/fajarkraton/fajar-lang) compiler enforcing:

- **No heap allocation** in hot paths (codebook lookups use stack-allocated buffers)
- **No tensor ops** that would require `@device` context
- **No external function calls** that could touch userspace state
- **Bounds-checked indexing** at compile time via const generics

This is the **first quantization library** with these guarantees. PyTorch
quant has none of them.

Test verification:

```bash
cd ../fajaros-x86
cat kernel/compute/fajarquant.fj | head -50
# all functions annotated @kernel, verified by `fj check`
```

---

## Citation

If you use FajarQuant in academic work, please cite:

```bibtex
@misc{putranto2026fajarquant,
  title={FajarQuant v3.1: Adaptive Per-Head KV Cache Quantization with Compile-Time Safety Guarantees},
  author={Putranto, Muhamad Fajar},
  year={2026},
  publisher={PrimeCore.id},
  url={https://github.com/fajarkraton/fajarquant}
}
```

---

## Status

### Arm A — Phase D IntLLM (v0.4.0)

| Component | Status |
|---|---|
| HGRNBitForCausalLM 1.58-bit ternary architecture | Production |
| Mistral v3 32K tokenizer + SlimPajama-6B streaming loader | Production |
| Calibrated gate methodology (Mini/Base/Medium/Stretch) | Production |
| **Mini c.1 training (491M tokens)** | **PASS gate by 0.12 nat margin** |
| **Base c.1 training (982M Chinchilla-optimal tokens)** | **PASS gate by 0.21 nat margin** |
| **Medium c.1 training (1.819B tokens, ~Chinchilla)** | **PASS gate by 0.28 nat margin** |
| Stretch c.1 training | **Deferred to V32 post-Phase-E2 paper** |
| Track B 5+1-layer interruption-safety (V31.C.P6.1-P6.6) | Production, validated end-to-end |
| `make test-train-watchdog` regression gate (24 tests + signal delivery) | Green |
| Bench canonical real (8 lm-eval tasks × Mini/Base/Medium) | Complete |
| `make verify-intllm-tables --strict` (12/13 claims PASS) | 1 pending: kernel E2E Mini tok/s (FajarOS-side artifact) |
| Bench knowledge real (mmlu, triviaqa, boolq) | **Deferred to V32 post-Phase-E2 paper** |
| BitNet 2B4T baseline comparison | **Deferred to V32 post-Phase-E2 paper** |
| In-kernel deployment via FajarOS Nova IntLLM kernel-path | Production (FajarOS v3.9.0) |
| Phase D paper (Table 2 wikitext + hellaswag rows) | Real numbers populated |
| Phase D paper LaTeX writeup (full §4) | **Deferred to Phase E paper (one combined MLSys submission)** |

### Arm C — Phase E Bilingual Kernel-LLM (in flight, post-V31.E2.4)

Phase D extends to Indonesian + English bilingual ternary LLM in kernel context. Plan v1.9 (Tier 1+2 only; Tier 3 tax-vertical deferred to Phase F per founder solo-execution mode).

| Component | Status |
|---|---|
| **Phase E1 Bilingual corpus v1.0** | **CLOSED 2026-04-27.** 25.67 B tokens at 60:40 ID:EN (15.40 B ID + 10.27 B EN). 0% synthetic, 0.0254% exact-hash dedup. `make verify-bilingual-corpus` 8/8 invariants. |
| Phase E2.0 pre-flight + Q5 baseline | **CLOSED 2026-04-27.** Q5 = 24K-step Mini bilingual training: val_loss(ID)=2.68, val_loss(EN)=4.73, ratio=1.77×. |
| **Phase E2.4 BilingualCalibrationSampler** | **CLOSED with HONEST NEGATIVE RESULT 2026-04-27.** `outlier_global_reduction = −82.13` (gate ≥ 0.10) — calibrated quantizer is 83× WORSE than upstream baseline. Demoted to Phase D infra-diagnostic; NOT a Phase E2 ablation feature. See `docs/FJQ_PHASE_E_E2_BILINGUAL_CALIB_DECISION.md`. |
| Phase E2.1 Hadamard rotation | **NEXT** (~3-5 days human + 3-5h GPU). E2.1.0 ablation harness scaffold already shipped. |
| Phase E2.2 FP8 mixed-precision (lm_head + attn.W_o) | Pending (after E2.1) |
| Phase E2.3 FP16 distillation during QAT | Pending (after E2.2) |
| Phase E2.5 Language-conditioned design (per-lang RMSNorm γ preliminary) | Pending (after E2.3) |
| Phase E2.6 Combined ablation (4 features post-E2.4 closure) | Pending (after E2.1-2.5) |
| Phase E3 Bilingual pretrain (Base/Medium scale) | Pending (after E2 gate decision) |
| Phase E4 Paper draft + verification | Pending (after E2+E3) |
| Phase F (tax-vertical TaxPrime) | **Deferred** (gated on Phase E paper acceptance + 2 weeks; review checkpoint 2026-05-24) |

### Arm B — v3.1 KV Cache Quant (mature paper artifact)

| Component | Status |
|---|---|
| Four innovations (adaptive, outlier-aware PCA, fused, hierarchical) | Production |
| Multi-model perplexity evaluation (Gemma 4 E2B + Mistral 7B + Qwen2-7B) | Complete |
| 9-cell cross-architecture comparison (3 models × 3 bit widths) | Complete |
| Ablation: v3.0 → v3.1 (PPL-guided selection + per-head PCA) | Complete |
| LaTeX paper (11 pages arXiv + 10 pages MLSys 2027 template) | Complete |
| Venue decision committed (`paper/SUBMISSION.md`) | Complete |
| CI: 28 paper claim checks on every push | Complete |
| 6 Fajar Lang demos (`examples/*.fj`) | Complete |
| Kernel port (FajarOS Nova) | Phase 1+2 complete |
| ORCID + Zenodo DOI wire-up into paper | **Pending (C3.6)** |
| MLSys 2027 paper submission | **Pending (external deadline)** |
| Wall-clock latency benchmarks (ms/token vs KIVI/TQ) | Deferred (future work) |

### Cross-Repo Linkage

This v0.4.0 release pairs with:
- **[Fajar Lang v31.0.0](https://github.com/fajarkraton/fajar-lang/releases/tag/v31.0.0)** — compiler dependency; Phase D IntLLM uses `@noinline` + `@inline` + `@cold` (V29.P1) and `@no_vectorize` (V31.B.P2) attributes.
- **[FajarOS Nova v3.9.0](https://github.com/fajarkraton/fajaros-x86/releases/tag/v3.9.0)** — runs Phase D `medium_final.pt` checkpoints inside `@kernel` context via the IntLLM kernel-path (`make test-intllm-kernel-path` 4-invariant gate).

All three repos share Apache 2.0 license (relicensed from MIT on 2026-04-24 for fajar-lang + fajaros-x86; fajarquant has been Apache 2.0 since inception).

See `paper/SUBMISSION.md` and
[V26 Production Plan Phase C](https://github.com/fajarkraton/fajar-lang/blob/main/docs/V26_PRODUCTION_PLAN.md)
for the full roadmap and gate dates.

---

## License

Apache License 2.0. See [LICENSE](LICENSE).

---

*FajarQuant crate v0.4.0 / Phase D IntLLM + KV quant v3.1 — Made in Indonesia by [Muhamad Fajar Putranto](https://github.com/fajarkraton) (PrimeCore.id) — built with [Fajar Lang](https://github.com/fajarkraton/fajar-lang) v31.0.0 + Claude Opus 4.7*
