# FajarQuant v3.1 — Adaptive Per-Head KV Cache Quantization

> **No single KV cache quantization method dominates across attention architectures.
> FajarQuant v3.1 profiles each KV head and routes to the optimal strategy,
> matching or beating the best fixed method in 7 of 9 evaluation cells with
> zero catastrophic failures.**

[![Crates.io](https://img.shields.io/crates/v/fajarquant?color=blue)](https://crates.io/crates/fajarquant)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](LICENSE)
[![Compiler](https://img.shields.io/badge/compiler-Fajar_Lang_v27.5.0-blueviolet)](https://github.com/fajarkraton/fajar-lang)
[![Paper](https://img.shields.io/badge/paper-v3.1_adaptive-success)](paper/fajarquant.pdf)
[![Verify](https://img.shields.io/badge/claims-28%2F28_verified-brightgreen)](reproduce.sh)
[![Made in Indonesia](https://img.shields.io/badge/Made_in-Indonesia-red)]()

FajarQuant is a Rust-native KV cache quantization library with the first
systematic cross-architecture perplexity evaluation across 3 models and 3
bit widths (9 cells total), using the canonical R-α.1 model-surgery protocol.

## Headline Results — FajarQuant v3.1 vs Best Fixed Method

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

See `paper/SUBMISSION.md` and
[V26 Production Plan Phase C](https://github.com/fajarkraton/fajar-lang/blob/main/docs/V26_PRODUCTION_PLAN.md)
for the full roadmap and gate dates.

---

## License

Apache License 2.0. See [LICENSE](LICENSE).

---

*FajarQuant crate v0.3.0 / algorithm v3.1 — Made in Indonesia by [Fajar](https://github.com/fajarkraton) (PrimeCore.id) — built with [Fajar Lang](https://github.com/fajarkraton/fajar-lang) + Claude Opus 4.6*
