# FajarQuant v3 — Literature Landscape (Phase B0.2)

> **Date:** 2026-04-13 | **Papers:** 14 surveyed | **Rule:** CLAUDE.md §6.9 R2

## Landscape Table

| Paper | Venue | Method Type | Per-Head? | Key Insight | v3 Gap Addressed |
|---|---|---|---|---|---|
| **KIVI** | ICML 2024 | Per-channel key, per-token value | No | Keys: channel outliers; values: token outliers | Fixed axis per K/V; v3 selects axis per head |
| **KVQuant** | NeurIPS 2024 | Per-channel + NUQ + dense-sparse | No (per-layer) | Sensitivity-weighted NUQ; pre-RoPE key quant | Per-layer misses head variance; v3 per-head |
| **SKVQ** | COLM 2024 | Sliding window + channel reorder | No | Recent tokens high-prec, old 2-bit | Uniform method for all heads; v3 varies per head |
| **TurboQuant** | arXiv 2025 | Random rotation + Lloyd-Max | No | Haar-random rotation → i.i.d. Beta | Same rotation for every head; v3 selects rotation type |
| **Coupled Quant** | NeurIPS 2024 | Joint multi-channel VQ | No | Joint entropy < marginal sum | Uniform group size; v3 adapts coupling per head |
| **GEAR** | arXiv 2024 | Residual (low-rank + sparse) | No | Error = low-rank bulk + sparse outlier | Global residual rank; v3 per-head residual budget |
| **QServe** | MLSys 2025 | W4A8KV4 + SmoothAttention | No | Channel rebalancing before KV4 | Uniform rebalancing; v3 skips when unnecessary |
| **MiKV** | arXiv 2024 | Mixed-precision (token-level) | Partial (token) | Low-precision retention > eviction | Token-level, not head-level; v3 2D adaptive |
| **KVTuner** | ICML 2025 | Layer-wise mixed-precision | Layer-wise | Per-layer sensitivity analysis for precision | Layer granularity; v3 extends to per-head |
| **RotateKV** | IJCAI 2025 | Outlier-aware adaptive rotation | Yes (grouped) | Channel reorder + FWHT per group | Groups heads but same quantizer; v3 varies both rotation AND quantizer |
| **SpinQuant** | arXiv 2024 | Learned rotation (W+A+KV) | No | Cayley-parameterized learned rotation | Global rotation, requires training; v3 calibration-only |
| **FlatQuant** | COLM 2025 | Learned affine transform | No | Per-layer affine flattening | Requires training; v3 calibration-only |
| **PALU** | ICLR 2025 | Low-rank projection | Yes (per-head rank) | SVD per-head; cache compressed latent | Dimensionality, not bit-width; composable with v3 |
| **llama.cpp #21385** | Engineering | Entropy-based bit allocation | **Yes** | Sink heads (2% lowest entropy) need f16 | Bit allocation only, not method selection; v3 selects quantizer type |

## Key Finding

**No published method does per-head method selection** — choosing among rotation, NUQ, residual, or scalar quantization on a head-by-head basis.

Closest work:
- **RotateKV**: grouped-head rotation, but same quantizer everywhere
- **KVTuner**: mixed-precision, but layer-level not head-level
- **PALU**: per-head rank, but dimensionality reduction not quantization
- **llama.cpp #21385**: per-head bit allocation, but single quantizer type

**FajarQuant v3's unique contribution:** first framework that profiles each head's distribution (kurtosis, σ1/σ2, outlier fraction, skewness, channel CV) and dispatches to the optimal quantizer from a 5-method portfolio.

## Papers to cite in v3 paper section

New citations needed (beyond v2's 17): KVTuner, GEAR, MiKV, QServe, Coupled Quantization, RotateKV, PALU.
