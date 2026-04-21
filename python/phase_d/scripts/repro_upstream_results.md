# V31.C.P2.1 FULL gate — MatMul-free 370M repro

**Model:** `ridger/MMfreeLM-370M`
**Elapsed:** 340.2 s (5.7 min) on RTX 4090 Laptop
**Gate:** §6.9 R3 baseline parity, ±1.0 avg point

| Task | Paper Table 1 | Our repro | Δ |
|---|---:|---:|---:|
| arc_easy | 42.60 | 38.85 | -3.75 |
| arc_challenge | 23.80 | 22.27 | -1.53 |
| hellaswag | 32.80 | 32.45 | -0.35 |
| openbookqa | 28.40 | 28.40 | +0.00 |
| piqa | 63.00 | 62.68 | -0.32 |
| winogrande | 49.20 | 49.72 | +0.52 |
| **AVERAGE** | **40.30** | **39.06** | **-1.24** |

**Result:** FAIL ✗ (|Δ avg| = 1.24)
