# FajarQuant Paper Rewrite Outline (Phase B6 Reference)

> **Purpose:** Pre-B6 outline so the paper rewrite has a clear target
> structure. Created during B1 execution to parallelize planning.
>
> **When to use:** after B4.G GO decision is committed.

---

## Paper structure (revised for v2.12)

### 1. Abstract (~250 words)

**Current (v1) claim to revise:**
> "PPL 80.1 at 2-bit vs TurboQuant 117.1, KIVI 231.9"

**Revised structure:**
- Problem: KV cache memory dominance in embedded LLM inference
- FajarQuant v1: PCA-based rotation, wins on MSE but loses on PPL under canonical protocol
- FajarQuant v2.12: [chosen design] addressing outlier alignment, calibration noise, [other fixes]
- Result: v2.12 beats TurboQuant by [X%] on [N/3] models at [bit-width] under canonical protocol
- Embedded deployment: native Fajar Lang implementation, bare-metal kernel, [speedup]
- Unique contribution: first KV quantization framework integrating with a systems programming language

### 2. Introduction

- Motivation: 7B model at 4K context = ~4 GB KV cache in FP16
- Prior work: KIVI per-channel/per-token; QuaRot/SpinQuant rotation; KVTC transform coding
- Problem: rotation-based methods dominate but all assume GPU-only deployment
- Our contribution: FajarQuant, a hardware-aware quantization system built into Fajar Lang
  - v1: PCA-based rotation — wins on MSE, but per-chunk instability + outlier alignment hurt PPL
  - v2.12: [design name] — addresses root causes via [mechanism], demonstrated on 3 architectures
  - Embedded story: runs on bare-metal kernel (FajarOS), not just GPU

### 3. Background & Related Work (EXPANDED from v1)

#### 3.1 KV Cache Quantization Landscape

**Must cite (from C1.6 literature sweep):**
- KIVI [Liu et al., ICML 2024] — per-channel K, per-token V, 2-bit baseline
- KVQuant [Hooper et al., NeurIPS 2024] — outlier handling, < 0.1 PPL @ 3-bit
- QuaRot [Ashkboos et al., 2024] — Hadamard rotation, lossless 6-8 bit
- SpinQuant [Liu et al., ICLR 2025] — learned Cayley rotation, 2.9 pt gap W4A4
- FlatQuant [Liu et al., ICML 2025] — learnable affine, +7.5% over SpinQuant
- RotateKV [IJCAI 2025] — outlier-aware adaptive Hadamard, < 0.3 PPL @ 2-bit
- OTT [ACL 2025] — outlier token tracing
- AsymKV [COLING 2025] — 1-bit KV cache
- TurboQuant [Zandieh & Mirrokni, ICLR 2026] — random ortho + outlier preservation, 6× compression
- KVTC [NVIDIA, ICLR 2026] — PCA + DP bit allocation + entropy coding, 20× compression
- KVLinC [Oct 2025] — Hadamard + linear correction

#### 3.2 Rotation Strategies
- Data-independent: Hadamard (QuaRot), random orthogonal (TurboQuant)
- Data-dependent: PCA (KVTC, FajarQuant v1), SVD
- Learned: Cayley SGD (SpinQuant), affine (FlatQuant)
- **Position:** FajarQuant v2.12 = [chosen strategy], motivated by RC1-RC5 analysis

#### 3.3 Outlier Handling
- SmoothQuant channel scaling
- KVQuant top-1% fp16
- TurboQuant top-15% bf16
- OTT dynamic token-level
- RotateKV channel reordering + attention-sink-aware
- **Position:** FajarQuant v2.12 includes [chosen outlier handling]

### 4. Method

#### 4.1 FajarQuant v1: PCA Rotation (baseline)
- Algorithm: per-chunk PCA on K/V → per-coord uniform quantization
- Strengths: data-driven decorrelation, MSE-optimal on clean data
- Weaknesses: RC1-RC5 (outlier alignment, per-chunk noise, RoPE interference, K-V independence, no outlier extraction)
- **This section is new — v1 paper didn't self-criticize**

#### 4.2 Root Cause Analysis (from B2 diagnosis)
- Table: RC1-RC5 with per-model PPL impact from B1 baseline data
- Figure: eigenvalue spectrum of Gemma/Mistral/Qwen2 showing outlier concentration
- **Honest self-critique — this is the paper's strongest signal of rigor**

#### 4.3 FajarQuant v2.12: [Chosen Design Name]
- Algorithm (per B2.D design spec)
- Per-component ablation (from B4 ablation table)
- Implementation: Python reference + Fajar Lang native

### 5. Experimental Setup

#### 5.1 Models
| Model | Params | Arch | KV heads | Head dim | Layers |
|---|---|---|---|---|---|
| Gemma 4 E2B | 5B | MQA | 1 | 256 | 35 |
| Mistral 7B v0.1 | 7B | GQA (32:8) | 8 | 128 | 32 |
| Qwen2-7B | 7B | GQA (28:4) | 4 | 128 | 28 |
| (Llama 2 7B, if access) | 7B | MHA (32:32) | 32 | 128 | 32 |

#### 5.2 Evaluation Protocol
- **PPL:** Canonical KVQuant/SKVQ protocol — non-overlapping 2048-token chunks, fresh DynamicCache per chunk, model surgery via per-architecture attention forward subclasses, shift_logits cross-entropy. WikiText-2 raw test split, 30 samples.
- **MSE:** Per the existing cross-model framework (n ≈ 19,400 samples from C1.4 production data).
- **Key citation:** "We follow the canonical evaluation protocol established by KVQuant [Hooper et al., NeurIPS 2024] and SKVQ [COLM 2024]."

#### 5.3 Baselines
- FP16 (no quantization)
- KIVI [ICML 2024] — per-channel K, per-token V
- TurboQuant [ICLR 2026] — random ortho + top-15% bf16 outlier preservation (FULL published spec)
- TurboQuant (naive) — for ablation only
- FajarQuant v1 — for v1→v2 delta story
- FajarQuant v2.12 — proposed method

### 6. Results

#### 6.1 Cross-Model PPL (main result)

**Table: tab:ppl_crossmodel_v2** (replaces old tab:ppl)

| Model | Bits | FP16 | FQ v1 | FQ v2 | KIVI | TQ (published) |
|---|---|---|---|---|---|---|
| Gemma | 2 | TBD | TBD | TBD | TBD | TBD |
| Gemma | 3 | TBD | TBD | TBD | TBD | TBD |
| Gemma | 4 | TBD | TBD | TBD | TBD | TBD |
| Mistral | 2 | ... | ... | ... | ... | ... |
| ... | | | | | | |

(3 models × 3 bits × 5 methods = 45 cells; currently TBD, filled in B4)

#### 6.2 Cross-Model MSE (preserved from v1)
- Existing Table tab:crossmodel (Gemma n=1500, Mistral n=12800, Qwen2 n=5600)
- **This table is UNCHANGED** — MSE data from C1.4 production runs is still valid
- "FajarQuant wins keys across all 3 architectures and all 3 bit widths (9 of 9 cells)"

#### 6.3 Ablation Study
- Table: v1 → v2-A → v2-D → v2-D+RoPE → v2-F per model per bit (from ablation roadmap)
- Highlight: which design component contributes most PPL improvement

#### 6.4 Structural Analysis
- Eigenvalue spectrum comparison (Gemma vs Mistral vs Qwen2) — explain why FQ performance varies by architecture
- Outlier channel distribution — how many channels need fp16 preservation

### 7. Embedded Deployment (NEW section, from B5)

#### 7.1 Fajar Lang Integration
- Native `Quantized<T, BITS>` type with compile-time safety
- `@device` context for zero-allocation quantization hot paths
- Compile-time calibration matrices as `const` tensors

#### 7.2 FajarOS Kernel Implementation
- `fajaros-x86/kernel/compute/fajarquant_v2.fj` — bare-metal implementation
- AVX2/AES-NI inline assembly for Hadamard + quantization
- Stack-allocated `QuantizedKVCache` — no heap in kernel context

#### 7.3 Performance vs Python Reference
- Table: native Fajar Lang vs Python per-layer quantization time
- "First KV quantization method natively integrated into a systems programming language"

### 8. Discussion

- v1 → v2 lessons: why per-chunk PCA failed, how literature informed the fix
- Comparison to KVTC: FajarQuant v2 achieves similar quality but without entropy coding
- Embedded vs GPU: different deployment targets, same algorithm
- Limitation: calibration cost (one-time, ~10 min), model-specific

### 9. Conclusion

- FajarQuant v2.12 achieves [X PPL] at 2-bit (within [Y] of FP16)
- Beats published TurboQuant by [Z%] on [N/3] models
- Unique contribution: first language-integrated KV quantization for embedded systems
- MSE cross-model validation on 3 architectures (19,400 samples) confirms structure
- Future work: Llama 2 (pending access), larger models, actual embedded deployment

### Appendices

- A: Full numerical results (all 45 PPL cells + all MSE cells)
- B: Calibration procedure details
- C: Fajar Lang code listing (key snippets, not full implementation)
- D: Reproducibility (hardware snapshot, seeds, `reproduce.sh` usage)

---

## Paper length estimate

| Section | Pages |
|---|---|
| Abstract | 0.25 |
| Introduction | 0.75 |
| Background & Related Work | 1.5 |
| Method | 1.5 |
| Experimental Setup | 0.75 |
| Results | 2.0 |
| Embedded Deployment | 0.5 |
| Discussion | 0.5 |
| Conclusion | 0.25 |
| References | 1.0 |
| **Total** | **~9 pages** (expanded from current 5-page v1) |

If targeting a workshop paper (4-6 pages), §7 Embedded Deployment can
be shortened and §3 Related Work trimmed. Core story (§4-§6) needs at
least 4 pages.

---

*Document version: 2026-04-12 v1.0. Will be updated after B4.G GO
decision with actual numbers replacing TBD cells.*
