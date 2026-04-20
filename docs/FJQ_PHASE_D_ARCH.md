# FajarQuant Phase D — Architecture Decision (C.P1.1)

> **Date:** 2026-04-21 | **Gate:** G3 mechanical decision per CLAUDE.md §6.8 Rule 6 | **Depends on:** C.P0 findings (`FJQ_PHASE_D_P0_FINDINGS.md`, commit `1fc64e2`)

## Decision

**Phase D primary architecture: MatMul-Free LLM (MLGRU token mixer + GLU channel mixer + BitLinear ternary dense), extended for FajarOS i64 kernel deployment with quantization-aware training.**

**Secondary/ablation architecture: RWKV-7 "Goose" stripped to the same recipe (NoPE, ternary BitLinear, identical training corpus)** — used as a scale-vs-recall comparison point at 1.5B+, triggered only if the primary hits a recall-quality ceiling at 2.7B on canonical benchmarks.

This commit unblocks C.P1.2 (ops inventory), C.P1.3 (paper outline), C.P1.4 (config matrix), and the full C.P2+ implementation sprint.

---

## 1. Candidate comparison (PDF-depth reads, not landscape)

Three deep-reads completed 2026-04-21 covering RWKV-7 (`2503.14456v2`,
Peng et al.), Mamba-2 (`2405.21060`, Dao & Gu), and MatMul-Free LLM
(`2406.02528v5`, Zhu et al. + FPGA impl). Full notes retained in
commit history; this doc records only the decision-relevant deltas.

### 1.1 Pad-collapse root-cause avoidance (the reason Phase D exists)

V31.R3 closure isolated the root cause to **large-gamma × many-norm
accumulation in x1000 fixed-point** — Gemma 3's per-layer RMSNorm γ
values reach 66.5 real, applied across 104 norm sites, compounding
past the fixed-point precision envelope. Representation direction is
lost. The Phase D architecture must structurally avoid this failure
mode, not mitigate it.

| Candidate | Per-layer γ magnification | Softmax accumulation | Long exp-accumulator | Verdict |
|---|---|---|---|---|
| **A** RWKV-7 | None (GroupNorm γ + sqrt clamp from v6→v7 conversion) | None (no softmax) | Bounded via decay floor 0.545 + L2-norm κ | **Structurally avoids** |
| **B** Mamba-2 | Standard RMSNorm + GroupNorm (2 per layer, γ present) | None | **exp(segsum(A))** — can grow large negative on long contexts | **Partially avoids** — exp(segsum) is V31.R3-adjacent hazard |
| **C** MatMul-Free | **No per-layer γ at all** (scale absorbed into β/Q_b post-⊛, where β is mean-abs-weight ~O(0.3-0.7), always shrinking) | None (no attention at all) | None | **Structurally eliminates the root cause** |

**C wins on the primary axis by construction.** A is close but keeps GN
γ. B inherits a related (not identical) accumulation hazard via
`exp(segsum)`.

### 1.2 i64 port complexity

Per-token per-layer transcendental inventory:

| Candidate | σ | tanh | exp | sqrt/isqrt | L2-norm | softplus | softmax | Total |
|---|---|---|---|---|---|---|---|---|
| A RWKV-7 | 4× | 1× | 2× (1 constant + 1 bounded (−0.6, 0)) | 1× (GN) | 1× (κ̂) | — | — | 8 sites + 1 complex composition |
| B Mamba-2 | 2-3× (SiLU sites) | — | 1× (segsum) | 2× (RMSNorm + GN) | — | 1× (Δ) | — | 6-7 sites + segsum accumulator |
| **C MatMul-Free** | **2× (f_t, g_t)** | — | — | **1× (isqrt per BitLinear, reuses km_isqrt)** | — | — | — | **3 sites, all bounded, all with existing FajarOS approximations** |

C is the cheapest port. `km_isqrt` already shipped in V28/V29; σ needs
one new LUT or PWL; SiLU reuses σ via `x · σ(x)`.

Plus: **C has zero matmul.** Ternary `⊛` is signed-add only, already
validated in-kernel via V28 ternary path work. The FPGA reference
implementation at Sec 5 of Zhu et al. uses 8-bit activations + i32-ish
accumulators and hits 13 W at 1B+ — **strictly weaker arithmetic
envelope than i64**, which is an existence proof that the integer
math is sound at scale.

### 1.3 Verified scale × openness × license

| Candidate | Top open weights | License | Training code | Training corpus | Eval depth |
|---|---|---|---|---|---|
| A RWKV-7 | **2.9B** multilingual (HellaSwag 76.4, MMLU 55.0, beats Llama-3.2-3B on multilingual) | Apache-2.0 | Yes | RWKV World v3 (3.1T tokens) — listed, not packaged | Deep: PPL + LAMBADA + 6 downstream + MMLU |
| B Mamba-2 | **2.7B** base (official) + 7B Codestral (code-only, Apache) + 7B Zamba2 (hybrid, not pure) | Apache-2.0 mixed | Yes | Pile (official); proprietary (Codestral/Zamba) | PPL curve on Pile |
| C MatMul-Free | **2.7B** base (`ridger/MMfreeLM-{370M,1.3B,2.7B}`) | Apache-2.0 | Yes (flash-linear-attention fork) | SlimPajama 100B | **Shallow: zero-shot 6-task only, no PPL, no MMLU, no long-context** |

A has the highest verified instruction-quality scale. B's pure-SSM
ecosystem is fragmented (Codestral is code-only, Zamba2 is hybrid).
C caps at 2.7B with shallow evals.

### 1.4 Authors' cited weaknesses

| Candidate | #1 author-flagged limitation |
|---|---|
| A RWKV-7 | **"Numerical precision of the WKV7 kernel"** is the paper's first listed limitation (§10.1). Directly adverse for i64 fixed-point port. |
| B Mamba-2 | Not explicitly flagged as #1, but the in-context-recall gap on MQAR is well-documented (Zoology benchmark). Every 7B deployment is hybrid. |
| C MatMul-Free | "Not tested on extremely large-scale models (e.g., 100B+)." Scale ceiling 2.7B is real; "beats Transformer at 10²³ FLOPs" is projection, not measurement. |

A's author-flagged risk is the most adverse for Phase D specifically
— they themselves say the kernel is precision-sensitive, and i64
fixed-point reduces precision further. C's scale-ceiling risk is
comparatively benign for a research artifact: Phase D's target scale
(25-50M params in C.P4 config matrix, 1B as stretch goal) is
comfortably within the verified envelope.

### 1.5 Ternary-weight feasibility (already-published recipes)

| Candidate | Published ternary recipe? | Reference |
|---|---|---|
| A RWKV-7 | No published QAT. Would be novel. | — |
| B Mamba-2 | Bi-Mamba (`2411.11843`) + Slender-Mamba (COLING 2025) + BitMamba-2 community impl. PPL gap +1.6 at 2.7B. Keeps A/B/C/Δ FP (compromise). | Published |
| C MatMul-Free | **Ternary IS the published architecture.** BitLinear is the default. No retrofit needed. | Published |

C ships ternary out of the box. B has a workable retrofit with a
quality compromise. A would require novel QAT research.

---

## 2. Decision rationale

**Candidate C (MatMul-Free LLM) wins on every primary axis:**

1. ✅ **Pad-collapse root-cause avoidance** — structurally, not
   mitigation. No per-layer γ magnification, no softmax, no RoPE.
2. ✅ **Simplest i64 port** — 3 transcendental sites total, zero
   matmul, FPGA-validated arithmetic envelope.
3. ✅ **Ternary is the baseline** — no QAT research burden.
4. ✅ **Apache-2.0 weights + code + HF-compatible model class.**
5. ⚠️ **Scale ceiling 2.7B** — acceptable for a research artifact.
   Phase D targets 25-50M + 1B stretch; 2.7B is a hard upper bound
   that exceeds FajarOS Nova's Gemma 3 1B target by 2.7×.
6. ⚠️ **Shallow benchmarks in base paper** — Phase D adds PPL
   benchmarks per §6.9 R1 canonical protocol; this is C.P6 work
   anyway and is a paper contribution, not a blocker.

**Candidate A (RWKV-7) wins on:**
- Highest verified open scale (2.9B multilingual instruct-quality)
- Deepest evaluation protocol
- Structural pad-collapse avoidance (close to C)

**Candidate A's blocker:** authors' own #1-cited limitation is kernel
precision sensitivity in the native fp32 implementation. Going to
i64 fixed-point *reduces* precision further. Phase D would inherit a
research risk the paper's own authors flag as unresolved, with no
published QAT baseline to anchor against.

**Candidate B (Mamba-2) loses on:**
- exp(segsum) accumulation hazard (V31.R3-adjacent)
- Fragmented open ecosystem at ≥1B (no pure Mamba-2 general-purpose
  instruct model at 7B — everything is hybrid or code-only)
- Ternary recipe keeps internals FP (compromise that dilutes the
  "fully integer" story)

---

## 3. Hedge: Candidate A as secondary ablation

Despite the kernel-precision risk, **RWKV-7 is retained as a
secondary ablation** to be run **only if** the MatMul-Free baseline
hits a recall ceiling on canonical benchmarks at 2.7B (specifically,
>3 PPL gap vs Transformer++ at matched params, or <40 on MMLU 5-shot
at 1.5B+).

Trigger conditions (committed in C.P4 gates):
- PPL gap vs Transformer++ at matched params exceeds 3 points
- Needle-in-a-haystack test fails beyond 2K context
- MQAR-style in-context-recall drops >20% vs Transformer++

If any trigger fires, spin up an A-track in parallel with C.P5. The
ablation re-uses the same dataset pipeline (C.P3), tokenizer, and
FajarQuant quantization harness — only the model class swaps. Budget:
+2 weeks if triggered; default 0 weeks if not.

---

## 4. Phase D novel contribution (what we publish)

Phase D is **not** a reimplementation of MatMul-Free LLM. The novel
contribution (positioning for MLSys / NeurIPS MLSys Workshop /
TinyML Summit) is:

1. **Pure-i64 kernel deployment** — the original paper validated an
   FPGA at 8-bit; Phase D ports to general-purpose i64 CPU (FajarOS
   Nova). First such demonstration published for this architecture.
2. **Quantization-aware training via FajarQuant** — the original
   paper uses absmean-static weight quant + absmax-static activation
   quant. Phase D adds learned per-coord adaptive bit allocation +
   outlier-aware channel reordering during QAT (Rule §6.9 R5).
3. **Canonical-protocol quality numbers** — the original paper
   publishes only zero-shot commonsense. Phase D adds Wikitext-103
   PPL, LAMBADA, MMLU, HellaSwag, ARC-Challenge, and
   needle-in-a-haystack evaluations. (Rule §6.9 R1)
4. **Kernel-integrated deployment in FajarOS Nova shell** — boot →
   model-load → ask workflow. First language model native to a
   research OS's shell environment. (Rule §6.9 R1 — E2E claim.)
5. **Ablation against RWKV-7 + Mamba-2 + BitNet-b1.58 at matched
   params** if the recall trigger fires, giving the broader SSM/RNN
   community a head-to-head i64 comparison that doesn't currently
   exist.

Paper positioning: *"First language model with a fully integer-native
kernel forward pass on general-purpose hardware. We extend Matmul-Free
LLM with quantization-aware training, canonical-protocol evaluation,
and kernel-integrated deployment, and present ablations against RWKV-7
and Mamba-2 at matched parameter counts. Empirically, we achieve X PPL
on Wikitext-103 at Y params, within Z of FP16 Transformer++ baseline,
at W watts on consumer CPU."* (X, Y, Z, W filled in C.P6.)

---

## 5. Configuration matrix (preview — authoritative numbers in `FJQ_PHASE_D_CONFIG.md`)

| Config | d_model | L | Params (V=32K) | Training tokens | RTX 4090 wall-clock |
|---|---|---|---|---|---|
| **Mini** | 256 | 6 | **~21.5M** | 500M | ~1h |
| **Base** (C.P4.1 primary) | 384 | 12 | **~46.4M** | 1B | ~4h |
| **Medium** | 512 | 12 | **~71.3M** | 2B | ~11h |
| **Stretch** | 1024 | 24 | **~369.1M** | 15B | ~17d |

> Preliminary param estimates here were 8M/25M/50M/370M; those were
> pre-formula. Authoritative numbers above use the exact accounting
> `12Ld² + 2Vd` (V=32K Mistral v3) shipped in C.P1.4. Stretch matches
> Zhu et al. 370M row exactly.

Starting from **Base (~46M @ 1B tokens)**. Mini is a fast-iteration
config for QAT sweeps + calibration ablations (all complete within a
working day). Medium is the first configuration to compare against
FP16 Transformer++ directly. Stretch (370M) matches the smallest
published MatMul-Free checkpoint, giving direct repro-check against
the original paper's Table 1 ARC/HS/OQ/PQ/WGe numbers.

1B stretch goal reserved for C.P4.3 (final QAT phase) only if Medium
hits the §6.9 R1 gate. Per §6.9 R6, we do NOT write results-section
paper text until the 1B canonical benchmarks are measured.

---

## 6. Risks + mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Recall gap at 2.7B (MLGRU fixed-state) | Medium | Trigger Candidate A ablation per §3 |
| Training instability above lr=2e-2 (paper §4.3) | Medium | Use paper-cited stable lr range 1e-3 to 1e-2; AdamW with warmup |
| FajarQuant QAT degrades ternary baseline | Low | Start from `ridger/MMfreeLM-370M` checkpoint; fine-tune with QAT only (not from scratch) |
| No published PPL — unclear baseline | Low | Re-measure on SlimPajama 100B held-out, same corpus as original |
| i64 overflow in MLGRU state over long context | Medium | Analytical overflow bounds per §6.9 R4: max state norm × sequence length × d ≤ i64 max. Add calibration pass. |
| Scale ceiling 2.7B blocks "production" claim | High | Phase D paper is explicitly a **research artifact + FajarOS capability demo** per §6.9 R6. Deployment-scale claim deferred. |

---

## 7. Unblocks

Per V31 Master Plan Track C dependency graph:

- **C.P1.2** (ops inventory) can start now — proceed with MLGRU +
  GLU + BitLinear + RMSNorm op inventory plus σ/SiLU LUT spec.
- **C.P1.3** (paper outline) can start now — MLSys template with
  novel-contribution framing per §4 above.
- **C.P1.4** (config matrix) formalized in §5 above; ready for
  review.
- **C.P2** (PyTorch reference) unblocked — fork `ridgerchu/matmulfreellm`
  as starting implementation, add QAT harness.
- **C.P3** (dataset pipeline) unblocked — tokenizer = Mistral v3
  (matches paper) or SmolLM's 49K (already-shipped FajarOS), TBD
  in C.P3.1.

---

## 8. §6.9 Research Integrity checklist

| Rule | Status |
|---|---|
| R1 — canonical protocol from ≥2 refs | ✅ Wikitext-103 + LAMBADA + MMLU-5shot committed as Phase D eval protocol (matches BitNet-2B4T and RWKV-7 overlaps) |
| R2 — ≥8-10 paper lit sweep | ✅ C.P0 shipped: 22 verified papers |
| R3 — baseline parity | ✅ Transformer++ (Llama-2 repro per Zhu et al. Table 1), plus Candidate A ablation if triggered |
| R4 — calibration once, not per-chunk | ✅ FajarQuant QAT uses per-matrix γ (absmean) computed ONCE during QAT, not per-forward |
| R5 — explicit outlier handling | ✅ Top-K per-coord adaptive bit allocation + channel reordering during QAT (novel contribution #2) |
| R6 — algorithmic validation precedes paper validation | ✅ Results-section text gated on C.P6 measurements per C.P7 plan |
| R7 — pre-publication audit gate | ⏳ `verify_paper_tables.py --strict` to be added in C.P7 before any release/PDF |

---

## 9. §6.8 Plan Hygiene checklist (for this decision commit)

| Rule | Status |
|---|---|
| R1 — pre-flight audit exists | ✅ C.P0 findings (`1fc64e2`) |
| R2 — verifications are runnable commands | ✅ §6 ablation triggers + §5 config matrix spec measurable |
| R3 — prevention layer added | ⏳ C.P7 will add `verify_paper_tables.py` CI gate |
| R4 — agent numbers cross-checked | ✅ 3 parallel PDF-depth reads cross-referenced against C.P0 landscape |
| R5 — variance tagged | Tagged in commit msg |
| R6 — mechanical decision gate | **✅ THIS FILE. Downstream C.P1.2-4 + C.P2 unblocked by commit of this file.** |
| R7 — public artifact sync | N/A for internal design doc |
| R8 — multi-repo state check | ✅ fajarquant ahead 0, fajar-lang ahead 0, fajaros-x86 ahead 0 confirmed pre-commit |
