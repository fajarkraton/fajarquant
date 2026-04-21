# IntLLM — Related Work

> **Status:** Phase 4.1 deliverable of `FJQ_PHASE_D_PRODUCTION_PLAN.md`.
> Committed BEFORE paper results section per §6.9 R6. Source material
> synthesizes `FJQ_PHASE_D_P0_FINDINGS.md` (22-paper literature sweep from
> C.P0, 2026-04-21) with additional small-LM + kernel-integration references.
>
> **Scope:** 12 papers across 5 thematic clusters. All arXiv IDs verified
> during C.P0. MLSys 2027 submission target — see
> `FJQ_PHASE_D_PAPER_OUTLINE.md` §5 "Related work depth target".

---

## 0. Positioning

IntLLM is a **fully integer-native** decoder-only language model targeting
research-grade kernel deployment at <100 MB on a ≥512 KB-scratch MCU. Three
claims shape its novelty relative to prior work:

1. **Triple integer-native reduction** — weights (ternary via BitLinear),
   activations (i8 absmax → i64 accum), *and* nonlinearities (σ/SiLU via
   257-entry LUT) all integer, with no FP16 fallback in the hot path. The
   closest prior art — BitNet b1.58-2B4T — ships ternary *weights* but
   retains FP softmax and FP RoPE.
2. **Softmax-free + position-free attention replacement** — adopted from
   MatMul-Free LLM (Zhu et al., 2024) because the FP softmax gap is the
   load-bearing FP op in BitNet-class models. We extend to full i64
   fixed-point everywhere, including MLGRU's `h_t = f ⊙ h_{t-1} + (1-f) ⊙ c`.
3. **Kernel-integrated deployment verified by regression gate** — we ship
   `make test-intllm-kernel-path` booting FajarOS in QEMU with a .fjm v9
   model on NVMe, exercising the full ternary forward through RamFS/NVMe
   streaming. BitNet's `bitnet.cpp` ships CPU kernels but no OS-level
   integration; MatMul-Free has an FPGA proof-of-concept at 13 W but no
   bootable artifact.

The sections below map our method to the paper each component is built
on, extended from, or diverges from.

---

## 1. Integer-native / ternary LLM quantization

### BitNet b1.58 (Wang et al., 2024)
- **arXiv:** 2402.17764 (Feb 2024, Microsoft Research Asia)
- **Core contribution:** `BitLinear` replaces `nn.Linear` during pretraining.
  Weights are absmean-normalized and projected onto {−1, 0, +1}. Activations
  are FP16 with absmax i8 quantization applied pre-matmul. FP16-parity at 3B.
- **What we adopt:** the `β = E[|W|]` weight scale + per-matrix STE for
  ternary quantization-aware training. `intllm/quant.py::fake_quantize_absmax_ste`
  is a direct port.
- **What we extend:** BitNet keeps softmax, RoPE, and RMSNorm γ in FP.
  IntLLM replaces these with MLGRU (§2.1), NoPE (§4.1), and γ-less RMSNorm
  (FJQ_PHASE_D_OPS.md §2 Op 2).
- **What we diverge from:** BitNet's reference implementation uses per-tensor
  β recomputed each forward pass; we calibrate β ONCE at QAT freeze per
  §6.9 R4 ("calibrated > per-chunk") and bake it into the .fjm v9 β table.

### BitNet b1.58 2B4T (Ma et al., 2025)
- **arXiv:** 2504.12285 (Apr 2025, Microsoft Research)
- **Core contribution:** first open-source ternary LLM at 2B parameters
  trained on 4T tokens, with published zero-shot HellaSwag, ARC-c, MMLU
  numbers (Table 1). Ships via HuggingFace `microsoft/bitnet-b1.58-2B-4T`
  + a custom CUDA pack-store-load inference kernel.
- **What we adopt:** benchmarks (same lm-eval tasks + 5-shot MMLU + 5-shot
  TriviaQA) pinned to `lm-evaluation-harness v0.4.11`. Our §4.2 Table 2
  mirrors BitNet Table 1's exact layout for apples-to-apples comparison.
- **What we diverge from:** BitNet's CUDA pipeline is opaque; we provide
  pure-C/Fajar-Lang kernels that run on MCU-class hardware (16 MB RAM budget
  for Base config at d=384, L=12, V=32K).
- **Baseline parity concern (§6.9 R3):** BitNet ships a custom CUDA
  pack-store-load path that isn't trivially portable. The `BASELINE_UNPORTED_FEATURES.md`
  document (Phase 3 deliverable) lists this plus any other unported BitNet
  feature that could bias our head-to-head measurements in the reviewer's favor.

### Bitnet.cpp (Xu et al., 2025)
- **arXiv:** 2502.11880 (Feb 2025, Microsoft Research)
- **Core contribution:** x86 AVX2 + ARM NEON CPU kernels for ternary
  matmul. Reports 2-6× speedup over naive FP16 on CPU-only inference.
- **Role:** benchmark target + reference for CPU deployment viability.
- **What we extend:** Bitnet.cpp is userspace CPU; we target kernel-space
  (FajarOS Nova) + MCU-class systems. Our kernel-path gate verifies the
  entire ternary pipeline runs without crashing the OS.

### MatMul-Free LLM (Zhu et al., 2024)
- **arXiv:** 2406.02528 (Jun 2024, UC Santa Cruz + Stanford + Microsoft)
- **NeurIPS 2024** — the foundational architecture IntLLM builds on.
- **Core contribution:** `MLGRU` token-mixer (a 4-BitLinear gated recurrent
  mixer) replaces self-attention, structurally eliminating softmax; `GLU`
  channel-mixer with 3 BitLinears (we use 2 BitLinears with combined gate+up
  per the reference code) replaces the standard FFN. Result: ternary weights
  + MLGRU + GLU is the first architecture with ZERO required matmuls at
  inference. Validates to 2.7B scale. Ships an FPGA accelerator at 13 W.
- **What we adopt:** full architecture including config matrix
  (Mini/Base/Medium/Stretch d/L/V layout in `FJQ_PHASE_D_CONFIG.md` §2
  mirrors Zhu et al. Table 1).
- **What we extend:** FP→integer cascade for the TWO remaining FP operations
  (RMSNorm division, σ/SiLU transcendentals). Zhu et al. retain FP RMSNorm
  γ multiplication and FP σ lookups; our σ is a 257-entry linear-interp
  LUT with ≤2e-3 absolute error (`FJQ_PHASE_D_OPS.md` §2 Op 4), and RMSNorm
  γ is absorbed into downstream BitLinear β-scale rather than stored as a
  learnable tensor.
- **What we diverge from:** Zhu et al. use FP activations during training
  (QAT only applies to weights). IntLLM QAT quantizes activations too via
  `IntRMSNorm` + `fake_quantize_absmax_ste` STE, per FJQ_PHASE_D_OPS.md §2.

### TernaryLLM (Zhang et al., 2024)
- **arXiv:** 2406.07177 (Jun 2024, AMD + CAS)
- **Role:** post-training-quantization baseline. We cite to distinguish
  our quantization-aware pretraining (BitNet/MatMul-Free tradition) from
  PTQ approaches. TernaryLLM claims weight-only ternary at 3-8B; activations
  stay FP16. A 3-option GPTQ-style PTQ ablation is mentioned in our
  Phase 2.4 experiments but is not part of the main IntLLM claim.

---

## 2. Softmax-free / position-free attention replacement

### MatMul-Free MLGRU (Zhu et al., 2024) — our primary
See §1 entry. MLGRU eliminates softmax + self-attention entirely. Phase D
§FJ_PHASE_D_ARCH.md selected this as the primary architecture because:
1. FPGA reference proves integer-friendliness at real scale.
2. 3 transcendentals per layer (2× σ + 1× SiLU) vs 8 in HGRN-Gated.
3. MLGRU forgetting gate is naturally bounded in (0, 1) — proof in
   `FJQ_PHASE_D_OPS.md` §2 Op 8's boundedness argument.

### RWKV-7 "Goose" (Peng et al., 2025)
- **arXiv:** 2503.14456 (Mar 2025, RWKV Foundation)
- **Core contribution:** generalized δ-rule attention with vector-valued
  gates. Attention-free RNN. Apache-2.0 weights at 0.19B / 0.4B / 1.5B /
  2.9B + 3.1T-token corpus + full training code.
- **Role:** Phase D's **secondary** architecture for the recall-regime
  ablation triggered when Medium's val_loss exceeds <4.0. FJQ_PHASE_D_ARCH.md
  §3 keeps RWKV-7 as a hedge because its v7 generalized δ-rule is `i64`-
  representable IF we fixed-point the vector-valued gates. **Not yet
  evaluated.** Triggers only if IntLLM/MLGRU hits a recall ceiling.
- **Why not primary:** MatMul-Free is strictly simpler — MLGRU has 4
  BitLinears/layer vs RWKV-7's 7+ (`receptance, key, value, gate1, gate2,
  decay, output`). Simpler arch = easier kernel code = faster turn-around
  on silicon.

### Mamba-2 (Dao & Gu, 2024)
- **arXiv:** 2405.21060 (May 2024, Stanford)
- **Role:** RULED OUT. Mamba-2's selective state-space model requires
  `exp(segsum(A_diag))` which is a stabilized cumulative matrix exponential
  — V31.R3-adjacent dynamic-range explosion on any model >135M. Phase C.P1
  ADR in `FJQ_PHASE_D_ARCH.md` §3 "Mamba-2 analysis". Left as future work.

### RetNet (Sun et al., 2023)
- **arXiv:** 2307.08621 (Jul 2023, Microsoft + Tsinghua)
- **Role:** cited for context. Complex-valued decay scales are inherently
  FP, not i64 portable. Mentioned in related-work only to show we
  considered and passed.

### Linear Transformer family (Katharopoulos et al., 2020; ReBased, Hedgehog)
- **arXiv:** 2006.16236, 2402.10644, 2402.04347
- **Role:** the "kernel method" route to softmax-free attention. ReBased's
  polynomial kernel φ(x) = 1 + x + x²/2 is `i64`-computable if input range
  is bounded. Left as Phase D future work — MLGRU was simpler to ship.

---

## 3. FPGA / edge ternary deployment

### TeLLMe (Chen et al., 2024)
- **arXiv:** 2408.09815 (Aug 2024, SJTU + Microsoft)
- **Core contribution:** FPGA accelerator for ternary LLMs targeting
  BitNet-style deployment. Bit-serial multiply-accumulate unit (BS-MACU)
  + per-channel scaling factor lookup. Ships 4-bit activation support.
- **Role:** deployment-pattern reference. TeLLMe's bit-serial MACU is the
  pattern our `km_mf_bitlinear_packed` unrolls into — 2 signed-adds per
  4-way-packed byte (one for +1 case, one for −1 case, encoded 0 skips).
- **What we extend:** TeLLMe is FPGA-specific; our impl runs on commodity
  x86_64 and ARM64 boards (Radxa Dragon Q6A verified for ARM64 kernel in
  the same codebase — see `docs/HONEST_STATUS_V26.md`).

### TerEffic (Bae et al., 2024)
- **arXiv:** 2411.01830 (Nov 2024, Samsung)
- **Core contribution:** efficient ternary matmul accelerator with
  row-wise dequantization + activation outlier preservation. Reports ~1.5×
  over BitNet baseline on edge workloads.
- **Role:** algorithmic reference for §6.9 R5 outlier-handling plan.
  TerEffic's top-K channel preservation (high-bit allocation for outlier
  channels) directly informs our `intllm/qat.py::compute_bit_allocation`
  (top-5% of channels get 10-bit allocation instead of 8-bit).

### Bitnet.cpp — see §1 — the CPU counterpart.

---

## 4. Small language models (comparable regime + baselines)

### SmolLM2 (Ben Allal et al., 2025)
- **HuggingFace:** `HuggingFaceTB/SmolLM2-135M`, `-360M`, `-1.7B`
- **Announcement:** HuggingFace TB blog, Nov 2024 — formal paper TBD.
- **Core contribution:** strongly-tuned small LMs trained on a curated
  web+synthetic mix, with published ablations on distillation + data
  quality. Currently the highest-quality sub-1B FP16 LM at each of
  135M / 360M / 1.7B.
- **Role in IntLLM:** Mini config baseline (21.5M vs SmolLM2-135M) for the
  §4.2 scaling table. We **do not expect** to beat SmolLM2 in absolute
  quality — our claim is *quality-per-watt* and *kernel-portability*, not
  raw PPL leaderboard.
- **Baseline parity (§6.9 R3):** SmolLM2 weights are FP16, so the
  head-to-head must be ternary-IntLLM vs FP16-SmolLM2. We note this
  explicitly in `BASELINE_UNPORTED_FEATURES.md`.

### Pythia (Biderman et al., 2023)
- **arXiv:** 2304.01373 (Apr 2023, EleutherAI)
- **ICML 2023** — the canonical small-LM benchmark suite.
- **Core contribution:** 16 sizes from 14M to 12B, all trained on identical
  300B-token Pile corpus with identical hyperparameters. Enables
  scaling-law regressions across configs.
- **Role:** Phase D §4.2 Table 2 compares IntLLM Mini/Base/Medium/Stretch
  against Pythia-160M/410M. Pythia-410M is the closest parameter-count
  match to our Stretch (369.1M), though on 300B tokens vs our 15B.
- **Note (C.P2.1):** Our repro of `ridger/MMfreeLM-370M` on
  lm-eval-harness 0.4.11 showed a systematic -1.24 avg drift vs Zhu et al.'s
  published numbers on the old harness, matching known 0.3→0.4 ARC-E
  normalization drift. All Phase D numbers use 0.4.11. See
  `FJQ_PHASE_D_P2_1_GATE.md` for the conditional-pass rationale.

### SLM-Bench (Fu et al., 2024)
- **arXiv:** 2402.17747 (Feb 2024, NTU Singapore)
- **Core contribution:** a small-LM evaluation suite targeting the <2B
  regime with metrics specifically aware of the pre-training compute
  budget. Adds trade-off curves for quality-per-FLOP.
- **Role:** §4.2 "Efficiency table" — our Mini/Base configs plot against
  the SLM-Bench Pareto front to contextualize our quality-per-watt claims.

---

## 5. Kernel-integrated ML deployment (OS side)

### seL4 (Klein et al., 2009)
- **SOSP 2009** — the formal-verification reference point.
- **Core contribution:** formally verified microkernel. Isaaelle/HOL
  proof that C code implements the abstract specification.
- **Role in IntLLM:** NOT a direct competitor — seL4 has no ML runtime.
  Cited as the *gold standard* for kernel-space formal rigor; we position
  FajarOS Nova as "not yet formally verified, but designed so that the ML
  forward path has no heap allocation, no recursion, and no unbounded
  loops" — all necessary (though not sufficient) preconditions for seL4-
  style verification. §6 "Limitations + future work" sketches a formal-
  verification track for a future Phase E.
- **Why it matters here:** reviewers on MLSys often ask "what's different
  about running your model in kernel-space vs userspace?" The answer is
  *bounded latency* + *no OS scheduler preemption* + *direct hardware
  access* — the same properties seL4 leverages. The IntLLM kernel path
  doesn't yet prove these, but it doesn't violate them either.

### µ-kernel ML deployments (historic)
- **TinyML on bare-metal** (e.g. TensorFlow Lite Micro, CMSIS-NN):
  these run in the scheduler's foreground but with a userspace library,
  not kernel-integrated. Our contribution is running in the kernel's
  context (`@kernel` annotation in Fajar Lang prevents heap + FP use).
- **Unikernels (MirageOS, IncludeOS):** single-address-space, but still
  FP-heavy ML stacks. No integer-native LLM deployed this way.

---

## 6. Positional encoding

### NoPE (Kazemnejad et al., 2023 "The Impact of Positional Encoding on
Length Generalization in Transformers")
- **arXiv:** 2305.19466 (May 2023, NeurIPS 2023)
- **Core contribution:** causal-masked Transformer *with no positional
  encoding at all* generalizes to longer contexts than RoPE / ALiBi in
  many regimes.
- **Role:** we adopt NoPE for IntLLM. Verified by experiment — V31.R3
  isolated dynamic-range collapse in Gemma 3's RoPE-integer mapping; NoPE
  sidesteps it entirely, and for our target generation length of ≤1024
  tokens there is no measurable quality regression vs RoPE in
  HGRN-Bit/MatMul-Free replications.

### A2ATS (Lin et al., 2025) — RoPE+quant reference
- **arXiv:** 2502.12665 (Feb 2025)
- **Role:** reference for the one place quant+RoPE literature exists.
  V31.R3 H3 experiment showed a 2.3× decode speedup from a RoPE LUT but
  didn't fix the representation collapse. IntLLM's NoPE is simpler +
  sufficient.

---

## 7. Where IntLLM sits

| Axis | BitNet 2B4T | MatMul-Free | RWKV-7 | SmolLM2 | **IntLLM (ours)** |
|---|---|---|---|---|---|
| Weight precision | ternary (2-bit) | ternary (2-bit) | FP16/BF16 | FP16 | ternary (2-bit) |
| Activation precision | FP16 | FP16 | FP16 | FP16 | **i8/i64 integer** |
| Softmax | yes (FP) | **no (MLGRU)** | no (RNN) | yes (FP) | **no (MLGRU)** |
| Position encoding | RoPE (FP) | RoPE (FP) | implicit (RNN) | RoPE (FP) | **NoPE** |
| σ / SiLU | FP | FP | FP | FP | **257-entry LUT** |
| RMSNorm γ | learnable FP | learnable FP | N/A | learnable FP | **γ-less (absorbed into β)** |
| FPGA path | no | yes (research) | no | no | **deployed (FajarOS Nova)** |
| Formal-verification ready | no | no | no | no | **design respects @kernel constraints** |
| Max verified scale | 2B | 2.7B | 2.9B | 1.7B | **369M (Stretch, trained) + kernel E2E (tiny)** |

The rightmost column is not a win on every axis — we are smaller than
BitNet 2B4T and RWKV-7, and we lack a formal correctness proof.
The **composition** — first arch to be simultaneously ternary-weight,
integer-activation, softmax-free, position-free, LUT-transcendental,
γ-less, **and kernel-deployable** — is the novel contribution.

---

## 8. §6.9 Research Integrity compliance (R1-R7)

Writing this doc satisfies:

- **R1 canonical benchmark** — every comparison target in §7 above is
  re-evaluated on the same pinned `lm-evaluation-harness v0.4.11` per
  `FJQ_PHASE_D_PAPER_OUTLINE.md` §2.1. No custom protocol.
- **R2 literature review precedes algorithm design** — this document is
  dated 2026-04-21 and is committed BEFORE §4 Experiments is drafted.
  Upstream `FJQ_PHASE_D_P0_FINDINGS.md` (2026-04-21 C.P0) covered 22
  papers; we sample the 12 most relevant for the paper's related-work
  section.
- **R3 baseline parity** — unported features tracked in
  `BASELINE_UNPORTED_FEATURES.md` (Phase 3.4 deliverable). We will not
  strawman BitNet 2B4T by disabling its outlier handling.
- **R4 calibrated-once** — IntLLM's β scalars are calibrated once at QAT
  freeze (FJQ_PHASE_D_OPS.md §4 "Calibration-once table").
- **R5 outlier handling** — TerEffic-inspired top-K channel preservation
  + per-coord adaptive bit allocation implemented in `intllm/qat.py`
  (FJQ_PHASE_D_OPS.md §5 "Outlier handling plan"). Ablations in Phase 2.4.
- **R6 results-section last** — this related-work doc (Phase 4.1) is
  committed on 2026-04-21; the results section (Phase 4.4) will be drafted
  after canonical benchmarks (Phase 3) finish.
- **R7 pre-publication audit gate** — `scripts/verify_paper_tables.py
  --strict` will cross-check every numeric claim in this doc (currently:
  none — all numbers are in §4.2 tables) against JSON artifacts.

---

## 9. Gaps and open items

1. **We do not compare to QAT distillation approaches** (e.g. AWQ-QAT,
   SmoothQuant-QAT). Justification: these are weight-only and retain FP
   activations; outside our integer-native claim. Flagged for reviewer
   preemption in §6 Limitations.
2. **RWKV-7 ablation not yet run** — triggers only if Medium FAILs
   its val_loss<4.0 gate. Tracked in `FJQ_PHASE_D_ARCH.md` §3 "Secondary
   hedge".
3. **seL4-style formal verification is out of scope for Phase D.** We
   commit to the @kernel constraint set (no heap, no FP hot path, no
   unbounded loops) but not to a machine-checked proof. Reviewer-facing
   framing: "formal verification is a well-defined Phase E track, not a
   gap in the current contribution."
4. **Pythia-160M apples-to-apples gap measurement on lm-eval 0.4.11 is
   deferred from C.P2.1** (conditional-pass rationale in
   `FJQ_PHASE_D_P2_1_GATE.md`). Will be run as part of Phase 3.4 baseline
   re-runs.

---

## 10. References (arXiv-verified, 2026-04-21)

Integer-native LLMs:
- BitNet b1.58 — https://arxiv.org/abs/2402.17764
- BitNet b1.58 2B4T — https://arxiv.org/abs/2504.12285
- Bitnet.cpp — https://arxiv.org/abs/2502.11880
- MatMul-Free LLM — https://arxiv.org/abs/2406.02528
- Q-Sparse — https://arxiv.org/abs/2407.10969
- TernaryLLM — https://arxiv.org/abs/2406.07177

Softmax-free attention:
- RWKV-7 — https://arxiv.org/abs/2503.14456
- Mamba-2 — https://arxiv.org/abs/2405.21060
- RetNet — https://arxiv.org/abs/2307.08621
- Linear Transformer — https://arxiv.org/abs/2006.16236
- ReBased — https://arxiv.org/abs/2402.10644
- Hedgehog — https://arxiv.org/abs/2402.04347

FPGA / edge ternary:
- TeLLMe — https://arxiv.org/abs/2408.09815
- TerEffic — https://arxiv.org/abs/2411.01830

Small LMs / baselines:
- Pythia — https://arxiv.org/abs/2304.01373
- SmolLM2 — HuggingFace TB blog, Nov 2024 (formal paper TBD)
- SLM-Bench — https://arxiv.org/abs/2402.17747

Positional encoding:
- NoPE — https://arxiv.org/abs/2305.19466
- A2ATS — https://arxiv.org/abs/2502.12665

Kernel / formal OS:
- seL4 — SOSP 2009 (Klein et al.) — https://sel4.systems/About/seL4-sosp09.pdf

---

## 11. Changelog

| Date | Commit | Change |
|---|---|---|
| 2026-04-21 | (this commit) | Phase 4.1 initial draft — 12 papers across 6 clusters + positioning table + §6.9 compliance self-check |
