# FajarQuant Phase D — P0 Findings (Literature Sweep)

> **Date:** 2026-04-21 | **Papers surveyed:** 22 verified (9 integer-LLM + 8 softmax-free attention + 5 PE) + 3 flagged unverified | **Rules:** CLAUDE.md §6.8 R1 (pre-flight audit), §6.9 R2 (≥8-10 paper sweep)

## 0. Scope and motivation

Phase D targets a transformer-class language model whose kernel forward pass
runs in **pure i64 fixed-point arithmetic**, without softmax, RoPE, or
FP-precision-sensitive ops. Motivation comes from V30 Gemma 3 port +
V31.R3 closure: pad-collapse is architectural, driven by fp16-trained
large-gamma RMSNorm (Gemma 3 peak γ≈66.5) × 104 norm applications in
x1000 fixed-point — representation direction is lost.

This P0 sweep surveys three parallel question areas to inform C.P1 arch
design:
1. **What weight precision regimes have been published at scale?**
2. **What attention mechanisms avoid softmax entirely?**
3. **What positional encodings are expressible in integer-only ops?**

Methodology: parallel WebSearch + WebFetch with arXiv-ID verification.
Future-dated or unverifiable IDs are flagged, not cited.

---

## 1. Integer-Native / Ternary / 1-bit LLMs (Section 1 landscape)

**Integer-ness score:** 0=pure FP, 1=low-bit weights + FP scales/activations,
2=low-bit weights + int-quantized activations but softmax/RoPE still FP,
3=fully integer forward path (no softmax, no transcendentals).

| cite-key | Title | arXiv | Venue/Year | Core contribution | Int | Phase D role |
|---|---|---|---|---|---|---|
| **BitNet-b1.58** | The Era of 1-bit LLMs | 2402.17764 | arXiv Feb 2024 (Microsoft) | Ternary {-1,0,+1} weights via BitLinear; matches FP16 LLaMA from 3B | 2 | Build on (weight encoding only; retains RoPE + softmax) |
| **BitNet-2B4T** | BitNet b1.58 2B4T Technical Report | 2504.12285 | arXiv Apr 2025 (Microsoft) | First open-source native 1.58-bit 2B LLM, 4T tokens; HF weights + kernels | 2 | Build on (validates ternary at 2B/4T scale). **Still FP softmax+RoPE.** |
| **Bitnet.cpp** | Efficient Edge Inference for Ternary LLMs | 2502.11880 | arXiv Feb 2025 (Microsoft) | x86/ARM CPU runtime, 2-6× speedup | 2 | Benchmark target |
| **OneBit** | Towards Extremely Low-bit LLMs | 2402.11295 | NeurIPS 2024 | 1-bit sign matrix + two FP scale vectors (SVID) | 1 | Diverge (FP scales defeat i64 goal) |
| **DB-LLM** | Dual-Binarization for Efficient LLMs | 2402.11960 | ACL Findings 2024 | 2-bit as two binary matrices + KD | 1 | Diverge (PTQ, not arch) |
| **MatMul-Free-LM** | Scalable MatMul-free Language Modeling | 2406.02528 | arXiv Jun 2024 | Ternary BitLinear + **MLGRU replaces self-attention**. FPGA at 13W, tested to 2.7B | **3** | **Build on — purest i64 candidate.** Code at `ridgerchu/matmulfreellm` |
| **Q-Sparse** | All LLMs can be Fully Sparsely-Activated | 2407.10969 | arXiv Jul 2024 (Microsoft) | Top-K activation sparsification, composes with BitNet | 2 | Compose as optional layer |
| **TernaryLLM** | Ternarized Large Language Model | 2406.07177 | arXiv Jun 2024 (AMD/CAS) | PTQ ternary via Dual Learnable Ternarization + KD | 1 | Diverge (weight-only, FP activations) |

**Key finding — integer-LLM landscape has two disjoint camps:**

- **Camp A (weight-only ternary):** BitNet family, OneBit, DB-LLM,
  TernaryLLM. Ternary weights but retain FP softmax + FP RoPE + FP
  RMSNorm gamma. Verified scale up to 2B params. Phase D pad-collapse
  root cause (large-gamma × many-norm accumulation) **is inherited**.
- **Camp B (pipeline-native integer):** MatMul-Free-LM is the sole member.
  Removes softmax AND matmul via MLGRU substitution. Max verified
  scale 2.7B. Direct i64 mapping possible.

No published method simultaneously:
  (a) removes softmax, (b) removes RoPE, (c) uses ternary weights,
  and (d) ships verified checkpoints at ≥1B params with instruction
  tuning. **This is the Phase D gap.**

---

## 2. Softmax-Free / Linear-Time Attention (Section 2 landscape)

| cite-key | Title | arXiv | Venue/Year | Attention formulation | Shipped ≥1B? | Phase D role |
|---|---|---|---|---|---|---|
| **Linear-Transformer** | Transformers are RNNs | 2006.16236 | ICML 2020 | φ(Q)(φ(K)ᵀV), φ=elu+1 | No | Foundational reference |
| **RetNet** | Retentive Network | 2307.08621 | arXiv Jul 2023 (Microsoft) | Multi-scale exponential decay retention | Research ≤6.7B | Diverge (complex-valued decay) |
| **Mamba** | Selective State Spaces | 2312.00752 | COLM 2024 | Input-dependent SSM, no attention | **Yes — 2.8B open** | Build on (FP exp in discretization) |
| **Mamba-2** | SSMs via Structured State Space Duality | 2405.21060 | ICML 2024 | SSD framework, 2-8× faster than Mamba-1 | **Yes — Falcon-Mamba 7B, Codestral-Mamba 7B, Zamba 7B** | **Build on — largest deployed softmax-free ecosystem** |
| **RWKV-7 "Goose"** | Expressive Dynamic State Evolution | 2503.14456 | arXiv Mar 2025 | Generalized delta rule, attention-free RNN, vector gating | **Yes — 0.19B / 0.4B / 1.5B / 2.9B Apache-2.0 + 3.1T corpus + training code** | **Build on — highest verified scale + openness** |
| **GLA** | Gated Linear Attention | 2312.06635 | ICML 2024 | Data-dependent gating, 2D matrix hidden state | Research ≤2.7B | Build on alternative |
| **Hedgehog** | Expressive Linear Attentions with Softmax Mimicry | 2402.04347 | ICLR 2024 | MLP feature map mimics softmax | No | Diverge (MLP φ = FP-heavy) |
| **ReBased** | Taylor-kernel Linear Transformers | 2402.10644 | arXiv Feb 2024 | φ(x) = 1+x+x²/2 (polynomial) | No | Possibly build on — **polynomial kernel is i64-computable** if input range bounded |

**Key finding — three softmax-free regimes are production-viable:**

1. **SSM / state-space** (Mamba, Mamba-2): largest ecosystem, 7B
   instruction-tuned releases. Softmax-free by construction. FP
   transcendental remaining: `exp(Δ·A)` in discretization — replaceable
   with Taylor-1 (`1 + Δ·A`) already standard in discrete-time SSM.
2. **Attention-free RNN** (RWKV-7): highest verified open scale (2.9B)
   with full training code + multilingual corpus. Delta rule uses
   only element-wise ops + sigmoid gate. Sigmoid is the main
   transcendental; replaceable with piecewise-linear or LUT.
3. **Polynomial linear attention** (ReBased): kernel φ(x) = 1+x+x²/2
   is pure polynomial, computable in i64 with bounded-input
   guarantees. Smaller verified scale but closest to "compute-trivial"
   for integer kernels.

Hedgehog is explicitly out: its MLP feature map is trained to mimic FP
softmax, defeating the goal.

---

## 3. Integer-Friendly Positional Encoding (Section 3 landscape)

| cite-key | Title | arXiv | Venue/Year | Mechanism | Integer? | Phase D role |
|---|---|---|---|---|---|---|
| **ALiBi** | Train Short, Test Long | 2108.12409 | ICLR 2022 | Scalar bias `-m·|i-j|` added to attention logits; fixed slopes per head | ✅ (integer subtract + bit-shift multiplier) | Build on *if* attention logits retained |
| **xPos** | Length-Extrapolatable Transformer | 2212.10554 | ACL 2023 (Microsoft) | RoPE + exponential decay `ζⁱ` | ❌ (sin/cos + exp) | Diverge — same FP problem as RoPE |
| **RoPE-integer variants** | A2ATS `2502.12665` et al. | 2502.12665 | arXiv Feb 2025 | KV-cache quant with windowed RoPE | Partial (underlying `sin(θ·i)` still FP) | Reference only. V31.R3 confirmed RoPE LUT gives 2.3× speedup but doesn't fix dynamic-range collapse |
| **NoPE** | Transformers without Positional Encodings Still Learn Position | 2203.16634 | Findings ACL 2022 | Causal mask alone suffices for decoder-only | ✅ (zero ops) | **Build on — strongest.** Mamba/RWKV-7 already NoPE |
| **T5 relbias** | Exploring the Limits of Transfer Learning | 1910.10683 | JMLR 2020 | Log-spaced relative distance buckets → learned scalar bias | ✅ (integer bucket + lookup) | Build on as secondary if attention logits retained |

**Key finding — NoPE is the default for softmax-free architectures.**
Mamba and RWKV-7 are both NoPE by construction (recurrent state carries
position implicitly). If Phase D chooses an attention-free recurrent
core, PE is a non-problem. If Phase D keeps some attention-logit path
(e.g., hybrid Mamba-Transformer), ALiBi + T5-relbias are the two
integer-computable fallbacks. RoPE is confirmed out.

---

## 4. Candidate Architecture Families for Phase D

Three families surfaced. None is an obvious winner — each optimizes a
different axis. Decision deferred to C.P1 (arch design phase).

### Candidate A — **RWKV-7 + ternary BitLinear + NoPE**

- Attention: RWKV-7 delta rule (attention-free RNN)
- Weights: BitNet b1.58 ternary
- Positional: NoPE (implicit via recurrence)
- Remaining transcendentals: sigmoid gate → PWL or LUT

**Strengths:** highest verified open scale (2.9B), training code + 3.1T
multilingual corpus released Apache-2.0, zero softmax/RoPE/sin/cos by
construction, 3B-class SoTA on multilingual.

**Risks:** delta-rule state accumulation needs i64 dynamic-range
analysis over long contexts. RMSNorm gamma × many-layer accumulation
(the V31.R3 pad-collapse cause) is partially inherited unless norm
layer is redesigned. RWKV community is smaller than SSM; fewer
downstream fine-tunes to transfer-learn from.

### Candidate B — **Mamba-2 + ternary weights + NoPE**

- Attention: Mamba-2 SSD (structured state-space duality)
- Weights: Ternary BitLinear on input/output/gating projections
- Positional: NoPE (SSM state implicit)
- Remaining transcendentals: `exp(Δ·A)` → Taylor-1 or PWL

**Strengths:** largest production ecosystem of softmax-free models
(Falcon-Mamba 7B, Codestral-Mamba 7B, Zamba 7B all instruction-tuned).
SSD structure maps cleanly to block-matmul i64 kernels. Closest drop-in
replacement for Gemma-3 inside FajarOS Nova chat workflow (tokenizer +
chat template transfer directly).

**Risks:** selective-scan kernel is non-trivial to rewrite in i64 (the
main Mamba-2 engineering cost). No open ternary-Mamba-2 release exists
— Phase D must train one. Some SSM variants show in-context-recall gap
vs Transformer at 7B+ (could affect tool-use workflows).

### Candidate C — **MatMul-Free LLM (MLGRU)**

- Attention: Element-wise Hadamard MLGRU replacing self-attention
- Weights: Ternary everywhere; no matmul anywhere in the model
- Positional: Implicit via GRU recurrence
- Remaining transcendentals: sigmoid/tanh in gates → PWL or LUT

**Strengths:** **only architecture in the sweep that removes softmax
AND matmul simultaneously by construction.** Direct fit for FajarOS
kernel's i64 constraint. FPGA at 13W demonstrates real low-power
feasibility at billion-param scale. Training code open.

**Risks:** max verified scale 2.7B (less than RWKV-7's 2.9B multilingual
instruct). Bigger architectural deviation from mainstream pretraining
recipes — instruction-tuning ecosystem does not transfer directly.
Single-paper body of work (higher research risk).

---

## 5. Phase D positioning statement

**IntLLM = first transformer-class language model with a fully
integer-native kernel forward pass.** Simultaneously:

1. **Softmax-free attention or attention-free recurrence** (removes the
   `exp()` + normalization chain that causes RoPE-level dynamic-range
   issues in integer kernels)
2. **NoPE or integer-only PE** (removes `sin/cos` LUTs)
3. **Ternary BitNet-class weights** (reduces matmul to integer sign ops)
4. **Integer RMSNorm variant** (sidesteps the V31.R3 pad-collapse root
   cause — large-gamma × many-norm representation-direction loss in
   x1000 fixed-point)

Closest prior art is **BitNet-b1.58-2B4T** (ternary weights, FP softmax
+ FP RoPE retained). IntLLM's novelty is extending integer-native
operation from the weight domain to the full forward path, resolving
the architectural impedance mismatch that V30 surfaced.

Closest conceptual ancestor is **MatMul-Free LLM**, which already
removes softmax + matmul. IntLLM extends that line with a broader
evaluation (≥2 canonical benchmarks per §6.9 R1), instruction tuning
(BitNet-2B4T recipe), and kernel-integrated deployment (FajarOS Nova
shell).

---

## 6. Flagged unverified items (do not cite in paper)

| arXiv ID | Search-surfaced as | Status |
|---|---|---|
| 2602.07374 | "TernaryLM: Memory-Efficient LM via Native 1.5-Bit" | Future-dated; abstract not fetched |
| 2601.02728 | "CRoPE: efficient RoPE parametrization for quantization" | Future-dated; needs author verification |
| 2601.17334 | "Power-based Partial Attention" | Future-dated; needs author verification |

Re-verify before any citation in the Phase D paper. arXiv occasionally
assigns IDs months in advance but papers may never appear or may
change substantially.

Additional data gap: **MatMul-Free LLM PPL values** — abstract claims
"comparable to SOTA Transformers up to 2.7B" but exact PPL figures
require PDF Table 2 fetch. Pull before C.P1 arch decision so Candidate
C's empirical case is quantified, not narrative.

---

## 7. Gate for C.P1

Per V31 Master Plan §4.2: C.P1 can start once this findings doc is
committed. C.P1 tasks:
- C.P1.1: Choose arch family (Candidate A/B/C) and commit decision in
  `FJQ_PHASE_D_ARCH.md`
- C.P1.2: Integer ops inventory (matmul, RMSNorm-int, gate-PWL,
  aggregate) with dtype + overflow bounds
- C.P1.3: Paper outline (MLSys template)
- C.P1.4: Config matrix (d_model ∈ {256, 384, 512}, L ∈ {6, 12, 18})

**Author recommendation (non-binding, for C.P1 consideration):**
Candidate A (RWKV-7 + ternary + NoPE) has the highest verified open
scale + training code + multilingual corpus, and the delta-rule state
update is the most i64-analyzable of the three. Candidate B (Mamba-2)
wins on deployment ecosystem but requires training a ternary variant
from scratch. Candidate C (MatMul-Free) is architecturally purest but
has the smallest empirical base. Final decision to include a concrete
ablation plan in C.P1.

---

## 8. References

### Integer-LLM
- BitNet b1.58 — https://arxiv.org/abs/2402.17764
- BitNet 2B4T — https://arxiv.org/abs/2504.12285
- Bitnet.cpp — https://arxiv.org/abs/2502.11880
- OneBit — https://arxiv.org/abs/2402.11295
- DB-LLM — https://arxiv.org/abs/2402.11960
- MatMul-Free LM — https://arxiv.org/abs/2406.02528
- Q-Sparse — https://arxiv.org/abs/2407.10969
- TernaryLLM — https://arxiv.org/abs/2406.07177

### Softmax-free attention
- Linear Transformer — https://arxiv.org/abs/2006.16236
- RetNet — https://arxiv.org/abs/2307.08621
- Mamba — https://arxiv.org/abs/2312.00752
- Mamba-2 — https://arxiv.org/abs/2405.21060
- RWKV-7 — https://arxiv.org/abs/2503.14456
- GLA — https://arxiv.org/abs/2312.06635
- Hedgehog — https://arxiv.org/abs/2402.04347
- ReBased — https://arxiv.org/abs/2402.10644

### Positional encoding
- ALiBi — https://arxiv.org/abs/2108.12409
- xPos — https://arxiv.org/abs/2212.10554
- A2ATS (RoPE + quant) — https://arxiv.org/abs/2502.12665
- NoPE — https://arxiv.org/abs/2203.16634
- NoPE length-generalization — https://arxiv.org/abs/2404.12224
- T5 — https://arxiv.org/abs/1910.10683
