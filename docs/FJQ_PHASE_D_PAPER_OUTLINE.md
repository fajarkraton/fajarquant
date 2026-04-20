# FajarQuant Phase D — Paper Outline (C.P1.3)

> **Date:** 2026-04-21 | **Target venue:** MLSys 2027 (primary) / NeurIPS MLSys Workshop 2026 (fallback) / TinyML Summit 2027 (industry track) | **Rule compliance:** §6.9 R6 (results-section text is LAST) + §6.9 R7 (mechanical verify script pre-pub)

## 0. Metadata + submission targets

- **Title (working):** *"IntLLM: A Fully Integer-Native Language Model for CPU Deployment"*
- **Alternate:** *"Ternary Language Modeling Without Floating Point: From Training to Kernel"*
- **Page budget:** 10 pages main + unlimited references + unlimited supplementary (MLSys template)
- **LaTeX template:** `mlsys2027` (fallback: `NeurIPS 2026` if MLSys rejects → workshop path)
- **Supplementary:** reproducibility artifact, extended ablations, kernel SystemVerilog-parity port
- **Reproducibility statement:** required by MLSys — `reproduce.sh` with 4 modes (smoke / verify / full / fallback) per V26 C3.2 precedent
- **Authorship:** Muhamad Fajar Putranto (primary) + contributors TBD
- **Code release:** `github.com/fajarkraton/fajarquant` Apache-2.0 (already the PhaseD home)

## 1. Abstract (200 words, written LAST per §6.9 R6)

4-sentence template — numbers remain as placeholders until C.P6 measures:
1. Motivation: LLM deployment on edge + research OS kernels demands
   integer-only arithmetic, which existing quantized LLMs do not provide
   (softmax, RoPE, LayerNorm γ cascade remain FP).
2. Method: IntLLM extends the MatMul-Free LLM architecture with
   quantization-aware training (FajarQuant), explicit outlier handling,
   and a 600-LOC kernel port for the FajarOS Nova research OS.
3. Results: Achieve **[X] PPL** on Wikitext-103 at **[Y] params**,
   within **[Z]** of FP16 Transformer++ baseline, at **[W]** watts on
   consumer x86-64 CPU. No softmax, no RoPE, no FP operations in the
   forward path.
4. Contribution: First end-to-end integer-native LLM demonstrated from
   QAT to kernel-integrated research-OS deployment.

## 2. Section skeleton

### §1. Introduction (~1.5 pages)

**Narrative arc:**
- LLM deployment on ultra-constrained hardware (embedded, edge, research OS
  kernels) is gated by floating-point residue in modern architectures.
- Prior work (BitNet b1.58, MatMul-Free LLM) quantizes weights to ternary
  but retains FP softmax, FP RoPE, FP LayerNorm γ — the architectural
  impedance mismatch that causes representation collapse under aggressive
  fixed-point quantization (forward reference to §6.3 collapse analysis).
- We contribute IntLLM: a fully integer forward path from embedding to
  argmax, validated at **[Y] params** on a research OS kernel running on
  consumer x86-64 CPUs.

**Contributions list (3-5 bullets, finalize after C.P6):**
1. First language model with a fully integer-native kernel forward pass on
   general-purpose hardware (not FPGA).
2. QAT recipe with per-coord adaptive bit allocation + channel reordering
   (ablated vs static BitNet recipe).
3. Canonical-protocol evaluation on Wikitext-103 + LAMBADA + MMLU that
   MatMul-Free LLM's original paper did not publish.
4. FajarOS Nova kernel integration: first LM native to a research OS
   shell environment (boot → model-load → `ask` workflow).
5. Ablation vs RWKV-7 + Mamba-2 + BitNet b1.58 at matched params (if
   recall trigger fires — see §6.1).

**Figure 1:** Architecture overview — input token → embed → N × (RMSNorm
→ BitLinear-f/c/g/o → MLGRU state → residual → RMSNorm → BitLinear-g/u/d
→ GLU → residual) → RMSNorm → BitLinear-LM-head → argmax. Every op
labeled with dtype (i8 / i32 / i64-temp) and LUT dependencies.

### §2. Background + related work (~1.5 pages)

Three subsections:

**§2.1 Low-bit LLM quantization** — BitNet family, OneBit, DB-LLM,
TernaryLLM, Q-Sparse. Common theme: quantize weights (sometimes
activations) but retain FP softmax/LayerNorm. Positioning: IntLLM
extends ternary to the **full forward path**, not just weights.

**§2.2 Softmax-free attention architectures** — Mamba/Mamba-2 SSMs,
RWKV family, Linear-Transformer, Hedgehog, ReBased, Gated Linear
Attention. Positioning: all remove softmax from attention, but most
still use FP norms/rotations; IntLLM combines softmax-free with
ternary + integer norms.

**§2.3 Integer-friendly positional encoding** — NoPE, ALiBi, T5
relbias. Positioning: IntLLM uses NoPE (MLGRU recurrence carries
position implicitly) — zero-op positional solution.

Include 1 table: comparison matrix of 10 prior methods across 5 axes
(ternary weights / int activations / no softmax / no RoPE / verified
open ≥1B checkpoint). IntLLM is the only row with all 5 ✓.

### §3. Method (~2.5 pages — core contribution section)

**§3.1 IntLLM architecture (fig + formulas)** — reproduce MLGRU + GLU +
BitLinear + RMSNorm equations from `FJQ_PHASE_D_OPS.md`. Fig 2: single
decoder block with per-op dtype annotations.

**§3.2 Integer ops via LUT + fixed-point** — σ LUT construction,
isqrt via Newton iteration (reused from V29), x1000 fixed-point
convention, overflow bounds table (from `FJQ_PHASE_D_OPS.md` §7).

**§3.3 QAT recipe** — three novel components:
- Per-coord adaptive bit allocation (§3.3.1)
- Ternary-aware channel reordering (§3.3.2)
- Periodic γ_x re-calibration (§3.3.3)

**§3.4 Kernel integration** — FajarOS Nova architecture recap (2
paragraphs), kernel op inventory (`matmulfree.fj` + `tfm_matmulfree.fj`
+ existing kmatrix extensions), compile-time safety (@kernel +
@no_vectorize guards per V31.B.P2).

### §4. Experiments (~1.5 pages)

**§4.1 Setup:**
- Training: RTX 4090 16 GB, SlimPajama subset, Mistral v3 tokenizer
  (32K vocab; TBD in C.P3.1)
- Configs: Mini 8M / Base 25M / Medium 50M / Stretch 370M per
  `FJQ_PHASE_D_CONFIG.md` §5
- Baselines: Transformer++ repro per Zhu et al. Table 1 (primary) +
  BitNet b1.58 at matched params (if trainable at budget) + optional
  RWKV-7 if recall trigger fires
- Evaluation: Wikitext-103 PPL + LAMBADA + MMLU 5-shot + ARC-C +
  HellaSwag + needle-in-haystack (follow RWKV-7 protocol exactly for
  §6.9 R1 canonical compliance)

**§4.2 Main results:**
- Table 1: 5 metrics × 4 model sizes × 2 architectures (IntLLM vs
  Transformer++). All numbers gated on C.P6 measurement (§6.9 R6).
- Table 2: IntLLM FP16-shadow vs IntLLM-QAT (measures quantization
  loss). Gap target: ≤ 1 PPL at Base, ≤ 2 PPL at Stretch.

**§4.3 Kernel-path E2E:**
- Table 3: FajarOS Nova boot → model-load → tok-load → `ask` workflow
  latency + tokens/sec + watts, at all 4 configs.
- Figure 3: Scaling curve — IntLLM tok/sec vs d_model on single-core
  consumer x86-64.

### §5. Ablations (~1 page)

**§5.1 QAT component ablation:**
- Baseline static BitNet recipe (no outlier handling)
- +per-coord adaptive bits
- +channel reordering
- +periodic γ_x re-calibration (full IntLLM)

Table 4: PPL at Base 25M/1B for each ablation row. Measures the
contribution of each §3.3 component.

**§5.2 Architecture ablation** (conditional, triggered only if §6.1
recall fails):
- IntLLM (MLGRU) baseline
- IntLLM with RWKV-7 time-mix replacing MLGRU (same ternary QAT)
- IntLLM with Mamba-2 SSM replacing MLGRU (same ternary QAT)

Table 5: PPL + MMLU at Base 25M/1B for each architecture.

**§5.3 Fixed-point scale sensitivity:**
- x100 scale, x1000 scale, x10000 scale
- Measures whether the x1000 convention is close to optimal for this
  architecture, or if a wider fixed-point helps.

### §6. Limitations + future work (~0.5 page)

**§6.1 Recall ceiling.** MLGRU's fixed-dimensional state is a known
limitation of linear RNNs on long-context factual recall. Needle-in-
haystack results at Stretch 370M may show degradation past ~2K
context; we document and discuss.

**§6.2 Scale ceiling.** Phase D validates up to 370M (with 1B stretch
if §4.2 Table 1 Base numbers clear canonical gate). 7B+ scaling is
future work — extrapolation from §4 scaling law only.

**§6.3 Why representation collapse did not occur** (backward reference
to §1 motivation). Mechanical analysis: β is always shrinking, no γ
magnification in the norm path, no softmax exp-accumulation. We
demonstrate (in §4.3) that the FajarOS Nova kernel produces stable
per-token representations across 64-token generations, in contrast to
the Gemma 3 1B deployment that motivated this work.

### §7. Conclusion (0.25 page)

Three bullets. Per §6.9 R6, write this LAST (after §4 Table 1).

## 3. Figure + table inventory (final count for §6.9 R7 verify script)

| # | Label | Source | Section |
|---|---|---|---|
| Fig 1 | Architecture overview with per-op dtype | New LaTeX | §1 |
| Fig 2 | Single decoder block with equations | New LaTeX | §3.1 |
| Fig 3 | Tok/sec vs d_model scaling curve | `bench_cpu.py` output | §4.3 |
| Table 1 | Comparison matrix of 10 prior methods | Manual + refs | §2 |
| Table 2 | Main results PPL+downstream | `eval_harness.py` | §4.2 |
| Table 3 | FP16-shadow vs IntLLM-QAT gap | `eval_harness.py` | §4.2 |
| Table 4 | FajarOS Nova E2E metrics | `test-intllm-e2e` | §4.3 |
| Table 5 | QAT ablation | `ablate_qat.py` | §5.1 |
| Table 6 | Arch ablation (conditional) | `ablate_arch.py` | §5.2 |
| Table 7 | Fixed-point scale sensitivity | `ablate_scale.py` | §5.3 |

`verify_paper_tables.py --strict` (shipped in C.P7) enforces every
numerical cell in Tables 2-7 is backed by an `eval_harness.py` JSON
artifact with a committed SHA.

## 4. Reproducibility artifact (C.P7 deliverable)

`reproduce.sh` with 4 modes (precedent: V26 FajarQuant v3.1 shipped
this successfully):

- `--smoke` (5 min) — tiny config proves training + eval harness works
- `--verify` (30 min) — Base 25M run from `.fjm` checkpoint, PPL within 1
- `--full` (8h on RTX 4090) — train Base 25M, measure all Table 2/3/4
- `--fallback` (offline) — read frozen JSON artifacts, no compute

## 5. Related work depth target

~40 citations. Grouped in BibTeX by theme:
- Weight-quantization (BitNet et al.) — ~10
- Softmax-free attention (Mamba/RWKV/Linear) — ~10
- Positional encoding (RoPE/ALiBi/NoPE) — ~5
- Kernel deployment + edge inference (llama.cpp, Bitnet.cpp, TinyML) — ~10
- OS-integrated AI (rare; may not find 5) — ~5

## 6. Submission milestone gate

Per V31 Master Plan §4.8 (C.P7):
- First draft: C.P7.1 (3d)
- Internal review: §6.9 R7 verify script green
- External review: 2 researchers outside FajarQuant (TBD)
- Camera-ready: after canonical PPL numbers stabilize (§6.9 R6)

Gate: **no submission until §6.9 R1+R3+R7 are all green**. Section text
that survives the verify-script check is what ships; anything else is
cut.

## 7. Dependencies unlocked by this outline

- C.P1.4 config matrix formalization — ready
- C.P2.1 PyTorch reference fork — ready
- C.P3.1 tokenizer decision — ready (outline uses "Mistral v3 or SmolLM
  49K TBD")
- C.P7.1 paper draft — sectioning locked; waits for §4-5 numbers

## 8. Risks + mitigations

| Risk | Mitigation |
|---|---|
| MatMul-Free LLM architecture already partially published — novelty question | §3.3 QAT recipe + §3.4 kernel integration + §4.3 FajarOS E2E are the novel contributions; position clearly in §1 |
| Recall ceiling at Base/Stretch blocks §4 Table 2 | §6.1 has a dedicated subsection; §5.2 arch ablation gives a data-backed explanation |
| MLSys reviewer asks "why not Mamba-2" | §2.2 + §5.2 answer this; §5.2 ablation has Mamba-2 row triggered if needed |
| Scale ceiling 2.7B looks unambitious vs 7B+ results elsewhere | §6.2 is upfront; compensate with §4.3 real FajarOS kernel deployment that no other paper has |
| Reviewer requests int-8 accumulator instead of i64 | §3.2 fixed-point scale ablation (§5.3) covers this |
