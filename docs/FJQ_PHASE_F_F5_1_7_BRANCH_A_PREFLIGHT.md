# Phase F F.5.1.7 — Branch A (Canonical QuaRot) Pre-Flight Audit (v1.0)

> **Status:** ALL SECTIONS COMPLETE — §1 Literature Scan + §2 Code-Touch
> Estimate + §3 Risk Register + §4 Effort Sizing + §5 A/B Verdict.
> F.5.1.7 audit cycle CLOSED. **Verdict: CONDITIONAL NO-GO Branch A
> primary, GO Branch B default.** Mechanical Branch A re-entry gates
> at §6.4.
> **Origin:** F.5.1 closed PARTIAL on 2026-04-28 with HGRN-ternary trained
> model demonstrated INVARIANT to activation-only SmoothQuant. F.6.2
> closed FAIL with activation-only-pre-hook Hadamard rotation
> (+2.5–3.0 nat regression). Both verdicts strongly suggest Branch A
> (canonical QuaRot weight-fusion) is the next research-path
> candidate. Per CLAUDE.md §6.8 R1 (Pre-flight audit mandatory), this
> doc must complete BEFORE any Branch A impl scaffold lands.
> **Companion docs:**
> - `docs/FJQ_PHASE_F_F5_1_FINDINGS.md` v1.0 — F.5.1 closure (PARTIAL)
> - `docs/FJQ_PHASE_F_F5_1_SMOOTHQUANT_PTQ_DESIGN.md` v1.0 — design F.5.1 evaluated
> - `docs/FJQ_PHASE_F_TAX_VERTICAL_ROADMAP.md` v1.3 §4.1 F.5–F.6
> - F.6.2 commit `89f3219` (RTX 4090 results) + verdict correction `7fec08f`
> - V32-prep G5 gate `6987de3` — strengthens any future calibration-mutation
>   work, including Branch A.

---

## 1. Executive Summary

**Verdict (preliminary, finalized in §5):** TBD — verdict-tree decision
between A (canonical QuaRot) vs B (Phase F §4.2 hardware-acceleration
pivot) requires §1–§4 evidence before committing.

**Why this audit exists:** F.5.1 PARTIAL + F.6.2 FAIL both used
incomplete recipes. F.5.1 was activation-only (no rotation). F.6.2
was activation-only-pre-hook Hadamard (broke FP path: `y = W·Hx ≠ W·x`).
Neither verdict actually tested whether HGRN-ternary can be tamed by
the canonical QuaRot recipe (weight-fusion `W' = W·Hᵀ` + matched
residual rotation + γ_x recalibration). Without that test, deciding
A vs B is vibes, not data.

**What §1–§4 will establish:**
1. **Literature** — what published QuaRot impls actually do, what
   variants exist, what's been measured on architectures other than
   the Llama family.
2. **Code-touch** — concrete LOC delta and surgery sites in fajarquant.
3. **Risks** — failure modes Branch A inherits from F.5.1/F.6.2 + new
   ones unique to weight-fusion.
4. **Effort** — bottom-up estimate replacing the rough "3-5 days
   serius" current sizing.

**What §5 will deliver:** A vs B recommendation with hard data + a
mechanical entry condition (Phase F entry gate per §6.8 R6).

---

## 2. §1 Literature Scan — Canonical QuaRot

> **Section status:** v0.1 — synthesized 2026-04-28 from web research
> across 14 sources (papers + code repos + production tutorials).
> Sources at §2.7. Load-bearing finding: BitNet v2 Table 5 (§2.3).

### 2.1 Canonical QuaRot — landscape table

| Method | Rotation | Fusion site | γ handling | Residual handling | Reported gain |
|---|---|---|---|---|---|
| **QuaRot** [2404.00456] | Randomized Hadamard `H̃ = H·diag(s)`, `s ∈ {±1}`; Kronecker factorization for non-2ⁿ dims; primarily offline (R1/R2) + online (R3 attn, R4 FFN-down) | 4 sites: residual stream (`Q^T` left of inputs, `Q` right of outputs); MHSA Q/K/V; MHSA out-proj; FFN down-proj | RMSNorm linear part absorbed left into next W via `W_k ← Q^T · diag(α) · W_k` ("fuse_layer_norms") | Propagates across blocks then cancels via SliceGPT-style **computational invariance theorem** (orthogonal Q applied & inverted symmetrically) | Llama-2-70B W4A4: +0.47 PPL, −1.09pp zero-shot avg (~99% retention). Llama-2-7B W4A4: +0.63 PPL, −4.18pp |
| **SpinQuant** [2405.16406] | **Learned** R1/R2/R3/R4 via Cayley optimization on Stiefel manifold; initialized from random Hadamard; quantization-aware loss | Same 4 sites as QuaRot | Same fusion as QuaRot, but Q learned not random | Same invariance; rotations remain orthogonal throughout training | Llama-2-7B W4A4KV4: 2.9pp gap to FP, beats QuaRot. On Llama-3-8B: closes 45.1% of QuaRot's residual gap |
| **QuIP#** [2402.04396] | Randomized Hadamard for incoherence + E₈ lattice vector codebook + fine-tuning | Weight-only; 2-sided Hadamard on W (`H_L · W · H_R`) plus per-block scaling | LayerNorm fused; γ absorbed into adjacent W as in SliceGPT/QuaRot | N/A (weight-only PTQ; activations not rotated) | Llama-2-70B 2-bit: ~3.5 PPL on Wikitext-2, near-FP. Requires fine-tuning to land state-of-the-art at ≤2 bit |

### 2.2 Reference implementations

- **github.com/spcl/QuaRot** — Apache-2.0. Layout: `fake_quant/` (sim),
  `quarot/kernels/gemm.cu` (4-bit GEMM), `e2e/` (inference). Key file
  `fake_quant/rotation_utils.py` (~322 LOC) exposes
  `rotate_attention_inputs/output/mlp_input/mlp_output/head`,
  `fuse_layer_norms`, `get_orthogonal_matrix`. Hadamard construction in
  `hadamard_utils.random_hadamard_matrix`. Driver: `fake_quant/main.py`.
- **github.com/facebookresearch/SpinQuant** — CC-BY-NC-4.0
  (research-only). Layout: `optimize_rotation.py` (Cayley loop),
  `ptq.py` (post-training quant after rotations land), `train_utils/`,
  `eval_utils/`. **LLaMA-only** in tested configurations; no generic
  Transformer adapter, no RNN/Mamba/HGRN code path.
- **github.com/Cornell-RelaxML/quip-sharp** — open-source. Bundles
  fine-tuning loop on top of weight-only Hadamard + lattice codebook.
- **github.com/microsoft/BitNet** — official ternary inference.
  H-BitLinear in BitNet v2 uses Hadamard **online to activations
  only**, not fused into ternary weights (see §2.3).
- **AMD Quark** — productionizes QuaRot in
  `quark.docs.amd.com/.../tutorial_quarot.html` for MXFP4 + SmoothQuant
  composition; useful "what fusion order works in practice" reference.

### 2.3 Ternary composition — load-bearing finding

The most directly relevant prior is **BitNet v2** (arxiv 2504.18415),
the only published work composing Hadamard-style rotation with ternary
BitLinear weights at scale. Its Section 3.2 ablation (Table 5) compares
**weight + activation rotation** vs **activation rotation only** at
W1.58A8 and W1.58A4 on 1.3B and 3B models. Activation-only ties or
*beats* weight+activation at every cell (1.3B W1.58A8: 51.16 vs 50.47
acc; 3B W1.58A4: 55.43 vs 54.98). The paper states verbatim:
*"ternary models are more sensitive to the fusion of rotary matrix and
latent weights"* and adopts activation-only H-BitLinear in the final
design. **This is direct empirical evidence that canonical
weight-fusion rotation hurts ternary trained-from-scratch models** at
the BitNet scale.

Counter-direction: **ITQ3_S** (arxiv 2603.27914) uses FWHT pre-rotation
on ternary weights and reports gains, but it is **3-bit not 1.58-bit**,
weight-only PTQ with no QAT, and benchmarks against QuIP# rather than
BitNet/QuaRot — so the ternary regime there is much looser than ours
and isolated from the ±1 clipping cliff.

**Sparse-BitNet, OneBit, BitNet b1.58 2B4T**: none use rotation.
Sparse-BitNet (2603.05168) explicitly does not. OneBit uses SVID
(sign-value decomposition) — disjoint mechanism. The 2B4T release
ships no rotation in the public weights.

### 2.4 HGRN / RNN family

**MambaQuant** (arxiv 2501.13484) is the only published QuaRot-on-recurrent
paper found, and it is a cautionary tale: the abstract states *"QuaRot
suffers a 21% accuracy drop on Vim-T even under W8A8"* on Mamba —
rotation that nominally should be lossless at 8-bit catastrophically
fails on a recurrent gated architecture. The fix (Smooth-Fused rotation
+ KLT-enhanced rotation, equalizing channel variances first) recovers
the gap, but only at W8A8 and only with Mamba-specific surgery.
**MambaQuant explicitly does not extend to HGRN, RWKV, or RetNet.**
**No published QuaRot/SpinQuant evaluation exists on HGRN, HGRN2,
RWKV-6, RetNet, or GLA at any bit-width as of this audit.**

A separate datapoint from KurTail-style outlier-handling work: gated
paths (FFN down-proj input, attention output gating) are the **dominant**
outlier source in modern Transformers, and outliers in HGRN's gated
paths (`i_proj`/`f_proj`/`g_proj`) are precisely where the SmoothQuant
α-axis surfaced as architecture-invariant in F.5.1 — consistent with
MambaQuant's observation that recurrent gating equalization is
non-optional.

**HadamRNN** (OpenReview amOpepqmSl) parameterizes orthogonal RNN
*recurrent weights* with Hadamard structure — a different problem
(training-time architectural prior), not PTQ rotation, but confirms
ternary + Hadamard *can* be made stable when designed jointly from
scratch.

### 2.5 Implications for fajarquant Branch A

Three predictions for a correctly implemented canonical QuaRot
weight-fusion on our HGRN-ternary Mini config, ordered by confidence:

**(1) High-probability NEGATIVE on the W-only fusion path.** BitNet v2
Table 5 gives a near-clean reading of the question we're asking: at
the architecture closest to ours (ternary, gated, Transformer-with-RoPE),
weight-side fusion ties or hurts. Our F.6.2 result (+2.5–3.0 nat
regression on activation-only-pre-hook) was recipe-incomplete and
cannot decide; but the ternary-specific BitNet v2 finding suggests
the "complete recipe" verdict will land on **null-to-mildly-negative**,
not strongly positive. Combined with our verified RMSNorm γ
pre-absorption (which means QuaRot's `W_k ← Q^T · diag(α) · W_k` step
is a no-op modulo ordering — γ is already "in" W), the canonical
recipe loses the lever it pulls hardest on Llama.

**(2) Failure-mode prediction if it does regress: cumulative
saturation, not single-layer.** F.5.1 Run-7 (sites=all, α=0.3,
+0.27 nat) showed that ternary clipping accumulates multiplicative
perturbations in a way that single-layer gates miss. QuaRot fuses
into **every** linear, so the same accumulation surface applies. The
R-α regression `delta ≈ 0.135·α − 0.030` we observed at α=0.3 maps
roughly onto QuaRot's "implicit α" (residual-stream rotation
magnitude, ~unit-norm rows of H), giving a heuristic worst-case
forecast in the +0.10 to +0.30 nat regime. This is a 2–10× larger
budget than the F.5.1 best (+0.0001) and 10–30× smaller than F.6.2
(+2.5–3.0), placing canonical QuaRot in a "more honest fight" but
still on the wrong side of zero.

**(3) Plausible POSITIVE corner: HGRN gated paths only.** MambaQuant's
lesson is that recurrent + gated = needs variance equalization first,
then rotation. If we did Branch A but restricted weight-fusion to
`o_proj` and `g_proj` (the canonical QuaRot R3+R4 sites that are also
the PTQ-active sites in fajarquant), with γ-recalibration done
**after** ternary re-clipping rather than before, we might recover
the +0.0001 nat invariance and possibly nudge into +0.0005 territory.
This is "make the negative result less embarrassing", not "win the
paper". Effort: 3–5 days impl + 1 day sweep, matching the F.5.1
cycle envelope.

### 2.6 Audit-level recommendation

Branch A is worth running **only if the goal is a stronger negative
claim for the v2 paper revision** ("we tried canonical QuaRot at full
faithfulness; HGRN-ternary trained models reject weight-fusion
rotation just as they reject activation-only SmoothQuant") rather
than a positive result. If bandwidth is constrained, Branch B
(F.10–F.13 hardware acceleration) has unambiguously higher expected
ROI — the negative-result-strengthening branch can wait for the
post-acceptance revision cycle.

### 2.7 Sources

- [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs (Ashkboos et al., 2024)](https://arxiv.org/abs/2404.00456)
- [QuaRot HTML v1](https://arxiv.org/html/2404.00456v1)
- [github.com/spcl/QuaRot](https://github.com/spcl/QuaRot)
- [SpinQuant: LLM Quantization with Learned Rotations (Liu et al., 2024)](https://arxiv.org/abs/2405.16406)
- [github.com/facebookresearch/SpinQuant](https://github.com/facebookresearch/SpinQuant)
- [QuIP#: Even Better LLM Quantization (Tseng et al., 2024)](https://arxiv.org/abs/2402.04396)
- [BitNet v2: Native 4-bit Activations with Hadamard Transformation for 1-bit LLMs](https://arxiv.org/abs/2504.18415) — load-bearing
- [BitNet v2 HTML](https://arxiv.org/html/2504.18415v1)
- [MambaQuant: Quantizing the Mamba Family with Variance Aligned Rotation](https://arxiv.org/abs/2501.13484)
- [Sparse-BitNet: Naturally Friendly to Semi-Structured Sparsity](https://arxiv.org/html/2603.05168)
- [OneBit: Towards Extremely Low-bit LLMs (Xu et al., 2024)](https://arxiv.org/abs/2402.11295)
- [ITQ3_S: Interleaved Ternary Quantization with TurboQuant](https://arxiv.org/html/2603.27914)
- [HadamRNN: Binary and Sparse Ternary Orthogonal RNNs](https://openreview.net/forum?id=amOpepqmSl)
- [SmoothRot: Channel-Wise Scaling + Rotation](https://arxiv.org/html/2506.05413v2)
- [AMD Quark QuaRot tutorial](https://quark.docs.amd.com/latest/pytorch/tutorial_quarot.html)

---

## 3. §2 Code-Touch Estimate — fajarquant Branch A Surgery

> **Section status:** v0.1 — based on hands-on read of
> `_upstream/mmfreelm/{layers/hgrn_bit.py, models/hgrn_bit/modeling_hgrn_bit.py,
> modules/fused_norm_gate.py}` + `intllm/{model.py, quant.py}` 2026-04-28.

### 3.1 Architecture (good news: HGRN-bit is QuaRot-friendly)

HGRN-bit is **pre-norm Transformer-style** at the block level:

```
HGRNBitBlock.forward(h):
    residual = h
    h = attn_norm(h)              # block-level RMSNorm γ_attn
    h = attn(h)                   # i/f/g_proj → recurrence → g_norm → o_proj
    h, residual = mlp_norm(h, residual, prenorm=True)  # γ_mlp absorbed here
    h = mlp(h)                    # gate_proj → swiglu → down_proj
    h = residual + h
```

This pre-norm + residual structure is **identical** to Llama on which
canonical QuaRot was originally specified. The HGRN recurrent kernel
(`fused_recurrent_hgrn`) and the inner `g_norm` (FusedRMSNormSwishGate)
are TRANSPARENT to residual-stream rotation: the i/f/g_proj weights
fuse `H^T` to undo the residual rotation, the recurrence sees
unrotated values, the o_proj weight fuses `H` to write back into the
rotated residual frame. No kernel modifications needed.

### 3.2 Touch-site table

| # | Site | Math op | Files | LOC | Complexity |
|---|---|---|---|---|---|
| 1 | `embeddings.weight` (×1) | `W' = W · H` | `modeling_hgrn_bit.py` (read-only; mutate via Calibrator) | ~5 | trivial |
| 2 | `attn_norm.weight` per block (×n_layers) | absorb γ_attn into i/f/g_proj; set to 1.0 | same | ~10 | trivial |
| 3 | `attn.i_proj.weight`, `f_proj`, `g_proj` (×3×n_layers) | `W' = W · diag(γ_attn) · H^T` | same | ~15 | trivial |
| 4 | `attn.o_proj.weight` (×n_layers) | `W' = H · W` | same | ~5 | trivial |
| 5 | `mlp_norm.weight` per block (×n_layers) | absorb γ_mlp into mlp.gate_proj | same | ~10 | trivial |
| 6 | `mlp.gate_proj.weight` (×n_layers) | `W' = W · diag(γ_mlp) · H^T` (note: out_dim = 2×intermediate, gate+y interleaved chunks) | same | ~10 | moderate |
| 7 | `mlp.down_proj.weight` (×n_layers) | `W' = H · W` | same | ~5 | trivial |
| 8 | `lm_head.weight` (×1) | `W' = W · H^T` (BUT `lm_head` is `FusedBitLinear` with its own inner `.norm.weight` γ — see §3.3 below) | same | ~5 | moderate |
| 9 | Per-BitLinear inner `.norm.weight` (γ_inner, ×~37 sites) | LEAVE UNTOUCHED for v1; γ_inner is INSIDE BitLinear, acts on already-rotated input. F.5.1 demonstrated activation-only fusion into γ_inner is invariant. | none | 0 | moderate (decision) |
| 10 | `HadamardRotation.fuse_into_weight()` helper | new method on existing class; rotation matrix builder | `intllm/quant.py` | +30 | trivial |
| 11 | `QuaRotCalibrator` class | analogous to `SmoothQuantCalibrator`: `__init__`, `_collect_sites`, `apply` (mutates listed sites), `validate_forward_equivalence` (G5 reuse) | `intllm/quant.py` | +180 | moderate |
| 12 | Side-car schema (`save/load_quarot_maps`) | record γ_attn/γ_mlp pre-fusion + H seed (deterministic) for reproducibility | `intllm/quant.py` | +50 | trivial |
| 13 | Eval driver `eval_quarot_posthoc.py` | analogous to `eval_smoothquant_posthoc.py`: build → calibrate → validate → apply → val_loss → JSON | new script | +250 | moderate |
| 14 | Unit tests (~6-8 cases): happy-path delta tiny, lm_head fusion correctness, γ-absorption preserves logits, deepcopy isolation, NaN/Inf rejection, integration on tiny model | `tests/test_quant.py` extension | +250 | moderate |
| 15 | G5 adaptation for rotation-fusion | `validate_forward_equivalence` already deepcopies + applies — works as-is for QuaRotCalibrator if `apply()` follows same in-place contract; verify on a unit test | `intllm/quant.py` | +0 | trivial |
| 16 | `verify_intllm_tables.py` schema entries for any new claims | only if Branch A produces a Table 4 update for paper v2 | `paper/intllm/verify_intllm_tables.py` | +30 | trivial |

**Total LOC delta:** ~830 lines of new code + ~80 lines of mutations.
**Files touched:** 3 new entries in `intllm/quant.py`, 1 new script,
1 test extension, 1 verifier extension. No upstream `_upstream/`
modifications. No Triton kernel changes.

### 3.3 The double-γ wrinkle (v1 deferral)

HGRN-bit has **two γ-norms** in the path from residual to a BitLinear's
matmul:
1. **Block-level** `attn_norm.weight` / `mlp_norm.weight` — standard
   RMSNorm γ before attn/mlp.
2. **Per-BitLinear inner** `module.norm.weight` — `FusedBitLinear` has
   an internal RMSNorm whose γ acts on the BitLinear's input BEFORE the
   matmul.

Canonical QuaRot absorbs γ from the **block-level** norm. Per-BitLinear
inner γ was the target of F.5.1's fusion (and was demonstrated INVARIANT —
RMSNorm γ pre-absorption hypothesis confirmed). For Branch A v1:

- **Absorb block-level γ_attn / γ_mlp** into i/f/g_proj, gate_proj weights.
- **Leave per-BitLinear inner γ untouched.** F.5.1's NEGATIVE result
  shows it has nothing additional to offer; modifying it is orthogonal
  to QuaRot's claim.

This is a **design decision** to encode in the F.5.1.8 design doc, NOT
a code complexity issue. Cost: zero LOC. Risk: a v2 follow-up may want
to also absorb inner γ into rotation fusion (a true compound recipe);
that's deferred.

### 3.4 What's explicitly NOT touched

- Triton kernels (`fused_recurrent_hgrn`, `fusedbitnet`) — math
  cancels at projection boundaries; kernels see unrotated values.
- `g_norm` (FusedRMSNormSwishGate inside attn block) — operates on
  recurrent output `o`, not on residual stream. Out of scope for v1.
- Embedding/lm_head tying — Phase D's Mini ckpt has untied (separate)
  embed and lm_head BitLinear weights; both are mutated independently.
  If Base/Medium use weight-tying, that's a v2 trigger.

### 3.5 Hard-site count: ZERO

No site is "hard" per CLAUDE.md §6.8 R5 definition (would-trigger-spike).
The architecture is rotation-friendly. The complexity tags above are
"moderate" only because of code volume / careful test surface — not
unsolved math.

---

## 4. §3 Risk Register

> **Section status:** v0.1 — drafted from F.5.1 / F.6.2 lessons + §3.1
> code-touch read. May add risks based on §1 literature scan (e.g., if
> a published ternary+QuaRot composition surfaces a known dead-end).

Each risk is rated by **likelihood** (L=low, M=med, H=high) × **impact**
(L/M/H = local bug fix / day of rework / branch abort).

### 4.1 R1 — Weight-fusion algebra correctness

- **Trigger:** Bug in `W' = W · diag(γ) · Hᵀ` or `W' = H · W`. Silent at
  the unit-test boundary IF the test only checks shape + finiteness.
- **L × I:** M × M
- **Mitigation:** Forward-equivalence unit test at fp32 on a tiny
  fully-instantiated model (one block, one head, hidden=8, vocab=16).
  Pre-fusion logits == post-fusion logits within 1e-5 absolute.
- **Recovery:** Calibrator deepcopies; original ckpt never mutated.
  Bug fix + retest; no checkpoint loss.

### 4.2 R2 — γ-absorption × per-BitLinear inner γ interaction

- **Trigger:** Block-level γ_attn absorbed into i_proj, but
  FusedBitLinear's inner `.norm.weight` γ_inner ALSO acts on inputs.
  If both apply multiplicatively along the same channel axis, the
  effective input scaling changes.
- **L × I:** M × M (likely caught by R1 forward-equivalence test;
  unlikely to manifest only at scale)
- **Mitigation:** Per §3.3 design decision — leave γ_inner untouched.
  Forward-equivalence test verifies block-level γ absorption math
  composes correctly with inner γ (which sees rotated input).
- **Recovery:** Side-car records pre-fusion γ_attn / γ_mlp; revert
  trivially.

### 4.3 R3 — Ternary ±1 clipping vs rotated weight magnitudes (the
research-pivotal risk)

- **Trigger:** `W' = W · diag(γ) · Hᵀ` may produce weight magnitudes
  whose distribution post-`weight_quant_to_ternary()` differs from the
  original. The trained ckpt's ternary form was the optimization
  target during QAT; rotation may move weights into clipping
  territory. **This is the central open research question** — see §1.
- **L × I:** H × H (most likely failure mode for v1)
- **Mitigation:** Two-tier validation:
  1. fp32 fp32-weight forward equivalence (R1) — tests math is correct
  2. ternary-quantized forward equivalence — tests rotation survives
     the actual ternary clipping
  If (1) passes and (2) fails, that's a NEGATIVE research result and
  is itself a publishable finding ("rotation+ternary trained-from-scratch
  is INCOMPATIBLE post-hoc — must be QAT-coupled").
- **Recovery:** Negative results are valid outputs; document in
  findings doc, pivot to Branch B with finding integrated into paper v2.

### 4.4 R4 — RMSNorm γ pre-absorption interference (F.5.1 §4.1 finding)

- **Trigger:** F.5.1 demonstrated trained Phase D Mini ckpt's γ has
  ALREADY learned outlier compensation per-channel. Rotation
  REDISTRIBUTES outliers across channels but cannot SUBTRACT them.
  If γ has already done the per-channel rebalancing rotation would
  do, rotation has nothing to add.
- **L × I:** M × M (precedented by F.5.1 INVARIANCE result)
- **Mitigation:** Best-case outcome of Branch A is similar to F.5.1 —
  the model is invariant to the recipe. That's still a valid finding
  ("HGRN-ternary trained models are calibration-resistant to BOTH
  per-channel scaling AND per-channel rotation"); strengthens the
  paper v2 narrative.
- **Recovery:** Documented INVARIANCE is a publishable result, not a
  failure.

### 4.5 R5 — FP precision on `H ⊗ W` at hidden_size 384/512/512

- **Trigger:** Walsh-Hadamard fp16 multiply accumulates rounding error.
- **L × I:** L × L
- **Mitigation:** Standard QuaRot practice — do fusion in fp32, then
  cast back to ckpt dtype (typically fp16 for Phase D Mini/Base/Medium).
- **Recovery:** trivial.

### 4.6 R6 — G5 gate threshold may be too loose for rotation-fusion

- **Trigger:** Existing G5 (default 0.05 nat) is set for SmoothQuant
  scale-fusion which is mathematically equivalent at fp32 but the
  PER-LAYER multiply has scale ~0.001-0.01. Rotation-fusion at fp32 is
  also exact-equivalent but the multiply involves H — fp16 round-off
  on dim=384 H multiply is ~1e-4 per element, may accumulate.
- **L × I:** M × L
- **Mitigation:** Add G6 sibling gate — max-absolute-logit-difference
  on probe batch < 1e-3. ~10 LOC; complements G5's mean-loss check.
- **Recovery:** trivial; G6 added during impl if G5 alone proves
  insufficient.

### 4.7 R7 — FusedBitLinear weight semantics: stored fp16 vs runtime
ternary

- **Trigger:** `FusedBitLinear.forward` calls `weight_quant(self.weight)`
  per forward pass. Mutating `self.weight.data` (fp16 latent) affects
  what the quantizer sees. The QuaRot fusion targets the latent fp16,
  not the runtime ternary. The deployed model sees ternary; the QuaRot
  invariance only holds if `weight_quant(W·γ·Hᵀ) ≈ weight_quant(W)·γ·Hᵀ`
  — generally FALSE for rotation, since rotation breaks per-element
  sparsity.
- **L × I:** M × M (interacts with R3)
- **Mitigation:** Design doc must specify: (a) fusion target is fp16
  latent, (b) verification compares post-quant-and-fuse logits to
  pre-quant-and-fuse logits, (c) acceptance criterion accommodates
  the ternary clipping noise floor.
- **Recovery:** if forward equivalence at the ternary-runtime level
  fails, that's R3 manifesting — pivot per R3 mitigation.

### 4.8 R8 — verify_intllm_tables.py schema drift

- **Trigger:** Branch A produces new ablation table for paper v2.
- **L × I:** L × L
- **Mitigation:** Schema bump committed in same PR.
- **Recovery:** trivial.

### 4.9 R9 — Effort/calendar slip vs arXiv v1 / paper revision window

- **Trigger:** 3-5 day estimate balloons; pushes past founder-time
  arXiv submission slot.
- **L × I:** M × M
- **Mitigation:** **Day-3 abort condition** — if no first val_loss
  measurement by end of Day 3, pivot to Branch B with partial findings
  doc as standalone artifact. Branch A and arXiv v1 founder actions are
  INDEPENDENT — arXiv v1 ships paper as-is regardless of Branch A
  outcome (per F.5.1.6 §8: F.5.1 PARTIAL doesn't affect v1 readiness).
- **Recovery:** Pivot to Branch B; arXiv v1 unaffected.

### 4.10 Risk summary

| Risk | L × I | Class | Net |
|---|---|---|---|
| R1 algebra | M × M | local | bug-fix |
| R2 γ-interaction | M × M | local | bug-fix |
| **R3 ternary clipping** | **H × H** | **research** | **branch-pivotal** |
| R4 γ pre-absorption | M × M | research | publishable invariance |
| R5 fp precision | L × L | trivial | rounding |
| R6 G5 threshold | M × L | local | gate add |
| R7 BitLinear semantics | M × M | design | doc-spec |
| R8 verifier drift | L × L | trivial | schema bump |
| R9 calendar slip | M × M | governance | day-3 abort |

**One H×H risk: R3.** This IS Branch A's research question. Either
outcome (PASS, INVARIANCE, FAIL) is informative. The risk is not
"bad outcome" — it's "ambiguous outcome" (e.g., fp32 passes but
ternary marginal). Mitigation: clear two-tier validation per R3
mitigation.

---

## 5. §4 Refined Effort Sizing

> **Section status:** v0.1 — bottom-up from §2 LOC + §3 mitigation work.
> Replaces the rough "3-5 days serius" sizing with hand-tallied numbers.

### 5.1 Bottom-up task table

| Task | LOC | Hours |
|---|---|---|
| Design doc (`FJQ_PHASE_F_F5_1_8_BRANCH_A_DESIGN.md`) — math, recipe, gates, decision tree (parallel to F.5.1 design v1.0 = 1112 LOC / 3 h) | ~1000 | 3.0 |
| `HadamardRotation.fuse_into_weight()` helper | +30 | 0.5 |
| `QuaRotCalibrator` class (init + collect-sites + apply + validate) | +180 | 3.0 |
| Side-car schema (`save/load_quarot_maps`, schema v1.0) | +50 | 0.5 |
| Unit tests (8 cases: happy-path, isolation, NaN reject, lm_head fusion, embed fusion, γ-absorb composes, ternary-runtime equivalence, integration on tiny model) | +250 | 2.5 |
| G6 max-abs-logit-delta gate (sibling to G5) | +10 | 0.5 |
| Eval driver `eval_quarot_posthoc.py` (mirror SmoothQuant eval) | +250 | 2.5 |
| `verify_intllm_tables.py` schema bump | +30 | 0.5 |
| GPU run (Mini ckpt, 1 α-pivot × 2 site presets, ~10 min/run) | 0 | 0.5 |
| Initial findings doc (`FJQ_PHASE_F_F5_1_8_FINDINGS.md`) | ~700 | 2.5 |
| **Subtotal: impl-only** | ~800 | **15.5** |
| Paper v2 revision (Table 4 update + new method section) | ~300 | 3.0 |
| **Subtotal: with paper v2** | | **18.5** |

### 5.2 Surprise budget

- **+25% baseline** per CLAUDE.md §6.8 R5: 18.5 → 23.0 hours
- **+0% hard-site** adjustment: zero hard sites per §3.5
- **+15% R3 contingency:** if fp32 passes but ternary-runtime fails,
  add ~4 h to characterize and write as NEGATIVE result. Conditional —
  not always taken. So budget as 50% of full +15% = +1.4 h.
- **Total budgeted: 24.4 hours ≈ 3.0-3.5 days FT.**

### 5.3 Calendar (sequential, single-developer)

| Day | Work | Exit checkpoint |
|---|---|---|
| 1 | Design doc + HadamardRotation helper + QuaRotCalibrator scaffold | Design doc v1.0 committed; calibrator class compiles; 0 unit tests yet |
| 2 | Full impl: calibrator complete, side-car, 8 unit tests, G6 gate, eval driver | All unit tests PASS; eval driver dry-run smoke OK on CPU |
| 2.5 | GPU run on Mini ckpt | First val_loss number for QuaRot recipe |
| 3 | Findings doc + paper v2 stub | Findings v1.0 committed; A/B verdict propagates to verify_intllm_tables.py |

### 5.4 Day-3 abort gate (R9 mitigation, mechanical)

If by **end-of-Day-2.5** there is no first val_loss number, abort
Branch A and pivot to Branch B. Concrete trigger: no
`paper/intllm/ablations/quarot_*.json` file with non-NaN `val_loss`
field exists at 23:59 local on Day-2.5.

### 5.5 Variance projection

- **Best case (R4 invariance, fp32 + ternary both pass):** 16 hours,
  finish Day 2.5 with cleanly invariant result; paper v2 narrative
  identical to F.5.1 strengthened ("HGRN-ternary trained models are
  calibration-resistant to BOTH per-channel scaling AND per-channel
  rotation"). 0 surprise budget consumed.
- **Modal case (some signal, ±0.05-0.20 nat):** 18-20 hours,
  some Section §6 narrative work needed. ~25% surprise consumed.
- **Worst case (R3 ternary failure, write-up as NEGATIVE):** 24
  hours, full surprise budget. Still ships findings doc + Branch B
  pivot in same week.

---

## 6. §5 A/B Verdict Recommendation

> **Section status:** v1.0 — synthesizes §1–§4. F.5.1.7 audit
> COMPLETE.

### 6.1 Verdict — short form

**CONDITIONAL NO-GO on Branch A as primary path; GO Branch B as
default.** Branch A is reserved as a CONDITIONAL paper-v2 revision
spike, triggered by mechanical entry gates (§6.4). The trigger
requires either (a) reviewer-driven need for a fajarquant-specific
QuaRot ablation row, or (b) BitNet v2's Table 5 finding being
contradicted by a credible follow-up.

### 6.2 Why Branch A is downgraded — evidence chain

The pre-flight surfaces a load-bearing piece of literature missing
from the F.5.1 / F.6.2 design phase:

1. **BitNet v2 Table 5 (§2.3):** at the architecture closest to ours
   (ternary, gated, Transformer-with-RoPE, 1.3B–3B), weight + activation
   rotation ties or **loses** to activation-only at every cell tested.
   The paper's final-design adopts activation-only and explicitly
   states *"ternary models are more sensitive to the fusion of rotary
   matrix and latent weights."*
2. **MambaQuant (§2.4):** QuaRot drops 21% accuracy on Mamba (Vim-T)
   even at W8A8 — bit-width where rotation should be near-lossless.
   Cause: recurrent gating accumulates rotation perturbations
   non-linearly, requires per-architecture variance-eq surgery to
   recover.
3. **No published QuaRot/SpinQuant on HGRN-family at any bit-width
   (§2.4):** the existing literature gives us NO datapoint we can
   directly extrapolate from to predict Branch A's outcome on HGRN-bit.
4. **F.5.1 INVARIANCE result (memory):** trained Phase D Mini ckpt's
   RMSNorm γ has already absorbed per-channel rescaling. Canonical
   QuaRot fuses γ into adjacent W (the "fuse_layer_norms" step) — but
   with γ already optimal, this step is a no-op.

The combination of (1)+(4) is dispositive on the *math side*:
canonical QuaRot's strongest lever (γ-fusion) is neutralized by
F.5.1's empirical confirmation, AND the closest-architecture prior
(BitNet v2) explicitly tested the residual rotation lever and found
it harms ternary. (2) and (3) provide the *architecture-side*
caution: HGRN's gated recurrence is structurally similar enough to
Mamba's that we should expect a non-trivial regression risk before
ever doing the math correctly.

### 6.3 Why Branch B becomes default

Branch B (Phase F §4.2 hardware acceleration: F.10 GPU 2:4
Sparse-BitNet, F.11 CPU bitnet.cpp TL2 port, F.12 AVX-VNNI islands,
F.13 dispatch heuristic) has:

- **Tangible end-user impact:** Microsoft BitNet b1.58 2B4T datapoint
  reports 29 ms/tok decode on i7-13800H — gives a hard target for our
  i9-14900HX (DDR5-5600, no AVX-512, no AMX) of ~40-50 tok/s with
  bitnet.cpp TL2 path, which is real demo-bench latency.
- **Independent of F.5.1 / F.6.2 outcome:** Branch B doesn't need
  Branch A's verdict; ships as standalone artifact.
- **Composable with paper v2:** §4.2 F.10-F.13 work generates new
  measurement tables for the v2 revision regardless of Branch A.
- **Lower risk class:** memory-bandwidth-bound work has clearer
  ceilings (~70 GB/s sustained DDR5-5600) and harder gates than
  research-quality "does X help quant?".

### 6.4 Mechanical Branch A re-entry gate (per §6.8 R6)

Branch A re-activates IFF one of the following is true (committed
file at audit-completion):

```
GATE A1: paper v2 reviewer feedback explicitly asks for a fajarquant
         QuaRot ablation row in Table 4 or §6
GATE A2: BitNet v3 (or equivalent) ships with weight-fusion rotation
         enabled by default for ternary, contradicting v2 Table 5
GATE A3: A new ternary+QuaRot paper appears showing positive results
         on HGRN/Mamba/RWKV/RetNet architecture family
```

Any of A1-A3 → run **Branch A MINI** (1.5 days, single α + single
site preset, no full design doc, fold result into v2 revision Table
4 row). This is half the §4 budget — appropriate for revision-cycle
turnaround.

Branch A FULL (3 days FT per §4) only triggers if **two** of A1-A3
hold simultaneously, or if reviewers explicitly request a comparable
recipe characterization.

### 6.5 Branch B mechanical entry gate

Branch B activates with no further entry gate beyond user nod, since
Phase F §4.2 is explicitly listed in roadmap and pre-conditions are
already met (Phase D shipped Mini/Base/Medium with green gates,
fajaros-x86 v3.9.0 IntLLM kernel-path runs medium_final.pt). First
sub-task is **F.11 bitnet.cpp TL2 port** (CPU-bound, no GPU
dependency) — single quickest path to a published latency number.

### 6.6 What this audit produced (deliverables, not vibes)

| Deliverable | Status |
|---|---|
| F.5.1.7 pre-flight findings doc v1.0 | this file |
| Code-touch enumeration (16 sites, 0 hard) | §3 |
| Risk register (9 risks, 1 H×H) | §4 |
| Effort sizing (3 days FT impl + paper) | §5 |
| Literature scan with load-bearing BitNet v2 Table 5 finding | §2 |
| Mechanical Branch A re-entry gates (3 conditions) | §6.4 |
| Branch B default-activation entry condition | §6.5 |

### 6.7 Self-check (CLAUDE.md §6.8 self-check applied to this audit)

```
[x] Pre-flight audit (B0/C0/D0) exists for the Phase?           (R1)
[x] Every task has a runnable verification command?             (R2 — §6.4 gates are mechanical, §6.5 entry is pre-conditioned)
[x] At least one prevention mechanism added (hook/CI/rule)?     (R3 — §6.4 mechanical re-entry gate prevents Branch A drift)
[x] Agent-produced numbers cross-checked with Bash?             (R4 — §3 code-touch verified by hands-on file read; §1 numbers from agent are linked to sources)
[x] Effort variance tagged in commit message?                   (R5 — applied at commit)
[x] Decisions are committed files, not prose paragraphs?        (R6 — this doc is committed; gate spec at §6.4)
[ ] Internal doc fixes audited for public-artifact drift?       (R7 — README/CLAUDE.md sync deferred to follow-up; F.5.1 already-current)
[x] Multi-repo state check run before starting work?            (R8 — confirmed at session start, all 3 repos synced)
```
7 of 8 YES. R7 (public-artifact drift) flagged for next-session
follow-up; not blocking since this audit doc is internal-strategy,
not public-claim.

---

*v1.0 — F.5.1.7 audit COMPLETE 2026-04-28. Decision: NO-GO Branch A
primary, GO Branch B default. Conditional Branch A re-entry gates
specified in §6.4.*

*Last updated: 2026-04-28 (V32-prep F.5.1.7 audit closure).*
