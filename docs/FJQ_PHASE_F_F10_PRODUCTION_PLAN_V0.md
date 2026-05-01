---
phase: F.10 — GPU 2:4 Sparse-BitNet integration (pre-flight scaffolding plan)
status: PRE-FLIGHT v0 — 2026-05-01
budget: this doc ~1h pre-flight; full F.10 chain estimate 3 days human + ~15h GPU
prereq: nvidia driver active (RTX 4090 visible) + paper submission (entry condition)
parent_roadmap: docs/FJQ_PHASE_F_TAX_VERTICAL_ROADMAP.md §F.10
---

# F.10 GPU 2:4 Sparse-BitNet — Production Plan v0 (pre-flight)

> **TL;DR.** F.10 strict execution is gated on paper submission (entry
> condition) which has 5 founder external actions pending. This doc is the
> §6.8 R1 pre-flight: maps Sparse-BitNet (Microsoft Research, public repo
> github.com/AAzdi/Sparse-BitNet, MIT, 13 stars) recipe to our Phase D
> infrastructure, identifies vendoring scope, lists 8 sub-tasks F.10.0-F.10.8,
> total ~3 days human + ~15h GPU. Not actioning execution this session;
> producing this plan so when paper unblocks, F.10 chain is click-and-go
> rather than start-from-scratch.

## 1. What Sparse-BitNet is

Per github.com/AAzdi/Sparse-BitNet README:
- Microsoft Research paper "Sparse-BitNet: 1.58-bit LLMs are Naturally
  Friendly to Semi-Structured Sparsity" (Zhang, Wu, Huang, Wang, Shao, Hao,
  Chi, Dong, Song, Xia, Sui, Wei)
- Empirical claim from roadmap §F.10: BitNet b1.58 weights post-training are
  ~42% naturally zero with tri-modal "quantization-valley" histogram; under
  2:4 structured sparsity, ternary BitNet degrades 3.3× less than FP16
  baseline.
- Recipe: ternary BitLinear ({-1, 0, +1}) + 2:4 N:M structured sparsity
  trained jointly (sparse-from-scratch QAT)
- Triton kernel for mask creation; Dual-STE for gradient flow through both
  quantization and sparsity
- Hardware: requires CUDA 12+, PyTorch 2.1+, Triton, NVIDIA Apex

## 2. Hardware compatibility check

| Component | Required | This machine | Status |
|---|---|---|---|
| GPU arch | Ada Lovelace (sm_89) for 4th-gen Tensor Core 2:4 | RTX 4090 Laptop = Ada Lovelace sm_89 | ✓ |
| CUDA | ≥12.0 | 12.4 (per F.13.1 v2 calibration JSON) | ✓ |
| PyTorch | ≥2.1 | 2.6.0+cu124 | ✓ |
| Triton | (any recent) | already used by HGRN-Bit upstream | ✓ |
| NVIDIA Apex | mixed-precision training | NOT INSTALLED | ⚠ — install in F.10.1 |
| cuSPARSElt | for runtime 2:4 GEMM | shipped with CUDA 12+ | ✓ |
| RAM | depends on Mini scale | 31 GB system + 16 GB VRAM | ✓ for Mini |

Net: **all critical compatibility ✓** modulo Apex install.

## 3. Repo scope assessment (vendor what, write what)

Sparse-BitNet repo `AAzdi/Sparse-BitNet` has 30 files. Files relevant to F.10:

| File | Role | F.10 action |
|---|---|---|
| `llm/kernel/mask_creator_kernel.py` | Triton kernel that produces 2:4 mask from weight magnitudes | **VENDOR** verbatim into `python/phase_d/intllm/sparse_kernel.py` |
| `llm/kernel/linear_cross_entropy.py` | Fused linear + cross-entropy (orthogonal) | SKIP — not core to 2:4 |
| `llm/arch/model.py` | BitLinear + sparse-mask integration | **STUDY** — adapt logic to our `intllm/quant.py:FusedBitLinear` (HGRN-Bit-aware) |
| `llm/train.py` | Dual-STE training loop | **STUDY** — adapt to our `intllm/train.py` |
| `llm/config.py` | Hyperparameters (sparsity warmup, mask refresh interval) | STUDY — extract defaults |
| `llm/biteval/eval_utils.py` | Sparsity-aware eval (per-layer sparsity stats) | OPTIONAL — useful for verification but not blocking |
| `setup.sh` + `scripts/*.sh` | Bash glue | SKIP — adapt to our Makefile pattern |

**Vendoring scope ≈ 1 file (mask_creator_kernel.py) + 2 study-and-adapt
(model.py, train.py).** Much smaller than F.11's 75 KB upstream kernel
vendoring. Matches roadmap §F.10 estimate "Triton kernels are public, no
wgmma → RTX 4090 compatible. Single cleanest experiment."

## 4. Sub-task breakdown

Total: ~3 days human + ~15h GPU. 9 sub-tasks F.10.0 - F.10.8.

| # | Task | Effort | GPU time | Verification |
|---|---|---|---|---|
| F.10.0 | Pre-flight (this doc) | ~1h human | 0 | this file + verify-f10-decision (future gate) |
| F.10.1 | Install NVIDIA Apex; vendor `mask_creator_kernel.py` into `python/phase_d/intllm/sparse_kernel.py` (verbatim, +license attribution); add unit test that mask is structurally 2:4 on synthetic input | ~3h human | <1 min | `pytest python/phase_d/tests/test_sparse_kernel.py` 2:4 invariant verified |
| F.10.2 | Extend `intllm/quant.py:FusedBitLinear` with optional `sparse_mask: torch.Tensor \| None` parameter; wire into forward pass with `weight * mask` after ternary quantization | ~4h human | 0 | unit test: dense path (mask=None) produces same output as today; sparse path (mask=2:4) zeros out 50% of weights |
| F.10.3 | Extend `intllm/train.py` with sparse-from-scratch loop: Dual-STE for both ternary quant + sparsity gradients; mask refresh every N steps via mask_creator_kernel | ~6h human | 0 | unit test: 100-step toy training with sparse mask; verify mask updates between steps |
| F.10.4 | Add `--sparse-2-4` flag to `train_mini_ablation.py`; wire feature into E2_REAL_FEATURES set | ~2h human | 0 | dry-run: `--tag sparse_2_4 --sparse-2-4 --dry-run` exits 0 with feature flag confirmed in JSON |
| F.10.5 | PoL smoke: Mini × `--sparse-2-4 --proof-of-life` (200 steps, ~5 min); verify scaffold runs end-to-end without OOM, produces JSON artifact, sparsity statistics show ~50% zeros in masked sites | ~30min human | ~5 min | `paper/intllm/ablations/mini_sparse_2_4_pol.json` with `proof_of_life=true` + sparsity stats |
| F.10.6 | **Full F.10.1 run** (gated on paper submission): Mini × `--sparse-2-4` 24K steps (~4h on RTX 4090 per Q5 baseline timing). Comparison: vs Q5 baseline 4.732 EN val_loss + sparse_2_4 must achieve ≥+0.05 nat IMPROVEMENT (gate G1 from roadmap entry condition) | ~30min human | ~4h | `paper/intllm/ablations/mini_sparse_2_4.json`; `verify_intllm_tables.py --strict` includes new claim |
| F.10.7 | Wall-clock prefill+decode speedup measurement: Mini × {batch=1, batch=8} × {dense, sparse} × {N=64 tokens}; require sparse path ≥1.10× wall-clock vs dense (gate G2). Use `bench_f13_dispatch_calibration.py` pattern | ~3h human | ~30min | `paper/intllm/results/sparse_2_4_speedup.json` with median speedup |
| F.10.8 | Decision-doc per gates G1+G2 + verify-f10-decision Makefile gate + commit chain | ~3h human | 0 | `make verify-f10-decision` strict 0 |

## 5. Critical "DO NOT" pursuit (per roadmap warning)

Roadmap §F.10 explicitly warns: **"Known dead-end (do NOT pursue): Post-hoc
2:4 prune of an existing Phase D checkpoint."** Sparse-BitNet Table 4 +
V31.E2 pattern (Mini-scale + training-from-scratch hostile to features
ported from post-hoc literature) → retro-fit will almost certainly fail
gates.

This plan adheres: F.10.1-F.10.8 all use **sparse-from-scratch QAT**, not
post-hoc prune.

## 6. Mechanical decision gates (per roadmap entry condition)

Two gates from roadmap entry condition for F.10:

| Gate | Definition | Pass criterion |
|---|---|---|
| **G1 val_loss** | Sparse-2:4 Mini final val_loss vs Q5 baseline 4.732 (EN) | sparse must be ≤4.732 OR ≥+0.05 nat improvement |
| **G2 wall-clock** | Sparse path end-to-end tok/s vs dense baseline at batch=1 / Mini | sparse ≥1.10× dense (memory-bandwidth-bound regime; 2× theoretical compute headroom does NOT auto-translate) |

Outcome matrix:

| G1 | G2 | Verdict |
|---|---|---|
| PASS | PASS | F.10.2 unlocked (extend to Base/Medium scale) |
| PASS | FAIL | Compute is real but memory bandwidth blocks; F.10.2 deferred. Document as paper §X.Y |
| FAIL | PASS | Speedup is real but quality regresses; F.10 demoted to "infra-only ship" similar to F.11 |
| FAIL | FAIL | Full demote of F.10 like F.11. Document as Phase E2-style honest negative. |

Per V31.E2 pattern, the FAIL+FAIL outcome is the most likely (~60%). PASS+PASS
is the wished-for differentiator (~15%).

## 7. Realistic expectation framing for paper v2

Sparse-BitNet's own paper claims 1.30× prefill / 1.18× decode at batch=128
on A100/B200. Their measurements are LARGE-batch + LARGE-model.

Our IntLLM regime (batch=1 + ≤100M params on RTX 4090 Laptop) is
**memory-bandwidth-bound**, NOT compute-bound at this scale. Mini × batch=1
already runs in ~5 ms/token (measured F.13.1 v2 = 202.6 tok/s with KV cache
+ torch.compile). Of that 5 ms, the dominant cost is memory bandwidth
(loading weights + KV cache), not GEMM throughput.

Realistic Sparse-BitNet wall-clock on our setup: **1.05-1.15× end-to-end at
best.** Possibly worse if mask-application overhead (Triton kernel launch
per step) eats into the savings. The 2× theoretical compute headroom from
2:4 Tensor Cores does not directly map.

Per roadmap §F.10 entry condition: gate G2 requires ≥1.10×. We're aiming
for the lower edge of the realistic band.

## 8. Entry condition: paper submission status

Per `docs/ARXIV_SUBMISSION.md` v1.1:
- v1 paper: ✓ ready (verify-gate green 40/40 PASS, tarball builds clean)
- 5 founder external actions pending (ORCID, Zenodo, arxiv.org account,
  editorial review, upload)

F.10 strict execution **does NOT begin** until paper goes live. F.10.0-F.10.5
(scaffolding + PoL smoke) CAN proceed before paper since they don't produce
paper-claim artifacts. F.10.6-F.10.8 (full runs + decision-doc) MUST wait
for paper acceptance.

## 9. What this pre-flight delivers

- ✓ Hardware compatibility verified (RTX 4090 Laptop + CUDA 12.4 + PyTorch 2.6 OK; only Apex needs install)
- ✓ Vendoring scope identified (1 file verbatim + 2 study-and-adapt vs F.11's 75 KB)
- ✓ 9 sub-tasks F.10.0-F.10.8 with effort + GPU time + verification per task
- ✓ "DO NOT pursue post-hoc prune" warning preserved per V31.E2 lesson
- ✓ Mechanical decision gates G1 (val_loss) + G2 (wall-clock) with 4-cell verdict matrix
- ✓ Honest realistic expectation: 1.05-1.15× end-to-end on our regime, not 2×
- ✓ Entry-condition status tracked: paper submission still pending

## 10. What this pre-flight does NOT do

- ✗ Vendor any code yet (F.10.1)
- ✗ Modify FusedBitLinear or train loop (F.10.2-3)
- ✗ Run any GPU work (F.10.5+)
- ✗ Make F.10 GO/NO-GO call (waits for paper + F.10.5 PoL outcome)

## 11. Recommended next step (post-paper-submission)

When paper goes live AND there's bandwidth for F.10 chain (~3 days human):

1. F.10.1 vendor + Apex install (~3h)
2. F.10.2 BitLinear sparse extension (~4h)
3. F.10.3 train loop Dual-STE (~6h)
4. F.10.4 ablation flag (~2h)
5. F.10.5 PoL smoke (~30min)
6. **DECISION POINT 1**: PoL OK? if no → debug; if yes → F.10.6
7. F.10.6 full Mini run (~4h)
8. F.10.7 wall-clock measurement (~3h)
9. F.10.8 decision-doc per gates (~3h)

Total post-paper budget: ~25h human + ~5h GPU = ~3 days human end-to-end.

## 12. Self-check (CLAUDE.md §6.8)

| Rule | Status |
|---|---|
| §6.8 R1 pre-flight audit | YES — this doc |
| §6.8 R2 runnable verification | YES — per-task col in §4 |
| §6.8 R3 prevention layer | planned — `make verify-f10-decision` in F.10.8 |
| §6.8 R5 surprise budget +25% | YES — 9 tasks with explicit estimates; cumulative ~3 days vs roadmap "3 days" matches |
| §6.8 R6 mechanical decision gates | YES — §6 G1+G2 + 4-cell verdict matrix |
| §6.8 R7 public-artifact sync | partial — full sync at F.10.8 closeout |
| §6.8 R8 multi-repo state check | N/A at pre-flight; will run at F.10.8 |

7/8 (R7 deferred to closeout). Pre-flight scope CLOSED.

---

*F.10 Pre-flight v0 closed 2026-05-01. Plan covers F.10.0-F.10.8 chain
(~3 days human + ~15h GPU). F.10.6+ gated on paper submission.
F.10.1-F.10.5 (scaffolding + PoL) can proceed in advance when bandwidth
allows.*
