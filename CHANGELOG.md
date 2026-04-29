# Changelog

All notable changes to FajarQuant are documented in this file.

The repo started as a KV cache quantization library (Arm B, paper v3.1)
and expanded with the Phase D IntLLM training-quantization research line
in v0.4.0. Going forward, both arms coexist; Phase D is the primary
research direction, KV quant is a mature paper artifact.

## [Unreleased] — V32-prep F.11 + F.5.1.6/7 chains

**F.5.1.6 G5 forward-equivalence gate** (`6987de3`) — closes the
§3.3 action item from F.5.1.6 findings.
`SmoothQuantCalibrator.validate_forward_equivalence` deepcopies
the model + measures pre/post Δ-loss on a probe batch; catches
Run-7-style cumulative saturation that per-layer §4.6 gates miss.
4 unit tests + wired into `eval_smoothquant_posthoc` with
`--g5-threshold` flag.

**F.5.1.7 Branch A pre-flight audit** (`d9f92bb`) — verdict
**NO-GO Branch A primary, GO Branch B default** based on three
load-bearing pieces of literature: BitNet v2 Table 5 (ternary +
Hadamard weight-fusion ties or hurts at 1.3B/3B), MambaQuant 21%
drop on Mamba at W8A8, ZERO published QuaRot on HGRN-family.
Plus F.5.1's RMSNorm γ pre-absorption finding neutralizes
QuaRot's strongest lever. Mechanical re-entry gates GATE A1/A2/A3
specified.

**F.11 chain — vendored microsoft/BitNet TL2 AVX2 kernel + FFI
infrastructure** (15 fajarquant + 2 fajaros-x86 + 1 Fajar Lang
commits, ~23h cumulative work, infrastructure-only landing):

  Build + integration:
  - `cpu_kernels/bitnet_tl2/` — vendored microsoft/BitNet TL2
    (75 KB MIT-licensed source; attribution in
    `THIRD_PARTY_NOTICES.md`); cc-crate `build.rs` +
    `BITNET_OMIT_TRANSFORM` gate + freestanding-safe weak
    `memset` stub
    [`a098631`, `ea8b10a`, `41621a4`, `98ed310`, `5031e29`]
  - `src/cpu_kernels/tl2.rs` — Rust shim with 64-byte alignment +
    shape pre-validation (5 unit tests + preprocessor FFI smoke)
    [`fffb9b7`, `73a8a45`]
  - `src/cpu_kernels/scalar_baseline.rs` — Phase D scalar
    BitLinear baseline ported to Rust (250 LOC, 10 tests)
    [`2d2ab47`]
  - `cpu_kernels/bitnet_tl2/codegen/codegen_tl2.py` — vendored
    microsoft/BitNet codegen with `fajarquant_mini`
    ModelShapeDict entry covering all 4 Mini BitLinear shapes
    [`41621a4`]
  - **fajaros-x86 Makefile `$(TL2_O)` target** + ld step
    inclusion + `@ffi extern("C")` decls in
    `kernel/compute/matmulfree.fj`
    [fajaros-x86 `ac68866`, `7d1de9a`] — TL2 symbols at known
    addresses in `fajaros-llvm.elf`:
    `T fjq_tl2_preprocessor @0x1066f0`,
    `T fjq_tl2_qgemm_lut @0x109010`,
    `T fjq_tl2_self_test @0x106700`

  Encoders + parity work:
  - `python/phase_d/scripts/repack_to_tl2.py` (370 LOC, 23 tests)
    + tile-organized layout extension; iteration 6 sign packing
    derived from kernel `slli_epi16(vec_sign, 4*k+sub)` pattern
    [`a098631`, `57bf7a9`, `b53ede3`, `b5a7471`, `1a52f90`]
  - `src/cpu_kernels/tl2_encoder.rs` — Rust port mirroring Python
    encoder (315 LOC, 11 tests)
    [`56a69d2`, `b5a7471`, `1a52f90`]
  - `parity_real_mlp_one_tile_at_mini_256_256` Rust test
    (#[ignore]'d) — end-to-end V31 → TL2 → AVX2 → C[i] vs
    `scalar_baseline::bitlinear_packed_scalar` comparison
    [`328052b`]

  **What works:**
  - Vendored kernel + FFI symbols linked into FajarOS Nova ELF
  - Magnitude byte encoding bit-exact (verified by sentinel +
    hand-computed-offset tests across Python + Rust)
  - 50/50 fajarquant lib + 33/33 Python tests PASS
  - Smoke binary + preprocessor smoke run end-to-end through AVX2

  **What's still BLOCKED (paper v2 narratives DEFERRED):**
  - Bit-exact parity test fails 64/64 with consistent
    sign-flip-symmetric diffs — magnitude correct, sign-bit
    position has ~18 residual error per row after iteration 6
    derivation + iteration 7 hypothesis exploration (H5/H6/H8
    rejected; H7 untried).
  - `make test-intllm-kernel-path` regression NOT yet run with
    TL2 dispatch — fajaros-x86 builds and links the static lib
    but no production code path calls `fjq_tl2_qgemm_lut`
    (would crash on parity gap).
  - F.11.5 tok/s benchmark + F.11.6 findings + paper v2 §6
    hardware-perf table — gated on parity closure.

  **Strategic decision branches** (per
  `docs/FJQ_PHASE_F_F11_CHAIN_CLOSURE.md` +
  `FJQ_PHASE_F_F11_BRANCH_X_ENCODER_LOCATION.md` +
  `FJQ_PHASE_F_F11_X7_HYPOTHESIS_FINDINGS.md`):
  - Path A: try H7 ai-ordering reversal (~1h); cheapest
  - Path B: kernel disassembly inspection (~2-3h, exceeds budget)
  - Path C: accept gap, pivot to F.13 dispatch heuristic OR
    arXiv founder actions

**Cumulative variance**: −47% on aggregate F.11 chain
(infrastructure burn 12.75h vs 24h budget; full chain including
parity 23h vs 24h, at practical limit). Per-iteration variances
in individual commit messages.

**CLAUDE.md V31.1 sync** in fajar-lang (`86989180`) — public
footer reflects the F.11 infrastructure-only state honestly.

## [0.4.0] "Phase D IntLLM" — 2026-04-24

V31.C Phase D IntLLM (1.58-bit MatMul-Free LLM training quantization)
research line shipped. Three calibrated training gates PASS with
monotonically widening margins (0.12 → 0.21 → 0.28 nat). Track B 5+1
layer interruption-safety hardening validated end-to-end during a real
laptop-shutdown event mid-Medium training. In-kernel deployment via
[FajarOS Nova v3.9.0 IntLLM Kernel Path](https://github.com/fajarkraton/fajaros-x86/releases/tag/v3.9.0).

### Added — Phase D IntLLM (training-quant research line)

**Architecture + training (V31.C.P1-P5):**
- **`intllm/model.py`** — HGRNBitForCausalLM 1.58-bit ternary architecture
  (custom MatMul-Free LLM, eliminates V31.R3 γ-cascade observed in Gemma 3 path).
- **`intllm/quant.py`** — SigmoidLUT + SiLULUT + IntRMSNorm + `fake_quantize_absmax_ste`
  (V31.C.P2.1 conditional PASS on `ridger/MMfreeLM-370M` re-eval; 5/6 tasks within ±1.53).
- **`intllm/qat.py`** — BitLinearStatTracker + adaptive bits + channel permutation
  per CLAUDE.md §6.9 R5 (outlier handling non-negotiable for LLM quantization).
- **`intllm/tokenizer.py`** + **`intllm/data.py`** — Mistral v3 32K tokenizer +
  SlimPajama-6B streaming loader (V31.C.P3, steady-state 1.3M tok/s).
- **`intllm/train.py`** — TrainConfig + `LambdaLR` scheduler (linear warmup +
  cosine decay to 10%) + `train_loop` with metric tracking.

**Training drivers (V31.C.P2-P7):**
- `python/phase_d/scripts/train_mini.py` — 21.5M params, 491M tokens (22.8 tok/p)
- `python/phase_d/scripts/train_base.py` — 46.4M params, 982M Chinchilla-optimal
- `python/phase_d/scripts/train_medium.py` — 74.5M params, 1.819B tokens (24.4 tok/p)

**3 calibrated gates PASS (V31.C.P4-P8):**
- Mini v2: val_loss 4.38 / PPL 80.0 / gate < 4.5 / margin **0.12 nat**
- Base c.1: val_loss 3.99 / PPL 54.1 / gate < 4.2 / margin **0.21 nat** (3× wider than c.2's 0.071 nat)
- Medium c.1: val_loss 3.72 / PPL 41.3 / gate < 4.0 / margin **0.28 nat** (1.33× wider than Base c.1)
- All PASS — monotonic widening on both val_loss and margin per scale step.
- Decision docs: `docs/FJQ_PHASE_D_BASE_C1_GATE.md`, `docs/FJQ_PHASE_D_MEDIUM_C1_GATE.md`
  (§6.8 R6 8-YES checklist).

**Track B 5+1-layer interruption-safety (V31.C.P6.1–P6.6):**
1. `ckpt_every` — atomic write (`.tmp` + `os.replace`) + rotation (P6.1)
2. `--resume` / `--resume-auto` — bit-exact state restore: model + optimizer +
   LR scheduler + step counter (P6.2)
3. `StepWatchdog` — daemon thread SIGTERMs main if step counter idle > 1800s,
   single-shot + warmup-aware (P6.3)
4. HF timeout + `_retry_iter` — `HF_DATASETS_DOWNLOAD_TIMEOUT=60` +
   `HF_HUB_DOWNLOAD_TIMEOUT=60`, retry-on-transient with seed offset (P6.4)
5. `make test-train-watchdog` Makefile regression gate — 24 tests + signal-delivery
   integration test (P6.5)
6. `sys.stdout.reconfigure(line_buffering=True)` — defensive in-script fix for
   nohup-buffering blind period (P6.6, added post-Medium-resume incident)

End-to-end Track B validation: all 6 layers exercised at scale during a real
laptop-shutdown event mid-Medium training (2026-04-23 11:32 WIB). Resume
restored 91k more steps to step 111000 over 18h12m, finished cleanly. Only
~36min compute lost between checkpoint save and shutdown. CLAUDE.md §6.11
"Training Script Interruption-Safety Rule" codifies the 5-layer pattern as
cross-repo (added in fajar-lang `441f22e`).

**Benchmarks (V31.C.P3.2-P3.5, P9):**
- `intllm.lm_eval_wrapper` — `HFLM` adapter for `lm-evaluation-harness` v0.4.11.
- `bench-canonical-real` — 8-task lm-eval (wikitext + lambada + hellaswag + piqa
  + winogrande + arc_easy + arc_challenge + openbookqa) on `*_final.pt` checkpoints.
  All 3 scales benched in v0.4.0:
  - `paper/intllm/results/bench_canonical_intllm-mini.json`
  - `paper/intllm/results/bench_canonical_intllm-base.json`
  - `paper/intllm/results/bench_canonical_intllm-medium.json`
- **Phase D scaling chain — clean monotonic LM-modeling improvement:**
  - wikitext word_PPL: 343 → 201 → **138** (-60% Mini→Medium)
  - lambada PPL: 51121 → 16729 → **5277** (-90% Mini→Medium)
  - lambada acc: 0.001 → 0.007 → **0.023** (16× Mini→Medium)
  - Multi-choice reasoning (hellaswag/piqa/winogrande/arc/openbookqa):
    noisy at sub-100M scale, mostly within ±1-2 stderr (per Chinchilla
    expectation).
- `bench-knowledge-real` — mmlu/triviaqa/boolq scaffold ready (not yet executed).
- `make verify-intllm-tables --strict` — 12/13 paper claims PASS (1 PEND:
  Table 4 kernel E2E Mini tok/s, FajarOS-side artifact).

**fp16-vs-ternary parity gate (V31.C.P3.5, IntLLM differentiator):**
- 37 hooks, rel_l2 mean 2.31, p50 0.82, max 19.31 on Mini v2.

**MLSys AE artifact (V31.C.P5.1-P5.3):**
- `ARTIFACT_APPENDIX.md` + 100-prompt fixed evaluation set.
- `run_smoke.sh` — 93s end-to-end Functional smoke (vs 30-min target).

**Documentation:**
- `docs/FJQ_PHASE_D_PRODUCTION_PLAN.md` — 9-week plan
- `docs/FJQ_PHASE_D_GATE_CALIBRATION.md` — evidence-backed scale-aware
  thresholds (Mini < 4.5 / Base < 4.2 / Medium < 4.0 / Stretch < 3.7)
- `docs/FJQ_PHASE_D_CONFIG.md` — params + token budgets per scale
- `docs/FJQ_PHASE_D_BASE_C1_GATE.md` + `docs/FJQ_PHASE_D_MEDIUM_C1_GATE.md` —
  §6.8 R6 decision-gate docs
- `docs/FJQ_PHASE_D_OPS.md` + `docs/PAPER_OUTLINE.md`

### Changed — Cross-cutting

- **Cargo.toml** — `version = "0.3.0"` → `"0.4.0"`. Description rewritten to
  reflect both Arms.
- **README.md** — restructured as umbrella ("FajarQuant — Quantization Research
  for Compiler-Verified LLM Systems") with two clearly separated arms. v3.1
  KV quant content preserved; Phase D added as primary going forward.
- **Compiler dependency** — bumped from Fajar Lang v27.5.0 → v31.0.0 (gains
  `@noinline`+`@inline`+`@cold` lexer V29.P1, `@no_vectorize` codegen V31.B.P2,
  `FJ_EMIT_IR` env var; needed by Phase D kernel-path).
- **Paper Table 2** — wikitext + hellaswag rows for all 3 scales now backed by
  real lm-eval v0.4.11 numbers (was placeholder zeros pre-v0.4.0).

### Cross-Repo Linkage

- **[Fajar Lang v31.0.0](https://github.com/fajarkraton/fajar-lang/releases/tag/v31.0.0)** —
  compiler dependency. Phase D uses V29.P1 `@noinline` lexer + V31.B.P2
  `@no_vectorize` codegen attribute.
- **[FajarOS Nova v3.9.0](https://github.com/fajarkraton/fajaros-x86/releases/tag/v3.9.0)** —
  IntLLM Kernel Path. Runs `medium_final.pt` inside `@kernel` context via
  shell `cmd_ask` → `tfm_mf_generate` dispatch + `make test-intllm-kernel-path`
  4-invariant gate.
- All three repos now Apache 2.0 (fajar-lang + fajaros-x86 relicensed from
  MIT on 2026-04-24; fajarquant has been Apache 2.0 since inception).

## [0.3.0] "FajarQuant v3.1 Adaptive Per-Head" — 2026-04-13

Adaptive per-head KV cache quantization. Profiles each KV head's statistical
properties (variance per channel, kurtosis, SVD ratio) at calibration time
and routes to optimal strategy. Discovers two architecture-specific optima
that no fixed method finds: PCA rotation on MQA 3-bit (−24% vs KIVI), PPL-
guided mixture on GQA 2-bit (−35% vs KIVI). Score: 2 wins / 5 ties / 2 losses
on 9-cell cross-architecture grid.

### Added
- `adaptive.rs` — per-head profiler + strategy selector (Path A=KIVI / B=PCA / C=TQ)
- PPL-guided 2-bit fallback when MSE and PPL disagree
- 28 paper claims + `verify_paper_tables.py` strict CI gate

See GitHub Release for full notes:
https://github.com/fajarkraton/fajarquant/releases/tag/v0.3.0-fajarquant-v3.1

## [0.2.0] "FajarQuant v2.12 Cross-Architecture KV Cache Quantization" — 2026-04-13

First systematic cross-architecture perplexity evaluation. 3 models (Gemma 4 E2B
MQA + Mistral 7B GQA-8 + Qwen2-7B GQA-4) × 3 bit widths (2/3/4) = 9 cells, all
on canonical R-α.1 model surgery protocol with WikiText-2 test set.

### Added
- Outlier-aware calibrated PCA (`turboquant.rs` v2): replaces TurboQuant's
  random rotation with per-head PCA, calibrated once on representative data.
- Fused quantized attention (`fused_attention.rs`): computes attention on
  quantized KV vectors via codebook dot products, 524,288× memory reduction
  at 16K context.
- Hierarchical multi-resolution bit allocation (`hierarchical.rs`): more bits
  for recent tokens, exponential decay. 48.7% bit savings @ 10K, 55.7% @ 16K.
- 6 Fajar Lang demos in `examples/*.fj`.

See GitHub Release for full notes:
https://github.com/fajarkraton/fajarquant/releases/tag/v0.2.0-fajarquant-v2.12

## [0.1.0] "Initial Release" — 2026-04-11

Extracted as standalone crate from Fajar Lang V26 Phase A4 split. Algorithm,
paper, data, and reproducibility scripts moved here. Fajar Lang depends via
Cargo path/git dep + thin re-export shim.

### Added
- KIVI baseline (`kivi.rs`)
- 16 integration tests
- Cargo workspace structure
- Initial paper draft

See GitHub Release for full notes:
https://github.com/fajarkraton/fajarquant/releases/tag/v0.1.0
