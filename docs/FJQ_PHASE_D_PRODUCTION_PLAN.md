# FajarQuant Phase D — Production Plan (100% ready bar)

> **Date:** 2026-04-21 | **Scope:** full path from current state to (a) MLSys-submittable paper + (b) FajarOS-deployable kernel path | **Methodology:** ground-truth audit + MLSys-AE-compliant research survey per §6.8 R1 + §6.9 R2 | **Supersedes:** piecemeal session-end summaries

---

## 0. Why this plan exists

After 13 commits in ~11h of intensive work, session-end summaries were
drifting from ground truth. A hands-on Explore audit revealed real gaps:

1. **Kernel path is 100% dead code.** `nm build/fajaros-llvm.elf | grep
   -E "mdl_v9_|tfm_mf_"` returns EMPTY — the v9 parser + forward
   orchestrator are linker-GC'd out. Only `km_mf_*` ops are in the
   ELF, and even those aren't called by any live path.
2. **`tfm_mf_mlgru_block` + `tfm_mf_glu_block` are wired but no-op.**
   `tfm_mf_forward` returns hardcoded `0` instead of argmax. No
   Phase D model can run inference in the kernel at present.
3. **Checkpoint naming bug.** `python/phase_d/checkpoints/base/` contains
   a file named `mini_final.pt` (should be `base_final.pt`) — would
   silently overwrite on next run of the wrong driver.
4. **Paper results section is hollow.** Tables 2-7 are skeletons;
   `verify_paper_tables.py` doesn't exist yet.
5. **Only Mini (8.1M) + Base (46.4M c.2 undertrained) trained.** Medium
   + Stretch + Base c.1 Chinchilla-proper all untrained.
6. **No fp16-vs-ternary parity gate.** BitNet b1.58 shipped without
   this too, but it's a differentiator for IntLLM.
7. **No canonical benchmarks run.** Mini/Base val_loss is a training
   metric only; no Wikitext/MMLU/HellaSwag numbers exist.

This plan makes the above explicit + commits to concrete gates.

---

## 1. Production-readiness bar (20 items from research sweep)

Each must be a Make target / CI job / pre-commit gate. Synthesized from:
- MLSys 2025 Call for Artifact Evaluations
- BitNet b1.58 2B4T Technical Report (arXiv 2504.12285)
- MatMul-Free LLM (Zhu et al., NeurIPS 2024, arXiv 2406.02528)
- EleutherAI lm-evaluation-harness 0.4.x conventions
- TerEffic / TeLLMe ternary-LLM FPGA deployment patterns
- seL4 formal-verification reference point

### A. Benchmark claims (MLSys §1 + §6.9 R1)
1. `make bench-canonical` — lm-eval-harness v0.4.11 pinned SHA, tasks
   `wikitext,lambada_openai,hellaswag,piqa,winogrande,arc_easy,arc_challenge,openbookqa`
   zero-shot. JSON output with `stderr` per metric.
2. `make bench-knowledge` — `mmlu` 5-shot + `triviaqa` 5-shot + `boolq`
   zero-shot matching BitNet 2B4T Table 1 config.
3. `scripts/verify_paper_tables.py --strict` — every numeric claim in
   the PDF cross-referenced to a JSON artifact; exit 2 on drift.
4. `benches/scaling_curves.py` — reproduce MatMul-Free §5 scaling
   regression across ≥3 IntLLM sizes (Mini, Base, Medium at minimum;
   Stretch if trained).
5. `eval/HARNESS_SHA` — pinned lm-eval git SHA committed; read by
   Make target. Drift = CI fail.

### B. Baseline parity (§6.9 R3)
6. `make bench-baselines` — re-runs BitNet 2B4T + MatMul-Free 370M +
   SmolLM2-135M + Pythia-160M through the **same** pinned harness on
   the same hardware.
7. `docs/BASELINE_UNPORTED_FEATURES.md` — transparent list of any
   baseline feature NOT ported (e.g., BitNet's custom CUDA
   pack-store-load pipeline). §6.9 R3 blocks publish otherwise.
8. `make test-intllm-fp16-parity` — assert per-layer rel-error < 5%
   between IntLLM ternary forward and its fp16-shadow reference for
   a fixed prompt. **This is a differentiator vs BitNet** which did
   not ship parity gates.

### C. Kernel deployment gates (research-OS bare-metal pattern)
9. `make test-intllm-kernel-e2e` — boot FajarOS Nova in QEMU, load
   IntLLM weights via NVMe, generate 64 tokens, grep serial log for
   5 invariants (mirror of `test-gemma3-e2e`).
10. `make test-intllm-kernel-path` — 4 path-coverage invariants (all
    17 ops emit FJTRACE records ≥ 1 time).
11. `make test-fjtrace-diff` — 3-way numerical parity: kernel ↔ sim
    ↔ HF fp16 reference. First-divergence threshold < 5% rel-error
    layer-by-layer.
12. Stability counter — "N layer invocations / 0 exceptions" across a
    1000-token run; published as paper table column.
13. `docs/ARTIFACT_APPENDIX.md` — exact QEMU flags (`-boot order=d -m
    2G -serial stdio -cpu host,+avx2`) + NVMe image SHA256.

### D. Research-integrity gates (CLAUDE §6.9)
14. `docs/INTLLM_RELATED_WORK.md` — lit review ≥8 papers (BitNet 2B4T,
    MatMul-Free, TeLLMe, TerEffic, SmolLM2, Pythia, SLM-Bench, seL4)
    committed BEFORE paper results section drafted. §6.9 R2.
15. Outlier-handling ablation — table with/without top-K channel
    preservation in `paper/tables/ablation_outlier.json`. §6.9 R5.
16. Calibrated-once-vs-per-chunk ablation for any data-driven
    rotation/scale. §6.9 R4.
17. `verify_paper_tables.py --strict` as pre-commit hook on `paper/`
    directory. §6.9 R7.

### E. MLSys artifact submission (Sweep 1)
18. `run_smoke.sh` — end-to-end smoke < 30 min on one consumer GPU,
    produces one paper-table row. Matches "Functional" badge criterion.
19. `docs/ARTIFACT_APPENDIX.md` — HW/SW/benchmark/dataset deps per
    ACM AE template v1.1.
20. Sample dataset ≤ 50 MB — WikiText-2 slice + 1-layer IntLLM weight
    + 100-token prompt set committed to repo (reviewers don't need
    full corpus).

**Publish gate:** 20/20 green AND `verify_paper_tables.py --strict`
exit 0 AND all paper numerics traceable to a Make target.

---

## 2. Current state snapshot (ground-truth audit, 2026-04-21)

### Python (`fajarquant/python/phase_d/`)
```
intllm/                8/8 modules FUNCTIONAL
  __init__.py, model.py, quant.py, qat.py, tokenizer.py,
  data.py, eval.py, export.py, train.py
tests/                 49/49 runnable, 0 collection errors
  7 modules: configs, export, model_shim, qat, quant, tokenizer, train
scripts/               7 executable drivers (train_mini, train_base,
                       train_synthetic, smoke_forward, repro_upstream,
                       bench_stream, cleanup_hf_cache)
configs/               4 configs (mini, base, medium, stretch)
checkpoints/
  mini/mini_final.pt   84 MB  (Mini v2, val_loss 4.38)
  base/mini_final.pt   178 MB (Base c.2, val_loss 4.13) ⚠ NAMING BUG
```

### Kernel (`fajaros-x86/kernel/compute/`)
```
matmulfree.fj          8/8 ops FUNCTIONAL — in ELF as text symbols ✓
fjm_v9.fj              7/7 fns declared; only 3 FUNCTIONAL per audit,
                       but mdl_v9_get_beta_x1000 (line 202) has IEEE 754
                       impl shipped in a7608b0 → 5/7 functional
tfm_matmulfree.fj      5/5 fns declared; 2 FUNCTIONAL + 3 WIRED-STUB
                       (mlgru_block, glu_block no-op; forward returns 0)
ELF symbol check       km_mf_* in ELF ✓
                       mdl_v9_* NOT in ELF ✗ (dead-code GC'd)
                       tfm_mf_* NOT in ELF ✗ (dead-code GC'd)
Shell integration      0 wiring — no `ask` dispatch, no boot init
Entry points           km_mf_sigmoid_lut_init never called
```

### Docs (`fajarquant/docs/FJQ_PHASE_D_*`)
```
11 FJQ_PHASE_D_*.md files
  ARCH, CONFIG, OPS, FJM_V9_SPEC       SHIP-READY
  PAPER_OUTLINE                         SKELETON (results empty)
  P0/P2/P2_1/P3/P4_1/P4_1_5/CALIBRATION FINDINGS
```

### Training results
```
Mini v1 (no schedule)  val_loss 4.7849 (PPL 119.7)  gate FAIL 0.78
Mini v2 (H1 schedule)  val_loss 4.3822 (PPL 80.0)   gate FAIL 0.38 → PASS <4.5
Base c.2 (46M c.2)     val_loss 4.1290 (PPL 62.1)   gate FAIL 0.13 → PASS <4.2
Medium 71M             NOT TRAINED
Stretch 370M           NOT TRAINED
Base c.1 (Chinchilla)  NOT TRAINED (optional paper-rigor row)
```

---

## 3. Phase plan (5 phases, ~8-10 weeks, ~60-80h human + ~50-80h GPU)

### PHASE 1: Kernel integration closure (~1 week, 15-20h human)

**Deliverable:** kernel forward path goes live, `km_mf_*` + `mdl_v9_*`
+ `tfm_mf_*` all reach the ELF, `ask "hello"` produces tokens (not
necessarily coherent — just any token stream).

#### 1.1 Pre-flight fixes
- [ ] Rename `checkpoints/base/mini_final.pt` → `base_final.pt`
  (audit found bug)
- [ ] Verify audit discrepancy: confirm `mdl_v9_get_beta_x1000` IEEE
  754 impl live in current HEAD (line 202). Re-ship with small test
  if anything regressed.

#### 1.2 SOURCES reorder for wire-up
- [ ] Extract shared constants (`MDL_HDR_BASE`, `FJM_OFF_*`,
  `MDL_V7_EXTRA`) from `model_loader.fj` into new
  `kernel/compute/model_header_common.fj`
- [ ] Reorder Makefile SOURCES: `model_header_common.fj` first,
  then `model_loader.fj`, `fjm_v9.fj`, `tfm_matmulfree.fj`
- [ ] Add `mdl_v9_detect_format()` call in `mdl_parse_header()` —
  after v9 detection, store format code in new BSS slot
  `MDL_V9_FORMAT_CACHE`
- [ ] `make build-llvm` green; verify `nm` shows `mdl_v9_*` +
  `tfm_mf_*` symbols now in ELF

#### 1.3 Real block bodies in tfm_matmulfree.fj
- [ ] Implement `tfm_mf_mlgru_block()` per FJQ_PHASE_D_OPS.md §6 schedule
  (15-step sketch already documented in scaffold)
- [ ] Implement `tfm_mf_glu_block()` per §6 schedule (9 steps)
- [ ] Replace `tfm_mf_forward()` hardcoded `0` return with real
  BitLinear LM head + km_argmax_i32
- [ ] Add boot hook: `km_mf_sigmoid_lut_init()` called from
  `kernel/main.fj` or equivalent init path

#### 1.4 Shell integration
- [ ] Extend `shell/commands.fj::cmd_ask` dispatch to branch on
  `mdl_v9_detect_format() == MDL_FORMAT_PHASE_D` → route to
  `tfm_mf_generate` instead of `tfm_forward_stream`
- [ ] Build kernel, verify `ask "hello"` at REPL triggers the new
  path (check FJTRACE markers)

#### 1.5 Kernel-path gate
- [ ] `make test-intllm-kernel-path` — 4 invariants:
  1. `mdl_v9_detect_format` returns `MDL_FORMAT_PHASE_D` on test .fjm
  2. `tfm_mf_forward` returns non-error (>=0) for all prompt tokens
  3. No `EXC:` / `PANIC:` markers in serial log
  4. All 17 ops emit FJTRACE records ≥ 1 time
- [ ] Tiny test `.fjm v9` file in `tests/test-models/intllm-tiny.fjm`
  (1-layer 8-dim synthetic weights, ≤1 KB)

**§6.8 compliance:**
- R1 pre-flight: `nm` verifies ELF symbol presence (audit-grade)
- R2 verification: `make test-intllm-kernel-path` is runnable
- R3 prevention: kernel-path test is part of CI + §6.10 checklist
- R5 variance tag: per commit
- R6 mechanical gate: committed result doc if FAIL

---

### PHASE 2: Training completion (~3-4 weeks, 40-70h GPU)

**Deliverable:** Full scaling curve measured. Mini + Base c.1 + Medium
+ Stretch all trained; ≥3 data points for scaling-law table.

#### 2.1 Base c.1 full Chinchilla (~11h GPU)
- [ ] Launch `python scripts/train_base.py` (no `--batch-size` override
  = default batch=8 = 982M tokens, 21 tok/p Chinchilla-optimal)
- [ ] Expected val_loss: 3.85-4.00 (projected from c.2 curve)
- [ ] Gate: val_loss < 4.2 (per GATE_CALIBRATION) — should easily PASS
- [ ] Paper Table 2 Base row finalized

#### 2.2 Medium training (~11h GPU)
- [ ] Write `scripts/train_medium.py` (mirror of train_base.py)
- [ ] Launch: 71.3M × 2B tokens (Chinchilla-optimal)
- [ ] Expected val_loss: 3.6-3.8 (from §3 GATE_CALIBRATION projection)
- [ ] Gate: val_loss < 4.0 — strong PASS projection
- [ ] If FAIL: arch has real ceiling → trigger RWKV-7 ablation
  (FJQ_PHASE_D_ARCH.md §3)

#### 2.3 Stretch training (~17 days GPU, gated)
- [ ] Conditional on Medium PASS
- [ ] Pre-stage SlimPajama-627B-Reupload subset (15B tokens ~ 60 GB)
- [ ] `scripts/cleanup_hf_cache.py --check-free --threshold-gb 30` as
  crontab during run (prevent disk exhaustion)
- [ ] Launch with checkpoint every 50K steps
- [ ] Gate: val_loss < 3.7 (matches Zhu et al. scale expectation)
- [ ] If Stretch FAIL: paper submits at Medium with Stretch as "future
  work"

#### 2.4 QAT ablations (sequential, each run ~4h)
- [ ] Baseline: BitNet-static (no adaptive bits, no permutation, no
  re-cal) — separate run on Base config
- [ ] +adaptive bits only
- [ ] +adaptive bits + permutation
- [ ] +adaptive bits + permutation + periodic re-cal (full IntLLM)
- [ ] Paper Table 4 QAT ablation populated

**§6.8 compliance per run:**
- R1 pre-flight: C.P4.0 POL gate before each config
- R5 variance tag per commit
- R6 gate decision doc per config

---

### PHASE 3: Canonical benchmarks + baseline parity (~1 week, 10-20h GPU + 10h human)

**Deliverable:** Full benchmark table for paper §4.2. All numerics
traceable to JSON artifacts.

#### 3.1 Pin evaluation harness
- [ ] `eval/HARNESS_SHA` — commit lm-evaluation-harness SHA (currently
  v0.4.11 ≈ SHA TBD at install time)
- [ ] `pip install -e "lm-evaluation-harness @ git+$SHA"` documented
  in `docs/REPRODUCIBILITY.md`
- [ ] CI fails if drift detected

#### 3.2 `make bench-canonical`
- [ ] Script: `scripts/bench_canonical.py` runs 8 zero-shot tasks
  (wikitext, lambada_openai, hellaswag, piqa, winogrande, arc_easy,
  arc_challenge, openbookqa) on each trained IntLLM size
- [ ] Emit JSON per model in `paper/results/bench_canonical_<model>.json`
- [ ] Target: 4 IntLLM sizes (Mini, Base c.1, Medium, Stretch) +
  apples-to-apples baselines

#### 3.3 `make bench-knowledge`
- [ ] Script: `scripts/bench_knowledge.py` — mmlu 5-shot + triviaqa
  5-shot + boolq 0-shot matching BitNet 2B4T Table 1
- [ ] Results per size in `paper/results/bench_knowledge_<model>.json`

#### 3.4 Baseline re-runs
- [ ] `scripts/bench_baselines.py` — download + evaluate on SAME
  harness:
  - BitNet b1.58 2B4T (`microsoft/bitnet-b1.58-2B-4T`)
  - MatMul-Free 370M / 1.3B / 2.7B (`ridger/MMfreeLM-*`)
  - SmolLM2-135M / 360M (`HuggingFaceTB/SmolLM2-*`)
  - Pythia-160M (`EleutherAI/pythia-160m`)
- [ ] `docs/BASELINE_UNPORTED_FEATURES.md` — explicit list of unported
  features (e.g., BitNet CUDA pack-store pipeline)

#### 3.5 fp16-vs-ternary parity gate (IntLLM differentiator)
- [ ] Implement `--fp16-reference` flag in training driver producing
  bf16 forward output
- [ ] `make test-intllm-fp16-parity` — assert per-layer rel-error
  < 5% between ternary + fp16 reference for fixed prompt
- [ ] This is NEW — BitNet 2B4T did not ship a parity gate

---

### PHASE 4: Paper draft + verification script (~2 weeks, 20-30h human)

**Deliverable:** MLSys 10-page PDF; `verify_paper_tables.py --strict`
exit 0.

#### 4.1 Related-work finalization
- [ ] `docs/INTLLM_RELATED_WORK.md` (≥8 papers cited) committed
  BEFORE results section drafted (§6.9 R2)
- [ ] Papers: BitNet 2B4T, MatMul-Free, SmolLM2, Pythia, TeLLMe,
  TerEffic, SLM-Bench, seL4

#### 4.2 Populate Tables 2-7
- [ ] Table 2 main results: per-config PPL + zero-shot × 4 IntLLM
  sizes + baselines
- [ ] Table 3 fp16-shadow vs QAT gap
- [ ] Table 4 FajarOS Nova E2E metrics (tok/s + watts per size)
- [ ] Table 5 QAT ablation (from Phase 2.4 runs)
- [ ] Table 6 arch ablation (conditional on recall trigger)
- [ ] Table 7 fixed-point scale sensitivity
- [ ] Every cell references a JSON file in `paper/results/`

#### 4.3 Verification script
- [ ] `scripts/verify_paper_tables.py` — parses PDF + cross-refs
  every number against JSON artifacts
- [ ] `--strict` mode: exit 2 on any cell not matching within 0.01
  absolute tolerance
- [ ] Pre-commit hook on `paper/` directory
- [ ] CI required check before release

#### 4.4 Results-section writing
- [ ] Per §6.9 R6: write results section LAST, after all Tables 2-7
  populated + verify script green
- [ ] No quantitative claim in paper text that isn't in a verified
  JSON artifact

---

### PHASE 5: MLSys artifact submission (~1 week, 10-15h human)

**Deliverable:** MLSys AE "Functional" badge ready.

#### 5.1 Artifact appendix
- [ ] `docs/ARTIFACT_APPENDIX.md` follows ACM AE template v1.1:
  - Hardware deps (GPU model, VRAM, RAM, disk)
  - Software deps (CUDA version, Python, PyTorch, lm-eval SHA,
    Fajar Lang commit SHA, FajarOS commit SHA)
  - Benchmark deps (lm-eval-harness exact version)
  - Dataset deps (DKYoon/SlimPajama-6B + Mistral-7B-v0.3 tokenizer)
  - Exact QEMU invocation for kernel path

#### 5.2 Sample dataset
- [ ] ≤ 50 MB committed to repo:
  - WikiText-2 slice (2 MB)
  - 1-layer IntLLM weight (~30 MB)
  - 100-token prompt set (10 KB)
- [ ] Reviewers run smoke without downloading 60 GB SlimPajama

#### 5.3 Smoke script
- [ ] `run_smoke.sh` — end-to-end under 30 min on one consumer GPU:
  - Downloads sample dataset (if not cached)
  - Runs inference on 10 prompts
  - Produces one Table 2 row (Mini config numbers)
  - Exits 0 if numbers within 10% of paper

#### 5.4 QEMU bootable ISO
- [ ] `make kernel-artifact` — produces `fajaros-intllm.iso` + sample
  `.fjm v9` model on NVMe image
- [ ] Reviewer workflow: download ISO, `qemu ... -cdrom ...`, see
  kernel boot + `ask "hello"` → token stream
- [ ] Serial-log assertion script for reviewer verification

#### 5.5 Public GitHub release
- [ ] Tag `v0.1.0-intllm-phase-d` on fajarquant + fajaros-x86
- [ ] GitHub Release with:
  - Artifact appendix
  - Sample dataset
  - QEMU ISO
  - Pinned dependency SHAs
  - `reproduce.sh` and `run_smoke.sh`

---

## 4. Timeline + effort rollup

| Phase | Calendar | Human (h) | GPU (h) | Gates |
|---|---|---|---|---|
| 1 Kernel integration | Week 1 | 15-20 | 0 | test-intllm-kernel-path |
| 2 Training completion | Weeks 2-5 | 10-15 | 40-70 | Medium + Stretch gates |
| 3 Canonical benchmarks | Week 6 | 10 | 10-20 | bench-canonical, bench-knowledge, fp16-parity |
| 4 Paper + verify | Weeks 7-8 | 20-30 | 0 | verify_paper_tables.py --strict |
| 5 MLSys artifact | Week 9 | 10-15 | 2-4 | run_smoke.sh < 30 min |
| **Total** | **~9 weeks** | **65-90** | **52-94** | **20/20 checklist items** |

GPU budget in USD (RTX 4090 @ 450W @ $0.15/kWh):
- Worst-case 94h × 0.45 × 0.15 ≈ **$6.35** electricity
- Plus ~$0.20 for Base c.1 + ~$0.10 for Medium in the long runs

Worst-case human time: 90h ≈ 2 weeks full-time or 9 weeks part-time.

---

## 5. Decision gates (§6.8 R6 mechanical)

Per-phase gates that commit to committed files:

| Phase | Decision file | Blocks |
|---|---|---|
| 1 | `FJQ_PHASE_D_P5_KERNEL_GATE.md` | Phase 2 downstream |
| 2.1 | Updated `FJQ_PHASE_D_P4_1_5_BASE_GATE.md` (c.1 row) | Phase 4.2 Table 2 Base row |
| 2.2 | `FJQ_PHASE_D_MEDIUM_GATE.md` | Phase 2.3 Stretch |
| 2.3 | `FJQ_PHASE_D_STRETCH_GATE.md` | Phase 3.2 Stretch bench |
| 3.5 | `FJQ_PHASE_D_FP16_PARITY_GATE.md` | Phase 4.2 Table 3 |
| 4.3 | `FJQ_PHASE_D_VERIFY_SCRIPT_GATE.md` | Phase 5 submission |
| 5 | `FJQ_PHASE_D_ARTIFACT_GATE.md` (final) | Publish + tag |

---

## 6. Risk register + mitigations

| Risk | Prob | Mitigation |
|---|---|---|
| Medium gate FAIL at <4.0 | Medium | §3 GATE_CALIBRATION projected PASS from scale-curve; if FAIL, trigger RWKV-7 ablation (FJQ_PHASE_D_ARCH.md §3) |
| Stretch 17d run crashes mid-way | Medium | Checkpoint every 50K steps; cleanup_hf_cache.py <30GB threshold; restart from last ckpt |
| `@no_vectorize` unavailable for @kernel leads to O2 miscompile | Low | Fallback to @noinline + gcc-compiled C helper (V30 Track 3 pattern). Verify via nm + objdump at Phase 1.5 |
| fp16 parity rel-error >5% | Low | Known kernel arith is ternary signed-add; rel-error depends on β precision (IEEE 754 decoder); if FAIL, widen β fixed-point scale to x10000 |
| MLSys AE rejection | Low | Only "Functional" badge required; sample dataset + run_smoke keep reproducibility cheap |
| Public artifact drift (README, tags) | Medium | §6.8 R7 checklist per commit; automate with GitHub Actions release workflow |

---

## 7. §6.8 + §6.9 self-check for THIS plan

- [x] **R1 Pre-flight audit done** — hands-on via Explore agent +
       research sweep via general-purpose agent. 2 audit
       discrepancies surfaced + noted.
- [x] **R2 Verifications are runnable commands** — every gate in §3
       is a `make <target>` or script invocation
- [x] **R3 Prevention layer per phase** — each phase ships ≥1 Make
       gate, pre-commit hook, or CI job
- [x] **R4 Multi-agent audit cross-checked** — 2 agents (Explore +
       research), both verified manually before commit (audit claim
       about `mdl_v9_get_beta_x1000` stub status corrected against
       actual file)
- [x] **R5 Surprise budget** — this is a 9-week plan; budget
       assumes +25% variance default, escalating to +40% if Phase 1
       kernel integration exceeds 20h
- [x] **R6 Mechanical decision gates** — §5 table enumerates 7
       committed-file decision gates
- [x] **R7 Public-facing artifact sync** — Phase 5.5 covers
       README/tags/releases
- [x] **R8 Multi-repo state check** — Phases 1 (fajaros-x86), 2-4
       (fajarquant), 5 (both); cross-repo sync required

- [x] **R1-R7 §6.9 Research Integrity** — Phase 3 canonical protocol
       (R1), Phase 4.1 literature review (R2), Phase 3.4 baseline
       parity (R3), calibrated-once-not-per-chunk inherited from
       FJQ_PHASE_D_OPS.md §4 (R4), outlier handling inherited from
       intllm/qat.py (R5), results-LAST §4.4 (R6), `verify_paper_
       tables.py --strict` §4.3 (R7)

---

## 8. What this document commits to

1. **20-item production checklist** is the binding bar. Shipping =
   20/20 green.
2. **5-phase sequenced plan** with pre-flight audits per phase
3. **7 mechanical decision gates** committed as files
4. **9-week + 90h + ~$7 GPU** realistic budget
5. **Audit-grade verification** via `nm`, pytest, Make targets

Current state: **~40% complete** (Python side full, kernel side 50%,
training 50%, paper 10%, artifact 0%). Gap-to-100% is the above plan.

---

## 9. Session handoff

Next session's options (order of strategic priority):

**P1 (must-do-first):** Phase 1.1-1.2 (kernel SOURCES reorder + v9
detection wire-up + checkpoint rename bug fix). ~4-6h. Unblocks the
remaining ~30 items.

**P2 (high value parallel):** Phase 2.1 Base c.1 Chinchilla run (~11h
GPU overnight). Fills paper Table 2 Base row.

**P3 (documentation):** Phase 4.1 related-work doc (~2h, no GPU).
Unblocks paper results section.

**P4 (meta):** this plan's own commit is the unblock-everything move.
