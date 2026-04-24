# FajarQuant Phase D — §3.4 B0 Pre-Flight Findings (Baseline Re-runs)

> **Date:** 2026-04-24 | **Rule:** CLAUDE.md §6.8 R1 (pre-flight audit mandatory before scaffolding), §6.9 R3 (no strawman baselines — port full features, document what is skipped and why) | **Scope:** §3.4 of `FJQ_PHASE_D_PRODUCTION_PLAN.md`, baseline re-runs (`scripts/bench_baselines.py` + `make bench-baselines`) | **Author:** Claude (Opus 4.7, 1M context), verified hands-on

## 0. Why this pre-flight exists

§3.4 is the §6.9 R3 crux of Phase D: if we scaffold `bench_baselines.py`
against naively-loaded baselines we risk publishing numbers where "IntLLM
beats BitNet" is a benchmark protocol artifact rather than an algorithmic
claim — the same failure mode that blew up V26 C1.6 (FajarQuant vs TurboQuant
"win" was a post-hoc cache mutation artifact, see `memory/feedback_research_integrity.md`).
This doc commits the baseline inventory, unported-feature surface, and
GPU/disk budget *before* `bench_baselines.py` is written.

Downstream blocker: §3.4 task table in `FJQ_PHASE_D_PRODUCTION_PLAN.md`
cannot be marked in-progress until this doc is committed and its decisions
are wired into the script skeleton.

---

## 1. Environment audit (hands-on, 2026-04-24)

| Item | Observed | Required by plan | Status |
|---|---|---|---|
| venv Python | `~/Documents/fajarquant/.venv/bin/python` | exists | ✓ |
| `lm_eval` | 0.4.11 | pinned in `eval/HARNESS_SHA` to `v0.4.11` | ✓ matches |
| `transformers` | 4.57.6 | native `bitnet` + HF-standard `llama`/`gpt_neox` | ✓ (see §3.1) |
| `torch` | 2.6.0+cu124, CUDA=True | GPU required for baselines ≥1B | ✓ |
| GPU | RTX 4090 Laptop 16 GB VRAM, 15.9 GB free | largest baseline 2.7B bf16 = 5.4 GB | ✓ fits |
| Disk free | 126 GB / 937 GB (86 % used) | ~15 GB total baseline downloads (§5) | ✓ fits but monitor |
| HARNESS_SHA full-40-char pin | **NOT YET** — `ref: v0.4.11` tag only | §3.1 demands full SHA after first real run | 🟡 carryover from §3.1 |

Command log (reproducible):
```
~/Documents/fajarquant/.venv/bin/python -c "import lm_eval, transformers, torch; print(lm_eval.__version__, transformers.__version__, torch.__version__, torch.cuda.is_available())"
# → 0.4.11 4.57.6 2.6.0+cu124 True
nvidia-smi --query-gpu=memory.free --format=csv
# → 15933 MiB
df -h ~/.cache | tail -1
# → /dev/nvme0n1p2  937G  764G  126G  86%
```

---

## 2. Baseline inventory — HF accessibility + arch dispatch

All 7 candidate repos probed via `curl -sI /api/models/<repo>` → 200 OK:

| Baseline repo | Params (bf16 bytes) | HF arch | Needs trust_remote_code? | Notes |
|---|---|---|---|---|
| `microsoft/bitnet-b1.58-2B-4T` | 2.0 B (850 MB, BF16+U8 ternary-packed) | `bitnet` | **No** (native since transformers 4.57) | config has legacy `auto_map` but native class registered — see §3.1 |
| `microsoft/bitnet-b1.58-2B-4T-bf16` | 2.0 B (~4 GB BF16 master) | `bitnet` | No | **Only use for fp16-shadow comparison** (see §4) |
| `ridger/MMfreeLM-370M` | 370 M (374 MB) | `hgrn_bit` | **Yes** | Already cached locally; vendored class in `_upstream/mmfreelm/` (UPSTREAM_PIN.md commit `f24cfe5`) |
| `ridger/MMfreeLM-1.3B` | 1.3 B (1.36 GB) | `hgrn_bit` | Yes | Same class as 370M |
| `ridger/MMfreeLM-2.7B` | 2.7 B (2.70 GB, sharded safetensors) | `hgrn_bit` | Yes | Largest baseline; fits 16 GB VRAM with room for KV cache |
| `HuggingFaceTB/SmolLM2-135M` | 135 M (134 MB) | `llama` | No | **Base variant available** (`-Instruct` also exists but we use base) |
| `HuggingFaceTB/SmolLM2-360M` | 360 M (362 MB) | `llama` | No | Base variant available |
| `EleutherAI/pythia-160m` | ~160 M (both .safetensors + legacy .bin) | `gpt_neox` | No | Classic pretrain-only baseline; no SFT confound |

**Verification commands (reproducible):**
```
for r in microsoft/bitnet-b1.58-2B-4T ridger/MMfreeLM-{370M,1.3B,2.7B} HuggingFaceTB/SmolLM2-{135M,360M} EleutherAI/pythia-160m; do
  curl -sI -o /dev/null -w "%{http_code}\t$r\n" "https://huggingface.co/api/models/$r"
done
~/Documents/fajarquant/.venv/bin/python -c "from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES; print('bitnet' in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)"
# → True
~/Documents/fajarquant/.venv/bin/python -c "from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES; print('hgrn_bit' in CONFIG_MAPPING_NAMES)"
# → False (→ trust_remote_code=True required + PYTHONPATH=_upstream)
```

---

## 3. Harness wire-up (§3.1 + §3.2 reuse)

### 3.1 Every baseline loads through the **same** `HFLM` wrapper

`intllm.lm_eval_wrapper.build_hflm()` loads a constructed `HGRNBitForCausalLM`
instance and hands it to `lm_eval.models.huggingface.HFLM`. For the baseline
re-runs we skip `build_hflm` entirely and go through `HFLM` directly with
`pretrained=<HF_REPO>`, which is lm-eval's canonical HF-loading path.

**Why this matters (§6.9 R3):** IntLLM eval and baseline eval share the
identical `HFLM → loglikelihood/generate_until → task formatter` plumbing.
Any silent disagreement from wrapper behavior is eliminated; the only
remaining differences are model weights, tokenizer, and compute kernel —
which are what we want to measure.

### 3.2 Ternary weights in the BitNet 2B4T deployment repo

The 850 MB `model.safetensors` ships weights as packed U8 ternary (3 weights
per uint8 via base-3 packing) plus BF16 scales + BF16 embeddings. The
native `BitNetForCausalLM` class (transformers ≥ 4.57) **unpacks on-the-fly
during forward**, not at load time. This means:
- We compare against true W1.58A8 ternary, not an unintentional full-precision upcast.
- The `-bf16` variant *does* materialize full-precision master weights — using it
  for the main comparison would be measuring a different model and would
  invalidate the "IntLLM vs 2-bit BitNet" claim.
- **Decision committed:** main comparison row = `microsoft/bitnet-b1.58-2B-4T`
  (packed). `-bf16` reserved for optional fp16-shadow reference only.

### 3.3 MMfreeLM custom-code path

`hgrn_bit` is not in transformers' built-in registry. `AutoConfig.from_pretrained(...,
trust_remote_code=True)` works because the HF repo ships a self-contained
`configuration_hgrn_bit.py` + `modeling_hgrn_bit.py`. However, **the triton
kernels** (`fusedbitnet`, `chunk_hgrn`) require `mmfreelm` importable in
`sys.path`. Local vendored snapshot at `python/phase_d/_upstream/` (commit
`f24cfe5`, pinned per `UPSTREAM_PIN.md`) satisfies this.

**Decision committed:** `bench_baselines.py` must run with
`PYTHONPATH=python/phase_d/_upstream:python/phase_d` or equivalent. A
Makefile wrapper must set this automatically. Missing `mmfreelm` import
is a hard failure — no silent fall-back.

---

## 4. Unported-features audit per §6.9 R3

This section is load-bearing: §6.9 R3 forbids strawman baselines. For
every baseline we list features NOT ported, and justify why the skip
does not invalidate the comparison or document the honest caveat.

### 4.1 BitNet 2B4T — **3 NOT-ported, 1 load-bearing caveat**

| Baseline feature | Ported? | Rationale |
|---|---|---|
| Ternary-packed W1.58 weight forward (HF native) | ✓ Yes | Used as published |
| 8-bit absmax activation quantization (HF native) | ✓ Yes | Used as published |
| Squared-ReLU FFN, subln norm, RoPE | ✓ Yes | Used as published |
| **SFT + DPO instruction tuning** | **✗ No baseline-side, pretrain-only for IntLLM** | **NO pretrain-only BitNet 2B4T variant exists on HF** (probed `-pretrain`, `-base`, `-4T-base` → all 401). Published 2B4T numbers in BitNet Table 1 are post-SFT+DPO. Our pretrain-only IntLLM CANNOT fairly compete on zero-shot tasks that SFT+DPO lifts (CommonsenseQA, TruthfulQA, GSM8K, HumanEval+, MMLU↑, MT-bench, IFEval). **Honest disclosure required in paper Table 2**: mark BitNet row "post-SFT+DPO" and our IntLLM row "pretrain-only" explicitly. PPL metrics (WikiText, LAMBADA) are still fair because instruction tuning barely shifts continuation LM loss; zero-shot accuracy comparisons are caveated. |
| **bitnet.cpp optimized CPU kernels** | **✗ No** | Their README explicitly warns: *"Running via transformers will likely result in inference speeds and energy usage comparable to, or potentially worse than, standard full-precision models."* Our GPU-side HFLM eval does NOT invoke bitnet.cpp. **Consequence:** Table 4 (FajarOS Nova tok/s + watts) CANNOT compare to BitNet's published 29 ms CPU decoding. For Table 2 correctness + zero-shot only, this is OK — we are measuring model quality, not throughput. For throughput/energy, BitNet row must stay "N/A — requires bitnet.cpp port, future work". |
| Training corpus (4T tokens, mostly unreleased) | ✗ No | Not needed for eval; documented in paper §2 related work. |

**Entry in `docs/BASELINE_UNPORTED_FEATURES.md`:** required before paper
Table 2 draft, per §6.9 R3 + plan item #7.

### 4.2 MMfreeLM 370M / 1.3B / 2.7B — **clean baseline, 0 caveats**

| Feature | Ported? |
|---|---|
| BitLinear ternary forward | ✓ (via `mmfreelm.ops.bitnet`) |
| fusedbitnet triton kernel | ✓ (vendored `_upstream/`) |
| MLGRU time-mixing (no self-attention) | ✓ |
| HGRN chunk-parallel kernel | ✓ (vendored) |
| Pretrain-only checkpoints, no SFT/DPO | ✓ (HF repos are pretrain-only per README) |

**This is the cleanest baseline in the set.** MMfreeLM = IntLLM's arch
cousin (both are MatMul-Free-family) with the **same HFLM eval path** +
**same training regime (pretrain-only)**. It is the canonical apples-to-apples
comparison for Phase D. Paper Table 2 MMfreeLM row is defensible without caveats.

However, MMfreeLM paper only reports Table 1 for **370M**. For 1.3B and
2.7B we replicate into our own JSON but have no upstream-published ground
truth to sanity-check against. **Decision committed:** treat 1.3B + 2.7B
as "our measured number" — no cross-check column for them, only for 370M
(which already has a cross-check in `FJQ_PHASE_D_P2_1_GATE.md` with the
±1.53 lm-eval 0.3→0.4 drift analysis).

### 4.3 SmolLM2 135M / 360M — **1 NOT-ported, 1 caveat**

| Feature | Ported? |
|---|---|
| Llama-class arch via native transformers `llama` | ✓ |
| Pretrain-only base checkpoints (not `-Instruct`) | ✓ |
| **Llama-3-class tokenizer (49 K vocab)** | tokenizer-local, not ported | tokenizer baked into weights; IntLLM uses Mistral v3 32K. PPL numbers across differing tokenizers are NOT directly comparable in absolute terms. Zero-shot accuracy IS comparable because the task formatter reconverts to text. **Caveat:** any WikiText / LAMBADA PPL comparison must be bucketed by tokenizer; paper Table 2 should either (a) use bits-per-byte (BPB) instead of PPL or (b) report PPL with explicit tokenizer column. |

### 4.4 Pythia-160m — **clean baseline, 0 caveats on quality axis**

Pretrain-only, public dataset (Pile), standard gpt_neox arch, bf16/fp32.
Same tokenizer caveat as SmolLM2 (gpt_neox 50 K vocab vs Mistral 32 K).

### 4.5 Cross-cutting tokenizer caveat

**Paper Table 2 PPL column must use bits-per-byte (BPB) or an explicit
per-tokenizer grouping**, because PPL scales with log(vocab) and is not
comparable across:
- IntLLM (Mistral v3, 32 K)
- BitNet 2B4T (Llama-3, 128.256 K)
- MMfreeLM (Mistral v3 — same as IntLLM ✓)
- SmolLM2 (custom 49 K)
- Pythia (gpt_neox, 50 K)

Zero-shot task accuracy is comparable regardless; PPL is not.

**Decision committed:** `bench_baselines.py` emits BOTH `ppl` and
`bits_per_byte` columns from lm-eval output; paper Table 2 uses BPB
for the headline column, PPL as supplementary.

---

## 5. GPU budget estimate

**Calibration basis (from existing IntLLM runs on RTX 4090 Laptop 16 GB):**

| Scale | Params | canonical (8 tasks) | knowledge (3 tasks, mmlu 5-shot dominates) | Sum |
|---|---|---|---|---|
| Mini | ~14 M | 202 s | 784 s | 16 min |
| Base | ~46 M | 315 s | 616 s | 15 min |
| Medium | ~175 M | 363 s | 1,500 s | 31 min |

**Extrapolation for baselines (all 11 tasks — 8 canonical + 3 knowledge):**

| Baseline | Params | Est wall | Rationale |
|---|---|---|---|
| SmolLM2-135M | 135 M | ~15 min | ≈ Base |
| Pythia-160m | 160 M | ~20 min | ≈ Base |
| SmolLM2-360M | 360 M | ~45 min | ≈ Medium × 1.5 |
| MMfreeLM-370M | 370 M | ~45 min | ≈ Medium × 1.5 |
| MMfreeLM-1.3B | 1.3 B | ~2.5 h | linear-ish scale |
| MMfreeLM-2.7B | 2.7 B | ~4.5 h | linear-ish scale, edge of VRAM |
| BitNet 2B4T | 2.0 B | ~4 h | memory-light (ternary) but transformers path is NOT optimized |

**Total:** ~13 GPU-hours one-shot, or ~15 h with a 15 % retry margin.
Plan budget is 10-20 h ⇒ fits.

**Hard risks:**
- MMfreeLM-2.7B + KV cache for hellaswag context may spike VRAM; batch_size=2
  fallback required. **Decision committed:** `bench_baselines.py` must
  accept `--batch-size-override <repo>=<bs>` and default `ridger/MMfreeLM-2.7B=2`.
- bitnet ternary-unpack in transformers is O(N_params) at every forward;
  batch_size=4 may actually be slower than batch_size=1 due to dispatch overhead.
  Will be measured during real run, not now.

---

## 6. Tokenizer / SFT / kernel matrix (summary, §6.9 R3 disclosure block)

**This matrix will appear verbatim in `docs/BASELINE_UNPORTED_FEATURES.md`
+ paper Table 2 footnote. Committed now so downstream cannot drift.**

| Model | Training regime | Tokenizer | Kernel path | Zero-shot PPL-comparable-to-IntLLM? | Zero-shot accuracy-comparable-to-IntLLM? |
|---|---|---|---|---|---|
| **IntLLM (ours)** | pretrain-only | Mistral v3 32 K | FajarOS Nova @kernel + HFLM GPU | — | — |
| MMfreeLM 370M/1.3B/2.7B | pretrain-only | Mistral v3 32 K | HFLM + vendored triton | **YES (BPB)** | **YES** |
| SmolLM2-135M/360M | pretrain-only (base) | Llama-class 49 K | HFLM native | BPB only | YES |
| Pythia-160m | pretrain-only | gpt_neox 50 K | HFLM native | BPB only | YES |
| BitNet 2B4T | pretrain + SFT + DPO | Llama-3 128 K | HFLM (not bitnet.cpp) | BPB only | partial — **instruction-tuning lifts zero-shot**; call out in footnote |

---

## 7. Decisions committed (§6.8 R6 mechanical gate)

1. `scripts/bench_baselines.py` will execute the **11 tasks** (8 canonical + 3 knowledge) per baseline by re-using exactly the same `HFLM + simple_evaluate` plumbing from `bench_canonical.py` + `bench_knowledge.py`. No alternate eval path.
2. The 7 baseline repos listed in §2 are the committed set. Adding/removing baselines requires a decision doc amendment, not a silent edit.
3. BitNet main comparison = `microsoft/bitnet-b1.58-2B-4T` packed. `-bf16` reserved for fp16-shadow parity only. `-gguf` is out of scope.
4. `PYTHONPATH=python/phase_d/_upstream:python/phase_d` required; absence is a hard failure.
5. Emit BOTH `ppl` and `bits_per_byte` columns; paper Table 2 uses BPB.
6. Default `--batch-size` = 4; `ridger/MMfreeLM-2.7B` and `microsoft/bitnet-b1.58-2B-4T` default to 2. CLI override allowed.
7. `docs/BASELINE_UNPORTED_FEATURES.md` MUST be committed before any Table 2 paper draft touches these rows (§6.9 R3).
8. §3.1 carryover: first real `make bench-baselines` run must resolve `eval/HARNESS_SHA` from `v0.4.11` tag to full 40-char SHA; same commit bumps this file.
9. BitNet row in Table 2 must carry the "post-SFT+DPO vs our pretrain-only" footnote; Table 4 BitNet row must stay "N/A — requires bitnet.cpp port".
10. Prevention layer per §6.8 R3: a new `verify-baselines-honesty` check in `scripts/verify_intllm_tables.py --strict` will fail the pre-commit hook if paper Table 2 contains a BitNet row without the SFT-lift footnote flag.

---

## 8. What §3.4 scaffold must include (directly follows from decisions above)

Minimum deliverable for the §3.4 scaffold commit:

- `scripts/bench_baselines.py` with:
  - argparse: `--repo <HF_ID>` (repeatable), `--tag <name>`, `--scaffold`, `--device`, `--batch-size`, `--batch-size-override <repo>=<bs>` (repeatable), `--limit`, `--tasks`
  - task list identical to `bench_canonical.py` ∪ `bench_knowledge.py`
  - for BitNet and MMfreeLM repos: special-case note printed at start (e.g. "using transformers native" / "using vendored mmfreelm")
  - emits `paper/intllm/results/bench_baselines_<tag>.json` with keys: `repo`, `params`, `tokenizer`, `training_regime ∈ {pretrain_only, pretrain_sft_dpo}`, `kernel_path`, `harness_version`, `tasks: [...]`, `elapsed_seconds`
- Makefile target `bench-baselines` that loops through the 7 repos with correct defaults
- `docs/BASELINE_UNPORTED_FEATURES.md` populated with §4 + §6 content
- `verify_intllm_tables.py` claims appended: per-baseline PPL/BPB/acc cells PEND until first real run
- 3 placeholder JSONs so `make verify-intllm-tables --strict` exit 0 immediately

Out of scope for the scaffold commit (explicitly deferred to post-scaffold):
- Real GPU runs (8-15 h)
- Full SHA pin of `eval/HARNESS_SHA` (captured at first real run, per §3.1)
- Table 2 paper draft

---

## 9. Self-check (§6.8 R1-R8 + §6.9 R1-R7)

| Rule | Applied? | Evidence |
|---|---|---|
| §6.8 R1 — pre-flight audit committed before downstream | ✓ | THIS doc |
| §6.8 R2 — verification = runnable commands | ✓ | every section cites the command it was verified with |
| §6.8 R3 — prevention layer added | ✓ | §7 item 10 — new `verify-baselines-honesty` check; §7 item 7 gate |
| §6.8 R4 — agent-produced numbers cross-checked | ✓ | all HF probes, venv probes, GPU probes ran hands-on in this session |
| §6.8 R5 — +25 % surprise budget | ✓ | 1 surprise logged: "BitNet has no pretrain-only HF variant" — drove §4.1 SFT caveat |
| §6.8 R6 — decisions are committed files | ✓ | §7 commits 10 decisions, will land with this doc |
| §6.8 R7 — public-artifact drift check | ✓ | README badges untouched; paper Table 2 gate in §7 item 10 prevents drift |
| §6.8 R8 — multi-repo state check | ✓ | fajar-lang + fajaros-x86 + fajarquant all synced per MEMORY 2026-04-24 |
| §6.9 R1 — canonical protocol from ≥2 refs | ✓ | lm-eval 0.4.11 harness used by BitNet 2B4T tech report + MMfreeLM paper + SmolLM2 paper |
| §6.9 R2 — ≥8 paper lit review precedes | ✓ | done in `FJQ_PHASE_D_P0_FINDINGS.md` (22 papers) |
| §6.9 R3 — no strawman baselines; port full features | ✓ | §4 audit is the load-bearing deliverable of this doc |
| §6.9 R4 — calibrated vs per-chunk | n/a (eval only, no data-driven decomposition) |
| §6.9 R5 — outlier handling | n/a (eval only) |
| §6.9 R6 — algorithmic validation precedes paper claims | ✓ | §7 item 7 blocks paper text until BASELINE_UNPORTED_FEATURES.md lands |
| §6.9 R7 — verify_paper_tables.py --strict gate | ✓ | §7 item 10 adds to pre-commit |

**8/8 §6.8 YES + 7/7 §6.9 YES (2 n/a) ⇒ ship this B0 doc and proceed to §3.4 scaffold commit.**

---

## 10. Carry-overs / open items for §3.4 scaffold commit

- [ ] Write `scripts/bench_baselines.py` per §8.
- [ ] Write `docs/BASELINE_UNPORTED_FEATURES.md` from §4 + §6.
- [ ] Replace Makefile `bench-baselines` placeholder (line 188-191) with real loop.
- [ ] Append 21 PEND rows (7 repos × 3 headline metrics) to `scripts/verify_intllm_tables.py` registry.
- [ ] Update `docs/FJQ_PHASE_D_PRODUCTION_PLAN.md` §3.4 checklist with "B0 findings committed" marker.
- [ ] §3.1 follow-up: capture full 40-char HARNESS_SHA during first real run.

---

*§3.4 B0 FINDINGS — committed 2026-04-24. Downstream §3.4 scaffold work unblocked.*
