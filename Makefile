# FajarQuant top-level Make targets — Phase D IntLLM gates.
#
# Per FJQ_PHASE_D_PRODUCTION_PLAN.md §3-§5, every paper claim is backed
# by one of these targets. Phase numbers reference that plan.
#
# Most targets shell out to `python/phase_d/scripts/*.py` and assume a
# pre-built venv at .venv/ with the Phase D dependencies installed.

.PHONY: help
help:
	@echo "FajarQuant Phase D Make targets:"
	@echo "  verify-intllm-tables       Phase 4.3  — paper-claim gate (--strict)"
	@echo "  test-intllm-fp16-parity    Phase 3.5  — fp16-vs-ternary smoke on Mini"
	@echo "  test-phase-d               Phase 4    — pytest tests/ in phase_d (82)"
	@echo "  test-train-watchdog        Track B §6.11 — training-script resilience gate"
	@echo "  fp16-parity-mini           Phase 3.5  — measure + emit JSON for Mini"
	@echo "  fp16-parity-base           Phase 3.5  — measure + emit JSON for Base"
	@echo ""
	@echo "  bench-canonical            Phase 3.2  — scaffold JSON for all 3 sizes (CPU, <1s)"
	@echo "  bench-canonical-real TAG=X Phase 3.2  — real lm-eval via HFLM (needs GPU+ckpt)"
	@echo "  bench-knowledge            Phase 3.3  — scaffold JSON (mmlu/triviaqa/boolq, CPU)"
	@echo "  bench-knowledge-real TAG=X Phase 3.3  — real lm-eval knowledge tasks (GPU+ckpt)"
	@echo "  bench-baselines            Phase 3.4  — scaffold JSON for all 7 baselines (CPU, <5s)"
	@echo "  bench-baselines-one REPO=X Phase 3.4  — scaffold a single baseline repo"
	@echo "  bench-baselines-real REPO=X Phase 3.4 — real lm-eval on one HF baseline (GPU)"
	@echo "  bench-baselines-real-all   Phase 3.4  — real lm-eval sweep (~13 GPU-hours)"
	@echo ""
	@echo "FajarQuant Phase E Make targets:"
	@echo "  dedup-corpus-id-dryrun     Phase E1.4 — MinHash LSH dry-run on 10K ID-corpus sample"
	@echo "  dedup-corpus-id            Phase E1.4 — MinHash LSH full ID-corpus dedup (resumable)"
	@echo "  test-dedup-resume          Phase E1.4.1 — resume-state smoke gate (synthetic, ~3s)"
	@echo "  audit-dedup-sweep          Phase E1.4.2 — post-sweep read-only audit"
	@echo "  dedup-corpus-id-exact-dryrun Phase E1.4.3 — sha256 exact-hash dry-run on 10K ID-corpus"
	@echo "  dedup-corpus-id-exact      Phase E1.4.3 — sha256 exact-hash full ID-corpus dedup (resumable)"
	@echo "  test-dedup-exact           Phase E1.4.3 — exact-hash resume-state smoke gate (synthetic, ~3s)"
	@echo "  test-intllm-en             Phase E1.2 — EN corpus shim smoke gate (constants + math, ~1s)"
	@echo "  verify-bilingual-corpus    Phase E1.6 — bilingual corpus packaging gate (spec ↔ manifests, ~1s)"
	@echo "  train-mini-ablation TAG=X  Phase E2.1.0 — Mini-scale ablation harness (TAG=baseline|hadamard|fp8_lmhead|distill|balanced_calib|lang_cond_a|combined; --proof-of-life or --dry-run for fast wiring sanity)"

PYTHON := .venv/bin/python
PHASE_D := python/phase_d
PHASE_E := python/phase_e
PYTHONPATH_PD := PYTHONPATH=$(PHASE_D)

# ─── Phase 4.3 ─── verify-intllm-tables ──────────────────────────────
.PHONY: verify-intllm-tables
verify-intllm-tables:
	@$(PYTHON) scripts/verify_intllm_tables.py --strict

# ─── Phase 4 test suite ─── all phase_d unit tests ──────────────────
.PHONY: test-phase-d
test-phase-d:
	@cd $(PHASE_D) && PYTHONPATH=. ../../$(PYTHON) -m pytest tests/ -q

# ─── Track B §6.11 ─── training-script resilience regression gate ───
#
# Exercises the full chain of interruption-safety features shipped in
# V31.C.P6.1–P6.5:
#   - StepWatchdog (real daemon thread + real SIGTERM delivery)
#   - ckpt_every intermediate saves + rotation
#   - --resume bit-exact state restoration
#   - HF streaming retry_iter
#
# Per CLAUDE.md §6.11, any PR that touches `train_*.py` or `intllm/{data,train}.py`
# MUST keep this target green. Wired into the pre-push hook so broken
# resilience code cannot reach origin.
.PHONY: test-train-watchdog
test-train-watchdog:
	@cd $(PHASE_D) && PYTHONPATH=. ../../$(PYTHON) -m pytest tests/ \
		-k "watchdog or ckpt or resume or retry_iter or timeout_env" -q

# ─── Phase 3.5 ─── fp16-vs-ternary parity (IntLLM differentiator) ───
#
# The full parity gate threshold is deferred until Base c.1 lands and
# we have a 2-point comparison (Mini + Base) to choose a defensible
# bound. Until then `test-intllm-fp16-parity` is a smoke that:
#   1. asserts the artifact JSON exists for Mini (i.e. someone has run
#      `make fp16-parity-mini` at least once after each retraining)
#   2. asserts the JSON has a populated summary block
#
# Re-run the measurement step (writes a fresh JSON) with
# `make fp16-parity-mini` after a new checkpoint.
.PHONY: test-intllm-fp16-parity
test-intllm-fp16-parity:
	@if [ ! -f paper/intllm/results/fp16_parity_intllm-mini.json ]; then \
		echo "[FAIL] missing paper/intllm/results/fp16_parity_intllm-mini.json"; \
		echo "       run \`make fp16-parity-mini\` first (needs Mini checkpoint)"; \
		exit 1; \
	fi
	@$(PYTHON) -c "import json; d = json.load(open('paper/intllm/results/fp16_parity_intllm-mini.json')); \
	    s = d['summary']; \
	    assert s['n_layers'] > 0, 'empty summary — runner produced no hooks'; \
	    print(f'[PASS] Mini fp16-parity scaffold green: {s[\"n_layers\"]} layers, ' \
	          f'rel_l2 mean={s[\"rel_error_l2_mean\"]:.4f} p50={s[\"rel_error_l2_p50\"]:.4f} ' \
	          f'max={s[\"rel_error_l2_max\"]:.4f}')"

.PHONY: fp16-parity-mini
fp16-parity-mini:
	@if [ ! -f $(PHASE_D)/checkpoints/mini/mini_final.pt ]; then \
		echo "[SKIP] no Mini checkpoint at $(PHASE_D)/checkpoints/mini/mini_final.pt"; \
		echo "       run train_mini.py to produce one"; \
		exit 0; \
	fi
	@cd $(PHASE_D) && PYTHONPATH=. ../../$(PYTHON) scripts/run_fp16_parity.py \
		--checkpoint checkpoints/mini/mini_final.pt --tag intllm-mini

.PHONY: fp16-parity-base
fp16-parity-base:
	@if [ ! -f $(PHASE_D)/checkpoints/base/base_final.pt ]; then \
		echo "[SKIP] no Base checkpoint at $(PHASE_D)/checkpoints/base/base_final.pt"; \
		echo "       Base c.1 must finish first (~12h GPU)"; \
		exit 0; \
	fi
	@cd $(PHASE_D) && PYTHONPATH=. ../../$(PYTHON) scripts/run_fp16_parity.py \
		--checkpoint checkpoints/base/base_final.pt --tag intllm-base

# ─── Phase 3.2 ─── canonical zero-shot benchmarks (scaffold today) ──
#
# Scaffold mode emits schema-valid JSON with 0.0 placeholders for every
# (task, metric) pair. This unblocks verify_intllm_tables.py Table 2
# claim registry (whose claims currently use tolerance=inf, so 0.0
# passes trivially). When a real bench run lands, it overwrites the
# scaffold values AND Phase 4.2 tightens the tolerance in
# scripts/verify_intllm_tables.py.
#
# Real mode (bench-canonical-real) requires an LM wrapper around
# HGRNBitForCausalLM — pending Phase 3.2.1 follow-up, not blocking.
.PHONY: bench-canonical
bench-canonical:
	@cd $(PHASE_D) && PYTHONPATH=. ../../$(PYTHON) scripts/bench_canonical.py \
		--tag intllm-mini --scaffold
	@cd $(PHASE_D) && PYTHONPATH=. ../../$(PYTHON) scripts/bench_canonical.py \
		--tag intllm-base --scaffold
	@cd $(PHASE_D) && PYTHONPATH=. ../../$(PYTHON) scripts/bench_canonical.py \
		--tag intllm-medium --scaffold

# Single-tag convenience: `make bench-canonical-one TAG=mini`
.PHONY: bench-canonical-one
bench-canonical-one:
	@if [ -z "$(TAG)" ]; then \
		echo "usage: make bench-canonical-one TAG=mini|base|medium"; exit 2; \
	fi
	@cd $(PHASE_D) && PYTHONPATH=. ../../$(PYTHON) scripts/bench_canonical.py \
		--tag intllm-$(TAG) --scaffold

# Real benchmark run (Phase 3.2.1 — uses intllm.lm_eval_wrapper.build_hflm)
#
# Optional vars:
#   LIMIT=<n>       cap docs per task (smoke); <1.0 = fraction, ≥1 = absolute
#   TASKS="<list>"  space-separated subset of canonical tasks
#   BATCH=<n>       HFLM batch size (default 4)
#   MAXLEN=<n>      HFLM max_length override (default: checkpoint's)
#
# Examples:
#   make bench-canonical-real TAG=mini
#   make bench-canonical-real TAG=mini LIMIT=50 TASKS="piqa hellaswag"
#   make bench-canonical-real TAG=base BATCH=8 MAXLEN=2048
.PHONY: bench-canonical-real
bench-canonical-real:
	@if [ -z "$(TAG)" ]; then \
		echo "usage: make bench-canonical-real TAG=mini|base|medium [LIMIT=n] [TASKS=\"...\"]"; \
		exit 2; \
	fi
	@cd $(PHASE_D) && PYTHONPATH=. ../../$(PYTHON) scripts/bench_canonical.py \
		--tag intllm-$(TAG) \
		--checkpoint checkpoints/$(TAG)/$(TAG)_final.pt \
		--batch-size $${BATCH:-4} \
		$(if $(LIMIT),--limit $(LIMIT),) \
		$(if $(MAXLEN),--max-length $(MAXLEN),) \
		$(if $(TASKS),--tasks $(TASKS),) \
		--strict

# ─── Phase 3.3 ─── knowledge / few-shot benchmarks ──────────────────
#
# Scaffold mode emits schema-valid JSON (mmlu 5-shot, triviaqa 5-shot,
# boolq 0-shot — matching BitNet 2B4T Table 1). Real mode reuses
# intllm.lm_eval_wrapper.build_hflm from §3.2.1.
.PHONY: bench-knowledge
bench-knowledge:
	@cd $(PHASE_D) && PYTHONPATH=. ../../$(PYTHON) scripts/bench_knowledge.py \
		--tag intllm-mini --scaffold
	@cd $(PHASE_D) && PYTHONPATH=. ../../$(PYTHON) scripts/bench_knowledge.py \
		--tag intllm-base --scaffold
	@cd $(PHASE_D) && PYTHONPATH=. ../../$(PYTHON) scripts/bench_knowledge.py \
		--tag intllm-medium --scaffold

# Real knowledge-benchmark run (Phase 3.3 + wrapper from §3.2.1).
# Optional vars: LIMIT / TASKS / BATCH / MAXLEN (same as bench-canonical-real).
.PHONY: bench-knowledge-real
bench-knowledge-real:
	@if [ -z "$(TAG)" ]; then \
		echo "usage: make bench-knowledge-real TAG=mini|base|medium [LIMIT=n] [TASKS=\"...\"]"; \
		exit 2; \
	fi
	@cd $(PHASE_D) && PYTHONPATH=. ../../$(PYTHON) scripts/bench_knowledge.py \
		--tag intllm-$(TAG) \
		--checkpoint checkpoints/$(TAG)/$(TAG)_final.pt \
		--batch-size $${BATCH:-4} \
		$(if $(LIMIT),--limit $(LIMIT),) \
		$(if $(MAXLEN),--max-length $(MAXLEN),) \
		$(if $(TASKS),--tasks $(TASKS),) \
		--strict

# ─── Phase 3.4 ─── baseline re-runs (scaffold + real) ───────────────
#
# Same HFLM plumbing as bench-canonical / bench-knowledge so IntLLM and
# baselines share the identical eval path (§6.9 R3 parity — see
# docs/FJQ_PHASE_D_3_4_B0_FINDINGS.md §3.1).
#
# The 7 baselines registered in scripts/bench_baselines.py::BASELINE_REGISTRY
# and docs/BASELINE_UNPORTED_FEATURES.md §1. If you add/remove a repo here,
# update those two files in the same commit (§7 decision #2).
#
# Real mode needs:
#   - venv with lm_eval 0.4.11 + transformers ≥ 4.54 + torch + CUDA
#   - for hgrn_bit repos, vendored mmfreelm at python/phase_d/_upstream/
#     (recreate via UPSTREAM_PIN.md)
#   - batch_size defaults per repo (2 for 2B+ models, 4 otherwise); override
#     with BATCH=<n>. See BASELINE_REGISTRY for per-repo defaults.
BASELINE_REPOS := \
	microsoft/bitnet-b1.58-2B-4T \
	ridger/MMfreeLM-370M \
	ridger/MMfreeLM-1.3B \
	ridger/MMfreeLM-2.7B \
	HuggingFaceTB/SmolLM2-135M \
	HuggingFaceTB/SmolLM2-360M \
	EleutherAI/pythia-160m

.PHONY: bench-baselines
bench-baselines:
	@for repo in $(BASELINE_REPOS); do \
		cd $(PHASE_D) && PYTHONPATH=_upstream:. ../../$(PYTHON) scripts/bench_baselines.py \
			--repo $$repo --scaffold || exit $$?; \
		cd ../..; \
	done

# Single-repo scaffold convenience: `make bench-baselines-one REPO=ridger/MMfreeLM-370M`
.PHONY: bench-baselines-one
bench-baselines-one:
	@if [ -z "$(REPO)" ]; then \
		echo "usage: make bench-baselines-one REPO=<hf_repo_id>"; \
		echo "registered: $(BASELINE_REPOS)"; exit 2; \
	fi
	@cd $(PHASE_D) && PYTHONPATH=_upstream:. ../../$(PYTHON) scripts/bench_baselines.py \
		--repo $(REPO) --scaffold

# Real baseline run — single repo.
#
# Optional vars (same pattern as bench-canonical-real):
#   LIMIT=<n>       cap docs per task (smoke); <1.0 = fraction, ≥1 = absolute
#   TASKS="<list>"  space-separated subset of the 11 tasks
#   BATCH=<n>       override BASELINE_REGISTRY default_batch_size
#   DEVICE=<d>      torch device (default: cuda)
#
# Examples:
#   make bench-baselines-real REPO=EleutherAI/pythia-160m
#   make bench-baselines-real REPO=microsoft/bitnet-b1.58-2B-4T LIMIT=50 TASKS="piqa boolq"
#   make bench-baselines-real REPO=ridger/MMfreeLM-2.7B BATCH=1
.PHONY: bench-baselines-real
bench-baselines-real:
	@if [ -z "$(REPO)" ]; then \
		echo "usage: make bench-baselines-real REPO=<hf_repo_id> [LIMIT=n] [TASKS=\"...\"] [BATCH=n]"; \
		echo "registered: $(BASELINE_REPOS)"; exit 2; \
	fi
	@cd $(PHASE_D) && PYTHONPATH=_upstream:. ../../$(PYTHON) scripts/bench_baselines.py \
		--repo $(REPO) \
		$(if $(BATCH),--batch-size $(BATCH),) \
		$(if $(LIMIT),--limit $(LIMIT),) \
		$(if $(TASKS),--tasks $(TASKS),) \
		$(if $(DEVICE),--device $(DEVICE),) \
		--strict

# Real sweep across all 7 baselines — ~13 GPU-hours on RTX 4090 Laptop 16 GB
# per FJQ_PHASE_D_3_4_B0_FINDINGS.md §5 GPU budget estimate. Use with care;
# this is the committed Phase 3.4 production run.
.PHONY: bench-baselines-real-all
bench-baselines-real-all:
	@for repo in $(BASELINE_REPOS); do \
		echo "=== $$repo ==="; \
		cd $(PHASE_D) && PYTHONPATH=_upstream:. ../../$(PYTHON) scripts/bench_baselines.py \
			--repo $$repo --strict || exit $$?; \
		cd ../..; \
	done

# ─── Phase E1.4 ─── MinHash LSH near-duplicate dedup ──────────────────
#
# RedPajama / Falcon-Edge recipe: 5-gram word shingles, 128 MinHash
# permutations, Jaccard threshold 0.85, first-seen wins. See
# FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md §E1.4.
#
# `dedup-corpus-id-dryrun` is the §6.8 R2 verification gate. Caps input
# at 10K docs, no shards written, prints stats only. Should finish in
# under a minute and validates: scan/shingle/hash/LSH-insert plumbing
# end-to-end before committing to the multi-hour full run.
#
# `dedup-corpus-id` is the full 10.67M-doc production sweep. Resilience
# layers (atomic per-shard output + .progress journal + --resume-auto)
# land in E1.4.1 once dry-run is validated; do NOT launch the full run
# without first reviewing the dry-run report.
.PHONY: dedup-corpus-id-dryrun
dedup-corpus-id-dryrun:
	@$(PYTHON) $(PHASE_E)/scripts/dedup_corpus.py \
		--source all --max-docs 10000 --dry-run --progress-every 1000

.PHONY: dedup-corpus-id
dedup-corpus-id:
	@$(PYTHON) $(PHASE_E)/scripts/dedup_corpus.py \
		--source all --resume --progress-every 10000

# Smoke gate (V31.E1.4.1): synthetic 11-doc corpus across 2 shards exercises
# state save → kill → resume → cross-run dedup detection → reference-vs-resume
# parity → dry-run side-effect-freeness. ~3s wall-clock; pre-push hook target
# for any change to dedup_corpus.py per CLAUDE §6.8 R3.
.PHONY: test-dedup-resume
test-dedup-resume:
	@$(PYTHON) $(PHASE_E)/scripts/test_dedup_resume.py

# Post-sweep audit (V31.E1.4.2): read-only summarizer of dedup-corpus-id
# state. Reports running/exited, manifest stats per source, disk usage,
# token-budget verification, and a proposed FJQ_PHASE_E_E1_FINDINGS.md
# v1.1 splice block. NEVER modifies files or commits anything — pure
# read + print.
.PHONY: audit-dedup-sweep
audit-dedup-sweep:
	@$(PYTHON) $(PHASE_E)/scripts/audit_dedup_sweep.py

# ─── Phase E1.4.3 ─── Exact-string sha256 dedup ──────────────────────
#
# Drop-in replacement for the LSH variant after it OOM-killed at 5.96M
# / 10M FineWeb-2 ID docs (LSH index hit 30 GB on a 31 GB box,
# 2026-04-26). Empirical dedup_rate from the LSH partial was 0.0093%
# (552 dupes / 5.96M docs) — at this rate, MinHash near-dup is overkill.
# sha256 over normalized text catches >99% at constant ~1 GB memory.
# See `memory/project_phase_e_e1_4_oom_remediation.md` Option A.
#
# `dedup-corpus-id-exact-dryrun` is the §6.8 R2 verification gate. Caps
# input at 10K docs, no shards written. Should finish in well under a
# minute on the ID corpus (≈22 GB raw text across 145+ shards).
#
# `dedup-corpus-id-exact` is the full ~10.67M-doc production sweep.
# Output goes to `data/phase_e/corpus_id_dedup_exact/` so the existing
# 84-shard LSH partial (`corpus_id_dedup/`) stays put for forensics
# until the exact-hash run is verified per the resume protocol.
.PHONY: dedup-corpus-id-exact-dryrun
dedup-corpus-id-exact-dryrun:
	@$(PYTHON) $(PHASE_E)/scripts/dedup_corpus_exact.py \
		--source all --max-docs 10000 --dry-run --progress-every 1000

.PHONY: dedup-corpus-id-exact
dedup-corpus-id-exact:
	@$(PYTHON) $(PHASE_E)/scripts/dedup_corpus_exact.py \
		--source all --resume --progress-every 100000

# Smoke gate (V31.E1.4.3): synthetic 12-doc corpus across 2 shards
# exercises state save → kill → resume → cross-run dedup detection →
# reference-vs-resume parity → dry-run side-effect-freeness → NFKC +
# lowercase + whitespace-collapse normalization. Pre-push hook target
# for any change to dedup_corpus_exact.py per CLAUDE §6.8 R3.
.PHONY: test-dedup-exact
test-dedup-exact:
	@$(PYTHON) $(PHASE_E)/scripts/test_dedup_exact.py

# ─── Phase E1.2 ─── EN corpus shim smoke gate ─────────────────────────
#
# `python/phase_e/intllm_en.py` is a constants + helpers module that
# records the bilingual mix design (ID:EN ratio, repo selection,
# tokenizer rates) without importing torch/transformers/datasets. This
# gate validates the constants stay in sync with the E0 tokenizer
# measurement file + E1.4 manifests, and that the math helpers behave
# correctly across 60:40 / 50:50 / 80:20 ratios. Per §6.8 R3 prevention
# layer for any change to intllm_en.py. Runtime ~1 s.
.PHONY: test-intllm-en
test-intllm-en:
	@$(PYTHON) $(PHASE_E)/scripts/test_intllm_en.py

# ─── Phase E2.1.0 ─── Mini-scale ablation harness (E2.x.3 prerequisite) ─
#
# Per FJQ_PHASE_E_E2_FINDINGS.md v1.0+Q4 closure: the harness produces
# `paper/intllm/ablations/mini_<TAG>.json` artifacts that all 5 E2.x.3
# rows reference and that verify_paper_tables.py asserts against. This
# is a hard prerequisite ahead of any E2.x feature work — without it,
# ablation rows have no JSON to assert.
#
# Currently no E2.x feature is implemented; toggling a feature flag
# emits a warning + records the request in JSON but proceeds with
# baseline behavior. Real wiring lands per E2.x sub-task as features
# ship in intllm.{quant,qat,model}.
#
# Usage:
#   make train-mini-ablation TAG=baseline DRYRUN=1     # fast schema sanity, no GPU
#   make train-mini-ablation TAG=baseline POL=1        # ~3-5 min POL on RTX 4090
#   make train-mini-ablation TAG=baseline              # full ~1 h Mini run
#
# Once E2.4 BilingualCalibrationSampler ships:
#   make train-mini-ablation TAG=balanced_calib FLAGS="--balanced-calib"
.PHONY: train-mini-ablation
train-mini-ablation:
	@if [ -z "$(TAG)" ]; then \
		echo "usage: make train-mini-ablation TAG=<name> [DRYRUN=1] [POL=1] [FLAGS=\"--feat ...\"]"; \
		echo "  standard tags: baseline, hadamard, fp8_lmhead, distill, balanced_calib, lang_cond_{a,b,c}, combined"; \
		exit 2; \
	fi
	@cd $(PHASE_D) && PYTHONPATH=. ../../$(PYTHON) scripts/train_mini_ablation.py \
		--tag $(TAG) \
		$(if $(DRYRUN),--dry-run,) \
		$(if $(POL),--proof-of-life,) \
		$(FLAGS)

# ─── Phase E1.6 ─── Bilingual corpus packaging gate ───────────────────
#
# Asserts `docs/FJQ_PHASE_E_BILINGUAL_CORPUS_V1.md` stays consistent
# with on-disk evidence: ID dedup manifests + EN config shim
# (intllm_en.py) + tokenizer measurement file. 8 invariants:
#   1-2 ID manifests exist + bytes_text_kept ≈ spec
#   3-4 docs_kept exact + dedup_rate ≈ spec
#   5-6 EN cap from shim + Stretch repo selection at 60:40
#   7   synthetic_mix_ratio = 0% (E1.5 deferred per plan v1.7)
#   8   license attribution covers all sources actually used
# Per §6.8 R3 prevention layer for E1.6 packaging. Pre-commit hook
# wiring in follow-up E1.6.1 sub-task.
.PHONY: verify-bilingual-corpus
verify-bilingual-corpus:
	@$(PYTHON) $(PHASE_E)/scripts/verify_bilingual_corpus.py
