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
	@echo "  bench-baselines            Phase 3.4  — TODO (BitNet, MMfreeLM, SmolLM2)"

PYTHON := .venv/bin/python
PHASE_D := python/phase_d
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

# ─── Phase 3.4 ─── baseline re-runs (scaffold follow-up) ────────────
.PHONY: bench-baselines
bench-baselines:
	@echo "[TODO] Phase 3.4 — baseline re-runs not yet scaffolded."
	@echo "       See FJQ_PHASE_D_PRODUCTION_PLAN.md §3.4 for spec."
	@exit 1
