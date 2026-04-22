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
	@echo "  test-phase-d               Phase 4    — pytest tests/ in phase_d (56)"
	@echo "  fp16-parity-mini           Phase 3.5  — measure + emit JSON for Mini"
	@echo "  fp16-parity-base           Phase 3.5  — measure + emit JSON for Base"
	@echo ""
	@echo "  bench-canonical            Phase 3.2  — TODO (lm-eval 8 zero-shot)"
	@echo "  bench-knowledge            Phase 3.3  — TODO (mmlu 5-shot etc.)"
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

# ─── Phase 3.2/3.3/3.4 ─── canonical / knowledge / baseline ─────────
.PHONY: bench-canonical bench-knowledge bench-baselines
bench-canonical bench-knowledge bench-baselines:
	@echo "[TODO] Phase 3.2/3.3/3.4 — not yet scaffolded."
	@echo "       See FJQ_PHASE_D_PRODUCTION_PLAN.md §3 for spec."
	@exit 1
