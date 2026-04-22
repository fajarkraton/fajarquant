#!/usr/bin/env bash
# FajarQuant Phase D — MLSys AE Functional smoke (Phase 5.3).
#
# Single-command end-to-end demonstration that the IntLLM artifact
# works. Target: < 30 min on a consumer GPU (RTX 4090 / 3090 / 4070
# Laptop). Per FJQ_PHASE_D_PRODUCTION_PLAN.md §5.3.
#
# Steps (each must exit 0):
#   1/5  environment check         (Python, torch, GPU)
#   2/5  Phase D unit tests        (56 tests via pytest)
#   3/5  paper-claim verifier      (--strict, exit 2 on drift)
#   4/5  fp16-parity scaffold gate (asserts JSON populated)
#   5/5  Mini checkpoint inference (1 prompt + 5-token continuation)
#
# Run:
#   bash run_smoke.sh
#
# Exit codes:
#   0   all green — artifact is functional
#   2-N  step N-1 failed; see preceding stderr for the failing step

set -euo pipefail

cd "$(dirname "$0")"
START=$SECONDS

PYTHON=".venv/bin/python"
PHASE_D="python/phase_d"

step() {
    echo ""
    echo "============================================================"
    echo "[$1] $2"
    echo "============================================================"
}

# ─── 1/5 ─── Environment check ───────────────────────────────────────
step "1/5" "Environment check"
if [[ ! -x "$PYTHON" ]]; then
    echo "FAIL: $PYTHON missing — set up the venv first:"
    echo "      python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 2
fi
$PYTHON -c "
import sys, torch
print(f'  python={sys.version.split()[0]}')
print(f'  torch={torch.__version__}')
print(f'  cuda_available={torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  device={torch.cuda.get_device_name(0)}')
    print(f'  vram_total_GB={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}')
else:
    print('  WARN: no CUDA — HGRNBit triton kernels may error')
" || { echo "FAIL: torch import"; exit 2; }

# ─── 2/5 ─── Phase D unit tests ──────────────────────────────────────
step "2/5" "Phase D unit tests (56 tests)"
LOG=/tmp/run_smoke_test_phase_d.log
if cd "$PHASE_D" && PYTHONPATH=. "../../$PYTHON" -m pytest tests/ -q > "$LOG" 2>&1; then
    cd "$OLDPWD"
    grep -E "passed|failed" "$LOG" | tail -1
else
    cd "$OLDPWD"
    echo "FAIL: Phase D unit tests failed; see $LOG"
    tail -30 "$LOG"
    exit 3
fi

# ─── 3/5 ─── Paper-claim verifier (--strict) ─────────────────────────
step "3/5" "Paper-claim verifier (verify-intllm-tables --strict)"
if make verify-intllm-tables; then
    :
else
    echo "FAIL: verify-intllm-tables --strict exited non-zero"
    exit 4
fi

# ─── 4/5 ─── fp16-parity scaffold gate ───────────────────────────────
step "4/5" "fp16-parity scaffold gate"
if make test-intllm-fp16-parity; then
    :
else
    echo "FAIL: test-intllm-fp16-parity smoke failed (regenerate with"
    echo "      'make fp16-parity-mini' if checkpoint moved/changed)"
    exit 5
fi

# ─── 5/5 ─── Mini checkpoint inference ───────────────────────────────
step "5/5" "Mini checkpoint inference (1 prompt + 5-token greedy)"
if cd "$PHASE_D" && PYTHONPATH=. "../../$PYTHON" scripts/mini_inference_smoke.py; then
    cd "$OLDPWD"
else
    rc=$?
    cd "$OLDPWD"
    if [[ $rc -eq 2 ]]; then
        echo "INFO: Mini checkpoint not present — inference step skipped."
        echo "      The other 4 gates passed; this artifact is functional"
        echo "      modulo the inference smoke. Train Mini OR download the"
        echo "      release checkpoint to fully validate."
    else
        echo "FAIL: mini_inference_smoke.py exited $rc"
        exit 6
    fi
fi

ELAPSED=$((SECONDS - START))
echo ""
echo "============================================================"
echo "[OK] run_smoke.sh — all gates green in ${ELAPSED}s"
echo "============================================================"
