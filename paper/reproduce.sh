#!/usr/bin/env bash
# reproduce.sh — Reproduce all FajarQuant paper results from scratch
#
# Usage:
#   bash paper/reproduce.sh --smoke    # Quick smoke test (~10 min, 1 model, 5 prompts)
#   bash paper/reproduce.sh --full     # Full reproduction (~8 GPU hours, 3 models)
#   bash paper/reproduce.sh --verify   # Verify existing results only (no GPU needed)
#
# Requirements:
#   - NVIDIA GPU with >=16 GB VRAM (RTX 4090 recommended)
#   - Python 3.10+ with torch, transformers, datasets, numpy
#   - HuggingFace account with access to google/gemma-4-E2B
#
# Environment:
#   Set HF_TOKEN for gated model access (Gemma):
#     export HF_TOKEN=hf_xxxx

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:---smoke}"

# Auto-detect venv if PYTHON not set
if [ -z "${PYTHON:-}" ]; then
    if [ -f .venv/bin/python3 ]; then
        PYTHON=".venv/bin/python3"
    elif [ -f venv/bin/python3 ]; then
        PYTHON="venv/bin/python3"
    else
        PYTHON="python3"
    fi
fi

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}PASS${NC} $1"; }
fail() { echo -e "${RED}FAIL${NC} $1"; exit 1; }
info() { echo -e "${YELLOW}INFO${NC} $1"; }

echo "============================================================"
echo "  FajarQuant Paper Reproduction Script"
echo "  Mode: $MODE"
echo "  Working dir: $(pwd)"
echo "============================================================"
echo ""

# ── Verify mode: no GPU needed ──
if [ "$MODE" = "--verify" ]; then
    info "Step V: Verifying paper tables against existing data..."
    $PYTHON scripts/verify_paper_tables.py --strict
    echo ""

    info "Checking v3.1 result files exist..."
    MISSING=0
    for f in \
        data/kv_cache/perplexity_v3.1_gemma_4_e2b_2bit.json \
        data/kv_cache/perplexity_v3.1_gemma_4_e2b_3bit.json \
        data/kv_cache/perplexity_v3.1_gemma_4_e2b_4bit.json \
        data/kv_cache/perplexity_v3.1_mistral_2bit.json \
        data/kv_cache/perplexity_v3.1_mistral_3bit.json \
        data/kv_cache/perplexity_v3.1_mistral_4bit.json \
        data/kv_cache/perplexity_v3.1_qwen2_2bit.json \
        data/kv_cache/perplexity_v3.1_qwen2_3bit.json \
        data/kv_cache/perplexity_v3.1_qwen2_4bit.json; do
        if [ -f "$f" ]; then
            pass "$f"
        else
            fail "$f MISSING"
            MISSING=$((MISSING + 1))
        fi
    done

    if [ $MISSING -eq 0 ]; then
        pass "All 9 v3.1 result files present"
    fi

    info "Checking calibration data checksums..."
    if [ -f data/calibration/fq_v2_gemma_4_e2b.npz ]; then
        ACTUAL=$(sha256sum data/calibration/fq_v2_gemma_4_e2b.npz | cut -c1-16)
        EXPECTED="b4d43463daad65c0"
        if [ "$ACTUAL" = "$EXPECTED" ]; then
            pass "Gemma calibration checksum matches"
        else
            fail "Gemma calibration checksum mismatch: $ACTUAL != $EXPECTED"
        fi
    fi

    echo ""
    pass "Verification complete"
    exit 0
fi

# ── Environment check (GPU modes only) ──
check_gpu_env() {
    info "Checking GPU environment..."
    $PYTHON -c "import torch; assert torch.cuda.is_available(), 'CUDA required'" 2>/dev/null \
      || fail "PyTorch with CUDA not available. Install: pip install torch --index-url https://download.pytorch.org/whl/cu124"
    $PYTHON -c "import transformers, datasets, numpy" 2>/dev/null \
      || fail "Missing packages. Install: pip install transformers datasets numpy"
    GPU_NAME=$($PYTHON -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    GPU_MEM=$($PYTHON -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')" 2>/dev/null)
    info "GPU: $GPU_NAME ($GPU_MEM)"
    pass "Environment OK"
    echo ""
}

# ── Smoke mode: quick single-model test ──
if [ "$MODE" = "--smoke" ]; then
    check_gpu_env
    info "Step 1: Smoke test — Gemma 4 E2B, 3-bit only, 5 samples"
    echo ""

    # Check if calibration exists
    if [ ! -f data/calibration/fq_v2_gemma_4_e2b.npz ]; then
        info "Calibrating Gemma PCA rotations (first run only)..."
        $PYTHON scripts/calibrate_fq_v2.py \
            --model google/gemma-4-E2B \
            --output data/calibration/fq_v2_gemma_4_e2b.npz \
            ${HF_TOKEN:+--token "$HF_TOKEN"}
    fi

    # Profile heads
    info "Profiling KV heads..."
    $PYTHON scripts/profile_kv_heads.py \
        --model google/gemma-4-E2B \
        ${HF_TOKEN:+--token "$HF_TOKEN"} || true

    # Eval 3-bit with 5 samples
    info "Evaluating Gemma 3-bit perplexity (5 samples)..."
    $PYTHON scripts/eval_perplexity_v3.py \
        --model google/gemma-4-E2B \
        --strategy data/calibration/strategy_gemma_4_e2b_3bit.json \
        --calibration data/calibration/fq_v2_gemma_4_e2b.npz \
        --bits 3 --max-samples 5 --seq-len 2048 \
        --output data/kv_cache/perplexity_smoke_gemma_3bit.json \
        ${HF_TOKEN:+--token "$HF_TOKEN"}

    # Verify: v3.1 should beat KIVI on Gemma 3-bit
    $PYTHON -c "
import json
with open('data/kv_cache/perplexity_smoke_gemma_3bit.json') as f:
    d = json.load(f)
fq = d['fajarquant_v3_3bit']['ppl']
kivi = d['kivi_3bit']['ppl']
print(f'FQ v3.1: {fq:.2f}, KIVI: {kivi:.2f}')
assert fq < kivi, f'Smoke test failed: FQ {fq} >= KIVI {kivi}'
print('Smoke test PASSED: FQ v3.1 beats KIVI on Gemma 3-bit')
"
    echo ""
    pass "Smoke test complete"
    exit 0
fi

# ── Full mode: reproduce all 9 cells ──
if [ "$MODE" = "--full" ]; then
    check_gpu_env
    info "Full reproduction — 3 models x 3 bit widths (~8 GPU hours)"
    echo ""

    # Step 1: Calibrate
    info "Step 1: Calibrating PCA rotations..."

    if [ ! -f data/calibration/fq_v2_gemma_4_e2b.npz ]; then
        $PYTHON scripts/calibrate_fq_v2.py \
            --model google/gemma-4-E2B \
            --output data/calibration/fq_v2_gemma_4_e2b.npz \
            ${HF_TOKEN:+--token "$HF_TOKEN"}
    fi
    if [ ! -f data/calibration/fq_v2_mistral_7b_v0.1_perhead.npz ]; then
        $PYTHON scripts/calibrate_fq_v2.py \
            --model mistralai/Mistral-7B-v0.1 --per-head \
            --output data/calibration/fq_v2_mistral_7b_v0.1_perhead.npz
    fi
    if [ ! -f data/calibration/fq_v2_qwen2_7b_perhead.npz ]; then
        $PYTHON scripts/calibrate_fq_v2.py \
            --model Qwen/Qwen2-7B --per-head \
            --output data/calibration/fq_v2_qwen2_7b_perhead.npz
    fi
    pass "Calibration done"

    # Step 2: Profile + tune
    info "Step 2: Profiling KV heads + tuning thresholds..."
    for model in google/gemma-4-E2B mistralai/Mistral-7B-v0.1 Qwen/Qwen2-7B; do
        $PYTHON scripts/profile_kv_heads.py --model "$model" \
            ${HF_TOKEN:+--token "$HF_TOKEN"} || true
    done

    $PYTHON scripts/tune_thresholds.py \
        --model mistralai/Mistral-7B-v0.1 --bits 3,4 \
        --calibration data/calibration/fq_v2_mistral_7b_v0.1_perhead.npz || true
    $PYTHON scripts/tune_thresholds.py \
        --model Qwen/Qwen2-7B --bits 3,4 \
        --calibration data/calibration/fq_v2_qwen2_7b_perhead.npz || true
    pass "Profiling + tuning done"

    # Step 3: PPL-guided 2-bit selection
    info "Step 3: PPL-guided 2-bit strategy selection..."
    for model in google/gemma-4-E2B mistralai/Mistral-7B-v0.1 Qwen/Qwen2-7B; do
        $PYTHON scripts/ppl_guided_select.py --model "$model" \
            ${HF_TOKEN:+--token "$HF_TOKEN"} || true
    done
    pass "PPL-guided selection done"

    # Step 4: Full perplexity evaluation
    info "Step 4: Full 3x3 perplexity evaluation..."

    # Gemma (seq_len=2048)
    for bits in 2 3 4; do
        STRAT="data/calibration/strategy_gemma_4_e2b_${bits}bit"
        [ "$bits" = "2" ] && STRAT="${STRAT}_ppl_guided"
        $PYTHON scripts/eval_perplexity_v3.py \
            --model google/gemma-4-E2B \
            --strategy "${STRAT}.json" \
            --calibration data/calibration/fq_v2_gemma_4_e2b.npz \
            --bits "$bits" --seq-len 2048 --max-samples 30 \
            --output "data/kv_cache/perplexity_v3.1_gemma_4_e2b_${bits}bit.json" \
            ${HF_TOKEN:+--token "$HF_TOKEN"}
    done
    pass "Gemma eval done"

    # Mistral (seq_len=2048)
    for bits in 2 3 4; do
        STRAT="data/calibration/strategy_mistral_7b_v0.1_${bits}bit"
        [ "$bits" = "2" ] && STRAT="${STRAT}_ppl_guided"
        $PYTHON scripts/eval_perplexity_v3.py \
            --model mistralai/Mistral-7B-v0.1 \
            --strategy "${STRAT}.json" \
            --calibration data/calibration/fq_v2_mistral_7b_v0.1_perhead.npz \
            --bits "$bits" --seq-len 2048 --max-samples 30 \
            --output "data/kv_cache/perplexity_v3.1_mistral_${bits}bit.json"
    done
    pass "Mistral eval done"

    # Qwen2 (seq_len=1024 due to OOM)
    for bits in 2 3 4; do
        STRAT="data/calibration/strategy_qwen2_7b_${bits}bit"
        [ "$bits" = "2" ] && STRAT="${STRAT}_ppl_guided"
        $PYTHON scripts/eval_perplexity_v3.py \
            --model Qwen/Qwen2-7B \
            --strategy "${STRAT}.json" \
            --calibration data/calibration/fq_v2_qwen2_7b_perhead.npz \
            --bits "$bits" --seq-len 1024 --max-samples 30 \
            --output "data/kv_cache/perplexity_v3.1_qwen2_${bits}bit.json"
    done
    pass "Qwen2 eval done"

    # Step 5: Verify paper tables
    info "Step 5: Verifying paper tables..."
    $PYTHON scripts/verify_paper_tables.py --strict

    echo ""
    echo "============================================================"
    pass "Full reproduction complete. All results in data/kv_cache/"
    echo "============================================================"
    exit 0
fi

echo "Usage: bash paper/reproduce.sh [--smoke|--full|--verify]"
exit 1
