#!/usr/bin/env python3
"""F.13.1 live calibration — measure CPU-vs-GPU tok/s on Mini for dispatch decision.

Replaces the projected anchors in `tests/fixtures/f13_dispatch_anchors.toml`
with measured numbers for the host envelope (i9-14900HX + RTX 4090 Laptop).

Per FJQ_PHASE_F_F13_PRODUCTION_PLAN.md — F.13.1 was DEFERRED from the
Z-narrow scope because nvidia driver was absent. Driver restored
2026-04-30; F.13.1 now runs.

Methodology:
  - Single Mini checkpoint (22 M params), greedy decode batch=1
  - Fixed deterministic prompt (24 tokens, same as mini_inference_smoke.py
    for cross-script reproducibility)
  - Warm-up phase (5 tokens, discarded) before timing
  - Timed phase (N=64 tokens for GPU, N=16 for CPU because CPU is slow)
  - Measure wall-clock per-token median + p95 across all timed steps
  - One forward through model() per token; we control the decode loop
    explicitly to avoid HuggingFace generate() KV-cache assumptions
    that may differ between upstream/our wrapper

Output: paper/intllm/results/f13_dispatch_calibration.json
        + console summary

Run:
    cd python/phase_d
    PYTHONPATH=. ../../.venv/bin/python scripts/bench_f13_dispatch_calibration.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # python/phase_d
REPO_ROOT = ROOT.parent.parent  # fajarquant
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from intllm.model import HGRNBitConfig, HGRNBitForCausalLM  # noqa: E402

PROMPT_TOKENS = [
    1, 345, 678, 901, 234, 567, 890, 123, 456, 789, 12,
    34, 56, 78, 90, 11, 22, 33, 44, 55, 66, 77, 88, 99,
]
N_WARMUP = 5
N_GPU = 64
N_CPU = 16  # CPU is ~5x slower; smaller N keeps total wall-clock reasonable

OUT_PATH = REPO_ROOT / "paper" / "intllm" / "results" / "f13_dispatch_calibration.json"


def load_model(device: str):
    ckpt_path = ROOT / "checkpoints" / "mini" / "mini_final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Mini ckpt missing at {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = state["arch"]
    cfg = HGRNBitConfig(
        vocab_size=arch["vocab_size"],
        hidden_size=arch["hidden_size"],
        num_hidden_layers=arch["num_hidden_layers"],
        max_position_embeddings=arch["max_position_embeddings"],
    )
    model = HGRNBitForCausalLM(cfg).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params, arch


def bench_decode(model, prompt_ids, n_warmup: int, n_timed: int, device: str) -> dict[str, Any]:
    """Greedy decode loop with per-token timing.

    Returns: {ms_per_token_median, ms_per_token_p95, ms_per_token_mean,
              tokens_per_sec, n_warmup, n_timed, total_seconds, generated_tokens}
    """
    cur = prompt_ids.to(device)
    per_token_ms: list[float] = []
    generated: list[int] = []

    with torch.no_grad():
        # Warmup — JIT compile, cache warm, GPU clocks settle
        for _ in range(n_warmup):
            out = model(cur)
            next_tok = int(out.logits[0, -1].argmax().item())
            cur = torch.cat([cur, torch.tensor([[next_tok]], device=device)], dim=1)

        # Timed phase
        for _ in range(n_timed):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(cur)
            next_tok = int(out.logits[0, -1].argmax().item())
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            per_token_ms.append((t1 - t0) * 1000.0)
            generated.append(next_tok)
            cur = torch.cat([cur, torch.tensor([[next_tok]], device=device)], dim=1)

    median_ms = statistics.median(per_token_ms)
    mean_ms = statistics.mean(per_token_ms)
    sorted_ms = sorted(per_token_ms)
    p95_idx = max(0, min(len(sorted_ms) - 1, int(0.95 * len(sorted_ms)) - 1))
    p95_ms = sorted_ms[p95_idx]

    return {
        "ms_per_token_median": median_ms,
        "ms_per_token_mean": mean_ms,
        "ms_per_token_p95": p95_ms,
        "tokens_per_sec_median": 1000.0 / median_ms,
        "tokens_per_sec_mean": 1000.0 / mean_ms,
        "n_warmup": n_warmup,
        "n_timed": n_timed,
        "total_timed_seconds": sum(per_token_ms) / 1000.0,
        "generated_tokens": generated[:8],  # sample only — full list is noise
    }


def main() -> int:
    print("F.13.1 live calibration — Mini × batch=1 × greedy decode")
    print(f"  Output: {OUT_PATH.relative_to(REPO_ROOT)}")
    print()

    safe_prompt = [t % 32000 for t in PROMPT_TOKENS]  # safe vocab fallback; will fix below per ckpt
    results: dict[str, Any] = {
        "phase": "F.13.1",
        "scope": "Mini × batch=1 × greedy decode (autoregressive, fresh KV per step)",
        "host": {},
        "model": {},
        "results": {},
    }

    # ─── Host envelope ───
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        results["host"]["gpu"] = {
            "name": gpu_props.name,
            "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
            "total_memory_gb": round(gpu_props.total_memory / 1e9, 2),
            "torch_cuda_version": torch.version.cuda,
        }
    results["host"]["torch_version"] = torch.__version__

    # ─── GPU bench ───
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[1/2] GPU bench on {device}")
        model, n_params, arch = load_model(device)
        # Re-fix prompt with actual vocab size
        safe_prompt = [t % arch["vocab_size"] for t in PROMPT_TOKENS]
        prompt_ids = torch.tensor([safe_prompt])
        results["model"] = {
            "n_params_million": round(n_params / 1e6, 2),
            "hidden_size": arch["hidden_size"],
            "num_hidden_layers": arch["num_hidden_layers"],
            "vocab_size": arch["vocab_size"],
        }
        print(f"      Mini: {n_params/1e6:.2f}M params, hidden={arch['hidden_size']}, L={arch['num_hidden_layers']}")
        print(f"      Warming up ({N_WARMUP} tokens)...")
        gpu_result = bench_decode(model, prompt_ids, N_WARMUP, N_GPU, device)
        results["results"]["gpu"] = gpu_result
        print(f"      GPU: {gpu_result['ms_per_token_median']:.2f} ms/tok median "
              f"({gpu_result['tokens_per_sec_median']:.1f} tok/s) — "
              f"p95 {gpu_result['ms_per_token_p95']:.2f} ms")
        del model
        torch.cuda.empty_cache()
    else:
        print("[1/2] GPU not available, skipping")
        results["results"]["gpu"] = None

    # ─── CPU bench ───
    print(f"\n[2/2] CPU bench on cpu (HGRN may use triton-CPU fallback or pure torch)")
    print(f"      WARNING: HGRN-Bit upstream uses fused-triton kernels — CPU path")
    print(f"      may use pure-torch fallback which is slower than F.11.3 scalar Rust.")
    try:
        device = "cpu"
        model, n_params, arch = load_model(device)
        safe_prompt = [t % arch["vocab_size"] for t in PROMPT_TOKENS]
        prompt_ids = torch.tensor([safe_prompt])
        print(f"      Warming up ({N_WARMUP} tokens)...")
        cpu_result = bench_decode(model, prompt_ids, N_WARMUP, N_CPU, device)
        results["results"]["cpu_torch"] = cpu_result
        print(f"      CPU (torch): {cpu_result['ms_per_token_median']:.2f} ms/tok median "
              f"({cpu_result['tokens_per_sec_median']:.1f} tok/s) — "
              f"p95 {cpu_result['ms_per_token_p95']:.2f} ms")
    except Exception as e:
        print(f"      CPU bench FAILED: {type(e).__name__}: {e}")
        results["results"]["cpu_torch"] = {"error": f"{type(e).__name__}: {e}"}

    # ─── Save ───
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_PATH.relative_to(REPO_ROOT)}")

    # ─── Summary line ───
    if results["results"].get("gpu") and results["results"].get("cpu_torch") and "error" not in results["results"]["cpu_torch"]:
        gpu_ts = results["results"]["gpu"]["tokens_per_sec_median"]
        cpu_ts = results["results"]["cpu_torch"]["tokens_per_sec_median"]
        ratio = cpu_ts / gpu_ts if gpu_ts > 0 else float('inf')
        print()
        print("F.13.1 verdict cross-check:")
        print(f"  Mini × batch=1: GPU = {gpu_ts:.1f} tok/s, CPU-torch = {cpu_ts:.1f} tok/s")
        print(f"  CPU/GPU ratio = {ratio:.2f} ({'CPU wins' if ratio > 1.0 else 'GPU wins'})")
        print(f"  F.13 anchor [gpu_small_model_batch1_decode] = 80-120 tok/s "
              f"({'in band' if 80 <= gpu_ts <= 120 else 'OUT OF BAND — verdict needs re-derivation'})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
