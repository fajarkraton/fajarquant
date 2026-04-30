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


def bench_decode_naive(model, prompt_ids, n_warmup: int, n_timed: int, device: str) -> dict[str, Any]:
    """Naive greedy decode loop — full forward on growing sequence each step.

    No KV cache, no compile. This is the worst-case stack and the baseline
    for showing how much optimization removes overhead.
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


def bench_decode_kvcache(model, prompt_ids, n_warmup: int, n_timed: int, device: str) -> dict[str, Any]:
    """KV-cached greedy decode — single-token forward each step using HGRN-Bit
    `RecurrentCache` (`use_cache=True` + `past_key_values`).

    This eliminates the dominant cost in bench_decode_naive: re-processing
    the full growing sequence each step. With cache, each step processes
    ONE token while reusing the recurrent state from the prefill.
    """
    per_token_ms: list[float] = []
    generated: list[int] = []

    with torch.no_grad():
        # Prefill — process full prompt once, store cache.
        prompt = prompt_ids.to(device)
        out = model(input_ids=prompt, use_cache=True)
        past = out.past_key_values
        next_tok = int(out.logits[0, -1].argmax().item())

        # Warmup — single-token decode steps until cache stabilizes
        for _ in range(n_warmup):
            single = torch.tensor([[next_tok]], device=device)
            out = model(input_ids=single, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = int(out.logits[0, -1].argmax().item())

        # Timed phase
        for _ in range(n_timed):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            single = torch.tensor([[next_tok]], device=device)
            out = model(input_ids=single, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = int(out.logits[0, -1].argmax().item())
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            per_token_ms.append((t1 - t0) * 1000.0)
            generated.append(next_tok)

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
        "generated_tokens": generated[:8],
    }


def bench_decode_kvcache_compiled(model, prompt_ids, n_warmup: int, n_timed: int, device: str, mode: str = "default") -> dict[str, Any]:
    """KV-cached + torch.compile greedy decode.

    Two compile modes attempted in fallback chain:
      - "reduce-overhead" (CUDA graphs) — fastest but needs explicit
        cudagraph_mark_step_begin() between recurrent-state mutations.
        HGRN-Bit's `init_state` does in-place `state += (...)` which
        breaks CUDAGraphs unless we add the marker between steps.
      - "default" (eager-with-fusion) — slower but compatible with
        in-place tensor mutation in the model code.

    Caller passes `mode` to select; main() invokes both as separate
    bench keys so we see whether CUDAGraphs work and what the gap is.
    """
    try:
        compiled_model = torch.compile(model, mode=mode, dynamic=False)
    except Exception as e:
        return {"error": f"torch.compile(mode={mode}) failed: {type(e).__name__}: {e}"}

    per_token_ms: list[float] = []
    generated: list[int] = []

    use_cudagraph_marker = (mode == "reduce-overhead")

    with torch.no_grad():
        try:
            # Prefill (with the compiled wrapper)
            if use_cudagraph_marker:
                torch.compiler.cudagraph_mark_step_begin()
            prompt = prompt_ids.to(device)
            out = compiled_model(input_ids=prompt, use_cache=True)
            past = out.past_key_values
            next_tok = int(out.logits[0, -1].argmax().item())
        except Exception as e:
            return {"error": f"compiled(mode={mode}) prefill failed: {type(e).__name__}: {e}"}

        # Extra warmup for torch.compile — first few invocations trigger
        # graph capture; need stable timing post-capture.
        compile_warmup = max(n_warmup, 10)
        try:
            for _ in range(compile_warmup):
                if use_cudagraph_marker:
                    torch.compiler.cudagraph_mark_step_begin()
                single = torch.tensor([[next_tok]], device=device)
                out = compiled_model(input_ids=single, past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_tok = int(out.logits[0, -1].argmax().item())
        except Exception as e:
            return {"error": f"compiled(mode={mode}) warmup failed: {type(e).__name__}: {e}"}

        # Timed phase
        for _ in range(n_timed):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if use_cudagraph_marker:
                torch.compiler.cudagraph_mark_step_begin()
            single = torch.tensor([[next_tok]], device=device)
            out = compiled_model(input_ids=single, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = int(out.logits[0, -1].argmax().item())
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            per_token_ms.append((t1 - t0) * 1000.0)
            generated.append(next_tok)

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
        "n_warmup": compile_warmup,
        "n_timed": n_timed,
        "total_timed_seconds": sum(per_token_ms) / 1000.0,
        "generated_tokens": generated[:8],
    }


def main() -> int:
    print("F.13.1 live calibration — Mini × batch=1 × greedy decode")
    print(f"  Output: {OUT_PATH.relative_to(REPO_ROOT)}")
    print()

    safe_prompt = [t % 32000 for t in PROMPT_TOKENS]  # safe vocab fallback; will fix below per ckpt
    results: dict[str, Any] = {
        "phase": "F.13.1 v2 (kvcache + torch.compile)",
        "scope": "Mini × batch=1 × greedy decode (4 paths: naive / kvcache / compile(default) / compile(reduce-overhead))",
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

    # ─── GPU bench (3 paths: naive / kvcache / kvcache+compiled) ───
    if torch.cuda.is_available():
        device = "cuda"
        # Pre-warmup: stabilize GPU clocks via dummy ops. Laptop RTX power
        # management can leave initial measurements 6× slower than steady state
        # (observed F.13.1 v1: 19.7 vs v2: 120.2 tok/s on identical naive path
        # — sole difference was prior session warmth). Burn ~50 small matmuls
        # to push P-state to boost.
        print(f"[0/4] GPU clock stabilization (50 dummy matmuls)...")
        a = torch.randn(512, 512, device=device)
        b = torch.randn(512, 512, device=device)
        for _ in range(50):
            c = a @ b
        torch.cuda.synchronize()
        del a, b, c

        # Path 1: naive (no KV cache, no compile) — original F.13.1 baseline
        print(f"[1/4] GPU bench — naive (no KV cache, no compile)")
        model, n_params, arch = load_model(device)
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
        gpu_result = bench_decode_naive(model, prompt_ids, N_WARMUP, N_GPU, device)
        results["results"]["gpu_naive"] = gpu_result
        # Backwards-compat key for older readers / fixture references:
        results["results"]["gpu"] = gpu_result
        print(f"      GPU naive: {gpu_result['ms_per_token_median']:.2f} ms/tok median "
              f"({gpu_result['tokens_per_sec_median']:.1f} tok/s) — "
              f"p95 {gpu_result['ms_per_token_p95']:.2f} ms")

        # Path 2: KV-cached (HGRN-Bit RecurrentCache, single-token forward per step)
        print(f"\n[2/4] GPU bench — kvcache (HGRN-Bit RecurrentCache)")
        try:
            kvcache_result = bench_decode_kvcache(model, prompt_ids, N_WARMUP, N_GPU, device)
            results["results"]["gpu_kvcache"] = kvcache_result
            print(f"      GPU kvcache: {kvcache_result['ms_per_token_median']:.2f} ms/tok median "
                  f"({kvcache_result['tokens_per_sec_median']:.1f} tok/s) — "
                  f"p95 {kvcache_result['ms_per_token_p95']:.2f} ms")
            speedup = kvcache_result['tokens_per_sec_median'] / gpu_result['tokens_per_sec_median']
            print(f"      Speedup vs naive: {speedup:.2f}x")
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"      KV-cache bench FAILED: {err}")
            results["results"]["gpu_kvcache"] = {"error": err}

        # Path 3a: KV-cached + torch.compile(default) — eager-with-fusion,
        #          compatible with in-place state mutation in HGRN-Bit
        print(f"\n[3a/4] GPU bench — kvcache + torch.compile(default)")
        compiled_default = bench_decode_kvcache_compiled(model, prompt_ids, N_WARMUP, N_GPU, device, mode="default")
        results["results"]["gpu_kvcache_compiled_default"] = compiled_default
        if "error" in compiled_default:
            print(f"      Compiled(default) bench FAILED: {compiled_default['error']}")
        else:
            print(f"      GPU kvcache+compiled(default): {compiled_default['ms_per_token_median']:.2f} ms/tok median "
                  f"({compiled_default['tokens_per_sec_median']:.1f} tok/s) — "
                  f"p95 {compiled_default['ms_per_token_p95']:.2f} ms")
            speedup = compiled_default['tokens_per_sec_median'] / gpu_result['tokens_per_sec_median']
            print(f"      Speedup vs naive: {speedup:.2f}x")

        # Path 3b: KV-cached + torch.compile(reduce-overhead) — CUDA graphs.
        #          Now with cudagraph_mark_step_begin() to handle HGRN's
        #          in-place recurrent state. Best-case if it works.
        print(f"\n[3b/4] GPU bench — kvcache + torch.compile(reduce-overhead) + cudagraph marker")
        compiled_reduce = bench_decode_kvcache_compiled(model, prompt_ids, N_WARMUP, N_GPU, device, mode="reduce-overhead")
        results["results"]["gpu_kvcache_compiled_reduce_overhead"] = compiled_reduce
        # Backwards-compat key: keep the old name pointing to whichever compiled
        # variant succeeded (prefer reduce-overhead if both work).
        if "error" not in compiled_reduce:
            results["results"]["gpu_kvcache_compiled"] = compiled_reduce
        elif "error" not in compiled_default:
            results["results"]["gpu_kvcache_compiled"] = compiled_default
        else:
            results["results"]["gpu_kvcache_compiled"] = compiled_reduce  # error path
        if "error" in compiled_reduce:
            print(f"      Compiled(reduce-overhead) bench FAILED: {compiled_reduce['error']}")
        else:
            print(f"      GPU kvcache+compiled(reduce-overhead): {compiled_reduce['ms_per_token_median']:.2f} ms/tok median "
                  f"({compiled_reduce['tokens_per_sec_median']:.1f} tok/s) — "
                  f"p95 {compiled_reduce['ms_per_token_p95']:.2f} ms")
            speedup = compiled_reduce['tokens_per_sec_median'] / gpu_result['tokens_per_sec_median']
            print(f"      Speedup vs naive: {speedup:.2f}x")

        del model
        torch.cuda.empty_cache()
    else:
        print("[1/4] GPU not available, skipping all 3 GPU paths")
        results["results"]["gpu_naive"] = None
        results["results"]["gpu_kvcache"] = None
        results["results"]["gpu_kvcache_compiled"] = None

    # ─── CPU bench ───
    print(f"\n[4/4] CPU bench on cpu (HGRN may use triton-CPU fallback or pure torch)")
    print(f"      WARNING: HGRN-Bit upstream uses fused-triton kernels — CPU path")
    print(f"      may use pure-torch fallback which is slower than F.11.3 scalar Rust.")
    try:
        device = "cpu"
        model, n_params, arch = load_model(device)
        safe_prompt = [t % arch["vocab_size"] for t in PROMPT_TOKENS]
        prompt_ids = torch.tensor([safe_prompt])
        print(f"      Warming up ({N_WARMUP} tokens)...")
        cpu_result = bench_decode_naive(model, prompt_ids, N_WARMUP, N_CPU, device)
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

    # ─── Comparison table + verdict cross-check ───
    print()
    print("F.13.1 v2 path comparison:")
    print(f"  {'Path':<32} {'tok/s':>8} {'ms/tok':>8} {'speedup':>8}")
    print(f"  {'-'*32} {'-'*8} {'-'*8} {'-'*8}")
    naive = results["results"].get("gpu_naive")
    if naive and "error" not in naive:
        baseline_ts = naive["tokens_per_sec_median"]
        print(f"  {'GPU naive (no cache, no compile)':<40} {baseline_ts:>8.1f} {naive['ms_per_token_median']:>8.2f} {1.00:>7.2f}x")
        for key, label in [
            ("gpu_kvcache", "GPU kvcache"),
            ("gpu_kvcache_compiled_default", "GPU kvcache + compile(default)"),
            ("gpu_kvcache_compiled_reduce_overhead", "GPU kvcache + compile(reduce-overhead)"),
        ]:
            r = results["results"].get(key)
            if r and "error" not in r:
                ts = r["tokens_per_sec_median"]
                print(f"  {label:<40} {ts:>8.1f} {r['ms_per_token_median']:>8.2f} {ts/baseline_ts:>7.2f}x")
            elif r and "error" in r:
                print(f"  {label:<40} {'FAIL':>8} {'-':>8} {'-':>8}")

    # Anchor cross-check (use BEST measured GPU as the anchor candidate)
    candidates = []
    for k in ("gpu_kvcache_compiled_reduce_overhead", "gpu_kvcache_compiled_default",
              "gpu_kvcache", "gpu_naive"):
        r = results["results"].get(k)
        if r and "error" not in r:
            candidates.append((k, r["tokens_per_sec_median"]))
    if candidates:
        best_path, best_ts = max(candidates, key=lambda x: x[1])
        print()
        print(f"F.13 anchor cross-check (best path = {best_path}):")
        print(f"  Best GPU tok/s = {best_ts:.1f}")
        print(f"  Anchor [gpu_small_model_batch1_decode] = 80-120 tok/s")
        if 80 <= best_ts <= 120:
            print(f"  → IN BAND — anchor validated by measurement, verdict G3 stable")
        elif best_ts < 80:
            print(f"  → BELOW BAND ({best_ts:.1f} < 80) — verdict G3 reinforced "
                  f"(GPU even less competitive than projection)")
        else:
            print(f"  → ABOVE BAND ({best_ts:.1f} > 120) — verdict G3 may need "
                  f"re-derivation if margin to CPU-TL2 lower bound (200) drops below 50 tok/s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
