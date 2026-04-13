#!/usr/bin/env python3
"""
ppl_guided_select.py — PPL-guided strategy selection for 2-bit quantization.

At 2-bit, MSE-based tuning picks wrong strategies because MSE doesn't
correlate well with PPL (rotation errors compound through multi-layer
attention). This script tests candidate strategies on a small PPL sample
(5 chunks) and picks the lowest PPL.

Usage:
    python scripts/ppl_guided_select.py \
        --model google/gemma-4-E2B \
        --calibration data/calibration/fq_v2_gemma_4_e2b.npz \
        --mse-strategy data/calibration/strategy_gemma_4_e2b_2bit.json \
        --bits 2 --n-chunks 5

Decision provenance: V26 FajarQuant v3.1, Phase B-fix2.4.
Root cause: MSE ≠ PPL at 2-bit (v3 C9 NO-GO finding).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_attention_v3 import (  # noqa: E402
    load_v3_config,
    patch_model_for_quantization_v3,
    unpatch_model,
)
from quant_attention import (  # noqa: E402
    patch_model_for_quantization,
    unpatch_model as unpatch_v1,
)


def evaluate_ppl_quick(model, tokenizer, text, seq_len=2048, n_chunks=5):
    """Quick PPL evaluation on n_chunks (default 5) for strategy comparison."""
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        from transformers import DynamicCache

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    total_len = input_ids.size(1)
    max_chunks = total_len // seq_len
    if n_chunks > max_chunks:
        n_chunks = max_chunks

    nlls, n_tokens = [], 0
    for i in range(n_chunks):
        chunk = input_ids[:, i * seq_len : (i + 1) * seq_len]
        with torch.no_grad():
            cache = DynamicCache()
            out = model(chunk, past_key_values=cache, use_cache=True)
        shift_logits = out.logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        nlls.append(loss.item())
        n_tokens += shift_labels.numel()
        del cache
        torch.cuda.empty_cache()

    ppl = float(torch.exp(torch.tensor(sum(nlls) / n_tokens)))
    return ppl, n_tokens, n_chunks


def make_uniform_strategy(n_layers: int, n_heads: int, path: str, bits: int, model_name: str) -> dict:
    """Create a strategy where all heads use the same path."""
    assignments = []
    for _ in range(n_layers):
        layer = [{"k": path, "v": path} for _ in range(n_heads)]
        assignments.append(layer)
    return {
        "_model": model_name,
        "_bits": bits,
        "_method": f"uniform_all_{path}",
        "assignments": assignments,
    }


def main():
    parser = argparse.ArgumentParser(
        description="PPL-guided strategy selection for 2-bit quantization"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration", default=None, help="v2 calibration .npz (for Path B)")
    parser.add_argument("--mse-strategy", default=None, help="MSE-tuned strategy JSON (optional candidate)")
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--n-chunks", type=int, default=5, help="Chunks for quick PPL (default: 5)")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--token", default=None)
    parser.add_argument("--output", default=None, help="Output strategy JSON (default: auto)")
    args = parser.parse_args()

    model_short = args.model.split("/")[-1].lower().replace("-", "_")
    if args.output is None:
        args.output = f"data/calibration/strategy_{model_short}_{args.bits}bit_ppl_guided.json"

    print(f"PPL-guided strategy selection")
    print(f"  Model: {args.model}")
    print(f"  Bits: {args.bits}")
    print(f"  Quick eval: {args.n_chunks} chunks × {args.seq_len} tokens")
    print(f"  Output: {args.output}")

    # Load dataset
    print("\nLoading WikiText-2 test set...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, token=args.token, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, token=args.token, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()

    # Detect model dimensions from attention layers
    n_layers = 0
    n_kv_heads = 0
    for module in model.modules():
        cls_name = type(module).__name__
        if cls_name in ("MistralAttention", "Qwen2Attention", "Gemma4TextAttention"):
            n_layers += 1
            if hasattr(module, "num_key_value_heads"):
                n_kv_heads = module.num_key_value_heads
    if n_kv_heads == 0:
        n_kv_heads = 1
    print(f"  Detected: {n_layers} layers, {n_kv_heads} KV heads")

    # Build candidate strategies
    candidates: list[tuple[str, dict]] = []

    # Candidate 1: all-A (KIVI)
    candidates.append(("all-A (KIVI)", make_uniform_strategy(
        n_layers, n_kv_heads, "A", args.bits, args.model)))

    # Candidate 2: all-C (Hadamard+outlier / TQ-style)
    candidates.append(("all-C (TQ outlier)", make_uniform_strategy(
        n_layers, n_kv_heads, "C", args.bits, args.model)))

    # Candidate 3: all-B (PCA rotation) — only if calibration available
    if args.calibration and os.path.exists(args.calibration):
        candidates.append(("all-B (PCA rotation)", make_uniform_strategy(
            n_layers, n_kv_heads, "B", args.bits, args.model)))

    # Candidate 4: MSE-tuned strategy (if provided)
    if args.mse_strategy and os.path.exists(args.mse_strategy):
        with open(args.mse_strategy) as f:
            mse_strat = json.load(f)
        mse_strat["_method"] = "mse_tuned"
        candidates.append(("MSE-tuned", mse_strat))

    # FP16 baseline first
    unpatch_model(model)
    unpatch_v1(model)
    print(f"\n{'=' * 60}")
    print(f"FP16 baseline ({args.n_chunks} chunks)")
    print(f"{'=' * 60}")
    t0 = time.time()
    fp16_ppl, _, _ = evaluate_ppl_quick(model, tokenizer, text, args.seq_len, args.n_chunks)
    print(f"  PPL: {fp16_ppl:.2f}  ({time.time()-t0:.1f}s)")

    # Evaluate each candidate
    results: list[dict] = []
    for name, strategy in candidates:
        unpatch_model(model)
        unpatch_v1(model)

        config = {"strategy": strategy, "calibration": None}
        if args.calibration and os.path.exists(args.calibration):
            from quant_attention_v2 import load_calibration
            config["calibration"] = load_calibration(args.calibration)

        patch_model_for_quantization_v3(model, args.bits, config)

        print(f"\n{'=' * 60}")
        print(f"Candidate: {name}")
        print(f"{'=' * 60}")
        t0 = time.time()
        ppl, ntok, nchk = evaluate_ppl_quick(model, tokenizer, text, args.seq_len, args.n_chunks)
        elapsed = time.time() - t0
        print(f"  PPL: {ppl:.2f}  ({ntok} tokens, {nchk} chunks, {elapsed:.1f}s)")

        results.append({
            "name": name,
            "ppl": ppl,
            "strategy": strategy,
            "tokens": ntok,
            "chunks": nchk,
        })

        unpatch_model(model)

    # Pick winner
    results.sort(key=lambda r: r["ppl"])
    winner = results[0]
    print(f"\n{'=' * 60}")
    print(f"WINNER: {winner['name']}  (PPL: {winner['ppl']:.2f})")
    print(f"{'=' * 60}")
    print(f"  FP16 baseline: {fp16_ppl:.2f}")
    for r in results:
        delta = r["ppl"] - fp16_ppl
        marker = " <<<" if r["name"] == winner["name"] else ""
        print(f"  {r['name']:<25} PPL: {r['ppl']:>10.2f}  ({delta:+.2f}){marker}")

    # Save winning strategy
    winning_strategy = winner["strategy"].copy()
    winning_strategy["_method"] = "ppl_guided"
    winning_strategy["_ppl_guided_selection"] = {
        "fp16_ppl": fp16_ppl,
        "winner": winner["name"],
        "winner_ppl": winner["ppl"],
        "n_chunks": args.n_chunks,
        "all_candidates": [
            {"name": r["name"], "ppl": r["ppl"]} for r in results
        ],
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(winning_strategy, f, indent=2)
    print(f"\nSaved winning strategy to {args.output}")


if __name__ == "__main__":
    main()
