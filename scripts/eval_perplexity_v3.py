#!/usr/bin/env python3
"""
eval_perplexity_v3.py — Perplexity evaluation with FajarQuant v3 per-head dispatch.

Usage:
    python scripts/eval_perplexity_v3.py \
        --model google/gemma-4-E2B \
        --strategy data/calibration/strategy_gemma_2bit.json \
        --calibration data/calibration/fq_v2_gemma_4_e2b.npz \
        --bits 2 --max-samples 30 --seq-len 2048 \
        --output data/kv_cache/perplexity_v3_gemma.json
"""

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


def load_wikitext2() -> str:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join([t for t in ds["text"] if t.strip()])


def evaluate_perplexity(model, tokenizer, text, seq_len=2048, max_samples=30):
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        from transformers import DynamicCache

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    total_len = input_ids.size(1)
    n_chunks = min(total_len // seq_len, max_samples)

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

    ppl = float(torch.exp(torch.tensor(sum(nlls) / n_tokens)))
    return ppl, n_tokens, n_chunks


def main():
    parser = argparse.ArgumentParser(description="FajarQuant v3 perplexity eval")
    parser.add_argument("--model", default="google/gemma-4-E2B")
    parser.add_argument("--strategy", required=True, help="Strategy JSON from strategy_selector.py")
    parser.add_argument("--calibration", default=None, help="v2 calibration .npz (for Path B)")
    parser.add_argument("--bits", default="2,3,4")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--token", default=None)
    parser.add_argument("--output", default="data/kv_cache/perplexity_v3_result.json")
    args = parser.parse_args()

    bits_list = [int(b) for b in args.bits.split(",")]

    print("Loading WikiText-2 test set...")
    text = load_wikitext2()

    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, token=args.token, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, token=args.token, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()

    results: dict = {}

    # FP16 baseline
    unpatch_model(model)
    unpatch_v1(model)
    print(f"\n{'=' * 60}")
    print("Evaluating: FP16 baseline")
    print(f"{'=' * 60}")
    t0 = time.time()
    ppl, ntok, nsamp = evaluate_perplexity(model, tokenizer, text, args.seq_len, args.max_samples)
    print(f"  PPL: {ppl:.2f}  ({ntok} tokens, {nsamp} chunks, {time.time()-t0:.1f}s)")
    results["fp16"] = {"ppl": ppl, "tokens": ntok, "samples": nsamp}

    for bits in bits_list:
        # Load strategy for this bit width
        # Strategy file may be bit-specific or generic
        strategy_path = args.strategy
        if "{bits}" in strategy_path:
            strategy_path = strategy_path.replace("{bits}", str(bits))

        config = load_v3_config(strategy_path, args.calibration)

        # v3 per-head dispatch
        unpatch_model(model)
        unpatch_v1(model)
        patch_model_for_quantization_v3(model, bits, config)

        print(f"\n{'=' * 60}")
        print(f"Evaluating: FajarQuant v3 @ {bits}-bit")
        strat = config["strategy"]
        print(f"  Strategy: {strat.get('path_counts', {})} ({strat.get('n_unique_paths', 0)} unique paths)")
        print(f"{'=' * 60}")
        t0 = time.time()
        ppl, ntok, nsamp = evaluate_perplexity(model, tokenizer, text, args.seq_len, args.max_samples)
        elapsed = time.time() - t0
        print(f"  PPL: {ppl:.2f}  ({ntok} tokens, {nsamp} chunks, {elapsed:.1f}s)")

        key = f"fajarquant_v3_{bits}bit"
        results[key] = {"ppl": ppl, "tokens": ntok, "samples": nsamp, "bits": bits}

        unpatch_model(model)

    # Also run KIVI and TQ outlier for comparison
    for method in ["kivi", "turboquant_outlier"]:
        for bits in bits_list:
            unpatch_model(model)
            unpatch_v1(model)
            patch_model_for_quantization(model, method, bits)

            print(f"\n{'=' * 60}")
            print(f"Evaluating: {method} @ {bits}-bit")
            print(f"{'=' * 60}")
            t0 = time.time()
            ppl, ntok, nsamp = evaluate_perplexity(model, tokenizer, text, args.seq_len, args.max_samples)
            elapsed = time.time() - t0
            print(f"  PPL: {ppl:.2f}  ({ntok} tokens, {nsamp} chunks, {elapsed:.1f}s)")

            key = f"{method}_{bits}bit"
            results[key] = {"ppl": ppl, "tokens": ntok, "samples": nsamp, "bits": bits}
            unpatch_v1(model)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"{'RESULTS — ' + args.model:^70}")
    print(f"{'=' * 70}")
    fp16_ppl = results["fp16"]["ppl"]
    print(f"FP16 baseline: {fp16_ppl:.2f}\n")
    for bits in bits_list:
        print(f"  --- {bits}-bit ---")
        for method in ["fajarquant_v3", "kivi", "turboquant_outlier"]:
            key = f"{method}_{bits}bit"
            if key in results:
                delta = results[key]["ppl"] - fp16_ppl
                print(f"  {method:<25} {results[key]['ppl']:>10.2f}  ({delta:+.2f})")
        print()

    # Save
    results["_protocol"] = {
        "name": "R-α.1 model surgery (v3)",
        "model": args.model,
        "strategy": args.strategy,
        "calibration": args.calibration,
        "seq_len": args.seq_len,
        "max_samples": args.max_samples,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
