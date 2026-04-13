#!/usr/bin/env python3
"""
eval_perplexity_v2.py — Perplexity evaluation including FajarQuant v2 methods.

Runs FP16 baseline + v1 methods + v2-D + v2-A (ablation) on a given model
using canonical R-α.1 non-overlapping chunk protocol.

Usage:
    python scripts/eval_perplexity_v2.py --model google/gemma-4-E2B \
        --calibration data/calibration/fq_v2_gemma_4_e2b.npz
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_attention import patch_model_for_quantization, unpatch_model  # noqa: E402
from quant_attention_v2 import (  # noqa: E402
    load_calibration,
    patch_model_for_quantization_v2,
    unpatch_model as unpatch_v2,
)


def load_wikitext2() -> str:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join([t for t in ds["text"] if t.strip()])


def evaluate_perplexity_canonical(model, tokenizer, text, seq_len=2048, max_samples=50):
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        from transformers import DynamicCache

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    total_len = input_ids.size(1)
    n_chunks = min(total_len // seq_len, max_samples)
    if n_chunks == 0:
        raise ValueError(f"text too short ({total_len}) for seq_len={seq_len}")

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
        del out, cache, shift_logits, shift_labels, loss
        torch.cuda.empty_cache()

    return math.exp(sum(nlls) / max(n_tokens, 1)), n_tokens, n_chunks


def main():
    parser = argparse.ArgumentParser(description="Perplexity eval with FajarQuant v2")
    parser.add_argument("--model", default="google/gemma-4-E2B")
    parser.add_argument("--calibration", required=True, help="Path to .npz calibration file")
    parser.add_argument("--bits", default="2,3,4")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--token", default=None)
    parser.add_argument("--output", default="data/kv_cache/perplexity_v2_results.json")
    args = parser.parse_args()

    bits_list = [int(b) for b in args.bits.split(",")]

    # v1 methods (for comparison) + v2 methods
    v1_methods = ["fajarquant", "kivi", "turboquant_outlier"]
    v2_methods = ["fajarquant_v2", "fajarquant_v2a"]

    print("Loading WikiText-2 test set...")
    text = load_wikitext2()

    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, token=args.token, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, token=args.token, dtype=torch.float16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()

    print(f"Loading calibration: {args.calibration}")
    cal = load_calibration(args.calibration)

    results: dict = {}

    # FP16 baseline
    unpatch_v2(model)
    unpatch_model(model)
    print(f"\n{'=' * 60}")
    print("Evaluating: FP16 baseline")
    print(f"{'=' * 60}")
    t0 = time.time()
    ppl, ntok, nsamp = evaluate_perplexity_canonical(
        model, tokenizer, text, args.seq_len, args.max_samples
    )
    print(f"  PPL: {ppl:.2f}  ({ntok} tokens, {nsamp} chunks, {time.time()-t0:.1f}s)")
    results["fp16"] = {"ppl": ppl, "tokens": ntok, "samples": nsamp}

    all_methods = v1_methods + v2_methods

    for bits in bits_list:
        for method in all_methods:
            # Patch
            if method in v2_methods:
                patch_model_for_quantization_v2(model, method, bits, cal)
            else:
                patch_model_for_quantization(model, method, bits)

            print(f"\n{'=' * 60}")
            print(f"Evaluating: {method} @ {bits}-bit")
            print(f"{'=' * 60}")
            t0 = time.time()
            ppl, ntok, nsamp = evaluate_perplexity_canonical(
                model, tokenizer, text, args.seq_len, args.max_samples
            )
            elapsed = time.time() - t0
            print(f"  PPL: {ppl:.2f}  ({ntok} tokens, {nsamp} chunks, {elapsed:.1f}s)")

            key = f"{method}_{bits}bit"
            results[key] = {"ppl": ppl, "tokens": ntok, "samples": nsamp, "bits": bits}

            # Unpatch
            if method in v2_methods:
                unpatch_v2(model)
            else:
                unpatch_model(model)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"{'RESULTS — ' + args.model:^70}")
    print(f"{'=' * 70}")
    fp16_ppl = results["fp16"]["ppl"]
    print(f"FP16 baseline: {fp16_ppl:.2f}\n")
    for bits in bits_list:
        print(f"  --- {bits}-bit ---")
        for method in all_methods:
            key = f"{method}_{bits}bit"
            if key in results:
                delta = results[key]["ppl"] - fp16_ppl
                print(f"  {method:<25} {results[key]['ppl']:>10.2f}  ({delta:+.2f})")
        print()

    # Save
    results["_protocol"] = {
        "name": "R-α.1 model surgery (v2)",
        "model": args.model,
        "calibration": args.calibration,
        "seq_len": args.seq_len,
        "max_samples": args.max_samples,
    }
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
