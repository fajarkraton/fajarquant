#!/usr/bin/env python3
"""
eval_fp16_baseline.py — B-fix.4: Standalone FP16 baseline evaluation.

No imports from quant_attention*.py. No patch/unpatch machinery.
Pure HuggingFace model + WikiText-2 perplexity evaluation.

Purpose: verify whether PPL=28.13 for Gemma-4-E2B is correct, or if
the patching/unpatching machinery in eval_perplexity_v3.py introduced
an artifact causing quantized methods to appear better than FP16.

Usage:
    python scripts/eval_fp16_baseline.py --model google/gemma-4-E2B
"""

import argparse
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_wikitext2() -> str:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join([t for t in ds["text"] if t.strip()])


def evaluate_perplexity_no_cache(model, tokenizer, text, seq_len=2048, max_samples=30):
    """Evaluate perplexity WITHOUT use_cache (no KV cache at all)."""
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    total_len = input_ids.size(1)
    n_chunks = min(total_len // seq_len, max_samples)

    nlls, n_tokens = [], 0
    for i in range(n_chunks):
        chunk = input_ids[:, i * seq_len : (i + 1) * seq_len]
        with torch.no_grad():
            out = model(chunk, use_cache=False)
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


def evaluate_perplexity_with_cache(model, tokenizer, text, seq_len=2048, max_samples=30):
    """Evaluate perplexity WITH DynamicCache (same as eval_perplexity_v3.py)."""
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
    parser = argparse.ArgumentParser(description="Standalone FP16 baseline PPL")
    parser.add_argument("--model", default="google/gemma-4-E2B")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    print(f"=== B-fix.4: Standalone FP16 baseline ===")
    print(f"Model: {args.model}")
    print(f"NO quant_attention imports. NO patch/unpatch.")
    print()

    text = load_wikitext2()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, token=args.token, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, token=args.token, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True,
    ).eval()

    # Test 1: No cache
    print(f"\n{'='*60}")
    print("Test 1: FP16 WITHOUT KV cache (use_cache=False)")
    print(f"{'='*60}")
    t0 = time.time()
    ppl_no_cache, ntok, nsamp = evaluate_perplexity_no_cache(
        model, tokenizer, text, args.seq_len, args.max_samples
    )
    print(f"  PPL: {ppl_no_cache:.4f}  ({ntok} tokens, {nsamp} chunks, {time.time()-t0:.1f}s)")

    # Test 2: With DynamicCache (same protocol as eval_perplexity_v3.py)
    print(f"\n{'='*60}")
    print("Test 2: FP16 WITH DynamicCache (same as v3 eval)")
    print(f"{'='*60}")
    t0 = time.time()
    ppl_with_cache, ntok, nsamp = evaluate_perplexity_with_cache(
        model, tokenizer, text, args.seq_len, args.max_samples
    )
    print(f"  PPL: {ppl_with_cache:.4f}  ({ntok} tokens, {nsamp} chunks, {time.time()-t0:.1f}s)")

    # Compare
    print(f"\n{'='*60}")
    print("Comparison")
    print(f"{'='*60}")
    print(f"  No cache:        {ppl_no_cache:.4f}")
    print(f"  With cache:      {ppl_with_cache:.4f}")
    print(f"  Delta:           {ppl_with_cache - ppl_no_cache:+.4f}")
    print(f"  v3 eval reported: 28.13")
    print()
    if abs(ppl_with_cache - 28.13) < 0.5:
        print("  CONCLUSION: v3 eval FP16 baseline is CONSISTENT with standalone.")
        print("  PPL < FP16 anomaly is a real property of the evaluation, not")
        print("  an artifact of the patching/unpatching mechanism.")
    else:
        print(f"  CONCLUSION: v3 eval FP16 baseline DIFFERS from standalone by "
              f"{abs(ppl_with_cache - 28.13):.2f}.")
        print("  The patching/unpatching mechanism may be affecting results.")


if __name__ == "__main__":
    main()
