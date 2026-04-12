#!/usr/bin/env python3
"""
FajarQuant — Perplexity Evaluation with Quantized KV Cache (R-α.1)

Computes WikiText-2 perplexity for FP16 baseline + 3 KV cache
quantization methods (FajarQuant, KIVI, TurboQuant) using the
canonical KIVI/KVQuant/SKVQ protocol:

  - Non-overlapping chunks of `--seq-len` tokens
  - Per-chunk fresh DynamicCache + full forward pass
  - Quantization applied at attention layer level via per-architecture
    forward subclasses in scripts/quant_attention.py (R-α.1 model surgery)
  - Standard shift_logits/shift_labels cross-entropy loss
  - Final perplexity = exp(total_nll / total_scored_tokens)

This replaces the prefix+target post-hoc cache mutation approach used
in commit c9b2ff5, which produced FP16-vs-quant ordering anomalies due
to protocol mismatch with the literature. See
docs/V26_C1_6_METHODOLOGY.md for the full methodology decision trail.

Usage:
    python scripts/eval_perplexity.py --model google/gemma-4-E2B
    python scripts/eval_perplexity.py --model mistralai/Mistral-7B-v0.1 \\
        --output data/kv_cache/perplexity_mistral_7b.json
    python scripts/eval_perplexity.py --model Qwen/Qwen2-7B --seq-len 1024 \\
        --output data/kv_cache/perplexity_qwen2_7b.json
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

# Make scripts/ importable so we can pull in quant_attention.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_attention import patch_model_for_quantization, unpatch_model  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
# Dataset loader
# ═══════════════════════════════════════════════════════════════════════


def load_wikitext2() -> str:
    """Load WikiText-2 test set as a single concatenated string."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    return text


# ═══════════════════════════════════════════════════════════════════════
# Canonical perplexity evaluator (KVQuant/SKVQ-style)
# ═══════════════════════════════════════════════════════════════════════


def evaluate_perplexity_canonical(
    model,
    tokenizer,
    dataset_text: str,
    seq_len: int = 2048,
    max_samples: int = 50,
):
    """Canonical perplexity evaluation à la KVQuant/SKVQ.

    Partitions the tokenized text into non-overlapping chunks of
    `seq_len` tokens. For each chunk:
      1. Allocate a fresh DynamicCache (so KV state from previous
         chunks does not leak across chunk boundaries).
      2. Forward the full chunk through the model with use_cache=True.
         If the model has been patched via
         quant_attention.patch_model_for_quantization, every attention
         layer applies its quant method to (key_states, value_states)
         before they're written to the fresh cache.
      3. Score next-token predictions: cross-entropy of
         logits[:, :-1, :] vs chunk[:, 1:], reduction='sum'.
      4. Accumulate negative log-likelihood and token count.

    Returns (ppl, total_scored_tokens, n_chunks).
    """
    encodings = tokenizer(dataset_text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)
    total_len = input_ids.size(1)

    n_chunks = min(total_len // seq_len, max_samples)
    if n_chunks == 0:
        raise ValueError(
            f"text too short ({total_len} tokens) for seq_len={seq_len}"
        )

    # Lazy import — DynamicCache moved between submodules across versions.
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        from transformers import DynamicCache  # type: ignore

    nlls: list[float] = []
    n_tokens = 0

    for i in range(n_chunks):
        chunk = input_ids[:, i * seq_len : (i + 1) * seq_len]

        with torch.no_grad():
            cache = DynamicCache()
            out = model(chunk, past_key_values=cache, use_cache=True)
            logits = out.logits  # (1, seq_len, vocab)

        # Standard next-token prediction loss (per-token sum)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        nlls.append(loss.item())
        n_tokens += shift_labels.numel()

        del out, cache, logits, shift_logits, shift_labels, loss
        torch.cuda.empty_cache()

    avg_nll = sum(nlls) / max(n_tokens, 1)
    ppl = math.exp(avg_nll)
    return ppl, n_tokens, n_chunks


# ═══════════════════════════════════════════════════════════════════════
# Main entry — runs FP16 baseline + 3 quant methods × {2,3,4}-bit
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Perplexity eval with quantized KV cache (R-α.1)"
    )
    parser.add_argument("--model", default="google/gemma-4-E2B")
    parser.add_argument("--bits", default="2,3,4")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Non-overlapping chunk length (KVQuant default: 2048)",
    )
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--token", default=None)
    parser.add_argument(
        "--output", default="data/kv_cache/perplexity_results.json"
    )
    args = parser.parse_args()

    bits_list = [int(b) for b in args.bits.split(",")]
    methods = ["fajarquant", "kivi", "turboquant"]

    print("Loading WikiText-2 test set...")
    text = load_wikitext2()
    print(f"  {len(text)} characters, ~{len(text.split())} words")

    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, token=args.token, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        token=args.token,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded on {model.device}")

    results: dict = {}

    # 1. FP16 baseline (ensure no patches active)
    unpatch_model(model)
    print(f"\n{'=' * 60}")
    print("Evaluating: FP16 baseline (no quantization)")
    print(f"{'=' * 60}")
    t0 = time.time()
    ppl, ntok, nsamp = evaluate_perplexity_canonical(
        model,
        tokenizer,
        text,
        seq_len=args.seq_len,
        max_samples=args.max_samples,
    )
    elapsed = time.time() - t0
    print(f"  PPL: {ppl:.2f}  ({ntok} tokens, {nsamp} chunks, {elapsed:.1f}s)")
    results["fp16"] = {"ppl": ppl, "tokens": ntok, "samples": nsamp}

    # 2. Quantized methods at each bit width
    for bits in bits_list:
        for method in methods:
            n_patched = patch_model_for_quantization(model, method, bits)
            print(f"\n{'=' * 60}")
            print(
                f"Evaluating: {method} @ {bits}-bit "
                f"(patched {n_patched} attention modules)"
            )
            print(f"{'=' * 60}")
            t0 = time.time()
            ppl, ntok, nsamp = evaluate_perplexity_canonical(
                model,
                tokenizer,
                text,
                seq_len=args.seq_len,
                max_samples=args.max_samples,
            )
            elapsed = time.time() - t0
            print(
                f"  PPL: {ppl:.2f}  ({ntok} tokens, {nsamp} chunks, {elapsed:.1f}s)"
            )
            key = f"{method}_{bits}bit"
            results[key] = {
                "ppl": ppl,
                "tokens": ntok,
                "samples": nsamp,
                "bits": bits,
            }
            unpatch_model(model)

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"{'PERPLEXITY RESULTS — ' + args.model:^70}")
    print(f"{'=' * 70}")
    print(f"{'Method':<25} {'PPL':>10} {'Tokens':>10}")
    print("-" * 50)
    print(
        f"{'FP16 (baseline)':<25} "
        f"{results['fp16']['ppl']:>10.2f} {results['fp16']['tokens']:>10}"
    )
    for bits in bits_list:
        print(f"\n  --- {bits}-bit ---")
        for method in methods:
            key = f"{method}_{bits}bit"
            if key in results:
                delta = results[key]["ppl"] - results["fp16"]["ppl"]
                sign = "+" if delta >= 0 else ""
                print(
                    f"  {method:<23} "
                    f"{results[key]['ppl']:>10.2f}  ({sign}{delta:.2f})"
                )

    # ── Save JSON (flat schema for verify_paper_tables.py compat) ────
    out_path = args.output
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Add a self-documenting protocol marker as a top-level key.
    # verify_paper_tables.py only reads "{method}_{bits}bit" entries, so
    # this extra key is harmless for backward compat.
    results["_protocol"] = {
        "name": "R-α.1 model surgery",
        "model": args.model,
        "seq_len": args.seq_len,
        "max_samples": args.max_samples,
        "source": "scripts/quant_attention.py",
        "methodology_doc": "docs/V26_C1_6_METHODOLOGY.md",
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
