#!/usr/bin/env python3
"""
FajarQuant — Perplexity Evaluation with Quantized KV Cache

Measures perplexity on WikiText-2 with different KV cache quantization methods:
1. FP16 baseline (no quantization)
2. FajarQuant (PCA rotation + per-coordinate quantization)
3. KIVI (per-channel keys / per-token values)
4. TurboQuant (random rotation + per-coordinate quantization)

Usage:
    python scripts/eval_perplexity.py --model google/gemma-4-E2B --bits 2,3,4
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy.stats import ortho_group
from transformers import AutoModelForCausalLM, AutoTokenizer


# ═══════════════════════════════════════════════════════════════
# Quantization Methods (operate on torch tensors for speed)
# ═══════════════════════════════════════════════════════════════

def quantize_per_coord(data: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-coordinate (per-channel) uniform quantization and dequantization."""
    levels = (1 << bits) - 1
    # data shape: (seq_len, d) or (heads, seq_len, d)
    if data.ndim == 3:
        mn = data.amin(dim=1, keepdim=True)
        mx = data.amax(dim=1, keepdim=True)
    else:
        mn = data.amin(dim=0, keepdim=True)
        mx = data.amax(dim=0, keepdim=True)
    rng = mx - mn
    rng = rng.clamp(min=1e-15)
    scale = rng / levels
    indices = ((data - mn) / scale).round().clamp(0, levels)
    return mn + indices * scale


def apply_fajarquant(kv_tensor: torch.Tensor, bits: int, is_key: bool = True) -> torch.Tensor:
    """FajarQuant: PCA rotation + per-coordinate quantization.
    kv_tensor: (batch, heads, seq_len, d_head) in float32/16
    """
    b, h, s, d = kv_tensor.shape
    device = kv_tensor.device
    result = torch.zeros_like(kv_tensor)

    for bi in range(b):
        for hi in range(h):
            data = kv_tensor[bi, hi].float()  # (seq, d)
            if s < 2:
                result[bi, hi] = data.to(kv_tensor.dtype)
                continue

            # PCA rotation
            mean = data.mean(dim=0)
            centered = data - mean
            cov = (centered.T @ centered) / max(s - 1, 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            idx = torch.argsort(eigenvalues, descending=True)
            R = eigenvectors[:, idx].T  # (d, d)

            rotated = (data - mean) @ R.T
            quantized = quantize_per_coord(rotated, bits)
            recon = quantized @ R + mean
            result[bi, hi] = recon.to(kv_tensor.dtype)

    return result


def apply_kivi(kv_tensor: torch.Tensor, bits: int, is_key: bool = True) -> torch.Tensor:
    """KIVI: per-channel for keys, per-token for values.
    kv_tensor: (batch, heads, seq_len, d_head)
    """
    b, h, s, d = kv_tensor.shape
    levels = (1 << bits) - 1
    result = torch.zeros_like(kv_tensor)

    for bi in range(b):
        for hi in range(h):
            data = kv_tensor[bi, hi].float()

            if is_key:
                # Per-channel (per d_head dimension)
                mn = data.amin(dim=0, keepdim=True)
                mx = data.amax(dim=0, keepdim=True)
            else:
                # Per-token (per sequence position)
                mn = data.amin(dim=1, keepdim=True)
                mx = data.amax(dim=1, keepdim=True)

            rng = (mx - mn).clamp(min=1e-15)
            scale = rng / levels
            indices = ((data - mn) / scale).round().clamp(0, levels)
            recon = mn + indices * scale
            result[bi, hi] = recon.to(kv_tensor.dtype)

    return result


def apply_turboquant(kv_tensor: torch.Tensor, bits: int, seed: int = 42) -> torch.Tensor:
    """TurboQuant: random orthogonal rotation + per-coordinate quantization."""
    b, h, s, d = kv_tensor.shape
    result = torch.zeros_like(kv_tensor)

    rng = np.random.default_rng(seed)
    R_np = ortho_group.rvs(d, random_state=rng)
    R = torch.from_numpy(R_np).float().to(kv_tensor.device)

    for bi in range(b):
        for hi in range(h):
            data = kv_tensor[bi, hi].float()
            rotated = data @ R.T
            quantized = quantize_per_coord(rotated, bits)
            recon = quantized @ R
            result[bi, hi] = recon.to(kv_tensor.dtype)

    return result


# ═══════════════════════════════════════════════════════════════
# Perplexity Evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate_perplexity(
    model,
    tokenizer,
    dataset_text: str,
    method: str = "none",
    bits: int = 4,
    max_length: int = 512,
    stride: int = 256,
    max_samples: int = 50,
):
    """Evaluate perplexity with optional KV cache quantization.

    Uses sliding window approach: process text in overlapping chunks,
    only score the non-overlapping portion of each chunk.
    """
    encodings = tokenizer(dataset_text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)
    total_len = input_ids.size(1)

    nlls = []
    n_tokens = 0
    samples = 0

    for begin in range(0, total_len - 1, stride):
        end = min(begin + max_length, total_len)
        chunk = input_ids[:, begin:end]
        target_len = end - begin - 1

        if chunk.size(1) < 4:
            break

        with torch.no_grad():
            outputs = model(chunk, use_cache=True)

            # Quantize KV cache and re-run if method != "none"
            if method != "none" and hasattr(outputs, "past_key_values"):
                past_kv = outputs.past_key_values
                if hasattr(past_kv, "layers"):
                    for layer in past_kv.layers:
                        k = layer.keys  # (batch, heads, seq, d)
                        v = layer.values
                        if method == "fajarquant":
                            layer.keys = apply_fajarquant(k, bits, is_key=True)
                            layer.values = apply_fajarquant(v, bits, is_key=False)
                        elif method == "kivi":
                            layer.keys = apply_kivi(k, bits, is_key=True)
                            layer.values = apply_kivi(v, bits, is_key=False)
                        elif method == "turboquant":
                            layer.keys = apply_turboquant(k, bits)
                            layer.values = apply_turboquant(v, bits)

                    # Re-forward with quantized cache (only last token to get logits)
                    outputs = model(
                        chunk[:, -1:],
                        past_key_values=past_kv,
                        use_cache=False,
                    )

            logits = outputs.logits

        # Compute NLL for scoring region
        if method == "none":
            # Full sequence logits available
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = chunk[:, 1:].contiguous()
        else:
            # Only last-token logits after quantized cache re-forward
            # Score just the last token prediction
            shift_logits = logits[:, -1:, :].contiguous()
            shift_labels = chunk[:, -1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        n_scored = shift_labels.numel()
        nlls.append(loss.item())
        n_tokens += n_scored
        samples += 1

        if samples >= max_samples:
            break

        # Free GPU memory
        del outputs
        torch.cuda.empty_cache()

    avg_nll = sum(nlls) / max(n_tokens, 1)
    ppl = math.exp(avg_nll)
    return ppl, n_tokens, samples


def load_wikitext2() -> str:
    """Load WikiText-2 test set as a single concatenated string."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    return text


def main():
    parser = argparse.ArgumentParser(description="Perplexity evaluation with quantized KV cache")
    parser.add_argument("--model", default="google/gemma-4-E2B")
    parser.add_argument("--bits", default="2,3,4")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--token", default=None)
    parser.add_argument("--output", default="data/kv_cache/perplexity_results.json")
    args = parser.parse_args()

    bits_list = [int(b) for b in args.bits.split(",")]

    print("Loading WikiText-2 test set...")
    text = load_wikitext2()
    print(f"  {len(text)} characters, ~{len(text.split())} words")

    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, token=args.token, dtype=torch.float16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded on {model.device}")

    results = {}

    # 1. FP16 baseline
    print(f"\n{'='*60}")
    print("Evaluating: FP16 baseline (no quantization)")
    print(f"{'='*60}")
    t0 = time.time()
    ppl, ntok, nsamp = evaluate_perplexity(
        model, tokenizer, text, method="none",
        max_length=args.max_length, stride=args.stride, max_samples=args.max_samples,
    )
    elapsed = time.time() - t0
    print(f"  PPL: {ppl:.2f}  ({ntok} tokens, {nsamp} samples, {elapsed:.1f}s)")
    results["fp16"] = {"ppl": ppl, "tokens": ntok, "samples": nsamp}

    # 2. Quantized methods at each bit width
    methods = ["fajarquant", "kivi", "turboquant"]
    for bits in bits_list:
        for method in methods:
            print(f"\n{'='*60}")
            print(f"Evaluating: {method} @ {bits}-bit")
            print(f"{'='*60}")
            t0 = time.time()
            ppl, ntok, nsamp = evaluate_perplexity(
                model, tokenizer, text, method=method, bits=bits,
                max_length=args.max_length, stride=args.stride, max_samples=args.max_samples,
            )
            elapsed = time.time() - t0
            print(f"  PPL: {ppl:.2f}  ({ntok} tokens, {nsamp} samples, {elapsed:.1f}s)")
            key = f"{method}_{bits}bit"
            results[key] = {"ppl": ppl, "tokens": ntok, "samples": nsamp, "bits": bits}

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'PERPLEXITY RESULTS — ' + args.model:^70}")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'PPL':>10} {'Tokens':>10}")
    print("-" * 50)
    print(f"{'FP16 (baseline)':<25} {results['fp16']['ppl']:>10.2f} {results['fp16']['tokens']:>10}")
    for bits in bits_list:
        print(f"\n  --- {bits}-bit ---")
        for method in methods:
            key = f"{method}_{bits}bit"
            if key in results:
                delta = results[key]["ppl"] - results["fp16"]["ppl"]
                print(f"  {method:<23} {results[key]['ppl']:>10.2f}  (+{delta:.2f})")

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
