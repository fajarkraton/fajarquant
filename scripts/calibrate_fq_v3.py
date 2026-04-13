#!/usr/bin/env python3
"""
calibrate_fq_v3.py — Unified FajarQuant v3 calibration pipeline.

Runs the full pipeline: profile → auto-tune → assign → verify → save.

Output: single .npz file containing:
  - Per-head profile statistics
  - Optimal thresholds
  - Strategy assignments (verified)
  - v2 calibration data (PCA matrices for Path B, if available)

Usage:
    python scripts/calibrate_fq_v3.py --model google/gemma-4-E2B \
        --v2-calibration data/calibration/fq_v2_gemma_4_e2b.npz \
        --bits 2,3,4 \
        --output data/calibration/fq_v3_gemma.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="FajarQuant v3 unified calibration")
    parser.add_argument("--model", default="google/gemma-4-E2B")
    parser.add_argument("--v2-calibration", default=None)
    parser.add_argument("--bits", default="2,3,4")
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--token", default=None)
    parser.add_argument("--output-dir", default="data/calibration")
    args = parser.parse_args()

    bits_list = [int(b) for b in args.bits.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    # Derive model short name for file naming
    model_short = args.model.split("/")[-1].lower().replace("-", "_")

    t_total = time.time()

    # ═══════════════════════════════════════════════════════════
    # Step 1: Profile
    # ═══════════════════════════════════════════════════════════
    profile_path = os.path.join(args.output_dir, f"profile_{model_short}.json")
    if os.path.exists(profile_path):
        print(f"[1/4] Profile exists: {profile_path} (reusing)")
        with open(profile_path) as f:
            profile = json.load(f)
    else:
        print(f"[1/4] Profiling {args.model}...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from profile_kv_heads import load_wikitext2, profile_model

        tokenizer = AutoTokenizer.from_pretrained(
            args.model, token=args.token, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, token=args.token, dtype=torch.float16,
            device_map="auto", trust_remote_code=True,
        ).eval()

        text = load_wikitext2()
        profile = profile_model(model, tokenizer, text, args.n_samples, args.seq_len)

        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2)
        print(f"  Saved to {profile_path}")

        del model
        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════
    # Step 2: Auto-tune thresholds (per bit width)
    # ═══════════════════════════════════════════════════════════
    from tune_thresholds import grid_search, evaluate_threshold_combo, DEFAULT_THRESHOLDS

    print(f"\n[2/4] Auto-tuning thresholds for {bits_list}...")
    thresholds_per_bits = {}
    for bits in bits_list:
        best_thresh, best_score = grid_search(profile, bits)
        thresholds_per_bits[bits] = best_thresh
        print(f"  {bits}-bit: {best_thresh} (score={best_score:.1f})")

    # ═══════════════════════════════════════════════════════════
    # Step 3: Assign strategies
    # ═══════════════════════════════════════════════════════════
    from strategy_selector import assign_strategies

    print(f"\n[3/4] Assigning strategies...")
    strategies_per_bits = {}
    for bits in bits_list:
        strategy = assign_strategies(profile, bits, thresholds_per_bits[bits])
        strategies_per_bits[bits] = strategy
        print(f"  {bits}-bit: {strategy['path_counts']} ({strategy['n_unique_paths']} unique)")

    # ═══════════════════════════════════════════════════════════
    # Step 4: Verify + save
    # ═══════════════════════════════════════════════════════════
    from strategy_verifier import verify_and_swap

    print(f"\n[4/4] Verifying strategies...")
    for bits in bits_list:
        verified = verify_and_swap(strategies_per_bits[bits], profile, bits)
        v = verified["_verification"]
        print(f"  {bits}-bit: {v['swaps']} swaps ({v['swap_rate']:.1%})")

        # Save strategy file
        strat_path = os.path.join(
            args.output_dir, f"strategy_{model_short}_{bits}bit.json"
        )
        with open(strat_path, "w") as f:
            json.dump(verified, f, indent=2)

    # Save unified config
    config = {
        "_model": args.model,
        "_model_short": model_short,
        "_n_heads": profile.get("_n_heads", 1),
        "_head_dim": profile.get("_head_dim", 128),
        "_n_layers": profile.get("_n_layers", 0),
        "_v2_calibration": args.v2_calibration,
        "_profile_path": profile_path,
        "bits_available": bits_list,
        "thresholds": thresholds_per_bits,
        "strategy_files": {
            bits: f"strategy_{model_short}_{bits}bit.json"
            for bits in bits_list
        },
    }
    config_path = os.path.join(args.output_dir, f"fq_v3_{model_short}.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    elapsed = time.time() - t_total
    print(f"\nDone in {elapsed:.1f}s — config saved to {config_path}")
    print(f"Strategy files: {[f'strategy_{model_short}_{b}bit.json' for b in bits_list]}")


if __name__ == "__main__":
    main()
