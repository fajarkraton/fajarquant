#!/usr/bin/env python3
"""V31.C.P3.5 — fp16-vs-ternary per-layer parity measurement runner.

Loads a trained IntLLM checkpoint and runs intllm.fp16_parity.measure_parity
on a deterministic prompt. Outputs:

  - paper/intllm/results/fp16_parity_intllm-{model}.json
        Per-layer rel_error_l2 + summary stats. Source-of-truth for
        paper Table 3.
  - stdout: human-readable summary table.

Per FJQ_PHASE_D_PRODUCTION_PLAN.md §3.5 this is the IntLLM
differentiator vs BitNet 2B4T which did NOT ship a parity gate.
The threshold for `make test-intllm-fp16-parity` will be set in a
follow-up commit once we have measured baselines from this run.

Usage:
    cd fajarquant/python/phase_d
    PYTHONPATH=. ../../.venv/bin/python scripts/run_fp16_parity.py \\
        --checkpoint checkpoints/mini/mini_final.pt --tag intllm-mini

    # Or for Base when c.1 lands:
    PYTHONPATH=. ../../.venv/bin/python scripts/run_fp16_parity.py \\
        --checkpoint checkpoints/base/base_final.pt --tag intllm-base
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO_ROOT = ROOT.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from intllm.fp16_parity import measure_parity, summary_stats  # noqa: E402
from intllm.model import HGRNBitConfig, HGRNBitForCausalLM  # noqa: E402

# Deterministic prompt — a 32-token English sentence the model has
# almost-certainly seen during SlimPajama training. Choose a long-enough
# context to exercise sequence-dependent kernels (the linear-attention
# path collapses to identity for seq_len < 2).
DEFAULT_PROMPT_TOKENS = [
    1,    # BOS
    345, 678, 901, 234, 567, 890, 123, 456, 789, 12,
    34, 56, 78, 90, 11, 22, 33, 44, 55, 66, 77, 88, 99,
    111, 222, 333, 444, 555, 666, 777, 888,
]
# 32 tokens incl. BOS — enough sequence for HGRN recurrence to matter.


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to .pt checkpoint saved by train_*.py")
    p.add_argument("--tag", required=True,
                   help="Output tag, e.g. 'intllm-mini' for "
                        "paper/intllm/results/fp16_parity_intllm-mini.json")
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "paper" / "intllm" / "results",
                   help="Where to write fp16_parity_<tag>.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-layers", type=int, default=None,
                   help="Cap number of captured BitLinear hooks (debug)")
    args = p.parse_args()

    if not args.checkpoint.exists():
        print(f"FATAL: checkpoint not found: {args.checkpoint}")
        return 2

    if args.device == "cpu":
        print("WARN: CPU device — HGRNBit upstream uses fused triton "
              "kernels and will likely error.")

    print(f"[1/4] loading checkpoint {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    arch_dict = state["arch"]
    train_dict = state.get("train", {})
    cfg = HGRNBitConfig(
        vocab_size=arch_dict["vocab_size"],
        hidden_size=arch_dict["hidden_size"],
        num_hidden_layers=arch_dict["num_hidden_layers"],
        max_position_embeddings=arch_dict["max_position_embeddings"],
    )
    model = HGRNBitForCausalLM(cfg).to(args.device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      params: {n_params:,}, hidden={arch_dict['hidden_size']}, "
          f"L={arch_dict['num_hidden_layers']}, V={arch_dict['vocab_size']}")

    print(f"[2/4] preparing prompt ({len(DEFAULT_PROMPT_TOKENS)} tokens)")
    # Clamp into vocab in case the checkpoint has a smaller vocab than expected.
    vocab = arch_dict["vocab_size"]
    safe_tokens = [t % vocab for t in DEFAULT_PROMPT_TOKENS]
    input_ids = torch.tensor([safe_tokens], device=args.device)

    print(f"[3/4] measuring parity (max_layers={args.max_layers}) ...")
    t0 = time.time()
    results = measure_parity(model, input_ids, max_layers=args.max_layers)
    elapsed = time.time() - t0
    print(f"      {len(results)} hooks captured in {elapsed:.1f} s")

    stats = summary_stats(results)
    print()
    print("[4/4] per-layer rel_error_l2 (sample):")
    print(f"      {'layer':<60s} rel_l2     abs_l2")
    items = list(results.items())
    head = items[: min(5, len(items))]
    tail = items[-min(5, len(items)):]
    for name, parity in head:
        print(f"      {name:<60s} {parity.rel_error_l2:8.4f}  {parity.abs_error_l2:8.4f}")
    if len(items) > 10:
        print(f"      {'... (' + str(len(items) - 10) + ' more) ...':<60s}")
    if len(items) > 5:
        for name, parity in tail:
            print(f"      {name:<60s} {parity.rel_error_l2:8.4f}  {parity.abs_error_l2:8.4f}")
    print()
    print(f"      n_layers           : {stats['n_layers']}")
    print(f"      rel_error_l2 mean  : {stats['rel_error_l2_mean']:.4f}")
    print(f"      rel_error_l2 max   : {stats['rel_error_l2_max']:.4f}")
    print(f"      rel_error_l2 p95   : {stats['rel_error_l2_p95']:.4f}")
    print(f"      rel_error_l2 p50   : {stats['rel_error_l2_p50']:.4f}")
    print(f"      rel_error_l2 min   : {stats['rel_error_l2_min']:.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"fp16_parity_{args.tag}.json"
    payload = {
        "_description": (
            "Per-layer fp16-vs-ternary forward parity. "
            "Source: intllm.fp16_parity.measure_parity. "
            "Run via scripts/run_fp16_parity.py."
        ),
        "_phase": "FJQ_PHASE_D_PRODUCTION_PLAN.md §3.5 (IntLLM differentiator)",
        "model": args.tag,
        "checkpoint_path": str(args.checkpoint.relative_to(ROOT) if args.checkpoint.is_relative_to(ROOT)
                                else args.checkpoint),
        "checkpoint_params": n_params,
        "arch": arch_dict,
        "train": train_dict,
        "prompt_tokens": safe_tokens,
        "elapsed_seconds": elapsed,
        "device": args.device,
        "summary": stats,
        "per_layer": {name: asdict(p) for name, p in results.items()},
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n      artifact: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
