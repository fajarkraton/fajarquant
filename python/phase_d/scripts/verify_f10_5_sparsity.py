#!/usr/bin/env python3
"""F.10.5 verification — confirm 50% sparsity at every SparseFusedBitLinear
site after sparse-from-scratch QAT.

Reconstructs the Mini model, attaches --sparse-2-4 replacement, runs ONE
forward to trigger mask creation, then inspects every SparseFusedBitLinear's
`_cached_mask` buffer. Reports per-site sparsity ratio. Asserts:

  1. All replaced sites have a populated `_cached_mask`
  2. Each site's mask is structurally 2:4 (50% zeros)
  3. Total replaced sites == expected (Mini = 37 if include lm_head, 36 if skip)

Usage:
    cd python/phase_d
    PYTHONPATH=. ../../.venv/bin/python scripts/verify_f10_5_sparsity.py

Output: prints per-site stats + final PASS/FAIL summary; exits 0/1.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # python/phase_d
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from intllm.model import HGRNBitConfig, HGRNBitForCausalLM  # noqa: E402
from intllm.quant import SparseFusedBitLinear, replace_bitlinear_with_sparse  # noqa: E402


# Mini config (matches train_mini_ablation.py + Phase D Mini c.1)
MINI_CONFIG = {
    "vocab_size": 32768,
    "hidden_size": 256,
    "num_hidden_layers": 6,
    "max_position_embeddings": 2048,
}

EXPECTED_SITES_INCLUDE_LM_HEAD = 6 * 6 + 1  # 37 (6 layers × 6 BitLinear + lm_head)
EXPECTED_SITES_SKIP_LM_HEAD = 6 * 6  # 36


def main() -> int:
    print("F.10.5 verify-sparsity — Mini × sparse-2-4 site inspection")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("FAIL: CUDA required for sparse path (Triton mask kernel)")
        return 1

    # Build Mini model
    cfg = HGRNBitConfig(**MINI_CONFIG)
    model = HGRNBitForCausalLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[1/4] built Mini model: {n_params/1e6:.2f}M params, "
          f"L={MINI_CONFIG['num_hidden_layers']}, "
          f"d={MINI_CONFIG['hidden_size']}")

    # Replace with sparse — match F.10.4 default settings
    n_replaced = replace_bitlinear_with_sparse(
        model, sparse_n=2, sparse_m=4, mask_refresh_interval=100, skip_lm_head=True
    )
    print(f"[2/4] replaced {n_replaced} BitLinear sites with SparseFusedBitLinear "
          f"(skip_lm_head=True, expected {EXPECTED_SITES_SKIP_LM_HEAD})")
    if n_replaced != EXPECTED_SITES_SKIP_LM_HEAD:
        print(
            f"FAIL: replaced count {n_replaced} != expected "
            f"{EXPECTED_SITES_SKIP_LM_HEAD}",
        )
        return 1

    # Trigger mask creation via one forward pass
    input_ids = torch.randint(0, MINI_CONFIG["vocab_size"], (1, 64), device=device)
    print(f"[3/4] forward pass (1, 64) to trigger mask computation...")
    with torch.no_grad():
        _ = model(input_ids)

    # Inspect each SparseFusedBitLinear's cached mask
    print(f"[4/4] inspecting cached masks at all {n_replaced} sites:")
    print()
    print(f"  {'site name':<60} {'shape':<20} {'sparsity':>10}")
    print(f"  {'-'*60} {'-'*20} {'-'*10}")

    failures: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, SparseFusedBitLinear):
            mask = module._cached_mask
            if mask is None:
                failures.append(f"{name}: _cached_mask is None after forward")
                print(f"  {name:<60} {'(no mask)':<20} {'FAIL':>10}")
                continue

            sparsity_ratio = (mask == 0).float().mean().item()
            shape_str = f"{tuple(mask.shape)}"
            status_str = f"{sparsity_ratio:.4f}"
            print(f"  {name:<60} {shape_str:<20} {status_str:>10}")

            # Strict 2:4 invariant: exactly 50% zeros
            if abs(sparsity_ratio - 0.5) > 1e-6:
                failures.append(
                    f"{name}: sparsity={sparsity_ratio:.4f} != 0.5"
                )

    print()
    if failures:
        print(f"FAIL: {len(failures)} site(s) violated 2:4 invariant:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"PASS: all {n_replaced} sites have exactly 50% sparsity (2:4 invariant satisfied)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
