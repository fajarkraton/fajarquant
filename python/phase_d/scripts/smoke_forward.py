#!/usr/bin/env python3
"""V31.C.P2.1c — minimal smoke test for vendored mmfreelm.

Loads `ridger/MMfreeLM-370M` via upstream's HGRNBit* classes, runs a
single forward pass on a test prompt, and verifies:
  - model loads without error
  - forward produces finite logits
  - greedy generation emits plausible token ids (no NaN / repeat lock)

This is NOT the full C.P2.1 gate (Wikitext-103 PPL repro) — that comes
next. This confirms the scaffolding works end-to-end.

Run:
    cd fajarquant/python/phase_d
    ../../.venv/bin/python scripts/smoke_forward.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Vendored snapshot path
HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parent / "_upstream"
if not UPSTREAM.is_dir():
    sys.exit(f"fatal: vendored snapshot missing at {UPSTREAM}. See UPSTREAM_PIN.md.")
sys.path.insert(0, str(UPSTREAM))

import torch  # noqa: E402
import mmfreelm  # noqa: F401, E402  — registers HGRNBit* model classes
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def main() -> int:
    model_name = "ridger/MMfreeLM-370M"
    prompt = "In a shocking finding, scientists discovered a herd of unicorns"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[1/4] torch {torch.__version__} on {device}")
    print(f"      CUDA available: {torch.cuda.is_available()}")
    if device == "cuda":
        print(f"      GPU: {torch.cuda.get_device_name(0)}")

    print(f"[2/4] loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).half()
    print(f"      params: {sum(p.numel() for p in model.parameters()):,}")

    print("[3/4] forward pass ...")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits
    print(f"      logits shape: {tuple(logits.shape)}")
    print(f"      logits finite: {bool(torch.isfinite(logits).all().item())}")
    print(f"      logits mean ± std: {logits.mean().item():.3f} ± {logits.std().item():.3f}")
    if not torch.isfinite(logits).all():
        print("      FAIL — NaN/Inf in logits")
        return 1

    print("[4/4] greedy generation ...")
    with torch.no_grad():
        gen = model.generate(input_ids, max_length=32, do_sample=False)
    text = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    print(f"      output:\n        {text}")

    # Smoke-test invariants
    prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    gen_only = gen[0, prompt_tokens:].tolist()
    if len(set(gen_only)) <= 1 and len(gen_only) > 1:
        print(f"      WARN: generation degenerated to single-token lock: {gen_only}")
    elif len(set(gen_only)) <= 3 and len(gen_only) > 4:
        print(f"      WARN: low token diversity (≤3 unique of {len(gen_only)})")

    print("\nPASS — upstream vendored snapshot loads + forward pass works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
