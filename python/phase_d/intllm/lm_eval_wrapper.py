"""V31.C.P3.2.1 — HFLM adapter for HGRNBitForCausalLM.

Because HGRNBitForCausalLM inherits from `transformers.PreTrainedModel` +
`GenerationMixin` (see `mmfreelm.models.hgrn_bit.modeling_hgrn_bit`), it is
fully HF-compatible and can be passed directly to
`lm_eval.models.huggingface.HFLM` as the `pretrained=` argument. No custom
`LM` subclass needed.

Entry point:
    build_hflm(checkpoint_path) -> HFLM

consumed by `scripts/bench_canonical.py` real-mode.

Design note (§6.9 R3 baseline parity): we deliberately reuse HFLM rather
than wrap a minimal LM subclass, so IntLLM eval uses the *same* loglikelihood
+ generation plumbing that lm-eval applies to the baseline comparison
models (BitNet 2B4T, MMfreeLM-370M, SmolLM2, Pythia) under Phase 3.4. This
eliminates a class of silent disagreement where "IntLLM scores differ from
baselines" could trace to the wrapper rather than the model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from intllm.model import HGRNBitConfig, HGRNBitForCausalLM
from intllm.tokenizer import get_tokenizer


def build_hflm(
    checkpoint: Path,
    device: str = "cuda",
    batch_size: int = 4,
    max_length: Optional[int] = None,
):
    """Load an IntLLM .pt checkpoint and wrap it as an lm_eval HFLM.

    `max_length` defaults to `arch.max_position_embeddings` from the
    checkpoint if not given.
    """
    # Import inside function so callers that only want the module (e.g.
    # for testing imports) don't pay the lm_eval import cost.
    from lm_eval.models.huggingface import HFLM

    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    arch = state["arch"]
    cfg = HGRNBitConfig(
        vocab_size=arch["vocab_size"],
        hidden_size=arch["hidden_size"],
        num_hidden_layers=arch["num_hidden_layers"],
        max_position_embeddings=arch["max_position_embeddings"],
    )
    model = HGRNBitForCausalLM(cfg).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    tokenizer = get_tokenizer()

    if max_length is None:
        max_length = arch["max_position_embeddings"]

    return HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )


def extract_results(
    lm_eval_output: dict,
) -> dict[str, dict[str, float]]:
    """Flatten lm_eval.simple_evaluate output into the canonical schema
    consumed by scripts/verify_intllm_tables.py::bench_canonical.

    lm-eval emits keys like 'acc,none' / 'acc_stderr,none' — the ',none'
    suffix identifies the filter group. We collapse it since IntLLM doesn't
    use per-filter variants today.
    """
    flat: dict[str, dict[str, float]] = {}
    for task, metrics in lm_eval_output.get("results", {}).items():
        flat[task] = {}
        for k, v in metrics.items():
            if k == "alias":
                continue
            clean = k.split(",")[0]
            try:
                flat[task][clean] = float(v)
            except (TypeError, ValueError):
                # skip non-numeric entries (e.g. 'samples')
                continue
    return flat
