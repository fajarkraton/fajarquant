"""Minimal training loop for IntLLM proof-of-life (C.P2.3).

This module is intentionally tiny — it exists to verify that
`intllm.model` (which re-exports upstream HGRNBit*) actually trains
end-to-end. The full C.P4 training pipeline (lr schedule, gradient
clipping, mixed precision, wandb logging, gradient checkpointing) is
built on top of this loop, not in this file.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.optim import AdamW


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    log_every: int = 50


@dataclass
class TrainResult:
    losses: list[float] = field(default_factory=list)
    initial_loss: float = float("nan")
    final_loss: float = float("nan")
    steps: int = 0


def _causal_lm_loss(model_output, labels: torch.Tensor) -> torch.Tensor:
    """Standard shift-by-one causal LM cross-entropy loss.

    Most HF causal LMs (including upstream HGRNBitForCausalLM) expose
    `.loss` directly when `labels` is passed to `forward`. We accept
    pre-computed loss when present, otherwise compute manually.
    """
    if hasattr(model_output, "loss") and model_output.loss is not None:
        return model_output.loss
    logits = model_output.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def train_loop(
    model: nn.Module,
    batches: Iterable[torch.Tensor],
    *,
    config: TrainConfig | None = None,
) -> TrainResult:
    """Run a minimal AdamW training loop over the given batches.

    Each batch is a `(B, T)` int64 tensor of token IDs; loss is the
    standard shift-by-one causal LM cross-entropy. Returns the loss
    trace + initial/final values for assertion-friendly inspection.

    Does NOT do mixed precision, gradient checkpointing, lr schedule,
    or QAT — those layer in via C.P2.4 and C.P4 (separately).
    """
    cfg = config or TrainConfig()
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    model.train()

    result = TrainResult()
    for step, ids in enumerate(batches):
        out = model(input_ids=ids, labels=ids)
        loss = _causal_lm_loss(out, ids)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        optimizer.step()

        loss_val = loss.detach().item()
        result.losses.append(loss_val)
        if step == 0:
            result.initial_loss = loss_val
        result.final_loss = loss_val
        result.steps = step + 1

        if cfg.log_every and (step + 1) % cfg.log_every == 0:
            print(f"  step {step + 1:4d} loss={loss_val:.4f}")

    model.eval()
    return result


def smoothed_min(losses: list[float], window: int = 10) -> float:
    """Average of the lowest-`window`-step rolling mean — robust loss
    floor estimator that ignores single-step spikes."""
    if not losses:
        return float("nan")
    if len(losses) < window:
        return float(sum(losses) / len(losses))
    rolling = [sum(losses[i : i + window]) / window for i in range(len(losses) - window + 1)]
    return float(min(rolling))
