"""Training loop for IntLLM with linear-warmup + cosine-decay LR schedule.

C.P2.3 shipped a constant-lr loop (proof-of-life). C.P4.1 Mini gate
FAIL identified the missing schedule as the H1 root cause (~+0.4 nat).
This revision adds a `LambdaLR` scheduler honouring `warmup_steps`,
`total_steps`, and `min_lr_ratio` from `TrainConfig`. Backward-
compatible: callers that don't set the new fields get a no-op
schedule (constant lr).
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    log_every: int = 50
    # LR schedule (C.P4.1 H1 fix). Defaults preserve old constant-lr behaviour.
    warmup_steps: int = 0          # 0 → no warmup (constant lr from step 0)
    total_steps: int = 0           # 0 → no decay (constant lr after warmup)
    min_lr_ratio: float = 0.1      # cosine decays to this fraction of peak lr
    # Track B step 1 (V31.C.P6.1 — interruption-safety). Defaults preserve
    # pre-hardening behaviour: ckpt_every=0 → no intermediate checkpoints.
    ckpt_every: int = 0            # 0 → disabled; N → save every N steps
    ckpt_dir: str | None = None    # where to write ckpt_step_{step:06d}.pt
    keep_last_n_ckpts: int = 3     # rotation — keep this many newest, prune older


def _lr_lambda(step: int, *, warmup: int, total: int, min_ratio: float) -> float:
    """Linear warmup → cosine decay to `min_ratio` of peak lr.

    `step` is 0-indexed. Returns the multiplier to apply to base lr.
    Behaviour:
      - warmup=0, total=0 → constant 1.0 (no schedule)
      - warmup>0           → linear ramp 0 → 1 over [0, warmup)
      - total>warmup       → cosine decay 1 → min_ratio over [warmup, total)
      - step >= total      → clamps to min_ratio
    """
    if warmup == 0 and total == 0:
        return 1.0
    if warmup > 0 and step < warmup:
        return float(step) / float(warmup)
    if total <= warmup:
        return 1.0
    progress = (step - warmup) / (total - warmup)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


@dataclass
class TrainResult:
    losses: list[float] = field(default_factory=list)
    initial_loss: float = float("nan")
    final_loss: float = float("nan")
    steps: int = 0
    checkpoints_written: list[str] = field(default_factory=list)


def _ckpt_path(ckpt_dir: Path, step: int) -> Path:
    """Filename convention: ckpt_step_{step:06d}.pt (sort-compatible)."""
    return ckpt_dir / f"ckpt_step_{step:06d}.pt"


def _save_checkpoint(
    ckpt_dir: Path,
    step: int,
    model: nn.Module,
    optimizer,
    scheduler,
    cfg: TrainConfig,
    loss_trace_tail: list[float],
) -> Path:
    """Atomically save a checkpoint — write to `.tmp` then `os.replace`.

    Atomic rename guarantees that a reader (resume) never sees a partial
    file even if the writer is SIGKILL'd mid-write. `os.replace` is
    POSIX-atomic across the same filesystem.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_path = _ckpt_path(ckpt_dir, step)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    payload = {
        "step": step,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": asdict(cfg),
        "loss_trace_tail": loss_trace_tail[-50:],
    }
    torch.save(payload, tmp_path)
    os.replace(tmp_path, final_path)
    return final_path


def _rotate_checkpoints(ckpt_dir: Path, keep_last_n: int) -> list[Path]:
    """Delete all but the `keep_last_n` newest `ckpt_step_*.pt` files.

    Sorting by filename works because the step is zero-padded to 6
    digits. Returns the list of deleted paths for logging.
    """
    if keep_last_n <= 0:
        return []
    existing = sorted(ckpt_dir.glob("ckpt_step_*.pt"))
    to_delete = existing[:-keep_last_n] if len(existing) > keep_last_n else []
    for p in to_delete:
        try:
            p.unlink()
        except OSError:
            pass  # best-effort; stale ckpts aren't fatal
    return to_delete


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
    """Run an AdamW training loop with linear-warmup + cosine-decay LR.

    Each batch is a `(B, T)` int64 tensor of token IDs; loss is the
    standard shift-by-one causal LM cross-entropy. Returns the loss
    trace + initial/final values for assertion-friendly inspection.

    LR schedule activates when `config.warmup_steps > 0` or
    `config.total_steps > 0`. With both at 0, behaves identically to
    the C.P2.3 constant-lr loop (regression-safe).

    Does NOT do mixed precision, gradient checkpointing, or QAT —
    those layer in via C.P2.4 + C.P4 separately.
    """
    cfg = config or TrainConfig()
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda s: _lr_lambda(
            s,
            warmup=cfg.warmup_steps,
            total=cfg.total_steps,
            min_ratio=cfg.min_lr_ratio,
        ),
    )
    ckpt_dir_path = Path(cfg.ckpt_dir) if cfg.ckpt_dir else None
    ckpt_enabled = cfg.ckpt_every > 0 and ckpt_dir_path is not None
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
        scheduler.step()

        loss_val = loss.detach().item()
        result.losses.append(loss_val)
        if step == 0:
            result.initial_loss = loss_val
        result.final_loss = loss_val
        result.steps = step + 1

        if cfg.log_every and (step + 1) % cfg.log_every == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"  step {step + 1:5d} loss={loss_val:.4f} lr={current_lr:.2e}")

        if ckpt_enabled and (step + 1) % cfg.ckpt_every == 0:
            path = _save_checkpoint(
                ckpt_dir_path, step + 1, model, optimizer, scheduler, cfg,
                result.losses,
            )
            result.checkpoints_written.append(str(path))
            _rotate_checkpoints(ckpt_dir_path, cfg.keep_last_n_ckpts)

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
