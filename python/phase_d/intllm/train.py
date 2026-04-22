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
import signal
import threading
import time
from collections.abc import Callable, Iterable
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
    # Track B step 2 (V31.C.P6.2 — resume from checkpoint).
    resume_from: str | None = None  # path to ckpt; loads model+optimizer+scheduler+step
    # Track B step 3 (V31.C.P6.3 — step-idle watchdog). 0 → disabled.
    # When >0, a daemon thread fires SIGTERM to the main process if the
    # step counter hasn't advanced for this many seconds. Motivated by
    # the c.1 hang (2026-04-22): training "running" but stuck in a dead
    # HF CDN socket for 8.5h with zero progress.
    watchdog_idle_seconds: int = 0
    watchdog_check_interval: int = 30  # how often the thread polls


class StepWatchdog:
    """Daemon thread that fires a signal if the training step counter
    doesn't advance for `idle_seconds`.

    Usage:
        wd = StepWatchdog(idle_seconds=1800)
        wd.start()
        try:
            for batch in batches:
                ...
                wd.touch()
        finally:
            wd.stop()

    `touch()` must be called once per successful step. During the
    "warmup" phase (before the first `touch()`), the watchdog does NOT
    fire — this covers legitimate startup costs (tokenizer init, model
    load, first CUDA kernel compile) that can take 30-60s.

    `on_fire` defaults to `os.kill(getpid(), SIGTERM)`, which lets the
    process's signal handler (or default terminate behaviour) clean up.
    Tests inject a recording callback instead.

    Rationale (post-c.1 hang): with a 30-min threshold and 60K-step
    Base run @ ~0.8s/step, the watchdog adds zero false-positive risk
    (normal pauses are seconds, not minutes) while capping worst-case
    waste-on-hang at ~30 min instead of 8.5h.
    """

    def __init__(
        self,
        idle_seconds: int,
        check_interval: int = 30,
        on_fire: Callable[[], None] | None = None,
    ) -> None:
        self.idle_seconds = int(idle_seconds)
        self.check_interval = max(1, int(check_interval))
        self._on_fire = on_fire or (lambda: os.kill(os.getpid(), signal.SIGTERM))
        self._last_step_ts: float | None = None
        self._fired = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return self.idle_seconds > 0

    @property
    def fired(self) -> bool:
        return self._fired

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="StepWatchdog"
        )
        self._thread.start()

    def touch(self) -> None:
        """Record that a training step just completed — resets the idle clock."""
        self._last_step_ts = time.time()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.check_interval + 1)
            self._thread = None

    def check_now(self, now: float | None = None) -> bool:
        """Synchronous idle check — exposed for unit tests.

        Returns True iff the watchdog WOULD fire given the current state.
        Does not actually fire; callers are expected to use `start()` for
        production and this method only in tests.
        """
        if not self.enabled or self._fired or self._last_step_ts is None:
            return False
        elapsed = (now or time.time()) - self._last_step_ts
        return elapsed > self.idle_seconds

    def _run(self) -> None:
        while not self._stop.wait(self.check_interval):
            if self._fired or self._last_step_ts is None:
                continue
            idle = time.time() - self._last_step_ts
            if idle > self.idle_seconds:
                self._fired = True
                print(
                    f"[watchdog] training step counter idle {idle:.0f}s "
                    f"(threshold {self.idle_seconds}s) — firing SIGTERM to "
                    f"pid {os.getpid()}",
                    flush=True,
                )
                try:
                    self._on_fire()
                except Exception as e:  # best-effort
                    print(f"[watchdog] on_fire failed: {e}", flush=True)
                return


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


def find_latest_checkpoint(ckpt_dir: str | Path) -> Path | None:
    """Return the highest-step ckpt_step_*.pt in `ckpt_dir`, or None.

    Filename zero-padding guarantees lexicographic sort == numeric sort.
    Used by `--resume-auto` in the training drivers to pick up after
    an interruption without the user having to name the file manually.
    """
    d = Path(ckpt_dir)
    if not d.exists():
        return None
    candidates = sorted(d.glob("ckpt_step_*.pt"))
    return candidates[-1] if candidates else None


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

    # Track B step 2: resume from checkpoint before entering the step
    # loop. LambdaLR's last_epoch is preserved in state_dict → cosine
    # decay continues from the right point. Note: batches iterator is
    # NOT rewound (HF streaming isn't easily skippable) — we just pick
    # up at the next batch. Minor data overlap is fine for LLM pretrain.
    starting_step = 0
    if cfg.resume_from:
        resume_path = Path(cfg.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(
                f"resume checkpoint not found: {resume_path}"
            )
        device = next(model.parameters()).device
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        starting_step = int(ckpt["step"])
        print(f"  resumed from {resume_path.name} at step {starting_step}")

    # Track B step 3: step-idle watchdog. Thread only starts if enabled.
    watchdog = StepWatchdog(
        idle_seconds=cfg.watchdog_idle_seconds,
        check_interval=cfg.watchdog_check_interval,
    )
    watchdog.start()
    if watchdog.enabled:
        print(f"  watchdog armed: SIGTERM if step idle > {cfg.watchdog_idle_seconds}s")

    model.train()

    result = TrainResult()
    result.steps = starting_step  # reflects resume point even if 0 batches consumed
    try:
        for step_idx, ids in enumerate(batches):
            current_step = starting_step + step_idx + 1
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
            if step_idx == 0:
                result.initial_loss = loss_val
            result.final_loss = loss_val
            result.steps = current_step
            watchdog.touch()

            if cfg.log_every and current_step % cfg.log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  step {current_step:5d} loss={loss_val:.4f} lr={current_lr:.2e}")

            if ckpt_enabled and current_step % cfg.ckpt_every == 0:
                path = _save_checkpoint(
                    ckpt_dir_path, current_step, model, optimizer, scheduler, cfg,
                    result.losses,
                )
                result.checkpoints_written.append(str(path))
                _rotate_checkpoints(ckpt_dir_path, cfg.keep_last_n_ckpts)
    finally:
        watchdog.stop()

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
