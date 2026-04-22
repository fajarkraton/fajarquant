"""Training-loop proof-of-life tests (C.P2.3).

Verifies that `intllm.model + intllm.data + intllm.train` compose into
a working end-to-end training stack. The gate is "loss drops
substantially on an overfit-a-single-batch test" — the simplest
possible signal that the stack isn't silently broken.

These tests run in ~10 s on GPU, ~30 s on CPU. They skip if CUDA isn't
available on machines where the upstream mmfreelm Triton kernels won't
build (CPU-only).
"""

from __future__ import annotations

import threading
import time

import pytest
import torch

from intllm.data import overfit_token_batches, synthetic_token_batches
from intllm.model import HGRNBitConfig, HGRNBitForCausalLM
from intllm.train import (
    StepWatchdog,
    TrainConfig,
    _lr_lambda,
    find_latest_checkpoint,
    smoothed_min,
    train_loop,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="mmfreelm Triton kernels require CUDA",
)

# Tiny config — 165K params, trains in ~10 s on RTX 4090 Laptop.
_TINY_VOCAB = 256
_TINY_HIDDEN = 64
_TINY_LAYERS = 2
_TINY_SEQLEN = 64
_BATCH = 8


def _make_tiny_model(device: str = "cuda") -> HGRNBitForCausalLM:
    cfg = HGRNBitConfig(
        vocab_size=_TINY_VOCAB,
        hidden_size=_TINY_HIDDEN,
        num_hidden_layers=_TINY_LAYERS,
        max_position_embeddings=_TINY_SEQLEN,
    )
    return HGRNBitForCausalLM(cfg).to(device)


def test_training_loop_overfits_a_single_batch() -> None:
    """Smoke test: model memorizes a fixed synthetic batch.

    Gate: loss drops ≥80% from random-init (`ln(V) ≈ 5.55`) over 200
    steps. 80% is deliberately loose — if the stack works at all the
    observed drop is >99%, and the test goal is catching "silently
    broken" not "perfectly trained."
    """
    torch.manual_seed(0)
    model = _make_tiny_model()
    batches = overfit_token_batches(
        vocab_size=_TINY_VOCAB,
        seq_len=_TINY_SEQLEN,
        batch_size=_BATCH,
        n_batches=200,
        seed=0,
        device="cuda",
    )
    result = train_loop(model, batches, config=TrainConfig(lr=3e-3, log_every=0))

    assert result.steps == 200
    assert torch.isfinite(torch.tensor(result.final_loss)).item(), (
        f"final loss not finite: {result.final_loss}"
    )

    floor = smoothed_min(result.losses)
    drop_pct = 100.0 * (result.initial_loss - floor) / result.initial_loss
    assert drop_pct >= 80.0, (
        f"loss only dropped {drop_pct:.1f}%; initial={result.initial_loss:.3f} floor={floor:.3f}"
    )


def test_training_loop_stays_finite_on_random_batches() -> None:
    """Stack-stability test: on random tokens (no learnable signal),
    loss hovers near `ln(V)` ≈ 5.55 but NEVER goes NaN/Inf.

    This catches a different class of bugs than the overfit test —
    e.g., unstable quantization + numerical saturation in the MLGRU
    state update would manifest as NaN over enough steps.
    """
    torch.manual_seed(1)
    model = _make_tiny_model()
    batches = synthetic_token_batches(
        vocab_size=_TINY_VOCAB,
        seq_len=_TINY_SEQLEN,
        batch_size=_BATCH,
        n_batches=50,
        seed=42,
        device="cuda",
    )
    result = train_loop(model, batches, config=TrainConfig(lr=1e-3, log_every=0))

    assert result.steps == 50
    assert all(
        torch.isfinite(torch.tensor(x)).item() for x in result.losses
    ), "non-finite loss during random-token training"
    import math

    expected = math.log(_TINY_VOCAB)
    # Loss should stay within ±1 of ln(V) — random tokens have no
    # learnable signal, so big deviations mean instability.
    assert all(abs(x - expected) < 1.0 for x in result.losses[-10:]), (
        f"random-token loss drifted; last 10 = {result.losses[-10:]}"
    )


# -----------------------------------------------------------------
# LR scheduler (C.P4.1 H1 fix)
# -----------------------------------------------------------------

def test_lr_lambda_no_schedule_when_both_zero() -> None:
    """warmup=0, total=0 → constant 1.0 (regression-safe default)."""
    for s in (0, 100, 10_000):
        assert _lr_lambda(s, warmup=0, total=0, min_ratio=0.1) == 1.0


def test_lr_lambda_linear_warmup() -> None:
    """warmup_steps=1000 → linear ramp 0 → 1 over [0, 1000)."""
    assert _lr_lambda(0, warmup=1000, total=2000, min_ratio=0.1) == 0.0
    assert _lr_lambda(500, warmup=1000, total=2000, min_ratio=0.1) == 0.5
    # at warmup boundary, ramp completes; cosine starts at 1.0
    assert abs(_lr_lambda(1000, warmup=1000, total=2000, min_ratio=0.1) - 1.0) < 1e-6


def test_lr_lambda_cosine_decay_to_min_ratio() -> None:
    """At total_steps, lr should equal min_ratio (clamped)."""
    val = _lr_lambda(2000, warmup=1000, total=2000, min_ratio=0.1)
    assert abs(val - 0.1) < 1e-6
    # Past total — clamp
    val = _lr_lambda(5000, warmup=1000, total=2000, min_ratio=0.1)
    assert abs(val - 0.1) < 1e-6


def test_lr_lambda_midpoint_decay() -> None:
    """Halfway through cosine decay → average of 1.0 and min_ratio."""
    # warmup=0, total=1000, midpoint 500
    val = _lr_lambda(500, warmup=0, total=1000, min_ratio=0.1)
    expected = 0.1 + (1.0 - 0.1) * 0.5  # cos(π/2)=0 → 0.5*(1+0)=0.5 cosine
    assert abs(val - expected) < 1e-6


# -----------------------------------------------------------------
# Track B step 1 (V31.C.P6.1) — interruption-safe checkpointing
# -----------------------------------------------------------------

def test_ckpt_disabled_by_default(tmp_path) -> None:
    """ckpt_every=0 → zero files written, backward-compatible."""
    torch.manual_seed(0)
    model = _make_tiny_model()
    batches = overfit_token_batches(
        vocab_size=_TINY_VOCAB, seq_len=_TINY_SEQLEN, batch_size=_BATCH,
        n_batches=20, seed=0, device="cuda",
    )
    result = train_loop(model, batches, config=TrainConfig(
        lr=1e-3, log_every=0, ckpt_every=0, ckpt_dir=str(tmp_path),
    ))
    assert result.checkpoints_written == []
    assert list(tmp_path.glob("ckpt_*.pt")) == []


def test_ckpt_written_every_n_steps(tmp_path) -> None:
    """ckpt_every=5, 15 steps → 3 checkpoints at steps 5, 10, 15."""
    torch.manual_seed(0)
    model = _make_tiny_model()
    batches = overfit_token_batches(
        vocab_size=_TINY_VOCAB, seq_len=_TINY_SEQLEN, batch_size=_BATCH,
        n_batches=15, seed=0, device="cuda",
    )
    result = train_loop(model, batches, config=TrainConfig(
        lr=1e-3, log_every=0, ckpt_every=5,
        ckpt_dir=str(tmp_path), keep_last_n_ckpts=10,
    ))
    assert result.steps == 15
    written = sorted(tmp_path.glob("ckpt_step_*.pt"))
    assert len(written) == 3, f"expected 3 ckpts, got {[p.name for p in written]}"
    assert [p.name for p in written] == [
        "ckpt_step_000005.pt", "ckpt_step_000010.pt", "ckpt_step_000015.pt",
    ]


def test_ckpt_payload_has_required_keys(tmp_path) -> None:
    """Loaded checkpoint must carry step + state_dict + optimizer +
    scheduler + config — everything needed by --resume (step 2)."""
    torch.manual_seed(0)
    model = _make_tiny_model()
    batches = overfit_token_batches(
        vocab_size=_TINY_VOCAB, seq_len=_TINY_SEQLEN, batch_size=_BATCH,
        n_batches=5, seed=0, device="cuda",
    )
    train_loop(model, batches, config=TrainConfig(
        lr=1e-3, log_every=0, ckpt_every=5, ckpt_dir=str(tmp_path),
    ))
    ckpt_path = tmp_path / "ckpt_step_000005.pt"
    assert ckpt_path.exists()
    payload = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    for key in ("step", "state_dict", "optimizer", "scheduler", "config"):
        assert key in payload, f"missing key: {key}"
    assert payload["step"] == 5
    assert isinstance(payload["state_dict"], dict)
    assert payload["config"]["ckpt_every"] == 5


def test_ckpt_rotation_keeps_only_last_n(tmp_path) -> None:
    """keep_last_n_ckpts=2, 6 save events → only 2 newest remain on disk."""
    torch.manual_seed(0)
    model = _make_tiny_model()
    batches = overfit_token_batches(
        vocab_size=_TINY_VOCAB, seq_len=_TINY_SEQLEN, batch_size=_BATCH,
        n_batches=12, seed=0, device="cuda",
    )
    train_loop(model, batches, config=TrainConfig(
        lr=1e-3, log_every=0, ckpt_every=2,
        ckpt_dir=str(tmp_path), keep_last_n_ckpts=2,
    ))
    remaining = sorted(tmp_path.glob("ckpt_step_*.pt"))
    # 12 steps / 2 = 6 save events, keep last 2 → steps 10 and 12 remain
    assert [p.name for p in remaining] == [
        "ckpt_step_000010.pt", "ckpt_step_000012.pt",
    ]


def test_ckpt_atomic_no_tmp_files_on_disk(tmp_path) -> None:
    """Atomic-rename invariant: after a clean run, zero .tmp files remain.

    This doesn't test the SIGKILL-mid-write case directly (hard to fake
    in-process), but asserts the clean path doesn't leak.
    """
    torch.manual_seed(0)
    model = _make_tiny_model()
    batches = overfit_token_batches(
        vocab_size=_TINY_VOCAB, seq_len=_TINY_SEQLEN, batch_size=_BATCH,
        n_batches=5, seed=0, device="cuda",
    )
    train_loop(model, batches, config=TrainConfig(
        lr=1e-3, log_every=0, ckpt_every=5, ckpt_dir=str(tmp_path),
    ))
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == [], f"leaked tmp files: {tmp_files}"


# -----------------------------------------------------------------
# Track B step 2 (V31.C.P6.2) — --resume from checkpoint
# -----------------------------------------------------------------

def test_find_latest_checkpoint_returns_highest_step(tmp_path) -> None:
    """find_latest_checkpoint returns the largest step, not just newest mtime."""
    # Create files in REVERSE step order so mtime-sort would give wrong answer
    for step in (15, 5, 10):
        (tmp_path / f"ckpt_step_{step:06d}.pt").write_bytes(b"dummy")
    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "ckpt_step_000015.pt"


def test_find_latest_checkpoint_none_when_empty(tmp_path) -> None:
    """Empty dir → None (so --resume-auto can fall back to fresh start)."""
    assert find_latest_checkpoint(tmp_path) is None
    # Non-existent dir also returns None (not FileNotFoundError)
    assert find_latest_checkpoint(tmp_path / "does_not_exist") is None


def test_resume_from_checkpoint_continues_step_counter(tmp_path) -> None:
    """After resume, TrainResult.steps reflects starting + new steps.

    E.g. resume from step 5 ckpt, run 3 more steps → result.steps == 8.
    """
    torch.manual_seed(0)
    model = _make_tiny_model()
    batches_a = overfit_token_batches(
        vocab_size=_TINY_VOCAB, seq_len=_TINY_SEQLEN, batch_size=_BATCH,
        n_batches=5, seed=0, device="cuda",
    )
    train_loop(model, batches_a, config=TrainConfig(
        lr=1e-3, log_every=0, ckpt_every=5, ckpt_dir=str(tmp_path),
    ))
    ckpt = tmp_path / "ckpt_step_000005.pt"
    assert ckpt.exists()

    # Fresh model, resume from the ckpt, train 3 more steps
    model2 = _make_tiny_model()
    batches_b = overfit_token_batches(
        vocab_size=_TINY_VOCAB, seq_len=_TINY_SEQLEN, batch_size=_BATCH,
        n_batches=3, seed=1, device="cuda",
    )
    result2 = train_loop(model2, batches_b, config=TrainConfig(
        lr=1e-3, log_every=0, resume_from=str(ckpt),
    ))
    assert result2.steps == 8, f"expected step 8 after resume+3, got {result2.steps}"


def test_resume_preserves_model_state(tmp_path) -> None:
    """After resume, the model must evaluate bit-exactly the same as
    right before the checkpoint was saved. Strongest possible test —
    any load bug (wrong keys, device mismatch, missing buffers) fails.
    """
    torch.manual_seed(0)
    model = _make_tiny_model()
    batches = overfit_token_batches(
        vocab_size=_TINY_VOCAB, seq_len=_TINY_SEQLEN, batch_size=_BATCH,
        n_batches=30, seed=0, device="cuda",
    )
    # Probe = the training batch itself (overfit_token_batches is
    # deterministic by seed — we can reconstruct it).
    probe = next(iter(overfit_token_batches(
        vocab_size=_TINY_VOCAB, seq_len=_TINY_SEQLEN, batch_size=_BATCH,
        n_batches=1, seed=0, device="cuda",
    )))

    # Train 30 steps and checkpoint at step 30.
    train_loop(model, batches, config=TrainConfig(
        lr=3e-3, log_every=0, ckpt_every=30, ckpt_dir=str(tmp_path),
    ))
    model.eval()
    with torch.no_grad():
        loss_pre_save = model(input_ids=probe, labels=probe).loss.item()

    # Fresh random model, resume from the checkpoint, zero extra steps.
    model2 = _make_tiny_model()
    train_loop(model2, iter([]), config=TrainConfig(
        lr=3e-3, log_every=0,
        resume_from=str(tmp_path / "ckpt_step_000030.pt"),
    ))
    model2.eval()
    with torch.no_grad():
        loss_post_resume = model2(input_ids=probe, labels=probe).loss.item()

    # Bit-exact within FP numerical noise — no training occurred in model2
    # beyond the load, so weights should match exactly.
    assert abs(loss_post_resume - loss_pre_save) < 1e-4, (
        f"resume drift: pre-save={loss_pre_save:.6f} post-resume={loss_post_resume:.6f}"
    )


def test_resume_missing_file_raises(tmp_path) -> None:
    """Bad --resume path raises FileNotFoundError with a clear message."""
    model = _make_tiny_model()
    batches = overfit_token_batches(
        vocab_size=_TINY_VOCAB, seq_len=_TINY_SEQLEN, batch_size=_BATCH,
        n_batches=1, seed=0, device="cuda",
    )
    bogus = tmp_path / "does_not_exist.pt"
    with pytest.raises(FileNotFoundError, match="resume checkpoint not found"):
        train_loop(model, batches, config=TrainConfig(
            lr=1e-3, log_every=0, resume_from=str(bogus),
        ))


# -----------------------------------------------------------------
# Track B step 3 (V31.C.P6.3) — StepWatchdog
# -----------------------------------------------------------------

def test_watchdog_disabled_when_idle_seconds_zero() -> None:
    """idle_seconds=0 → start() is a no-op, enabled is False."""
    fired = []
    wd = StepWatchdog(idle_seconds=0, on_fire=lambda: fired.append(True))
    assert not wd.enabled
    wd.start()  # must not crash; no thread spawned
    assert wd._thread is None
    wd.stop()
    assert fired == []


def test_watchdog_does_not_fire_during_warmup() -> None:
    """Before first touch(), watchdog never fires — lets model load etc."""
    fired = []
    wd = StepWatchdog(idle_seconds=1, on_fire=lambda: fired.append(True))
    # No touch() → _last_step_ts is None → check_now returns False regardless.
    import time
    time.sleep(1.2)
    assert wd.check_now() is False
    assert fired == []


def test_watchdog_check_now_reports_idle_past_threshold() -> None:
    """After touch(), check_now() returns True iff elapsed > threshold."""
    wd = StepWatchdog(idle_seconds=5)
    wd.touch()
    now_ts = wd._last_step_ts
    # Within threshold
    assert wd.check_now(now=now_ts + 3) is False
    # Past threshold
    assert wd.check_now(now=now_ts + 10) is True


def test_watchdog_touch_resets_idle_clock() -> None:
    """touch() resets the idle clock — long-running steps shouldn't fire
    as long as each step eventually calls touch()."""
    wd = StepWatchdog(idle_seconds=5)
    wd.touch()
    t0 = wd._last_step_ts
    # Simulate a 4s step → still under threshold
    assert wd.check_now(now=t0 + 4) is False
    # Step completes, touch again
    wd.touch()
    t1 = wd._last_step_ts
    # 4s after the NEW touch → still fine
    assert wd.check_now(now=t1 + 4) is False


def test_watchdog_daemon_thread_fires_on_idle() -> None:
    """Real thread test: start watchdog with 0.3s threshold + 0.1s poll;
    never touch; within ~0.5s on_fire should be called."""
    import time
    fired = threading.Event()
    wd = StepWatchdog(
        idle_seconds=0,  # placeholder; override below
        check_interval=0,  # placeholder; override below
        on_fire=lambda: fired.set(),
    )
    # Force enabled with a tiny threshold for testability.
    wd.idle_seconds = 1
    wd.check_interval = 0  # poll as fast as possible; actual wait uses max(1, interval)
    # touch() BEFORE start() so the _run loop has a baseline.
    wd.touch()
    # Wind back the clock so the thread sees "idle > 1s" immediately.
    wd._last_step_ts = time.time() - 5
    wd.check_interval = 1  # Event.wait min is the arg; set to 1s
    wd.start()
    # Expect fire within 2 poll cycles (~2s).
    assert fired.wait(timeout=4), "watchdog did not fire within 4s"
    assert wd.fired
    wd.stop()


def test_watchdog_fires_only_once() -> None:
    """After firing, further check_now calls return False — single-shot."""
    wd = StepWatchdog(idle_seconds=1)
    wd.touch()
    wd._last_step_ts = time.time() - 10
    wd._fired = True  # pretend it already fired
    assert wd.check_now() is False


def test_watchdog_default_on_fire_sends_sigterm_to_own_process() -> None:
    """End-to-end signal-delivery test. Uses the DEFAULT on_fire (which
    is os.kill(getpid(), SIGTERM)) — installs a SIGTERM handler so the
    test runner doesn't actually terminate. Proves the watchdog's real
    production code path works, not just the injected callback in the
    other tests. This is the §6.8 R3 prevention regression.
    """
    import signal
    sigterm_received = threading.Event()

    def _handler(signum, frame):
        sigterm_received.set()

    old = signal.signal(signal.SIGTERM, _handler)
    try:
        wd = StepWatchdog(idle_seconds=1, check_interval=1)  # default on_fire
        wd.touch()
        wd._last_step_ts = time.time() - 5  # already idle past threshold
        wd.start()
        assert sigterm_received.wait(timeout=4), (
            "SIGTERM was not delivered within 4s via default on_fire path"
        )
        wd.stop()
    finally:
        signal.signal(signal.SIGTERM, old)


def test_train_loop_respects_watchdog_idle_seconds_zero(tmp_path) -> None:
    """Regression: watchdog_idle_seconds=0 in TrainConfig must not spawn
    a thread or interfere with normal training (backward-compat)."""
    torch.manual_seed(0)
    model = _make_tiny_model()
    batches = overfit_token_batches(
        vocab_size=_TINY_VOCAB, seq_len=_TINY_SEQLEN, batch_size=_BATCH,
        n_batches=10, seed=0, device="cuda",
    )
    result = train_loop(model, batches, config=TrainConfig(
        lr=1e-3, log_every=0, watchdog_idle_seconds=0,
    ))
    assert result.steps == 10


def test_resumed_ckpts_named_with_true_total_step(tmp_path) -> None:
    """When resuming from step 5 + training 10 more steps with ckpt_every=5,
    new ckpts are named by TRUE total step (10, 15) — not relative step."""
    torch.manual_seed(0)
    model = _make_tiny_model()
    batches_a = overfit_token_batches(
        vocab_size=_TINY_VOCAB, seq_len=_TINY_SEQLEN, batch_size=_BATCH,
        n_batches=5, seed=0, device="cuda",
    )
    train_loop(model, batches_a, config=TrainConfig(
        lr=1e-3, log_every=0, ckpt_every=5, ckpt_dir=str(tmp_path),
        keep_last_n_ckpts=10,
    ))
    # Resume from step 5, train 10 more → should write ckpts at 10 and 15.
    model2 = _make_tiny_model()
    batches_b = overfit_token_batches(
        vocab_size=_TINY_VOCAB, seq_len=_TINY_SEQLEN, batch_size=_BATCH,
        n_batches=10, seed=1, device="cuda",
    )
    train_loop(model2, batches_b, config=TrainConfig(
        lr=1e-3, log_every=0, ckpt_every=5, ckpt_dir=str(tmp_path),
        keep_last_n_ckpts=10,
        resume_from=str(tmp_path / "ckpt_step_000005.pt"),
    ))
    all_ckpts = sorted(p.name for p in tmp_path.glob("ckpt_step_*.pt"))
    assert all_ckpts == [
        "ckpt_step_000005.pt",
        "ckpt_step_000010.pt",
        "ckpt_step_000015.pt",
    ]
