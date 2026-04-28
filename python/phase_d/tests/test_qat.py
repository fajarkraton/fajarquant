"""Tests for `intllm.qat` — verify hooks attach to upstream BitLinear,
stats accumulate correctly, bit-allocation + permutation respect spec.
"""

from __future__ import annotations

import pytest
import torch

from intllm.data import overfit_token_batches
from intllm.model import HGRNBitConfig, HGRNBitForCausalLM
from intllm.qat import (
    BitLinearStatTracker,
    QATConfig,
    advance_trackers,
    attach_stat_trackers,
    compute_bit_allocation,
    compute_channel_permutation,
    detach_stat_trackers,
    save_calibration_maps,
)
from intllm.train import TrainConfig, train_loop


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="mmfreelm Triton kernels require CUDA",
)


def _make_tiny_model() -> HGRNBitForCausalLM:
    cfg = HGRNBitConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        max_position_embeddings=64,
    )
    return HGRNBitForCausalLM(cfg).to("cuda")


# -----------------------------------------------------------------
# QATConfig
# -----------------------------------------------------------------

def test_qat_config_defaults_are_safe() -> None:
    """Default QATConfig has all features OFF — pure BitNet baseline.
    Critical for ablation Table 4 reproducibility."""
    cfg = QATConfig()
    assert cfg.enable_adaptive_bits is False
    assert cfg.enable_channel_permutation is False
    assert cfg.enable_periodic_recal is False


# -----------------------------------------------------------------
# BitLinearStatTracker
# -----------------------------------------------------------------

def test_stat_tracker_running_max_is_elementwise() -> None:
    tracker = BitLinearStatTracker(in_features=4)
    fake_module = torch.nn.Linear(4, 4)  # not a BitLinear, just a placeholder
    # Simulate two forward calls with different per-channel magnitudes
    x1 = torch.tensor([[1.0, -2.0, 3.0, -0.5]])
    x2 = torch.tensor([[-0.5, 4.0, 1.0, -2.0]])
    tracker(fake_module, (x1,), torch.empty(0))
    tracker(fake_module, (x2,), torch.empty(0))
    expected = torch.tensor([1.0, 4.0, 3.0, 2.0])
    assert torch.allclose(tracker.running_max, expected), (
        f"expected {expected.tolist()}, got {tracker.running_max.tolist()}"
    )
    assert tracker.n_calls == 2


def test_stat_tracker_rejects_dim_mismatch() -> None:
    tracker = BitLinearStatTracker(in_features=4)
    fake = torch.nn.Linear(4, 4)
    with pytest.raises(ValueError, match="channel-dim mismatch"):
        tracker(fake, (torch.randn(1, 8),), torch.empty(0))


# -----------------------------------------------------------------
# attach_stat_trackers
# -----------------------------------------------------------------

def test_attach_finds_bitlinear_layers_in_hgrn_bit_model() -> None:
    """Attaching to a tiny HGRNBit model should find ≥4 BitLinear sites:
    upstream's HGRN architecture has time-mix + channel-mix BitLinears
    per layer. With L=2 we expect a multiple of 4-7 hooks."""
    model = _make_tiny_model()
    trackers = attach_stat_trackers(model)
    n_layers = model.config.num_hidden_layers
    assert len(trackers) >= n_layers * 4, (
        f"expected ≥{n_layers * 4} BitLinear sites in L={n_layers} model, got {len(trackers)}"
    )
    # All trackers start fresh
    for name, t in trackers.items():
        assert t.n_calls == 0, f"tracker {name} pre-populated with n_calls={t.n_calls}"
    detach_stat_trackers(model)


def test_attach_then_train_records_statistics() -> None:
    """After a few training steps, hooks should have accumulated calls
    and non-zero running_max values."""
    model = _make_tiny_model()
    trackers = attach_stat_trackers(model)

    batches = overfit_token_batches(
        vocab_size=256, seq_len=32, batch_size=4, n_batches=10, device="cuda"
    )
    train_loop(model, batches, config=TrainConfig(lr=1e-3, log_every=0))

    for name, t in trackers.items():
        assert t.n_calls >= 10, f"{name}.n_calls = {t.n_calls}; expected ≥10"
        assert (t.running_max > 0).any(), (
            f"{name}.running_max stayed all-zero — hook never saw nonzero input"
        )
    detach_stat_trackers(model)


# -----------------------------------------------------------------
# compute_bit_allocation
# -----------------------------------------------------------------

def test_bit_allocation_assigns_high_bits_to_top_k() -> None:
    tracker = BitLinearStatTracker(in_features=100)
    # Construct a known-skewed running_max: 5 channels are 10x larger
    rm = torch.full((100,), 1.0)
    rm[[3, 17, 42, 67, 88]] = 10.0
    tracker.running_max = rm
    tracker.n_calls = 1  # bypass the "no observations" guard

    bits = compute_bit_allocation(tracker, top_k_pct=5.0, low_bits=8, high_bits=10)
    assert bits.shape == (100,)
    assert bits.dtype == torch.int32
    # Exactly 5 channels (5% of 100) should be at high_bits
    assert (bits == 10).sum().item() == 5
    assert (bits == 8).sum().item() == 95
    # And those 5 should be the ones we made larger
    expected_top = torch.tensor([3, 17, 42, 67, 88])
    actual_top = torch.where(bits == 10)[0].sort().values
    assert torch.equal(actual_top, expected_top)


def test_bit_allocation_handles_small_n_channels() -> None:
    """With <20 channels, top_k=5% rounds to 1 channel (max(1, ...))."""
    tracker = BitLinearStatTracker(in_features=10)
    tracker.running_max = torch.tensor([0.1, 5.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    tracker.n_calls = 1
    bits = compute_bit_allocation(tracker, top_k_pct=5.0, low_bits=8, high_bits=10)
    assert (bits == 10).sum().item() == 1  # only the highest (5.0)
    assert torch.where(bits == 10)[0].item() == 1


def test_bit_allocation_rejects_empty_tracker() -> None:
    tracker = BitLinearStatTracker(in_features=10)
    with pytest.raises(ValueError, match="no observations"):
        compute_bit_allocation(tracker)


# -----------------------------------------------------------------
# compute_channel_permutation
# -----------------------------------------------------------------

def test_channel_permutation_sorts_by_running_max_desc() -> None:
    tracker = BitLinearStatTracker(in_features=4)
    tracker.running_max = torch.tensor([0.5, 3.0, 1.0, 2.0])
    tracker.n_calls = 1
    perm = compute_channel_permutation(tracker)
    # Expected: indices sorted by value descending → [1 (=3.0), 3 (=2.0), 2 (=1.0), 0 (=0.5)]
    assert torch.equal(perm, torch.tensor([1, 3, 2, 0]))


# -----------------------------------------------------------------
# Convergence still works with hooks attached (regression guard)
# -----------------------------------------------------------------

def test_qat_hooks_do_not_break_training_convergence() -> None:
    """Critical regression check: attaching the QAT stat trackers should
    not interfere with the upstream forward pass. Training must still
    converge on the overfit-batch test."""
    model = _make_tiny_model()
    attach_stat_trackers(model)
    batches = overfit_token_batches(
        vocab_size=256, seq_len=64, batch_size=8, n_batches=200, device="cuda"
    )
    result = train_loop(model, batches, config=TrainConfig(lr=3e-3, log_every=0))
    drop_pct = 100.0 * (result.initial_loss - result.final_loss) / result.initial_loss
    assert drop_pct >= 80.0, (
        f"QAT-hooked model only dropped {drop_pct:.1f}%; init={result.initial_loss:.3f} "
        f"final={result.final_loss:.3f}"
    )
    detach_stat_trackers(model)


# -----------------------------------------------------------------
# E2.4.A.2 — save_calibration_maps
# -----------------------------------------------------------------

def test_save_calibration_maps_writes_well_formed_pt(tmp_path) -> None:
    """Round-trip: build trackers, populate, save, reload, verify shape."""
    # Two synthetic trackers — different in_features to exercise the
    # heterogeneous-layer case (real models have varying widths too).
    t1 = BitLinearStatTracker(in_features=8)
    t1.running_max = torch.tensor([1.0, 5.0, 0.5, 3.0, 0.1, 4.0, 2.0, 0.8])
    t1.n_calls = 100
    t2 = BitLinearStatTracker(in_features=4)
    t2.running_max = torch.tensor([2.0, 0.1, 1.5, 0.05])
    t2.n_calls = 50

    out_path = tmp_path / "test_maps.pt"
    payload = save_calibration_maps(
        {"layer_a": t1, "layer_b": t2},
        out_path,
        top_k_pct=25.0,  # 25% of 8 = 2 channels high; 25% of 4 = 1 channel high
        low_bits=8, high_bits=10,
    )

    # In-memory return value matches what was written.
    loaded = torch.load(out_path, weights_only=False)
    assert set(loaded.keys()) == {"layer_a", "layer_b", "_meta"}
    assert loaded["_meta"]["n_layers"] == 2
    assert loaded["_meta"]["top_k_pct"] == 25.0
    assert loaded["_meta"]["low_bits"] == 8
    assert loaded["_meta"]["high_bits"] == 10

    # layer_a: top 2 by running_max are channels 1 (=5.0) and 5 (=4.0) → 10-bit
    bits_a = loaded["layer_a"]["bits"]
    assert bits_a.dtype == torch.int32
    assert bits_a.shape == (8,)
    assert int(bits_a[1]) == 10 and int(bits_a[5]) == 10
    # All others 8-bit
    assert int(bits_a[0]) == 8 and int(bits_a[7]) == 8

    # layer_a: permutation sorts indices by descending running_max
    # → [1 (5.0), 5 (4.0), 3 (3.0), 6 (2.0), 0 (1.0), 7 (0.8), 2 (0.5), 4 (0.1)]
    expected_perm_a = torch.tensor([1, 5, 3, 6, 0, 7, 2, 4])
    assert torch.equal(loaded["layer_a"]["permutation"], expected_perm_a)

    # n_calls preserved
    assert loaded["layer_a"]["n_calls"] == 100
    assert loaded["layer_b"]["n_calls"] == 50

    # Return value structure matches loaded structure
    assert payload["_meta"]["n_layers"] == 2


def test_save_calibration_maps_extra_meta_merged(tmp_path) -> None:
    """`extra_meta` keys land in the `_meta` block alongside defaults."""
    t = BitLinearStatTracker(in_features=4)
    t.running_max = torch.tensor([1.0, 2.0, 3.0, 4.0])
    t.n_calls = 10

    out_path = tmp_path / "maps_with_meta.pt"
    save_calibration_maps(
        {"layer": t}, out_path,
        extra_meta={"tag": "balanced_calib", "id_share": 0.6, "n_steps": 24000},
    )
    loaded = torch.load(out_path, weights_only=False)
    assert loaded["_meta"]["tag"] == "balanced_calib"
    assert loaded["_meta"]["id_share"] == 0.6
    assert loaded["_meta"]["n_steps"] == 24000
    # Defaults still present
    assert loaded["_meta"]["top_k_pct"] == 5.0
    assert loaded["_meta"]["_schema_version"] == "1.2"


def test_save_calibration_maps_empty_dict_raises(tmp_path) -> None:
    """No trackers → ValueError before any disk write."""
    with pytest.raises(ValueError, match="trackers dict is empty"):
        save_calibration_maps({}, tmp_path / "empty.pt")
    assert not (tmp_path / "empty.pt").exists()


def test_save_calibration_maps_zero_observations_raises(tmp_path) -> None:
    """Tracker with n_calls=0 → ValueError; partial save not committed."""
    fresh = BitLinearStatTracker(in_features=4)  # n_calls = 0
    populated = BitLinearStatTracker(in_features=4)
    populated.running_max = torch.tensor([1.0, 2.0, 3.0, 4.0])
    populated.n_calls = 5

    out_path = tmp_path / "partial.pt"
    with pytest.raises(ValueError, match="0 observations"):
        save_calibration_maps(
            {"populated": populated, "fresh": fresh}, out_path,
        )
    # Atomic write: tmp file should not be left behind on failure path.
    # (We raise before torch.save, so .pt and .pt.tmp both absent.)
    assert not out_path.exists()
    assert not out_path.with_suffix(".pt.tmp").exists()


def test_save_calibration_maps_atomic_write(tmp_path) -> None:
    """Successful save → final file exists, .tmp does not."""
    t = BitLinearStatTracker(in_features=4)
    t.running_max = torch.tensor([1.0, 2.0, 3.0, 4.0])
    t.n_calls = 1

    out_path = tmp_path / "atomic.pt"
    save_calibration_maps({"l": t}, out_path)
    assert out_path.exists()
    # The .tmp must have been os.replace'd to the final name.
    assert not out_path.with_suffix(".pt.tmp").exists()


# -----------------------------------------------------------------
# F.5.3 — skip-warmup calibration
# -----------------------------------------------------------------

def test_stat_tracker_skips_calls_before_warmup_boundary() -> None:
    """With start_recording_at_step=W, the first W forward calls (while
    current_step < W) are dropped: n_skipped accumulates, n_calls and
    running_max stay at their initial values."""
    tracker = BitLinearStatTracker(in_features=4, start_recording_at_step=5)
    fake = torch.nn.Linear(4, 4)
    big = torch.tensor([[100.0, 100.0, 100.0, 100.0]])

    # current_step starts at 0 → 5 calls below the boundary all skip
    for step in range(5):
        tracker.current_step = step
        tracker(fake, (big,), torch.empty(0))
    assert tracker.n_calls == 0
    assert tracker.n_skipped == 5
    assert torch.all(tracker.running_max == 0.0), (
        "running_max must remain at init while warmup is active; "
        f"got {tracker.running_max.tolist()}"
    )

    # current_step = 5 reaches the boundary → record
    tracker.current_step = 5
    small = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    tracker(fake, (small,), torch.empty(0))
    assert tracker.n_calls == 1
    assert tracker.n_skipped == 5
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert torch.allclose(tracker.running_max, expected), (
        "post-warmup running_max should reflect ONLY post-boundary inputs; "
        f"got {tracker.running_max.tolist()}"
    )


def test_stat_tracker_default_warmup_is_zero_legacy_behavior() -> None:
    """No-arg construction matches legacy behavior: every call records."""
    tracker = BitLinearStatTracker(in_features=4)
    assert tracker.start_recording_at_step == 0
    fake = torch.nn.Linear(4, 4)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    # Don't touch current_step → stays 0 → 0 >= 0 records.
    tracker(fake, (x,), torch.empty(0))
    assert tracker.n_calls == 1
    assert tracker.n_skipped == 0


def test_stat_tracker_negative_warmup_rejected() -> None:
    with pytest.raises(ValueError, match="start_recording_at_step must be ≥ 0"):
        BitLinearStatTracker(in_features=4, start_recording_at_step=-1)


def test_advance_trackers_increments_all() -> None:
    a = BitLinearStatTracker(in_features=4, start_recording_at_step=10)
    b = BitLinearStatTracker(in_features=4, start_recording_at_step=10)
    c = BitLinearStatTracker(in_features=8, start_recording_at_step=10)
    trackers = {"a": a, "b": b, "c": c}

    for _ in range(7):
        advance_trackers(trackers)
    assert a.current_step == 7
    assert b.current_step == 7
    assert c.current_step == 7


def test_advance_trackers_resets_to_global_step() -> None:
    """Resume-from-checkpoint path: re-sync all trackers to a known step."""
    a = BitLinearStatTracker(in_features=4, start_recording_at_step=100)
    b = BitLinearStatTracker(in_features=4, start_recording_at_step=100)
    a.current_step = 3
    b.current_step = 5
    advance_trackers({"a": a, "b": b}, global_step=200)
    assert a.current_step == 200
    assert b.current_step == 200


def test_advance_trackers_negative_global_step_rejected() -> None:
    a = BitLinearStatTracker(in_features=4)
    with pytest.raises(ValueError, match="global_step must be ≥ 0"):
        advance_trackers({"a": a}, global_step=-1)


def test_save_calibration_maps_records_warmup_in_meta(tmp_path) -> None:
    """F.5.3 audit trail: _meta.start_recording_at_step_{min,max} reflect
    the warmup setting; per-layer entries carry n_skipped + the per-tracker
    threshold so the consumer can reconstruct what was dropped."""
    t1 = BitLinearStatTracker(in_features=4, start_recording_at_step=200)
    t1.running_max = torch.tensor([1.0, 2.0, 3.0, 4.0])
    t1.n_calls = 800
    t1.n_skipped = 200
    t1.current_step = 1000

    t2 = BitLinearStatTracker(in_features=4, start_recording_at_step=200)
    t2.running_max = torch.tensor([0.5, 1.5, 2.5, 3.5])
    t2.n_calls = 800
    t2.n_skipped = 200
    t2.current_step = 1000

    out_path = tmp_path / "warmup_maps.pt"
    save_calibration_maps({"l1": t1, "l2": t2}, out_path)
    loaded = torch.load(out_path, weights_only=False)

    # Schema bumped to 1.1 (F.5.3 warmup-skip fields), then 1.2
    # (F.5.2 accumulator_mode + ema_alpha fields).
    assert loaded["_meta"]["_schema_version"] == "1.2"
    assert loaded["_meta"]["start_recording_at_step_min"] == 200
    assert loaded["_meta"]["start_recording_at_step_max"] == 200
    assert loaded["_meta"]["n_skipped_min"] == 200
    assert loaded["_meta"]["n_skipped_max"] == 200

    # Per-layer entries carry the warmup-skip count + threshold.
    assert loaded["l1"]["n_skipped"] == 200
    assert loaded["l1"]["start_recording_at_step"] == 200
    assert loaded["l1"]["n_calls"] == 800
    assert loaded["l2"]["n_skipped"] == 200


def test_save_calibration_maps_legacy_no_warmup_meta(tmp_path) -> None:
    """Default (no warmup) trackers still serialize cleanly; the new
    meta fields report 0/0 — drop-in safe for downstream loaders that
    branch on `start_recording_at_step_min > 0`."""
    t = BitLinearStatTracker(in_features=4)
    t.running_max = torch.tensor([1.0, 2.0, 3.0, 4.0])
    t.n_calls = 100
    out_path = tmp_path / "legacy_maps.pt"
    save_calibration_maps({"l": t}, out_path)
    loaded = torch.load(out_path, weights_only=False)
    assert loaded["_meta"]["start_recording_at_step_min"] == 0
    assert loaded["_meta"]["start_recording_at_step_max"] == 0
    assert loaded["_meta"]["n_skipped_min"] == 0
    assert loaded["_meta"]["n_skipped_max"] == 0
    assert loaded["l"]["n_skipped"] == 0
    assert loaded["l"]["start_recording_at_step"] == 0


def test_attach_stat_trackers_propagates_warmup() -> None:
    """`attach_stat_trackers(model, start_recording_at_step=W)` should
    install all trackers with the same threshold."""
    model = _make_tiny_model()
    trackers = attach_stat_trackers(model, start_recording_at_step=2400)
    assert len(trackers) > 0
    for name, t in trackers.items():
        assert t.start_recording_at_step == 2400, (
            f"tracker {name} got start_recording_at_step={t.start_recording_at_step}, "
            f"expected 2400"
        )
        assert t.current_step == 0
        assert t.n_skipped == 0
    detach_stat_trackers(model)


def test_attach_stat_trackers_default_warmup_is_zero() -> None:
    """No-kwarg attach behaves like the pre-F.5.3 API (records every call)."""
    model = _make_tiny_model()
    trackers = attach_stat_trackers(model)
    for t in trackers.values():
        assert t.start_recording_at_step == 0
    detach_stat_trackers(model)


# -----------------------------------------------------------------
# F.5.2 — EMA accumulator
# -----------------------------------------------------------------

def test_stat_tracker_default_mode_is_max() -> None:
    """No-arg tracker uses legacy element-wise max — backwards compat."""
    t = BitLinearStatTracker(in_features=4)
    assert t.accumulator_mode == "max"
    assert t.ema_alpha == 0.99  # field present even when unused


def test_stat_tracker_ema_first_observation_seeds_running_estimate() -> None:
    """EMA bootstrap: first post-warmup call copies per_channel directly so
    the running estimate isn't biased toward the zero init."""
    t = BitLinearStatTracker(in_features=4, accumulator_mode="ema", ema_alpha=0.99)
    fake = torch.nn.Linear(4, 4)
    x = torch.tensor([[10.0, 20.0, 5.0, 15.0]])
    t(fake, (x,), torch.empty(0))
    assert t.n_calls == 1
    expected = torch.tensor([10.0, 20.0, 5.0, 15.0])
    assert torch.allclose(t.running_max, expected), (
        "first EMA observation should seed running_max directly; "
        f"got {t.running_max.tolist()}"
    )


def test_stat_tracker_ema_smooths_subsequent_observations() -> None:
    """α=0.99: running_max ← 0.99·prev + 0.01·current.
    Feeding [10] then [1] gives 0.99·10 + 0.01·1 = 9.91."""
    t = BitLinearStatTracker(in_features=1, accumulator_mode="ema", ema_alpha=0.99)
    fake = torch.nn.Linear(1, 1)
    t(fake, (torch.tensor([[10.0]]),), torch.empty(0))
    t(fake, (torch.tensor([[1.0]]),), torch.empty(0))
    assert t.n_calls == 2
    expected = 0.99 * 10.0 + 0.01 * 1.0
    assert abs(float(t.running_max[0]) - expected) < 1e-5, (
        f"expected {expected:.4f}, got {float(t.running_max[0]):.4f}"
    )


def test_stat_tracker_ema_alpha_zero_tracks_latest() -> None:
    """α=0 → running_max equals the most recent observation (pure tracking)."""
    t = BitLinearStatTracker(in_features=1, accumulator_mode="ema", ema_alpha=0.0)
    fake = torch.nn.Linear(1, 1)
    t(fake, (torch.tensor([[100.0]]),), torch.empty(0))
    t(fake, (torch.tensor([[3.0]]),), torch.empty(0))
    t(fake, (torch.tensor([[7.0]]),), torch.empty(0))
    # Bootstrap on call 1, then α=0 means subsequent calls fully overwrite.
    assert abs(float(t.running_max[0]) - 7.0) < 1e-6


def test_stat_tracker_ema_alpha_one_freezes_after_bootstrap() -> None:
    """α=1 → after the bootstrap observation, the EMA never updates."""
    t = BitLinearStatTracker(in_features=1, accumulator_mode="ema", ema_alpha=1.0)
    fake = torch.nn.Linear(1, 1)
    t(fake, (torch.tensor([[5.0]]),), torch.empty(0))
    t(fake, (torch.tensor([[1000.0]]),), torch.empty(0))
    t(fake, (torch.tensor([[1000.0]]),), torch.empty(0))
    # Bootstrap recorded 5.0; subsequent updates are 1.0·5.0 + 0.0·… = 5.0
    assert abs(float(t.running_max[0]) - 5.0) < 1e-6


def test_stat_tracker_ema_mode_does_not_track_max_anymore() -> None:
    """Sanity: under EMA, running_max is the EMA, not the per-call max.
    Feed a small-then-huge sequence under α=0.99 → result << huge value."""
    t = BitLinearStatTracker(in_features=1, accumulator_mode="ema", ema_alpha=0.99)
    fake = torch.nn.Linear(1, 1)
    t(fake, (torch.tensor([[1.0]]),), torch.empty(0))  # bootstrap
    for _ in range(10):
        t(fake, (torch.tensor([[1.0]]),), torch.empty(0))
    # Now a single huge spike — should barely move the EMA
    t(fake, (torch.tensor([[1000.0]]),), torch.empty(0))
    # Running estimate ≈ 0.99·1.0 + 0.01·1000 ≈ 10.99, dramatically less than 1000
    assert float(t.running_max[0]) < 50.0, (
        f"EMA should resist single spikes; got {float(t.running_max[0]):.2f}"
    )


def test_stat_tracker_invalid_mode_rejected() -> None:
    with pytest.raises(ValueError, match="accumulator_mode must be one of"):
        BitLinearStatTracker(in_features=4, accumulator_mode="median")


def test_stat_tracker_invalid_alpha_rejected() -> None:
    with pytest.raises(ValueError, match=r"ema_alpha must be in \[0.0, 1.0\]"):
        BitLinearStatTracker(in_features=4, accumulator_mode="ema", ema_alpha=1.5)
    with pytest.raises(ValueError, match=r"ema_alpha must be in \[0.0, 1.0\]"):
        BitLinearStatTracker(in_features=4, accumulator_mode="ema", ema_alpha=-0.1)


def test_stat_tracker_ema_composes_with_skip_warmup() -> None:
    """F.5.2 + F.5.3 composability: warmup skip drops the first N calls,
    THEN the EMA bootstraps on the first post-warmup observation."""
    t = BitLinearStatTracker(
        in_features=1,
        start_recording_at_step=3,
        accumulator_mode="ema",
        ema_alpha=0.5,
    )
    fake = torch.nn.Linear(1, 1)
    # Steps 0..2 (current_step < 3): skipped, including the giant value
    for step in range(3):
        t.current_step = step
        t(fake, (torch.tensor([[1000.0]]),), torch.empty(0))
    assert t.n_skipped == 3
    assert t.n_calls == 0
    assert torch.all(t.running_max == 0.0), (
        "running_max must remain zero through warmup even under EMA mode"
    )
    # Step 3: first post-warmup → bootstrap to 7.0 directly
    t.current_step = 3
    t(fake, (torch.tensor([[7.0]]),), torch.empty(0))
    assert t.n_calls == 1
    assert abs(float(t.running_max[0]) - 7.0) < 1e-6


def test_attach_stat_trackers_propagates_ema_kwargs() -> None:
    """attach_stat_trackers passes accumulator_mode + ema_alpha to every
    installed tracker."""
    model = _make_tiny_model()
    trackers = attach_stat_trackers(
        model,
        accumulator_mode="ema",
        ema_alpha=0.95,
    )
    for name, t in trackers.items():
        assert t.accumulator_mode == "ema", f"{name} mode={t.accumulator_mode}"
        assert t.ema_alpha == 0.95, f"{name} alpha={t.ema_alpha}"
    detach_stat_trackers(model)


def test_save_calibration_maps_records_ema_in_meta(tmp_path) -> None:
    """F.5.2 audit: per-layer entries carry mode + α; _meta aggregates
    distinct modes/alphas across trackers."""
    t = BitLinearStatTracker(in_features=4, accumulator_mode="ema", ema_alpha=0.97)
    t.running_max = torch.tensor([1.0, 2.0, 3.0, 4.0])
    t.n_calls = 50
    out_path = tmp_path / "ema_maps.pt"
    save_calibration_maps({"l": t}, out_path)
    loaded = torch.load(out_path, weights_only=False)
    assert loaded["_meta"]["_schema_version"] == "1.2"
    assert loaded["_meta"]["accumulator_modes"] == ["ema"]
    assert loaded["_meta"]["ema_alphas"] == [0.97]
    assert loaded["l"]["accumulator_mode"] == "ema"
    assert loaded["l"]["ema_alpha"] == 0.97


def test_save_calibration_maps_legacy_max_mode_meta(tmp_path) -> None:
    """Default (max-mode) trackers serialize with the new fields too."""
    t = BitLinearStatTracker(in_features=4)
    t.running_max = torch.tensor([1.0, 2.0, 3.0, 4.0])
    t.n_calls = 100
    out_path = tmp_path / "max_maps.pt"
    save_calibration_maps({"l": t}, out_path)
    loaded = torch.load(out_path, weights_only=False)
    assert loaded["_meta"]["accumulator_modes"] == ["max"]
    assert loaded["_meta"]["ema_alphas"] == [0.99]
    assert loaded["l"]["accumulator_mode"] == "max"
