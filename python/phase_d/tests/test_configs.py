"""Sanity tests for Mini/Base/Medium/Stretch configs.

Verifies each config:
  - Imports cleanly + instantiates the dataclass
  - Produces a valid HGRNBitConfig
  - Param count matches FJQ_PHASE_D_CONFIG.md §2 within ±10%
  - Token-budget = batch_size × seq_len × n_steps lands in the
    expected range per FJQ_PHASE_D_CONFIG.md §3.2
"""

from __future__ import annotations

import pytest

from intllm.model import HGRNBitConfig, HGRNBitForCausalLM


# Expected (params, tokens) pairs from FJQ_PHASE_D_CONFIG.md §2/§3.2
EXPECTATIONS: dict[str, tuple[int, int]] = {
    "mini": (21_500_000, 491_000_000),
    "base": (46_400_000, 982_000_000),
    "medium": (71_300_000, 1_999_000_000),
    "stretch": (369_100_000, 14_991_000_000),
}
TOLERANCE_PCT = 10.0


@pytest.mark.parametrize("config_name", ["mini", "base", "medium", "stretch"])
def test_config_imports_and_param_count(config_name: str) -> None:
    """Each config dataclass loads + produces a model with params close
    to FJQ_PHASE_D_CONFIG.md §2 spec."""
    if config_name == "mini":
        from configs.mini import MiniArchConfig as Arch
    elif config_name == "base":
        from configs.base import BaseArchConfig as Arch
    elif config_name == "medium":
        from configs.medium import MediumArchConfig as Arch
    else:
        from configs.stretch import StretchArchConfig as Arch
    arch = Arch()

    cfg = HGRNBitConfig(
        vocab_size=arch.vocab_size,
        hidden_size=arch.hidden_size,
        num_hidden_layers=arch.num_hidden_layers,
        max_position_embeddings=arch.max_position_embeddings,
    )
    model = HGRNBitForCausalLM(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    expected_params, _ = EXPECTATIONS[config_name]
    pct_diff = abs(n_params - expected_params) / expected_params * 100
    assert pct_diff <= TOLERANCE_PCT, (
        f"{config_name}: params {n_params:,} differs from spec {expected_params:,} "
        f"by {pct_diff:.1f}% (tol {TOLERANCE_PCT}%)"
    )


@pytest.mark.parametrize("config_name", ["mini", "base", "medium", "stretch"])
def test_config_token_budget_in_range(config_name: str) -> None:
    """Total tokens = batch × seq × n_steps must match plan budget within
    ±10%."""
    if config_name == "mini":
        from configs.mini import MiniTrainConfig as Train
    elif config_name == "base":
        from configs.base import BaseTrainConfig as Train
    elif config_name == "medium":
        from configs.medium import MediumTrainConfig as Train
    else:
        from configs.stretch import StretchTrainConfig as Train
    t = Train()
    actual_tokens = t.batch_size * t.seq_len * t.n_steps

    _, expected_tokens = EXPECTATIONS[config_name]
    pct_diff = abs(actual_tokens - expected_tokens) / expected_tokens * 100
    assert pct_diff <= TOLERANCE_PCT, (
        f"{config_name}: tokens {actual_tokens:,} differs from spec "
        f"{expected_tokens:,} by {pct_diff:.1f}% (tol {TOLERANCE_PCT}%)"
    )
