"""
B-fix.1.3 regression tests for strategy_selector.py.

Ensures architecture gates cannot be re-introduced:
- MQA (1 head) must NOT force Path C
- Wide-GQA (8+ heads) must NOT force Path A
- All architectures use the same threshold-based decision tree
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from strategy_selector import select_strategy, DEFAULT_THRESHOLDS


# ═══════════════════════════════════════════════════════════════════════
# B-fix.1.3: MQA must NOT be forced to Path C
# ═══════════════════════════════════════════════════════════════════════


def test_mqa_high_cv_gets_path_a():
    """MQA with high channel variance should get Path A (KIVI), not C."""
    stats = {"channel_var_cv": 3.0, "svd_ratio": 1.0, "kurtosis": 0.5, "skewness": 0.1}
    result = select_strategy(stats, bits=2, n_kv_heads=1)
    assert result == "A", f"MQA with high cv should get A, got {result}"


def test_mqa_low_stats_gets_default():
    """MQA with all stats below thresholds should get default (A), not C."""
    stats = {"channel_var_cv": 0.5, "svd_ratio": 1.0, "kurtosis": 0.3, "skewness": 0.05}
    result = select_strategy(stats, bits=3, n_kv_heads=1)
    assert result == "A", f"MQA with low stats should get default A, got {result}"


def test_mqa_high_kurtosis_gets_path_c():
    """MQA with genuinely high kurtosis should get Path C via threshold, not gate."""
    stats = {"channel_var_cv": 0.5, "svd_ratio": 1.0, "kurtosis": 5.0, "skewness": 0.1}
    result = select_strategy(stats, bits=2, n_kv_heads=1)
    assert result == "C", f"MQA with high kurtosis should get C, got {result}"


# ═══════════════════════════════════════════════════════════════════════
# B-fix.1.3: Wide-GQA must NOT be forced to Path A
# ═══════════════════════════════════════════════════════════════════════


def test_wide_gqa_high_kurtosis_gets_path_c():
    """Wide-GQA with high kurtosis should get Path C, not forced A."""
    stats = {"channel_var_cv": 0.5, "svd_ratio": 1.0, "kurtosis": 5.0, "skewness": 0.1}
    result = select_strategy(stats, bits=2, n_kv_heads=8)
    assert result == "C", f"Wide-GQA with high kurtosis should get C, got {result}"


def test_wide_gqa_high_svd_gets_path_b():
    """Wide-GQA with high SVD ratio should get Path B, not forced A."""
    stats = {"channel_var_cv": 0.5, "svd_ratio": 4.0, "kurtosis": 0.5, "skewness": 0.1}
    result = select_strategy(stats, bits=3, n_kv_heads=8)
    assert result == "B", f"Wide-GQA with high svd should get B, got {result}"


# ═══════════════════════════════════════════════════════════════════════
# Architecture-independence: same stats → same path regardless of n_kv_heads
# ═══════════════════════════════════════════════════════════════════════


def test_same_stats_same_path_across_architectures():
    """Same stats should produce same path for MQA, narrow-GQA, and wide-GQA."""
    for stats, expected in [
        ({"channel_var_cv": 3.0, "svd_ratio": 1.0, "kurtosis": 0.5, "skewness": 0.1}, "A"),
        ({"channel_var_cv": 0.5, "svd_ratio": 4.0, "kurtosis": 0.5, "skewness": 0.1}, "B"),
        ({"channel_var_cv": 0.5, "svd_ratio": 1.0, "kurtosis": 5.0, "skewness": 0.1}, "C"),
        ({"channel_var_cv": 0.5, "svd_ratio": 1.0, "kurtosis": 0.5, "skewness": 0.8}, "E"),
    ]:
        for n_heads in [1, 4, 8, 32]:
            result = select_strategy(stats, bits=3, n_kv_heads=n_heads)
            assert result == expected, (
                f"n_kv_heads={n_heads}: expected {expected}, got {result} "
                f"for stats={stats}"
            )


# ═══════════════════════════════════════════════════════════════════════
# Decision tree order: cv > svd > kurt > skew > residual > default
# ═══════════════════════════════════════════════════════════════════════


def test_cv_takes_priority_over_kurtosis():
    """When both cv and kurtosis exceed thresholds, cv (Path A) wins."""
    stats = {"channel_var_cv": 3.0, "svd_ratio": 1.0, "kurtosis": 5.0, "skewness": 0.1}
    result = select_strategy(stats, bits=2, n_kv_heads=4)
    assert result == "A", f"cv should take priority over kurtosis, got {result}"


def test_residual_only_at_2bit():
    """Path D (residual) only triggers at bits <= 2."""
    stats = {"channel_var_cv": 0.5, "svd_ratio": 1.0, "kurtosis": 1.5, "skewness": 0.1}
    assert select_strategy(stats, bits=2, n_kv_heads=4) == "D"
    assert select_strategy(stats, bits=3, n_kv_heads=4) == "A"  # default, not D
    assert select_strategy(stats, bits=4, n_kv_heads=4) == "A"


def test_default_is_path_a():
    """All stats below thresholds → default Path A."""
    stats = {"channel_var_cv": 0.1, "svd_ratio": 1.0, "kurtosis": 0.1, "skewness": 0.01}
    for bits in [2, 3, 4]:
        result = select_strategy(stats, bits=bits, n_kv_heads=4)
        assert result == "A", f"Default should be A at {bits}-bit, got {result}"
