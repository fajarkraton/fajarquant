#!/usr/bin/env python3
"""
Phase E E1.2 — Smoke gate for `intllm_en.py` config shim.

Per CLAUDE §6.8 R3 (prevention layer per phase): wired to
`make test-intllm-en` and asserted by hand on every change to
`python/phase_e/intllm_en.py`.

What it covers (must stay green):
    1. E0 tokenizer-rate file exists + matches ID/EN_BYTES_PER_TOKEN constants
    2. E1.4 manifests yield ~15.40 B ID tokens (matches ID_TOKENS_AVAILABLE)
    3. compute_en_token_cap math (60:40 → 10.27 B; 50:50 → 15.40 B; 80:20 → 3.85 B)
    4. select_slimpajama_repo pivots at 6 B threshold
    5. bilingual_mix_summary is pure (no network/torch/transformers needed)
    6. Re-importing intllm_en succeeds with stdlib + json only

Runtime: < 1 s.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PHASE_E = REPO_ROOT / "python" / "phase_e"


def _tol(expected: float, actual: float, tol: float) -> bool:
    return abs(expected - actual) <= tol


def main() -> int:
    sys.path.insert(0, str(PHASE_E))
    intllm_en = importlib.import_module("intllm_en")

    # 1. E0 tokenizer file exists + matches constants ──────────────────
    e0_path = (
        REPO_ROOT
        / "paper"
        / "intllm"
        / "results"
        / "phase_e"
        / "e0_tokenizer_compression.json"
    )
    assert e0_path.is_file(), f"E0 tokenizer file missing: {e0_path}"
    e0 = json.loads(e0_path.read_text(encoding="utf-8"))
    id_rate = e0["per_language"]["Indonesian"]["measurement"]["bytes_per_token"]
    en_rate = e0["per_language"]["English"]["measurement"]["bytes_per_token"]
    assert _tol(intllm_en.ID_BYTES_PER_TOKEN, id_rate, 0.01), (
        f"ID_BYTES_PER_TOKEN drift: const={intllm_en.ID_BYTES_PER_TOKEN}, "
        f"E0={id_rate}"
    )
    assert _tol(intllm_en.EN_BYTES_PER_TOKEN, en_rate, 0.01), (
        f"EN_BYTES_PER_TOKEN drift: const={intllm_en.EN_BYTES_PER_TOKEN}, "
        f"E0={en_rate}"
    )

    # 2. E1.4 manifests + ID_TOKENS_AVAILABLE ──────────────────────────
    id_tokens_measured = intllm_en.id_tokens_from_e1_4_manifests(REPO_ROOT)
    # Should be within ±0.05 B of the constant (allows for the float
    # rounding when the constant was committed).
    assert _tol(intllm_en.ID_TOKENS_AVAILABLE, id_tokens_measured, 0.05e9), (
        f"ID_TOKENS_AVAILABLE drift: const={intllm_en.ID_TOKENS_AVAILABLE/1e9:.3f} B, "
        f"manifests={id_tokens_measured/1e9:.3f} B"
    )

    # 3. compute_en_token_cap math ─────────────────────────────────────
    cap_60_40 = intllm_en.compute_en_token_cap(15.40e9, 0.6) / 1e9
    cap_50_50 = intllm_en.compute_en_token_cap(15.40e9, 0.5) / 1e9
    cap_80_20 = intllm_en.compute_en_token_cap(15.40e9, 0.8) / 1e9
    assert abs(cap_60_40 - 10.27) < 0.01, f"60:40 cap: {cap_60_40} vs expected 10.27"
    assert abs(cap_50_50 - 15.40) < 0.01, f"50:50 cap: {cap_50_50} vs expected 15.40"
    assert abs(cap_80_20 - 3.85) < 0.01, f"80:20 cap: {cap_80_20} vs expected 3.85"

    # Edge cases on ratio bounds (must reject 0.0 / 1.0).
    for bad in (0.0, 1.0, -0.1, 1.5):
        try:
            intllm_en.compute_en_token_cap(15.40e9, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"compute_en_token_cap accepted bad ratio: {bad}")

    # 4. select_slimpajama_repo pivots ──────────────────────────────────
    assert (
        intllm_en.select_slimpajama_repo(2e9) == intllm_en.SLIMPAJAMA_DEFAULT_REPO
    ), "2 B EN target should hit DEFAULT (6B repo)"
    assert (
        intllm_en.select_slimpajama_repo(5.99e9) == intllm_en.SLIMPAJAMA_DEFAULT_REPO
    ), "5.99 B EN target should hit DEFAULT (just under threshold)"
    assert (
        intllm_en.select_slimpajama_repo(6.01e9) == intllm_en.SLIMPAJAMA_STRETCH_REPO
    ), "6.01 B EN target should hit STRETCH (just over threshold)"
    assert (
        intllm_en.select_slimpajama_repo(15e9) == intllm_en.SLIMPAJAMA_STRETCH_REPO
    ), "15 B EN target should hit STRETCH (627B repo)"

    # 5. bilingual_mix_summary purity ──────────────────────────────────
    summary = intllm_en.bilingual_mix_summary()  # default 60:40, manifest-driven
    assert summary["ratio_id_share"] == 0.6
    assert summary["fineweb_edu_addon"] is False
    assert summary["slimpajama_repo"] == intllm_en.SLIMPAJAMA_STRETCH_REPO, (
        f"60:40 with 15.40 B ID requires Stretch repo (10.27 B EN > 6 B threshold), "
        f"got: {summary['slimpajama_repo']}"
    )
    # Total tokens consistency check
    expected_total = summary["id_tokens"] + summary["en_tokens"]
    assert _tol(summary["total_tokens"], expected_total, 1.0), (
        f"total_tokens inconsistent: {summary['total_tokens']} vs "
        f"id+en={expected_total}"
    )

    # 6. Module imports with stdlib + json only ─────────────────────────
    # Sanity check: re-import in a clean namespace; must not require
    # torch / transformers / datasets.
    blocked = {
        "torch", "transformers", "datasets", "datasketch",
        "pyarrow", "numpy", "pandas",
    }
    in_intllm_en = {
        name for name in sys.modules
        if name in blocked and not name.startswith("_")
    }
    # Note: torch may already be loaded by other test runs in the same
    # interpreter — what we actually care about is that intllm_en itself
    # doesn't IMPORT them. Verify by checking the source.
    src = (PHASE_E / "intllm_en.py").read_text(encoding="utf-8")
    for blocked_pkg in blocked:
        assert f"import {blocked_pkg}" not in src, (
            f"intllm_en.py must not import heavy dep: {blocked_pkg}"
        )
        assert f"from {blocked_pkg}" not in src, (
            f"intllm_en.py must not import heavy dep: {blocked_pkg}"
        )

    # All checks passed.
    print("test-intllm-en: 6/6 invariants PASS")
    print(f"  ID rate: {intllm_en.ID_BYTES_PER_TOKEN} byte/tok (E0: {id_rate:.4f})")
    print(f"  EN rate: {intllm_en.EN_BYTES_PER_TOKEN} byte/tok (E0: {en_rate:.4f})")
    print(f"  ID tokens (manifests): {id_tokens_measured/1e9:.3f} B")
    print(f"  60:40 EN cap: {cap_60_40:.2f} B → {summary['slimpajama_repo']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
