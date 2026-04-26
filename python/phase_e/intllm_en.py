"""Phase E E1.2 — English subset config + reuse shim for SlimPajama.

Purpose
-------
"Registry/config" deliverable per `FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md`
§3 E1.2. Records the bilingual-mix design decision and exposes constants +
helpers so Phase E training code can compose ID-corpus-on-disk parquet
(produced by E1.4 exact-hash dedup) with EN streaming via Phase D's
`intllm.data.slimpajama_stream` without each call site re-deriving the
math.

This module contains NO streaming code. SlimPajama streaming lives in
Phase D (`python/phase_d/intllm/data.py`) and is imported by Phase E
training drivers when Phase E begins (E2/E3 phases). E1.2's job is just
to record what mix to use and how to compute it.

Decision recorded (v0)
----------------------
- **Ratio:** ID:EN = 60:40 (matches §E2.4.1 default for the bilingual
  calibration sampler). Optionally re-pinnable via `compute_en_token_cap(
  id_tokens, ratio_id_share=0.5)` for 50:50 ablation.
- **EN repo:** `DKYoon/SlimPajama-6B` for Mini/Base/Medium (≤ ~6 B EN
  tokens). Stretch (≥15 B EN tokens) pivots to `gmongaras/SlimPajama-
  627B_Reupload` per Phase D's existing convention (`intllm.data.
  DEFAULT_SLIMPAJAMA_REPO` + the docstring at line ~30 of `data.py`).
- **FineWeb-Edu add-on:** SKIPPED for v0. Re-evaluate at §E5/E6 if
  Mini-scale ablation shows EN coverage is the bottleneck.
- **Tokenizer:** Mistral v3 32K (same as Phase D + Phase E ID side).
  Compression rates from E0.0.3 measurement on 1000-article samples.

Invariants + verify
-------------------
Run `make test-intllm-en` (smoke, ~1 s) to validate:
  1. E0 tokenizer file exists and rates match the constants below
  2. E1.4 ID corpus manifests exist and yield ~15.40 B tokens via
     ID_BYTES_PER_TOKEN
  3. compute_en_token_cap(15.40e9, ratio_id_share=0.6) ≈ 10.27 B
     within ±0.05 B
  4. Re-importing this module does not crash without network/
     PyTorch/transformers (i.e., constants-only module is dependency-light)
"""

from __future__ import annotations

import json
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────
# Compression rates (E0.0.3 measurement, Mistral v3 32K, 1000 articles each)
# Source: `paper/intllm/results/phase_e/e0_tokenizer_compression.json`
# ────────────────────────────────────────────────────────────────────────

# Indonesian: less efficient (smaller vocab coverage in the Mistral
# pretraining mix) → more tokens per byte of input text.
ID_BYTES_PER_TOKEN: float = 2.527

# English: highly efficient (dominant in Mistral pretrain) → fewer tokens
# per byte of input.
EN_BYTES_PER_TOKEN: float = 3.830

# ────────────────────────────────────────────────────────────────────────
# Bilingual ratio defaults
# ────────────────────────────────────────────────────────────────────────

# 60:40 ID-heavy default — matches §E2.4.1
# `BilingualCalibrationSampler` default ratio.
BILINGUAL_RATIO_DEFAULT: float = 0.6

# Available ID corpus token count (post-E1.4 exact-hash dedup,
# from `data/phase_e/corpus_id_dedup_exact/*/_manifest.json` aggregate
# bytes_text_kept / ID_BYTES_PER_TOKEN). Treated as a constant going
# forward; re-derive via `id_tokens_from_e1_4_manifests()` if the
# corpus is regenerated.
ID_TOKENS_AVAILABLE: float = 15.40e9

# ────────────────────────────────────────────────────────────────────────
# SlimPajama repo selection (mirrors Phase D `intllm.data` convention)
# ────────────────────────────────────────────────────────────────────────

# Default for Mini/Base/Medium scales (Phase D production).
SLIMPAJAMA_DEFAULT_REPO: str = "DKYoon/SlimPajama-6B"

# Stretch / full-pretrain pivot when EN budget exceeds ~6 B unique tokens.
# Phase D switches at 15 B; Phase E inherits the same threshold.
SLIMPAJAMA_STRETCH_REPO: str = "gmongaras/SlimPajama-627B_Reupload"

# Stretch threshold in tokens. Above this, SLIMPAJAMA_STRETCH_REPO is
# required to avoid silent multi-epoch wraparound on the 6B repo (which
# `slimpajama_stream` does NOT do — it stops on StopIteration).
SLIMPAJAMA_STRETCH_THRESHOLD_TOKENS: float = 6.0e9

# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────


def compute_en_token_cap(
    id_tokens: float,
    ratio_id_share: float = BILINGUAL_RATIO_DEFAULT,
) -> float:
    """Given a fixed ID-token count and the desired ID share of the
    bilingual mix, return the matching EN token cap.

    >>> round(compute_en_token_cap(15.40e9, 0.6) / 1e9, 2)
    10.27
    >>> round(compute_en_token_cap(15.40e9, 0.5) / 1e9, 2)
    15.4
    >>> round(compute_en_token_cap(15.40e9, 0.8) / 1e9, 2)
    3.85
    """
    if not 0.0 < ratio_id_share < 1.0:
        raise ValueError(
            f"ratio_id_share must be in (0, 1), got {ratio_id_share}"
        )
    total = id_tokens / ratio_id_share
    return total - id_tokens


def compute_en_byte_cap(en_tokens: float) -> float:
    """EN bytes that, when tokenized at `EN_BYTES_PER_TOKEN`, yield
    approximately `en_tokens` tokens.

    >>> round(compute_en_byte_cap(10.27e9) / 1e9, 2)
    39.33
    """
    return en_tokens * EN_BYTES_PER_TOKEN


def select_slimpajama_repo(en_token_target: float) -> str:
    """Mini/Base/Medium → 6B repo; Stretch → 627B repo. Pivot at
    `SLIMPAJAMA_STRETCH_THRESHOLD_TOKENS`.

    >>> select_slimpajama_repo(2e9)
    'DKYoon/SlimPajama-6B'
    >>> select_slimpajama_repo(15e9)
    'gmongaras/SlimPajama-627B_Reupload'
    """
    if en_token_target > SLIMPAJAMA_STRETCH_THRESHOLD_TOKENS:
        return SLIMPAJAMA_STRETCH_REPO
    return SLIMPAJAMA_DEFAULT_REPO


def id_tokens_from_e1_4_manifests(repo_root: Path | None = None) -> float:
    """Read both E1.4 manifest JSONs and return the aggregate post-dedup
    ID token count. Re-derived rather than cached so corpus regeneration
    is reflected automatically.

    Falls back to `ID_TOKENS_AVAILABLE` constant if manifests are missing
    (e.g., on a fresh clone before the dedup sweep runs).
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent
    bytes_kept = 0
    for src in ("HuggingFaceFW__fineweb-2", "wikimedia__wikipedia"):
        m = (
            repo_root
            / "data"
            / "phase_e"
            / "corpus_id_dedup_exact"
            / src
            / "_manifest.json"
        )
        if not m.is_file():
            return ID_TOKENS_AVAILABLE
        bytes_kept += json.loads(m.read_text(encoding="utf-8"))["bytes_text_kept"]
    return bytes_kept / ID_BYTES_PER_TOKEN


def bilingual_mix_summary(
    id_tokens: float | None = None,
    ratio_id_share: float = BILINGUAL_RATIO_DEFAULT,
) -> dict:
    """Single-call summary suitable for embedding in findings docs +
    smoke-test assertions. Pure function over constants + manifests; no
    network, no PyTorch, no transformers.
    """
    if id_tokens is None:
        id_tokens = id_tokens_from_e1_4_manifests()
    en_tokens = compute_en_token_cap(id_tokens, ratio_id_share)
    en_bytes = compute_en_byte_cap(en_tokens)
    repo = select_slimpajama_repo(en_tokens)
    return {
        "id_tokens": id_tokens,
        "en_tokens": en_tokens,
        "total_tokens": id_tokens + en_tokens,
        "ratio_id_share": ratio_id_share,
        "ratio_en_share": 1.0 - ratio_id_share,
        "en_bytes_estimate": en_bytes,
        "id_bytes_per_token": ID_BYTES_PER_TOKEN,
        "en_bytes_per_token": EN_BYTES_PER_TOKEN,
        "slimpajama_repo": repo,
        "stretch_threshold_tokens": SLIMPAJAMA_STRETCH_THRESHOLD_TOKENS,
        "fineweb_edu_addon": False,  # v0; re-evaluate at E5/E6
    }


__all__ = [
    "ID_BYTES_PER_TOKEN",
    "EN_BYTES_PER_TOKEN",
    "BILINGUAL_RATIO_DEFAULT",
    "ID_TOKENS_AVAILABLE",
    "SLIMPAJAMA_DEFAULT_REPO",
    "SLIMPAJAMA_STRETCH_REPO",
    "SLIMPAJAMA_STRETCH_THRESHOLD_TOKENS",
    "compute_en_token_cap",
    "compute_en_byte_cap",
    "select_slimpajama_repo",
    "id_tokens_from_e1_4_manifests",
    "bilingual_mix_summary",
]
