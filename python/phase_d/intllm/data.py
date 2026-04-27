"""IntLLM dataset loaders.

Five modes:
  - `synthetic_token_batches` — fresh random tokens per batch (entropy
    floor = ln(V); no learnable structure; useful for "no-NaN" smoke
    tests but **cannot** verify loss-decrease).
  - `overfit_token_batches`   — one fixed batch, repeated N times. Model
    memorizes; loss drops toward 0. Convergence sanity check.
  - `slimpajama_stream`       — real text via HF datasets streaming,
    tokenised on the fly with the Mistral v3 tokenizer (C.P3.3).
    Track B step 4 (V31.C.P6.4) added per-chunk read timeout + retry
    so transient HF CDN flaps no longer hang the training process.
  - `indonesian_parquet_stream` — Phase E1.1+E1.4 ID corpus from local
    parquet shards (`data/phase_e/corpus_id_dedup_exact/`). 15.40 B
    tokens across ~10.66M docs (Wikipedia + FineWeb-2 ID). Same
    document-packing pattern as `slimpajama_stream`. Phase E2 Q5.
  - `bilingual_stream`        — interleaves the two real-text streams
    above at a configurable ID:EN ratio (default 60:40 from
    `intllm_en.BILINGUAL_RATIO_DEFAULT`). Per-batch language choice
    via seeded RNG. Phase E2 Q5 + E2.4 (BilingualCalibrationSampler).

The C.P3.3 streaming loader is the production training path — no
on-disk pre-staging needed for Mini/Base (≤2B tokens). For Stretch
(15B), pre-tokenize and shard offline (deferred to C.P4).
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerBase

# DKYoon/SlimPajama-6B is the default Mini/Base/Medium source —
# small enough to stream easily, large enough to cover ≤2B-token runs.
# Stretch (15B) pivots to gmongaras/SlimPajama-627B_Reupload.
DEFAULT_SLIMPAJAMA_REPO = "DKYoon/SlimPajama-6B"

# Track B step 4 (V31.C.P6.4 — streaming-hang prevention).
#
# Root cause memory: c.1 on 2026-04-22 hung 8.5h because a dead HF CDN
# socket (post-laptop-suspend) blocked urllib3 indefinitely. The two
# env vars below cap any single HTTP chunk read at 60s — a dead
# socket raises a timeout instead of hanging forever, which triggers
# the `_retry_iter` wrapper below, which reconstructs the stream.
#
# `setdefault` so users can still override via shell env if needed.
os.environ.setdefault("HF_DATASETS_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

# Tuple of exception types we treat as "transient network blips" —
# retriable. NOT retriable: data-level errors (KeyError, ValueError)
# which indicate a logic bug rather than network flakiness. Lazily
# built to avoid importing urllib3/requests at module load if datasets
# is not installed.
def _build_retryable_exceptions() -> tuple[type[BaseException], ...]:
    exc: list[type[BaseException]] = [ConnectionError, OSError, TimeoutError]
    try:
        import requests.exceptions as _req
        exc.extend([_req.Timeout, _req.ConnectionError, _req.ChunkedEncodingError])
    except ImportError:  # pragma: no cover - requests always present w/ datasets
        pass
    try:
        import urllib3.exceptions as _u3
        exc.extend([_u3.ReadTimeoutError, _u3.ProtocolError])
    except ImportError:  # pragma: no cover
        pass
    return tuple(exc)


def _retry_iter(
    factory: Callable[[int], Iterator],
    *,
    retryable: tuple[type[BaseException], ...] | None = None,
    max_attempts: int = 5,
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> Iterator:
    """Yield items from `factory(attempt_number)` with retry-on-transient.

    `factory(attempt)` MUST return a freshly-constructed iterator each
    call — we cannot resume a dead iterator, we can only replace it.
    Attempts are 0-indexed for the factory, 1-indexed for max_attempts
    counting (so max_attempts=5 allows up to 5 factory() calls total).

    Backoff on failure: exponential, capped at 60s. Tests pass a no-op
    `on_retry` to skip the wall-clock wait.

    Raises the last exception if max_attempts is exhausted.
    """
    retry_exc = retryable or _build_retryable_exceptions()
    attempt = 0
    while True:
        try:
            yield from factory(attempt)
            return  # iterator exhausted naturally
        except retry_exc as e:
            attempt += 1
            if attempt >= max_attempts:
                raise
            if on_retry is not None:
                on_retry(attempt, e)
            else:
                backoff = min(60, 2 ** attempt)
                print(
                    f"[retry_iter] attempt {attempt}/{max_attempts} failed "
                    f"({type(e).__name__}: {e}); sleeping {backoff}s",
                    flush=True,
                )
                time.sleep(backoff)


def synthetic_token_batches(
    *,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    n_batches: int,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> Iterator[torch.Tensor]:
    """Yield `n_batches` of fresh random integer-token tensors.

    Each batch is independently drawn uniformly from `[0, vocab_size)`,
    shape `(batch_size, seq_len)`. The expected next-token loss for any
    model on this data is `ln(vocab_size)` — no learnable structure
    exists. Use this for stack-stability smoke tests, not for
    convergence checks.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(n_batches):
        ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, seq_len),
            generator=g,
            dtype=torch.long,
        )
        yield ids.to(device)


def overfit_token_batches(
    *,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    n_batches: int,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> Iterator[torch.Tensor]:
    """Yield the SAME randomly-generated batch `n_batches` times.

    The model can memorize the fixed batch; loss should fall toward 0.
    This is the standard "does the training stack work?" sanity test —
    if the model can't memorize a single batch, something is broken
    upstream of any quantization or QAT concern.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    fixed_batch = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        generator=g,
        dtype=torch.long,
    ).to(device)
    for _ in range(n_batches):
        # Same tensor reference each step — autograd handles fine since
        # the optimiser doesn't touch input tensors.
        yield fixed_batch


def slimpajama_stream(
    *,
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
    batch_size: int,
    repo: str = DEFAULT_SLIMPAJAMA_REPO,
    split: str = "train",
    text_column: str = "text",
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> Iterator[torch.Tensor]:
    """Stream SlimPajama via HF datasets, tokenize on-the-fly, yield
    `(batch_size, seq_len)` int64 batches.

    No pre-tokenization, no on-disk caching beyond HF's normal hub
    cache. Suitable for Mini/Base/Medium configs that need ≤ a few
    billion tokens. Stretch (15B) should pre-tokenize + shard once and
    re-read from disk to avoid retokenising every epoch.

    Documents shorter than `seq_len` are concatenated with EOS
    separators until the buffer fills `batch_size · seq_len`. This
    "packing" pattern matches the standard MosaicML/StreamingDataset
    convention and keeps GPU utilisation high without needing
    sentence-level alignment heuristics.
    """
    from datasets import load_dataset  # type: ignore[import-not-found]

    eos_id = tokenizer.eos_token_id
    target_per_batch = batch_size * seq_len
    buf: list[int] = []

    # Track B step 4: rebuild the iterator on transient network errors
    # (HF CDN chunked-transfer drops, urllib3 read timeouts, etc.) so a
    # laptop-suspend / WiFi flap / CDN blip no longer requires operator
    # intervention. Seed is offset by attempt # so each retry sees a
    # slightly different shuffle order — avoids hammering the same
    # problem shard on back-to-back retries.
    def _factory(attempt: int) -> Iterator:
        s = seed + attempt
        return iter(
            load_dataset(repo, split=split, streaming=True).shuffle(
                seed=s, buffer_size=10_000,
            )
        )

    for example in _retry_iter(_factory):
        text = example.get(text_column)
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        buf.extend(ids)
        buf.append(eos_id)
        while len(buf) >= target_per_batch:
            chunk = buf[:target_per_batch]
            buf = buf[target_per_batch:]
            t = torch.tensor(chunk, dtype=torch.long).reshape(batch_size, seq_len)
            yield t.to(device)


# Phase E1 corpus root — written by `python/phase_e/scripts/dedup_corpus_exact.py`,
# packaged per `docs/FJQ_PHASE_E_BILINGUAL_CORPUS_V1.md`. Resolved relative to
# the fajarquant repo root (this file lives at
# `python/phase_d/intllm/data.py` → `..` × 3 = repo root).
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ID_CORPUS_ROOT = _REPO_ROOT / "data" / "phase_e" / "corpus_id_dedup_exact"
DEFAULT_ID_SOURCES = ("wikimedia__wikipedia", "HuggingFaceFW__fineweb-2")


def indonesian_parquet_stream(
    *,
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
    batch_size: int,
    corpus_root: Path | str = DEFAULT_ID_CORPUS_ROOT,
    sources: tuple[str, ...] = DEFAULT_ID_SOURCES,
    text_column: str = "text",
    device: str | torch.device = "cpu",
    seed: int = 0,
    parquet_batch_size: int = 1024,
) -> Iterator[torch.Tensor]:
    """Stream the Phase E1 ID corpus from local parquet shards, tokenize
    on the fly, yield `(batch_size, seq_len)` int64 batches.

    Same document-packing pattern as `slimpajama_stream`: documents are
    concatenated with EOS separators until the buffer fills
    `batch_size · seq_len`. Shard order is shuffled deterministically
    by `seed`; the iterator is infinite (re-shuffles each epoch).

    Wraps the local read in `_retry_iter` for symmetry with the HF
    path; on-disk reads rarely fail transiently, but we get the same
    failure surface (factory rebuild on `OSError`) for free.
    """
    import pyarrow.parquet as pq  # local import — keeps test imports light

    corpus_root = Path(corpus_root)
    if not corpus_root.is_dir():
        raise FileNotFoundError(
            f"ID corpus root not found: {corpus_root}. "
            f"Build it via `python/phase_e/scripts/dedup_corpus_exact.py` first.",
        )
    shard_paths: list[Path] = []
    for source in sources:
        src_dir = corpus_root / source
        if not src_dir.is_dir():
            continue  # tolerate a partial corpus in tests
        shard_paths.extend(sorted(src_dir.glob("shard_*.parquet")))
    if not shard_paths:
        raise FileNotFoundError(
            f"No `shard_*.parquet` under any of {sources} in {corpus_root}",
        )

    eos_id = tokenizer.eos_token_id
    target_per_batch = batch_size * seq_len
    buf: list[int] = []

    def _factory(attempt: int) -> Iterator[str]:
        # Reseed per attempt so retries see a different shuffle order.
        rng = random.Random(seed + attempt)
        order = list(shard_paths)
        rng.shuffle(order)
        # Infinite re-shuffle loop — production training is step-bounded,
        # not epoch-bounded, so the stream never naturally terminates.
        while True:
            for path in order:
                pf = pq.ParquetFile(path)
                for record_batch in pf.iter_batches(
                    batch_size=parquet_batch_size, columns=[text_column],
                ):
                    for text in record_batch.column(text_column).to_pylist():
                        if text:
                            yield text
            rng.shuffle(order)

    for text in _retry_iter(_factory):
        ids = tokenizer.encode(text, add_special_tokens=False)
        buf.extend(ids)
        buf.append(eos_id)
        while len(buf) >= target_per_batch:
            chunk = buf[:target_per_batch]
            buf = buf[target_per_batch:]
            t = torch.tensor(chunk, dtype=torch.long).reshape(batch_size, seq_len)
            yield t.to(device)


def bilingual_stream(
    *,
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
    batch_size: int,
    id_share: float = 0.6,
    device: str | torch.device = "cpu",
    seed: int = 0,
    en_repo: str = DEFAULT_SLIMPAJAMA_REPO,
    id_corpus_root: Path | str = DEFAULT_ID_CORPUS_ROOT,
    id_sources: tuple[str, ...] = DEFAULT_ID_SOURCES,
) -> Iterator[torch.Tensor]:
    """Interleave `indonesian_parquet_stream` + `slimpajama_stream` at
    `id_share` ratio (default 0.6 = 60:40 ID:EN per
    `intllm_en.BILINGUAL_RATIO_DEFAULT`).

    Per-batch language choice via seeded RNG. Both source streams use
    full `(batch_size, seq_len)` batches, so an `id_share`-fraction of
    *batches* equals exactly that fraction of *tokens*. Long-run ratio
    converges to `id_share` ± O(1/sqrt(n_batches)).

    The two underlying streams use distinct seeds (`seed` for ID,
    `seed+1` for EN, `seed+2` for the per-batch coin flip) so the
    coin-flip RNG cannot accidentally correlate with shard ordering.
    """
    if not 0.0 <= id_share <= 1.0:
        raise ValueError(f"id_share must be in [0, 1], got {id_share}")

    id_iter = indonesian_parquet_stream(
        tokenizer=tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
        corpus_root=id_corpus_root,
        sources=id_sources,
        device=device,
        seed=seed,
    )
    en_iter = slimpajama_stream(
        tokenizer=tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
        repo=en_repo,
        device=device,
        seed=seed + 1,
    )

    rng = random.Random(seed + 2)
    while True:
        if rng.random() < id_share:
            yield next(id_iter)
        else:
            yield next(en_iter)
