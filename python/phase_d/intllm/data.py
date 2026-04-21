"""IntLLM dataset loaders.

Three modes:
  - `synthetic_token_batches` — fresh random tokens per batch (entropy
    floor = ln(V); no learnable structure; useful for "no-NaN" smoke
    tests but **cannot** verify loss-decrease).
  - `overfit_token_batches`   — one fixed batch, repeated N times. Model
    memorizes; loss drops toward 0. Convergence sanity check.
  - `slimpajama_stream`       — real text via HF datasets streaming,
    tokenised on the fly with the Mistral v3 tokenizer (C.P3.3).

The C.P3.3 streaming loader is the production training path — no
on-disk pre-staging needed for Mini/Base (≤2B tokens). For Stretch
(15B), pre-tokenize and shard offline (deferred to C.P4).
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from transformers import PreTrainedTokenizerBase

# DKYoon/SlimPajama-6B is the default Mini/Base/Medium source —
# small enough to stream easily, large enough to cover ≤2B-token runs.
# Stretch (15B) pivots to gmongaras/SlimPajama-627B_Reupload.
DEFAULT_SLIMPAJAMA_REPO = "DKYoon/SlimPajama-6B"


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

    ds = load_dataset(repo, split=split, streaming=True).shuffle(seed=seed, buffer_size=10_000)
    eos_id = tokenizer.eos_token_id
    target_per_batch = batch_size * seq_len
    buf: list[int] = []
    for example in ds:
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
