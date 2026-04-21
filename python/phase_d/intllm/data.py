"""IntLLM dataset loaders.

C.P2.3 ships only synthetic-token loaders — enough to verify the
training stack converges. Real corpus loaders (SlimPajama streaming)
land in C.P3.

Two synthetic modes:
  - `synthetic_token_batches` — fresh random tokens per batch (entropy
    floor = ln(V); no learnable structure; useful for "no-NaN" smoke
    tests but **cannot** verify loss-decrease).
  - `overfit_token_batches`   — one fixed batch, repeated N times. Model
    memorizes; loss drops toward 0. **Use this** to verify the training
    stack converges end-to-end.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch


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
