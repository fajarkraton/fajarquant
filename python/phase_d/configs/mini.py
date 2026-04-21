"""Mini config — fast iteration target per FJQ_PHASE_D_CONFIG.md §2.

This is the C.P4.1 primary gate config. Trains in ~1h on RTX 4090
Laptop, hits ~21.5M params on Mistral v3 (V=32,768).

Importing this module does NOT instantiate the model — it only
exposes config dataclasses + constants. The driver script
`scripts/train_mini.py` consumes these.
"""

from __future__ import annotations

from dataclasses import dataclass

from intllm.tokenizer import DEFAULT_VOCAB_SIZE


@dataclass(frozen=True)
class MiniArchConfig:
    """Architecture: HGRNBitConfig kwargs for Mini scale."""

    vocab_size: int = DEFAULT_VOCAB_SIZE  # 32,768 Mistral v3
    hidden_size: int = 256
    num_hidden_layers: int = 6
    max_position_embeddings: int = 1024


@dataclass(frozen=True)
class MiniTrainConfig:
    """Training hyper-parameters per FJQ_PHASE_D_CONFIG.md §4."""

    seq_len: int = 1024
    batch_size: int = 8           # 8 × 1024 = 8,192 tokens/step
    n_steps: int = 60_000         # 8,192 × 60K ≈ 491M tokens (~Mini 500M target)
    lr: float = 2e-3              # paper §4.3 says ≤ 2e-2 stable; we stay 10× safer
    weight_decay: float = 0.1
    warmup_steps: int = 2_000
    grad_clip_norm: float = 1.0
    log_every: int = 100
    val_every: int = 5_000        # eval val loss this often
    val_steps: int = 100          # 100 batches × 8 × 1024 ≈ 800K val tokens
    ckpt_every: int = 10_000      # save every ~80M tokens
