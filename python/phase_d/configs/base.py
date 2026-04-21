"""Base config — primary FP-shadow vs QAT comparison target.

Per FJQ_PHASE_D_CONFIG.md §2: d=384, L=12, V=32K Mistral v3, ~46.4M params.
Plan budget: 1B training tokens, ~4h RTX 4090 wall-clock.

C.P4.1 Mini gate must PASS before launching Base. PASS conditions for
Base (FJQ_PHASE_D_CONFIG.md §5.2):
  - Wikitext-103 PPL ≤ 25 (Transformer++ baseline ~24 at 46M)
  - LAMBADA-o acc ≥ 38%
  - No saturation events in kernel FJTRACE (§6.9 R5)

Effective batch size kept modest (8) to share infrastructure with the
Mini driver and stay well under the 16 GB VRAM ceiling. Token throughput
scales by raising seq_len + n_steps, not batch.
"""

from __future__ import annotations

from dataclasses import dataclass

from intllm.tokenizer import DEFAULT_VOCAB_SIZE


@dataclass(frozen=True)
class BaseArchConfig:
    """Architecture: HGRNBitConfig kwargs for Base scale."""

    vocab_size: int = DEFAULT_VOCAB_SIZE  # 32,768 Mistral v3
    hidden_size: int = 384
    num_hidden_layers: int = 12
    max_position_embeddings: int = 2048


@dataclass(frozen=True)
class BaseTrainConfig:
    """Training hyper-parameters per FJQ_PHASE_D_CONFIG.md §4."""

    seq_len: int = 2048
    batch_size: int = 8           # 8 × 2048 = 16,384 tokens/step
    n_steps: int = 60_000         # 16,384 × 60K ≈ 982M tokens (~Base 1B target)
    lr: float = 2e-3              # paper §4.3 stable range; same as Mini
    weight_decay: float = 0.1
    warmup_steps: int = 2_000
    grad_clip_norm: float = 1.0
    log_every: int = 100
    val_every: int = 5_000
    val_steps: int = 100
    ckpt_every: int = 10_000
