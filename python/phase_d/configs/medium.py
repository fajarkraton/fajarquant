"""Medium config — first vs-Transformer++ table-2 data point.

Per FJQ_PHASE_D_CONFIG.md §2: d=512, L=12, V=32K Mistral v3, ~71.3M params.
Plan budget: 2B training tokens, ~11h RTX 4090 wall-clock.

Base gate must PASS before launching Medium. PASS conditions for
Medium (FJQ_PHASE_D_CONFIG.md §5.3):
  - Wikitext-103 PPL gap vs Transformer++ at matched params ≤ 3
  - MMLU 5-shot ≥ 22%
  - FajarOS Nova E2E boot+ask yields coherent output (3 prompts)

If Medium FAILS, Stretch is aborted and the paper submits at Medium
scale only — per §6.9 R6 results-section text comes LAST.

Same seq_len as Base for compute-cost continuity; lr stepped down 2×
because larger d_model needs slightly slower learning.
"""

from __future__ import annotations

from dataclasses import dataclass

from intllm.tokenizer import DEFAULT_VOCAB_SIZE


@dataclass(frozen=True)
class MediumArchConfig:
    """Architecture: HGRNBitConfig kwargs for Medium scale."""

    vocab_size: int = DEFAULT_VOCAB_SIZE  # 32,768 Mistral v3
    hidden_size: int = 512
    num_hidden_layers: int = 12
    max_position_embeddings: int = 2048


@dataclass(frozen=True)
class MediumTrainConfig:
    """Training hyper-parameters per FJQ_PHASE_D_CONFIG.md §4."""

    seq_len: int = 2048
    batch_size: int = 8           # 8 × 2048 = 16,384 tokens/step
    n_steps: int = 122_000        # 16,384 × 122K ≈ 2.0B tokens (~Medium target)
    lr: float = 1e-3              # 2× lower than Base — d=512 wants slower lr
    weight_decay: float = 0.1
    warmup_steps: int = 4_000
    grad_clip_norm: float = 1.0
    log_every: int = 200
    val_every: int = 10_000
    val_steps: int = 100
    ckpt_every: int = 20_000
