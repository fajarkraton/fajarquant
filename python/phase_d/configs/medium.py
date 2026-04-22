"""Medium config — first vs-Transformer++ table-2 data point.

Per FJQ_PHASE_D_CONFIG.md §2: d=512, L=12, V=32K Mistral v3.
Measured params (2026-04-23 POL): 74,523,648 (74.5M).
Chinchilla-optimal target: 74.5M × 20 tok/param = 1.49B tokens.
Plan budget: 1.49B training tokens at 91K steps × 0.84s/step measured
on RTX 4090 Laptop = ~21h wall-clock.

Base gate must PASS before launching Medium. PASS conditions for
Medium (FJQ_PHASE_D_CONFIG.md §5.3):
  - Wikitext-103 PPL gap vs Transformer++ at matched params ≤ 3
  - MMLU 5-shot ≥ 22%
  - FajarOS Nova E2E boot+ask yields coherent output (3 prompts)

If Medium FAILS, Stretch is aborted and the paper submits at Medium
scale only — per §6.9 R6 results-section text comes LAST.

Same seq_len as Base for compute-cost continuity; lr stepped down 2×
because larger d_model needs slightly slower learning.

Token budget calibration history:
  - 2026-04 initial: 122K steps = 2.0B tokens (1.34× Chinchilla, over-trained)
  - 2026-04-23 trimmed: 91K steps = 1.49B tokens (Chinchilla-optimal, ~21h)
    Rationale: match Mini + Base (both Chinchilla-optimal ~20 tok/param)
    for clean scaling-plot story. Trimming saves ~7h without meaningful
    quality loss (Chinchilla point is optimal by Kaplan/Hoffmann curve).
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
    n_steps: int = 91_000         # 16,384 × 91K ≈ 1.49B tokens = 20 tok/param Chinchilla
    lr: float = 1e-3              # 2× lower than Base — d=512 wants slower lr
    weight_decay: float = 0.1
    warmup_steps: int = 3_000     # 3.3% of n_steps (matches Mini + Base ratio)
    grad_clip_norm: float = 1.0
    log_every: int = 200
    val_every: int = 10_000
    val_steps: int = 100
    ckpt_every: int = 20_000
