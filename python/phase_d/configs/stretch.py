"""Stretch config — direct repro of Zhu et al. Table 1 370M row.

Per FJQ_PHASE_D_CONFIG.md §2: d=1024, L=24, V=32K Mistral v3, ~369.1M params.
Plan budget: 15B training tokens, ~17 days RTX 4090 wall-clock — the
biggest commit in Phase D.

Medium gate must PASS before launching Stretch. PASS conditions for
Stretch (FJQ_PHASE_D_CONFIG.md §5.4):
  - Match Zhu et al. Table 1 numbers within ±1 avg-point on 6 zero-shot
    tasks (repro check, §6.9 R3 baseline parity)
  - Wikitext-103 PPL within ±2 of FP16 Transformer++ at matched params
  - Kernel-path E2E on FajarOS Nova: no saturation, no collapse,
    tok/s ≥ 1 on consumer x86-64

If Stretch FAILS, paper submits at Medium with Stretch as future work.

Pre-staging requirement (§6.8 R3 prevention layer):
  - Pre-tokenize SlimPajama-627B-Reupload subset to disk before launch
  - cleanup_hf_cache.py monitor must keep ≥30 GB free during run
  - rotate checkpoints every 500 steps to bound disk growth

Batch=4 to stay under 16 GB VRAM with seq_len=2048 at L=24. Token
throughput compensates via n_steps.
"""

from __future__ import annotations

from dataclasses import dataclass

from intllm.tokenizer import DEFAULT_VOCAB_SIZE


@dataclass(frozen=True)
class StretchArchConfig:
    """Architecture: HGRNBitConfig kwargs for Stretch scale.

    Direct match to ridger/MMfreeLM-370M Zhu et al. config.
    """

    vocab_size: int = DEFAULT_VOCAB_SIZE  # 32,768 Mistral v3
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    max_position_embeddings: int = 2048


@dataclass(frozen=True)
class StretchTrainConfig:
    """Training hyper-parameters per FJQ_PHASE_D_CONFIG.md §4.

    The 17-day run: lowest lr in the matrix, longest warmup, most
    frequent checkpoints, most thorough validation.
    """

    seq_len: int = 2048
    batch_size: int = 4           # 4 × 2048 = 8,192 tokens/step (memory-bound at L=24)
    n_steps: int = 1_830_000      # 8,192 × 1.83M ≈ 15B tokens (Stretch target)
    lr: float = 5e-4              # half of Medium — biggest model wants slowest lr
    weight_decay: float = 0.1
    warmup_steps: int = 10_000
    grad_clip_norm: float = 1.0
    log_every: int = 500
    val_every: int = 50_000
    val_steps: int = 200
    ckpt_every: int = 50_000      # ~50,000 × 8K = 410M tokens between ckpts
