#!/usr/bin/env python3
"""Phase E E2.0+Q5 — bilingual coherence baseline at Mini scale.

Closes the last open blocker for E2.0 → v1.1 per
`docs/FJQ_PHASE_E_E2_FINDINGS.md` §5: trains a Mini-scale HGRNBit on
the bilingual corpus (60:40 ID:EN, 200M tokens) and measures the
**ratio val_loss(ID) / val_loss(EN)** so subsequent E2.5 ablation
deltas can be interpreted against a real "no-feature" baseline. The
E2.5.4 gate is "ratio ≤ 1.5× at Mini scale"; without this baseline we
cannot tell whether a given language-conditioning option *helps* or
just shifts the both-languages-equally floor.

This is NOT an ablation row — it does not produce
`paper/intllm/ablations/mini_<TAG>.json`. It writes
`paper/intllm/ablations/q5_bilingual_baseline.json` with a different
schema (ID/EN losses + ratio) so `verify_paper_tables.py` does not
treat it as an E2.x feature row.

Usage:

    cd python/phase_d

    # Smoke (200 steps, ~5 min on RTX 4090)
    PYTHONPATH=. ../../.venv/bin/python scripts/q5_bilingual_baseline.py \\
        --proof-of-life

    # Full Q5 baseline (~1.5–3 h on RTX 4090, 200M tokens at 60:40)
    PYTHONPATH=. ../../.venv/bin/python scripts/q5_bilingual_baseline.py

JSON schema written:

    {
      "_schema_version": "1.0",
      "phase": "E2.0+Q5",
      "id_share": 0.6,
      "n_steps": int,
      "training_seconds": float,
      "initial_loss": float,
      "final_loss": float,
      "val_loss_id": float,
      "val_loss_en": float,
      "ratio": float,            # max(id, en) / min(id, en)
      "ppl_id": float,
      "ppl_en": float,
      "model_params": int,
      "proof_of_life": bool,
      "timestamp": str (ISO 8601)
    }
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPO_ROOT = ROOT.parent.parent
ABLATIONS_DIR = REPO_ROOT / "paper" / "intllm" / "ablations"

# 200M tokens / (8 batch × 1024 seq) = 24,414 steps; round to 24,000 for
# warmup-friendly multiples (Mini default warmup_steps=2000).
DEFAULT_N_STEPS_FOR_200M_TOK = 24_000

# E2.5.4 ratio gate per plan v1.8 §3 PHASE E2.
E2_5_4_RATIO_GATE = 1.5


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_q5_json(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, out_path)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Phase E2.0+Q5 bilingual coherence baseline (Mini scale).",
    )
    p.add_argument(
        "--proof-of-life",
        action="store_true",
        help="Cap at 200 steps + 5 val batches each side; no checkpointing.",
    )
    p.add_argument(
        "--n-steps", type=int, default=None,
        help=f"Override training steps (default {DEFAULT_N_STEPS_FOR_200M_TOK}).",
    )
    p.add_argument(
        "--id-share", type=float, default=0.6,
        help="ID:EN ratio (default 0.6 = 60:40 per intllm_en.BILINGUAL_RATIO_DEFAULT).",
    )
    p.add_argument(
        "--val-steps", type=int, default=50,
        help="Number of val batches per language (default 50).",
    )
    p.add_argument(
        "--ckpt-dir",
        type=Path,
        default=ROOT / "checkpoints" / "q5_bilingual_baseline",
    )
    p.add_argument("--device", default=None)
    p.add_argument(
        "--out", type=Path, default=None,
        help="Override output JSON. Default: paper/intllm/ablations/q5_bilingual_baseline.json",
    )
    args = p.parse_args()

    out_path = args.out or (ABLATIONS_DIR / "q5_bilingual_baseline.json")

    # ── Imports (after sys.path setup) ────────────────────────────────
    import torch
    from configs.mini import MiniArchConfig, MiniTrainConfig
    from intllm.data import bilingual_stream, indonesian_parquet_stream, slimpajama_stream
    from intllm.eval import run_held_out_loss
    from intllm.model import HGRNBitConfig, HGRNBitForCausalLM
    from intllm.tokenizer import get_tokenizer
    from intllm.train import TrainConfig, train_loop

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    arch = MiniArchConfig()
    train_hp = MiniTrainConfig()

    if args.proof_of_life:
        train_hp = replace(train_hp, n_steps=200, log_every=25)
        val_steps_each = 5
    else:
        n_steps = args.n_steps if args.n_steps is not None else DEFAULT_N_STEPS_FOR_200M_TOK
        train_hp = replace(train_hp, n_steps=n_steps)
        val_steps_each = args.val_steps

    print(f"[1/5] config: arch={asdict(arch)}")
    print(f"      train: {asdict(train_hp)}")
    print(f"      device: {device}")
    print(f"      id_share: {args.id_share} (target {int(args.id_share*100)}:{100-int(args.id_share*100)} ID:EN)")

    print("[2/5] tokenizer + bilingual data stream")
    tok = get_tokenizer()
    train_batches = bilingual_stream(
        tokenizer=tok,
        seq_len=train_hp.seq_len,
        batch_size=train_hp.batch_size,
        id_share=args.id_share,
        device=device,
        seed=0,
    )

    print("[3/5] model")
    cfg = HGRNBitConfig(
        vocab_size=arch.vocab_size,
        hidden_size=arch.hidden_size,
        num_hidden_layers=arch.num_hidden_layers,
        max_position_embeddings=arch.max_position_embeddings,
    )
    model = HGRNBitForCausalLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      params: {n_params:,}")

    def capped_batches():
        for i, b in enumerate(train_batches):
            if i >= train_hp.n_steps:
                break
            yield b

    print(f"[4/5] training {train_hp.n_steps} steps on bilingual stream")
    if args.proof_of_life:
        warmup, total = 0, 0
    else:
        warmup, total = train_hp.warmup_steps, train_hp.n_steps
    ckpt_every = 0 if args.proof_of_life else train_hp.ckpt_every

    t0 = time.time()
    result = train_loop(
        model,
        capped_batches(),
        config=TrainConfig(
            lr=train_hp.lr,
            weight_decay=train_hp.weight_decay,
            grad_clip_norm=train_hp.grad_clip_norm,
            log_every=train_hp.log_every,
            warmup_steps=warmup,
            total_steps=total,
            min_lr_ratio=0.1,
            ckpt_every=ckpt_every,
            ckpt_dir=str(args.ckpt_dir) if ckpt_every > 0 else None,
            keep_last_n_ckpts=3,
            watchdog_idle_seconds=0 if args.proof_of_life else 1800,
        ),
    )
    elapsed = time.time() - t0
    print(f"      elapsed       : {elapsed:.1f} s ({elapsed/60:.1f} min)")
    print(f"      initial loss  : {result.initial_loss:.4f}")
    print(f"      final loss    : {result.final_loss:.4f}")

    print(f"[5/5] measuring val_loss on ID-only and EN-only streams ({val_steps_each} batches each)")

    id_val_stream = indonesian_parquet_stream(
        tokenizer=tok,
        seq_len=train_hp.seq_len,
        batch_size=train_hp.batch_size,
        device=device,
        seed=999,
    )
    val_loss_id = run_held_out_loss(
        model, batches=id_val_stream, n_steps=val_steps_each, device=device,
    )
    print(f"      val_loss_id   : {val_loss_id:.4f}  (PPL {math.exp(val_loss_id):,.1f})")

    en_val_stream = slimpajama_stream(
        tokenizer=tok,
        seq_len=train_hp.seq_len,
        batch_size=train_hp.batch_size,
        device=device,
        seed=998,
    )
    val_loss_en = run_held_out_loss(
        model, batches=en_val_stream, n_steps=val_steps_each, device=device,
    )
    print(f"      val_loss_en   : {val_loss_en:.4f}  (PPL {math.exp(val_loss_en):,.1f})")

    ratio = max(val_loss_id, val_loss_en) / min(val_loss_id, val_loss_en)
    higher_lang = "ID" if val_loss_id >= val_loss_en else "EN"
    gate_pass = ratio <= E2_5_4_RATIO_GATE
    print(f"      ratio         : {ratio:.4f}  (worse side: {higher_lang})")
    print(f"      E2.5.4 gate   : ratio ≤ {E2_5_4_RATIO_GATE} → {'PASS ✓' if gate_pass else 'OBS — coherence gap present'}")

    payload = {
        "_schema_version": "1.0",
        "phase": "E2.0+Q5",
        "id_share": args.id_share,
        "n_steps": result.steps,
        "training_seconds": elapsed,
        "initial_loss": result.initial_loss,
        "final_loss": result.final_loss,
        "val_loss_id": val_loss_id,
        "val_loss_en": val_loss_en,
        "ratio": ratio,
        "ppl_id": math.exp(val_loss_id),
        "ppl_en": math.exp(val_loss_en),
        "ratio_gate_threshold": E2_5_4_RATIO_GATE,
        "ratio_gate_pass": gate_pass,
        "higher_loss_language": higher_lang,
        "model_params": n_params,
        "proof_of_life": args.proof_of_life,
        "val_steps_each": val_steps_each,
        "timestamp": _now_iso(),
    }
    _write_q5_json(out_path, payload)
    print(f"\n  Q5 baseline JSON: {out_path}")

    del model
    del tok
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
