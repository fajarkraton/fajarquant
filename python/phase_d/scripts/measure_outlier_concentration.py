#!/usr/bin/env python3
"""Phase F.6.1 — outlier-concentration measurement on a trained Mini.

Hard prerequisite for any post-hoc QuaRot-style experiment per
`docs/FJQ_PHASE_F_TAX_VERTICAL_ROADMAP.md` v1.3 §4.1 F.6.1: if HGRN
o_proj inputs do NOT show outlier concentration (per-channel max-to-RMS
ratio close to 1, indicating uniform energy spread), then post-hoc
Hadamard rotation has nothing to spread either, and Phase F.6.2-6.4
become moot.

Also addresses the §7.3 cause-2 hindsight admission in the V31 paper:
"We did not empirically measure outlier concentration on o_proj inputs
before running this ablation." Running F.6.1 against the Phase D Mini
final checkpoint NOW gives the paper a concrete measurement to cite
rather than a hedge.

Methodology:
  1. Load python/phase_d/checkpoints/mini/mini_final.pt (Phase D
     English-only Mini c.1 checkpoint).
  2. Walk model.named_modules() to find every BitLinear (37 sites:
     6 layers × 6 BitLinears + lm_head).
  3. Attach forward pre-hooks that capture inputs at each site.
  4. Run 100 batches of bilingual_stream(id_share=0.6, seed=42)
     through the model in eval mode.
  5. For each captured input tensor, compute per-channel running
     absmax + per-channel running mean(|x|).
  6. After all 100 batches: compute per-channel max-to-mean ratio
     and per-layer aggregate stats (mean ratio, max ratio, p95).
  7. Save results to paper/intllm/ablations/outlier_concentration_mini.json.

Usage:
  cd python/phase_d
  PYTHONPATH=. ../../.venv/bin/python scripts/measure_outlier_concentration.py

CPU-bound (no GPU required for measurement-only forward; ~3 min on
RTX 4090 Laptop CPU host with 100 batches).

Output schema:
  {
    "_schema_version": "1.0",
    "checkpoint": "python/phase_d/checkpoints/mini/mini_final.pt",
    "n_batches": 100,
    "stream": "bilingual(id_share=0.6, seed=42)",
    "n_sites": 37,
    "global_summary": {
      "mean_ratio": float,    # mean of per-layer mean_ratio
      "max_ratio":  float,    # global max across all layers + channels
      "p95_ratio":  float,    # 95th-percentile across all channels
    },
    "per_layer": {
      "<layer_name>": {
        "in_features": int,
        "n_observations": int,
        "channel_max":  list[float],   # length in_features
        "channel_mean_abs": list[float],
        "ratio_mean": float,
        "ratio_max":  float,
        "ratio_p50":  float,
        "ratio_p95":  float,
        "n_outlier_channels_3x": int,  # ratio > 3.0
        "n_outlier_channels_5x": int,  # ratio > 5.0
      },
      ...
    },
    "by_role": {
      "i_proj":  { ... aggregated },
      "f_proj":  { ... aggregated },
      "g_proj":  { ... aggregated },
      "o_proj":  { ... aggregated },
      "mlp.gate_proj": { ... },
      "mlp.down_proj": { ... },
      "lm_head":  { ... },
    },
    "interpretation": str,    # human-readable summary
    "timestamp": str (ISO 8601),
  }
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
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
CHECKPOINT_DEFAULT = ROOT / "checkpoints" / "mini" / "mini_final.pt"
OUT_DEFAULT = REPO_ROOT / "paper" / "intllm" / "ablations" / "outlier_concentration_mini.json"


def role_of(layer_name: str) -> str:
    """Categorize a BitLinear site by its role in the HGRN block.
    Used for the by_role aggregation."""
    if "attn.i_proj" in layer_name:
        return "i_proj"
    if "attn.f_proj" in layer_name:
        return "f_proj"
    if "attn.g_proj" in layer_name:
        return "g_proj"
    if "attn.o_proj" in layer_name:
        return "o_proj"
    if "mlp.gate_proj" in layer_name:
        return "mlp.gate_proj"
    if "mlp.down_proj" in layer_name:
        return "mlp.down_proj"
    if layer_name == "lm_head":
        return "lm_head"
    return "other"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="F.6.1 outlier-concentration measurement.")
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DEFAULT)
    p.add_argument("--n-batches", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    import torch
    from configs.mini import MiniArchConfig
    from intllm.data import bilingual_stream
    from intllm.model import HGRNBitConfig, HGRNBitForCausalLM
    from intllm.qat import is_bitlinear
    from intllm.tokenizer import get_tokenizer

    sys.path.insert(0, str(REPO_ROOT / "python" / "phase_e"))
    from intllm_en import BILINGUAL_RATIO_DEFAULT

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/5] device: {device}")
    print(f"      checkpoint: {args.checkpoint}")
    print(f"      n_batches: {args.n_batches}")

    print("[2/5] loading checkpoint")
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    arch_dict = ckpt["arch"]
    arch = MiniArchConfig(**arch_dict)
    cfg = HGRNBitConfig(
        vocab_size=arch.vocab_size,
        hidden_size=arch.hidden_size,
        num_hidden_layers=arch.num_hidden_layers,
        max_position_embeddings=arch.max_position_embeddings,
    )
    model = HGRNBitForCausalLM(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"      params: {sum(p.numel() for p in model.parameters()):,}")

    # ── attach pre-hooks on every BitLinear ─────────────────────────
    print("[3/5] attaching forward pre-hooks on BitLinear sites")
    # Per-site state: running absmax + running sum-of-abs + count.
    state: dict[str, dict] = {}
    name_by_module: dict[int, str] = {}
    for name, module in model.named_modules():
        if not is_bitlinear(module):
            continue
        in_f = module.in_features
        state[name] = {
            "in_features": in_f,
            "running_max": torch.zeros(in_f, dtype=torch.float32, device=device),
            "running_sum_abs": torch.zeros(in_f, dtype=torch.float32, device=device),
            "n_obs": 0,
        }
        name_by_module[id(module)] = name

    def make_hook(layer_name: str):
        s = state[layer_name]

        def _pre_hook(_module, inputs):
            x = inputs[0].detach().reshape(-1, s["in_features"]).float()
            torch.maximum(s["running_max"], x.abs().amax(dim=0), out=s["running_max"])
            s["running_sum_abs"].add_(x.abs().sum(dim=0))
            s["n_obs"] += int(x.shape[0])
        return _pre_hook

    handles = []
    for module in model.modules():
        if id(module) in name_by_module:
            layer_name = name_by_module[id(module)]
            handles.append(module.register_forward_pre_hook(make_hook(layer_name)))
    print(f"      attached to {len(handles)} BitLinear sites")

    # ── stream batches through the model ────────────────────────────
    print(f"[4/5] streaming {args.n_batches} batches through model.eval()")
    tok = get_tokenizer()
    batches = bilingual_stream(
        tokenizer=tok,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        id_share=BILINGUAL_RATIO_DEFAULT,
        device=device,
        seed=42,
    )

    t0 = time.time()
    with torch.no_grad():
        for i, ids in enumerate(batches):
            if i >= args.n_batches:
                break
            _ = model(input_ids=ids, labels=ids)
            if (i + 1) % 25 == 0:
                print(f"      batch {i+1}/{args.n_batches}  ({time.time()-t0:.1f}s elapsed)")

    elapsed = time.time() - t0
    print(f"      elapsed: {elapsed:.1f} s")

    for h in handles:
        h.remove()

    # ── compute per-layer + global stats ────────────────────────────
    print("[5/5] computing per-layer stats")
    per_layer: dict[str, dict] = {}
    by_role: dict[str, list] = {}
    all_ratios = []
    for layer_name in sorted(state.keys()):
        s = state[layer_name]
        n = s["n_obs"]
        if n == 0:
            continue
        chan_max = s["running_max"].cpu().tolist()
        chan_mean_abs = (s["running_sum_abs"] / max(n, 1)).cpu().tolist()
        # Per-channel ratio = max(|x|) / mean(|x|); use eps to avoid /0
        ratios = [m / (mu + 1e-12) for m, mu in zip(chan_max, chan_mean_abs)]
        ratios_t = torch.tensor(ratios)
        layer_summary = {
            "in_features": s["in_features"],
            "n_observations": n,
            "channel_max": chan_max,
            "channel_mean_abs": chan_mean_abs,
            "ratio_mean": float(ratios_t.mean()),
            "ratio_max":  float(ratios_t.max()),
            "ratio_p50":  float(ratios_t.median()),
            "ratio_p95":  float(ratios_t.quantile(0.95)),
            "n_outlier_channels_3x": int((ratios_t > 3.0).sum()),
            "n_outlier_channels_5x": int((ratios_t > 5.0).sum()),
        }
        per_layer[layer_name] = layer_summary
        by_role.setdefault(role_of(layer_name), []).append(layer_summary)
        all_ratios.extend(ratios)

    # By-role aggregates: mean over each layer's ratio_mean within role
    by_role_agg: dict[str, dict] = {}
    for role, layer_list in by_role.items():
        by_role_agg[role] = {
            "n_layers": len(layer_list),
            "mean_of_ratio_mean": float(sum(l["ratio_mean"] for l in layer_list) / len(layer_list)),
            "max_of_ratio_max": float(max(l["ratio_max"] for l in layer_list)),
            "mean_of_ratio_p95": float(sum(l["ratio_p95"] for l in layer_list) / len(layer_list)),
            "total_outlier_channels_3x": int(sum(l["n_outlier_channels_3x"] for l in layer_list)),
            "total_outlier_channels_5x": int(sum(l["n_outlier_channels_5x"] for l in layer_list)),
        }

    all_ratios_t = torch.tensor(all_ratios)
    global_summary = {
        "n_layers_measured": len(per_layer),
        "n_channels_total": len(all_ratios),
        "mean_ratio": float(all_ratios_t.mean()),
        "max_ratio":  float(all_ratios_t.max()),
        "p95_ratio":  float(all_ratios_t.quantile(0.95)),
        "n_outlier_channels_3x_global": int((all_ratios_t > 3.0).sum()),
        "n_outlier_channels_5x_global": int((all_ratios_t > 5.0).sum()),
    }

    # Interpretation: F.6 entry condition is ≥3× max/mean ratio on at
    # least one BitLinear site. Apply it here to set the
    # F.6-go/no-go disposition.
    o_proj = by_role_agg.get("o_proj", {})
    o_proj_max = o_proj.get("max_of_ratio_max", 0.0)
    if o_proj_max >= 5.0:
        interp = (
            f"o_proj inputs show STRONG outlier concentration "
            f"(per-layer-max max/mean = {o_proj_max:.2f}). Phase F.6 "
            f"post-hoc QuaRot is RECOMMENDED — Hadamard rotation has "
            f"meaningful outlier energy to spread."
        )
    elif o_proj_max >= 3.0:
        interp = (
            f"o_proj inputs show MODERATE outlier concentration "
            f"(per-layer-max max/mean = {o_proj_max:.2f}). Phase F.6 "
            f"meets the ≥3× entry threshold; expect modest improvement."
        )
    else:
        interp = (
            f"o_proj inputs show LOW outlier concentration "
            f"(per-layer-max max/mean = {o_proj_max:.2f}). Phase F.6 "
            f"does NOT meet the ≥3× entry threshold; post-hoc Hadamard "
            f"has little outlier energy to spread; F.6 likely moot."
        )

    payload = {
        "_schema_version": "1.0",
        "phase": "F.6.1",
        "checkpoint": str(args.checkpoint.relative_to(REPO_ROOT)) if args.checkpoint.is_absolute()
                      else str(args.checkpoint),
        "n_batches": args.n_batches,
        "stream": f"bilingual(id_share={BILINGUAL_RATIO_DEFAULT}, seed=42)",
        "n_sites": len(per_layer),
        "elapsed_seconds": elapsed,
        "global_summary": global_summary,
        "by_role": by_role_agg,
        "per_layer": per_layer,
        "interpretation": interp,
        "timestamp": now_iso(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, args.out)

    print()
    print("  ── F.6.1 outlier-concentration measurement DONE ──")
    print(f"  global mean_ratio       : {global_summary['mean_ratio']:.3f}")
    print(f"  global max_ratio        : {global_summary['max_ratio']:.3f}")
    print(f"  global p95_ratio        : {global_summary['p95_ratio']:.3f}")
    print(f"  outlier channels (>3×)  : {global_summary['n_outlier_channels_3x_global']}")
    print(f"  outlier channels (>5×)  : {global_summary['n_outlier_channels_5x_global']}")
    print()
    print("  By role:")
    for role, agg in sorted(by_role_agg.items()):
        print(f"    {role:18s}  n={agg['n_layers']}  mean_ratio={agg['mean_of_ratio_mean']:.2f}  "
              f"max_ratio={agg['max_of_ratio_max']:.2f}  outliers>3×={agg['total_outlier_channels_3x']}")
    print()
    print(f"  {interp}")
    print()
    print(f"  JSON: {args.out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
