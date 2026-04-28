#!/usr/bin/env python3
"""V32-prep F.5.1.2 — SmoothQuant PTQ eval harness.

Per `docs/FJQ_PHASE_F_F5_1_SMOOTHQUANT_PTQ_DESIGN.md` v1.0 §5.3:
load Phase D Mini final checkpoint, optionally apply SmoothQuant
calibration, measure held-out val_loss with SmoothQuant ON vs OFF, and
record verdict per §6 gate criteria (G1/G2/G3 → PASS/PARTIAL/FAIL).

Methodology (real eval path):

  1. Load `python/phase_d/checkpoints/mini/mini_final.pt`.
  2. Run `no_smoothquant` baseline: fresh model, val_loss measurement.
  3. Run `smoothquant` mode: fresh model, attach SmoothQuantCalibrator,
     calibrate on `bilingual_stream(id_share=0.6, seed=42)` for
     n-calibration-batches batches, validate gates per §4.6, fuse
     `s` into `(γ, W)` per §3.3, save side-car file, run val_loss.
  4. Compute `delta_vs_baseline = smoothquant.val_loss − no_smoothquant.val_loss`.
  5. Write JSON output with verdict per §6.1 gates G1 (no-regression,
     ≤+0.05 nat), G2 (quant-error ≥+0.10 — deferred; not measured here),
     G3 (val_loss improvement, ≤−0.05 nat).

Per F.5 entry condition ("paper submitted" + "≥+10% MSE reduction smoke"),
F.5.1 EXECUTION is gated; this script's `--dry-run` path is the
unblocked CPU-only smoke for V32-prep validation. Real eval execution
follows the F.6.1/F.6.2 precedent of allowing measurement-class work
ahead of the formal gate.

Output schema:

  paper/intllm/ablations/smoothquant_<sites>_a<α>.json:
    {
      "_schema_version": "1.0",
      "ckpt_path": str,
      "alpha": float,
      "sites": str | list[str],
      "calibration": {
        "n_sequences": int,
        "stream": "bilingual(id_share=0.6, seed=42)",
        "wall_clock_seconds": float,
        "validation_gates": {layer_name: {gate: pass}, "_aggregate": ...},
        "edge_cases_total": {n_act_clamped, n_w_clamped, n_s_clamped_*},
        "side_car_maps_path": str
      },
      "modes": {
        "no_smoothquant": {"val_loss": F, "ppl": F},
        "smoothquant":    {"val_loss": F, "ppl": F, "delta_vs_baseline": F}
      },
      "verdict": {
        "G1_no_regression":   bool,    # ≤+0.05 nat
        "G2_quant_error":     null,    # deferred; needs separate metric
        "G3_val_loss_improvement": bool,   # ≤-0.05 nat
        "outcome":            "STRONG-PASS"|"WEAK-PASS"|"NEUTRAL"|"HARD-FAIL"|...
        "interpretation":     str,
        "gate_nat_threshold": 0.05
      },
      "device":  "cuda"|"cpu",
      "timestamp": str (ISO 8601)
    }

Usage:

    cd python/phase_d

    # Smoke (CPU, no val): build + calibrate + validate, no final eval.
    PYTHONPATH=. ../../.venv/bin/python scripts/eval_smoothquant_posthoc.py --dry-run

    # Real eval (auto-GPU; Mini ~2 min on RTX 4090, ~30 min on CPU):
    PYTHONPATH=. ../../.venv/bin/python scripts/eval_smoothquant_posthoc.py \\
        --alpha 0.5 --sites o
"""

from __future__ import annotations

import argparse
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
DEFAULT_CKPT = ROOT / "checkpoints" / "mini" / "mini_final.pt"
ABLATIONS_DIR = REPO_ROOT / "paper" / "intllm" / "ablations"

# Gate threshold matches F.5.1 design §6.1 (also the E2.1/F.6.2 precedent).
F51_GATE_NAT = 0.05


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _build_and_load(ckpt_path: Path, arch, device: str):
    import torch
    from intllm.model import HGRNBitConfig, HGRNBitForCausalLM

    cfg = HGRNBitConfig(
        vocab_size=arch.vocab_size,
        hidden_size=arch.hidden_size,
        num_hidden_layers=arch.num_hidden_layers,
        max_position_embeddings=arch.max_position_embeddings,
    )
    model = HGRNBitForCausalLM(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _classify_outcome(g1: bool, g3: bool, delta: float, gate: float) -> tuple[str, str]:
    """Map (G1, G3) per §6.2 outcome table. G2 (quant-error) is not
    measured by this script (separate metric pass); flag as null in
    JSON, exclude from outcome label.
    """
    if not g1:
        return ("HARD-FAIL", f"val_loss regressed by {delta:+.4f} nat (gate -{gate:+.2f})")
    if g3:
        return ("STRONG-PASS-VAL", f"val_loss improved by {-delta:.4f} nat ≥ {gate:.2f}")
    if abs(delta) < gate:
        return ("NEUTRAL", f"|delta|={abs(delta):.4f} < gate {gate:.2f}; no measurable effect")
    return ("WEAK-PASS-VAL", f"val_loss improved by {-delta:.4f} nat (sub-gate)")


def main() -> int:
    p = argparse.ArgumentParser(description="V32-prep F.5.1.2 SmoothQuant PTQ eval.")
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                   help=f"Path to Phase D Mini final checkpoint. Default: {DEFAULT_CKPT}")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="SmoothQuant migration strength α ∈ [0, 1]. Default 0.5; F.5.1 §3.2 "
                   "expects ternary may want α<0.5; primary sweep covers [0.3, 0.7].")
    p.add_argument("--sites", default="o",
                   help="Site preset name (o|igfo|igf|o_down|all) OR comma-list of suffixes.")
    p.add_argument("--n-calibration-batches", type=int, default=64,
                   help="Calibration batches (default 64 ≈ 512 sequences at bs=8 per design §4.1).")
    p.add_argument("--n-val-batches", type=int, default=50,
                   help="Val batches for run_held_out_loss (default 50 to match Mini ablation).")
    p.add_argument("--device", default=None,
                   help="cpu|cuda. Default: cuda if available, else cpu.")
    p.add_argument("--out", type=Path, default=None,
                   help="JSON output path. Default: paper/intllm/ablations/smoothquant_<sites>_a<α>.json")
    p.add_argument("--out-maps", type=Path, default=None,
                   help="Side-car maps path. Default: same stem as --out with _maps.pt suffix.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build + calibrate + validate, no val_loss runs. CPU-only smoke (~30 sec).")
    p.add_argument("--strict-gates", action="store_true",
                   help="Abort if validation gates §4.6 fail (default: warn but proceed).")
    args = p.parse_args()

    if not args.ckpt.exists():
        print(f"[error] checkpoint not found: {args.ckpt}", file=sys.stderr)
        return 2

    import torch
    from configs.mini import MiniArchConfig, MiniTrainConfig
    from intllm.quant import SmoothQuantCalibrator, save_smoothquant_maps
    arch = MiniArchConfig()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sites_for_filename = args.sites.replace(",", "_").replace(".", "")

    if args.out is None:
        args.out = ABLATIONS_DIR / f"smoothquant_{sites_for_filename}_a{args.alpha}.json"
    if args.out_maps is None:
        args.out_maps = args.out.with_suffix("").with_suffix("") / "_maps.pt"
        args.out_maps = args.out.parent / f"{args.out.stem}_maps.pt"

    print(f"[1/5] device={device}, ckpt={args.ckpt}")
    print(f"      alpha={args.alpha}, sites={args.sites}")
    print(f"      n_calibration_batches={args.n_calibration_batches}, n_val_batches={args.n_val_batches}")
    print(f"      dry_run={args.dry_run}, strict_gates={args.strict_gates}")

    # Resolve sites: preset name OR comma-list
    if "," in args.sites:
        sites = [s.strip() for s in args.sites.split(",")]
    else:
        sites = args.sites  # preset name; SmoothQuantCalibrator resolves it

    if not args.dry_run:
        from intllm.data import bilingual_stream
        from intllm.eval import run_held_out_loss
        from intllm.tokenizer import get_tokenizer
        sys.path.insert(0, str(REPO_ROOT / "python" / "phase_e"))
        from intllm_en import BILINGUAL_RATIO_DEFAULT

        train_hp = MiniTrainConfig()
        print(f"[2/5] tokenizer + streams (bs={train_hp.batch_size}, seq={train_hp.seq_len})")
        tok = get_tokenizer()

    # ── PASS 1: no_smoothquant baseline ──────────────────────────────
    print("\n[3/5] mode=no_smoothquant (baseline)")
    model = _build_and_load(args.ckpt, arch, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      params: {n_params:,}")

    if args.dry_run:
        results_no = {"val_loss": float("nan"), "ppl": float("nan"), "dry_run": True}
    else:
        val_stream_bl = bilingual_stream(
            tokenizer=tok,
            seq_len=train_hp.seq_len,
            batch_size=train_hp.batch_size,
            id_share=BILINGUAL_RATIO_DEFAULT,
            device=device,
            seed=999,
        )
        val_loss_bl = run_held_out_loss(
            model, batches=val_stream_bl, n_steps=args.n_val_batches, device=device,
        )
        print(f"      val_loss = {val_loss_bl:.4f}  (PPL {math.exp(val_loss_bl):,.1f})")
        results_no = {"val_loss": float(val_loss_bl), "ppl": float(math.exp(val_loss_bl))}

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── PASS 2: smoothquant ──────────────────────────────────────────
    print(f"\n[4/5] mode=smoothquant (α={args.alpha}, sites={args.sites})")
    model = _build_and_load(args.ckpt, arch, device)
    cal = SmoothQuantCalibrator(model, alpha=args.alpha, sites=sites, device=device)

    if args.dry_run:
        # Build a tiny synthetic calibration batch (CPU) — just to exercise hooks
        print("      [DRY-RUN] using synthetic calibration batches (4 batches × bs=2 × seq=64)")
        synth_batches = [torch.randint(0, arch.vocab_size, (2, 64), device=device) for _ in range(4)]
        calibration_data = cal.calibrate(synth_batches, n_batches=4)
        cal_wall = 0.0
    else:
        calib_stream = bilingual_stream(
            tokenizer=tok,
            seq_len=train_hp.seq_len,
            batch_size=train_hp.batch_size,
            id_share=BILINGUAL_RATIO_DEFAULT,
            device=device,
            seed=42,
        )
        cal_t0 = time.time()
        calibration_data = cal.calibrate(calib_stream, n_batches=args.n_calibration_batches)
        cal_wall = time.time() - cal_t0
        print(f"      calibration wall: {cal_wall:.1f} s ({cal_wall / 60:.1f} min)")

    print(f"      sites matched: {len(calibration_data)}")
    gates = cal.validate(calibration_data)
    all_pass = gates["_aggregate"]["all_layers_pass"]
    print(f"      validation gates: aggregate all_pass={all_pass}, n_layers={gates['_aggregate']['n_layers']}")
    if not all_pass:
        msg = f"validation gates §4.6 failed; details: {gates}"
        if args.strict_gates:
            raise RuntimeError(msg)
        print(f"      [WARN] {msg}", file=sys.stderr)

    # Aggregate edge cases (sum across layers for at-a-glance diagnostic)
    edge_total = {"n_act_clamped": 0, "n_w_clamped": 0, "n_s_clamped_lo": 0, "n_s_clamped_hi": 0}
    for name, layer in calibration_data.items():
        for k in edge_total:
            edge_total[k] += int(layer["edge_cases"][k])
    print(f"      edge cases: {edge_total}")

    # Save side-car maps
    save_smoothquant_maps(
        args.out_maps, calibration_data,
        alpha=args.alpha,
        sites=args.sites,
        ckpt_path=str(args.ckpt),
        n_calibration_sequences=int(args.n_calibration_batches * (
            MiniTrainConfig().batch_size if not args.dry_run else 2
        )),
        calibration_seed=42,
        calibration_stream="bilingual(id_share=0.6, seed=42)" if not args.dry_run else "synthetic_dry_run",
    )
    print(f"      side-car maps: {args.out_maps}")

    # Apply fusion
    cal.apply(model, calibration_data, in_place=True)
    print("      §3.3 fusion applied (γ ← γ/s, W ← s·W)")

    if args.dry_run:
        results_sq = {"val_loss": float("nan"), "ppl": float("nan"), "dry_run": True}
    else:
        val_stream_sq = bilingual_stream(
            tokenizer=tok,
            seq_len=train_hp.seq_len,
            batch_size=train_hp.batch_size,
            id_share=BILINGUAL_RATIO_DEFAULT,
            device=device,
            seed=999,
        )
        val_loss_sq = run_held_out_loss(
            model, batches=val_stream_sq, n_steps=args.n_val_batches, device=device,
        )
        print(f"      val_loss = {val_loss_sq:.4f}  (PPL {math.exp(val_loss_sq):,.1f})")
        results_sq = {"val_loss": float(val_loss_sq), "ppl": float(math.exp(val_loss_sq))}

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── PASS 5: verdict + JSON ──────────────────────────────────────
    if not args.dry_run:
        delta = results_sq["val_loss"] - results_no["val_loss"]
        results_sq["delta_vs_baseline"] = float(delta)
        g1 = bool(delta <= F51_GATE_NAT)
        g3 = bool(delta <= -F51_GATE_NAT)
        outcome, interpretation = _classify_outcome(g1, g3, delta, F51_GATE_NAT)
        verdict = {
            "G1_no_regression": g1,
            "G2_quant_error": None,  # not measured; separate compute_quant_error_per_channel pass
            "G3_val_loss_improvement": g3,
            "delta_vs_baseline": float(delta),
            "outcome": outcome,
            "interpretation": interpretation,
            "gate_nat_threshold": F51_GATE_NAT,
        }
    else:
        verdict = {"dry_run": True}

    payload = {
        "_schema_version": "1.0",
        "ckpt_path": str(args.ckpt),
        "alpha": args.alpha,
        "sites": args.sites,
        "calibration": {
            "n_sequences": int(args.n_calibration_batches * (
                MiniTrainConfig().batch_size if not args.dry_run else 2
            )),
            "stream": "bilingual(id_share=0.6, seed=42)" if not args.dry_run else "synthetic_dry_run",
            "wall_clock_seconds": cal_wall,
            "validation_gates": gates,
            "edge_cases_total": edge_total,
            "side_car_maps_path": str(args.out_maps),
        },
        "modes": {
            "no_smoothquant": results_no,
            "smoothquant": results_sq,
        },
        "verdict": verdict,
        "device": device,
        "timestamp": _now_iso(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, args.out)

    print(f"\n[5/5] wrote {args.out}")
    if not args.dry_run:
        print(f"      delta_vs_baseline: {verdict['delta_vs_baseline']:+.4f} nat")
        print(f"      outcome: {verdict['outcome']}")
        print(f"      interpretation: {verdict['interpretation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
