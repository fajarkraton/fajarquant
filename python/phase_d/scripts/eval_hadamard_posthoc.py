#!/usr/bin/env python3
"""V32-prep F.6.2 — post-hoc Hadamard rotation eval on trained Mini checkpoint.

⚠ SCOPE CAVEAT (read first): this script tests the ACTIVATION-ONLY
PRE-HOOK variant of rotation (`x → Hx` injected via forward-pre-hook
on BitLinear inputs), NOT the canonical QuaRot/SpinQuant weight-fusion
recipe. The pre-hook BREAKS the FP path:

    canonical:    y = (W·Hᵀ) · (H·x) = W·x        (preserved exactly)
    pre-hook:     y = W · (H·x)         ≠ W·x      (computation altered)

Any trained model is EXPECTED to regress under the pre-hook variant
because downstream layers (residual + RMSNorm + MLP) were trained to
expect `Wx`, not `WHx`. A `rotation_hurts` verdict from this script
CANNOT be used to claim "HGRN architecture is incompatible with
rotation" — that conflates recipe-incompleteness with architectural
incompatibility. Disambiguating (a) "training-from-scratch fights
rotation" vs (b) "HGRN-rotation incompatibility" requires the FULL
canonical recipe:

    1. Compute orthogonal Hadamard H of size hidden_size
    2. Fuse H into weights: W' = W·Hᵀ (so y = W'·Hx = W·x in FP)
    3. Apply matching rotation in residual stream (RMSNorm γ + entry
       projections) so the input to attn block is genuinely H·x_residual
    4. Re-calibrate per-channel γ_x for the rotated input distribution

This script implements step 1 only as a forward-pre-hook (no step 2,
3, 4). Useful as a SANITY CHECK that the rotation primitive composes
with HGRN's BitLinear layers and the existing E2.1.2 attach pattern;
NOT useful as a paper-claim about HGRN's compatibility with rotation.

What F.6.2 (this incomplete variant) DOES tell us:
  - Whether the pre-hook attach mechanics work end-to-end on a
    trained checkpoint (yes if eval completes without crash)
  - The MAGNITUDE of FP-path breakage as a function of which sites
    are rotated — i.e. how heavily downstream layers depend on the
    exact pre-rotation activation distribution

What F.6.2 DOES NOT tell us:
  - Whether HGRN benefits from canonical (weight-fused) rotation
  - Whether E2.1's negative was training-from-scratch vs architectural
  - Whether rotation_hurts here generalizes to other architectures
    (it doesn't — same recipe would also hurt a transformer; this
    is a recipe-class verdict, not an architecture-class verdict)

F.6.1 (commit pre-existing) measured outlier concentration on o_proj
inputs at 51.6× mean / 421× max ratio — STRONG concentration, so the
canonical recipe REMAINS WORTH TESTING in a future first-step.

Methodology:

  1. Load python/phase_d/checkpoints/mini/mini_final.pt.
  2. For each mode in ["no_rotation", "o", "igf", "igfo"]:
       a. Re-build a fresh HGRNBitForCausalLM, load state_dict.
       b. Attach HadamardRotation pre-hooks per mode (skip for
          "no_rotation"). All sites share a single HadamardRotation
          (hidden_size) module — safe per F.6.3 dim audit (commit
          0c6839c).
       c. Run `run_held_out_loss` over `n_val_batches` batches of
          `bilingual_stream(id_share=0.6, seed=999)` — same val
          stream the Mini ablation harness uses, so deltas are
          apples-to-apples vs Q5 baseline.
       d. Detach hooks, free model, record val_loss.
  3. Compute deltas vs no_rotation. Verdict: "rotation helps" if
     ANY rotated mode beats no_rotation by >=0.05 nat (matches the
     E2.1 gate threshold). "neutral" if within ±0.05 nat. "hurts"
     if all rotated modes are >=0.05 nat worse.

Output schema (paper/intllm/ablations/posthoc_hadamard_mini.json):
  {
    "_schema_version": "1.0",
    "ckpt_path": "...",
    "n_val_batches": 50,
    "stream": "bilingual(id_share=0.6, seed=999)",
    "modes": {
      "no_rotation": {"val_loss": F, "ppl": F},
      "o":   {"val_loss": F, "ppl": F, "delta_vs_baseline": F, "n_attached": int},
      "igf": {"val_loss": F, "ppl": F, "delta_vs_baseline": F, "n_attached": int},
      "igfo": {"val_loss": F, "ppl": F, "delta_vs_baseline": F, "n_attached": int},
    },
    "verdict": {
      "best_mode": str,
      "rotation_helps": bool,   # any mode beats baseline by 0.05+ nat
      "f62_outcome": str,       # "training-from-scratch was the issue" |
                                # "HGRN-architecture issue (online-QuaRot variant)" |
                                # "ambiguous"
    },
    "timestamp": str (ISO 8601)
  }

Usage:
  cd python/phase_d

  # Smoke (CPU, no val): verify script imports + ckpt load + hook attach.
  PYTHONPATH=. ../../.venv/bin/python scripts/eval_hadamard_posthoc.py --dry-run

  # Real eval (auto-detects GPU; Mini ~2 min on RTX 4090, ~30 min on CPU):
  PYTHONPATH=. ../../.venv/bin/python scripts/eval_hadamard_posthoc.py

NOTE: This is V32-prep work. Per Phase F roadmap §4.1 entry condition
("Phase E paper accepted (or at least submitted)"), F.6.2 execution is
formally gated on paper submission. F.6.1 (commit pre-existing) was
allowed to run pre-submission as measurement-only; F.6.2 follows the
same precedent — measurement-only, no training, single forward pass.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
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
OUT_PATH = REPO_ROOT / "paper" / "intllm" / "ablations" / "posthoc_hadamard_mini.json"

SITE_SUFFIXES = {
    "o":    [".attn.o_proj"],
    "igfo": [".attn.i_proj", ".attn.f_proj", ".attn.g_proj", ".attn.o_proj"],
    "igf":  [".attn.i_proj", ".attn.f_proj", ".attn.g_proj"],
}

# Gate threshold matches E2.1 closure: |delta| < 0.05 = neutral, >=+0.05 = improvement.
F62_GATE_NAT = 0.05


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _attach_hadamard_hooks(model, mode: str, hidden_size: int, device: str):
    """Attach HadamardRotation forward-pre-hook to every BitLinear matching
    the suffix list for `mode`. Returns (handles, n_attached, hadamard_module)
    so the caller can detach + free.
    """
    from intllm.qat import is_bitlinear
    from intllm.quant import HadamardRotation

    if mode == "no_rotation":
        return [], 0, None

    hadamard_module = HadamardRotation(hidden_size).to(device)

    def _pre_hook(_module, inputs):
        x = inputs[0]
        return (hadamard_module(x),)

    suffixes = SITE_SUFFIXES[mode]
    handles = []
    for name, module in model.named_modules():
        if any(name.endswith(s) for s in suffixes) and is_bitlinear(module):
            h = module.register_forward_pre_hook(_pre_hook)
            handles.append(h)
    return handles, len(handles), hadamard_module


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
    # Phase D final ckpts: {arch, train, state_dict, result}
    # Track B training-loop ckpts: {step, state_dict, optimizer, ...}
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> int:
    p = argparse.ArgumentParser(description="V32-prep F.6.2 post-hoc Hadamard eval.")
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                   help=f"Path to Phase D Mini final checkpoint. Default: {DEFAULT_CKPT}")
    p.add_argument("--n-val-batches", type=int, default=50,
                   help="Number of val batches for run_held_out_loss (default 50, matches Mini ablation).")
    p.add_argument("--device", default=None,
                   help="cpu|cuda. Default: cuda if available, else cpu.")
    p.add_argument("--out", type=Path, default=OUT_PATH,
                   help=f"Output JSON path. Default: {OUT_PATH}")
    p.add_argument("--dry-run", action="store_true",
                   help="Build + load + attach (each mode), but skip val_loss runs. "
                   "Verifies script wiring on CPU in ~30 sec without GPU eval.")
    args = p.parse_args()

    if not args.ckpt.exists():
        print(f"[error] checkpoint not found: {args.ckpt}", file=sys.stderr)
        return 2

    import torch
    from configs.mini import MiniArchConfig
    arch = MiniArchConfig()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/4] device={device}, ckpt={args.ckpt}, n_val_batches={args.n_val_batches}, dry_run={args.dry_run}")

    if not args.dry_run:
        from configs.mini import MiniTrainConfig
        from intllm.data import bilingual_stream
        from intllm.eval import run_held_out_loss
        from intllm.tokenizer import get_tokenizer
        sys.path.insert(0, str(REPO_ROOT / "python" / "phase_e"))
        from intllm_en import BILINGUAL_RATIO_DEFAULT

        train_hp = MiniTrainConfig()
        print(f"[2/4] tokenizer + val stream (bs={train_hp.batch_size}, seq={train_hp.seq_len}, id_share=0.6, seed=999)")
        tok = get_tokenizer()

    modes = ["no_rotation", "o", "igf", "igfo"]
    results: dict[str, dict] = {}

    for mode in modes:
        print(f"\n[3/4] mode={mode}")
        model = _build_and_load(args.ckpt, arch, device)
        handles, n_attached, hada_mod = _attach_hadamard_hooks(model, mode, arch.hidden_size, device)
        print(f"      attached: {n_attached} sites")

        if args.dry_run:
            results[mode] = {"val_loss": float("nan"), "ppl": float("nan"),
                             "n_attached": n_attached, "dry_run": True}
        else:
            val_stream = bilingual_stream(
                tokenizer=tok,
                seq_len=train_hp.seq_len,
                batch_size=train_hp.batch_size,
                id_share=BILINGUAL_RATIO_DEFAULT,
                device=device,
                seed=999,
            )
            val_loss = run_held_out_loss(
                model, batches=val_stream, n_steps=args.n_val_batches, device=device,
            )
            results[mode] = {
                "val_loss": float(val_loss),
                "ppl": float(math.exp(val_loss)),
                "n_attached": n_attached,
            }
            print(f"      val_loss = {val_loss:.4f}  (PPL {math.exp(val_loss):,.1f})")

        for h in handles:
            h.remove()
        del model
        if hada_mod is not None:
            del hada_mod
        if device == "cuda":
            torch.cuda.empty_cache()

    # Compute deltas + verdict (real run only).
    if not args.dry_run:
        baseline = results["no_rotation"]["val_loss"]
        for mode in modes:
            if mode == "no_rotation":
                continue
            results[mode]["delta_vs_baseline"] = float(results[mode]["val_loss"] - baseline)

        rotated = {m: results[m]["val_loss"] for m in modes if m != "no_rotation"}
        best_rotated_mode = min(rotated, key=rotated.get)
        best_delta = results[best_rotated_mode]["delta_vs_baseline"]
        rotation_helps = best_delta <= -F62_GATE_NAT
        rotation_hurts = all(results[m]["delta_vs_baseline"] >= F62_GATE_NAT for m in rotated)

        if rotation_helps:
            outcome = "helps"
            interpretation = (
                "Activation-only pre-hook rotation IMPROVED val_loss — unusual under "
                "an FP-path-breaking recipe. Likely indicates the trained checkpoint "
                "had latent compatibility with rotated inputs (e.g. via residual-stream "
                "self-correction). Worth investigating, but not a confirmation of "
                "canonical QuaRot benefit."
            )
        elif rotation_hurts:
            outcome = "hurts"
            interpretation = (
                "Activation-only pre-hook rotation REGRESSED val_loss — EXPECTED. "
                "Pre-hook breaks the FP path (y = W·Hx ≠ W·x), so any trained model "
                "should regress regardless of architecture. This is a RECIPE-CLASS "
                "outcome (the recipe is incomplete), NOT an architecture-class verdict. "
                "To disambiguate (a) training-from-scratch fights rotation vs "
                "(b) HGRN-rotation incompatibility, the full canonical QuaRot recipe "
                "(weight fusion + matched residual rotation + γ_x recalibration) must "
                "be implemented and run — see script docstring §SCOPE CAVEAT."
            )
        else:
            outcome = "neutral"
            interpretation = (
                "Activation-only pre-hook rotation was within ±0.05 nat of baseline — "
                "unusual; FP-path breakage typically causes >>0.05 nat regression. "
                "May indicate rotation amplitude was small enough that downstream "
                "layers absorbed it without losing much information."
            )

        verdict = {
            "test_recipe": "activation-only-pre-hook (NOT canonical QuaRot weight-fusion)",
            "canonical_quarot_weight_fusion_tested": False,
            "best_mode": "no_rotation" if not rotation_helps else best_rotated_mode,
            "best_rotated_mode": best_rotated_mode,
            "best_delta_vs_baseline": best_delta,
            "rotation_helps": rotation_helps,
            "rotation_hurts": rotation_hurts,
            "rotation_outcome": outcome,
            "interpretation": interpretation,
            "caveat": (
                "Activation-only pre-hook breaks the FP path: y = W·Hx ≠ W·x in "
                "general. Any trained model is expected to regress under this "
                "recipe. F.6.2 result CANNOT distinguish (a) training-from-scratch "
                "fights rotation vs (b) HGRN-architecture-rotation-incompatibility — "
                "both hypotheses require canonical QuaRot (weight fusion + matched "
                "residual rotation + γ_x recalibration), which was NOT executed here. "
                "Treat 'hurts' verdict as RECIPE-INCOMPLETENESS evidence only, not "
                "as an HGRN-specific architectural finding."
            ),
            "gate_nat_threshold": F62_GATE_NAT,
        }
    else:
        verdict = {"dry_run": True}

    payload = {
        "_schema_version": "1.0",
        "ckpt_path": str(args.ckpt),
        "n_val_batches": args.n_val_batches,
        "stream": "bilingual(id_share=0.6, seed=999)",
        "device": device,
        "modes": results,
        "verdict": verdict,
        "timestamp": _now_iso(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, args.out)

    print(f"\n[4/4] wrote {args.out}")
    if not args.dry_run:
        print(f"      rotation_outcome: {verdict['rotation_outcome']}")
        print(f"      best_mode={verdict['best_mode']}  best_delta={verdict['best_delta_vs_baseline']:+.4f} nat")
        print(f"      ⚠ test_recipe: {verdict['test_recipe']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
