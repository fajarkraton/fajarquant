#!/usr/bin/env python3
"""V32-prep F.6.3 measurement — i/f/g_proj input dim audit.

Phase E E2.1 closure flagged a "non-canonical" warning when applying
Hadamard rotation to anything other than `attn.o_proj`: the i/f/g_proj
sites in HGRNBitAttention may take inputs of a different dimensionality
than o_proj (which sees `input_dim = hidden_size * expand_ratio`),
making a single shared HadamardRotation(hidden_size) module unsuitable.
Per docs/FJQ_PHASE_F_TAX_VERTICAL_ROADMAP.md §4.1 F.6.3 the implementation
is "~1 day implementation + 1 ablation" but only AFTER confirming the
attach point is dimensionally sound.

This audit walks each Phase D config (Mini, Base, Medium) and records
per-layer (i_proj.in, f_proj.in, g_proj.in, o_proj.in) to verify:

  R1: i_proj.in == f_proj.in == g_proj.in within every layer
  R2: i/f/g_proj.in == hidden_size (i.e. expand_ratio == 1 holds)
  R3: o_proj.in == hidden_size * expand_ratio (= hidden_size at ratio=1)

If R1+R2+R3 all hold for a given config, F.6.3 can attach a SHARED
HadamardRotation(hidden_size) module to i/f/g/o_proj sites in one pass
(matching the pattern E2.1.2 used for o_proj only). If any rule fails,
F.6.3 needs per-site rotation modules.

CPU-only, no checkpoint required (dims are structural). Runs in <5 sec.

Usage:

    cd python/phase_d
    PYTHONPATH=. ../../.venv/bin/python scripts/measure_igf_proj_dim_audit.py

Output: paper/intllm/ablations/igf_proj_dim_audit.json
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPO_ROOT = ROOT.parent.parent
OUT_PATH = REPO_ROOT / "paper" / "intllm" / "ablations" / "igf_proj_dim_audit.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _audit_config(scale: str):
    """Construct the model for `scale` ∈ {mini, base, medium} and walk
    every HGRNBitAttention block to record projection input dims.
    """
    if scale == "mini":
        from configs.mini import MiniArchConfig as ArchConfig
    elif scale == "base":
        from configs.base import BaseArchConfig as ArchConfig
    elif scale == "medium":
        from configs.medium import MediumArchConfig as ArchConfig
    else:
        raise ValueError(f"unknown scale {scale!r}")
    from intllm.model import HGRNBitConfig, HGRNBitForCausalLM

    arch = ArchConfig()
    cfg = HGRNBitConfig(
        vocab_size=arch.vocab_size,
        hidden_size=arch.hidden_size,
        num_hidden_layers=arch.num_hidden_layers,
        max_position_embeddings=arch.max_position_embeddings,
    )
    model = HGRNBitForCausalLM(cfg)

    expected_input_dim = int(arch.hidden_size * getattr(cfg, "expand_ratio", 1))

    per_layer = []
    for name, module in model.named_modules():
        if not name.endswith(".attn"):
            continue
        if not (
            hasattr(module, "i_proj")
            and hasattr(module, "f_proj")
            and hasattr(module, "g_proj")
            and hasattr(module, "o_proj")
        ):
            continue
        per_layer.append(
            {
                "layer": name,
                "i_proj_in": int(module.i_proj.in_features),
                "f_proj_in": int(module.f_proj.in_features),
                "g_proj_in": int(module.g_proj.in_features),
                "o_proj_in": int(module.o_proj.in_features),
            },
        )

    rule_r1 = all(
        layer["i_proj_in"] == layer["f_proj_in"] == layer["g_proj_in"]
        for layer in per_layer
    )
    rule_r2 = all(layer["i_proj_in"] == arch.hidden_size for layer in per_layer)
    rule_r3 = all(layer["o_proj_in"] == expected_input_dim for layer in per_layer)

    return {
        "scale": scale,
        "hidden_size": int(arch.hidden_size),
        "num_hidden_layers": int(arch.num_hidden_layers),
        "expand_ratio": int(getattr(cfg, "expand_ratio", 1)),
        "expected_input_dim": expected_input_dim,
        "n_layers_audited": len(per_layer),
        "per_layer": per_layer,
        "R1_igf_inputs_match": rule_r1,
        "R2_igf_inputs_eq_hidden_size": rule_r2,
        "R3_o_input_eq_input_dim": rule_r3,
        "f63_shared_hadamard_safe": rule_r1 and rule_r2 and rule_r3,
    }


def main() -> int:
    results = {scale: _audit_config(scale) for scale in ("mini", "base", "medium")}
    payload = {
        "_schema_version": "1.0",
        "purpose": "V32-prep F.6.3 hidden_size mismatch audit (i/f/g/o_proj)",
        "audited_configs": list(results.keys()),
        "results": results,
        "all_configs_safe": all(r["f63_shared_hadamard_safe"] for r in results.values()),
        "timestamp": _now_iso(),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, OUT_PATH)

    print(f"[audit] wrote {OUT_PATH}")
    for scale, r in results.items():
        verdict = "SAFE" if r["f63_shared_hadamard_safe"] else "UNSAFE"
        print(
            f"  {scale:7s}  d={r['hidden_size']:4d} L={r['num_hidden_layers']:2d} "
            f"expand_ratio={r['expand_ratio']}  →  {verdict}  "
            f"(R1={r['R1_igf_inputs_match']}, R2={r['R2_igf_inputs_eq_hidden_size']}, "
            f"R3={r['R3_o_input_eq_input_dim']})",
        )
    return 0 if payload["all_configs_safe"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
