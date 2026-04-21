"""IntLLM model classes — thin shim over vendored upstream.

For Phase D C.P2.1 the architecture is exactly the upstream MatMul-Free LM
(HGRNBit family from `ridgerchu/matmulfreellm @ f24cfe5`). The shim
re-exports the upstream classes under the `intllm.model` namespace so
Phase D code never depends on the upstream module layout directly — if
we later fork the architecture (QAT-specific layer variants, or the
Candidate A RWKV-7 ablation triggered in FJQ_PHASE_D_ARCH.md §3), we
change this shim rather than every import site.

Later C.P2.2 work will add IntLLM-specific subclasses here:
  - IntBitLinear — BitLinear with per-coord adaptive bit allocation
  - IntRMSNorm   — γ-less RMSNorm with explicit fixed-point scale
  - QATHook     — periodic γ_x re-calibration (§6.9 R5)
"""

from __future__ import annotations

import sys
from pathlib import Path

_UPSTREAM = Path(__file__).resolve().parent.parent / "_upstream"
if not _UPSTREAM.is_dir():
    raise ImportError(
        f"Vendored upstream missing at {_UPSTREAM}. "
        "See python/phase_d/UPSTREAM_PIN.md to recreate."
    )
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))

# Re-exports. `import mmfreelm` runs the side-effect that registers
# HGRNBitForCausalLM / HGRNBitModel with the HuggingFace Transformers
# AutoClass registry, which is what lets `AutoModelForCausalLM.from_pretrained`
# recognise `ridger/MMfreeLM-*` checkpoints.
import mmfreelm  # noqa: F401
from mmfreelm.models import HGRNBitConfig, HGRNBitForCausalLM, HGRNBitModel  # noqa: E402

__all__ = [
    "HGRNBitConfig",
    "HGRNBitForCausalLM",
    "HGRNBitModel",
]
