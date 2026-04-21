"""Smoke test for `intllm.model` re-export shim.

Verifies the vendored upstream is importable via `intllm.model` without
users having to know about `_upstream/` or `mmfreelm`.
"""

from __future__ import annotations


def test_shim_reexports_hgrnbit_classes() -> None:
    from intllm.model import HGRNBitConfig, HGRNBitForCausalLM, HGRNBitModel

    cfg = HGRNBitConfig()
    assert cfg.hidden_size > 0, "HGRNBitConfig should have a valid hidden_size default"
    assert HGRNBitModel is not None
    assert HGRNBitForCausalLM is not None


def test_automodel_recognises_mmfreelm() -> None:
    """`import mmfreelm` side effect should have registered HGRNBit*
    with transformers AutoModel."""
    import intllm.model  # noqa: F401 — triggers the registration
    from transformers import MODEL_FOR_CAUSAL_LM_MAPPING

    registered = {cls.__name__ for cls in MODEL_FOR_CAUSAL_LM_MAPPING.values()}
    assert "HGRNBitForCausalLM" in registered, (
        f"HGRNBitForCausalLM not in transformers registry; saw {sorted(registered)[:5]}..."
    )
