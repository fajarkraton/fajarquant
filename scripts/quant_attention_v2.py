#!/usr/bin/env python3
"""
quant_attention_v2.py — FajarQuant v2 (F2-D): calibrated PCA + outlier extraction.

Loads pre-computed calibration data (.npz from calibrate_fq_v2.py), then applies:
  1. Outlier extraction: top-1% high-variance channels kept in fp16
  2. Calibrated PCA rotation on clean (non-outlier) channels
  3. Per-coordinate uniform quantization in the rotated space
  4. Inverse rotation + outlier merge

Reuses the R-α.1 monkey-patch infrastructure from quant_attention.py
(per-architecture forward subclasses). Only the quantization function changes.

Decision provenance: V26 C1.6 Path B, Phase B2.D → F2-D.
Algorithm spec: docs/V26_C1_6_V2_DESIGN.md

Usage:
    from quant_attention_v2 import (
        load_calibration,
        patch_model_for_quantization_v2,
        unpatch_model,
    )

    cal = load_calibration("data/calibration/fq_v2_gemma_4_e2b.npz")
    patch_model_for_quantization_v2(model, "fajarquant_v2", 2, cal)
    ppl = run_eval(model, ...)
"""

from __future__ import annotations

import os
import sys
import types
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

# Reuse v1 infrastructure
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_attention import (  # noqa: E402
    _ORIGINAL_FORWARDS,
    _quantize_per_coord_4d,
    apply_fajarquant_4d,
    apply_kivi_keys_4d,
    apply_kivi_values_4d,
    apply_turboquant_4d,
    apply_turboquant_4d_with_outliers,
)

# ═══════════════════════════════════════════════════════════════════════
# Module-level state
# ═══════════════════════════════════════════════════════════════════════

_QUANT_METHOD_V2: str = "none"
_QUANT_BITS_V2: int = 0
_CALIBRATION: dict | None = None  # loaded calibration data


# ═══════════════════════════════════════════════════════════════════════
# Calibration loader
# ═══════════════════════════════════════════════════════════════════════


def load_calibration(path: str) -> dict:
    """Load a .npz calibration file produced by calibrate_fq_v2.py.

    Supports both per-layer (v2 original) and per-head (v3.1) formats.
    Auto-detects format by checking for `k_0_h0_pca_rotation` key.

    Returns a dict with metadata and per-layer calibration arrays.
    Per-head format adds `cal["layers"][i]["heads"]` list.
    Per-layer format uses `cal["layers"][i]["k_pca_rotation"]` directly.
    """
    raw = dict(np.load(path, allow_pickle=True))

    n_layers = int(raw["_n_layers"])
    head_dim = int(raw["_head_dim"])
    n_heads = int(raw["_n_heads"]) if "_n_heads" in raw else 1
    is_per_head = "k_0_h0_pca_rotation" in raw

    cal: dict = {
        "model": str(raw["_model"]),
        "n_layers": n_layers,
        "head_dim": head_dim,
        "n_heads": n_heads,
        "per_head": is_per_head,
        "layers": [],
    }

    for i in range(n_layers):
        if is_per_head:
            # Per-head format: cal["layers"][i]["heads"][h]["k_pca_rotation"]
            layer_cal = {"heads": []}
            for h in range(n_heads):
                head_cal = {
                    "k_outlier_mask": raw[f"k_{i}_h{h}_outlier_mask"],
                    "k_pca_rotation": raw[f"k_{i}_h{h}_pca_rotation"],
                    "k_pca_mean": raw[f"k_{i}_h{h}_pca_mean"],
                    "v_outlier_mask": raw[f"v_{i}_h{h}_outlier_mask"],
                    "v_pca_rotation": raw[f"v_{i}_h{h}_pca_rotation"],
                    "v_pca_mean": raw[f"v_{i}_h{h}_pca_mean"],
                }
                layer_cal["heads"].append(head_cal)
            # Also keep layer-level access for backward compat (use head 0)
            layer_cal["k_outlier_mask"] = layer_cal["heads"][0]["k_outlier_mask"]
            layer_cal["k_pca_rotation"] = layer_cal["heads"][0]["k_pca_rotation"]
            layer_cal["k_pca_mean"] = layer_cal["heads"][0]["k_pca_mean"]
            layer_cal["v_outlier_mask"] = layer_cal["heads"][0]["v_outlier_mask"]
            layer_cal["v_pca_rotation"] = layer_cal["heads"][0]["v_pca_rotation"]
            layer_cal["v_pca_mean"] = layer_cal["heads"][0]["v_pca_mean"]
            cal["layers"].append(layer_cal)
        else:
            # Per-layer format (original v2)
            layer_cal = {
                "k_outlier_mask": raw[f"k_{i}_outlier_mask"],
                "k_pca_rotation": raw[f"k_{i}_pca_rotation"],
                "k_pca_mean": raw[f"k_{i}_pca_mean"],
                "v_outlier_mask": raw[f"v_{i}_outlier_mask"],
                "v_pca_rotation": raw[f"v_{i}_pca_rotation"],
                "v_pca_mean": raw[f"v_{i}_pca_mean"],
            }
            cal["layers"].append(layer_cal)

    return cal


def _get_gpu_tensors(
    layer_cal: dict, kv_type: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lazily convert numpy arrays to GPU tensors and cache them."""
    cache_key = f"_{kv_type}_gpu_{device}"
    if cache_key not in layer_cal:
        prefix = kv_type  # "k" or "v"
        outlier_mask = torch.from_numpy(
            layer_cal[f"{prefix}_outlier_mask"]
        ).to(device)
        R = torch.from_numpy(layer_cal[f"{prefix}_pca_rotation"]).to(device)
        mean = torch.from_numpy(layer_cal[f"{prefix}_pca_mean"]).to(device)
        clean_mask = ~outlier_mask
        layer_cal[cache_key] = (outlier_mask, clean_mask, R, mean)
    return layer_cal[cache_key]


# ═══════════════════════════════════════════════════════════════════════
# F2-D quantization core
# ═══════════════════════════════════════════════════════════════════════


def apply_fajarquant_v2_4d(
    kv: torch.Tensor,
    bits: int,
    layer_idx: int,
    is_key: bool,
) -> torch.Tensor:
    """FajarQuant v2 (F2-D) on a (B, H, S, D) tensor.

    Uses calibrated PCA rotation + outlier extraction:
      1. Split into outlier channels (fp16) and clean channels
      2. Center clean channels using calibrated mean
      3. Rotate using calibrated PCA basis
      4. Per-coordinate uniform quantize
      5. Inverse rotate + re-add mean
      6. Merge outlier channels back at original positions

    Requires _CALIBRATION to be loaded via load_calibration().
    """
    if _CALIBRATION is None:
        raise RuntimeError(
            "FajarQuant v2 requires calibration data. "
            "Call load_calibration() first."
        )

    B, H, S, D = kv.shape
    if S < 2:
        return kv

    layer_cal = _CALIBRATION["layers"][layer_idx]
    kv_type = "k" if is_key else "v"
    outlier_mask, clean_mask, R, mean = _get_gpu_tensors(
        layer_cal, kv_type, kv.device
    )

    work = kv.float()

    # Step 1: Split
    # outlier_mask shape: (D,) bool
    # kv_outlier: (B, H, S, n_outlier) — kept in fp16
    kv_outlier = work[:, :, :, outlier_mask]
    # kv_clean: (B, H, S, D')
    kv_clean = work[:, :, :, clean_mask]

    # Step 2: Center using calibrated mean
    # mean shape: (D',) — broadcast to (1, 1, 1, D')
    kv_centered = kv_clean - mean.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    # Step 3: Rotate using calibrated PCA basis
    # R shape: (D', D') — R has eigenvectors as rows
    # rotated = kv_centered @ R.T
    kv_rotated = torch.matmul(kv_centered, R.T)

    # Step 4: Per-coordinate uniform quantize
    kv_quant = _quantize_per_coord_4d(kv_rotated, bits)

    # Step 5: Inverse rotate + re-add mean
    # recon = kv_quant @ R + mean
    kv_recon = torch.matmul(kv_quant, R) + mean.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    # Step 6: Merge back
    # Create output tensor, scatter clean and outlier channels
    result = torch.empty_like(work)
    result[:, :, :, outlier_mask] = kv_outlier
    result[:, :, :, clean_mask] = kv_recon

    return result.to(kv.dtype)


def apply_fajarquant_v2a_4d(
    kv: torch.Tensor,
    bits: int,
    layer_idx: int,
    is_key: bool,
) -> torch.Tensor:
    """FajarQuant v2-A (ablation): calibrated PCA only, NO outlier extraction.

    Same as v2 but uses ALL channels for PCA (no split). This isolates
    the contribution of calibrated-vs-per-chunk PCA (RC2 fix) from the
    outlier extraction (RC5 fix).
    """
    if _CALIBRATION is None:
        raise RuntimeError("v2-A requires calibration data.")

    B, H, S, D = kv.shape
    if S < 2:
        return kv

    layer_cal = _CALIBRATION["layers"][layer_idx]
    kv_type = "k" if is_key else "v"
    _, clean_mask, R, mean = _get_gpu_tensors(layer_cal, kv_type, kv.device)

    work = kv.float()

    # Use only clean channels (same PCA basis) but quantize ALL of them —
    # no outlier preservation. This means outlier channels are quantized
    # too, through their projection onto the clean-data PCA basis.
    # We need a full-dim rotation for this. Instead, use the clean PCA
    # on clean channels and quantize outlier channels with simple uniform.
    kv_clean = work[:, :, :, clean_mask]
    kv_outlier = work[:, :, :, ~clean_mask]

    # Calibrated PCA on clean channels
    kv_centered = kv_clean - mean.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    kv_rotated = torch.matmul(kv_centered, R.T)
    kv_quant = _quantize_per_coord_4d(kv_rotated, bits)
    kv_recon = torch.matmul(kv_quant, R) + mean.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    # Outlier channels: quantize with simple per-coord uniform (NOT fp16)
    kv_outlier_quant = _quantize_per_coord_4d(kv_outlier, bits)

    # Merge
    result = torch.empty_like(work)
    result[:, :, :, clean_mask] = kv_recon
    result[:, :, :, ~clean_mask] = kv_outlier_quant

    return result.to(kv.dtype)


# ═══════════════════════════════════════════════════════════════════════
# Dispatch — routes method name to quantization function
# ═══════════════════════════════════════════════════════════════════════


def _quantize_kv_v2(
    kv: torch.Tensor, method: str, bits: int, is_key: bool, layer_idx: int
) -> torch.Tensor:
    """Dispatch to quantization method (v2-aware, supports layer_idx)."""
    if method == "none" or bits <= 0:
        return kv
    if method == "fajarquant_v2":
        return apply_fajarquant_v2_4d(kv, bits, layer_idx, is_key)
    if method == "fajarquant_v2a":
        return apply_fajarquant_v2a_4d(kv, bits, layer_idx, is_key)
    # Fall through to v1 methods for comparison
    if method == "fajarquant":
        return apply_fajarquant_4d(kv, bits)
    if method == "kivi":
        return apply_kivi_keys_4d(kv, bits) if is_key else apply_kivi_values_4d(kv, bits)
    if method == "turboquant":
        return apply_turboquant_4d(kv, bits)
    if method == "turboquant_outlier":
        return apply_turboquant_4d_with_outliers(kv, bits)
    raise ValueError(f"unknown quantization method: {method}")


# ═══════════════════════════════════════════════════════════════════════
# Per-architecture quantized forwards (v2 — adds layer_idx to dispatch)
# ═══════════════════════════════════════════════════════════════════════


def mistral_quantized_forward_v2(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Patched MistralAttention.forward (v2 — passes layer_idx)."""
    from transformers.models.mistral.modeling_mistral import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if _QUANT_METHOD_V2 != "none":
        key_states = _quantize_kv_v2(
            key_states, _QUANT_METHOD_V2, _QUANT_BITS_V2, is_key=True, layer_idx=self.layer_idx
        )
        value_states = _quantize_kv_v2(
            value_states, _QUANT_METHOD_V2, _QUANT_BITS_V2, is_key=False, layer_idx=self.layer_idx
        )

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx
        )

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=getattr(self.config, "sliding_window", None),
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def qwen2_quantized_forward_v2(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Patched Qwen2Attention.forward (v2 — passes layer_idx)."""
    from transformers.models.qwen2.modeling_qwen2 import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if _QUANT_METHOD_V2 != "none":
        key_states = _quantize_kv_v2(
            key_states, _QUANT_METHOD_V2, _QUANT_BITS_V2, is_key=True, layer_idx=self.layer_idx
        )
        value_states = _quantize_kv_v2(
            value_states, _QUANT_METHOD_V2, _QUANT_BITS_V2, is_key=False, layer_idx=self.layer_idx
        )

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx
        )

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def gemma4_quantized_forward_v2(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Patched Gemma4TextAttention.forward (v2 — passes layer_idx)."""
    from transformers.models.gemma4.modeling_gemma4 import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    cos, sin = position_embeddings

    query_states = self.q_proj(hidden_states).view(hidden_shape)
    query_states = self.q_norm(query_states)
    query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
    query_states = query_states.transpose(1, 2)

    if self.is_kv_shared_layer and past_key_values is not None:
        key_states, value_states = past_key_values.shared_layers[self.kv_shared_layer_index]
        key_states = key_states.to(query_states.device)
        value_states = value_states.to(query_states.device)
    else:
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = (
            self.v_proj(hidden_states).view(hidden_shape)
            if self.v_proj is not None
            else key_states
        )
        key_states = self.k_norm(key_states)
        key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
        key_states = key_states.transpose(1, 2)
        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

        if _QUANT_METHOD_V2 != "none":
            key_states = _quantize_kv_v2(
                key_states, _QUANT_METHOD_V2, _QUANT_BITS_V2,
                is_key=True, layer_idx=self.layer_idx,
            )
            value_states = _quantize_kv_v2(
                value_states, _QUANT_METHOD_V2, _QUANT_BITS_V2,
                is_key=False, layer_idx=self.layer_idx,
            )

    if past_key_values is not None:
        if not self.is_kv_shared_layer:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )
        if self.store_full_length_kv:
            if not hasattr(past_key_values, "shared_layers"):
                past_key_values.shared_layers = {}
            past_key_values.shared_layers[self.layer_idx] = key_states, value_states

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states, attention_mask,
        dropout=self.attention_dropout if self.training else 0.0,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


# ═══════════════════════════════════════════════════════════════════════
# Patch/unpatch API
# ═══════════════════════════════════════════════════════════════════════

_V2_FORWARDS_BY_CLASS: dict[str, Callable] = {
    "MistralAttention": mistral_quantized_forward_v2,
    "Qwen2Attention": qwen2_quantized_forward_v2,
    "Gemma4TextAttention": gemma4_quantized_forward_v2,
}


def patch_model_for_quantization_v2(
    model: nn.Module, method: str, bits: int, calibration: dict | None = None
) -> int:
    """Patch attention modules with v2 forwards.

    For v2 methods (fajarquant_v2, fajarquant_v2a), calibration is required.
    For v1 methods (fajarquant, kivi, turboquant, turboquant_outlier), calibration
    is not needed — the v2 forwards fall through to v1 functions.
    """
    global _QUANT_METHOD_V2, _QUANT_BITS_V2, _CALIBRATION

    valid_methods = (
        "none", "fajarquant_v2", "fajarquant_v2a",
        "fajarquant", "kivi", "turboquant", "turboquant_outlier",
    )
    if method not in valid_methods:
        raise ValueError(f"unknown method: {method}. Valid: {valid_methods}")
    if method != "none" and bits not in (2, 3, 4):
        raise ValueError(f"bits must be 2, 3, or 4 (got {bits})")
    if method in ("fajarquant_v2", "fajarquant_v2a") and calibration is None:
        raise ValueError(f"{method} requires calibration data")

    _QUANT_METHOD_V2 = method
    _QUANT_BITS_V2 = bits
    if calibration is not None:
        _CALIBRATION = calibration

    patched = 0
    for module in model.modules():
        cls_name = type(module).__name__
        if cls_name not in _V2_FORWARDS_BY_CLASS:
            continue
        mid = id(module)
        if mid not in _ORIGINAL_FORWARDS:
            _ORIGINAL_FORWARDS[mid] = module.forward
        module.forward = types.MethodType(_V2_FORWARDS_BY_CLASS[cls_name], module)
        patched += 1

    return patched


def unpatch_model(model: nn.Module) -> int:
    """Restore original forwards. Reuses v1's _ORIGINAL_FORWARDS store."""
    global _QUANT_METHOD_V2, _QUANT_BITS_V2
    _QUANT_METHOD_V2 = "none"
    _QUANT_BITS_V2 = 0

    restored = 0
    for module in model.modules():
        mid = id(module)
        if mid in _ORIGINAL_FORWARDS:
            module.forward = _ORIGINAL_FORWARDS[mid]
            restored += 1
    return restored


# ═══════════════════════════════════════════════════════════════════════
# Self-test (no GPU)
# ═══════════════════════════════════════════════════════════════════════


def _self_test() -> None:
    """Shape + sanity check for v2 quantization functions."""
    print("[quant_attention_v2] self-test starting...")

    D = 128
    B, H, S = 1, 4, 64
    k = torch.randn(B, H, S, D, dtype=torch.float32)

    # Create synthetic calibration data
    n_layers = 2
    cal: dict = {"model": "test", "n_layers": n_layers, "head_dim": D, "layers": []}

    for _ in range(n_layers):
        # Simulate: top 1% = 1-2 channels are outliers
        outlier_mask = np.zeros(D, dtype=np.bool_)
        outlier_mask[0] = True  # channel 0 is outlier
        clean_dim = D - outlier_mask.sum()

        # Random PCA basis for clean channels
        R = np.eye(clean_dim, dtype=np.float32)
        mean = np.zeros(clean_dim, dtype=np.float32)

        layer_cal = {
            "k_outlier_mask": outlier_mask,
            "k_pca_rotation": R,
            "k_pca_mean": mean,
            "v_outlier_mask": outlier_mask,
            "v_pca_rotation": R,
            "v_pca_mean": mean,
        }
        cal["layers"].append(layer_cal)

    global _CALIBRATION
    _CALIBRATION = cal

    # Test v2 quantization
    result = apply_fajarquant_v2_4d(k, 2, layer_idx=0, is_key=True)
    assert result.shape == k.shape, f"v2 shape mismatch: {result.shape}"
    assert torch.isfinite(result).all(), "v2 produced NaN/Inf"

    # Outlier channel should be preserved exactly (fp16 passthrough)
    outlier_orig = k[:, :, :, 0]
    outlier_recon = result[:, :, :, 0]
    assert torch.allclose(outlier_orig, outlier_recon, atol=1e-6), (
        "outlier channel not preserved"
    )

    # v2-A ablation
    result_a = apply_fajarquant_v2a_4d(k, 2, layer_idx=0, is_key=True)
    assert result_a.shape == k.shape, f"v2-A shape mismatch"
    assert torch.isfinite(result_a).all(), "v2-A produced NaN/Inf"

    # v2 should have lower MSE than v2-A (outlier preservation helps)
    mse_v2 = (result - k).pow(2).mean().item()
    mse_v2a = (result_a - k).pow(2).mean().item()
    print(f"  v2 MSE:  {mse_v2:.6f}")
    print(f"  v2-A MSE: {mse_v2a:.6f}")
    # With identity rotation and 1 outlier channel, v2 should be ≤ v2-A
    assert mse_v2 <= mse_v2a + 1e-6, (
        f"v2 ({mse_v2:.6f}) worse than v2-A ({mse_v2a:.6f})"
    )

    # Dispatch test
    result_dispatch = _quantize_kv_v2(k, "fajarquant_v2", 2, is_key=True, layer_idx=0)
    assert torch.allclose(result, result_dispatch, atol=1e-5), "dispatch mismatch"

    # v1 fallthrough
    from quant_attention import apply_fajarquant_4d as fq_v1
    result_v1_via_v2 = _quantize_kv_v2(k, "fajarquant", 2, is_key=True, layer_idx=0)
    result_v1_direct = fq_v1(k, 2)
    assert torch.allclose(result_v1_via_v2, result_v1_direct, atol=1e-5), "v1 fallthrough mismatch"

    _CALIBRATION = None
    print("[quant_attention_v2] self-test PASS")


if __name__ == "__main__":
    _self_test()
