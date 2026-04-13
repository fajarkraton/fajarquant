#!/usr/bin/env python3
"""
quant_attention_v3.py — FajarQuant v3: adaptive per-head method selection.

Per-head dispatch: each (layer, head, k/v) gets its own quantization path
based on the strategy assignment from strategy_selector.py.

Paths:
  A: KIVI-like per-channel symmetric (reuse quant_attention.py)
  B: PCA rotation + per-coord (reuse quant_attention_v2.py)
  C: Hadamard + adaptive outlier extraction (adapted from TQ outlier)
  D: Residual quantization (base + residual pass)
  E: Asymmetric per-channel (non-zero zero_point)

Usage:
    from quant_attention_v3 import (
        load_v3_config,
        patch_model_for_quantization_v3,
        unpatch_model,
    )

    config = load_v3_config("data/calibration/strategy_gemma_2bit.json",
                            "data/calibration/fq_v2_gemma_4_e2b.npz")
    patch_model_for_quantization_v3(model, 2, config)
    ppl = run_eval(model, ...)
"""

from __future__ import annotations

import json
import types
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

# Reuse v1/v2 individual path implementations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_attention import (  # noqa: E402
    apply_kivi_keys_4d,
    apply_kivi_values_4d,
    apply_turboquant_4d_with_outliers,
    _ORIGINAL_FORWARDS,
)
from quant_attention_v2 import load_calibration  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════
# Module-level v3 state
# ═══════════════════════════════════════════════════════════════════════

_V3_STRATEGY: dict | None = None     # strategy_selector output
_V3_BITS: int = 0
_V3_CALIBRATION: dict | None = None  # v2 calibration (for Path B)


# ═══════════════════════════════════════════════════════════════════════
# Path implementations
# ═══════════════════════════════════════════════════════════════════════

def _path_a_kivi(kv: torch.Tensor, bits: int, is_key: bool) -> torch.Tensor:
    """Path A: KIVI-like per-channel symmetric."""
    if is_key:
        return apply_kivi_keys_4d(kv, bits)
    return apply_kivi_values_4d(kv, bits)


def _path_b_pca(kv: torch.Tensor, bits: int, layer_idx: int, is_key: bool,
                cal: dict | None) -> torch.Tensor:
    """Path B: Calibrated PCA rotation + per-coord quantization.

    Uses v2 calibration data from load_calibration() which returns:
      {"layers": [{"k_outlier_mask": ..., "k_pca_rotation": ..., ...}, ...]}
    Lazy GPU tensor caching via _get_gpu_tensors (from quant_attention_v2).

    B-fix D6: fixed calibration key format — was using flat keys
    "layer_0_k_outlier_mask" that never existed in the nested cal dict.
    """
    if cal is None:
        import warnings
        warnings.warn(f"Path B: no calibration data, falling back to Path A (layer {layer_idx})")
        return _path_a_kivi(kv, bits, is_key)

    layers = cal.get("layers")
    if layers is None or layer_idx >= len(layers):
        import warnings
        warnings.warn(f"Path B: layer {layer_idx} not in calibration ({len(layers) if layers else 0} layers), "
                       "falling back to Path A")
        return _path_a_kivi(kv, bits, is_key)

    layer_cal = layers[layer_idx]
    kv_type = "k" if is_key else "v"

    outlier_mask_np = layer_cal.get(f"{kv_type}_outlier_mask")
    pca_rotation_np = layer_cal.get(f"{kv_type}_pca_rotation")
    pca_mean_np = layer_cal.get(f"{kv_type}_pca_mean")

    if outlier_mask_np is None or pca_rotation_np is None or pca_mean_np is None:
        import warnings
        warnings.warn(f"Path B: missing calibration arrays for layer {layer_idx} {kv_type}, "
                       "falling back to Path A")
        return _path_a_kivi(kv, bits, is_key)

    device = kv.device
    dtype = kv.dtype
    B, H, S, D = kv.shape

    # Lazy GPU tensor caching (same pattern as v2 _get_gpu_tensors)
    cache_key = f"_v3_{kv_type}_gpu_{device}"
    if cache_key not in layer_cal:
        layer_cal[cache_key] = (
            torch.from_numpy(outlier_mask_np).to(device),
            torch.from_numpy(pca_rotation_np).float().to(device),
            torch.from_numpy(pca_mean_np).float().to(device),
        )
    outlier_mask_t, rot_t, mean_t = layer_cal[cache_key]
    clean_mask = ~outlier_mask_t

    D_clean = int(clean_mask.sum())
    result = kv.clone()
    for h in range(H):
        head_data = kv[:, h, :, :]  # (B, S, D)
        # Preserve outlier channels in full precision
        outlier_data = head_data[:, :, outlier_mask_t].clone()
        # Clean channels: center → rotate → quantize → inverse rotate → uncenter
        clean = head_data[:, :, clean_mask].float()
        centered = clean - mean_t[:D_clean]
        rotated = centered @ rot_t[:D_clean, :D_clean].T
        # Per-coord asymmetric quantize (matching v2's unsigned uniform grid)
        levels = (1 << bits) - 1
        rmin = rotated.amin(dim=(0, 1), keepdim=True)
        rmax = rotated.amax(dim=(0, 1), keepdim=True)
        scale = (rmax - rmin).clamp(min=1e-8) / levels
        quantized = ((rotated - rmin) / scale).round().clamp(0, levels)
        dequantized = quantized * scale + rmin
        # Inverse rotate + uncenter
        restored = dequantized @ rot_t[:D_clean, :D_clean] + mean_t[:D_clean]
        result[:, h, :, clean_mask] = restored.to(dtype)
        result[:, h, :, outlier_mask_t] = outlier_data
    return result


def _path_c_hadamard_outlier(kv: torch.Tensor, bits: int) -> torch.Tensor:
    """Path C: Hadamard rotation + adaptive outlier extraction."""
    return apply_turboquant_4d_with_outliers(kv, bits)


def _path_d_residual(kv: torch.Tensor, bits: int) -> torch.Tensor:
    """Path D: Residual quantization (base + residual pass)."""
    B, H, S, D = kv.shape
    max_q = (1 << (bits - 1)) - 1
    # Base pass
    scale1 = kv.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / max_q
    q1 = (kv / scale1).round().clamp(-max_q, max_q)
    recon1 = q1 * scale1
    # Residual pass
    residual = kv - recon1
    scale2 = residual.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / max_q
    q2 = (residual / scale2).round().clamp(-max_q, max_q)
    recon2 = q2 * scale2
    return recon1 + recon2


def _path_e_asymmetric(kv: torch.Tensor, bits: int) -> torch.Tensor:
    """Path E: Asymmetric per-channel quantization."""
    B, H, S, D = kv.shape
    max_q = (1 << bits) - 1  # unsigned range
    result = kv.clone()
    for h in range(H):
        head = kv[:, h, :, :]  # (B, S, D)
        ch_min = head.amin(dim=(0, 1))  # (D,)
        ch_max = head.amax(dim=(0, 1))
        ch_range = (ch_max - ch_min).clamp(min=1e-8)
        scale = ch_range / max_q
        zp = ch_min
        q = ((head - zp) / scale).round().clamp(0, max_q)
        result[:, h, :, :] = q * scale + zp
    return result


_PATH_DISPATCH = {
    "A": lambda kv, bits, is_key, layer_idx, cal: _path_a_kivi(kv, bits, is_key),
    "B": lambda kv, bits, is_key, layer_idx, cal: _path_b_pca(kv, bits, layer_idx, is_key, cal),
    "C": lambda kv, bits, is_key, layer_idx, cal: _path_c_hadamard_outlier(kv, bits),
    "D": lambda kv, bits, is_key, layer_idx, cal: _path_d_residual(kv, bits),
    "E": lambda kv, bits, is_key, layer_idx, cal: _path_e_asymmetric(kv, bits),
}


# ═══════════════════════════════════════════════════════════════════════
# Per-head dispatch core
# ═══════════════════════════════════════════════════════════════════════

def _quantize_kv_v3(
    kv: torch.Tensor,
    bits: int,
    is_key: bool,
    layer_idx: int,
    strategy: dict,
    cal: dict | None,
) -> torch.Tensor:
    """Per-head dispatch: applies the assigned strategy to each head."""
    B, H, S, D = kv.shape
    assignments = strategy.get("assignments", [])
    if layer_idx >= len(assignments):
        return kv  # no strategy for this layer, pass through

    layer_strat = assignments[layer_idx]
    kv_type = "k" if is_key else "v"

    # Check if all heads use the same path → batch optimization
    paths = [head.get(kv_type, "A") for head in layer_strat[:H]]
    unique_paths = set(paths)

    if len(unique_paths) == 1:
        # All same path: apply to full tensor (no per-head loop)
        path = paths[0]
        return _PATH_DISPATCH[path](kv, bits, is_key, layer_idx, cal)

    # Mixed paths: per-head dispatch
    result = kv.clone()
    for h in range(min(H, len(layer_strat))):
        path = layer_strat[h].get(kv_type, "A")
        head_kv = kv[:, h:h+1, :, :]  # (B, 1, S, D)
        head_result = _PATH_DISPATCH[path](head_kv, bits, is_key, layer_idx, cal)
        result[:, h:h+1, :, :] = head_result
    return result


# ═══════════════════════════════════════════════════════════════════════
# Config + Patching API
# ═══════════════════════════════════════════════════════════════════════

def load_v3_config(strategy_path: str, calibration_path: str | None = None) -> dict:
    """Load v3 configuration: strategy assignments + optional v2 calibration."""
    with open(strategy_path) as f:
        strategy = json.load(f)
    cal = None
    if calibration_path and os.path.exists(calibration_path):
        cal = load_calibration(calibration_path)
    return {"strategy": strategy, "calibration": cal}


def patch_model_for_quantization_v3(model: nn.Module, bits: int, config: dict):
    """Patch model attention layers for v3 per-head quantization."""
    global _V3_STRATEGY, _V3_BITS, _V3_CALIBRATION
    _V3_STRATEGY = config["strategy"]
    _V3_BITS = bits
    _V3_CALIBRATION = config.get("calibration")

    # Detect architecture and patch — match by exact class name (like v1)
    _V3_FORWARD_MAKERS = {
        "MistralAttention": _make_v3_forward_mistral,
        "Qwen2Attention": _make_v3_forward_qwen2,
        "Gemma4TextAttention": _make_v3_forward_gemma4,
    }

    layer_idx = 0
    for module in model.modules():
        cls_name = type(module).__name__
        if cls_name not in _V3_FORWARD_MAKERS:
            continue
        mid = id(module)
        if mid not in _ORIGINAL_FORWARDS:
            _ORIGINAL_FORWARDS[mid] = module.forward
        maker = _V3_FORWARD_MAKERS[cls_name]
        module.forward = types.MethodType(maker(layer_idx), module)
        layer_idx += 1


def unpatch_model(model: nn.Module):
    """Restore original forwards."""
    for module in model.modules():
        mid = id(module)
        if mid in _ORIGINAL_FORWARDS:
            module.forward = _ORIGINAL_FORWARDS[mid]
    _ORIGINAL_FORWARDS.clear()


# ═══════════════════════════════════════════════════════════════════════
# Per-architecture patched forwards (same as v1/v2 but call _quantize_kv_v3)
# ═══════════════════════════════════════════════════════════════════════

def _make_v3_forward_mistral(layer_idx: int):
    """Create a closure for Mistral v3 forward at a specific layer."""
    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kwargs):
        from transformers.models.mistral.modeling_mistral import (
            ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb,
        )
        bsz, q_len, _ = hidden_states.size()
        qs = self.q_proj(hidden_states)
        ks = self.k_proj(hidden_states)
        vs = self.v_proj(hidden_states)
        qs = qs.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        ks = ks.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        vs = vs.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        qs, ks = apply_rotary_pos_emb(qs, ks, cos, sin)
        # v3: per-head quantization
        ks = _quantize_kv_v3(ks, _V3_BITS, True, layer_idx, _V3_STRATEGY, _V3_CALIBRATION)
        vs = _quantize_kv_v3(vs, _V3_BITS, False, layer_idx, _V3_STRATEGY, _V3_CALIBRATION)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": kwargs.get("cache_position")}
            ks, vs = past_key_values.update(ks, vs, layer_idx, cache_kwargs)
        attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        ao, aw = attn_fn(self, qs, ks, vs, attention_mask, **kwargs)
        ao = ao.reshape(bsz, q_len, -1).contiguous()
        ao = self.o_proj(ao)
        return ao, aw
    return forward


def _make_v3_forward_qwen2(layer_idx: int):
    """Create a closure for Qwen2 v3 forward."""
    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kwargs):
        from transformers.models.qwen2.modeling_qwen2 import (
            ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb,
        )
        bsz, q_len, _ = hidden_states.size()
        qs = self.q_proj(hidden_states)
        ks = self.k_proj(hidden_states)
        vs = self.v_proj(hidden_states)
        qs = qs.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        ks = ks.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        vs = vs.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        qs, ks = apply_rotary_pos_emb(qs, ks, cos, sin)
        ks = _quantize_kv_v3(ks, _V3_BITS, True, layer_idx, _V3_STRATEGY, _V3_CALIBRATION)
        vs = _quantize_kv_v3(vs, _V3_BITS, False, layer_idx, _V3_STRATEGY, _V3_CALIBRATION)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": kwargs.get("cache_position")}
            ks, vs = past_key_values.update(ks, vs, layer_idx, cache_kwargs)
        attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        ao, aw = attn_fn(self, qs, ks, vs, attention_mask, **kwargs)
        ao = ao.reshape(bsz, q_len, -1).contiguous()
        ao = self.o_proj(ao)
        return ao, aw
    return forward


def _make_v3_forward_gemma4(layer_idx: int):
    """Create a closure for Gemma4 v3 forward.

    Reuses the VERBATIM v1 gemma4_quantized_forward from quant_attention.py,
    only swapping `_quantize_kv(...)` calls to `_quantize_kv_v3(...)`.
    This avoids subtle Gemma4-specific bugs in re-implementation.
    """
    from quant_attention import gemma4_quantized_forward as _v1_gemma4_fwd

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kwargs):
        from transformers.models.gemma4.modeling_gemma4 import (
            ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb, eager_attention_forward,
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

            # v3: per-head quantization (ONLY difference from v1)
            key_states = _quantize_kv_v3(key_states, _V3_BITS, True, layer_idx,
                                         _V3_STRATEGY, _V3_CALIBRATION)
            value_states = _quantize_kv_v3(value_states, _V3_BITS, False, layer_idx,
                                           _V3_STRATEGY, _V3_CALIBRATION)

        if past_key_values is not None:
            if not self.is_kv_shared_layer:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx
                )
            if self.store_full_length_kv:
                if not hasattr(past_key_values, "shared_layers"):
                    past_key_values.shared_layers = {}
                past_key_values.shared_layers[self.layer_idx] = key_states, value_states

        attention_interface: callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    return forward
