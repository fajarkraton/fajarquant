#!/usr/bin/env python3
"""
quant_attention.py — R-α.1 literal model surgery for KV cache quantization.

Implements per-architecture attention forward subclasses that apply
quantization to key/value tensors immediately before they're written to
the model's KV cache via past_key_values.update(). This matches the
canonical KIVI/KVQuant/SKVQ protocol where quantization happens at the
attention layer level during forward pass, not as a post-hoc cache
mutation.

Decision provenance: V26 Phase C1.6, option R-α.1 (literal subclass).
Supersedes the prefix+target post-hoc cache mutation patch in
scripts/eval_perplexity.py commit c9b2ff5, which produced the
quant < FP16 anomaly documented in the literature investigation
(commit task #9 / docs/V26_C1_6_METHODOLOGY.md amendment).

Architectures supported (transformers 5.5.0):
  - MistralAttention      (mistral/modeling_mistral.py:122)
  - Qwen2Attention        (qwen2/modeling_qwen2.py:187)
  - Gemma4TextAttention   (gemma4/modeling_gemma4.py:1126)

Usage:
    from quant_attention import patch_model_for_quantization, unpatch_model
    model = AutoModelForCausalLM.from_pretrained(...).eval()

    # FP16 baseline (no patch needed; ensure clean state)
    unpatch_model(model)
    ppl_fp16 = run_eval(model, ...)

    # FajarQuant 2-bit
    patch_model_for_quantization(model, "fajarquant", 2)
    ppl_fq2 = run_eval(model, ...)

    # KIVI 2-bit
    patch_model_for_quantization(model, "kivi", 2)
    ppl_kivi2 = run_eval(model, ...)

    # Cleanup before next model load
    unpatch_model(model)
"""

from __future__ import annotations

import types
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

# ═══════════════════════════════════════════════════════════════════════
# Module-level state — set by patch_model_for_quantization, read by
# the patched forward methods at attention time.
# ═══════════════════════════════════════════════════════════════════════

_QUANT_METHOD: str = "none"   # "none" / "fajarquant" / "kivi" / "turboquant"
_QUANT_BITS: int = 0
_TURBO_R_CACHE: dict[tuple[int, str], torch.Tensor] = {}  # (head_dim, device_str) → R
_ORIGINAL_FORWARDS: dict[int, Callable] = {}  # id(module) → original forward


def get_quant_state() -> tuple[str, int]:
    """Read current quantization config (debug helper)."""
    return _QUANT_METHOD, _QUANT_BITS


# ═══════════════════════════════════════════════════════════════════════
# Vectorized quantization primitives — operate on (B, H, S, D) tensors
# directly without per-(b,h) Python loops. Slow paths fall back to a
# float32 promotion path; result is cast back to the input dtype.
# ═══════════════════════════════════════════════════════════════════════


def _quantize_per_coord_4d(data: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-coordinate min/max uniform quantization across the sequence
    dimension. Operates on (B, H, S, D) → (B, H, S, D).

    The quantization grid is computed per (B, H, D) — i.e. each
    coordinate of each head's sequence is independently mapped to
    `2**bits - 1 + 1` levels.
    """
    levels = (1 << bits) - 1
    mn = data.amin(dim=2, keepdim=True)  # (B, H, 1, D)
    mx = data.amax(dim=2, keepdim=True)
    rng = (mx - mn).clamp(min=1e-15)
    scale = rng / levels
    indices = ((data - mn) / scale).round().clamp(0, levels)
    return mn + indices * scale


def apply_fajarquant_4d(kv: torch.Tensor, bits: int) -> torch.Tensor:
    """FajarQuant on a (B, H, S, D) K or V tensor.

    Per (B, H), the function:
      1. Centers the sequence (subtract per-(B,H,D) mean over S).
      2. Computes the (D, D) covariance matrix from the centered data.
      3. Eigendecomposes; sorts eigenvectors by descending eigenvalue.
      4. Rotates to the principal-axis basis.
      5. Per-coordinate uniform quantizes (since axes are decorrelated).
      6. Inverse-rotates and re-adds the mean.

    Returns a tensor of the same shape and dtype as the input.
    """
    B, H, S, D = kv.shape
    if S < 2:
        return kv

    work = kv.float()
    mean = work.mean(dim=2, keepdim=True)  # (B, H, 1, D)
    centered = work - mean

    # Covariance per (b, h): (B, H, D, D)
    cov = torch.einsum("bhsi,bhsj->bhij", centered, centered) / max(S - 1, 1)

    # Symmetric eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(cov)  # eigvals (B,H,D), eigvecs (B,H,D,D)

    # Sort eigenvectors by descending eigenvalue
    idx = torch.argsort(eigvals, dim=-1, descending=True)  # (B, H, D)
    # Reorder eigvecs columns by idx
    idx_expanded = idx.unsqueeze(-2).expand(-1, -1, D, -1)  # (B, H, D, D)
    eigvecs_sorted = torch.gather(eigvecs, dim=-1, index=idx_expanded)
    # R has eigenvectors as rows: (B, H, D, D)
    R = eigvecs_sorted.transpose(-1, -2)

    # Forward rotation: rotated[b,h,s,e] = sum_d centered[b,h,s,d] * R[b,h,e,d]
    rotated = torch.einsum("bhsd,bhed->bhse", centered, R)

    quantized = _quantize_per_coord_4d(rotated, bits)

    # Inverse rotation: recon[b,h,s,d] = sum_e quantized[b,h,s,e] * R[b,h,e,d]
    recon = torch.einsum("bhse,bhed->bhsd", quantized, R) + mean

    return recon.to(kv.dtype)


def apply_kivi_keys_4d(k: torch.Tensor, bits: int) -> torch.Tensor:
    """KIVI per-channel quantization for keys.

    Min/max are computed per (B, H, D) across the S dimension — i.e.
    each channel of each head gets its own quantization grid based on
    that channel's range over the sequence. Equivalent to
    eval_perplexity.apply_kivi(..., is_key=True).
    """
    levels = (1 << bits) - 1
    work = k.float()
    mn = work.amin(dim=2, keepdim=True)  # (B, H, 1, D)
    mx = work.amax(dim=2, keepdim=True)
    rng = (mx - mn).clamp(min=1e-15)
    scale = rng / levels
    indices = ((work - mn) / scale).round().clamp(0, levels)
    return (mn + indices * scale).to(k.dtype)


def apply_kivi_values_4d(v: torch.Tensor, bits: int) -> torch.Tensor:
    """KIVI per-token quantization for values.

    Min/max are computed per (B, H, S) across the D dimension — i.e.
    each token's value vector gets its own quantization grid. Equivalent
    to eval_perplexity.apply_kivi(..., is_key=False).
    """
    levels = (1 << bits) - 1
    work = v.float()
    mn = work.amin(dim=3, keepdim=True)  # (B, H, S, 1)
    mx = work.amax(dim=3, keepdim=True)
    rng = (mx - mn).clamp(min=1e-15)
    scale = rng / levels
    indices = ((work - mn) / scale).round().clamp(0, levels)
    return (mn + indices * scale).to(v.dtype)


def apply_turboquant_4d(data: torch.Tensor, bits: int, seed: int = 42) -> torch.Tensor:
    """TurboQuant: random orthogonal rotation + per-coord quantization.

    A single fixed random orthogonal matrix R of shape (D, D) is
    generated per head dimension and cached. Equivalent to
    eval_perplexity.apply_turboquant.
    """
    B, H, S, D = data.shape
    cache_key = (D, str(data.device))
    R = _TURBO_R_CACHE.get(cache_key)
    if R is None:
        from scipy.stats import ortho_group

        rng = np.random.default_rng(seed)
        R_np = ortho_group.rvs(D, random_state=rng)
        R = torch.from_numpy(R_np).float().to(data.device)
        _TURBO_R_CACHE[cache_key] = R

    work = data.float()
    # Forward rotation: rotated[b,h,s,e] = sum_d work[b,h,s,d] * R[e,d]
    # = work @ R.T
    rotated = torch.einsum("bhsd,ed->bhse", work, R)

    quantized = _quantize_per_coord_4d(rotated, bits)

    # Inverse: recon = quantized @ R
    recon = torch.einsum("bhse,ed->bhsd", quantized, R)
    return recon.to(data.dtype)


def _quantize_kv(
    kv: torch.Tensor, method: str, bits: int, is_key: bool
) -> torch.Tensor:
    """Dispatch to the selected quantization method.

    Returns the input unchanged if method == "none".
    """
    if method == "none" or bits <= 0:
        return kv
    if method == "fajarquant":
        return apply_fajarquant_4d(kv, bits)
    if method == "kivi":
        return apply_kivi_keys_4d(kv, bits) if is_key else apply_kivi_values_4d(kv, bits)
    if method == "turboquant":
        return apply_turboquant_4d(kv, bits)
    raise ValueError(f"unknown quantization method: {method}")


# ═══════════════════════════════════════════════════════════════════════
# Per-architecture quantized forward methods — literal copies of the
# transformers 5.5.0 attention forward, with quantization inserted
# between RoPE application and the past_key_values.update() call.
# ═══════════════════════════════════════════════════════════════════════


def mistral_quantized_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Patched MistralAttention.forward.

    Verbatim copy of transformers 5.5.0 mistral/modeling_mistral.py:139
    with quantization inserted between RoPE and cache update.
    """
    from transformers.models.mistral.modeling_mistral import (  # type: ignore
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

    # ★★★ INSERTED QUANTIZATION (R-α.1) ★★★
    if _QUANT_METHOD != "none":
        key_states = _quantize_kv(key_states, _QUANT_METHOD, _QUANT_BITS, is_key=True)
        value_states = _quantize_kv(value_states, _QUANT_METHOD, _QUANT_BITS, is_key=False)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx
        )

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=getattr(self.config, "sliding_window", None),
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def qwen2_quantized_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Patched Qwen2Attention.forward.

    Verbatim copy of transformers 5.5.0 qwen2/modeling_qwen2.py:206
    with quantization inserted between RoPE and cache update.
    Differs from Mistral only in `self.sliding_window` lookup
    (instance attribute, not config).
    """
    from transformers.models.qwen2.modeling_qwen2 import (  # type: ignore
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

    # ★★★ INSERTED QUANTIZATION (R-α.1) ★★★
    if _QUANT_METHOD != "none":
        key_states = _quantize_kv(key_states, _QUANT_METHOD, _QUANT_BITS, is_key=True)
        value_states = _quantize_kv(value_states, _QUANT_METHOD, _QUANT_BITS, is_key=False)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx
        )

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def gemma4_quantized_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Patched Gemma4TextAttention.forward.

    Verbatim copy of transformers 5.5.0 gemma4/modeling_gemma4.py:1179
    with quantization inserted between RoPE and cache update.

    Gemma4 quirks vs Mistral/Qwen2:
      - Per-K/V `q_norm`/`k_norm`/`v_norm` (RMSNorm) before RoPE
      - Optional `is_kv_shared_layer`: K/V come from earlier layer's
        shared_layers dict — quantization is SKIPPED for shared layers
        because the source layer already quantized them when they were
        first computed (consistency with KVQuant pattern).
      - Optional `store_full_length_kv`: K/V also stored to shared_layers
        dict for downstream layers; quantized values flow through here.
      - `v_proj` may be `None` (then value_states = key_states).
      - RoPE uses `unsqueeze_dim=2` (vs default 1 for Mistral/Qwen2).
    """
    from transformers.models.gemma4.modeling_gemma4 import (  # type: ignore
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
        # Shared K/V — already quantized by source layer when stored.
        # Skip quantization here to avoid double-quantization.
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

        # ★★★ INSERTED QUANTIZATION (R-α.1) ★★★
        # Only on non-shared layers; shared layers reuse already-
        # quantized K/V from the source layer's stored shared_layers.
        if _QUANT_METHOD != "none":
            key_states = _quantize_kv(key_states, _QUANT_METHOD, _QUANT_BITS, is_key=True)
            value_states = _quantize_kv(value_states, _QUANT_METHOD, _QUANT_BITS, is_key=False)

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
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=self.attention_dropout if self.training else 0.0,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


# ═══════════════════════════════════════════════════════════════════════
# Patch dispatch — class name → quantized forward function
# ═══════════════════════════════════════════════════════════════════════

_QUANTIZED_FORWARDS_BY_CLASS: dict[str, Callable] = {
    "MistralAttention": mistral_quantized_forward,
    "Qwen2Attention": qwen2_quantized_forward,
    "Gemma4TextAttention": gemma4_quantized_forward,
}


def patch_model_for_quantization(model: nn.Module, method: str, bits: int) -> int:
    """Replace the forward method of every supported attention module
    in `model` with the matching quantized variant. Sets module-level
    state so the patched forwards apply (method, bits) at call time.

    Returns the number of attention modules patched (for logging).

    Re-callable: subsequent calls update method/bits and re-attach the
    patched forward (idempotent in effect). The original forward is
    preserved on the first patch via _ORIGINAL_FORWARDS so unpatch
    can restore it later.
    """
    global _QUANT_METHOD, _QUANT_BITS

    if method not in ("none", "fajarquant", "kivi", "turboquant"):
        raise ValueError(f"unknown quantization method: {method}")
    if method != "none" and bits not in (2, 3, 4):
        raise ValueError(f"bits must be 2, 3, or 4 (got {bits})")

    _QUANT_METHOD = method
    _QUANT_BITS = bits

    patched = 0
    for module in model.modules():
        cls_name = type(module).__name__
        if cls_name not in _QUANTIZED_FORWARDS_BY_CLASS:
            continue
        mid = id(module)
        if mid not in _ORIGINAL_FORWARDS:
            _ORIGINAL_FORWARDS[mid] = module.forward
        module.forward = types.MethodType(_QUANTIZED_FORWARDS_BY_CLASS[cls_name], module)
        patched += 1

    return patched


def unpatch_model(model: nn.Module) -> int:
    """Restore the original forward method of every patched attention
    module. Sets _QUANT_METHOD to "none" so any straggler invocation
    via the patched forward becomes a no-op.

    Returns the number of attention modules restored.
    """
    global _QUANT_METHOD, _QUANT_BITS
    _QUANT_METHOD = "none"
    _QUANT_BITS = 0

    restored = 0
    for module in model.modules():
        mid = id(module)
        if mid in _ORIGINAL_FORWARDS:
            module.forward = _ORIGINAL_FORWARDS[mid]
            restored += 1
    return restored


# ═══════════════════════════════════════════════════════════════════════
# Self-test (no GPU): import sanity + quantization function shape check
# ═══════════════════════════════════════════════════════════════════════


def _self_test() -> None:
    """Run import + shape sanity checks. Does NOT touch CUDA."""
    print("[quant_attention] self-test starting...")

    # Tiny synthetic K tensor: (B=1, H=4, S=64, D=128)
    k = torch.randn(1, 4, 64, 128, dtype=torch.float32)

    # Each method should return same shape, no NaN/Inf
    fq2 = apply_fajarquant_4d(k, 2)
    assert fq2.shape == k.shape, f"FQ shape mismatch: {fq2.shape} vs {k.shape}"
    assert torch.isfinite(fq2).all(), "FQ produced NaN/Inf"

    kk2 = apply_kivi_keys_4d(k, 2)
    assert kk2.shape == k.shape and torch.isfinite(kk2).all()

    kv2 = apply_kivi_values_4d(k, 2)
    assert kv2.shape == k.shape and torch.isfinite(kv2).all()

    tq2 = apply_turboquant_4d(k, 2)
    assert tq2.shape == k.shape and torch.isfinite(tq2).all()

    # Round-trip sanity: at 4-bit, FQ should be very close to input
    fq4 = apply_fajarquant_4d(k, 4)
    rel_err = (fq4 - k).abs().mean().item() / k.abs().mean().item()
    assert rel_err < 0.5, f"FQ 4-bit unreasonably lossy: rel_err={rel_err:.3f}"

    # Dispatch
    fq_via_dispatch = _quantize_kv(k, "fajarquant", 2, is_key=True)
    assert torch.allclose(fq_via_dispatch, fq2, atol=1e-5), "dispatch mismatch"

    none_via_dispatch = _quantize_kv(k, "none", 0, is_key=True)
    assert torch.equal(none_via_dispatch, k), "none dispatch mutated input"

    print("[quant_attention] self-test PASS")
    print(f"  FP32→FQ-2bit relerr: {(fq2 - k).abs().mean().item() / k.abs().mean().item():.4f}")
    print(f"  FP32→FQ-4bit relerr: {rel_err:.4f}")


if __name__ == "__main__":
    _self_test()
