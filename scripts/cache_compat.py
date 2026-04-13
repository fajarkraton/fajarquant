"""
cache_compat.py — DynamicCache API compatibility layer.

transformers 4.x: cache.key_cache[i] / cache.value_cache[i] (tensors)
transformers 5.x: cache.layers[i].keys / cache.layers[i].values (layer objects)

This module provides get_cache_kv() that works with both.
"""

from __future__ import annotations
import torch


def get_cache_n_layers(cache) -> int:
    """Get number of layers in a DynamicCache."""
    if hasattr(cache, "key_cache") and cache.key_cache:
        return len(cache.key_cache)
    if hasattr(cache, "layers"):
        return len(cache.layers)
    return 0


def get_cache_kv(cache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Get (keys, values) tensors for a layer from DynamicCache.

    Returns tensors with batch dim: (B, H, S, D).
    Both transformers 4.x and 5.x APIs supported.
    """
    if hasattr(cache, "key_cache") and cache.key_cache:
        # transformers 4.x API
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    # transformers 5.x API
    layer = cache.layers[layer_idx]
    return layer.keys, layer.values
