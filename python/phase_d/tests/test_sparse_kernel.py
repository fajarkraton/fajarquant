"""Tests for `intllm.sparse_kernel` — vendored from AAzdi/Sparse-BitNet
(F.10.1 vendor step 2026-05-01).

Verifies the N:M sparsity invariant: for each group of M consecutive
elements, exactly N are non-zero in the produced mask. All other
elements are zero. The N kept positions correspond to the largest-
magnitude entries in the group (top-N by abs).

This is a vendoring smoke test, NOT an integration test. F.10.2 wires
the kernel into BitLinear; F.10.3 wires it into the train loop.
F.10.5 is the GPU PoL smoke that exercises end-to-end training.
"""

from __future__ import annotations

import pytest
import torch

# CUDA is required for the Triton path; CPU has only the PyTorch
# fallback (`mask_creator_pytorch`). On CPU-only test machines, the
# Triton-based tests are skipped via `pytest.mark.skipif`.
HAS_CUDA = torch.cuda.is_available()

from intllm.sparse_kernel import (
    mask_creator_pytorch,
    mask_creator_triton_optimized,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _verify_n_m_invariant(mask: torch.Tensor, N: int, M: int) -> None:
    """Assert mask has structurally exactly N nonzeros per M consecutive elements,
    and that all values in mask are 0 or 1.
    """
    n = N
    m = M
    assert mask.numel() % m == 0, f"mask numel {mask.numel()} not divisible by M={m}"

    flat = mask.flatten().cpu()
    # Values must be binary
    unique_vals = torch.unique(flat).tolist()
    assert set(unique_vals).issubset({0.0, 1.0}), f"non-binary values in mask: {unique_vals}"

    # Per-group invariant
    groups = flat.reshape(-1, m)
    nonzeros_per_group = (groups != 0).sum(dim=1)
    assert (nonzeros_per_group == n).all().item(), (
        f"N:M invariant violated: expected exactly {n} nonzeros per {m}-group; "
        f"got distribution: min={nonzeros_per_group.min().item()}, "
        f"max={nonzeros_per_group.max().item()}, "
        f"mean={nonzeros_per_group.float().mean().item():.3f}"
    )


def _verify_top_n_by_abs(tensor: torch.Tensor, mask: torch.Tensor, N: int, M: int) -> None:
    """Assert that the N positions kept by mask in each group of M correspond
    to the N largest absolute-value entries (with index-based tie-breaking).
    """
    n = N
    m = M
    flat_t = tensor.flatten().cpu()
    flat_m = mask.flatten().cpu()
    assert flat_t.numel() == flat_m.numel()

    n_groups = flat_t.numel() // m
    for g in range(n_groups):
        group_t = flat_t[g * m : (g + 1) * m]
        group_m = flat_m[g * m : (g + 1) * m]
        kept_indices = (group_m == 1).nonzero(as_tuple=True)[0].tolist()
        assert len(kept_indices) == n

        # Top-n indices by abs (with stable sort for tie-breaking)
        abs_vals = group_t.abs()
        # The kernel uses tie-breaking: equal abs values prefer LOWER
        # index. argsort is stable in PyTorch so it sorts by index
        # within equal values; we want descending abs so we use -abs.
        # For ties on -abs, argsort stable returns lower index first;
        # that matches "smaller index for tie-breaking" per the
        # upstream README.
        ranking = torch.argsort(-abs_vals, stable=True)
        top_n_expected = sorted(ranking[:n].tolist())
        assert sorted(kept_indices) == top_n_expected, (
            f"group {g}: top-n by abs mismatch. "
            f"kept={sorted(kept_indices)}, "
            f"expected={top_n_expected}, "
            f"abs_vals={abs_vals.tolist()}"
        )


# ---------------------------------------------------------------------------
# CPU (PyTorch fallback) tests
# ---------------------------------------------------------------------------


def test_pytorch_fallback_2_4_invariant():
    """On CPU via PyTorch fallback: 2:4 mask preserves invariant."""
    torch.manual_seed(0xF104_2026)
    tensor = torch.randn(16, 32)  # 16 × 32 = 512 elements = 128 groups of 4
    mask = mask_creator_pytorch(tensor, N=2, M=4)
    _verify_n_m_invariant(mask, N=2, M=4)
    _verify_top_n_by_abs(tensor, mask, N=2, M=4)


def test_pytorch_fallback_4_8_invariant():
    """4:8 sparsity (4 nonzeros per 8 elements), CPU path."""
    torch.manual_seed(0xF104_2027)
    tensor = torch.randn(8, 16)  # 128 elements = 16 groups of 8
    mask = mask_creator_pytorch(tensor, N=4, M=8)
    _verify_n_m_invariant(mask, N=4, M=8)
    _verify_top_n_by_abs(tensor, mask, N=4, M=8)


def test_pytorch_fallback_returns_binary_mask():
    """Mask values must be exactly {0, 1}, dtype float (matches upstream)."""
    tensor = torch.randn(4, 4)
    mask = mask_creator_pytorch(tensor, N=2, M=4)
    assert mask.dtype == torch.float32 or mask.dtype == torch.float64, (
        f"unexpected mask dtype: {mask.dtype}"
    )
    unique = torch.unique(mask).tolist()
    assert set(unique).issubset({0.0, 1.0})


def test_pytorch_fallback_preserves_shape():
    """Output mask shape == input tensor shape."""
    for shape in [(8, 16), (4, 4, 8), (32,)]:
        t = torch.randn(*shape)
        mask = mask_creator_pytorch(t, N=2, M=4)
        assert mask.shape == t.shape, f"shape mismatch: {mask.shape} != {t.shape}"


def test_pytorch_fallback_handles_zero_input():
    """All-zero input — mask should still satisfy invariant
    (kernel picks first n by index for ties)."""
    tensor = torch.zeros(2, 8)
    mask = mask_creator_pytorch(tensor, N=2, M=4)
    _verify_n_m_invariant(mask, N=2, M=4)


# ---------------------------------------------------------------------------
# CUDA (Triton) tests — gated on GPU availability
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available; Triton path requires GPU")
def test_triton_2_4_invariant_cuda():
    """On CUDA via Triton path: 2:4 mask preserves invariant."""
    torch.manual_seed(0xF104_2028)
    tensor = torch.randn(64, 64, device="cuda")
    mask = mask_creator_triton_optimized(tensor, N=2, M=4)
    _verify_n_m_invariant(mask, N=2, M=4)
    _verify_top_n_by_abs(tensor, mask, N=2, M=4)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available; Triton path requires GPU")
def test_triton_matches_pytorch_fallback_cuda():
    """Triton kernel produces same mask as PyTorch fallback for the same input.

    This is the key correctness check: if Triton diverges from PyTorch,
    F.10's training math is broken at the mask-creation step.
    """
    torch.manual_seed(0xF104_2029)
    tensor = torch.randn(32, 32, device="cuda")
    mask_triton = mask_creator_triton_optimized(tensor, N=2, M=4)
    mask_pytorch = mask_creator_pytorch(tensor, N=2, M=4)
    assert torch.equal(mask_triton.cpu(), mask_pytorch.cpu()), (
        "Triton and PyTorch fallback produced different masks for same input"
    )


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available; Triton path requires GPU")
def test_triton_2_4_specialized_kernel_cuda():
    """The 2:4 specialized kernel produces ~50% sparsity (correct ratio)."""
    torch.manual_seed(0xF104_2030)
    tensor = torch.randn(128, 128, device="cuda")
    mask = mask_creator_triton_optimized(tensor, N=2, M=4)
    sparsity = (mask == 0).float().mean().item()
    # Exactly 2:4 = 50% sparsity = 50% nonzeros
    assert abs(sparsity - 0.5) < 1e-6, f"expected 0.5 sparsity, got {sparsity}"
