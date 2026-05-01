"""Tests for `intllm.quant.SparseFusedBitLinear` — the F.10.2 sparse
extension of FusedBitLinear (composition pattern wrapping upstream).

Verifies:
  1. Dense path (sparse_n=0) produces same forward output as a freshly-
     constructed plain FusedBitLinear with equal weights/norm.
  2. Sparse path (sparse_n=2, sparse_m=4) produces structurally-sparse
     output (50% zeros at the weight level after masking).
  3. Backward pass: gradients flow only to positions where mask==1.
  4. End-to-end forward + backward succeeds on CUDA.

These are the F.10.2 test plan items 1-4 from
`docs/FJQ_PHASE_F_F10_2_BITLINEAR_DESIGN.md` §5.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Trigger the same _upstream sys.path injection that intllm.model does,
# so mmfreelm is importable in tests without ordering dependence.
_UPSTREAM = Path(__file__).resolve().parent.parent / "_upstream"
if _UPSTREAM.is_dir() and str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))

HAS_CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Test 1 — dense path (sparse_n=0) bit-exact to upstream FusedBitLinear
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_CUDA,
    reason="FusedBitLinear's fused autograd kernel requires CUDA tensors",
)
def test_sparse_n_zero_matches_parent_forward():
    """sparse_n=0 → forward() output matches upstream FusedBitLinear
    when constructed with same weights + norm parameters.
    """
    from intllm.quant import SparseFusedBitLinear
    from mmfreelm.ops.fusedbitnet import FusedBitLinear

    torch.manual_seed(0xF102_0001)
    in_f, out_f = 64, 128

    # Two layers with identical seeds → same init weights/norm.
    sparse_disabled = SparseFusedBitLinear(in_f, out_f, sparse_n=0).cuda()
    plain = FusedBitLinear(in_f, out_f).cuda()

    # Copy params from sparse to plain so we compare on equal weights.
    plain.weight.data.copy_(sparse_disabled._inner.weight.data)
    plain.norm.weight.data.copy_(sparse_disabled._inner.norm.weight.data)

    x = torch.randn(2, 8, in_f, device="cuda")
    y_sparse = sparse_disabled(x)
    y_plain = plain(x)

    # Bit-exact match expected: both call the same fused autograd Function.
    assert torch.allclose(y_sparse, y_plain, atol=1e-5, rtol=1e-5), (
        f"dense path diverged from parent: max diff "
        f"{(y_sparse - y_plain).abs().max().item():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 2 — sparse path produces structural 50% zeros
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_CUDA, reason="sparse path uses Triton mask kernel")
def test_sparse_n_two_produces_50pct_mask_zeros():
    """sparse_n=2, sparse_m=4 → mask zeros out exactly 50% of weights
    in the effective (masked) weight applied during forward.

    We can't directly observe the masked weight from forward output,
    but we can verify by calling the mask creator on the same weight
    + checking the sparsity ratio matches exactly 0.5.
    """
    from intllm.quant import SparseFusedBitLinear
    from intllm.sparse_kernel import mask_creator_triton_optimized
    from mmfreelm.ops.fusedbitnet import weight_quant

    torch.manual_seed(0xF102_0002)
    layer = SparseFusedBitLinear(64, 128, sparse_n=2, sparse_m=4).cuda()

    # The mask is computed on the QUANTIZED weight, not raw.
    w = layer._inner.weight
    w_quant = weight_quant(w)  # ternary {-1, 0, +1} after the upstream quant op
    mask = mask_creator_triton_optimized(w_quant.detach(), N=2, M=4)

    sparsity_ratio = (mask == 0).float().mean().item()
    # 2:4 sparsity → exactly 50% zeros (2 out of 4 elements per group)
    assert abs(sparsity_ratio - 0.5) < 1e-6, (
        f"expected 0.5 sparsity ratio, got {sparsity_ratio}"
    )


# ---------------------------------------------------------------------------
# Test 3 — sparse path forward + backward succeeds on CUDA
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_CUDA, reason="sparse path requires CUDA")
def test_sparse_path_forward_backward_cuda():
    """End-to-end sparse_n=2 forward + loss + backward succeeds without
    CUDA errors, NaN, or graph corruption. Verifies the STE path is
    autograd-correct (mask treated as constant; gradient flows through
    quantized weight).
    """
    from intllm.quant import SparseFusedBitLinear

    torch.manual_seed(0xF102_0003)
    layer = SparseFusedBitLinear(64, 128, sparse_n=2, sparse_m=4).cuda()
    x = torch.randn(4, 16, 64, device="cuda", requires_grad=True)
    y = layer(x)
    assert y.shape == (4, 16, 128), f"unexpected output shape: {y.shape}"

    # Synthetic scalar loss + backward
    loss = y.pow(2).mean()
    loss.backward()

    # Verify gradient on weight is finite + non-zero (some positions get
    # gradient since mask isn't all-zero).
    assert layer._inner.weight.grad is not None, "no gradient propagated to weight"
    assert torch.isfinite(layer._inner.weight.grad).all(), "non-finite gradient"
    assert layer._inner.weight.grad.abs().sum().item() > 0, "all gradients are zero"


# ---------------------------------------------------------------------------
# Test 4 — gradient flows only to unmasked positions (STE correctness)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_CUDA, reason="F.10.3 mask refresh test requires CUDA")
def test_mask_refresh_interval_default_recomputes_every_forward():
    """interval=0 (default) → mask recomputed every forward.

    Verify by counting how many times the cached mask differs across
    consecutive forwards on a layer whose weight gets perturbed between
    forwards. With every-forward refresh, every perturbation should be
    reflected in the cached mask immediately.
    """
    from intllm.quant import SparseFusedBitLinear

    torch.manual_seed(0xF103_0001)
    layer = SparseFusedBitLinear(
        64, 128, sparse_n=2, sparse_m=4, mask_refresh_interval=0
    ).cuda()
    x = torch.randn(2, 8, 64, device="cuda")

    # First forward — establishes initial mask
    _ = layer(x)
    mask_a = layer._cached_mask.clone()

    # Perturb weight, expect mask to change on next forward
    with torch.no_grad():
        layer._inner.weight.data += 0.5 * torch.randn_like(layer._inner.weight)
    _ = layer(x)
    mask_b = layer._cached_mask.clone()

    # Different mask expected — perturbation should change mask under
    # every-forward refresh policy.
    assert not torch.equal(mask_a, mask_b), (
        "interval=0 failed to refresh mask after weight perturbation"
    )


@pytest.mark.skipif(not HAS_CUDA, reason="F.10.3 mask refresh test requires CUDA")
def test_mask_refresh_interval_caches_for_n_steps():
    """interval=10 → first forward refreshes; next 9 reuse cache; 11th
    refreshes again.

    Verify by perturbing weight between forwards but keeping interval=10:
    the cached mask should NOT change for the first 9 perturbed forwards
    (cache is reused), then update on the 10th.
    """
    from intllm.quant import SparseFusedBitLinear

    torch.manual_seed(0xF103_0002)
    layer = SparseFusedBitLinear(
        64, 128, sparse_n=2, sparse_m=4, mask_refresh_interval=10
    ).cuda()
    x = torch.randn(2, 8, 64, device="cuda")

    # Step 0 (counter=0): recomputes mask (always refreshes at counter=0)
    _ = layer(x)
    mask_0 = layer._cached_mask.clone()

    # Steps 1-9: perturb weight + forward; expect cached mask UNCHANGED
    for step in range(1, 10):
        with torch.no_grad():
            layer._inner.weight.data += 0.5 * torch.randn_like(layer._inner.weight)
        _ = layer(x)
        assert torch.equal(layer._cached_mask, mask_0), (
            f"step {step}: expected cached mask unchanged for interval=10, "
            f"but mask diverged (this means refresh fired prematurely)"
        )

    # Step 10 (counter==10, 10 % 10 == 0): refresh expected
    with torch.no_grad():
        layer._inner.weight.data += 0.5 * torch.randn_like(layer._inner.weight)
    _ = layer(x)
    mask_10 = layer._cached_mask.clone()
    assert not torch.equal(mask_0, mask_10), (
        "step 10: expected mask refresh, but cache unchanged"
    )


@pytest.mark.skipif(not HAS_CUDA, reason="F.10.3 manual refresh test requires CUDA")
def test_explicit_refresh_mask_method():
    """layer.refresh_mask() recomputes + returns mask without running forward.

    Useful for training-loop callbacks that want to refresh between epochs
    or at custom intervals (e.g. cosine-decay refresh frequency).
    """
    from intllm.quant import SparseFusedBitLinear

    torch.manual_seed(0xF103_0003)
    layer = SparseFusedBitLinear(
        64, 128, sparse_n=2, sparse_m=4, mask_refresh_interval=1000
    ).cuda()
    x = torch.randn(2, 8, 64, device="cuda")

    # Trigger first forward → cache established at step 0
    _ = layer(x)
    mask_initial = layer._cached_mask.clone()

    # Perturb weight + call refresh_mask explicitly
    with torch.no_grad():
        layer._inner.weight.data += 0.5 * torch.randn_like(layer._inner.weight)
    new_mask = layer.refresh_mask()

    # refresh_mask should return a NEW mask (different from initial)
    assert not torch.equal(mask_initial, new_mask), (
        "refresh_mask() did not produce a new mask after weight perturbation"
    )
    # And cache should now hold the new mask
    assert torch.equal(layer._cached_mask, new_mask), (
        "refresh_mask() returned a mask different from the cached one"
    )


@pytest.mark.skipif(not HAS_CUDA, reason="sparse path requires CUDA")
def test_sparse_grad_flows_only_to_unmasked_positions():
    """For positions where mask==0, gradient on `weight` should be 0.
    For positions where mask==1, gradient should be non-zero (in
    expectation — depends on activation pattern, but for random input
    + small layer the chance of exactly-zero gradient is ~0).

    This is the key autograd-correctness check for the STE pattern in
    SparseFusedBitLinear.forward.
    """
    from intllm.quant import SparseFusedBitLinear
    from intllm.sparse_kernel import mask_creator_triton_optimized
    from mmfreelm.ops.fusedbitnet import weight_quant

    torch.manual_seed(0xF102_0004)
    layer = SparseFusedBitLinear(64, 128, sparse_n=2, sparse_m=4).cuda()
    x = torch.randn(4, 16, 64, device="cuda")
    y = layer(x)
    loss = y.pow(2).sum()
    loss.backward()

    # Recompute the mask the same way forward did (deterministic — same weight)
    w = layer._inner.weight
    w_quant = weight_quant(w).detach()
    mask = mask_creator_triton_optimized(w_quant, N=2, M=4)

    grad = layer._inner.weight.grad
    masked_zero_positions = mask == 0
    grad_at_masked_positions = grad[masked_zero_positions]

    # Strict invariant: gradient at mask==0 positions is exactly 0.
    # Why? Because forward computes w_quant_sparse = w_quant * mask, and at
    # mask==0 positions the value is 0 regardless of w. Backward through
    # multiplication: ∂(w_quant * mask)/∂w_quant = mask. STE: ∂w_quant/∂w = 1
    # (at unmasked positions). So gradient at mask==0 is 0.
    assert grad_at_masked_positions.abs().max().item() < 1e-7, (
        f"gradient leak to mask==0 positions: max "
        f"{grad_at_masked_positions.abs().max().item():.2e}"
    )

    # Sanity: gradient at unmasked positions is non-zero
    grad_at_unmasked = grad[~masked_zero_positions]
    assert grad_at_unmasked.abs().sum().item() > 0, (
        "all unmasked-position gradients are zero — autograd path may be broken"
    )
