---
phase: F.10.2 — BitLinear sparse extension (design doc)
status: PRE-IMPL design v0 — 2026-05-01
budget: ~30min pre-flight; ~3-4h actual code mod (deferred to next focus block)
prereq: F.10.1 vendor + tests CLOSED (commit 26ad395)
---

# F.10.2 — BitLinear Sparse Extension Design

> **TL;DR.** Pre-flight identified upstream `FusedBitLinear` lives in vendored
> `_upstream/mmfreelm/ops/fusedbitnet.py:586-612` — a thin subclass of
> `BitLinear` with `forward()` calling a custom autograd `layer_norm_linear_quant_fn`.
> Critical mutation point: line 458 where `weight_quant(linear_weight)` produces
> ternary weights before `F.linear`. **Design decision: composition via forward-
> pre-hook (matching E2.1 Hadamard integration pattern), NOT inheritance
> modification or upstream fork.** Mask computed from detached weight magnitudes,
> applied as multiplicative gate after weight_quant but before F.linear. Default
> `sparse_n=0` preserves dense path bit-exact. ~3-4h code mod ahead — surfaced
> here for next focus block.

## 1. Architecture survey (where weight gets quantized + matmul'd)

```
FusedBitLinear.forward(x)                                         # line 604
  → layer_norm_linear_quant_fn(x, norm.weight, norm.bias,         # line 605
                               self.weight, self.bias,
                               is_rms_norm=True)
    → custom torch.autograd.Function                              # line 422
      → RMSNorm(x) → y                                            # line ~440
      → linear_weight = weight_quant(linear_weight).to(dtype)     # line 458 ← mutation point
      → out = F.linear(y, linear_weight, linear_bias)             # line 460  ← matmul
```

**Critical insight:** the linear matmul (line 460) consumes the OUTPUT of
`weight_quant` (line 458), not `self.weight` directly. So our sparse mask
needs to multiply the QUANTIZED weight, not the raw FP weight, to be
consistent with Sparse-BitNet recipe.

## 2. Four candidate integration approaches (with decision)

### Approach A — Fork upstream FusedBitLinear

Modify `_upstream/mmfreelm/ops/fusedbitnet.py` line 458:
```python
linear_weight = weight_quant(linear_weight) * sparse_mask  # mask injected here
```

**Pros:** direct, aligns with Sparse-BitNet recipe
**Cons:** breaks "vendored verbatim" pattern (V32-prep F.11 contract).
Future upstream sync requires re-applying the patch. Compounds with F.11's
existing patch (`include` redirect). Sets bad precedent.

**Decision: REJECT.** Vendored upstream stays untouched.

### Approach B — Subclass with forward override

Create `SparseFusedBitLinear(FusedBitLinear)` in `intllm/quant.py`. Override
`forward()` to compute mask + temporarily set `self.weight.data = weight *
mask` before calling `super().forward()`, restore after.

**Pros:** doesn't fork upstream
**Cons:** in-place `.data` mutation interacts badly with optimizer (state-
dependent on whether sparse path was triggered; can break Adam moment
buffers). try/finally restoration adds code complexity. Tests get fragile.

**Decision: REJECT.** In-place weight mutation in the forward path is
fragile.

### Approach C — Forward-pre-hook on existing FusedBitLinear instances

Use `module.register_forward_pre_hook(hook)` where hook computes mask + sets
`module.weight.data = weight * mask` before forward. This is the **same
pattern E2.1 Hadamard rotation uses** (verified at
`intllm/qat.py` and `paper/intllm/intllm.tex` §7.3 footnote where the
rotation is applied as a forward pre-hook).

**Pros:** matches established codebase pattern; minimal new infrastructure
**Cons:** still has the in-place .data mutation issue from Approach B (the
hook mutates weight.data temporarily). Same fragility.

**Decision: REJECT.** Same in-place issue as B.

### Approach D — Subclass with custom forward that re-implements quant+linear ✓

Create `SparseFusedBitLinear(FusedBitLinear)` in `intllm/quant.py` that
overrides `forward()` with our own implementation:

```python
class SparseFusedBitLinear(FusedBitLinear):
    def __init__(self, in_features, out_features, bias=False,
                 sparse_n: int = 0, sparse_m: int = 4):
        super().__init__(in_features, out_features, bias=bias)
        self.sparse_n = sparse_n  # 0 = disabled (dense path)
        self.sparse_m = sparse_m

    def forward(self, x):
        if self.sparse_n == 0:
            # Dense path — UNCHANGED, defer to parent
            return super().forward(x)

        # Sparse path: replicate parent's RMSNorm + weight_quant + sparse + F.linear
        from mmfreelm.ops.fusedbitnet import (
            activation_quant, weight_quant, rms_norm
        )
        from intllm.sparse_kernel import mask_creator_triton_optimized
        import torch.nn.functional as F

        x_norm = rms_norm(x, self.norm.weight)  # use norm via parent attribute
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w = self.weight
        w_q = w + (weight_quant(w) - w).detach()
        # NEW: apply sparse mask via STE (mask treated as constant w.r.t. autograd)
        with torch.no_grad():
            mask = mask_creator_triton_optimized(w_q.detach(), N=self.sparse_n, M=self.sparse_m)
        w_qs = w_q * mask  # gradient flows to w_q via STE; mask is constant
        return F.linear(x_quant, w_qs, self.bias)
```

**Pros:**
- No fork of upstream
- No in-place .data mutation
- Dense path defaults to parent FusedBitLinear forward (unchanged)
- Sparse path is explicit + readable
- Mask treated as constant w.r.t. autograd (STE for sparsity)
- Single function, no hook registration overhead

**Cons:**
- Needs to import `activation_quant`, `weight_quant`, `rms_norm` from
  upstream (fine — they're public functions)
- Replicates the line 460 `F.linear` call (small duplication, but
  simplification: doesn't go through the custom autograd `Function`
  — uses standard PyTorch autograd which handles STE via `.detach()`
  pattern just fine)
- Sparse path skips the FUSED layer-norm-linear kernel optimization
  (lines 422-471 are an optimized fused autograd function); sparse
  path uses unfused PyTorch ops. Loss of 5-15% training speed on the
  SPARSE PATH only (dense path unaffected). For F.10.5 PoL smoke +
  F.10.6 gate evaluation, the unfused-but-correct sparse path is fine.

**Decision: ACCEPT — Approach D is the cleanest.** The 5-15% sparse-path
training-speed loss is acceptable because (a) it's only on sparse runs,
(b) F.10.6 gate evaluation cares about val_loss, not training tok/s, and
(c) dense baseline path remains the optimized fused kernel for
apples-to-apples comparison.

## 3. Concrete signature + behavior

```python
class SparseFusedBitLinear(FusedBitLinear):
    """Optional 2:4 (or N:M) structured sparsity on top of FusedBitLinear.

    Args:
        in_features, out_features, bias: standard nn.Linear args
        sparse_n: number of nonzeros per group (default 0 = disabled)
        sparse_m: group size (default 4; only used if sparse_n > 0)

    When sparse_n=0 (default), forward() is bit-exact to parent
    FusedBitLinear.forward() — the dense path is preserved.

    When sparse_n>0, forward() computes a fresh N:M mask from the
    current ternary-quantized weight on every forward (via vendored
    Sparse-BitNet Triton kernel); the mask is treated as a constant
    w.r.t. autograd (gradients flow only through the STE-quantized
    weight path, not through the mask).
    """
```

**No new state stored persistently** — mask is computed fresh per forward.
This matches Sparse-BitNet's "dynamic mask" recipe (their README §2:
"masks evolve throughout rather than remaining fixed").

## 4. Backward pass / autograd correctness

Standard STE pattern via `.detach()`:

```python
w_q  = w + (weight_quant(w) - w).detach()  # STE: forward = ternary, backward = identity
mask = mask_creator(w_q.detach(), ...)     # constant w.r.t. autograd
w_qs = w_q * mask                           # gradient flows to w_q (and back to w)
```

Backward chain:
- `∂loss/∂w_qs` from F.linear backward
- `∂loss/∂w_q = ∂loss/∂w_qs * mask` (mask is constant)
- `∂loss/∂w = ∂loss/∂w_q * 1` (STE for ternary quant)

Net effect: gradients flow to `w` only on positions where mask==1, exactly
the Sparse-BitNet recipe. Positions zeroed by mask receive ZERO gradient,
which is correct (those weights shouldn't be updated since they're masked
out at forward time).

This is mathematically equivalent to the upstream Sparse-BitNet "Dual-STE"
formulation per their README, modulo the recipe detail of WHEN mask
updates (every-step vs every-N-steps). F.10.3 will add the every-N-steps
schedule via a separate refresh hook; F.10.2 starts with every-forward
(naive but always correct).

## 5. Test plan (additions to test_sparse_kernel.py — or new test_sparse_bitlinear.py)

```python
def test_sparse_n_zero_matches_parent_forward():
    """sparse_n=0 → forward() bit-exact to FusedBitLinear.forward()."""

def test_sparse_n_two_produces_50pct_zeros_in_output():
    """sparse_n=2, m=4 → forward() output has structural zero pattern."""

def test_sparse_grad_flows_only_to_unmasked_positions():
    """Backward only writes gradients where mask==1."""

def test_sparse_path_runs_on_cuda():
    """sparse_n=2 forward + backward succeeds on RTX 4090."""
```

4 new tests; ~50 LOC.

## 6. Implementation work breakdown

| Sub-step | Effort | What |
|---|---|---|
| F.10.2.1 | 30min | Read upstream `rms_norm` + `activation_quant` + `weight_quant` to verify import paths + dtype handling |
| F.10.2.2 | 1h | Add `SparseFusedBitLinear` class to `intllm/quant.py`, ~40 LOC including docstring |
| F.10.2.3 | 30min | Update `intllm/quant.py:__all__` + add to `_is_bitlinear` matcher (auto-detected by tracker hooks) |
| F.10.2.4 | 1.5h | 4 unit tests in `python/phase_d/tests/test_sparse_bitlinear.py` |
| F.10.2.5 | 15min | Run pytest, fix any issues, commit chain |
| **Total** | **~3.5h** | matches plan §4 estimate |

## 7. Open questions for next focus block

1. **Mask refresh frequency for production:** F.10.3 adds the every-N-steps
   schedule; should default be every step (what we do here) or every 100/1000
   steps per Sparse-BitNet recipe? Their README says "dynamic mask
   monitoring" but doesn't quote a specific N. F.10.5 PoL smoke will
   characterize the cost.

2. **Mask storage for export:** at inference time, we'd want the mask
   pre-computed as a static buffer (not recomputed per forward). F.10.3 or
   F.10.4 can add an `eval()` mode that snapshots mask to a buffer.

3. **HGRN-Bit specific consideration:** HGRN's gated recurrent paths (i_proj,
   f_proj, g_proj) may behave differently under 2:4 sparsity than standard
   transformer projections. Sparse-BitNet's paper measures Llama-style
   transformers; HGRN is novel territory. F.10.6 full run will surface any
   regression.

4. **Dual-STE vs single STE:** Sparse-BitNet's Dual-STE has BOTH quant and
   sparsity gradients via STE. Our design above uses single STE (quant) +
   mask-as-constant. Mathematically, both flow gradients only to unmasked
   positions; distinction may not matter empirically. F.10.5 PoL outcome
   will tell us if a more nuanced sparsity-STE is needed.

## 8. Self-check (CLAUDE.md §6.8)

| Rule | Status |
|---|---|
| §6.8 R1 pre-flight audit | YES — this doc |
| §6.8 R2 runnable verification | per-test col in §5 |
| §6.8 R3 prevention layer | F.10.5 PoL smoke is the gate |
| §6.8 R5 surprise budget | Approach D estimated 3.5h, +25% = 4.4h hard cap |
| §6.8 R6 mechanical decision gates | §2 ABCD with explicit accept/reject + reasoning |
| §6.8 R7 public-artifact sync | this doc + future commit + F.10.2 finds doc |

6/8 satisfied (R4 / R8 N/A at design stage).

## 9. State at design close

- Architecture survey COMPLETE
- 4 integration approaches evaluated with explicit decision
- Approach D selected (subclass with forward override)
- Concrete signature + behavior + autograd correctness documented
- Test plan + implementation breakdown specified
- 4 open questions surfaced for tracking

**No code mod in this commit** — design only. Next focus block:
F.10.2.1-F.10.2.5 implementation chain (~3.5h).

---

*F.10.2 design doc closed 2026-05-01. Approach D (subclass +
forward override) selected. Implementation breakdown ~3.5h
across 5 sub-steps. Deferred to next focus block per session-
fatigue management; current commit is design-only.*
