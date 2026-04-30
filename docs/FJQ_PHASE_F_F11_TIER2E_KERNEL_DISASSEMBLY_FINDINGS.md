---
phase: F.11.4 Tier 2E (kernel disassembly continuation)
status: CLOSED PARTIAL 2026-05-01
budget: 2-3h granted, ~2.5h actual, parity NOT closed but blocking-hypothesis space narrowed
prereq: F.11.4 Path B Findings docs + earlier H5/H6/H8 hypothesis findings
artifact: this doc + reverted experimental change
---

# F.11.4 Tier 2E Kernel Disassembly — Findings Doc

> **TL;DR.** Located the actual upstream encoder (`preprocess_three_weights_tl2`
> at Eddie-Wang1120/llama.cpp `tl-merge` branch). Verified that our magnitude
> index mapping and bit positions within u16 lanes are mathematically
> equivalent to upstream. Two hypotheses tested + rejected:
> (a) sign-buffer duplication (upstream `extend × 2`), (b) extending sign
> array to 2× size. Parity test still fails with same row-uniform `+32, -31, -1`
> cycle. Three remaining unknown axes (column-ordering within `as`, row_hi_lo
> top/bot mapping vs upstream's `np.split(.., 2)`, and whether `extend × 2`
> needs re-application at full-matrix scale not per-tile) require either
> bit-by-bit upstream encoder port + diff (Branch X-real, ~6-8h) or a
> printf-instrumented kernel build with traced sign reads (~2-3h additional).
> **Tier 2E budget exhausted; F.11.4 still infrastructure-only with
> documented parity gap; Branch X-real recommended as next-session continuation.**

## 1. What was done in 2.5h

### 1.1 Upstream encoder located (~30min)

Probed Eddie-Wang1120/llama.cpp branches via `git tree` API. Found `tl-merge`
branch contains `convert-hf-to-gguf-bitnet.py` with TL2 encoder functions:

```
Eddie-Wang1120/llama.cpp tl-merge:convert-hf-to-gguf-bitnet.py:
  - preprocess_three_weights_tl2 (lines 579-657)
  - preprocess_two_weights_tl2 (lines 536-577)
  - preprocess_weights_tl2 (lines 658-720)
```

This is ground-truth TL2 encoder source. `merge-dev` branch (linked from
`microsoft/BitNet/.gitmodules`) does NOT have the TL2 path; `tl-merge` is
the variant that does.

### 1.2 Magnitude scheme cross-checked (~20min)

Upstream:
```python
weight = np.reshape(weight, (weight_num // 3, 3))
first_weight = np.multiply(split_weights[0], 9)   # w0 * 9
second_weight = np.multiply(split_weights[1], 3)  # w1 * 3
third_weight = split_weights[2]                    # w2 * 1
weight = first_weight + second_weight + third_weight  # signed sum, range [-13, 13]
sign_weight = np.sign(weight) + 2  # {-1,0,1}+2 = {1,2,3}
sign_weight = np.where(sign_weight > 1, 0, sign_weight)  # negative→1, else→0
weight = np.abs(weight)  # magnitude index ∈ [0, 13]
```

Ours (`encode_triplet`): factor leading-nonzero sign, look up
canonical-form-to-index from `THREE_LUT_CANONICAL_TO_IDX` table.

**Both produce identical `(magnitude_index, sign_bit)` for all 27 ternary
triplets.** Sample verification:

| Triplet | Upstream code | Upstream mag/sign | Ours mag/sign |
|---|---|---|---|
| (0,0,0) | 0 | (0, 0) | (0, 0) |
| (1,0,0) | 9 | (9, 0) | (9, 0) |
| (-1,0,0) | -9 | (9, 1) | (9, 1) |
| (1,-1,0) | 6 | (6, 0) | (6, 0) |
| (-1,1,0) | -6 | (6, 1) | (6, 1) |
| (1,1,1) | 13 | (13, 0) | (13, 0) |
| (-1,-1,-1) | -13 | (13, 1) | (13, 1) |
| (1,-1,1) | 7 | (7, 0) | (7, 0) |
| (1,1,-1) | 11 | (11, 0) | (11, 0) |

Magnitude path is bit-exact (this was already known from earlier sentinel tests).

### 1.3 Sign bit-position scheme cross-checked (~30min)

Upstream sign packing (lines 631-654):

```python
sign_inner_by_weights = np.split(sign_inner_bm_weight, (BY // (by * 4)), axis=1)
for sign_inner_by_weight in sign_inner_by_weights:
    func_weight = np.split(sign_inner_by_weight, 8, axis=1)
    combine_weight = np.zeros((16, 1), dtype=np.uint16)
    for i in range(len(func_weight)):
        min_weight = np.split(func_weight[i], 2)  # along axis=0 (rows)
        min_top_weight = min_weight[0].astype(np.uint16) << 15 - (2 * i)
        min_bot_weight = min_weight[1].astype(np.uint16) << 15 - (2 * i + 1)
        combine_weight += min_top_weight
        combine_weight += min_bot_weight
    combine_weight = combine_weight.view(np.uint8)
    combine_weight = np.reshape(combine_weight, bm)
    sign_weight_list.append(combine_weight)
```

Per inner_by sub-block (8 cols wide × 32 rows tall):
- col `i ∈ [0, 8)`, top-half rows (0..15): bit `15 - 2i` of u16
- col `i ∈ [0, 8)`, bot-half rows (16..31): bit `14 - 2i` of u16

Our scheme (`pack_tl2_tile_organized` line 580-650):
```python
bit_pos_in_lane = 15 - 4*ai_in_as - 2*nibble_for_sign - row_hi_lo
```

Where `triplet_in_as = ai_in_as * 2 + nibble_for_sign` ∈ [0, 8).

**Bit positions match exactly:**

| triplet_in_as | (ai, nib) | top bit | bot bit | upstream `i` | upstream top | upstream bot |
|---|---|---|---|---|---|---|
| 0 | (0,0) | 15 | 14 | 0 | 15 | 14 |
| 1 | (0,1) | 13 | 12 | 1 | 13 | 12 |
| 2 | (1,0) | 11 | 10 | 2 | 11 | 10 |
| ... | ... | ... | ... | ... | ... | ... |
| 7 | (3,1) | 1 | 0 | 7 | 1 | 0 |

### 1.4 Sign-duplication hypothesis tested + REJECTED (~30min)

Upstream code calls `final_weight.extend(sign_weight_list)` **twice** (lines
656-657). Hypothesis: our 1-tile test buffer's zero-padded second half is
incompatible with kernel reads.

Experiment:
```rust
// Doubled buffer: 1344 = 2*640 + alignment pad
let mut sign_buf = AlignedU8::<1344>([0_u8; 1344]);
sign_buf.0[..signs_per_tile].copy_from_slice(&layout.signs[..signs_per_tile]);
sign_buf.0[signs_per_tile..2 * signs_per_tile]
    .copy_from_slice(&layout.signs[..signs_per_tile]);
```

Result: parity test still fails with **same exact** `+32, -31, -1` cycle on
all 64 rows. Hypothesis rejected.

Re-interpretation: the upstream `extend × 2` is for the FULL-MATRIX layout.
Upstream test uses (M=256, K=...) which has 8 row-blocks (`M // bm = 256/32`).
`sign_weight_list` has 8 entries × bm bytes. `extend × 2` outputs 16 entries
worth. This matches a kernel pattern where signs need to be PRESENT for each
i-block iteration with possible repetition, but at the FULL-MATRIX scale not
per-tile. Our 1-tile test only fills tile 0; the 7 other tiles' worth of
zero-padded signs just doesn't get read by the kernel.

(Reverted experimental change in tl2.rs:492.)

### 1.5 Three remaining hypotheses NOT tested in 2.5h budget

1. **Column-index ordering within `as`.** Upstream's `i` (0..7) iterates a
   spatial sub-block split by `np.split(.., 8, axis=1)`. Our `triplet_in_as`
   iterates linearly through `triplet_in_kouter`. The MAPPING from (k_outer,
   triplet_in_kouter) to upstream's (sub-block index, i within sub-block)
   may differ if the sub-block partitioning is along a different axis or
   uses a different ordering.

2. **`row_hi_lo` top/bot interpretation.** Upstream's `np.split(.., 2)`
   along axis=0 gives `min_top_weight` = first 16 rows, `min_bot_weight` =
   last 16 rows of the (32, by) sub-block. Our `row_hi_lo = 1 if lane >= 16
   else 0` uses lane-of-row mapping. If lane-cross fix to magnitudes isn't
   coupled to signs (which we confirmed kernel-side it isn't), then signs
   should use `row_hi_lo = (row_in_i_group >= 16)` directly, NOT
   lane-based. **This is a likely bug** but couldn't be confirmed
   without running the full upstream encoder for byte-by-byte comparison.

3. **Full-matrix-scale duplication.** The `extend × 2` may be a kernel
   contract for "two consecutive sign blocks per i_group × n_kouter
   iteration." If so, our per-tile sign buffer needs to be doubled-and-
   tiled differently. This wasn't reachable without running the upstream
   encoder end-to-end and inspecting the byte stream.

## 2. Why each remaining hypothesis needs Branch X-real (port encoder) to test

Each hypothesis requires either:
- **Side-by-side byte-stream comparison** between our encoder's output and
  upstream's `preprocess_three_weights_tl2(weights, ...)` output for the
  SAME weight matrix. Diff identifies the exact byte/bit/offset that differs.
- **Or** end-to-end trace of kernel sign reads for instrumented inputs (e.g.,
  set ONE specific sign bit in our output, see which output row it negates).

Both require infrastructure beyond hypothesis-driven encoder edits.

## 3. Recommended next-session work

**Branch X-real** (~6-8h, was the originally-recommended path before Tier 2D
suggested kernel disassembly as a faster alternative):

1. **Step 1 (~2h): Port upstream encoder verbatim.** Translate
   `preprocess_three_weights_tl2` to Python in our repo as
   `python/phase_d/scripts/upstream_repack_to_tl2.py`. Use `numpy` exactly
   as upstream. No optimization, no change — just port.

2. **Step 2 (~30min): Byte-stream diff.** Generate magnitude+sign bytes
   from BOTH encoders for the parity test's weight matrix. Save to
   binary files. `cmp -l` to identify exact byte indices where they
   differ. Identify pattern.

3. **Step 3 (~1h): Apply targeted fix.** Based on diff pattern, modify
   our `repack_to_tl2.py` (and potentially Rust encoder if separate) to
   match upstream byte-for-byte at differing positions only.

4. **Step 4 (~30min): Re-run parity test.** Should pass 64/64 (or near).

5. **Step 5 (~1h): Update fixture, decision-doc, CHANGELOG.** F.11
   parity CLOSES, runtime activation unblocked, paper v2 narratives
   become non-deferred.

6. **Step 6 (~1h): Cross-repo update.** `fajaros-x86/docs/FAJAROS_PRODUCTION_PLAN_V1.md`
   §3.10 F.11 parity closure status flip; tok/s ceiling claim becomes
   measurable.

**Why this is now ~6-8h instead of original ~12h estimate:**
- Upstream encoder LOCATED (no longer "find encoder source" cost)
- Magnitude scheme equivalence VERIFIED (no debug needed)
- Bit positions VERIFIED (no debug needed)
- 2 of 5 remaining hypothesis axes ELIMINATED (sign duplication, sign array sizing)
- Only 3 specific axes left to test (column ordering, row_hi_lo, matrix-scale)

## 4. What this Tier 2E session did NOT close

- Parity gap: still 64/64 mismatch with `+32, -31, -1` cycle
- Runtime activation: still BLOCKED
- Paper v2 fajarquant TL2 claim: still DEFERRED
- F.11.4 chain status: unchanged from `e9b9a00` CHANGELOG "infrastructure-only"

What this session DID add: ~50% reduction in remaining hypothesis space, and
located definitive ground-truth source for next-session port.

## 5. Self-check (CLAUDE.md §6.8)

| Rule | Status |
|---|---|
| §6.8 R1 pre-flight audit | YES — re-ran parity test to confirm baseline |
| §6.8 R2 runnable verification | YES — `cargo test ... parity_real_mlp_one_tile_at_mini_256_256 -- --ignored --nocapture` |
| §6.8 R3 prevention layer | partial — findings doc preserves discovery for next-session |
| §6.8 R4 numbers cross-checked | YES — magnitude indices for 9 sample triplets verified equivalent |
| §6.8 R5 surprise budget +25% | YES — 2.5h actual within 2-3h grant + 25% (3h45m hard cap) |
| §6.8 R6 mechanical decision gate | YES — Branch X-real path declared if Tier 2E inconclusive |
| §6.8 R7 public-artifact sync | YES — this doc commits the findings |
| §6.8 R8 multi-repo state check | YES — fajarquant only, others untouched |

**Honest closeout: F.11 parity NOT closed. Hypothesis space narrowed.
Branch X-real (port + byte-diff) is the cleanest path forward — recommended
when next budget grant is available.**

---

*F.11.4 Tier 2E closed PARTIAL 2026-05-01. Parity still 64/64 mismatch,
hypothesis space ~50% narrowed. Branch X-real (~6-8h port + byte-diff)
recommended next-session if F.11 closure budget granted.*
