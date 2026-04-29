# Phase F F.11 Branch X-real iteration 7 — Hypothesis Exploration Findings (v1.0)

> **Status:** 3 hypotheses tested in-place (H5, H6, H8); none beat
> the iteration 6 baseline. Encoder reverted to iter 6 state. New
> finding: **the kernel only USES 40 of 42 ai-vectors per inner
> block** (KK/8 = 5 outer k iterations × 4 ai's = 20 used; vec_as[20]
> loaded but unused). This means the kernel is OBSERVING ONLY 80 of
> 84 triplets per row — 4 triplets (12 cols) implicitly DROPPED per
> inner block.
> **Origin:** F.11.4 Branch X-real iteration 6 (`1a52f90`) reduced
> parity diff 23x but didn't reach zero. Iteration 7 explored
> sign-bit position perturbations.

## Hypotheses tested

### H5: row_hi_lo polarity flipped

```
old: bit_pos = 15 - 4*ai - 2*nibble - row_hi_lo
H5:  bit_pos = 14 - 4*ai - 2*nibble + row_hi_lo
```

Result:
  row 0: scalar=0,    tl2=316, diff=+316 (10x WORSE than baseline)
  row 1: scalar=-127, tl2=30,  diff=+157
  row 2: scalar=127,  tl2=-126, diff=-253

Verdict: REJECTED. row_hi_lo polarity in iteration 6 was correct.

### H6: byte_in_lane_pair endianness flipped

```
old: byte_in_lane_pair = bit_pos / 8
H6:  byte_in_lane_pair = 1 - (bit_pos / 8)
```

Result:
  IDENTICAL to baseline (32, -31, -1).

Verdict: NULL EFFECT. Kernel reads BOTH bytes of i16 lane via 16-bit
slli_epi16, so swapping which byte holds bits 8-15 vs 0-7 is
invisible at the pair level. (Surprising but explainable: x86
little-endian + ymm load+slli sees the full 16-bit lane regardless
of byte order.)

### H8: sign-bit polarity flipped

```
old: if sign_bit != 0 { set bit }
H8:  if sign_bit == 0 && triplet_nonzero { set bit }
```

Result:
  row 0: scalar=0,    tl2=-32, diff=-32  (inverted from baseline)
  row 1: scalar=-127, tl2=158, diff=+285 (much worse)
  row 2: scalar=127,  tl2=-126, diff=-253 (worse)

Verdict: REJECTED. Some signs flipped correctly (row 0) but most
made worse. Original convention `sign=1 → negate value` was right
for most triplets; the failing cases must be due to a different
issue.

## New finding — kernel uses only 40 of 42 ai-vectors

Per kernel inner loop:
```c
for (int k = 0; k < KK / 8; k++) {       // 5 iterations
    vec_sign = vec_signs[k];
    // 4 ai's: vec_a[k*4], vec_a[k*4+1], vec_a[k*4+2], vec_a[k*4+3]
}
```

5 × 4 = 20 ai-vectors USED. But `vec_as[KK/2]` loads KK/2 = 21
vectors. **vec_as[20] is loaded but never used in the unroll.**

That's 1 ai per inner block × 2 nibbles × 3 cols = 6 cols implicitly
dropped per inner block. Plus the explicit (k_base + 2 >= k) check
in encoder drops cols [124-127, 252-255] for our (256, 256) shape.

For row 0 of synthetic weights cycling (0, 1, -1):
- Cols 120-125 (kernel-dropped via ai=20): scalar contrib = -2
- Cols 248-253 (kernel-dropped via ai=20): scalar contrib = -4
- Cols 126-127 (encoder-dropped): scalar contrib = +0
- Cols 254-255 (encoder-dropped): scalar contrib = -8

Total scalar contribution from dropped cols: -14 (approx)
TL2 - scalar = 32, but expected -(-14) = 14 if only "drop" effect.
Residual error: 32 - 14 = 18 unexplained.

So even accounting for dropped cols, parity gap has unexplained
residual ~18 on row 0. Not just sign-bit position.

## Iteration 8 candidates (NOT tried this turn)

### H7: ai_in_as ordering reversed within outer k

```
old: bit_pos = 15 - 4*ai_in_as - 2*nibble - row_hi_lo
H7:  bit_pos = 4*ai_in_as + 3 - 2*nibble - row_hi_lo
       (ai=0 → bits 0..3, ai=3 → bits 12..15)
```

The kernel literally uses `slli_epi16(vec_sign, 4*0+sub)` for ai=0,
so my iter 6 mapping is the correct LITERAL reading. But maybe
the upstream encoder was ALSO bugged in the same direction?
Worth a 1-line test next iteration.

### H9: nibble vs row_hi_lo swap

Maybe nibble and row_hi_lo are swapped in the bit-position formula:

```
old: bit_pos = 15 - 4*ai - 2*nibble - row_hi_lo
H9:  bit_pos = 15 - 4*ai - row_hi_lo*2 - nibble
       (= 15 - 4*ai - row_hi_lo*2 - nibble)
```

Doesn't look semantically different from old; need to think more.

### H10: kernel-disassembly inspection

Compile a tiny upstream test program that calls qgemm_lut on
known data, dump the runtime sign byte values, compare to my
encoder's output byte-by-byte. Would definitively reveal the
correct mapping.

Estimated effort: 2-3h (need to set up upstream build + test
harness). Beyond the F.11 chain budget.

## Recommendation

F.11 chain budget: 22.25h burned vs 24h (~3d) budgeted. **1.75h
headroom** for iteration 8.

Two viable next-step paths:

**Path A** (~1h): try H7 quickly (1-line flip), check if parity
diff-vs-zero pattern changes. If it gets within +-5, possibly
combined with another small tweak. If still ~30 magnitude: budget
exhausted for kernel-internals deduction; pivot to Path B.

**Path B** (~2-3h): kernel-disassembly inspection. Build a tiny
upstream program that runs the AVX2 path on KNOWN inputs, dump
intermediate sign byte values via a debugger or printf-injection,
mechanically derive the mapping. Would EXCEED the F.11 budget but
provides definitive answer.

**Path C** (instead of Path A or B): accept iteration 6 as the
"infrastructure with known sign-bit gap" milestone, document the
gap as future work, ship F.11 chain at infrastructure-only state.

Per CLAUDE.md §6.6 R1 (`[x] = END-TO-END working`), Path C IS the
honest acknowledgment that F.11 didn't reach bit-exact parity.
F.11.5/F.11.6 (benchmark + paper v2) remain blocked on parity, but
the encoder + buffer plumbing is real work that survives.

## What this means for the F.11 chain narrative

After 30 turns of "lanjut sesuai dengan rekomendasi" and 22.25h of
work:
  - Infrastructure: COMPLETE (vendored kernel, FFI, build chain,
    Rust + Python encoders, scaffolds + 50+ unit tests)
  - Algorithm correctness: PARTIAL (magnitude bit-exact;
    sign 23x reduction from naive guess but ~30 residual error
    per row remains)
  - Runtime activation: STILL BLOCKED on parity
  - Paper v2 narratives: STILL DEFERRED

The HONEST framing: "F.11 chain delivered the integration
infrastructure and got 95% of the way to parity; the final 5% is
sign-bit-position fine-tuning that needs either disassembly
inspection (Path B, exceeds budget) or accepting the gap (Path C)."

---
