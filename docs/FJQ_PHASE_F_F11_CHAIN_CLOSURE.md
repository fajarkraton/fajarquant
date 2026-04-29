# Phase F F.11 Chain Closure (v1.0)

> **Status:** F.11 chain CLOSED at infrastructure-only state.
> Runtime dispatch + bit-exact parity DEFERRED to Branch X
> (encoder port from microsoft/BitNet llama.cpp fork) or Branch Z
> (pivot to F.13 dispatch heuristic).
> **Origin:** F.11.4(b) Branch Y execution surfaced a structural
> hole in the audit's plan — V31 weight format ≠ TL2 tile-organized
> wire format, so the kernel-path test isn't a useful safety net
> for "skip parity" since there's no path to invoke `qgemm_lut`
> without the encoder.

---

## 1. What F.11 chain DELIVERED

End-to-end infrastructure for AVX2 ternary-quant inference in
FajarOS Nova kernel context, blocked from RUNTIME activation by
one missing piece (the tile-permutation weight encoder).

### 1.1 Achievements (15 commits across 2 repos)

**fajarquant (commits 6987de3 → 5031e29):**

| Commit | Sub-task | Deliverable |
|---|---|---|
| `6987de3` | F.5.1.6 G5 | Forward-equivalence gate (separate from F.11) |
| `d9f92bb` | F.5.1.7 | Branch A pre-flight (separate; verdict NO-GO) |
| `b7349c4` | F.11 design v1.0 | 686-line design doc |
| `a098631` | F.11.0 | Python V31→TL2 algorithmic encoder (370 LOC + 23 tests) |
| `ea8b10a` | F.11.1 | TL2 vendor + cc crate + smoke binary |
| `fffb9b7` | F.11.2 | Rust safe shim (5 alignment + shape tests) |
| `2d2ab47` | F.11.3 | Phase D scalar baseline ported (250 LOC + 10 tests) |
| `41621a4` | F.11.3.5 | Mini-shape codegen (4 BitLinear shapes) |
| `73a8a45` | F.11.4(a) | Preprocessor FFI smoke (87 LOC, AVX2 path verified) |
| `c78d608` | F.11.4 search | Encoder NOT in upstream Python — finding |
| `839862d` | F.11.4(b) audit | Branch Y entry conditions all GREEN (or so we thought) |
| `98ed310` | F.11.4(b).1 | BITNET_OMIT_TRANSFORM gate — freestanding builds |
| `5031e29` | F.11.4(b).3 | Freestanding `memset` weak stub (gated to FREESTANDING-only after hosted-build infinite-recursion) |

**fajaros-x86 (commits ac68866 → 7d1de9a):**

| Commit | Sub-task | Deliverable |
|---|---|---|
| `ac68866` | F.11.4(b).3 | Makefile `$(TL2_O)` rule + ld step inclusion |
| `7d1de9a` | F.11.4(b).4 | `@ffi extern("C") fn fjq_tl2_qgemm_lut + preprocessor` decls |

### 1.2 What this gets you

- TL2 AVX2 kernel `fajarquant_bitnet_tl2.o` builds clean under
  `-ffreestanding -nostdlib`, links into `fajaros-llvm.elf`, three
  FFI symbols at known addresses:
  ```
  T fjq_tl2_preprocessor   @ 0x1066f0
  T fjq_tl2_qgemm_lut      @ 0x109010
  T fjq_tl2_self_test      @ 0x106700
  ```
- Fajar Lang `@ffi extern("C")` decls in `matmulfree.fj` —
  callable from `@kernel` context.
- Mini-shape codegen (256/1536/256/32768 dims) matches Phase D
  Mini ckpt's 4 unique BitLinear shapes.
- Phase D scalar baseline ported to portable Rust (`bitlinear_packed_scalar`)
  — works on every fajarquant target, not just x86_64.
- Apache-2.0 + MIT attribution properly tracked
  (`THIRD_PARTY_NOTICES.md`).

### 1.3 What this does NOT get you

- **No runtime activation.** No production code path calls
  `fjq_tl2_qgemm_lut` from a `@kernel` function. Dead-code-only.
- **No bit-exact parity.** `fjq_tl2_qgemm_lut` ↔
  `bitlinear_packed_scalar` equivalence remains an open question.
- **No tok/s benchmark.** F.11.5 deferred.
- **No paper v2 §6 hardware-perf table entries.** F.11.6 deferred.

---

## 2. The structural hole in Branch Y

The F.11.4(b) audit (`839862d`) classified Branch Y entry
conditions as "all GREEN" but missed a load-bearing prerequisite:

> "if Branch Y skips parity, the safety net is `make
> test-intllm-kernel-path` — that test catches output
> regressions empirically."

For this to work, `km_mf_bitlinear_packed_tl2` must be **invokable**
on real V31-format weights from the .fjm file. But:

  - `fjq_tl2_qgemm_lut`'s `a_addr` parameter expects **TL2
    tile-organized** weight bytes. Per F.11.4(a) audit
    (`c78d608`), this layout has strides like
    `((uint8_t*)A)[(k_outer * BBK / 3 / 2 * BM)]` — NOT
    row-major V31 2-bit-per-weight.
  - The .fjm v9 spec stores weights in V31 row-major format.
  - F.11.0's flat repacker doesn't emit tile-organized output
    either; it's the algorithmic encoder, not the wire encoder.

**Calling `fjq_tl2_qgemm_lut(v31_addr, ...)` is UB at the C level.**
It would dereference V31 bytes as TL2 magnitude indices, fetch
arbitrary bytes from the LUT region, and accumulate into C[i]
using random sign bits. The kernel-path test wouldn't necessarily
catch this — UB output could happen to satisfy the 4 invariants
(crash-freedom, output magnitude bounds, sentinel-token presence,
decode-loop stability) by luck.

The test isn't a safety net for UB-class bugs; it's a safety net
for INCORRECT-but-DEFINED-BEHAVIOR bugs. Branch Y's plan
conflated these.

---

## 3. Two paths forward

### 3.1 Branch X — port encoder from llama.cpp fork (~1-1.5d)

Per the encoder-search doc (`c78d608`) §3:

1. Locate microsoft/BitNet's llama.cpp fork (separate repo, likely
   pinned in their build's `3rdparty/`).
2. Find the C function that emits the TL2 wire format. Probable
   candidates: `quantize_row_tl2` or `ggml_compute_forward_tl2`
   in `ggml-quants.c` (1-2h to locate).
3. Read the tile-permutation logic (1-2h).
4. Port to Rust as `pack_tl2_tiles(weights, m, k, BM, BBK) ->
   (a_buf, sign_buf)` in `src/cpu_kernels/tl2.rs` (~4-6h).
5. Resume F.11.4(b).5-7 with a working encoder:
   - F.11.4(b).5 wrapper calls TL2 encoder at .fjm load time OR
     pre-encodes offline + ships TL2-format .fjm
   - F.11.4(b).6 dispatch + AVX2 enable check
   - F.11.4(b).7 kernel-path regression PASS = ship
6. Add `parity_real_mlp` test (~2h) — bit-exact validation.

**Total: 1-1.5d = ~8-12h.** Fits within F.11 chain's 11.25h
remaining budget.

**Verdict if this lands: F.11.5 + F.11.6 unblock.** Paper v2 gets
real tok/s + parity claims.

### 3.2 Branch Z — pivot to F.13 dispatch heuristic (~1d)

F.13 (CPU-vs-GPU dispatch heuristic in Nova kernel-path runtime) is
INDEPENDENT of TL2 correctness. It's a runtime decision about
which inference path to use — the existing scalar BitLinear
remains the CPU path, and F.13 just tunes the crossover with GPU.

- F.11 chain ships at the current "infrastructure-only" state.
- Future F.11 work resumes if/when:
  - User has bandwidth for Branch X
  - External audit requires TL2 perf data
  - Microsoft ships a tile encoder in their public Python

**Total: ~1d for F.13 prerequisites + initial heuristic.**

**Verdict: F.11 stays open as "future work" with infrastructure
landed.** F.13 gives a separate paper-v2 section.

---

## 4. Cumulative F.11 chain accounting

- Wall-clock burn: **12.75h** across 23 turns of "lanjut sesuai
  rekomendasi"
- Budget per F.11 design: **~3 days = 24h**
- Headroom remaining: **11.25h** — fits Branch X (~8-12h) cleanly.
  Insufficient for Branch X + F.11.5 + F.11.6 in same window if
  Branch X overruns.

Variance summary across F.11 sub-tasks:

| Sub-task | Estimate | Actual | Variance |
|---|---|---|---|
| F.11 design | 2h | 2.5h | +25% |
| F.11.0 repacker | 0.5d | 1.5h | −62% |
| F.11.1 vendor | 1d | 2.5h | −69% |
| F.11.2 shim | 0.5d | 1h | −75% |
| F.11.3 scalar | 1d | 1h | −88% (PARTIAL) |
| F.11.3.5 codegen | 2-4h | 1.5h | −50% |
| F.11.4(a) preproc | 1-2d | 1h | −94% (PARTIAL) |
| F.11.4 enc-search | 1-1.5d | 1h | BLOCKED |
| F.11.4(b) audit | 1h | 1h | 0% |
| F.11.4(b).1+.2 | 0.75h | 1h | +33% |
| F.11.4(b).3 | 1h | 30min | −50% |
| F.11.4(b).4 | 1h | 20min | −67% |
| **F.11 chain total** | **~3d (24h)** | **~12.75h** | **−47%** |

The aggregate underrun is real but masks the BLOCKED status of
the load-bearing sub-task (encoder). Variance accounting per
CLAUDE.md §6.8 R5 is honest individually but the chain as a whole
is "on time but not done."

---

## 5. Recommendation

**Branch X.** The F.11 chain has invested 12.75h of infrastructure
work that will be wasted if we abandon to Branch Z. Spending an
additional 8-12h on the encoder port unblocks the parity test +
runtime dispatch + paper v2 narratives in one tight increment.
Branch Z keeps the door open but ships zero F.11 user-facing
value.

Branch X risks:
- llama.cpp fork's encoder may itself be hand-written assembly
  (unlikely — TL2 is "type-two" lookup which sounds Python-friendly,
  but possible)
- Mini-shape kernels may have shape-specific tile-permutation
  rules not captured by a single generic encoder
- Bit-exact parity may surface a subtle bug in the codegen or
  vendoring chain that costs additional time

If any of those triggers, abort to Branch Z with the parity gap
documented + paper v2 cites the infrastructure-only landing.

---

## 6. What user should do next

The "lanjut sesuai dengan rekomendasi" loop has accumulated 23+
increments. This is a natural off-ramp: the next step is a
strategic call between X (commit to ~10h more) vs Z (ship F.13
with F.11 as future work).

**Default per resume protocol = Branch X**, since the audit
recommended Branch Y (now invalidated) and X is the algorithmic
sibling.

Other equally-defensible calls:
- **Pause + review.** 15 commits is a natural review point. User
  inspects the infrastructure, decides next direction.
- **Pivot to founder actions.** arXiv submission has been waiting
  since F.5.1.6 (~9 commits ago). Founder time, parallel-able
  with Branch X.

---

*v1.0 — F.11 chain CLOSED at infrastructure-only state
2026-04-29. Branch X (encoder port) recommended next; user
decision required.*

*Last updated: 2026-04-29 (V32-prep F.11 chain closure).*
