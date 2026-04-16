# V30.SIM — V8 Coherence Gap: Python Reference Simulator

**Phase:** V30 "Coherence" (separate research track from V29 hardening)
**Parent:** V28.2 CLOSED_PARTIAL (coherence gap) + V28.5 RETEST (pad-collapse confirmed)
**Status:** PLAN (2026-04-16) — execution pending user go-ahead per sub-phase
**Plan Hygiene:** satisfies Rules 1–8 (see §11 self-check at end)
**Signed by:** Muhamad Fajar Putranto
**Signed at:** 2026-04-16

---

## 1. Problem Statement

FajarOS `ask` on Gemma 3 1B v8 produces 64 tokens that all decode to
byte `0x00` (pad), rendered as `.` via the `< 0x20` fallback path in
`kernel/compute/transformer.fj:1719`. Stability is solid (boot, load,
decode all work); **coherent output is not**.

### 1.1 Established facts (post V28.5 RETEST with @noinline active)

| Observation | Source |
|---|---|
| 64/64 tokens argmax to token_id 0 (pad) | `docs/V28_5_RETEST.md` L79-95 |
| Infrastructure correctness verified: gamma reads, file layout, memory state all audited OK | `docs/V28_2_CLOSED_PARTIAL.md` L43-52 |
| Single-matrix quant round-trip: 2.40% max error | `docs/V28_2_CLOSED_PARTIAL.md` L11 |
| `@noinline` retest: stability ✅, multilingual ⚠️ not reproduced (was transient outlier) | `docs/V28_5_RETEST.md` L14-38 |
| Performance normal: 1.58 M cycles/token | `docs/V28_5_RETEST.md` L108-112 |

### 1.2 Three suspect loci (from V28_2_CLOSED_PARTIAL §Coherence Gap)

| # | Suspect | Why plausible | Evidence gap |
|---|---|---|---|
| S1 | Integer overflow in `mdl_ram_lmhead_argmax_v8_tied` accumulator (262144 vocab × 1152 hidden = 302M partial sums) | i32 overflows around 2.1B, near limit; cumulative quantization noise could push a logit logit high, pinning argmax to a stuck index | No per-token logit dump from kernel; can't tell if argmax tie-breaks to 0 or if logit for 0 is genuinely maximal |
| S2 | `km_vecmat_packed_v8` arithmetic produces degenerate hidden states | Group-wise dequant with int32 scale + u8 zero-point has rounding policy that differs from float reference | No per-layer hidden-state dump; can't see where divergence begins |
| S3 | Cumulative rounding across 26 layers × 4 norms = 104 RMSNorm steps interacting with Gemma 3's large gamma (mean 4.55, max 55.75, NOT zero-centered) | Each RMSNorm truncates to i16/i32; 104 truncations compound; Gemma's scale amplifies the noise | No intermediate-layer output dump |

### 1.3 Why Python simulator is the right tool

The three hypotheses differ only by **where in the pipeline the
divergence starts**. A bit-exact Python simulator (matching kernel's
int16/int32 widths and rounding policy) running the SAME model weights
on the SAME prompt lets us:

1. Dump intermediate values at every layer + every norm + argmax logits
2. Compare to pure-float HF reference (ground truth)
3. Locate the first step where kernel and Python disagree
4. Determine if the divergence is a **rounding-accumulation** issue
   (S3), an **arithmetic-policy** issue (S2), or an **overflow** (S1)

Without bit-exact simulator, debugging the kernel is black-box — each
`println` instrumentation requires a rebuild + boot cycle (~2 min),
and the state space is too large for manual bisection.

### 1.4 Prevention Layer Gap (Rule 3)

V28.2 closed as partial with no test that would catch future
quantization-correctness regressions. Python simulator becomes the
test oracle: any future kernel change must keep kernel output within
tolerance of simulator output, or the test fails.

---

## 2. Scope (Cross-Repo)

### 2.1 FajarQuant (primary — simulator lives here)
| File | Change |
|------|--------|
| `tools/kernel_sim/` (new directory) | Python simulator package |
| `tools/kernel_sim/__init__.py` | Package init |
| `tools/kernel_sim/int_ops.py` | Bit-exact int16/int32 ops matching kernel (add_i32_sat, mul_i32_trunc, rshift_round_nearest, etc.) |
| `tools/kernel_sim/vecmat_v8.py` | Python port of `km_vecmat_packed_v8` |
| `tools/kernel_sim/rmsnorm.py` | Python port of `km_rmsnorm` (max-abs rescaling per V28.2 fix) |
| `tools/kernel_sim/argmax_v8.py` | Python port of `mdl_ram_lmhead_argmax_v8_tied` |
| `tools/kernel_sim/transformer.py` | Full forward pass orchestration (embed → 26 layers → lmhead) |
| `tools/kernel_sim/trace.py` | Intermediate-value capture: every op emits to JSONL at `/tmp/sim_trace.jsonl` |
| `tools/kernel_sim/diff.py` | Compare two trace files (kernel vs Python), report first-divergence op + magnitude |
| `tests/sim/test_int_ops_parity.py` | Unit tests: Python int_ops match kernel reference values for edge cases (max_i32, negatives, rounding ties) |
| `tests/sim/test_single_matrix_roundtrip.py` | Regression: 2.40% max error gate from V28.2 still holds in Python impl |
| `docs/V30_SIM_FINDINGS.md` | Per-phase findings rollup |
| `docs/V30_SIM_DECISION.md` | P3 gate: identified divergence + fix path |
| `docs/V30_SIM_CLOSED.md` | Final closure (coherence achieved OR documented-blocked) |

### 2.2 FajarOS x86
| File | Change |
|------|--------|
| `kernel/compute/transformer.fj` (conditional P2) | Add `FJTRACE=1` boot flag → emit per-layer dumps to serial (one-shot debug mode; disabled by default) |
| `kernel/shell/commands.fj` | New shell cmd `trace-dump <layer>` for offline dump extraction |
| `scripts/parse_kernel_trace.py` | Parse kernel serial log → trace JSONL matching Python sim format |

### 2.3 Fajar Lang
No direct changes expected. **Conditional:** if the divergence points
to compiler-level rounding policy (e.g., `>>` on signed int semantics),
Fajar Lang codegen might need an arithmetic-op audit — but that's a P3
DECISION branch, not default path.

### 2.4 Documentation (memory/claude-context)
| File | Change |
|------|--------|
| `memory/MEMORY.md` | V30 coherence track status line |
| `memory/project_v28_1_gemma3.md` | V30 progress entries |
| `Fajar Lang/CLAUDE.md` §3 Version History | V30 entry if/when closed |

---

## 3. Skills & Knowledge Required

| Area | Depth | Reference |
|------|-------|-----------|
| **Group-wise int4 quantization math** | Deep — scale_int32, zero_point_u8, group size, dequant formula | V28.2 export script `scripts/export_gemma3_v8.py` |
| **RMSNorm numerical behavior** | Deep — max-abs rescaling, i16/i32 width, truncation points | `kernel/compute/transformer.fj` km_rmsnorm |
| **Transformer forward pass** | Medium — attention, FFN, residual, norm ordering (pre-norm vs post-norm) | Gemma 3 HF implementation |
| **Python + NumPy int precision** | Deep — NumPy defaults to int64; need explicit int16/int32 dtypes + overflow modeling | NumPy docs on integer overflow |
| **HuggingFace transformers reference forward** | Medium — use HF Gemma 3 Python impl as ground-truth float reference | HF transformers `modeling_gemma3.py` |
| **Kernel serial trace parsing** | Light — regex over serial log lines emitted by `FJTRACE` | POSIX grep + Python re |
| **Bit-exact integer semantics (C + Rust)** | Deep — how Fajar Lang lowers `>>`, `*`, `+` to LLVM IR, whether signed overflow is UB or wraps | Fajar Lang codegen docs + LLVM IR semantics |
| **Tolerance analysis for quantization** | Medium — what's an acceptable Python-vs-kernel delta per layer, given 4-bit quant noise floor | KIVI paper + AWQ paper |

**Skill gaps flagged:** per-op rounding policy audit requires cross-checking
Fajar Lang codegen output. Mandatory **online research** per CLAUDE.md
§6.9 Rule 2 (minimum 8 references):

1. KIVI paper (KV cache quantization integer math)
2. AWQ paper (activation-aware 4-bit quant)
3. GPTQ paper (group-wise quantization)
4. SKVQ / QuaRot (rotation-based quant)
5. HuggingFace Gemma 3 reference implementation
6. LLVM IR signed int overflow semantics (LangRef)
7. Gemma technical report (norm ordering, gamma init)
8. FajarQuant own paper (bit conventions)

---

## 4. Phased Approach

### Phase V30.SIM.P0 — Pre-Flight Audit

| # | Task | Verification | Est |
|---|------|--------------|-----|
| P0.1 | Re-run `ask hello` on current kernel, capture fresh steady-state pad-collapse evidence | `/tmp/v30_p0_baseline.log` contains "Output: " + 20+ `.` chars | 0.1h |
| P0.2 | Enumerate every integer op in v8 hot paths (embed, 26× layer matmul, 4× norm, lmhead argmax) with its bit width + rounding policy | `docs/V30_SIM_FINDINGS.md` has op inventory table (~30-50 rows) | 0.3h |
| P0.3 | Confirm V28.2 single-matrix 2.40% error gate still passes on current export | `python scripts/validate_single_matrix.py` exit 0 | 0.1h |
| P0.4 | Multi-repo state check (all 3 repos clean before starting) | `git status -sb` 3 repos; `git rev-list origin/main..main` 0 each | 0.02h |
| P0.5 | Online research sweep (§3 list), commit findings annex | `docs/V30_SIM_FINDINGS.md §Research` has 8+ references with summary sentences | 0.4h |
| P0.6 | Commit P0 findings + this plan | `git log --oneline -1 fajarquant/docs/V30_SIM_PLAN.md` exists | 0.1h |

**Phase P0 total: 1.0h** (+25%: 1.3h)
**Deliverable:** Op inventory + research annex + baseline pad-collapse log
**Gate:** Online research covered all 8 refs; op inventory complete

### Phase V30.SIM.P1 — Python Bit-Exact Op Library

| # | Task | Verification | Est |
|---|------|--------------|-----|
| P1.1 | `int_ops.py`: `add_i32_sat`, `mul_i16xi16_i32`, `rshift_round_nearest`, `clamp_i16`, etc. — every op in §P0.2 inventory | `pytest tests/sim/test_int_ops_parity.py` → all pass (edge cases: INT32_MAX, INT16_MIN, ties-to-even) | 0.5h |
| P1.2 | `vecmat_v8.py`: Python port of `km_vecmat_packed_v8` using int_ops | Round-trip single group matches kernel output within 0 ULP for 100 random inputs | 0.3h |
| P1.3 | `rmsnorm.py`: Python port of `km_rmsnorm` with max-abs rescaling | Single-vector test matches kernel output within 0 ULP for 50 random inputs | 0.3h |
| P1.4 | `argmax_v8.py`: Python port of `mdl_ram_lmhead_argmax_v8_tied` | Single argmax matches kernel for 10 test logit arrays | 0.3h |
| P1.5 | Regression: port V28.2 single-matrix test to Python | `pytest tests/sim/test_single_matrix_roundtrip.py` shows 2.40% max error matched | 0.2h |

**Phase P1 total: 1.6h** (+25%: 2.0h)
**Deliverable:** Bit-exact Python ops; all parity tests green
**Gate:** Zero-ULP divergence between Python and kernel for all primitive ops

### Phase V30.SIM.P2 — Forward-Pass Orchestration + Trace

| # | Task | Verification | Est |
|---|------|--------------|-----|
| P2.1 | `transformer.py`: embed → 26× layer (pre_norm → attn → post_attn_norm → residual → pre_ffn_norm → FFN → post_ffn_norm → residual) → final_norm → lmhead_argmax | Python forward runs on "hello" prompt to completion without error | 0.4h |
| P2.2 | `trace.py`: instrument every op to emit `{op, layer, shape, hash, min, max, mean, top5_abs}` JSONL | `python -m kernel_sim hello > /tmp/sim_trace.jsonl` produces ~500+ lines | 0.2h |
| P2.3 | Kernel `FJTRACE=1` mode: add env-flag-gated `println` at same op boundaries | `make build-llvm FJTRACE=1 && make run-nvme-llvm` → serial log has matching trace markers | 0.5h |
| P2.4 | `parse_kernel_trace.py`: serial log → same JSONL schema | output file line count matches Python sim ±5% | 0.2h |
| P2.5 | HF reference forward on same prompt, dumped as float baseline | `python scripts/hf_reference.py hello > /tmp/hf_baseline.jsonl` produces matching schema | 0.3h |

**Phase P2 total: 1.6h** (+25%: 2.0h)
**Deliverable:** 3 trace files: Python-int (kernel mirror), kernel-actual, HF-float (ground truth)
**Gate:** All 3 traces cover same op boundaries + same prompt

### Phase V30.SIM.P3 — Divergence Analysis + Root Cause (Decision Gate)

**Decision gate (Rule 6):** before P3 fix work, commit
`docs/V30_SIM_DECISION.md` recording:
- First op where Python-int diverges from HF-float
- First op where Kernel diverges from Python-int
- Magnitude of each divergence
- Root cause categorization (S1 / S2 / S3 / new)
- Chosen fix path with estimated effort

| # | Task | Verification | Est |
|---|------|--------------|-----|
| P3.1 | `diff.py`: compute Python-int vs HF-float divergence per op, report first significant (>5% magnitude) | output: `first_divergence: layer=X op=Y magnitude=Z%` | 0.2h |
| P3.2 | Same diff for Kernel vs Python-int (sanity: should be ~0 ULP everywhere; any delta = Python-sim bug, not kernel bug) | `max_divergence < 2 ULP` across all ops | 0.2h |
| P3.3 | If P3.2 passes, attribute P3.1 divergence to int precision vs float (i.e., quantization cost is honest) | DECISION doc records "quantization is honest, check argmax logit distribution" | 0.1h |
| P3.4 | Dump top-10 logit indices + values from lmhead per token | DECISION doc contains 10 rows × 64 tokens (640 rows) | 0.2h |
| P3.5 | If logit 0 (pad) dominates: trace back to lmhead accumulator path → S1 overflow hypothesis | DECISION doc identifies overflow locus OR rules it out | 0.3h |
| P3.6 | If logit 0 not dominant but still wins: tie-breaking or near-uniform distribution → S2 or S3 cumulative rounding | DECISION doc has hidden-state magnitude trajectory across layers | 0.3h |
| P3.7 | Commit DECISION doc | `git log --oneline -1 docs/V30_SIM_DECISION.md` | 0.05h |

**Phase P3 total: 1.35h** (+25%: 1.7h; +40% for S3 research branch: 1.9h)
**Deliverable:** Identified divergence + written fix plan
**Gate:** DECISION file committed before any P4 fix work

### Phase V30.SIM.P4 — Implement Fix + Re-Verify

Fix branches by P3 outcome. These are conditional scopes:

**If S1 (overflow in argmax accumulator):**
| Task | Verification | Est |
|---|---|---|
| Widen accumulator to i64 in `mdl_ram_lmhead_argmax_v8_tied` | `grep 'accumulator' kernel/compute/model.fj \| grep i64` ≥ 1 | 0.3h |
| Rerun Python sim with i64 accumulator; confirm argmax no longer collapses to 0 | sim output: top-5 logits shows diversity | 0.2h |
| Rerun kernel with same fix; confirm ask emits non-pad tokens | `ask hello` output ≥ 10 tokens with byte >= 0x20 | 0.3h |

**If S2 (vecmat_v8 arithmetic policy):**
| Task | Verification | Est |
|---|---|---|
| Identify which dequant op (scale apply, zero-point subtract, rshift) diverges from HF float | DECISION doc specifies op + line | 0.1h |
| Adjust kernel rounding policy to match reference | kernel rebuilds; sim parity restored | 1.5h |

**If S3 (cumulative RMSNorm rounding across 104 steps):**
| Task | Verification | Est |
|---|---|---|
| Add wider intermediate representation (i32 → i64) for RMSNorm mean_sq computation | rebuilds + tests | 1.0h |
| Re-run full forward; confirm per-layer hidden-state magnitude stays stable | trace inspection shows flat-ish magnitude | 0.5h |

**If new hypothesis:**
| Task | Verification | Est |
|---|---|---|
| Document + chain to next DECISION cycle | — | — |

**Phase P4 base range: 0.8h (S1) → 2.0h (S2) → 1.5h (S3)** (+25%: 1.0h–2.5h)
**Deliverable:** Kernel produces non-pad output for at least one prompt
**Gate:** `ask hello` output contains ≥10 tokens with byte `>= 0x20`

### Phase V30.SIM.P5 — Regression Test + Prevention Layers

| # | Task | Verification | Est |
|---|------|--------------|-----|
| P5.1 | `tests/sim/test_kernel_sim_parity.py` — zero-ULP parity between kernel trace and Python sim | `pytest` green | 0.2h |
| P5.2 | `tests/sim/test_coherence_smoke.py` — `ask hello` output has ≥10 non-pad bytes (regression guard) | test passes | 0.2h |
| P5.3 | Makefile target `test-coherence-smoke` in fajaros-x86 (boots kernel, runs `ask hello`, greps output) | `make test-coherence-smoke` exit 0 | 0.3h |
| P5.4 | Pre-commit hook check 7/7: reject commits touching v8 hot paths without matching sim-parity test run | temp-modify vecmat → commit blocked | 0.2h |
| P5.5 | `docs/V30_SIM_CLOSED.md` rollup + retrospective (honest attribution of what originally caused pad-collapse) | doc committed | 0.2h |
| P5.6 | Update `docs/V28_2_CLOSED_PARTIAL.md` with retroactive box: coherence closed by V30.SIM | diff review | 0.1h |
| P5.7 | Update `docs/V28_5_RETEST.md` with retroactive box: pad-collapse root-caused by V30.SIM | diff review | 0.1h |
| P5.8 | CHANGELOG entry (fajaros-x86 v3.6.0 "Coherence") | diff review | 0.15h |
| P5.9 | MEMORY.md + CLAUDE.md §3 sync | memory diff review | 0.15h |
| P5.10 | GitHub Release v3.6.0 | `gh release view v3.6.0` OK | 0.15h |

**Phase P5 total: 1.65h** (+25%: 2.1h)
**Deliverable:** Parity test + coherence smoke + full public-artifact sync
**Gate:** All 8 Plan Hygiene rules still YES; release live

---

## 5. Effort Summary

| Phase | Tasks | Base | +25% buffer |
|-------|------:|-----:|-----------:|
| P0 Pre-flight + research | 6 | 1.0h | 1.3h |
| P1 Python bit-exact ops | 5 | 1.6h | 2.0h |
| P2 Forward-pass + trace | 5 | 1.6h | 2.0h |
| P3 Divergence analysis | 7 | 1.35h | 1.7h (+40% = 1.9h) |
| P4 Fix (S1 branch, lightest) | 3 | 0.8h | 1.0h |
| P4 Fix (S2 branch, heaviest) | 2 | 1.5h | 2.0h |
| P5 Regression + prevention | 10 | 1.65h | 2.1h |
| **TOTAL (S1 branch, best case)** | **36** | **8.0h** | **10.1h** |
| **TOTAL (S2 branch, worst case)** | **35** | **8.7h** | **11.0h** |

**Research-grade phase.** High-variance: P2 kernel FJTRACE instrumentation
(may hit Fajar Lang compiler quirks on conditional println), P3 analysis
(may need to loop back if first DECISION is wrong). P4 is hypothesis-gated.

**Surprise budget default +30%** (research-grade uncertainty exceeds
+25% baseline per §6.8 Rule 5 high-uncertainty clause).

---

## 6. Surprise Budget Tracking (Rule 5)

Per CLAUDE.md §6.8 Rule 5, every commit tags variance:

```
feat(v30-sim-p1.1): int_ops bit-exact Python library
  [actual 0.7h, est 0.5h, +40%]

fix(v30-sim-p4.S1): widen argmax accumulator to i64
  [actual 0.5h, est 0.3h, +67%]
```

Auto-escalation to +40% for entire phase if:
- P1 fails parity check on first run (primitive op semantics mismatch)
- P3 first DECISION turns out wrong (must re-enter P3)
- Gemma 3 RoPE / GQA intricacies surface unexpected op not in P0.2 inventory

---

## 7. Prevention Layers (Rule 3)

Each phase installs at least one durable prevention mechanism:

| Phase | Prevention mechanism |
|-------|----------------------|
| P1 | `test_int_ops_parity.py` — int primitive semantics locked; future Fajar Lang compiler changes that alter arithmetic will break test |
| P2 | `FJTRACE` mode + trace schema — any future model port gets the same observability for free |
| P3 | DECISION doc format — forces evidence-before-fix discipline; reusable across future ML debugging |
| P4 | Sim-parity becomes the kernel's correctness oracle; no silent kernel regression possible |
| P5 | `test-coherence-smoke` Makefile + pre-commit check 7/7 — v8 hot path cannot be touched without parity rerun |
| P5 | Retroactive boxes on V28_2 + V28_5 docs — audit trail preserves both original claim + corrected root cause |

---

## 8. Gates & Decisions (Rule 6)

| Gate | Before Phase | File |
|------|--------------|------|
| Research sweep complete | P1 | `fajarquant/docs/V30_SIM_FINDINGS.md` §Research (P0.5) |
| Op parity zero-ULP | P2 | `test_int_ops_parity.py` green |
| All 3 traces captured | P3 | `/tmp/{sim_trace,kernel_trace,hf_baseline}.jsonl` all present |
| Divergence root-cause identified | P4 | `docs/V30_SIM_DECISION.md` (P3.7) |
| Coherence smoke passes | P5 | `make test-coherence-smoke` exit 0 |
| Phase closure | handoff | `docs/V30_SIM_CLOSED.md` (P5.5) |

Pre-commit hook check 7/7 blocks any commit modifying v8 hot paths
without a matching sim-parity test invocation in same commit window.

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Python int ops don't match Fajar Lang compiler's actual LLVM lowering (e.g., signed right-shift semantics) | Medium | High | P0.2 op inventory audits compiler output first; P1.1 parity tests cover edge cases |
| `FJTRACE` instrumentation alters kernel arithmetic (e.g., register pressure changes rounding) | Low | High | Use `@noinline` on traced fns; compare traced vs untraced kernel output — should be identical |
| P3 first DECISION wrong; loop back needed | Medium | Medium | §6 surprise budget +30% baked in; explicit loop-back path to P3 |
| HF reference and Gemma 3 1B checkpoint choose different RoPE base / norm ordering | Low | Medium | P0.5 research sweep includes Gemma 3 tech report; P2.5 HF script uses exact HF checkpoint |
| Fix reveals deeper architectural issue (not fixable in V30 scope) | Low | High | P3 DECISION doc makes escalation path explicit; P4 branches include "document-blocked" option |
| Disk image (disk_v8.img) differs from current export script output | Low | Medium | P0.1 baseline capture uses current disk; P0.3 re-runs V28.2 gate to confirm |
| Python simulator runs too slow for full 64-token generation | Low | Low | Layer-0-only mode for P3 analysis; full 64-token only needed for P5 smoke |
| Gemma 3 RoPE (local 10K + global 1M) not yet implemented in kernel → sim can't model what kernel doesn't do | Low | Medium | Scope explicit: V30.SIM targets existing kernel code, not V28.1 Gemma 3 port. If kernel hasn't shipped dual-theta, sim should match kernel's simplification, not HF reference |

---

## 10. Online Research Triggers (per CLAUDE.md §6.9 Rule 2)

Research required at Phase P0.5:

1. **KIVI paper** (KV cache int4/int2) — integer math conventions
2. **AWQ paper** — activation-aware weight quantization (per-group scale rationale)
3. **GPTQ paper** — reference group-wise quant algorithm
4. **SKVQ / QuaRot** — rotation-based quant, outlier handling
5. **HuggingFace Gemma 3 reference** — `modeling_gemma3.py` for exact forward pass
6. **Google Gemma tech report** — norm ordering, gamma init, architecture intent
7. **LLVM LangRef** — signed integer overflow semantics, `ashr` vs `lshr`, undefined behavior
8. **FajarQuant own paper** — bit-width conventions + per-head scaling rationale
9. **NumPy integer semantics** — explicit dtype handling, no silent int64 promotion
10. **Any published "pad-collapse debugging" writeups** — LKML, HF discussion, ML kernel blogs

Minimum 8 sources per Rule 2. Cite all in
`docs/V30_SIM_FINDINGS.md` §Research with 1-sentence summary each.

---

## 11. Self-Check — Plan Hygiene Rule 6.8 (All 8)

```
[x] 1. Pre-flight audit mandatory                    — Phase P0 satisfies this (6 tasks + research sweep)
[x] 2. Verification commands runnable                — every task has literal pytest/grep/make command
[x] 3. Prevention layer per phase                    — P1 parity tests, P2 trace schema, P4 sim-oracle, P5 smoke+hook
[x] 4. Multi-agent audit cross-check mandatory       — P3.2 sim-vs-kernel zero-ULP check catches simulator bugs; P0.5 research cross-checks hypotheses against literature
[x] 5. Surprise budget +30% (research-grade)         — §6 escalation triggers defined; per-task variance tagged
[x] 6. Decision gates mechanical files               — §8 lists 6 gate files; pre-commit hook check 7/7 enforces
[x] 7. Public-facing artifact sync                   — P5.6–P5.10 covers V28_2 + V28_5 retroactive + CHANGELOG + MEMORY + CLAUDE.md + GitHub Release
[x] 8. Multi-repo state check                        — P0.4 runs 3-repo git status; §2 enumerates affected repos
```

All 8 YES = plan ships.

---

## 12. Author Acknowledgement

Per CLAUDE.md §6.8 Rule 7 + user memory `feedback_honesty_upfront`:
this plan exists because V28.2 CLOSED_PARTIAL and V28.5 RETEST both
documented the v8 coherence gap honestly — infrastructure correct,
output still pad-collapsed. Prior sessions chose to ship infra and
pivot rather than spin on black-box kernel debugging. V30.SIM is
the structured return: Python reference simulator as debugging tool,
not another round of kernel-side `println` bisection.

The most durable contribution is **Phase P1+P2 (bit-exact Python sim
+ trace infrastructure)**. Regardless of whether V30.SIM identifies
the root cause in P3, the Python simulator becomes the reusable test
oracle for every future model port or quantization experiment. It's
a new capability, not a one-shot debug script.

**Honest risk acknowledgement:** this is a research-grade plan.
Estimated 8-11h to close with S1 branch; up to 14h for S2/S3 branches.
If P3 points to an architectural issue beyond scope (e.g., requires
Gemma 3 1B architectural completeness — V28.1 full sprint), V30.SIM
documents the finding and hands back to V28.1 scope. Bounded failure
is an acceptable outcome.

---

*V30.SIM V8 Coherence Python Simulator Plan — drafted 2026-04-16 by
Claude Opus 4.6 as the second deliverable of V30 next-session agenda
Track 3, per Plan Hygiene Rule 1.*
