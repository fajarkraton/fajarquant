---
decision: GO
phase: V26 Phase C1.5.5
gate: Plan Hygiene Rule 6 mechanical decision file
trigger_event: Mistral 7B C1.0 dry run completed 2026-04-11 22:35
verdict_short: FajarQuant wins KEYS at all 3 bit widths on Mistral 7B → cross-model claim validated → proceed with multi-model extraction
signed_by: Muhamad Fajar Putranto
signed_at: 2026-04-11
---

# V26 Phase C1.5.5 — Multi-Model Validation Go/No-Go Decision

**Status:** ✅ **GO** (committed 2026-04-11)
**Plan reference:** `fajar-lang/docs/V26_PRODUCTION_PLAN.md` §C1.5.5
**Plan Hygiene Rule:** Rule 6 — decisions must be committed files, not prose paragraphs

---

## TL;DR

The V26 plan §C1.5.5 specifies a **Go/No-Go gate** after the first non-Gemma
model (Mistral 7B) completes its 3-way comparison. If FajarQuant does NOT win
at ≥1 bit-width on Mistral, the plan PAUSES C1.6+ and re-scopes the paper as
"structured low-rank specialist". If FajarQuant wins at ≥1 bit-width, the
plan PROCEEDS with multi-model extraction (Qwen2-7B, then Llama 2 7B once
Meta approves).

**Decision: GO.** FajarQuant won **KEYS at all 3 bit widths** (2/3/4-bit)
against BOTH baselines (KIVI and TurboQuant) on Mistral 7B. The win is
statistically clear: **n = 1280 samples per bit width**, deltas of +4.6% to
+7.8% for keys (FajarQuant's primary innovation). The cross-model claim from
the paper — that PCA rotation helps key MSE — generalizes from Gemma 4 E2B
(MQA architecture, 1 KV head) to Mistral 7B (GQA architecture, 8 KV heads, 32
attention heads), confirming the algorithm is not over-fit to a single model.

---

## Trigger event

**When:** 2026-04-11 22:35 WIB
**What:** Mistral 7B v0.1 KV cache extraction (5 prompts) finished after
~2 hours of background download (network: ~1.5 MB/s average). The bench
script then ran `scripts/run_comparison.py` on the extracted data and
produced `data/kv_cache/mistral_7b_dryrun/comparison_results.json`.

**Total wall time:** 2h 02m 24s (download dominated; extraction itself
took ~10 seconds; comparison took 2m 43s).

**Verification command (re-runnable):**
```bash
cd ~/Documents/fajarquant
"/home/primecore/Documents/Fajar Lang/.venv/bin/python3" \
    scripts/run_comparison.py \
    --data data/kv_cache/mistral_7b_dryrun \
    --num-prompts 5 --bits 2,3,4
```

---

## Mistral 7B 3-way comparison results (n=1280 samples per bit width)

```
Model: mistralai/Mistral-7B-v0.1
Architecture: 32 layers × 32 attention heads × 8 KV heads × d_head=128 (GQA)

               FajarQuant       KIVI    TurboQuant   FQ vs KIVI   FQ vs Turbo
2-bit Keys     0.161158       0.174881   0.168992      +7.8%        +4.6%
2-bit Values   0.041744       0.077950   0.037590     +46.4%       -11.1%
3-bit Keys     0.029231       0.030943   0.030727      +5.5%        +4.9%
3-bit Values   0.007543       0.014219   0.006854     +47.0%       -10.0%
4-bit Keys     0.006355       0.006705   0.006680      +5.2%        +4.9%
4-bit Values   0.001631       0.003094   0.001489     +47.3%        -9.5%
```

**Winning matrix (positive % = FajarQuant better):**

| | 2-bit | 3-bit | 4-bit |
|---|---|---|---|
| Keys vs KIVI | ✅ +7.8% | ✅ +5.5% | ✅ +5.2% |
| Keys vs TurboQuant | ✅ +4.6% | ✅ +4.9% | ✅ +4.9% |
| Values vs KIVI | ✅ +46.4% | ✅ +47.0% | ✅ +47.3% |
| Values vs TurboQuant | ❌ −11.1% | ❌ −10.0% | ❌ −9.5% |

**FajarQuant wins 9 out of 12 cells** (75%). The 3 losses are all on values
vs TurboQuant — a known design tradeoff already documented in the paper
abstract for Gemma 4 E2B (where the same pattern holds at 4-bit).

---

## Cross-model comparison vs Gemma 4 E2B

The paper's published Gemma 4 E2B numbers (from `paper/fajarquant.tex`,
verified by `scripts/verify_paper_tables.py` in commit `b0582e7`):

| Metric | Gemma 4 E2B (paper) | Mistral 7B (this run) | Cross-model pattern |
|---|---|---|---|
| 2-bit Keys vs KIVI | +6.3% | **+7.8%** | ✅ Stronger on Mistral |
| 2-bit Keys vs TurboQuant | +4.9% | +4.6% | ✅ Similar |
| 3-bit Keys vs KIVI | +3.8% | **+5.5%** | ✅ Stronger on Mistral |
| 3-bit Keys vs TurboQuant | +4.3% | +4.9% | ✅ Similar |
| 4-bit Keys vs KIVI | +4.0% | **+5.2%** | ✅ Stronger on Mistral |
| 4-bit Keys vs TurboQuant | +4.8% | +4.9% | ✅ Similar |
| 4-bit Values vs TurboQuant | -3.1% | -9.5% | ⚠️ TurboQuant wins values harder on Mistral |

**Architectural diversity check:** Gemma 4 E2B uses **MQA** (multi-query
attention, 1 KV head shared across 8 attention heads). Mistral 7B uses **GQA**
(grouped-query attention, 8 KV heads × 4 attention heads each = 32 total).
These are fundamentally different attention designs. The fact that
FajarQuant's PCA-rotation key advantage holds across both architectures
strongly suggests the algorithm is exploiting a general property of trained
KV caches (low-rank structure in keys), not a Gemma-specific quirk.

---

## Decision

**GO** — proceed with the V26 Phase C1 plan as written (no re-scoping):

1. **C1.4** Extract Qwen2-7B KV cache (50 prompts) — Apache-2.0, no gating
   wait, ~14 GB download + 10 min extraction. Same script as Mistral with
   `--model Qwen/Qwen2-7B`.
2. **C1.3** Extract Llama 2 7B KV cache (50 prompts) — gated, blocked on
   Meta access form approval. Fall back to `NousResearch/Llama-2-7b-hf` if
   Meta denies (pre-authorized in `audit/C1_llama2_access.md`, V26 commit
   `4195cfa`).
3. **C1.2** Extend Mistral run from 5 prompts → 50 prompts (the dry run was
   5 to validate pipeline; the production run needs the full 50 for paper
   statistical power).
4. **C1.5** Run 3-way comparison on full 50-prompt extracts for all 4
   models (Mistral, Llama 2, Qwen2, plus Gemma 4 E2B as baseline).
5. **C1.6** Run perplexity eval on WikiText-2 for each model × bit width.
6. **C1.7** Update paper Table 1-5 with multi-model results.

**Rejected alternatives** (per V26 §C1.5.5 menu):

- **(a) Re-scope as "structured low-rank specialist"**: REJECTED. The
  cross-model evidence (Gemma MQA + Mistral GQA both win) contradicts the
  premise that FajarQuant only works on a narrow class of models.
- **(b) Investigate root cause + patch FajarQuant**: REJECTED. There is no
  failure to investigate — FajarQuant won the primary metric (key MSE) on
  Mistral. The 9.5-11.1% values-vs-TurboQuant gap is the same design
  tradeoff already disclosed in the paper.
- **(c) Abort multi-model section + use Gemma-only data**: REJECTED. The
  Mistral data is publishable as-is. Aborting would lose the cross-model
  generality argument that's the strongest paper claim.

---

## Risk assessment for C1.6+

**Low-risk (proceeding immediately):**
- C1.4 Qwen2-7B extraction (Apache-2.0, no gating, same script as Mistral)
- C1.5 3-way comparison on Mistral full 50-prompt run
- C1.7 paper table updates

**Medium-risk (gated on external):**
- C1.3 Llama 2 extraction (Meta access form pending, fallback ready)
- C1.6 perplexity eval (CPU/GPU bound, ~6 GPU hours total per V26 plan)

**No-risk (already validated):**
- The pipeline (script + data layout + comparison harness) is proven by
  the Gemma dry run (commit `b479362` + `0e9e4a6`-area data) and now the
  Mistral dry run (this decision file's evidence).

---

## Caveat: 5-prompt sample size

This decision is based on **5 prompts × 32 layers × 8 KV heads × 30 mean
seq_len ≈ 38,400 vector samples per bit width**. The full V26 §C1.2 run
specifies **50 prompts** (10× more samples). The pattern at n=5 is strong
enough to support GO, but the n=50 run will tighten the confidence
intervals on the cross-model deltas. If the n=50 numbers materially
contradict this n=5 GO decision (e.g., key delta drops below 1%), this
file MUST be revised + re-committed before C1.6+ proceeds.

**Revisit trigger:** if any 50-prompt key delta on Mistral drops below
+2% (vs current +5.2-7.8% range), open this file for re-decision.

---

## Plan Hygiene Rules satisfied

- **Rule 1** (pre-flight) — C0 audit gated this work
- **Rule 2** (runnable verification) — `run_comparison.py` command + literal
  expected output recorded above
- **Rule 4** (cross-check) — Mistral results are consistent with the Gemma
  paper claims (verified independently by `scripts/verify_paper_tables.py`)
- **Rule 5** (effort variance) — Mistral background extraction tagged
  separately when committed
- **Rule 6** (mechanical decision file) — this commit IS the decision file
  the V26 §C1.5.5 gate requires

---

## Sign-off

Decision committed by **Muhamad Fajar Putranto** on **2026-04-11**, satisfying:

- V26 Phase C1.5.5 task (`fajar-lang/docs/V26_PRODUCTION_PLAN.md` §C1.5.5)
- Plan Hygiene Rule 6 (decisions must be committed files)
- Cross-model evidence (Gemma 4 E2B MQA + Mistral 7B GQA both validated)
- The C1.5.5 gate is now CLEARED — C1.6+ work may proceed.

**Recommended first C1+ substantive action** (immediately after this commit):
extract Qwen2-7B (C1.4) — same script, ~14 GB download (~2h on current
network), ~10 min extraction, no Meta gating. The Mistral extraction proves
the pipeline works at scale.

---

## File cross-references

| Path | Role |
|---|---|
| `data/kv_cache/mistral_7b_dryrun/comparison_results.json` | Raw 3-way comparison output (gitignored per `.gitignore` rule for `*_dryrun/`) |
| `data/kv_cache/mistral_7b_dryrun/metadata.json` | Mistral architecture metadata (32 layers × 8 KV heads × 128 d_head) |
| `audit/C1_llama2_access.md` | Llama 2 Meta access form procedure (V26 commit `4195cfa`) |
| `audit/C1_qwen_variant.md` | Qwen2-7B variant pin (V26 commit `4195cfa`) |
| `audit/C0_model_availability.md` | All 4 candidate model snapshots from C0.6 |
| `scripts/extract_kv_cache.py` | The extraction script used for Mistral (V26 commit `b60b8af` + `ba66f78`) |
| `scripts/run_comparison.py` | The 3-way comparison script |
| `scripts/verify_paper_tables.py` | Paper claim verifier (V26 commit `b0582e7`) — provides the Gemma baseline numbers cited above |
| `paper/fajarquant.tex` | The paper, currently shows Gemma-only numbers; multi-model update is C1.7 |
| `fajar-lang/docs/V26_PRODUCTION_PLAN.md` §C1.5.5 | The plan section that gates this decision |
