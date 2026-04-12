# V26 Phase C1.6 — Perplexity Evaluation Methodology Decision

**Date:** 2026-04-12
**Phase:** C1.6 (perplexity eval, multi-model)
**Author:** Fajar (via Claude Opus 4.6)
**Rule:** CLAUDE.md §6.8 Rule 6 — mechanical decision gate
**Plan reference:** `~/Documents/Fajar Lang/docs/V26_PRODUCTION_PLAN.md` row C1.6.0

---

## Amendment 2026-04-12 (later same day) — superseded by R-α.1

**Status:** option (b) below was implemented (commit `c9b2ff5`,
`scripts/eval_perplexity.py` symmetric prefix+target scoring) and
smoke-tested on Gemma 4 E2B at 256-ctx (5 samples) and 512-ctx (10
samples). The 512-ctx smoke test produced a structurally implausible
result: **all 3 quantized methods scored 5–6 PPL points BELOW FP16
baseline** (FQ 32.79, KIVI 34.22, TQ 33.39 vs FP16 38.68). Quantization
noise cannot improve a model — the result indicated a protocol bug.

**Literature investigation (P2 in the C1.6.1.6 task):** all three
reference KV-quantization repos use a fundamentally different protocol:

| Repo | Chunking | Quantization application |
|---|---|---|
| KVQuant (NeurIPS 2024) | Non-overlapping `model.seqlen` chunks | Model-internal patching during forward |
| SKVQ (COLM 2024) | Non-overlapping `input_len` chunks | `plug_quantizer_into_model()` modifies model in-place |
| KIVI (ICML 2024) | Delegated to lm-eval-harness | Custom `LlamaForCausalLM_KIVI` patches attention forward |

**Pattern:** all three patch attention layers to apply quantization
*during* the forward pass, before K/V are written to cache. The
prefix+target post-hoc cache mutation approach in commit `c9b2ff5`
does not match this canonical pattern, and produced the observed
anomaly because:

1. The mutation breaks the cache continuity in a way the model handles
   inconsistently between FP16 and quantized paths.
2. Even when mutation works, the protocol measures something different
   from "quantization during attention" — target tokens have fresh
   self-K/V mixed in, diluting the impact of the quantized prefix
   cache.

The post-hoc cache mutation approach is **not viable** for paper-quality
PPL evaluation comparable to KIVI/KVQuant/SKVQ.

### New decision: R-α.1 — literal per-architecture attention surgery

**Effective 2026-04-12 (this amendment):**

Implement model surgery via per-architecture quantized forward subclasses,
matching the canonical KIVI/KVQuant/SKVQ pattern. Quantization is
inserted **between RoPE application and `past_key_values.update()`**
in each attention layer's forward method, so the cache stores
already-quantized K/V and subsequent attention reads use the quantized
versions throughout the forward pass.

**Architectures patched (transformers 5.5.0):**

| Class | Source | Quirks handled |
|---|---|---|
| `MistralAttention` | `mistral/modeling_mistral.py:122` | Vanilla |
| `Qwen2Attention` | `qwen2/modeling_qwen2.py:187` | `bias=True`, instance `sliding_window` |
| `Gemma4TextAttention` | `gemma4/modeling_gemma4.py:1126` | K/V `RMSNorm`, `is_kv_shared_layer` (skip re-quant), `store_full_length_kv`, optional `v_proj=None`, `unsqueeze_dim=2` for RoPE |

**Implementation files:**

- `scripts/quant_attention.py` (NEW, ~480 LOC) — vectorized 4D quant
  primitives, three per-architecture quantized forwards, monkey-patch
  helpers (`patch_model_for_quantization` / `unpatch_model`), self-test.
- `scripts/eval_perplexity.py` (REWRITE) — drops `evaluate_perplexity`
  prefix+target version. New `evaluate_perplexity_canonical` does
  non-overlapping `--seq-len` chunks, fresh `DynamicCache` per chunk,
  full forward, standard shift-labels cross-entropy. Calls `patch_model
  _for_quantization` per (method, bits) and `unpatch_model` between.

**Plumbing:** module-level `_QUANT_METHOD` / `_QUANT_BITS` globals in
`quant_attention.py`. `patch_model_for_quantization(model, method, bits)`
walks `model.modules()`, identifies supported attention modules by class
name, saves their original `forward` (per-instance via `id(module)` →
`_ORIGINAL_FORWARDS` dict), and replaces with the matching
`{class}_quantized_forward` bound via `types.MethodType`. `unpatch_model`
restores from the dict and resets globals to `("none", 0)`.

**Step B verification (no GPU, completed):**

```
quant_attention self-test:
  FP32→FQ-2bit relerr: 0.4485
  FP32→FQ-4bit relerr: 0.0901
  PASS (all 4 quant fns return correct shape, finite values, dispatch works)

eval_perplexity import + CLI smoke:
  imports clean, exports evaluate_perplexity_canonical / load_wikitext2 / main
  CLI shows --model, --bits, --seq-len, --max-samples, --token, --output
```

Step C (GPU smoke test on Gemma 4 E2B, 10 samples, 512-ctx, 2-bit) is
the next checkpoint — pending user authorization per the C1.6 plan.

**Risk that the amendment failed to address:**

- (R1) `attention_interface(self, ...)` may have undeclared `self`
  attribute requirements. Monkey-patching preserves `self`, so
  unlikely.
- (R2) Gemma4 `is_kv_shared_layer` skip-quantization assumption: if
  the source layer's K/V were not yet quantized when stored to
  `shared_layers`, the share would propagate unquantized. The
  implementation places quantization BEFORE the `store_full_length_kv`
  block, so this should be correct. Verify in Step C.
- (R3) `seq_len=2048` may OOM with 7B models on RTX 4090 16GB.
  Fallback: `--seq-len 1024` or 512. Decided per smoke test.
- (R4) PCA degenerate at very short S: handled — `apply_fajarquant_4d`
  early-returns the input unchanged when `S < 2`. Eval seq_len ≥ 512
  is far above this guard.

The remainder of this file (below) records the original option (b)
decision for historical traceability. **The active decision for C1.6
is R-α.1, implemented in commit TBD.**

---

## Decision: (b)  *(SUPERSEDED — see amendment above)*

**Fix `scripts/eval_perplexity.py` to symmetric prefix+target scoring, then re-run Gemma 4 E2B alongside Mistral 7B + Qwen2-7B.**

---

## Background — what triggered this decision

While preparing C1.6 (multi-model perplexity eval) the existing
`data/kv_cache/perplexity_results.json` (Gemma 4 E2B baseline, committed
in Phase A as part of the FajarQuant extraction) was inspected and found
to contain a **methodology asymmetry** between the FP16 baseline and the
quantized methods:

```
fp16:             7650 tokens, 30 samples   ← full-sequence scoring
fajarquant_2bit:    30 tokens, 30 samples   ← last-token only
kivi_2bit:          30 tokens, 30 samples   ← last-token only
turboquant_2bit:    30 tokens, 30 samples   ← last-token only
... (same pattern for 3-bit and 4-bit)
```

Tracing `scripts/eval_perplexity.py:135-225` confirmed the cause:

1. Each sliding-window chunk is forwarded with `use_cache=True` →
   produces full-sequence logits AND a `past_key_values` cache.
2. **FP16 path:** scores all `chunk[:, 1:]` positions (≈511 per chunk;
   30 chunks ≈ 7650 tokens after stride/overlap).
3. **Quantized path:** quantizes the cache, then re-forwards
   `chunk[:, -1:]` (a single token) with the quantized cache, and scores
   only that one token. So 30 chunks × 1 token = 30 tokens.

**Why this is broken:**

- **Apples-to-oranges:** FP16 PPL is averaged over ~7650 tokens; quantized
  PPL is averaged over 30 tokens. The two numbers cannot be compared
  numerically without methodology disclosure.
- **Statistical noise:** 30-token averages on 7B-scale models have
  enormous variance. The current "FajarQuant 2-bit PPL = 80.1" is
  effectively a single noisy measurement.
- **Logit alignment bug (secondary):** the re-forward feeds
  `chunk[:, -1:]` after the cache already contains position N-1, which
  double-counts the last position. The output logits are predictions for
  position N (one beyond the chunk), but they get compared to
  `chunk[:, -1:]` (position N-1). The shift is mis-aligned by 1.
  This means the existing "PPL" numbers are not even what they claim to
  be — they are a function of the model's distribution over a misaligned
  target rather than honest perplexity.
- **Headline claim is on shaky ground:** the paper currently states
  "FajarQuant achieves the lowest perplexity at 2-3 bit" (lines 40, 219,
  579 of `paper/fajarquant.tex`) and Table `tab:ppl` lines 212-214 give
  9 PPL numbers. All 9 are computed from this broken protocol.

---

## Options considered

### Option (a) — keep last-token scoring asymmetry

- **Pros:** zero rework. C1.6 just runs the existing script on Mistral
  + Qwen2-7B with `--model` and adds rows to `tab:ppl`.
- **Cons:** propagates a known methodological flaw to 2 more models,
  multiplying the credibility risk by 3. Reviewers comparing
  `fp16: 7650 tokens` against `quantized: 30 tokens` in the released
  data will reject the paper outright. The asymmetry is documented in
  the JSON itself and `verify_paper_tables.py`, so it cannot be hidden.
  The logit shift bug also makes the existing "PPL" numbers technically
  incorrect regardless of asymmetry.
- **Risk:** paper rejection on methodology grounds at the venue's
  reviewer discretion. The MSE story (Table `tab:crossmodel`) survives,
  but the PPL story collapses.

### Option (b) — fix the script + re-run Gemma

- **Pros:** symmetric apples-to-apples evaluation. FP16 and quantized
  methods score the same set of tokens with the same protocol. Fixes
  the logit shift bug as a side effect. Cross-model PPL table becomes
  defensible.
- **Cons:** patch is non-trivial (~0.8h), Gemma must be re-run (~30 min
  GPU), and the Gemma quantized PPL numbers WILL change. The "FajarQuant
  wins at 2-3 bit" headline could flip — see "Acknowledged risks" below.
- **Risk:** if FajarQuant loses 2-bit or 3-bit under the corrected
  methodology, the abstract (line 40), Table `tab:ppl`, Table `tab:e2e`
  rows, §Perplexity Evaluation "Key finding" (line 219), and conclusion
  (line 579) all need revision. The MSE Cross-Model story (already
  committed) survives because it does not depend on PPL.

---

## Decision: (b)

**Rationale:**

1. **Correctness > usability** (CLAUDE.md §6.1) — propagating a known
   broken methodology to a public artifact violates the project's first
   principle.
2. **Plan Hygiene Rule 3** — every fix must spawn a prevention layer.
   Fixing the script IS the prevention layer; option (a) leaves a
   pre-existing trap for future evaluators.
3. **The MSE story is already strong.** The paper's load-bearing claim
   ("PCA rotation reduces key MSE 4-6% across 3 architectures") is
   verified by `Table tab:crossmodel` with $n=1{,}500 + 12{,}800 + 5{,}600
   ≈ 19{,}900$ samples per bit width. PPL is a complementary metric,
   not the primary. If the corrected PPL flips, the paper's main thesis
   still holds.
4. **Reviewer-resistance.** Symmetric scoring is what reviewers will
   replicate. Releasing asymmetric data and asking them to "trust the
   numbers" is a non-starter.
5. **Cost is bounded.** 0.8h patch + 0.5h Gemma re-run + 1h Mistral +
   1h Qwen2-7B + 1h paper updates ≈ 4.6h total. Within Phase C +30%
   surprise budget.

---

## Acknowledged risks

| Risk | Mitigation |
|---|---|
| Gemma quantized PPL flips, FajarQuant no longer wins 2-3 bit | Update paper §Perplexity "Key finding" to reflect new winners. MSE Cross-Model story still carries the abstract claim "PCA rotation reduces key MSE 4-6%". |
| FajarQuant becomes mid-pack on PPL across all 3 models | Re-frame paper: PCA rotation as a structured-low-rank specialist on KEY MSE (proven, $n≈19{,}900$). PPL becomes a "minor regression / no-harm" supporting table rather than headline. |
| One or more methods produce NaN/Inf PPL with the patched scoring | Investigate per-model. Most likely cause: numerical issue in PCA on a degenerate covariance for some heads. Fall back to per-head safety guard. |
| Gemma re-run takes 5x longer than estimated due to forward+score pass cost | Use smaller `--max-samples` for Gemma (e.g. 50 instead of 30, but with correct token budget). Document budget vs actual in commit message. |
| Paper PDF compilation breaks after table rewrites | `cd paper && make` after each table edit. Pre-commit hook in fajarquant repo runs verify_paper_tables.py on .tex changes. |

---

## Patch specification — what `eval_perplexity.py` becomes

### Old `evaluate_perplexity` flow (broken)

```
for each sliding window chunk:
    forward(chunk, use_cache=True) → past_kv, full_logits
    if quantize:
        apply_quant(past_kv)
        forward(chunk[:, -1:], past_key_values=past_kv) → last_token_logits
        score: cross_entropy(last_token_logits, chunk[:, -1:])  ← misaligned
    else:
        score: cross_entropy(full_logits[:, :-1], chunk[:, 1:])  ← clean
```

### New `evaluate_perplexity` flow (fixed)

```
prefix_len = max_length // 2   (e.g. 256 for max_length=512)

for each sliding window chunk:
    chunk = input_ids[:, begin:end]   (length up to max_length)
    if len(chunk) < prefix_len + 2: break
    prefix = chunk[:, :prefix_len]
    target = chunk[:, prefix_len:]    (length target_len, ≥ 2 tokens)

    forward(prefix, use_cache=True) → past_kv      (covers positions 0..prefix_len-1)

    if method != "none":
        for layer in past_kv.layers:
            layer.keys, layer.values = quantize(layer.keys, layer.values)

    forward(target, past_key_values=past_kv, use_cache=False) → target_logits
    # target_logits[t] = prediction for token at (prefix_len + t + 1)
    # given prefix cache (possibly quantized) + target tokens [0..t]

    shift_logits = target_logits[:, :-1, :]   # (1, target_len - 1, vocab)
    shift_labels = target[:, 1:]              # (1, target_len - 1)
    loss = cross_entropy(shift_logits, shift_labels, reduction='sum')

    nlls += loss
    n_tokens += target_len - 1
```

**Both FP16 and quantized methods use this protocol.** The only
difference is whether the `quantize(...)` block runs. This makes the
two paths symmetric and the resulting PPL numbers directly comparable.

**Token budget:** for `max_length=512`, `prefix_len=256`, `max_samples=30`,
each chunk scores `target_len - 1 = 255` tokens → 30 × 255 ≈ 7,650 tokens
per method per bit width. Matches the historical FP16 baseline count.

**Verification after re-run:**

```bash
jq '.fp16.tokens, .fajarquant_2bit.tokens' data/kv_cache/perplexity_results.json
# both should be ≥ 5000 (was: 7650 and 30)
```

If FP16 tokens ≠ quantized tokens (within ±1 due to numerical edge),
the patch is wrong and we revert.

---

## Files to update after re-run

1. `data/kv_cache/perplexity_results.json` — overwritten by Gemma re-run
2. `data/kv_cache/perplexity_mistral_7b.json` — new
3. `data/kv_cache/perplexity_qwen2_7b.json` — new
4. `paper/fajarquant.tex`:
   - Line 40 (abstract): PPL numbers in "PPL~80.1 at 2-bit vs TurboQuant~117.1, KIVI~231.9"
   - Lines 212-214 (`tab:ppl`): 9 numbers
   - Line 219 (Key finding): may need wording change
   - Lines 382-383 (`tab:e2e`): PPL rows
   - Line 579 (conclusion): "lowest perplexity at 2-3 bit" claim
   - **NEW:** add `tab:ppl_crossmodel` with 3 models × 3 bit widths × 3 methods
   - **NEW:** §Cross-Model Validation gets a "Perplexity" subsection
5. `scripts/verify_paper_tables.py`:
   - Update 4 existing PPL claims (lines 95-121) to new Gemma values
   - Add 6-12 new PPL claims for Mistral + Qwen2-7B
6. `paper/fajarquant.pdf` — regenerated by `cd paper && make`

---

## Sub-tasks (mirrors plan rows C1.6.0 - C1.6.6)

- C1.6.0 — this file (Rule 6 decision gate) ← in progress
- C1.6.1 — patch `scripts/eval_perplexity.py`
- C1.6.1.5 — smoke test patched script (5 samples on Gemma)
- C1.6.2 — full Gemma re-run, overwrite `perplexity_results.json`
- C1.6.3 — Mistral 7B PPL run
- C1.6.4 — Qwen2-7B PPL run
- C1.6.5 — paper updates + verify_paper_tables.py extension
- C1.6.6 — commit + push + multi-repo state check (Rule 8)

Each gets its own commit with `[actual Xh, est Yh, ±Z%]` tag (Rule 5).
