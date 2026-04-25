# Phase E E0.0.3 — Tokenizer Decision

> **Status:** PASS — Mistral-v3 32K retained as primary tokenizer for Phase E bilingual ID+EN.
> **Decision date:** 2026-04-25
> **Empirical evidence:** `paper/intllm/results/phase_e/e0_tokenizer_compression.json`
> **Plan reference:** `FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.2 §3 E0.0.3 + §2 Q3
> **Companion script:** `python/phase_e/scripts/measure_tokenizer.py`

---

## Empirical measurement (2000 Wikipedia articles per language)

| Metric | Indonesian | English | ID/EN ratio |
|---|---|---|---|
| Bytes/token | **2.527** | 3.830 | 0.660× |
| Chars/token | 2.521 | 3.814 | 0.661× |
| Tokens/word | **2.948** | 1.665 | **1.770×** |
| Total tokens (1000 articles) | 4,063,491 | 6,052,942 | — |
| Total bytes (1000 articles) | 10.27 MB | 23.18 MB | 0.443× |

## Decision: PASS — Keep Mistral-v3 32K

**Bytes/token threshold (plan §2 Q3):** 2.527 ≤ 3.5 (efficient reference) → `keep_mistral_v3_32k` decision branch.

**Custom 24K Indonesian-optimized tokenizer NOT mandatory in E2.** Mistral-v3 32K efficient enough on the bytes-per-token axis.

## Nuance — `tokens/word` reveals the real cost

The plan threshold (`bytes/token > 4.0`) was simplistic. The deeper finding:

- Indonesian uses **2.948 tokens per word** vs English **1.665 tokens per word**
- = **+77% more tokens per word** for same semantic content
- Implication: training compute on Indonesian content costs ~1.77× more than equivalent English content per "thought"
- Why: Mistral-v3 32K was trained on a corpus dominated by English + European languages. Indonesian morphology (prefix/suffix-heavy: *me-* / *di-* / *ber-* / *-kan* / *-i* / *-an*) gets fragmented.

### Why bytes/token is misleading here

Indonesian Wikipedia text is largely ASCII (1 char ≈ 1 byte). English is too. So bytes/token ≈ chars/token for both. The IMBALANCE is in chars/token: ID 2.5 vs EN 3.8 — meaning each ID token covers ~34% fewer characters. Combined with longer Indonesian words (more compounding), this multiplies into the 1.77× tokens/word penalty.

### Implication for Phase E budget

If pretrain budget is 30 B tokens (Phase D Stretch scale):
- All-English: covers ~18 B "word-equivalents"
- All-Indonesian: covers ~10 B "word-equivalents"
- Bilingual 60ID/40EN: covers ~13 B "word-equivalents"

This means our bilingual model sees ~28% less semantic content per token than a pure-EN model would. **Compute budget should account for this** — either increase token budget or accept slightly less coverage. Memory cost at inference is the SAME (same vocab size), so this is a TRAINING-cost issue, not a deployment-cost issue.

## Alternative paths considered

### A. Custom 24K Indonesian-optimized tokenizer

Train SentencePiece BPE on Indonesian corpus only (24K vocab). Likely improves Indonesian tokens/word toward 1.6–1.8 range (parity with English on Mistral-v3).

**Pros:**
- Reduces Indonesian training compute by ~30%
- Better Indonesian token quality (less fragmentation of morphology)

**Cons:**
- Loses Mistral-v3 ecosystem comparability (BitNet/MMfreeLM Table 1 numbers use Mistral-v3 — direct comparison breaks)
- Phase D `intllm.tokenizer` API change required
- New v9.0 → v10.0 .fjm spec bump for kernel side
- Tokenizer quality validation (re-run § eval suite, etc.)
- Adds ~1 week E2 work

**Verdict:** DEFERRED to v2. Not blocking Phase E. Revisit in Phase F if Indonesian-only inference becomes priority OR if compute budget is tight.

### B. Mistral-v3 32K + Indonesian-vocab augmentation

Add ~2-4K Indonesian-specific tokens to existing 32K vocab (extend to 34-36K). Limited gain because attention/MLP weights would need re-init for new tokens (catastrophic during pretrain initialization).

**Verdict:** Not pursued. Effort >> benefit.

### C. Keep Mistral-v3 32K (CHOSEN)

**Pros:**
- Apples-to-apples with BitNet 2B4T, MMfreeLM, Phase D Mini/Base/Medium baselines
- Phase D `intllm.tokenizer` already in production
- No spec churn (.fjm v9 retained)
- Bytes/token already efficient at 2.527

**Cons:**
- Indonesian training compute 1.77× higher per word vs English (acceptable per §1 R2 cost analysis)
- Slight quality compromise on Indonesian morphology (mitigated by §14 Lapis 2 synthetic augmentation)

**Verdict:** ✅ ADOPTED for Phase E.

## Re-measurement plan

Re-run `measure_tokenizer.py` on:
- E1.6 final bilingual corpus (after dedup), not just Wikipedia. Different corpora → potentially different bytes/token. Acceptance: bytes/token < 4.0 still holds.
- After E2 if any algorithm catch-up affects tokenization (none expected)

## Output artifact

```json
{
  "decision": "keep_mistral_v3_32k",
  "tokenizer_name": "mistralai/Mistral-7B-v0.3",
  "tokenizer_vocab_size": 32768,
  "indonesian_bytes_per_token": 2.527,
  "english_bytes_per_token": 3.830,
  "indonesian_tokens_per_word": 2.948,
  "english_tokens_per_word": 1.665,
  "id_en_compute_ratio_per_word": 1.770
}
```

Full report: `paper/intllm/results/phase_e/e0_tokenizer_compression.json`

---

*Decision committed 2026-04-25. Author: Claude Opus 4.7 + Fajar.*
*Phase E plan §3 E0.0.3 gate: PASS (mechanical decision file landed).*
*Reproducibility: re-run via `cd ~/Documents/fajarquant && .venv/bin/python python/phase_e/scripts/measure_tokenizer.py --articles 2000`.*
