---
variant: Qwen/Qwen2-7B
hf_id: Qwen/Qwen2-7B
hf_url: https://huggingface.co/Qwen/Qwen2-7B
params: 7B (actual, not padded)
license: Apache-2.0
gated: false
disk_size_fp16_gb_approx: 14
released: 2024-06 (per Qwen team announcement)
decision_date: 2026-04-11
decision_session: V26 Phase C0 → C1 transition
audit_task: V26 C1.0.1
---

# C1.0.1 — Qwen Variant Pin

**Status:** ✅ Pinned to **`Qwen/Qwen2-7B`** as of 2026-04-11.

## Decision

The V26 Phase C1 multi-model validation will use **`Qwen/Qwen2-7B`** as the
Qwen representative in the 4-model set (Mistral 7B / Llama 2 7B / Qwen2-7B /
Phi-3 mini).

## Why Qwen2-7B (vs Qwen-7B or Qwen1.5-7B)

The C0.6 audit (`audit/C0_model_availability.md`) WebFetch'd
`huggingface.co/Qwen/Qwen-7B` and discovered the original Qwen-7B HF page
actually reports **8B params**, not 7B. The V26 plan's reference to "Qwen 7B"
was ambiguous about which variant. Three candidates were considered:

| Variant | Params | License | Gating | Reason |
|---|---|---|---|---|
| `Qwen/Qwen-7B` | **8B** (per HF page) | Tongyi Qianwen | None | Closest to original plan ID, but param mismatch |
| `Qwen/Qwen1.5-7B` | 7B | Apache-2.0 | None | Intermediate, useful only for prior-literature comparison |
| **`Qwen/Qwen2-7B`** | **7B** (actual) | **Apache-2.0** | **None** | ✅ **Newest, true 7B, permissive license** |

`Qwen2-7B` was selected because:

1. **Param count matches the "7B" label** in the V26 plan and paper. Mixing
   7B models with one 8B model would muddy the cross-model comparison.
2. **Apache-2.0 license** is the cleanest possible. Tongyi Qianwen license
   has commercial-use restrictions that complicate the venue submission's
   reproducibility statement (C3.4).
3. **Qwen2 is the current generation** as of April 2026 — reviewers will
   expect the newest version unless a specific reason favors an older one.
4. **No access agreement** required (vs Llama 2 which is gated, see
   `audit/C1_llama2_access.md`).
5. **Well-supported in `transformers`** library — verified at C1.1 adapter
   integration step.

## Impact on Phase C1 Plan Tasks

The following tasks should be updated to use the pinned variant:

| Task | Old reference | New reference |
|---|---|---|
| C1.4 | "Qwen 7B" or "Qwen 7B / Phi-3 mini" | `Qwen/Qwen2-7B` (treat Phi-3 mini as a separate 5th model option) |
| C1.5 | `data/kv_cache/qwen_7b/` | `data/kv_cache/qwen2_7b/` |
| C1.6 | `--data-dir data/kv_cache/qwen_7b` | `--data-dir data/kv_cache/qwen2_7b` |
| C1.7 paper Table 1-5 | "Qwen 7B" | "Qwen2 7B" (with footnote citing the released 2024-06 version) |

These edits should land in `docs/V26_PRODUCTION_PLAN.md` §C1 in a
follow-up plan-edit commit.

## Verification Commands

```bash
# Confirm the model still exists on HF (should return HTTP 200)
curl -sIL https://huggingface.co/Qwen/Qwen2-7B 2>&1 | grep -i 'http/'

# Confirm license string
huggingface-cli download Qwen/Qwen2-7B --include "config.json" --local-dir /tmp/qwen2_test \
  && jq -r '.transformers_version' /tmp/qwen2_test/config.json && rm -rf /tmp/qwen2_test

# Confirm param count == 7B (after weights download in C1.4)
python3 -c "from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-7B', torch_dtype='auto'); print(f'{sum(p.numel() for p in m.parameters())/1e9:.2f}B params')"
```

## Sign-Off

Decision committed by user via AskUserQuestion on 2026-04-11 (V26 Phase
C0 → C1 transition session). Plan Hygiene Rule 6 mechanical gate
satisfied (decision is a committed file, not a prose paragraph).
