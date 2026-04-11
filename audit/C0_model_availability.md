# C0.6 — HF Model Availability Snapshot

**Audit date:** 2026-04-11
**Audit task:** V26 Phase C0.6 (`docs/V26_PRODUCTION_PLAN.md` §C0)
**Target use:** Phase C1 multi-model validation of FajarQuant
**Verification method:** WebFetch against `huggingface.co/<model>` for each candidate

## Summary Table

| Model | HF ID | Public? | License | Access friction | Disk (FP16/BF16) |
|---|---|---|---|---|---|
| Mistral 7B | `mistralai/Mistral-7B-v0.1` | ✅ Yes | Apache-2.0 | None | ~14 GB |
| Llama 2 7B | `meta-llama/Llama-2-7b-hf` | ❌ Login required | LLAMA 2 Community | **Meta gated, agreement required** | ~14 GB |
| Qwen 7B / 8B | `Qwen/Qwen-7B` | ✅ Yes | Tongyi Qianwen | Commercial use needs Alibaba form | ~16 GB (HF page says **8B params**, not 7B) |
| Phi-3 mini | `microsoft/Phi-3-mini-4k-instruct` | ✅ Yes | MIT | None | ~7.6 GB (3.8B params × 2 bytes) |

**Total disk budget:** ~52 GB (one-time download). Per-model VRAM at FP16 inference: ~14-16 GB for the 7B-class models, ~7.6 GB for Phi-3 mini. Fits the RTX 4090 Laptop's 16 GB VRAM **only one at a time**; sequential extraction is required.

## Per-Model Notes

### 1. Mistral 7B v0.1 — `mistralai/Mistral-7B-v0.1` ✅ EASY

- **Public:** Yes, no login required
- **License:** Apache-2.0 (`apache-2.0`)
- **Access agreement:** None
- **Size:** ~14 GB FP16 (7B params × 2 bytes); BF16 published
- **Status for C1:** **READY** — direct download via `huggingface_hub` or `transformers`
- **Risk:** None

### 2. Llama 2 7B — `meta-llama/Llama-2-7b-hf` ⚠️ GATED

- **Public:** **No** — requires HF login
- **License:** LLAMA 2 Community License Agreement (custom commercial-conditional)
- **Access agreement:** **Yes** — must share contact info with Meta and accept the agreement on the HF page before downloads work
- **Size:** ~14 GB FP16 (7B params × 2 bytes)
- **Status for C1:** **BLOCKED until Meta access is granted**
- **Action required before C1.0 dry run:** Author must visit https://huggingface.co/meta-llama/Llama-2-7b-hf logged in with HF account, click "Submit" on the access form, wait for Meta approval (typically a few hours to a day), then re-run a download check
- **Risk:** Approval can be denied or take days. **Backup plan:** swap Llama 2 for `NousResearch/Llama-2-7b-hf` (community redistribution, no gating) — verify before C1.5.5 go/no-go

### 3. Qwen 7B / 8B — `Qwen/Qwen-7B` ⚠️ NAMING DRIFT

- **Public:** Yes
- **License:** Tongyi Qianwen License Agreement
- **Access agreement:** Required only for **commercial use** (Alibaba form at https://dashscope.console.aliyun.com/openModelApply/qianwen). Research use does not require the form.
- **Size:** HF page reports **8B params** (not 7B as commonly cited and as listed in the V26 plan). Likely tokenizer-included or embedding-padded count. ~16 GB BF16
- **Status for C1:** **READY for research use** but the V26 plan should reference "Qwen 7B/8B" or pin a specific variant
- **Risk:** Naming inconsistency in the plan. **Action:** decide between `Qwen/Qwen-7B` (this), `Qwen/Qwen1.5-7B`, or `Qwen/Qwen2-7B` (newer) — and pin the choice in `docs/V26_PRODUCTION_PLAN.md` C1 task table before C1.0 dry run

### 4. Phi-3 mini — `microsoft/Phi-3-mini-4k-instruct` ✅ EASY

- **Public:** Yes
- **License:** MIT
- **Access agreement:** None
- **Size:** ~7.6 GB BF16 (3.8B params × 2 bytes)
- **Status for C1:** **READY**
- **Note:** This is a 3.8B model (smaller than the others), included as a control for the 7B-class group. Useful for budget runs.
- **Risk:** None

## C0.6 Findings → C1 Plan Deltas

1. **Llama 2 7B Meta gating is a real C1.0 blocker.** Author must request access **before** the C1.0 single-model dry run. This is not in the current C1 task table. **Add C1.0.0** = "Confirm Llama 2 access OR pin `NousResearch/Llama-2-7b-hf` as the variant".

2. **Qwen variant naming must be pinned.** The V26 plan currently says "Qwen 7B" but the canonical `Qwen/Qwen-7B` HF page reports 8B. Update the plan to either pin `Qwen/Qwen-7B` (and rename the row "Qwen 7B/8B") or switch to `Qwen2-7B` (which actually has 7B). **Decision needed before C1.0.**

3. **VRAM budget is sequential, not parallel.** RTX 4090 Laptop has 16 GB VRAM. Mistral and Llama 2 fit at FP16, Qwen 8B is borderline (likely needs 8-bit base loading), Phi-3 mini fits comfortably. Plan must process models **one at a time** with `del model; torch.cuda.empty_cache()` between runs. **Update C1.0 task to document this constraint.**

4. **Disk budget ~52 GB.** Verify free disk before C1.0:
   ```
   df -h ~/Documents/fajarquant/data/models 2>&1 || df -h ~/Documents/fajarquant
   ```
   Reserve ≥60 GB for safety (model weights + tokenizers + temp files).

## Verification Commands (for re-running this audit)

```bash
# Re-fetch model pages (each should still resolve)
curl -sIL https://huggingface.co/mistralai/Mistral-7B-v0.1 | grep -i 'http/'
curl -sIL https://huggingface.co/meta-llama/Llama-2-7b-hf | grep -i 'http/'
curl -sIL https://huggingface.co/Qwen/Qwen-7B | grep -i 'http/'
curl -sIL https://huggingface.co/microsoft/Phi-3-mini-4k-instruct | grep -i 'http/'

# After Meta gating is approved, this should succeed:
huggingface-cli download meta-llama/Llama-2-7b-hf --include "config.json" \
    --local-dir /tmp/llama2_test && rm -rf /tmp/llama2_test
```

## Sign-Off

C0.6 audit completed 2026-04-11 by Claude Code session continuation. All 4
candidate models confirmed to exist on HuggingFace. **Llama 2 gating is the
single hard blocker for C1.0** and must be cleared before the dry run.
**Qwen variant naming is the second outstanding decision.**
