# C1.0.0 — Llama 2 7B Meta Access Request

**Audit task:** V26 Phase C1.0.0 (added in C0 audit, see `fajar-lang/docs/V26_C0_FINDINGS.md` §10)
**Created:** 2026-04-11
**Status:** **PENDING USER ACTION** — Claude cannot complete this step
**Blocks:** C1.3 (Llama 2 7B KV cache extraction). C1.0 (single-model dry run on Mistral) does NOT depend on this.

## Why This File Exists

The C0 pre-flight audit (Plan Hygiene Rule 1) discovered that
`meta-llama/Llama-2-7b-hf` is **gated** on HuggingFace — Meta requires
an access agreement before any download is permitted. This was not in
the original V26 plan §C1 task table; it was caught by C0.6's WebFetch
verification.

If C1 starts the multi-model extraction without prior approval, **C1.3
will fail** at the first `huggingface_hub.snapshot_download(...)` call
with a `401 Client Error`. That wastes ~5 minutes of attempted download
and the time to diagnose the failure.

This file is a Plan Hygiene Rule 6 mechanical reminder: it sits in the
audit/ directory until the access is granted, then gets deleted (with
a commit) once the access check passes.

## Action Required (30 seconds + waiting on Meta)

1. Open https://huggingface.co/meta-llama/Llama-2-7b-hf in a browser
2. **Log in** to HuggingFace (create an account if needed; free)
3. The page shows: *"You need to share contact information with Meta to access this model"*
4. Click **"Submit"** on the access form (provides your name + email + intended use)
5. **Wait** for Meta approval. Typical turnaround: a few hours to 1 day. You'll get an email.
6. Verify access works:
   ```bash
   pip install -U huggingface_hub
   huggingface-cli login   # paste your HF access token from https://huggingface.co/settings/tokens
   huggingface-cli download meta-llama/Llama-2-7b-hf --include "config.json" \
       --local-dir /tmp/llama2_test && echo "OK" && rm -rf /tmp/llama2_test
   ```
   If the command prints `OK` (no `401 Client Error`), access is granted and C1.3 can proceed.
7. Delete this file with a commit: `git rm audit/C1_llama2_access.md && git commit -m "fix(v26-c1.0.0): Llama 2 Meta access granted"`

## Fallback (if Meta denies or takes too long)

The V26 plan §6 risk register pre-authorizes a fallback: swap to
`NousResearch/Llama-2-7b-hf` (community redistribution, no gating).
The model weights are bit-identical to Meta's release. To activate the
fallback **without waiting for Meta**:

1. Edit `docs/V26_PRODUCTION_PLAN.md` §C1.3:
   - Replace `meta-llama/Llama-2-7b-hf` with `NousResearch/Llama-2-7b-hf`
2. Edit `audit/C0_model_availability.md`: add a note "Falling back to NousResearch redistribution per C1.0.0 fallback"
3. Commit: `git commit -m "fix(v26-c1.0.0): fall back to NousResearch Llama 2 — Meta gate skipped"`
4. Delete this file in the same commit

## Why Both Paths Are OK

The 3-way comparison (FajarQuant vs KIVI vs TurboQuant) only depends on
the **architecture** and **weight values**, not the licensing source.
NousResearch's redistribution is a verbatim copy. The paper's claim of
"validated on Llama 2 7B" is honest either way.

If the venue is strict about reproducibility provenance, cite the model
as `meta-llama/Llama-2-7b-hf (via NousResearch redistribution due to gating)`
in the supplementary materials (C3.4).

## Sign-Off

This file exists as a **Plan Hygiene Rule 6 mechanical reminder**.
It blocks C1.3 conceptually (the file is the gate), but does not
mechanically block git commits — there is no hook tied to its
existence. The commit-msg hook for V26 C3.2 is unaffected.

**Recommended next step for the human:** open the HF link above NOW
(while reading this) and submit the form. By the time C1.0 dry run
completes (~1 GPU hour), Meta has usually approved.
