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

## Action Required (browser steps + ~5 min config)

### Browser-only steps (irreducible — only the human can do these)

1. **Revoke any leaked tokens** at https://huggingface.co/settings/tokens. If a
   token was ever pasted into an untrusted environment (chat, public log, screen
   share, screenshot), revoke it. Tokens grant API access to your HF account.
2. **Create a new token** at https://huggingface.co/settings/tokens with **`Read`
   scope only**. Copy it once — you cannot view it again later.
3. **Log in** to HuggingFace if not already.
4. Open https://huggingface.co/meta-llama/Llama-2-7b-hf logged in.
5. The page shows: *"You need to share contact information with Meta to access
   this model"*. Click **"Submit"** on the access form.
6. **Wait** for Meta approval (typical: a few hours to 1 day; you'll get an email).

### Local config (one-time, ~30 seconds — token never touches the command line)

Pick **one** of the two patterns. Both keep the token out of shell history, `ps`
listings, and committed files.

```bash
# OPTION A — env var (cleanest for one-off CLI use)
echo 'export HF_TOKEN=hf_<your-new-token>' >> ~/.huggingface_env
chmod 600 ~/.huggingface_env
echo 'source ~/.huggingface_env' >> ~/.bashrc   # or ~/.zshrc
source ~/.huggingface_env
```

```bash
# OPTION B — huggingface-cli (recommended if you also use the HF Python API)
pip install -U huggingface_hub
huggingface-cli login   # paste token interactively, hidden from shell history
# token is cached at ~/.cache/huggingface/token (file mode 600)
```

**Both options work transparently with `extract_kv_cache.py` — no `--token` arg needed.** The script's token resolution order is:

1. `--token` CLI arg (DISCOURAGED — leaks via ps + shell history)
2. `$HF_TOKEN` env var (Option A)
3. `~/.cache/huggingface/token` cache (Option B, via huggingface_hub fallback)

### Verify access works

```bash
# Verification 1 — only checks login, not Meta gate
pip install -U huggingface_hub
huggingface-cli whoami   # should print your HF username

# Verification 2 — checks both login AND Meta gate (this is the real one)
huggingface-cli download meta-llama/Llama-2-7b-hf --include "config.json" \
    --local-dir /tmp/llama2_test && echo "ACCESS OK" && rm -rf /tmp/llama2_test
```

If verification 2 prints `ACCESS OK` (no `401 Client Error: Cannot access gated repo`), you're done. Delete this file with a commit:

```bash
cd ~/Documents/fajarquant
git rm audit/C1_llama2_access.md
git commit -m "fix(v26-c1.0.0): Llama 2 Meta access granted, cleanup gate"
```

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
