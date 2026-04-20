# Upstream Pin — ridgerchu/matmulfreellm

Vendored snapshot of `ridgerchu/matmulfreellm` lives at
`python/phase_d/_upstream/`. `.gitignore` excludes the directory from
git; this file records the pinned commit so any developer can recreate
the exact snapshot.

## Current pin

| Attribute | Value |
|---|---|
| Upstream | `https://github.com/ridgerchu/matmulfreellm` |
| Commit SHA | `f24cfe5` |
| Commit title | "fix running bug on triton and transformers" |
| Date pinned | 2026-04-21 |
| License | Apache-2.0 |

## Recreate the snapshot

```bash
cd fajarquant/python/phase_d
git clone --depth 1 https://github.com/ridgerchu/matmulfreellm.git _upstream
cd _upstream
git fetch --depth 1 origin f24cfe5
git checkout f24cfe5
```

## Why vendored (not `pip install git+...`)

1. §6.8 Rule 3 — prevention. If upstream removes the repo or rewrites
   history, Phase D reproducibility breaks.
2. Phase D adds features (QAT harness, per-coord bit allocation,
   ternary-aware channel reordering) that need to reach into upstream
   internals. Importing as a dep would require monkeypatching or PR
   back upstream; vendoring lets us reach freely.
3. The paper will need to cite the exact commit; a vendored snapshot is
   the honest artifact.

## Upstream license compliance

Apache-2.0 requires:
- Retain `LICENSE` file — present at `_upstream/LICENSE` (unchanged)
- Attribute in documentation — done here + in `FJQ_PHASE_D_PAPER_OUTLINE.md` §2
- State changes made — Phase D is a superset; changes documented per
  commit in the `fajarquant` git log with `v31-c-p2.*` prefix

## Update cadence

Re-pin when upstream publishes:
- Training code (the Surprise #1 in `FJQ_PHASE_D_P2_FINDINGS.md`)
- A BitNet-b1.58 compatibility bridge
- Official v0.2 release

Otherwise stay frozen. Phase D stability > upstream currency.
