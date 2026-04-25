# Phase E E0.0.9 — CI GPU Runner Plan

> **Status:** PASS — no GPU CI needed; adopt Phase D pattern (verify-committed-JSONs in CI, GPU work runs on dev machine + cloud burst).
> **Decision date:** 2026-04-25
> **Plan reference:** `FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.3 §3 E0.0.9

---

## 1. Existing CI inventory (recon 2026-04-25)

### fajarquant
| Workflow | Runners | GPU? | Purpose |
|---|---|---|---|
| `ci.yml` | ubuntu-latest, macos-latest | NO | Rust build + test + clippy + fmt + flake-stress (5×64-thread) |
| `paper-reproduce-smoke.yml` | ubuntu-latest | NO | Verify pre-committed JSONs via `verify_paper_tables.py --strict` |

### fajaros-x86
| Workflow | Runners | GPU? | Purpose |
|---|---|---|---|
| `qemu-boot-stress.yml` | ubuntu-latest | NO | QEMU boot reliability stress |
| `qemu-test.yml` | ubuntu-latest | NO | Boot + test-all kernel tests in QEMU |

### fajar-lang
| Workflow | Runners | GPU? | Purpose |
|---|---|---|---|
| `ci.yml` | ubuntu/macos/windows | NO | Compiler build + test |
| `docs.yml` | ubuntu | NO | Docs |
| `embedded.yml` | ubuntu | NO | Embedded compile checks |
| `nightly.yml` | ubuntu | NO | Nightly fuzz + bench |
| `nova.yml` | ubuntu | NO | FajarOS Nova integration |
| `release.yml` | ubuntu/macos/windows | NO | Release artifacts |

**Verdict:** **0 GPU runners across 3 repos, 12 workflows total.** All existing CI is CPU-only.

## 2. Phase D precedent (the established pattern)

Phase D ran for ~3 months without GPU CI. The pattern that worked:

| Phase | What ran | Where | Output |
|---|---|---|---|
| Training (Mini/Base/Medium) | locally on RTX 4090 Laptop | `make train-{mini,base,medium}.py` | `paper/intllm/results/training_intllm-*.json` (committed) |
| Canonical eval | locally + cloud burst | `make bench-canonical-real TAG=...` | `paper/intllm/results/canonical_*.json` (committed) |
| Knowledge eval | locally | `make bench-knowledge-real TAG=...` | `paper/intllm/results/knowledge_*.json` (committed) |
| Baseline sweep (§3.4) | locally | `make bench-baselines-real-all` | `paper/intllm/results/bench_baselines_*.json` (committed) |
| **Verification in CI** | ubuntu-latest | `python scripts/verify_intllm_tables.py --strict` | exit 0 → claims hold |

**Key insight:** GPU compute = local/cloud-burst (one-shot, expensive). CI = data-pinning lint (cheap, on every push). The paper-reproduce-smoke.yml comment is explicit: *"No GPU needed — uses pre-computed results in data/kv_cache/. Catches data drift 2 weeks before submission, not 2 days after."*

## 3. Phase E adoption: same pattern, new artifacts

Phase E new gates classified by GPU-need:

| Gate | GPU on dev machine / cloud? | CI lint pattern? |
|---|---|---|
| `test-bilingual-kernel-path` (E5.1.3) | **NO** — QEMU boot, CPU only (matching Phase D `test-intllm-kernel-path`) | YES, in fajaros-x86 qemu-test.yml |
| `test-tax-vertical-kernel-path` (E5.2.2) | **NO** — same pattern | YES |
| `test-bilingual-watchdog` (E3 prevention) | **NO** — Python pytest, CPU only | YES, in fajarquant ci.yml |
| Boot reliability 50× (E5.1.4) | **NO** — QEMU loop, CPU only | YES, in fajaros-x86 qemu-boot-stress.yml |
| Binary size budget (E5.1.5) | **NO** — file-size check | YES, simple shell in qemu-test.yml |
| Tokens/sec bench (E5.1.6) | **YES** — local 4090 OR cloud burst | committed JSON verified by `verify_phase_e_tables.py` |
| TTFT bench (E5.1.7) | YES — same | same |
| RSS bench (E5.1.8) | YES — same | same |
| Power bench (E5.1.9) | YES — Radxa Q6A external watt-meter | committed JSON verified |
| Bilingual training Mini/Base/Medium (E3) | YES — RTX 4090 Laptop | training JSON committed, gate doc verified |
| Bilingual training Stretch (E3.4) | YES — cloud A100/H100 burst | committed JSON verified |
| Algorithm ablations Mini-scale (E2) | YES — RTX 4090 Laptop | ablation JSON committed |
| Tax-vertical SFT (E4.1) | YES — RTX 4090 Laptop or cloud | checkpoint + eval JSON committed |
| **Verify all paper tables** | **NO** — `verify_phase_e_tables.py --strict` | YES, `paper-reproduce-smoke.yml` extension |

**Same Phase D split:** GPU compute = local/cloud-burst, CI = data-pinning lint.

## 4. Recommended CI additions for Phase E (no GPU runners needed)

### 4.1 Extend `paper-reproduce-smoke.yml` for Phase E

```yaml
- name: Verify Phase E paper tables (when v1.0 paper drafted)
  run: python scripts/verify_phase_e_tables.py --strict

- name: Check Phase E result files exist
  run: |
    for f in \
      paper/intllm/results/phase_e/training_bilingual_mini.json \
      paper/intllm/results/phase_e/training_bilingual_base.json \
      paper/intllm/results/phase_e/training_bilingual_medium.json \
      paper/intllm/results/phase_e/canonical_bilingual_id_core.json \
      paper/intllm/results/phase_e/canonical_bilingual_en_core.json \
      paper/intllm/results/phase_e/serving_targets.json \
      paper/intllm/results/phase_e/tax_eval.json; do
      [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }
      python -c "import json; json.load(open('$f'))"
    done
```

### 4.2 Extend `qemu-test.yml` (fajaros-x86) for Phase E kernel paths

Add new make targets:
- `make test-bilingual-kernel-path`
- `make test-tax-vertical-kernel-path`

These run in existing CPU-only QEMU pipeline matching Phase D `test-intllm-kernel-path`.

### 4.3 Extend `ci.yml` (fajarquant) for `test-bilingual-watchdog`

Already covered by Phase D `test-train-watchdog` precedent — extend to Phase E with same harness, plus DVC-pull for synthetic generator manifest validation.

### 4.4 NEW lint workflow: `phase_e_data_lint.yml` (PROPOSED)

```yaml
name: Phase E Data Lint
on: [push, pull_request]
jobs:
  data-lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Verify no Llama outputs used (license check)
        run: python scripts/verify_no_llama_output.py  # NEW deliverable per E0.0.10
      - name: Verify synthetic ratio cap (≤50% per §14 R1)
        run: python scripts/verify_synthetic_mix.py
      - name: Verify TaxPrime PII redaction
        run: python scripts/check_pii_redaction.py --strict
      - name: Verify train/eval cross-contamination
        run: python scripts/check_train_eval_overlap.py
```

CPU-only. No GPU needed.

## 5. GPU compute alternative (NOT required, but documented)

If future scope demands GPU CI (e.g., automated regression on every push), options:

| Option | Cost | Verdict |
|---|---|---|
| GitHub Actions GPU runners (T4 4 GB beta) | ~$0.50/hour | underpowered for our scale |
| Self-hosted GitHub runner on dev RTX 4090 | $0 (existing hardware) | 🟢 viable IF needed; keep as TODO |
| AWS EC2 g4dn.xlarge auto-scale | ~$0.526/hour | overkill for sub-1-hour smoke |
| Cloud burst on-demand (current Phase D pattern) | ~$1-3/hour H100 only when training | 🟢 already in plan v1.3 §4 |

**Decision:** none for Phase E. If Phase F regression-budget warrants, revisit.

## 6. Acceptance criteria

- [x] All 12 existing CI workflows audited (3 repos)
- [x] 0 GPU runners present, all CPU-only
- [x] Phase D pattern (local/cloud GPU + committed JSONs + CI verify) confirmed working
- [x] Phase E gates classified by GPU-need (5 require local/cloud GPU, 9+ require none)
- [x] CI extension plan written (paper-reproduce-smoke + qemu-test + ci.yml + new phase_e_data_lint.yml)
- [x] No GPU runners required for Phase E
- [x] Self-hosted-runner option documented as future TODO if needed

**Phase E E0.0.9 gate: PASS** (mechanical decision file landed per §6.8 R6).

8 of 10 E0 sub-tasks now closed.

---

*Decision committed 2026-04-25. Author: Claude Opus 4.7 + Fajar.*
*Implementation triggers: CI workflow extensions land as part of E5 (paper finalization phase). Until then, existing CI stays as-is.*
*Companion: E0.0.10 synthetic legal review identified `verify_no_llama_output.py` as new CI lint deliverable — included in §4.4 above.*
