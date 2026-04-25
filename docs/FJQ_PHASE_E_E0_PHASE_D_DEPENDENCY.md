# Phase E E0.0.6 — Phase D §3.4 Baseline Sweep Dependency Resolution

> **Status:** PASS — partial-baseline acceptance path adopted per plan v1.1 §2 Q6.
> **Decision date:** 2026-04-25
> **Plan reference:** `FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.3 §2 Q6 + §3 E0.0.6
> **Predecessor doc:** `FJQ_PHASE_D_3_4_B0_FINDINGS.md` (scaffold landed `64fa0a9`)

---

## 1. Phase D §3.4 sweep status (2026-04-25)

Phase D `bench-baselines-real-all` was launched 2026-04-25 11:29 WIB to populate paper Table 2 with 7 real baseline measurements (BitNet, MMfreeLM-370M/1.3B/2.7B, SmolLM2-135M/360M, Pythia-160m). The sweep was killed twice in the same day due to network-induced CLOSE_WAIT hangs:

| # | Repo | Status | Eval elapsed | JSON state |
|---|---|---|---|---|
| 1 | microsoft/bitnet-b1.58-2B-4T | ✅ **REAL** | 5172 s (86 min) | `bench_baselines_microsoft__bitnet-b1_58-2B-4T.json` modified, **unstaged** |
| 2 | ridger/MMfreeLM-370M | ✅ **REAL** | 1395 s (23 min) | `bench_baselines_ridger__MMfreeLM-370M.json` modified, **unstaged** |
| 3 | ridger/MMfreeLM-1.3B | 🔴 hung twice (CLOSE_WAIT) | 0 (download stuck) | scaffold (status: scaffold, results: 0.0) |
| 4 | ridger/MMfreeLM-2.7B | ⏸ pending | — | scaffold |
| 5 | HuggingFaceTB/SmolLM2-135M | ⏸ pending | — | scaffold |
| 6 | HuggingFaceTB/SmolLM2-360M | ⏸ pending | — | scaffold |
| 7 | EleutherAI/pythia-160m | ⏸ pending | — | scaffold |

**Failure mode:** `bench_baselines.py` lacks the §6.11 retry+timeout hardening that training scripts have. When wifi blip causes server-side socket close, the urllib3 reads hang indefinitely on dead sockets. `HF_HUB_DOWNLOAD_TIMEOUT=60` env var doesn't catch socket-level reads on already-ESTABLISHED connections.

Memory: `feedback_hf_streaming_hang.md` documents the exact pattern (CLOSE_WAIT + 8 ESTAB zombies + 117 threads in futex_do_wait).

## 2. Decision: ACCEPT PARTIAL per plan v1.1 §2 Q6

Plan v1.1 §2 Q6 (preserved in v1.2/v1.3) decision rule:

> *"if Phase D §3.4 not closed within 1 week from this plan, accept partial baseline (BitNet ✅ + MMfreeLM-370M ✅) and defer 1.3B/2.7B/SmolLM2/pythia comparisons to Phase E paper appendix; do NOT block E1 corpus assembly on it."*

**Triggering early — same-day decision, not waiting full 1-week window:** the failure mode is structural (script-level hardening missing, not network luck). Re-launching without proper §6.11 hardening will hang again on next wifi event. Best to accept partial NOW + plan the proper fix as a separate concern.

## 3. What's preserved vs deferred

### ✅ Preserved for paper Table 2 (canonical real measurements)

| Baseline | Coverage |
|---|---|
| **microsoft/bitnet-b1.58-2B-4T** | All 11 lm-eval tasks (wikitext, lambada, hellaswag, piqa, winogrande, arc_e/c, openbookqa, mmlu, triviaqa, boolq) |
| **ridger/MMfreeLM-370M** | All 11 lm-eval tasks |

These give us **the 2 most strategically important baselines:**
- BitNet b1.58 2B4T is the **direct competitor** at deployment + ternary architecture
- MMfreeLM-370M is the **same architecture cousin** (matmul-free, mistral-v3 32k tokenizer) — cleanest apples-to-apples for IntLLM

### ⏸ Deferred to paper appendix or v1.1 sweep

| Baseline | Disposition |
|---|---|
| ridger/MMfreeLM-1.3B | deferred — paper Table 2 row marked "deferred (sweep hardening required)" |
| ridger/MMfreeLM-2.7B | deferred |
| HuggingFaceTB/SmolLM2-135M | deferred |
| HuggingFaceTB/SmolLM2-360M | deferred |
| EleutherAI/pythia-160m | deferred (fast eval candidate for re-run; cached locally) |

## 4. What this UNBLOCKS (Phase E forward progress)

- ✅ **E1 corpus assembly** can proceed without waiting
- ✅ **E2 ablations** can use BitNet + MMfreeLM-370M as comparison anchors
- ✅ **E3 Mini/Base/Medium** scaling-chain has 2 baselines for paper Table 2
- ✅ **Paper draft** can populate Table 2 with 2/7 rows + appendix note
- ✅ **TaxPrime team** can start dataset creation in parallel (no dependency)

## 5. What this DOES NOT block

- E0.0.4 (cloud GPU decision)
- E0.0.7 (data licensing review)
- E0.0.8 (dataset versioning strategy)
- E0.0.9 (CI GPU runner availability)
- All forward Phase E work

## 6. Recovery plan (when full sweep is desired)

**Pre-condition:** add §6.11-style retry+timeout hardening to `bench_baselines.py` (~30-60 min coding work):

| Hardening layer | Implementation |
|---|---|
| L1 — Module-level HF env vars | `HF_HUB_DOWNLOAD_TIMEOUT=60`, `HF_DATASETS_DOWNLOAD_TIMEOUT=60` set at import via `os.environ.setdefault` |
| L2 — Socket-level read timeout | requests/urllib3 `Session` with `timeout=(30, 60)` (connect, read) |
| L3 — Retry wrapper | `_retry_iter(factory)` matching `intllm.data` pattern: 5 attempts, exp backoff, transient-error catch (socket.timeout, ConnectionError, ReadTimeoutError) |
| L4 — Per-repo failure isolation | wrapper-script-level `|| continue` (already done in `/tmp/fajarquant_logs/sweep_remaining.sh`) |
| L5 — Watchdog | optional StepWatchdog-style daemon on per-task progress; SIGTERM if no advance for 30 min |

**Post-hardening sweep:**
- 5 remaining repos, ~10-15h GPU at hardened sweep
- Cost: ~$50-100 cloud burst OR free on dev machine over 2 days
- Outputs: 5 additional `bench_baselines_*.json` files
- Paper Table 2 row update with full 7/7 coverage

**Schedule:** treat as a Phase E **side-task** for E5 paper finalization. Not on critical path. Owner: technical lead in week of E5.3 paper drafting. Logged as Phase E TODO in this doc.

## 7. Paper Table 2 implication

Paper §4.1 Table 2 layout:

```
| Model       | Params | Tokens | Wikitext PPL | Lambada | HellaSwag | PIQA | ... | MMLU | TriviaQA | BoolQ |
|-------------|--------|--------|--------------|---------|-----------|------|-----|------|----------|-------|
| BitNet 2B4T | 2.0 B  | 4.0 T  | <real>       | <real>  | <real>    | <real>| ... | <real>| <real>  | <real>|
| MMfreeLM 370M | 0.37 B | 100 B | <real>       | <real>  | <real>    | <real>| ... | <real>| <real>  | <real>|
| MMfreeLM 1.3B | 1.3 B  | 100 B | DEFERRED†    | DEFERRED†| DEFERRED†| ... | ... | ... | ... | ... |
| MMfreeLM 2.7B | 2.7 B  | 100 B | DEFERRED†    | ... |
| SmolLM2 135M  | 0.135 B | ?     | DEFERRED†    | ... |
| SmolLM2 360M  | 0.36 B | ?     | DEFERRED†    | ... |
| Pythia 160m   | 0.16 B | 300 B | DEFERRED†    | ... |
| IntLLM Mini   | 0.007 B | ~70 M | <real>       | ... |
| IntLLM Base   | 0.046 B | ~500 M| <real>       | ... |
| IntLLM Medium | 0.14 B  | ~1.4 B| <real>       | ... |
| IntLLM Stretch| ~3 B    | ~30 B | <pending>    | ... |
| IntLLM Bilingual Mini  | 0.007 B | ~70 M  | <pending E3.1> | ... |
| IntLLM Bilingual Base  | 0.046 B | ~500 M | <pending E3.2> | ... |
| IntLLM Bilingual Medium | 0.14 B | ~1.4 B | <pending E3.3> | ... |
| IntLLM Bilingual Stretch | ~3 B  | ~30 B  | <pending E3.4> | ... |

† DEFERRED row note: "Baseline sweep halted due to repeated network-induced
CLOSE_WAIT hangs in bench_baselines.py. Re-run pending §6.11-style retry+timeout
hardening (Phase E side-task TODO)."
```

The deferred rows are honest — paper preserves transparency rather than hiding the gap.

## 8. Decision: PASS

- [x] Partial-baseline acceptance path adopted (per v1.1 §2 Q6 abort rule, triggered early)
- [x] BitNet ✅ + MMfreeLM-370M ✅ preserved as canonical Table 2 rows (commit JSONs in separate commit per audit hygiene)
- [x] 5 deferred baselines documented with disposition
- [x] Recovery plan written (5-layer §6.11 hardening for bench_baselines.py)
- [x] Phase E forward progress unblocked
- [x] Paper Table 2 layout includes "DEFERRED†" note for transparency

**Phase E E0.0.6 gate: PASS** (mechanical decision file landed per §6.8 R6).

6 of 10 E0 sub-tasks now closed: E0.0.1 ✅, E0.0.2 ✅, E0.0.3 ✅, E0.0.5 ✅, E0.0.10 ✅, **E0.0.6 ✅**.

---

*Decision committed 2026-04-25. Author: Claude Opus 4.7 + Fajar.*
*Companion: forthcoming commit will stage the 2 real baseline JSONs (BitNet + MMfreeLM-370M) as separate Phase D §3.4 partial-closure commit.*
*Recovery side-task ownership: technical lead, week of E5.3 paper drafting.*
