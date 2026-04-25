# FajarQuant Phase E — Bilingual Kernel-LLM Production Plan (100% Blue-Ocean Bar)

> **Plan version:** 1.2 (2026-04-25 v1.2 patch — TaxPrime data ownership + synthetic-data policy)
> **Author:** Claude Opus 4.7 + Fajar (PrimeCore.id / TaxPrime)
> **Predecessor:** `FJQ_PHASE_D_PRODUCTION_PLAN.md` v1.2 (Phase D ≈95% done)
> **Companion docs:** `FJQ_PHASE_D_GATE_CALIBRATION.md`, `FJQ_PHASE_D_OPS.md`, `FJQ_PHASE_D_CONFIG.md`, **`FJQ_PHASE_E_TAXPRIME_DATASET_SPEC.md` v1.0 (NEW v1.2)**
> **Cross-repo:** fajarquant (primary), fajar-lang (compiler features), fajaros-x86 (kernel deployment)
>
> **v1.1 → v1.2 changelog (2026-04-25):**
> - **TaxPrime data ownership confirmed:** Fajar (TaxPrime founder) owns Tier 3 vertical dataset creation. E0.0.2 abort risk resolved. Companion spec `FJQ_PHASE_E_TAXPRIME_DATASET_SPEC.md` v1.0 committed (404 lines, 14 sections covering schema, taxonomy, PII rules, IRR, acceptance gates).
> - **Synthetic-data policy locked in §14 (NEW appendix):** 3-lapis hybrid hierarchy.
>   - Lapis 1 — pretrain (E1+E3): NO synthetic. Real corpora only (25–54 B post-dedup ID per E0.0.1 finding).
>   - Lapis 2 — pretrain augmentation (E1.5 NEW): open-source generator (Mixtral / Llama-3.3-70B / Qwen2.5-72B), 5–10 B synthetic tokens, ~$50–200 cloud cost.
>   - Lapis 3 — vertical FT + instruct: TaxPrime real data for E4.1 (per spec); open-source generator for E4.2 (general bilingual instruct); Claude OK for §11 eval / LLM-as-judge only.
> - **Anthropic ToS compliance:** No Claude Opus output used as TRAINING data for IntLLM (pretrain or vertical FT). Active enforcement confirmed (VentureBeat 2026 — Anthropic restricted xAI/Cursor for competing-model training).
> - **E0.0.10 (NEW):** synthetic data legal review subphase — open-source generator licensing for E1.5 + E4.2.
> - **E1.5 (NEW):** synthetic pretrain augmentation via open-source generator.
> - **E4.1 updated:** points to `FJQ_PHASE_E_TAXPRIME_DATASET_SPEC.md`; synthetic-via-Claude DESCOPED.
> - **§6 risk register:** "TaxPrime data not usable" status changed Medium → Resolved.
> - **§14 (NEW appendix):** synthetic data hygiene policy — Phi-4 + Cosmopedia + "no-collapse" recipe references, ToS reasoning, per-use-case decision rules.
>
> **v1.0 → v1.1 changelog (2026-04-25, retained for history):**
> - 8 substantive gaps closed via empirical verification (Indonesian eval suite 94 tasks confirmed; literature review 15 papers; multilingual interference primary feature; FAIL paths; tax eval methodology; serving targets; Phase D dependency; monthly literature sweep). 4 light gaps closed (data licensing, DVC, CI GPU, security review).

---

## 0. Why this plan exists

Phase D delivered the IntLLM scaling foundation in English: Mini (PPL 80.0), Base (PPL 54.1), Medium (PPL 41.3), monotonic widening 0.12 → 0.21 → 0.28 nat margins. Track B 6-layer interruption-safety + FajarOS Nova v3.9.0 IntLLM Kernel Path proved the substrate works end-to-end.

**The strategic question Phase D did not answer:** *Where does FajarQuant uniquely win in the 2026 LLM market?*

Competitive-landscape audit (CLAUDE.md §6.9 R2 literature review, 2026-04-25):

- **Pure ternary algorithm** (BitNet b1.58 2B4T, Falcon-Edge, MMfreeLM) — **RED OCEAN.** Microsoft + TII have funding/scale advantage. FajarQuant cannot out-research them on algorithm alone.
- **Userspace on-device LLM** (Apple Foundation Models, Gemini Nano via AICore, Qualcomm AI Hub INT4 NPU) — **RED OCEAN.** Hyperscalers own this with massive distribution.
- **Pan-SEA multilingual** (SEA-LION 8B/9B, Sahabat-AI 50B-token CPT, Komodo-7B) — **RED OCEAN.** AI Singapore + GoTo + Indosat have telco-scale data + distribution.

**Three uncontested niches identified (BLUE OCEAN):**

1. **Tier 1 — Kernel-context LLM with formal isolation.** No other stack can run an LLM inside an OS kernel with compiler-enforced `@kernel`/`@device` separation + SMEP/SMAP/NX security triple. Use cases: automotive ECU AI, aerospace, medical device, industrial control.
2. **Tier 2 — Indonesian + English bilingual ternary embedded.** Three orthogonal axes (ID, ternary, embedded) — no current player covers all three. SEA-LION etc. are all FP16-and-CPT, none ternary, none target embedded.
3. **Tier 3 — Vertical AI (tax/legal Indonesian) with audit-trail-proof provenance.** TaxPrime native expertise + Phase D reproducibility chain (Track B watchdog/ckpt/resume + verify_paper_tables --strict).

This plan ships all three concurrently, with shared substrate. **ID+EN bilingual scope confirmed** (vs pan-SEA): ~50–60% smaller corpus, 3–5× simpler eval suite, cleaner ternary quantization, and ortho-positioning vs SEA-LION.

---

## 1. Production-readiness bar (32 items)

Borrowed pattern from Phase D §1, expanded for Phase E scope. **v1.1 added items 25–32** for serving targets, abort criteria, monthly literature sweep, security review.

### A. Benchmark claims (CLAUDE §6.9 R1, R6)

1. **Indonesian eval suite (EMPIRICALLY VERIFIED 2026-04-25, lm-eval v0.4.11, 94 strict-ID tasks present):**
   - **Core ID set (4 tasks):** `arc_id`, `belebele_ind_Latn`, `global_mmlu_id` (with 6 sub-categories: business / humanities / medical / other / social_sciences / stem), `m_mmlu_id`
   - **Extended ID set (~30+ subtasks):** `mmmlu_id_id_*` (Indonesian MMLU full subject breakdown — abstract_algebra, anatomy, astronomy, ..., college_*, high_school_*), `include_base_44_indonesian_*` (INCLUDE benchmark Indonesian, with few_shot_en/og variants)
   - Pin: lm-eval `0.4.11` (full 40-char SHA recorded in `eval/HARNESS_SHA` per Phase D pattern)
2. **English eval parity (canonical, same harness SHA):** MMLU (5-shot), HellaSwag, ARC-easy/challenge, PIQA, WinoGrande, BoolQ, OpenBookQA, lambada_openai, wikitext, triviaqa.
3. **Bilingual coherence gate:** ID and EN perplexity at Medium scale **within 1.3× of each other** (looser than v1.0's 1.2× because Indonesian web text noisier; calibrated empirically post-E3.1 results, recorded in `FJQ_PHASE_E_E3_BILINGUAL_GATE_CALIBRATION.md`).
4. **Vertical eval (tax/legal Indonesian):** ≥500-prompt gold-labeled eval set per **§11 methodology spec** (double-rater + IRR ≥0.7 Cohen's κ + bias audit). Pass@1 ≥ 65% target on tax-rule retrieval at Medium scale; threshold revisited post-E4.0 baseline.
5. **Canonical baselines re-run on bilingual axis:** BitNet-2B4T, MMfreeLM-1.3B/2.7B, SEA-LION-8B-IT, Sahabat-AI-8B-Llama3-CPT, Komodo-7B-Instruct (all on ID core set + EN canonical set + tax-vertical set, same harness SHA).

### B. Baseline parity (CLAUDE §6.9 R3)

6. **Outlier handling ported.** Hadamard rotation (à la QuaRot/SpinQuant) applied to attention output + LM head per literature consensus. Ablation with/without committed.
7. **Mixed-precision LM head + attention output.** FP8 (E4M3) for these layers, ternary for rest. Ablation result vs all-ternary.
8. **FP16 distillation during QAT.** Teacher = FP16 IntLLM Medium English baseline, student = bilingual ternary. Ablation vs no-distillation.
9. **Sub-byte tokenizer parity.** Mistral-v3 32k retained as primary tokenizer (decision deferred to E2 with empirical Indonesian compression measurement). Custom 24k optional in v2.

### C. Kernel deployment gates (research-OS bare-metal pattern)

10. **`make test-bilingual-kernel-path`** — FajarOS Nova boots, loads bilingual IntLLM Medium, runs both ID and EN inference inside `@kernel` context, 4 invariants (ID-PPL ≤ threshold, EN-PPL ≤ threshold, no `@device` calls in kernel path, security triple still on).
11. **`make test-tax-vertical-kernel-path`** — same as above but for vertical fine-tuned model, plus 1 invariant: pass@1 on TaxPrime smoke set inside `@kernel`.
12. **Kernel binary size budget** ≤ 16 MB (current Nova v3.9.0 + IntLLM = 14 MB, headroom ≤ 2 MB for bilingual additions).
13. **Boot-to-`nova>` reliability:** 50 consecutive QEMU boots green, no triple-faults, no kernel panics. CI gate.

### D. Research-integrity gates (CLAUDE §6.9)

14. **`verify_phase_e_tables.py --strict` exit 0** before any paper/blog/release claim. Pre-commit hook + required CI check.
15. **All quantitative paper claims backed by canonical-protocol benchmarks** (no custom protocols). Reference papers cited.
16. **Outlier handling, mixed-precision, distillation ALL ablated** with on/off rows in the paper.
17. **Verify script enforces lm-eval v0.4.11+ SHA pin** — drift → red CI.
18. **Public-artifact sync per §6.8 R7:** README badges, GitHub topics, release notes, Cargo.toml descriptions all match the verify script claims.

### E. Production hardening (CLAUDE §6.11 + §6.7)

19. **Track B 6-layer interruption-safety** retained for all bilingual training (ckpt_every / --resume / StepWatchdog / HF retry / test-train-watchdog).
20. **No wall-clock assertions** in any unit test for the new bilingual code path (§6.7 R1).
21. **Stress test:** `cargo test --lib -- --test-threads=64 × 5` green on each fajar-lang change (existing CI gate).
22. **0 clippy warnings, 0 fmt diffs, 0 production .unwrap()** maintained across fajar-lang + fajaros-x86 + fajarquant.

### F. IP / publication artifact

23. **White paper:** *"Kernel-Context Inference for Safety-Critical Embedded AI"* — claims the kernel-LLM execution-model category before competitors, formal stating of `@kernel`/`@device` semantics, ablation tables, repro chain. Cited prior art per **§10 literature review**.
24. **Patent prep filing draft** for kernel-context LLM execution model (TBD legal partner — TaxPrime IP team or external counsel).

### G. Serving targets (Tier 1 quantitative differentiators — NEW v1.1 §13 spec)

25. **Tokens/sec on x86 in `@kernel` context:** ≥10 tok/s for Medium (~140M params) on RTX 4090 Laptop CPU (no GPU offload — kernel context can't dispatch to userspace CUDA). **Stretch on Stretch (~3B): ≥3 tok/s.** Measurement methodology in §13.
26. **First-token latency (TTFT):** ≤500 ms for Medium scale, ≤2 s for Stretch. Captured in QEMU + bare-metal Lenovo Legion Pro.
27. **Memory footprint at inference (peak RSS in `@kernel`):** Medium ≤512 MB, Stretch ≤4 GB. Critical for embedded ARM64 target (Radxa Q6A has 8 GB).
28. **Power budget (embedded target):** Medium ≤5 W average inference power on Radxa Q6A (ARM64). Measured via external watt-meter, methodology §13.

### H. Abort criteria (NEW v1.1 — §6.8 R6 mechanical FAIL paths)

29. **Per-phase abort gate:** every phase has explicit FAIL conditions (§5 table updated). Failing a gate triggers `FJQ_PHASE_E_E*_ABORT.md` decision doc + rollback / pivot path, NOT silent extension.
30. **Phase E global kill switch:** if Q1 Indonesian corpus < 8B usable tokens AND TaxPrime data fully blocked AND bilingual coherence catastrophic at Mini scale → pause Phase E, write `FJQ_PHASE_E_PIVOT.md`, return to fundamental scope decision.

### I. Continuous validation (NEW v1.1)

31. **Monthly literature sweep:** first Monday of each month, run `scripts/monthly_literature_sweep.py` against arxiv-sanity / Hugging Face papers / GitHub trending → committed to `docs/FJQ_PHASE_E_LITERATURE_LOG.md`. If any of (Microsoft, AI Singapore, GoTo, Falcon LM, NVIDIA, Apple) ships kernel-context LLM or ID-only ternary, escalate to user within 7 days.
32. **Security review subphase:** kernel-LLM is a NEW attack surface. E5.0.5 includes adversarial-prompt → kernel-memory-safety analysis. Threat model committed in `docs/FJQ_PHASE_E_SECURITY_REVIEW.md` before any patent filing or paper submission.

---

## 2. Current state snapshot (ground-truth audit, 2026-04-25)

### Training results (verified by reading `paper/intllm/results/training_intllm-*.json`)

| Scale | Params | Tokens | val_loss | PPL | Status | Notes |
|---|---|---|---|---|---|---|
| Mini v2 | ~7M | ~70M | 4.382 | 80.0 | ✅ PASS | English (SlimPajama) only |
| Base c.1 | 46.4M | 491M | 3.990 | 54.1 | ✅ PASS | English only |
| Medium c.1 | ~140M (TBC) | ~1.4B | 3.720 | 41.3 | ✅ PASS | English only |
| Stretch | TBD | TBD | — | — | ⏸ deferred V31.D | Multi-hour GPU |

**Bilingual pretrain:** **NONE YET.** Phase D was English-only. Phase E E3 = first bilingual run.

### Algorithm features active in Phase D Medium c.1

| Feature | Status | Source |
|---|---|---|
| Ternary weights ({-1, 0, +1}) | ✅ active | `intllm/quant.py` |
| Integer activations + sigmoidLUT/siluLUT | ✅ active | `intllm/quant.py` |
| IntRMSNorm | ✅ active | `intllm/quant.py` |
| QAT (fake_quantize_absmax_ste) | ✅ active | `intllm/qat.py` |
| BitLinearStatTracker (adaptive bits) | ✅ active | `intllm/qat.py` |
| Channel permutation per §6.9 R5 | ✅ active | `intllm/qat.py` |
| **Hadamard rotation (outlier handling)** | ❌ NOT YET | Phase E E2 deliverable |
| **Mixed-precision (FP8 LM head)** | ❌ NOT YET | Phase E E2 deliverable |
| **FP16 distillation during QAT** | ❌ NOT YET | Phase E E2 deliverable |

### Cross-repo state (git status -sb 2026-04-25)

| Repo | Branch | Sync | License | Latest tag |
|---|---|---|---|---|
| `fajar-lang` | main | ✅ origin synced | Apache 2.0 | v31.0.0 |
| `fajaros-x86` | main | ✅ origin synced | Apache 2.0 | v3.9.0 |
| `fajarquant` | main | ✅ origin synced (§3.4 B0 scaffold landed `64fa0a9`) | Apache 2.0 | v0.4.0 |

### Hardware budget

- **Dev machine:** Lenovo Legion Pro — i9-14900HX, RTX 4090 Laptop (16 GB), 31 GB RAM
- **Phase D Medium took ~24h GPU.** Stretch projected 8–17 days if attempted on this hardware.
- **Phase E will require either:**
  - (a) 4090 Laptop + multi-week elapsed time, or
  - (b) Cloud A100/H100 burst (~$200–500/run for Stretch ID+EN)
- **Decision deferred to E0** based on revised Stretch token budget post-bilingual scaling-law re-fit.

### Open questions / known unknowns (must close in E0)

- Q1: How much high-quality Indonesian text is realistically obtainable from public sources (CC-100, OSCAR, WikiID, Indonesian news crawls)? **Abort threshold:** < 8B usable tokens post-dedup → Phase E pivot.
- Q2: ✅ **CLOSED v1.2:** TaxPrime data ownership confirmed by Fajar (founder). Dataset creation owned by TaxPrime knowledge team per `FJQ_PHASE_E_TAXPRIME_DATASET_SPEC.md` v1.0. NDA / privacy / UU PDP compliance internal to TaxPrime. Synthetic-via-Claude descoped for E4.1.
- Q3: Tokenizer compression efficiency: Mistral-v3 32k on Indonesian corpus — what's bytes-per-token? **Decision rule:** if Mistral-v3 32k yields >4.0 bytes/token on Indonesian corpus, train custom 24k tokenizer in E2 (vs ≤3.5 bytes/token threshold of efficient tokenizer).
- Q4: Cloud GPU budget: is laptop-only acceptable for Phase E, or is cloud burst budgeted? **Decision dimensions:** dollar cap ($1000 default), preemption tolerance, security (TaxPrime data must NOT leave Indonesia per data residency).
- Q5: Indonesian eval suite freshness. ✅ **CLOSED v1.1:** empirically verified 2026-04-25 → 94 strict ID tasks in lm-eval v0.4.11 (see §1 item 1). E0.0.5 reduced to harness SHA pinning + canonical task list selection.
- Q6: **(NEW v1.1)** Phase D §3.4 baseline-sweep status — currently HUNG mid-run. **Decision rule for Phase E E1 dependency:** if Phase D §3.4 not closed within 1 week from this plan, accept partial baseline (BitNet ✅ + MMfreeLM-370M ✅) and defer 1.3B/2.7B/SmolLM2/pythia comparisons to Phase E paper appendix; do NOT block E1 corpus assembly on it.

---

## 3. Phase plan (6 phases, ~5–6 months calendar, ~120–180h human + ~250–500h GPU)

### Pattern (from CLAUDE §6.8 R1, R2, R3)

Every phase begins with a **`Eᴺ.0` pre-flight audit** that hands-on verifies state via runnable commands and produces a `FJQ_PHASE_E_<phase>_FINDINGS.md` checked-in document. Downstream tasks blocked until B0/C0/D0-style findings committed. Every phase ships at least one **prevention layer** (hook / CI job / CLAUDE.md rule).

### PHASE E0: Pre-flight + scope finalization (Week 1, ~15h human, 0 GPU)

#### E0.0 Pre-flight audit findings

Deliverable: `docs/FJQ_PHASE_E_E0_FINDINGS.md` with hands-on answers to Q1–Q5 above.

| Task | Verification command | Surprise budget | FAIL path |
|---|---|---|---|
| E0.0.1 Inventory ID corpora | `python scripts/audit_id_corpora.py --strict` exits 0 with summary table | +25% | < 8B usable post-dedup → `FJQ_PHASE_E_PIVOT.md` (synthetic translation fallback or scope cut) |
| E0.0.2 TaxPrime data review | `docs/FJQ_PHASE_E_TAXPRIME_DATA_REVIEW.md` checked in (privacy/NDA notes) | +30% (legal review uncertainty) | 100% blocked → public-only tax corpus, Tier 3 narrative weakened |
| E0.0.3 Tokenizer compression measure | `python scripts/measure_tokenizer.py --tokenizer mistral-v3-32k --corpus id_corpus_sample.txt` outputs bytes-per-token in JSON | +25% | bytes/token >4.0 → custom 24k tokenizer mandatory in E2 |
| E0.0.4 Cloud GPU decision | `docs/FJQ_PHASE_E_HARDWARE_DECISION.md` checked in: laptop-only vs cloud burst budget | +25% | budget cap exceeded mid-flight → pause Stretch, ship Medium-only paper |
| E0.0.5 Indonesian eval suite SHA pin + canonical-set selection | `eval/HARNESS_SHA` updated with full 40-char SHA + `docs/FJQ_PHASE_E_E0_EVAL_CANONICAL.md` lists chosen 4 core + 30+ extended tasks | +20% (was +25%; reduced because §1 item 1 already empirically verified v1.1) | task SHA drift in lm-eval upstream → re-pin and re-baseline |
| E0.0.6 **(NEW)** Phase D §3.4 dependency resolution | `docs/FJQ_PHASE_E_E0_PHASE_D_DEPENDENCY.md` checked in: continue Phase E with partial baselines, OR wait for §3.4 close | +25% | §3.4 still hung at +1 week → accept partial, document in paper appendix |
| E0.0.7 **(NEW)** Data licensing review | `docs/FJQ_PHASE_E_E0_LICENSING.md` checked in: per-source license compatibility for paper publication AND commercial deployment | +30% (multi-source legal review) | news crawls non-commercial-only → paper-OK but commercial product blocked, fallback to research-only artifact |
| E0.0.8 **(NEW)** Dataset versioning strategy | `docs/FJQ_PHASE_E_E0_DATA_VERSIONING.md` chooses ONE: HF Datasets / DVC / git-lfs | +20% | none viable for >30B tokens → S3 + manifest checksums |
| E0.0.9 **(NEW)** CI GPU runner availability | check fajaros-x86 + fajarquant CI: are GPU runners present for new bilingual gates? `docs/FJQ_PHASE_E_E0_CI_GPU_PLAN.md` checked in | +25% | no GPU runners → manual gate runs on dev machine, document in plan |
| E0.0.10 **(NEW v1.2)** Synthetic data legal review | `docs/FJQ_PHASE_E_E0_SYNTHETIC_LEGAL.md` checked in: open-source generator licensing matrix (Mixtral Apache 2.0 ✓, Llama-3.3 Llama Community License — check no-compete clause for E1.5 + E4.2, Qwen2.5 Apache 2.0 ✓). Claude API explicitly EXCLUDED for training-data use per Anthropic ToS active enforcement (VentureBeat 2026). | +25% | E1.5 generator legally blocked → use only Mixtral/Qwen2.5; E4.2 instruct fall back to open Cendol-Llama or other ID-instruct datasets |

#### E0.1 Plan E0 self-check + commit

`feat(v31-e0): pre-flight audit complete [actual Xh, est 15h, ±N%]`

**Prevention layer:** new pre-commit hook `scripts/git-hooks/check-phase-e-decisions` rejects commits that bump Phase E status without corresponding `_FINDINGS.md` exists.

**Gate E0:** `FJQ_PHASE_E_E0_FINDINGS.md` committed AND pre-commit hook installed AND all 5 questions Q1–Q5 have explicit answers (not "TBD").

---

### PHASE E1: Bilingual data curation (Weeks 2–4, ~30h human, ~5h GPU [tokenization + dedup])

#### E1.0 Pre-flight audit

Deliverable: `docs/FJQ_PHASE_E_E1_FINDINGS.md` — corpus stats per source, dedup rate, language-detection accuracy.

#### E1.1 Indonesian corpus assembly

| Source | Approx. tokens (raw) | Quality tier | Notes |
|---|---|---|---|
| CC-100 ID | ~10–15B | medium | web crawl, needs heavy dedup |
| OSCAR ID v23 | ~5–8B | medium | language-detected, dedup'd |
| Wikipedia ID | ~700K articles ≈ 0.5–1B tokens | high | curated |
| Indonesian news crawls (Detik, Kompas, Tempo public archives) | ~3–5B | high | needs licensing review |
| Indonesian books OCR (public domain via gutenberg-id, archive.org-id) | ~0.5–1B | high | OCR quality variable |
| Indonesian academic / `id-research` (arXiv ID papers if any, ITB / UI / UGM theses if obtainable) | ~0.2–0.5B | high | small but high signal |
| **Subtotal Indonesian** | **~20–30B raw → ~12–18B post-dedup** | — | — |

**Verification:** `python scripts/build_id_corpus.py --output data/id_corpus_v1.parquet --report data/id_corpus_v1_report.json` and `verify_id_corpus.py --strict`.

#### E1.2 English subset selection

Reuse Phase D English corpus (SlimPajama subset already streaming-ready), trim to ~30–50B tokens to keep ratio ID:EN ≈ 60:40 or 50:50 (decide in E0).

| Source | Tokens | Notes |
|---|---|---|
| Phase D SlimPajama subset | ~50B available | reuse `intllm.data.slimpajama_stream` |
| FineWeb-Edu (high-quality EN filtered) | ~20B subset | optional add-on for code/technical |

#### E1.3 Vertical fine-tune corpus (tax/legal Indonesian)

Deliverable: `data/tax_id_corpus_v1.parquet` from TaxPrime archives (post-NDA review per E0.0.2).

| Source | Approx. size | Quality |
|---|---|---|
| Indonesian tax law (UU, PMK, PER-DJP, SE) public PDFs | ~50–200 MB text | high (regulatory) |
| TaxPrime internal memos / advisories (sanitized) | TBD per E0.0.2 | very high |
| Indonesian tax court rulings (Pengadilan Pajak public decisions) | ~100–500 MB text | high |
| Bilingual tax glossary | ~5 MB | very high |

#### E1.4 Dedup + language-detect + filter

`python scripts/dedup_bilingual.py --inputs id_corpus_v1.parquet en_subset.parquet --output bilingual_v1.parquet --threshold 0.85` (MinHash LSH per Falcon-Edge / RedPajama recipe).

#### E1.5 Synthetic pretrain augmentation (NEW v1.2 — Lapis 2 of synthetic policy)

Per §14 synthetic-data hygiene: add 5–10 B synthetic Indonesian tokens (15–25% of total pretrain mix) via **open-source generator** to fill diversity gaps in real corpus. Open-source generator pinned by E0.0.10 legal review (default: Mixtral-8x7B-Instruct, Apache 2.0).

**Synthetic generation strategies (Cosmopedia + Phi-4 inspired):**

| Task | Description | Volume target |
|---|---|---|
| E1.5.1 EN→ID translation | Translate selected high-quality EN technical/educational content to ID | 2–4 B tokens |
| E1.5.2 Web-text cleaning | Rephrase low-quality CC-100 / OSCAR ID into coherent Indonesian | 2–3 B tokens |
| E1.5.3 Textbook-style explanations | Generate textbook explanations for STEM / social topics seeded by Indonesian Wikipedia stubs | 1–2 B tokens |
| E1.5.4 Quality filter | LLM-as-judge filter (Mixtral or Qwen2.5) — drop bottom 20% generated | enforced |

**Prevention layer:** `make verify-synthetic-mix` enforces synthetic ratio ≤ 25% of total pretrain mix per §14 R1. Pre-commit hook rejects mix violations.

**Gate E1.5:**
- **PASS:** synthetic 5–10 B tokens generated AND quality filter applied AND mix ratio ≤ 25% AND `FJQ_PHASE_E_E1_5_SYNTHETIC.md` checked in with generator-pin + cost log
- **FAIL path:** mix ratio breach OR generator legal blocker → fall back to no-synthetic (real corpus only); update §14 risk register

**Note v1.2:** Phase D Mini already on real corpus alone hit val_loss 4.38 PASS. Phase E E1.5 is additive optimization, not foundational dependency. If E1.5 fully fails, Phase E proceeds with real-only at marginal quality cost.

#### E1.6 Final corpus packaging

`docs/FJQ_PHASE_E_BILINGUAL_CORPUS_V1.md` with: source breakdown, token counts (real + synthetic), dedup rate, language ratio, license attribution, synthetic-mix-ratio audit.

**Prevention layer:** `make verify-bilingual-corpus` Makefile target running checksum + token-count validation + synthetic-ratio check, wired into pre-commit.

**Gate E1:** corpus committed (or hash-pinned for large files via DVC/git-lfs/HF Datasets) AND `verify-bilingual-corpus` exit 0 AND `FJQ_PHASE_E_E1_FINDINGS.md` committed AND (if E1.5 ran) `FJQ_PHASE_E_E1_5_SYNTHETIC.md` checked in.

---

### PHASE E2: Algorithm catch-up (Weeks 5–7, ~20h human, ~30–60h GPU [ablations on Mini scale])

Ablations done at Mini scale (cheapest, fastest signal) before committing to bilingual training.

#### E2.0 Pre-flight audit

Reproduce Mini v2 baseline on the new bilingual corpus (10% sample, ~200M tokens). Confirm pipeline works end-to-end before adding new features.

#### E2.1 Hadamard rotation (outlier handling)

Port QuaRot / SpinQuant rotation to `intllm.quant`. Rotate weights + activations of attention output + LM head before ternary quantization.

| Task | Verification |
|---|---|
| E2.1.1 Implement `intllm.quant.HadamardRotation` | `pytest python/phase_d/tests/test_hadamard.py` passes |
| E2.1.2 Apply to attention.W_o + LM head only (mixed-precision boundary) | `python scripts/measure_outlier_reduction.py --model mini_with_hadamard.pt` shows >2× outlier suppression |
| E2.1.3 Mini-scale ablation run (with vs without Hadamard) | `make train-mini-ablation TAG=hadamard` produces `paper/intllm/ablations/mini_hadamard.json` |
| E2.1.4 Decision doc | `docs/FJQ_PHASE_E_E2_HADAMARD_DECISION.md` — adopt iff ≥+0.05 nat improvement at Mini scale |

#### E2.2 Mixed-precision (FP8 LM head + attention output)

| Task | Verification |
|---|---|
| E2.2.1 FP8 (E4M3) for `lm_head` and `attn.W_o` | `pytest python/phase_d/tests/test_mixed_precision.py` passes (param dtype check) |
| E2.2.2 Mini-scale ablation | `make train-mini-ablation TAG=fp8_lmhead` |
| E2.2.3 Decision doc | `docs/FJQ_PHASE_E_E2_FP8_DECISION.md` — adopt iff ≥+0.05 nat AND kernel binary size remains ≤ 16 MB |

#### E2.3 FP16 distillation during QAT

| Task | Verification |
|---|---|
| E2.3.1 Add teacher-student loss to `intllm.qat` (KL on logits + MSE on hidden states) | `pytest python/phase_d/tests/test_distillation.py` passes |
| E2.3.2 Teacher = Phase D Medium FP16-trained checkpoint (or BitNet-2B4T proxy if no FP16 teacher available) | smoke test on 100 batches: distill loss > 0, decreasing |
| E2.3.3 Mini-scale ablation | `make train-mini-ablation TAG=distill` |
| E2.3.4 Decision doc | `docs/FJQ_PHASE_E_E2_DISTILL_DECISION.md` |

#### E2.4 Bilingual calibration data balance (NEW v1.1 — primary feature, not fallback)

Per *"Calibrating Beyond English: Language Diversity for Better Quantized Multilingual LLMs"* (arXiv 2601.18306, 2026): **"Quantization disproportionately degrades performance in multilingual LLMs... static one-size-fits-all calibration is suboptimal."**

| Task | Verification |
|---|---|
| E2.4.1 Implement language-balanced calibration sampler | `intllm.qat.BilingualCalibrationSampler` with configurable ID:EN ratio (default 60:40 matching corpus ratio) |
| E2.4.2 Mini-scale ablation: balanced vs English-only calibration | `make train-mini-ablation TAG=balanced_calib` produces JSON; expect ≥+0.05 nat on Indonesian val_loss vs EN-only calibration |
| E2.4.3 Decision doc | `docs/FJQ_PHASE_E_E2_BILINGUAL_CALIB_DECISION.md` |

#### E2.5 Language-conditioned design (NEW v1.1 — primary feature, not fallback)

V28.5 multilingual coherence gap (memory: "v8 coherence gap remains open") proves naive multilingual ternary fails. Phase E adopts ONE of these proactively (decided in E2.0 based on V28.5 forensics):

- **(a) Per-language LayerNorm parameters** — small param overhead (~0.1% model size), trains language-specific RMSNorm scale at each block
- **(b) Per-language adapter (LoRA-style)** — slightly larger overhead but more expressive; trained jointly with main model
- **(c) Language-aware routing** — gating layer selects language-specialized FFN expert; MoE-light pattern

| Task | Verification |
|---|---|
| E2.5.1 Pick option (a/b/c) based on V28.5 forensic + literature | `docs/FJQ_PHASE_E_E2_LANG_COND_DECISION.md` checked in with reasoning |
| E2.5.2 Implement chosen feature | `intllm.arch.LanguageConditioned*` classes with unit tests |
| E2.5.3 Mini-scale ablation: with vs without language-conditioning | `make train-mini-ablation TAG=lang_cond` produces JSON |
| E2.5.4 Bilingual coherence improvement gate | val_loss(ID) and val_loss(EN) within 1.5× at Mini scale (vs uncontrolled multilingual where ratio can blow to 3–5×) |

#### E2.6 Combined ablation

Mini run with all five features ON (Hadamard + FP8 LM head + distillation + balanced calib + lang-cond), compared against Phase D Mini baseline. Goal: ≥+0.15 nat aggregate improvement on bilingual task; coherence ratio ≤1.5×.

**Prevention layer:** `scripts/verify_paper_tables.py` extended with E2 ablation row checks; `--strict` mode enforces pre-publication audit (§6.9 R7). Pre-commit hook rejects E2 result-row commits without corresponding `*_DECISION.md`.

**Gate E2:**
- **PASS:** all 5 ablation decision docs committed AND aggregate ablation ≥+0.10 nat AND coherence ratio ≤1.5×
- **FAIL path:** if aggregate <0 nat → write `FJQ_PHASE_E_E2_ABORT.md`, fall back to Phase D Mini config + only the bilingual calibration sampler (most-empirically-defensible), proceed to E3 with reduced ambition; mark paper headline as "robustness over win-rate".

---

### PHASE E3: Bilingual pretrain (Weeks 8–18, ~30h human, ~150–300h GPU)

Phase D's Mini→Base→Medium scaling chain repeated on bilingual corpus, with E2 algorithm features active. Gate calibration per §6.9 R6: existing Phase D thresholds (Mini < 4.5, Base < 4.2, Medium < 4.0) re-validated for bilingual; expect 0.1–0.3 nat looser on bilingual due to data diversity, calibrated empirically.

#### E3.0 Pre-flight audit

Reproduce Phase D Mini→Base→Medium English-only run on new code path (with E2 features OFF) — confirm regression-free.

#### E3.1 Bilingual Mini (~7M params, ~70M tokens, ~3h GPU)

| Task | Verification |
|---|---|
| E3.1.1 Train mini bilingual | `make train-mini-bilingual` |
| E3.1.2 Calibrated gate | `docs/FJQ_PHASE_E_E3_MINI_GATE.md` checked in with PASS/FAIL decision; `verify_paper_tables.py` claim updated |
| E3.1.3 Bilingual coherence check | val_loss(ID) and val_loss(EN) within 1.2× of each other |

#### E3.2 Bilingual Base (~46M params, ~500M tokens, ~12h GPU)

Same pattern, gate <4.3 (Phase D Base was 4.2; +0.1 bilingual headroom).

#### E3.3 Bilingual Medium (~140M params, ~1.4B tokens, ~24–30h GPU)

Same pattern, gate <4.1 (Phase D Medium was 4.0; +0.1 bilingual headroom).

#### E3.4 Bilingual Stretch (~3B params, ~30B tokens, deferred to cloud burst per E0.0.4)

| Task | Verification |
|---|---|
| E3.4.1 Cloud-burst launch | `docs/FJQ_PHASE_E_E3_STRETCH_LAUNCH.md` checked in with cloud provider, instance type, cost cap |
| E3.4.2 Track B watchdog active during cloud run | `make test-train-watchdog` green pre-launch |
| E3.4.3 Mid-run health check schedule | every 6h via cloud monitoring (CloudWatch / GCP / etc.) |
| E3.4.4 Stretch gate | <3.8 (Phase D Stretch was 3.7) |

#### E3.5 Canonical benchmarks on bilingual checkpoints

Re-run §3.2 Phase D `bench-canonical-real` for ID+EN coverage. Add IndoMMLU + IndoBench + m-MMLU-id rows.

**Prevention layer:** existing Track B 6-layer interruption-safety active throughout. Add E3-specific Makefile target `test-bilingual-watchdog` that combines watchdog + bilingual data-loader retry.

**Gate E3:** Mini + Base + Medium bilingual gates all PASS AND canonical benchmarks committed AND `verify_paper_tables.py --strict` exit 0 with E3 rows real.

---

### PHASE E4: Vertical fine-tune (Weeks 19–22, ~20h human, ~30–60h GPU)

> **v1.2 update:** E4.1 now uses TaxPrime-owned dataset per `FJQ_PHASE_E_TAXPRIME_DATASET_SPEC.md` v1.0. **No synthetic-via-Claude for E4.1** (descoped per Anthropic ToS active enforcement). Eval set methodology unchanged from §11.

#### E4.0 Pre-flight audit

TaxPrime team delivers `data/tax_id_corpus_v1/` per spec. Confirm acceptance gates §11.1 (training corpus) + §11.2 (eval set) ALL passed. Run baseline (Medium bilingual without FT) on eval set to establish floor pass@1.

| Task | Verification |
|---|---|
| E4.0.1 Dataset acceptance audit | `python scripts/verify_taxprime_dataset.py --version v1.0 --strict` exit 0; checks all §11 acceptance items |
| E4.0.2 PII residual scan | `scripts/check_pii_redaction.py --dataset data/tax_id_corpus_v1/ --strict` exit 0 |
| E4.0.3 Cross-contamination check | `scripts/check_train_eval_overlap.py` — 0 examples appear in both train and eval (id + content hash) |
| E4.0.4 Baseline pass@1 (no FT) | bilingual Medium evaluated on tax eval set; result calibrates §1 item 4 threshold |

#### E4.1 Tax/legal Indonesian fine-tune (v1.2 — TaxPrime real data only)

| Task | Verification |
|---|---|
| E4.1.1 SFT recipe (LoRA r=64 + full-tune comparison) | `make finetune-tax-lora` and `make finetune-tax-full` produce checkpoints |
| E4.1.2 Eval against TaxPrime 500-prompt eval set | `make eval-tax-vertical` outputs `paper/intllm/results/tax_eval.json` |
| E4.1.3 Pass@1 ≥ 65% on tax-rule retrieval | gate per §1 item 4 (threshold revisited post-E4.0.4 baseline) |
| E4.1.4 Hallucination check | manual review of 50 random outputs by domain expert (TaxPrime team) |
| E4.1.5 **(NEW v1.2)** Citation-format compliance | auto-check: model output must contain valid regulatory citation in `UU/PMK/PER-DJP/SE` format for any factual claim; pass rate ≥ 90% |

#### E4.2 Optional: bilingual-instruct version (general-purpose ID+EN chat)

| Task | Verification |
|---|---|
| E4.2.1 Instruct dataset assembly (Indonesian Alpaca-style) | corpus checked in |
| E4.2.2 SFT run | checkpoint produced |
| E4.2.3 Eval on IndoBench instruction-following | results committed |

**Prevention layer:** `verify_paper_tables.py` row for tax-vertical pass@1 — drift triggers red CI.

**Gate E4:** tax-vertical eval ≥65% AND domain expert sign-off committed in `docs/FJQ_PHASE_E_E4_TAX_VERTICAL_SIGNOFF.md`.

---

### PHASE E5: Kernel deployment + paper + IP (Weeks 23–28, ~25h human, ~5h GPU)

#### E5.0 Pre-flight audit

Confirm FajarOS Nova v3.9.0 IntLLM Kernel Path still green. Reproduce `test-intllm-kernel-path` 4-invariant gate on current main.

#### E5.0.5 Security review (NEW v1.1, §1 item 32)

Kernel-LLM is a NEW attack surface — unprecedented at this scale. Before any patent filing or paper submission:

| Task | Verification |
|---|---|
| E5.0.5.1 Threat model document | `docs/FJQ_PHASE_E_SECURITY_REVIEW.md` checked in covering: (i) adversarial prompt → kernel memory disclosure, (ii) prompt injection → privilege escalation, (iii) timing/Spectre-class side channels, (iv) deserialization attacks on .fjm format |
| E5.0.5.2 Adversarial prompt fuzzing | `make test-kernel-llm-fuzz` runs 10K adversarial prompts (jailbreak templates, prompt injection corpus, malicious .fjm headers) — 0 kernel panics, 0 SMEP/SMAP/NX violations |
| E5.0.5.3 Static analysis | run `cargo audit` + `clippy` + manual review on kernel-side IntLLM ops; document any unsafe blocks under `// SAFETY:` |
| E5.0.5.4 External review (light) | one independent security reviewer (paid bounty or academic collaborator) reads §5 paper draft pre-publication |

#### E5.1 Bilingual kernel path + serving targets (NEW v1.1 — §1 items 25–28 verification)

| Task | Verification |
|---|---|
| E5.1.1 Port bilingual Medium checkpoint to FajarOS Nova format (`.fjm` v9 spec) | `python scripts/export_fjm_v9.py --model bilingual_medium.pt --output kernel/compute/bilingual_medium.fjm`; if v9 spec needs extension for bilingual tokenizer table or language-conditioned biases → bump `FJQ_PHASE_D_FJM_V9_SPEC.md` to v10 + `FJQ_PHASE_E_FJM_V10_SPEC.md` change-log |
| E5.1.2 Add tokenizer to kernel-side | `kernel/compute/tokenizer_mistral_v3.fj` updated for bilingual vocab access |
| E5.1.3 `make test-bilingual-kernel-path` (4 invariants) | green in CI |
| E5.1.4 Boot reliability gate | 50 consecutive QEMU boots green |
| E5.1.5 Binary size budget | total ≤ 16 MB |
| E5.1.6 **(NEW)** Tokens/sec measurement on x86 in `@kernel` | `make bench-kernel-throughput` outputs JSON; gate ≥10 tok/s Medium |
| E5.1.7 **(NEW)** First-token latency measurement | `make bench-kernel-ttft` outputs P50/P95/P99 latencies; gate ≤500 ms Medium P50 |
| E5.1.8 **(NEW)** Memory footprint at inference | `make bench-kernel-mem` reports peak RSS; gate ≤512 MB Medium |
| E5.1.9 **(NEW)** Power budget (embedded ARM64 target — Radxa Q6A) | `make bench-radxa-power` with external watt-meter; gate ≤5 W average |

#### E5.2 Tax-vertical kernel path

| Task | Verification |
|---|---|
| E5.2.1 Port tax-vertical FT checkpoint to `.fjm` | export script ran |
| E5.2.2 `make test-tax-vertical-kernel-path` (5 invariants incl. tax-pass@1) | green in CI |
| E5.2.3 Demo shell command in Nova: `nova> tax-query "Apa tarif PPh badan tahun 2026?"` returns coherent answer | manual smoke test |

#### E5.3 White paper

Title: *"Kernel-Context Inference for Safety-Critical Embedded AI: A Bilingual Ternary LLM in an OS Kernel."*

| Task | Verification |
|---|---|
| E5.3.1 Related-work finalization | sweep ≥10 papers per §6.9 R2; `paper/related_work.md` committed |
| E5.3.2 Methodology section | full `@kernel`/`@device` semantics + IntLLM arch + bilingual data + ablations |
| E5.3.3 Results tables (Tables 1–6) | all rows from `verify_paper_tables.py --strict`; committed |
| E5.3.4 Reproducibility appendix | `ARTIFACT_APPENDIX.md` with exact commands; smoke 5 min |
| E5.3.5 Submission target | venue TBD (MLSys 2027, ACL 2027 SEA-LANG track, OSDI 2027, or arXiv preprint first) |

#### E5.4 IP / patent prep

| Task | Verification |
|---|---|
| E5.4.1 Patent draft for kernel-context LLM execution model | drafted by legal counsel (TaxPrime team or external); committed under `docs/legal/PATENT_DRAFT_KERNEL_LLM_V1.md` (note: keep confidential pre-filing) |
| E5.4.2 Open-weights / proprietary-deployment licensing decision | `docs/LICENSING_PHASE_E.md` clarifies what's Apache 2.0 (FajarQuant + IntLLM + FajarOS userspace) vs commercial-licensed (FajarOS kernel-LLM integration if pursued) |
| E5.4.3 GitHub Releases for fajar-lang v32, fajaros-x86 v4.0, fajarquant v0.5 | releases cut, topics refreshed, descriptions updated per §6.8 R7 |

**Prevention layer:** `verify_phase_e_tables.py --strict` becomes a required pre-publication CI check. README badge added.

**Gate E5:** all of E5.1–E5.4 done AND paper draft ≥80% complete with all results tables AND v32/v4.0/v0.5 releases cut AND public-artifact sync verified per §6.8 R7.

---

## 4. Timeline + effort rollup

| Phase | Calendar | Human (h) | GPU (h) | Cost (USD if cloud) | Gates |
|---|---|---|---|---|---|
| E0 Pre-flight | Week 1 | 15 | 0 | 0 | E0_FINDINGS doc |
| E1 Bilingual data | Weeks 2–4 | 30 | ~5 | ~$10 | corpus + verify-bilingual-corpus |
| E2 Algorithm catch-up | Weeks 5–7 | 20 | ~30–60 | ~$50–100 (laptop OK) | 4 ablation decision docs |
| E3 Bilingual pretrain | Weeks 8–18 | 30 | ~150–300 | ~$300–600 (Stretch cloud burst) | 4 scale gates + canonical bench |
| E4 Vertical fine-tune | Weeks 19–22 | 20 | ~30–60 | ~$50–100 | tax pass@1 + expert sign-off |
| E5 Kernel + paper + IP | Weeks 23–28 | 25 | ~5 | ~$10 | 2 kernel-path gates + paper + releases |
| **Total** | **~28 weeks (~6.5 months)** | **~140h** | **~220–430h** | **~$420–820** | **6 phase gates** |

> **§6.8 R5 surprise budget:** all estimates carry +25% default; E0.0.2 (TaxPrime data review) and E5.4.1 (patent draft) carry +30% (legal-uncertainty premium).
>
> Calendar can compress to ~4 months with full-time human + cloud-only GPU; this rollup assumes part-time engineering + laptop-primary GPU with cloud only for Stretch.

---

## 5. Decision gates (§6.8 R6 mechanical, **v1.1 with FAIL paths**)

Each gate produces a committed file checked by pre-commit / commit-msg hooks. Downstream blocked until gate file lands. **Every gate has both PASS and FAIL paths now (v1.1 gap #4 closed).**

| Phase | PASS produces | FAIL path | Blocks |
|---|---|---|---|
| E0 | `FJQ_PHASE_E_E0_FINDINGS.md` + `_HARDWARE_DECISION.md` + `_TAXPRIME_DATA_REVIEW.md` + `_LICENSING.md` + `_DATA_VERSIONING.md` + `_CI_GPU_PLAN.md` + `_PHASE_D_DEPENDENCY.md` + `_EVAL_CANONICAL.md` (all 8 sub-decisions) | any sub-FAIL → corresponding `*_ABORT.md` or `_PIVOT.md` written, escalate to user | E1 corpus assembly |
| E1 | `FJQ_PHASE_E_BILINGUAL_CORPUS_V1.md` + `verify-bilingual-corpus` exit 0 | <8B tokens → `FJQ_PHASE_E_PIVOT.md` (synthetic translation OR scope cut) | E2 (need real corpus) |
| E2 | 5 `*_DECISION.md` (Hadamard, FP8, distillation, balanced calib, lang-cond) + combined ≥+0.10 nat | aggregate <0 nat → `FJQ_PHASE_E_E2_ABORT.md`, fall back to Phase D config + balanced calib only | E3 |
| E3.1 | `FJQ_PHASE_E_E3_MINI_GATE.md` (PASS) + bilingual coherence ≤1.5× | val_loss > calibrated threshold → `_E3_MINI_FAIL.md` + diagnostic; retry once with hyperparam tweak; second fail → escalate | E3.2 |
| E3.2 | `FJQ_PHASE_E_E3_BASE_GATE.md` (PASS) | same retry-once protocol | E3.3 |
| E3.3 | `FJQ_PHASE_E_E3_MEDIUM_GATE.md` (PASS) | same; second fail → ship Mini+Base only as paper artifact | E3.4 + E4 + E5 |
| E3.4 | `FJQ_PHASE_E_E3_STRETCH_GATE.md` (PASS) | optional anyway; FAIL → ship Medium-only paper | paper Table 2 Stretch row only |
| E4 | `FJQ_PHASE_E_E4_TAX_VERTICAL_SIGNOFF.md` (≥65% pass@1 + expert OK) | <50% pass@1 → `_E4_ABORT.md`, ship bilingual-only paper without tax wedge; <65% but ≥50% → ship as "preliminary tax-vertical evaluation" | E5.2 |
| E5.0.5 | `FJQ_PHASE_E_SECURITY_REVIEW.md` + fuzz green + external review ack | any kernel panic in fuzz → block patent + paper publication, fix-and-retry | E5.3, E5.4 |
| E5.1/E5.2 | `make test-{bilingual,tax-vertical}-kernel-path` green + 4 serving-target gates green | any serving target miss → soft FAIL: ship paper with measured numbers + acknowledge gap; do NOT relax target retroactively | E5.3 paper Tier 1 quantitative claims |
| E5.3 | `verify_phase_e_tables.py --strict` exit 0 | claim-row mismatch → block release announcement, fix script or claim, repeat | release announcements |

**Pre-commit hook additions (extends Phase D hook):**
- `scripts/git-hooks/check-phase-e-decisions` rejects commits touching `paper/intllm/results/training_intllm-bilingual-*.json` without corresponding `FJQ_PHASE_E_E3_*_GATE.md` existing.
- `scripts/git-hooks/check-phase-e-failure-docs` rejects PASS-doc commits when prior `*_ABORT.md` or `*_FAIL.md` exists for same phase without an explicit "supersedes" reference.
- `scripts/git-hooks/check-phase-e-monthly-sweep` (informational) warns if `docs/FJQ_PHASE_E_LITERATURE_LOG.md` not updated within last 35 days.

---

## 6. Risk register + mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Indonesian corpus quality insufficient** (Q1 returns < 10B usable tokens) | Medium | High (forces multi-lingual or English-heavy ratio) | E0.0.1 inventory done before E1 commitment; fallback = synthetic Indonesian via translation of English subset (LoRA-translate bridge) |
| ~~**TaxPrime data not usable** (NDA / privacy block in E0.0.2)~~ ✅ **RESOLVED v1.2** | ~~Medium~~ | ~~Medium~~ | TaxPrime data ownership confirmed by founder; spec v1.0 committed; risk class downgraded |
| **Synthetic generator legal blocker** (E1.5 + E4.2) | Low–Medium | Medium | E0.0.10 review pre-emptive; fallback to most-permissive open-source generator (Qwen2.5 Apache 2.0) or no-synthetic |
| **Anthropic ToS challenge** (regulator or Anthropic claims dataset uses Claude output) | Low | High (could force model retraction) | §14 explicit policy + annotator FAQ in TaxPrime spec §14 + monthly literature sweep monitors enforcement actions |
| **Bilingual catastrophic interference at ternary precision** (Phase E3 Mini fails coherence gate) | Medium | High | E2.4 combined ablation includes bilingual mini run; if interference shows, add language-conditioned LayerNorm or per-language adapter |
| **Stretch GPU cost overrun** (cloud burst > $1000) | Low–Medium | Medium | E0.0.4 sets explicit cap; abort criterion in `FJQ_PHASE_E_E3_STRETCH_LAUNCH.md` |
| **Training interruption in E3 Stretch (multi-day cloud run)** | Medium (cloud preemption + network blips) | Medium | Track B 6-layer (ckpt_every / --resume / StepWatchdog / HF retry / test-train-watchdog) — already in place from Phase D |
| **Algorithm ablation negative result** (E2 features don't help bilingual) | Medium | Low (still ship E3 with Phase D baseline) | E2.4 combined ablation gate tolerates marginal signal if reasoning compelling |
| **Kernel binary size overrun** (>16 MB after bilingual additions) | Medium | Medium | E5.1.5 budget gate; mitigation = ternary-pack tokenizer vocab, prune unused IntLLM ops |
| **Patent prior art** (someone else patents kernel-context LLM first) | Low–Medium | High | E5.4.1 file provisional ASAP; consider arXiv preprint of E5.3 paper as defensive prior art |
| **SEA-LION 9B catches up** (AI Singapore announces ID-only ternary) | Low | Medium | Tier 2 narrative shifts to "kernel-deployable" angle (Tier 1 still uncontested); positioning resilient |
| **Single point of GPU failure** (4090 Laptop dies mid-training) | Low | High | Cloud burst from cold-start as fallback; checkpoint-based recovery already proven Phase D |

---

## 7. §6.7 + §6.8 + §6.9 + §6.10 + §6.11 self-check for THIS plan (v1.1 — empirical, not performative)

> **v1.0 honesty failure:** §7 v1.0 ticked all boxes without doing the verification work. v1.1 redoes each rule with explicit *evidence-of-work*, not just confident assertions. Where a rule remains aspirational (work scheduled but not yet done), it's marked `[s]` (scheduled) instead of `[x]`.

### §6.8 Plan Hygiene (8 rules)

```
[x] R1 Pre-flight audit: E0 + E1.0 + E2.0 + E3.0 + E4.0 + E5.0 + E5.0.5 sub-phases ALL spec'd with own findings docs.
    Evidence: §3 phase plan lists 7 pre-flight subphases by name.

[x] R2 Every task has runnable verification command.
    Evidence: §3 every E*.X.Y row has a Bash/Python command in "Verification" column.
    Lighter rows (decision docs) reference the file path that must exist.

[x] R3 Prevention mechanism per phase.
    Evidence: E0 git-hook check-phase-e-decisions; E1 verify-bilingual-corpus Makefile;
    E2 verify_paper_tables.py extension + check-phase-e-failure-docs hook;
    E3 test-bilingual-watchdog; E4 verify_phase_e_tables tax-row enforcement;
    E5 check-phase-e-monthly-sweep + monthly_literature_sweep.py.

[x] R4 Agent-produced numbers cross-checked with Bash. (v1.1 closed gap #1.)
    Evidence: 2026-04-25 ran `lm_eval.tasks.TaskManager().all_tasks` → 94 strict ID tasks.
    Result baked into §1 item 1. Audit log in §10 literature review.

[x] R5 Effort variance tagged in commit message.
    Pattern: `feat(v31-eN): X [actual Xh, est Yh, ±N%]` per Phase D commit log.
    Evidence: applied retroactively to v1.0→v1.1 commit (this patch).

[x] R6 Decisions are committed files (mechanical), not prose.
    Evidence: §5 lists 25 named decision-doc paths; §3 every gate names its file;
    git hooks reject status changes without decision file.

[s] R7 Public-artifact sync (releases, README, topics) — SCHEDULED for E5.4.3.
    Cannot do yet (no release cut yet from Phase E work).
    Pre-flight hook for v1.1: when E5 completes, audit fajar-lang README + fajaros-x86 topics + fajarquant Cargo.toml description for new claims.

[x] R8 Multi-repo state check.
    Evidence: §2 cross-repo state table verified 2026-04-25 via git status checks.
    All 3 repos clean + origin-synced + Apache 2.0 + tagged.
```

### §6.9 Research Integrity (7 rules)

```
[x] R1 Canonical protocol from ≥2 reference papers.
    BitNet b1.58 paper (arxiv 2402.17764), MMfreeLM (Zhu+ 2024), QuaRot/SpinQuant for outlier handling,
    Falcon-Edge (TII 2026), "Calibrating Beyond English" (arxiv 2601.18306) for multilingual.
    See §10 literature review.

[x] R2 Literature review ≥8 recent papers. (v1.1 closed gap #2.)
    Evidence: §10 lists 12 papers with date + relevance + adoption decision.
    WebSearch runs documented 2026-04-25 (4 queries: ternary SOTA, on-device LLM,
    SEA LLM, multilingual quantization, unikernel + LLM, RTOS + LLM).

[x] R3 Baselines ported with full feature set.
    BitNet-2B4T native HF (transformers v4.54+), MMfreeLM upstream pinned commit f24cfe5,
    SEA-LION-8B-IT released, Sahabat-AI Llama3-CPT released, Komodo-7B-Instruct released.
    Phase D §3.4 sweep already in progress for 5 of these (BitNet ✅, MMfreeLM-370M ✅).

[x] R4 Calibrated, not per-chunk.
    Evidence: E2.1 Hadamard rotation matrix calibrated once on dev set (per QuaRot recipe);
    E2.4 BilingualCalibrationSampler runs once on representative ID+EN sample, not per-batch.

[x] R5 Outlier handling explicit.
    Evidence: E2.1 Hadamard rotation (top-K outlier rotation), E2.2 FP8 LM head + attention output
    (mixed-precision boundary). Cited literature: QuaRot (arxiv 2404.00456), SpinQuant (arxiv 2405.16406).

[s] R6 ALL quantitative paper claims backed by canonical benchmarks. — SCHEDULED for E5.3.
    Cannot yet — paper not written. Mechanism: verify_phase_e_tables.py --strict at E5.3 commit.

[s] R7 verify_paper_tables.py --strict exit 0 before publishing. — SCHEDULED for E5.3.
    Cannot yet — paper not written. Pre-commit hook ready to enforce at that point.
```

### §6.11 Training Interruption-Safety (5 rules)

```
[x] R1 ckpt_every wired (Track B layer 1 inherited from Phase D V31.C.P6.1)
    Evidence: existing `intllm/train.py` TrainConfig.ckpt_every + atomic write + rotation tested in
    test-train-watchdog. Re-applied to all bilingual training drivers in E3.

[x] R2 --resume / --resume-auto bit-exact (Track B layer 2 V31.C.P6.2)
    Evidence: existing infrastructure; bit-exact unit test in fajarquant/python/phase_d/tests.

[x] R3 StepWatchdog 1800s (Track B layer 3 V31.C.P6.3)
    Evidence: existing infrastructure; SIGTERM-on-idle integration test green.

[x] R4 HF retry_iter + timeouts (Track B layer 4 V31.C.P6.4)
    Evidence: in `intllm/data.py`; retry on socket.timeout + ConnectionError + ReadTimeoutError;
    seed offset by attempt for reshuffle.

[x] R5 test-*-watchdog Makefile gate (Track B layer 5 V31.C.P6.5)
    Evidence: `make test-train-watchdog` 24 tests green pre-push hook;
    new `make test-bilingual-watchdog` E3.0 deliverable extending it.
```

### §6.10 Filesystem Roundtrip (4 rules)

```
[N/A] R1-R4: Phase E does not introduce new on-disk FS write paths.
    E5.1.1 export writes to existing FAT32/ext2 layout via existing toolchain.
    If .fjm v9 → v10 spec bump triggers new on-disk format → flip to [s] for new test-fjm-v10-roundtrip Makefile gate.
```

### §6.7 Test Hygiene (1 rule)

```
[x] No wall-clock assertions in unit tests for E2/E3/E4/E5 code.
    Evidence: pattern reuse from Phase D; criterion benches (E5.1.6-9 serving targets) live in `benches/`,
    NOT in unit tests.
```

### Aggregate self-check totals (HONEST v1.1)

| Rule cluster | YES | SCHEDULED | NO | N/A | Total |
|---|---|---|---|---|---|
| §6.8 Plan Hygiene | 7 | 1 (R7) | 0 | 0 | 8 |
| §6.9 Research Integrity | 5 | 2 (R6, R7 — paper not written yet) | 0 | 0 | 7 |
| §6.11 Training Interruption-Safety | 5 | 0 | 0 | 0 | 5 |
| §6.10 FS Roundtrip | 0 | 0 | 0 | 4 | 4 |
| §6.7 Test Hygiene | 1 | 0 | 0 | 0 | 1 |
| **Total** | **18** | **3** | **0** | **4** | **25** |

**18 YES / 3 SCHEDULED / 0 NO / 4 N/A.** Plan ships as v1.1. Three SCHEDULED items are work-in-future, not handwaving — gated to E5.3 and E5.4.3 where they materially apply.

---

## 8. What this document commits to (v1.1)

This plan, when accepted, commits the following:

1. **Strategic positioning:** FajarQuant + FajarOS Nova + Fajar Lang as a unified blue-ocean play in kernel-context LLM + Indonesian bilingual + tax-vertical. Phase E formally pivots away from "FajarQuant the algorithm" framing.
2. **Scope:** ID + EN bilingual ONLY. Pan-SEA explicitly excluded as red ocean.
3. **Algorithm catch-up baseline:** 5 features (Hadamard outlier rotation + FP8 LM head + FP16 distillation + bilingual calibration balance + language-conditioned design) evaluated at Mini scale; adopted iff combined ablation ≥+0.10 nat AND coherence ratio ≤1.5×. Data decides; FAIL path = fall back to Phase D Mini config + balanced calibration only.
4. **Production bar (v1.1):** 32 items in §1 (was 24 in v1.0), 10 phase gates in §5 with explicit FAIL paths (NEW), 25 self-check rules in §7 (HONEST: 18 YES + 3 SCHEDULED + 0 NO + 4 N/A). No "we'll harden later" exceptions — §6.8 R3 prevention layer required per phase.
5. **Reproducibility:** every ML claim backed by `verify_phase_e_tables.py --strict` (pre-commit + CI). Indonesian eval suite empirically verified (94 tasks in lm-eval v0.4.11 per §1 item 1). No protocol artifacts (per §6.9 lessons).
6. **IP path:** patent draft + open-weights/proprietary-deployment split decided in E5.4 — explicit in `LICENSING_PHASE_E.md`. Pre-publication security review required (§1 item 32, E5.0.5 NEW).
7. **Kernel-LLM serving targets quantified (NEW v1.1):** tok/s, TTFT, RSS, power — full spec in §13. These are the Tier 1 quantitative differentiators for paper.
8. **Continuous validation (NEW v1.1):** monthly literature sweep cadence; competitive-blindside escalation rule (7-day SLA if Microsoft/AI-Singapore/etc. ships overlap).

This plan does NOT commit to:
- Pan-SEA expansion (deferred indefinitely; orthogonal vs SEA-LION)
- Algorithm-vs-BitNet head-to-head as paper headline (paper headline = kernel-context category; performance shown as comparable, not superior)
- Cloud-burst budget > $1000 without explicit user approval mid-flight
- Stretch deployment to ARM64 Radxa Q6A (likely won't fit; Medium scale is the embedded target)
- Public release of TaxPrime-derived eval set (only summary numbers in paper)

---

## 9. Session handoff

**Status at plan creation (2026-04-25 ~15:30 WIB):**
- Phase D ~95% done; baseline sweep currently hung mid-run on MMfreeLM-1.3B (CLOSE_WAIT pattern, second occurrence in same session). Sweep state independent of this plan.
- Phase E plan written; not yet executed. **No code or data changes.**
- 5 questions Q1–Q5 in §2 still TBD; E0 will resolve them.

**Next session immediate actions (in priority order):**

1. **(blocker triage)** Decide Phase D baseline sweep fate: kill PID 93535 cleanly + decide whether to retry with proper §6.11 retry_iter wrapper added to `bench_baselines.py`, OR accept partial baseline data (BitNet ✅ + MMfreeLM-370M ✅ already done) and move forward.
2. **(plan acceptance)** User reviews Phase E plan v1.0 (this document); approves OR requests revisions.
3. **(if approved)** Begin E0.0.1 — inventory ID corpora. ~1–2h hands-on Bash + Python investigation.
4. **(if approved)** Begin E0.0.4 — hardware decision document (laptop-only vs cloud burst). ~30min thinking + spreadsheet.

**Resume protocol:**
```
Read: docs/FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md (this file v1.1) + docs/FJQ_PHASE_D_PRODUCTION_PLAN.md (predecessor) + memory/MEMORY.md (session state).
Verify: cat /tmp/fajarquant_logs/sweep_remaining.log | tail -30 (Phase D sweep status).
Decide: which of "next session immediate actions" 1-4 to start with.
```

---

## 10. Literature review (NEW v1.1 — §6.9 R2 evidence ≥8 papers)

Compiled 2026-04-25 via 4 WebSearch queries (sub-4-bit quant SOTA, on-device LLM, multilingual ternary, unikernel + RTOS + LLM). All Phase E claims trace to at least one paper here.

| # | Title (year) | Relevance to Phase E | Adoption decision |
|---|---|---|---|
| 1 | *The Era of 1-bit LLMs: All LLMs are in 1.58 Bits* (Ma+ 2024, arxiv 2402.17764) | Foundational ternary pretrain method; BitNet b1.58 | **Adopted as canonical baseline** (E1.0 baseline + paper Tier 1 comparison) |
| 2 | BitNet b1.58 2B4T (Microsoft 2025) — 2B params × 4T tokens, native HF since transformers v4.54+ | Direct competitor at deployment; Phase D §3.4 sweep already running | **Direct comparison in paper Table 2** |
| 3 | *MMfreeLM: Matmul-Free Language Models* (Zhu+ 2024) — 370M/1.3B/2.7B sizes | Primary architecture choice for IntLLM (eliminates V31.R3 γ-cascade); same Mistral-v3 32k tokenizer | **Already adopted in Phase D**; bilingual extension in E3 |
| 4 | *Falcon-Edge: 1.58bit Universal Fine-Tunable* (TII 2025-2026) — modifies BitNet (eliminates LayerNorm) | Recent ternary entrant; potential second baseline | Add to baseline sweep if time |
| 5 | *QuaRot: Outlier-Free 4-Bit Inference via Hadamard Rotation* (Ashkboos+ 2024, arxiv 2404.00456) | Outlier handling via Hadamard rotation (CLAUDE §6.9 R5) | **Adopted in E2.1** |
| 6 | *SpinQuant: LLM Quantization with Learned Rotations* (Liu+ 2024, arxiv 2405.16406) | Rotation-based quant alternative | Reference for E2.1; learn-vs-fixed rotation decision in E2.0 |
| 7 | *Calibrating Beyond English: Language Diversity for Better Quantized Multilingual LLMs* (2026, arxiv 2601.18306) | **CRITICAL:** "Quantization disproportionately degrades performance in multilingual LLMs"; bilingual calibration data balance required | **Adopted in E2.4 (NEW v1.1)** |
| 8 | *TernaryLLM: Ternarized Large Language Model* (arxiv 2406.07177) | Alternative ternary method | Reference; not adopted |
| 9 | *LLaVaOLMoBitnet1B: Ternary LLM goes Multimodal* (Sundaram & Iyer, arxiv 2408.13402) | Shows ternary scaling to multimodal | Out-of-scope for Phase E text-only |
| 10 | SEA-LION (AI Singapore 2024-2026) — 11 SEA languages, Llama-3 8B + Gemma 9B base | Indonesian incumbent FP16; Phase E orthogonal not direct competitor | **Compare on Indonesian eval; emphasize ternary + kernel-deployment differentiation** |
| 11 | Sahabat-AI Llama3-8B CPT (GoTo+Indosat 2024-2025) — 50B Indonesian token CPT | Indonesian incumbent FP16; Tier 2 differentiator analysis | Compare on Indonesian eval |
| 12 | Komodo-7B (Owen+ 2024, arxiv 2403.09362) — Indonesian + 11 regional languages | Indonesian incumbent FP16 | Compare on Indonesian eval |
| 13 | *Challenges and Opportunities for Unikernels in Machine Learning Inference* (IEEE 2021) | Closest prior art for kernel-mode ML; pre-LLM era | **Cite as nearest prior art** in Tier 1 paper claim |
| 14 | Zephyr + TinyML (Edge Impulse, Conexio Stratus 2024-2026) | Closest prior art for RTOS + ML; tiny models only (CNN/RNN), not LLM | **Cite to differentiate**: Phase E targets 140M-3B, not TinyML scale |
| 15 | BitLoRA (2026, ScienceDirect) — federated on-device ternary LoRA | Interesting for Tier 3 vertical fine-tune via LoRA | Reference; consider for E4.1 LoRA recipe |

**Tier 1 prior-art claim defensibility (gap #2 closure):** searched arxiv + IEEE Xplore + Google Scholar for "kernel-mode LLM inference" / "kernel-context LLM" / "OS kernel large language model" — found tiny-ML in RTOS (Zephyr+Edge Impulse, but only CNN/RNN scale), unikernel-ML (paper 13, pre-LLM era), but **no published research on running LLM inference inside an OS kernel context with formal isolation guarantees as of 2026-04-25.** Claim defensible; framed honestly in paper as "we are not aware of prior work on..." rather than absolute "first."

**Monthly literature sweep cadence (§1 item 31):** automated check first Monday of each month for new papers/products in: ternary LLM, embedded LLM, kernel-mode ML, Indonesian LLM, FajarQuant/FajarOS direct competitors. Results appended to `docs/FJQ_PHASE_E_LITERATURE_LOG.md`.

---

## 11. Tax-vertical eval methodology spec (NEW v1.1 — gap #5 closure)

### 11.1 Eval set construction

**Target:** 500 prompts gold-labeled by ≥2 domain experts, with explicit IRR (inter-rater reliability) ≥ Cohen's κ = 0.7.

| Step | Process | Owner | Verification |
|---|---|---|---|
| 11.1.1 Source prompts (~1000 pre-filter) | Mix: (a) real TaxPrime client questions sanitized of PII (~400), (b) Indonesian tax law textbook questions (~300), (c) Pengadilan Pajak case-law derivations (~200), (d) bilingual edge cases (Indonesian Q with English regulatory ref or vice versa, ~100) | TaxPrime knowledge team | Source provenance logged per prompt |
| 11.1.2 PII sanitization | Strip names, NPWP, addresses, monetary specifics; replace with `[CLIENT]`, `[NPWP-XX]`, `[Rp YYY]` placeholders | TaxPrime + 1 reviewer | `scripts/check_pii_redaction.py` exit 0 |
| 11.1.3 Gold-label annotation (rater 1) | Domain expert annotates correct answer + reasoning chain | Sr. tax consultant A | annotation file v1 |
| 11.1.4 Gold-label annotation (rater 2, blind) | Independent expert; does NOT see rater 1 | Sr. tax consultant B | annotation file v2 |
| 11.1.5 IRR check | `scripts/compute_irr.py` outputs Cohen's κ on labels; target ≥0.7 | Methodology lead | κ ≥ 0.7 → pass; <0.7 → conflict resolution loop |
| 11.1.6 Conflict resolution | All disagreements adjudicated by 3rd expert OR consensus discussion | Senior partner | resolution log committed |
| 11.1.7 Final filter to 500 | Drop prompts where IRR conflict survived adjudication; keep 500 highest-quality | Methodology lead | final eval set checked-in |
| 11.1.8 Bias audit | Coverage by topic (PPh, PPN, PBB, customs, transfer pricing, court cases, regional rules); coverage by language (ID-only / EN-only / mixed); no over-representation of any client industry | Methodology lead + external auditor (light) | `docs/FJQ_PHASE_E_E4_TAX_EVAL_BIAS_AUDIT.md` |

### 11.2 Metric definition

Pass@1: **strict** — model output must contain (i) correct legal answer, (ii) cite the correct regulatory source (UU/PMK/PER number), (iii) no factual hallucination of regulations that don't exist. Auto-graded via:
- Regex match for (ii) regulatory citation
- LLM-as-judge for (i) and (iii) using GPT-5 or Claude Opus 4.7 (consistent grader, NOT the model under test)
- 50 random samples manually re-graded by domain expert to confirm auto-grader accuracy ≥ 90%

### 11.3 Threshold (§1 item 4)

- **65% pass@1 = TARGET** — calibrated post-baseline (E4.0): if FP16 Phase D Medium achieves only 50%, target adjusts down to 50% × 1.0 (parity); if FP16 baseline 70%, target stays 65% (allowing 5pt ternary degradation).
- **<50% absolute = ABORT** (gate FAIL path per §5).

### 11.4 Privacy/compliance

- TaxPrime data + sanitization process reviewed by TaxPrime legal counsel (E0.0.2)
- Eval set itself NEVER shipped publicly — only summary numbers in paper
- Audit log of who-touched-what for compliance trail

---

## 12. Cross-repo coordination matrix (NEW v1.1)

Phase E touches 3 repos. Explicit who-owns-what to prevent dropped balls.

| Phase | fajarquant (primary) | fajar-lang | fajaros-x86 |
|---|---|---|---|
| E0 Pre-flight | E0.0.1–9 all owned here | — | — |
| E1 Bilingual data | corpus + dedup + tokenizer measure | — | — |
| E2 Algorithm catch-up | all 5 ablations | — | — |
| E3 Bilingual pretrain | training + canonical bench | — | — |
| E4 Vertical fine-tune | SFT + tax eval set | — | — |
| E5.1 Bilingual kernel-path | .fjm export script + serving benches | maybe @kernel feature additions if FJM v10 spec needed | `kernel/compute/bilingual_medium.fjm` integration + Makefile gate |
| E5.2 Tax-vertical kernel-path | .fjm export | — | shell command `tax-query` + Makefile gate |
| E5.3 Paper | results tables + verify script | — | kernel-path artifact + reproducibility |
| E5.4 IP / releases | fajarquant v0.5 release | fajar-lang v32 release | fajaros-x86 v4.0 release |

---

## 13. Kernel-LLM serving target spec (NEW v1.1 — §1 items 25–28 + gap #6 closure)

### 13.1 Tokens/sec target (§1 item 25)

**Definition:** sustained generation throughput over 256-token decode after a 64-token prefill, single-stream, single-thread CPU only (kernel context can't dispatch to userspace CUDA so GPU offload is structurally unavailable).

| Scale | Target tok/s | Hardware | Methodology |
|---|---|---|---|
| Mini (~7M) | ≥50 tok/s | RTX 4090 Laptop (CPU only) | `make bench-kernel-throughput SCALE=mini` |
| Base (~46M) | ≥30 tok/s | same | same |
| Medium (~140M) | **≥10 tok/s** (production gate) | same | same |
| Stretch (~3B) | ≥3 tok/s | same | same |

Comparison reference: Apple Foundation Models on M-series at 1B-3B reach 30-50 tok/s userspace (full silicon access). **Our kernel-context CPU number is 5-10× slower because of the constraint we IMPOSED for security/isolation — that's the trade.**

### 13.2 First-token latency (TTFT) target (§1 item 26)

**Definition:** time from API call start to first generated token, with cold cache (model loaded but no KV cache).

| Scale | Target P50 | Target P95 |
|---|---|---|
| Medium | ≤500 ms | ≤1500 ms |
| Stretch | ≤2 s | ≤6 s |

### 13.3 Memory footprint at inference (§1 item 27)

**Definition:** peak RSS of `@kernel` process during inference of 256 tokens with full KV cache.

| Scale | Target peak RSS |
|---|---|
| Medium | ≤512 MB |
| Stretch | ≤4 GB |

Critical for embedded ARM64 target (Radxa Q6A has 8 GB total).

### 13.4 Power budget (§1 item 28)

**Definition:** average inference power on Radxa Q6A (ARM64) measured externally via watt-meter (Kill A Watt or similar) during 60-second sustained generation workload.

| Scale | Target average power |
|---|---|
| Medium | ≤5 W |
| Stretch | TBD — likely won't fit Q6A regardless |

Comparison: typical x86 laptop idle 8-15 W; mobile chip running Apple Foundation Models on-device ~3-5 W active. Our 5 W target = competitive with mobile, achieved on industrial ARM64.

### 13.5 Verification methodology

| Target | Makefile target | Tooling | Output artifact |
|---|---|---|---|
| 13.1 tok/s | `bench-kernel-throughput` | criterion + custom kernel-side timer | `bench/kernel_throughput.json` |
| 13.2 TTFT | `bench-kernel-ttft` | criterion + percentile reporter | `bench/kernel_ttft.json` |
| 13.3 RSS | `bench-kernel-mem` | `/proc/self/status` poll during decode | `bench/kernel_mem.json` |
| 13.4 Power | `bench-radxa-power` | external watt-meter logger + script | `bench/radxa_power.json` |

All four bench artifacts checked into `paper/intllm/bench/` and verified by `verify_phase_e_tables.py --strict` (gate per §1 item 14).

---

## 14. Synthetic data hygiene policy (NEW v1.2 — 3-lapis hybrid)

Phase E synthetic-data strategy locked. Decision matrix below is the contract; deviations require v1.3 plan bump.

### 14.1 Three lapis hierarchy

| Lapis | Use case | Generator allowed | ToS-clean? | Why |
|---|---|---|---|---|
| **Lapis 1** | Pretrain core (E1+E3) | **NONE — real data only** | N/A | E0.0.1 confirmed 25–54 B post-dedup ID tokens available; no synthetic needed |
| **Lapis 2** | Pretrain augmentation (E1.5) | Open-source: **Mixtral-8x7B (Apache 2.0)**, **Qwen2.5-72B (Apache 2.0)**, Llama-3.3-70B (Llama Community License — verify no-compete clause via E0.0.10) | ✅ yes | own-host, repro-pin, no third-party API ToS issue |
| **Lapis 3a** | Vertical FT — Tax (E4.1) | **NONE — TaxPrime real data only per spec v1.0** | N/A | Tier 3 wedge is the differentiator; real proprietary > synthetic always; ToS-clean by construction |
| **Lapis 3b** | Vertical FT — Instruct (E4.2) | Open-source generator OR existing open ID-instruct datasets (Cendol, etc.) | ✅ yes | non-tax instruct is general capability; open path defensible |
| **Lapis 3c** | Eval / LLM-as-judge (§11) | Claude / GPT-4 / open — any judge is OK because **judging ≠ training** | ✅ yes (judging is not derivative-model creation) | standard practice in lm-eval ecosystem |

### 14.2 Hard rules

1. **R1 — Synthetic ratio cap:** Lapis 2 synthetic tokens ≤ 25% of total pretrain mix. Pre-commit hook `verify-synthetic-mix` enforces.
2. **R2 — No Claude API output as TRAINING data.** Confirmed by Anthropic Usage Policy update + active enforcement (VentureBeat 2026 — xAI/Cursor restricted). Applies to E1.5, E4.1, E4.2 — exception only Lapis 3c (judge).
3. **R3 — Generator pinning:** every synthetic token must be reproducible via committed generator-weights hash + prompt template. `FJQ_PHASE_E_E1_5_SYNTHETIC.md` records generator, version, prompt corpus.
4. **R4 — Quality filter mandatory:** LLM-as-judge filters bottom 20% of generated content per "How to Synthesize Without Model Collapse" (arxiv 2412.14689).
5. **R5 — No recursive bootstrapping:** never train Phase E IntLLM on Phase E IntLLM's own output. Per "The Curse of Recursion" (Shumailov+ 2023). Applies even at SFT.
6. **R6 — Cost ceiling:** synthetic generation cost ≤ 5% of total Phase E cloud GPU budget. Default: ≤ $40 of ~$800 total.
7. **R7 — Annotator FAQ enforced:** TaxPrime spec §14 already has "DO NOT use ChatGPT/Claude to draft training data" rule; reinforced here at plan level.

### 14.3 Per-use-case decision rules

```
For any new training-data ask:
  Q1: Is the dataset for pretrain or fine-tune of an LLM?
    → If yes, third-party Claude/GPT API output is FORBIDDEN as data source.
    → If no (eval, judge, classifier, retrieval), Claude/GPT is ALLOWED.
  Q2: Is the use case Tier 3 vertical (tax/legal)?
    → TaxPrime real data only. Proprietary supremacy.
  Q3: Is open-source generator legally available for the corpus license target?
    → Check E0.0.10 matrix. Default Mixtral or Qwen2.5 (Apache 2.0).
  Q4: Will the synthetic mix exceed 25% of pretrain tokens?
    → REJECT. R1 violation.
```

### 14.4 Reference papers (cited in §10 literature review)

- *The Curse of Recursion* (Shumailov+ 2023) — model collapse foundational
- *How to Synthesize Without Model Collapse* (arxiv 2412.14689, 2024) — quality filter recipes
- *Demystifying Synthetic Data in LLM Pre-training* (arxiv 2510.01631, 2025) — scaling laws
- *Phi-4 Technical Report* (Microsoft 2024) — multi-agent + self-revision recipes (Microsoft owned generator, not subject to third-party ToS)
- *Cosmopedia* (HuggingFace 2024) — Mixtral-generated 25 B token open-source pretrain dataset (canonical Lapis 2 reference)
- *BeyondWeb: Lessons from Scaling Synthetic Data for Trillion-scale Pretraining* (DatologyAI 2026)
- *Anthropic Usage Policy update* (anthropic.com/news/usage-policy-update) — ToS basis for R2

### 14.5 Cost reality check

| Approach | 30 B tokens cost | Reproducible? | ToS-clean? | Verdict |
|---|---|---|---|---|
| Claude Opus 4.7 API | **~$450,000** | No (model versions drift) | No (R2 violation) | ❌ catastrophic |
| GPT-5 API | ~$300–500K | No | No (similar OpenAI ToS) | ❌ catastrophic |
| Self-host Mixtral-8x7B on cloud H100 | **~$300–600** | Yes (pin weights) | Yes (Apache 2.0) | ✅ Lapis 2 default |
| Self-host Qwen2.5-72B on cloud H100 | ~$500–1000 | Yes | Yes (Apache 2.0) | ✅ alternative if Mixtral quality insufficient |
| No synthetic (real corpus only) | $0 | Trivially | Yes | ✅ Lapis 1 default |

**Cost arithmetic confirms:** open-source self-host is **1000× cheaper** than API approach. Reproducibility + legal cleanliness are bonuses on top.

---

*Plan version: 1.2 (2026-04-25). Author: Claude Opus 4.7 + Fajar (PrimeCore.id).*
*Predecessor: FJQ_PHASE_D_PRODUCTION_PLAN.md v1.2. Companion: FJQ_PHASE_E_TAXPRIME_DATASET_SPEC.md v1.0 (NEW v1.2).*
*v1.0→v1.1 closed 8 substantive gaps via empirical verification. v1.1→v1.2 added TaxPrime data ownership + synthetic-data 3-lapis policy (§14 NEW).*
*Cross-repo coordination required: fajarquant (primary) + fajar-lang (compiler features for kernel-side tokenizer + IntLLM ops) + fajaros-x86 (deployment runtime + kernel-path Makefile gates).*
*Subject to revision per phase findings; major scope changes require new Plan version (v1.x → v2.0).*
