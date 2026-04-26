# FajarQuant Phase E — Bilingual Kernel-LLM Production Plan (100% Production Bar, Tier 1+2 Scope)

> **Plan version:** 1.8 (2026-04-26 v1.8 patch — Tier 3 tax-vertical DEFERRED to Phase F; Tier 1+2 only)
>
> **v1.7 → v1.8 changelog (2026-04-26 same-session, post-E1.4+E1.2 closure):**
> - **🎯 Strategic re-scope: Tier 3 tax-vertical (TaxPrime) DEFERRED to Phase F.** Phase E v1.8 commits to **Tier 1 (kernel-context LLM) + Tier 2 (Indonesian + English bilingual ternary) ONLY**. Three drivers:
>   - (1) **Founder bandwidth honesty** — solo execution + day-job + V31.E1 in flight. Tier 3 was 190-270 person-hours of TaxPrime archive ingest + dataset assembly + eval set construction; deferring frees calendar for clean bilingual base ship.
>   - (2) **Cleaner research narrative for first paper** — "Bilingual ID+EN ternary LLM in kernel context" stands alone as MLSys-2027-grade contribution; tax-vertical was the third leg of the three-tier story but adds NDA dependency + single-rater methodology caveat. Drop strengthens reproducibility (all-public-corpus) and external validity (no proprietary archive).
>   - (3) **De-risk first artifact** — E1.3 TaxPrime archive ingest was founder-blocked (NDA review pending). E4 entire phase (4 weeks) blocked on E1.3. Deferring removes the largest dependency risk in the calendar.
> - **Tier 3 work PRESERVED** as Phase F roadmap, not deleted:
>   - `docs/FJQ_PHASE_F_TAX_VERTICAL_ROADMAP.md` NEW (v1.8) — pointer document tying together preserved companion specs.
>   - `docs/FJQ_PHASE_E_TAXPRIME_DATASET_SPEC.md` v1.1 retained as authoritative spec for Phase F (header updated to reflect deferral).
>   - §11 Tax-vertical eval methodology preserved in this plan with "DEFERRED — Phase F" header (not deleted; future-self can lift verbatim).
> - **Calendar reduction:** 52-65 weeks (12-15 months solo per v1.6) → **36-44 weeks (8-10 months solo)**. Drops:
>   - Phase E4 (Vertical fine-tune, 4 weeks) — entire phase deferred.
>   - E1.3 TaxPrime ingest in Phase E1 (~5h human) — deferred.
>   - Tax-vertical paper claims + tax patent prep in E5 (~8-12 weeks calendar overhead) — deferred.
> - **Production bar (§1):** 32 items → **~26 items**. Stripped: item 4 (vertical eval pass@1), item 11 (test-tax-vertical-kernel-path), tax claims in items 24+30. Renumbering preserved with audit-trail "(deferred to Phase F)" tags.
> - **Phase E1 sync to v1.8 status (closed via separate commits):**
>   - E1.1 ID first slice CLOSED (`47a8892` chain) — 10.67M docs / 38.92 GB raw.
>   - E1.2 EN reuse CLOSED (`f912054`+`e82b1c4`) — `intllm_en.py` shim, 60:40 ratio, Stretch repo.
>   - E1.4 dedup CLOSED (`4ebb437`+`8dc4fab`) — exact-hash sha256, 15.40 B post-dedup tokens (1.92× over §2 abort threshold).
>   - E1.3 DEFERRED (this plan revision).
>   - E1.5 DEFERRED (carried over from v1.7).
>   - E1.6 packaging scope reduced: bilingual base attribution + license file only (no tax-vertical attribution path).
> - **Risk register (§6) updated:** TaxPrime-related risks → RESOLVED via Phase F deferral. New risk: "Tier 3 deferral cannibalizes paper differentiator vs SEA-LION / Sahabat-AI" (Med likelihood, Med impact, mitigated by kernel-context Tier 1 differentiator + ternary bilingual being unique combo).
> - **Companion docs:** TaxPrime dataset spec header updated to "Phase F roadmap" framing; body unchanged.
> - **Self-check (§6.8 R7 artifact sync):** README badges + GitHub Releases unaffected (Phase E hasn't released yet); MEMORY.md resume protocol updated separate commit; FJQ_PHASE_E_E1_FINDINGS.md v1.2 already references E1.3 deferred.
>
> **v1.6 → v1.7 changelog (2026-04-26 same-day):**
> - **🎯 Cash budget zero:** Phase E total external cash cost = **$0**. Founder self-funds all compute (laptop) + self-reviews privacy (founder is qualified counsel SH., MH.) + skips DVC/S3 cloud (plain local filesystem).
> - **DVC + S3 ap-southeast-1 DROPPED** (E0.0.8 simplified): plain `data/` directory structure on laptop. Reproducibility now via methodology spec + public corpus pin via HF Datasets revision SHA + git-committed result JSONs. Saved ~$105/yr + 30-60 min AWS setup. Risk accepted: laptop hardware fail = 4-7 month restart; mitigated via methodology-equivalent recreation possible from public corpus + pinned generator.
> - **External privacy counsel DROPPED** (Spec §9 + §11.2 simplified): founder is SH., MH. + 14yr TaxPrime practice = qualified Indonesian privacy counsel. Founder self-reviews UU PDP compliance. Risk accepted: defensibility slightly weaker than independent review; mitigated via founder credentials + spec methodology disclosure.
> - **Lapis 2 (synthetic augmentation E1.5) DEFERRED TO PHASE F** per Option B: real corpus alone (25-54 B post-dedup ID + EN per E0.0.1) is sufficient. Phase D Mini/Base/Medium succeeded real-only — empirical proof. Lose BeyondWeb 7.7× speedup; calendar +0 (vs +3 mo for laptop-Gemma-4-9B option A). Methodology defensible — paper acknowledges "Lapis 2 synthetic augmentation deferred to future work for resource reasons."
>   - E1.5.0 prompt templates v1.0 (`python/phase_e/prompts/LAPIS2_TEMPLATES.md`) PRESERVED as Phase F future-reference. Annotated as "DEFERRED" in template header.
>   - E1.5.5/E1.5.6 Claude filter+audit DEFERRED with the rest.
>   - Lapis 2.5 (Claude meta-design role) NOT NEEDED for v1.7 (no synthetic to filter); footnote retained for Phase F.
> - **Cloud-burst rescue path REMOVED** from §6 risk register: previously "$300-500 emergency" — now founder accepts hardware-fail restart cost as Tier 3 risk + uses external SSD for periodic backup (founder's choice, not formally required).
> - **§4 calendar:** total external cost $300-700 → **$0**. Calendar duration unchanged at ~13.5 months solo (Lapis 2 dropped saves no time net since it was a parallelizable sub-task).
> - **§14.5 cost reality check** simplified to single-row: "Phase E v1.7 total = $0."
>
> **v1.5 → v1.6 changelog (2026-04-26):**
>
> **v1.5 → v1.6 changelog (2026-04-26 same-day):**
> - **Reality update:** founder confirmed solo execution — no TaxPrime team available for parallel dataset assembly. All 190-270 person-hours of Tier 3 work fall on the founder alongside engineering + day-job.
> - **Tier 3 (tax-vertical) SCOPE-DOWN per Option A** (strategic decision 2026-04-26):
>   - Eval set: 500 prompts → **100-200 prompts**
>   - Training corpus: 5K-50K examples → **1K-5K examples**
>   - Methodology: double-rater Cohen κ ≥ 0.7 → **single-rater + self-consistency calibration runs**
>   - External bias auditor: removed (informal self-check)
>   - Paper transparency: claims framed as *"preliminary single-expert-rated evaluation"*
> - **§11 reference now points to TaxPrime spec v1.1** (solo-execution methodology — companion patch).
> - **§4 calendar EXTENDED:** 34 weeks (with-team assumption) → **~52-65 weeks (12-15 months solo, 10h/week realistic)** — accommodates founder bandwidth across engineering + Tier 3 + day-job.
> - **§6 risk register:** "TaxPrime team unavailable" RESOLVED via solo-mode adopt; **"founder bandwidth burnout"** added (Med likelihood, High impact); **"single-rater eval validity"** added (Med likelihood, Med impact, mitigation = explicit paper transparency + self-consistency calibration).
> - **§1 item 4** updated with solo-rater caveat: pass@1 ≥ 65% target retained but flagged "preliminary" until external validation in future Phase F.
> - **§13 serving target validation** unaffected (kernel-path tests still solo-runnable on laptop).
>
> **v1.4 → v1.5 changelog (2026-04-26):**
>
> **v1.4 → v1.5 changelog (2026-04-26):**
> - **§14.1 Lapis 2.5 NEW** — Claude Opus 4.7 admitted as **quality filter + audit + meta-design** role for synthetic data pipeline. Categorically distinct from Lapis 2 generation (ToS-blocked) — Claude here judges/audits/designs, never produces training tokens.
> - **§14.2 R2 expanded with explicit boundary clarification:** judging/filtering/auditing/prompt-design = OK; rewriting/expanding/translating/generating = NOT OK. Bright-line distinction added to plan-level hard rules.
> - **§3 E1.5 sub-tasks restructured:**
>   - E1.5.0 NEW: Claude-assisted BeyondWeb prompt-template engineering (meta-design — Claude doesn't see corpus, designs prompts only).
>   - E1.5.5 updated: quality filter using Claude Opus 4.7 (was Gemma 4 self-filter; Claude expected ~5-15% better discrimination per Indonesian-fluency benchmarks).
>   - E1.5.6 NEW: Claude audit pass on top-quality synthetic batch (5-10% sample for hallucination/bias/factual-error detection).
> - **§14.5 cost reality check** updated with Claude filter cost row (~$50-100 for 5-10B token rating with prompt caching). Total Phase E synthetic-augmentation cost: ~$1050-2100 (Gemma 4 generation $1000-2000 + Claude filter $50-100).
> - **§14.4 reference papers** add: ToS-clean precedent for LLM-as-judge usage in pretrain pipelines.
>
> **v1.3 → v1.4 changelog (2026-04-25 same-day patch):**
>
> **v1.3 → v1.4 changelog (2026-04-25 same-day patch):**
> - 🎉 **Phase E E0 (pre-flight) phase: 100% CLOSED** — all 10 sub-tasks delivered with mechanical decision files. See §3 status updates per sub-task.
> - **Hardware decision (E0.0.4 closed):** **LAPTOP-ONLY RTX 4090** strategy adopted by founder. No cloud burst budgeted. Cost saved ~$420–820 vs plan v1.3; trade-off = +6–8 weeks calendar (28 → 34 weeks). See `FJQ_PHASE_E_HARDWARE_DECISION.md`.
> - **Stretch replan (per E0.0.4):** option (C) **reduced Stretch ~1.5B × 15B tokens** (~7-10 days laptop) replaces plan v1.3 cloud-burst Stretch. Preserves Mini→Base→Medium→Stretch monotonic narrative; loses ~50% scale ambition.
> - **Two-tier deployment scoping (E0.0.7 closed):** Tier A (paper artifact) uses ALL sources; Tier B1 (commercial-clean) drops CC-100 + news + academic-restricted. Both prepared in parallel. Tier C (Indonesian-market commercial) defensible via UU Hak Cipta Pasal 43-44 fair-use-like for tax-vertical public-interest carve-out. See `FJQ_PHASE_E_E0_LICENSING.md`.
> - **E1.1.1 NEW sub-task:** Tier B1 commercial-clean corpus subset assembly (light fork of E1.1; ~+10% data curation effort).
> - **§4 calendar revised:** 28 weeks (cloud) → **34 weeks laptop-only with reduced Stretch** OR **28 weeks Medium-only**.
> - **§6 risk register:** "GPU cost overrun" RESOLVED (laptop-only). "Laptop hardware SPOF" added (Low likelihood, High impact, mitigation chain). "CC-100 commercial license" added. "Wikipedia share-alike commercial" added.
> - **§13.1 serving targets:** unchanged on metrics, but E5.1.6-9 measurement now done on laptop instead of cloud (no impact on numerical targets — kernel-context CPU only anyway).
>
> **v1.2 → v1.3 changelog (2026-04-25):**
>
> **v1.2 → v1.3 changelog (2026-04-25 same-day patch):**
> - **§14 Lapis 2 recipe correction:** explicit BeyondWeb-style **REPHRASE real text** (not generate-from-scratch). Cite arxiv 2508.10975 (DatologyAI 2026) — 50:50 real:rephrase synthetic = SOTA, 7.7× faster training than open web alone, 2.7× faster than Nemotron-Synth.
> - **Generator list CORRECTED based on fresh license research:**
>   - ✅ ADDED **Gemma 4** (Google, Apache 2.0 — *unrestricted commercial use, fine-tuning, derivative models*) as primary clean generator
>   - ✅ Mixtral-8x7B-Instruct retained (Apache 2.0)
>   - ⚠️ Qwen2.5 caveat: weights Apache 2.0, but **outputs custom-licensed — <100M MAU cannot train other LLMs besides Qwen/derivatives**. TaxPrime/PrimeCore.id under threshold → restricted. Verify exact ToS in E0.0.10.
>   - ❌ DROPPED **Llama-3.3** — Llama Community License **forbids using Llama outputs to train other LLMs (besides Llama derivatives)** for any party with <700M MAU. Was previously listed with caveat in v1.2 — now confirmed blocker.
> - **Fair-use defense citation added:** Bartz v. Anthropic (N.D. Cal. June 2025) — LLM training on copyrighted works = "quintessentially transformative" fair use. Caveat: pirated source material = no fair-use defense ($1.5B Anthropic settlement).
> - **Cost numbers corrected** for Claude Opus 4.7 — output **$25/MTok** (not $15 as v1.2 estimate); 100B token full ID+EN scope = **$1.25M (Batch API) to $3.4M (worst case)**, 1500–5000× more expensive than open-source rephrase path.
> - **§14.2 R8 NEW hard rule:** synthetic content must be REPHRASE-of-real, not generate-from-scratch (BeyondWeb canonical recipe).
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

**Two uncontested niches in Phase E v1.8 scope (BLUE OCEAN):**

1. **Tier 1 — Kernel-context LLM with formal isolation.** No other stack can run an LLM inside an OS kernel with compiler-enforced `@kernel`/`@device` separation + SMEP/SMAP/NX security triple. Use cases: automotive ECU AI, aerospace, medical device, industrial control.
2. **Tier 2 — Indonesian + English bilingual ternary embedded.** Three orthogonal axes (ID, ternary, embedded) — no current player covers all three. SEA-LION etc. are all FP16-and-CPT, none ternary, none target embedded.

**Phase F roadmap (deferred from Phase E v1.8 per 2026-04-26 pivot):**

3. **Tier 3 — Vertical AI (tax/legal Indonesian) with audit-trail-proof provenance.** TaxPrime native expertise + Phase D reproducibility chain. **DEFERRED:** preserved as Phase F roadmap (`docs/FJQ_PHASE_F_TAX_VERTICAL_ROADMAP.md` + `FJQ_PHASE_E_TAXPRIME_DATASET_SPEC.md` v1.1 + §11 below). Activation gated on Phase E base success + founder-bandwidth re-check.

This plan ships **Tier 1+2** concurrently, with shared substrate. **ID+EN bilingual scope confirmed** (vs pan-SEA): ~50–60% smaller corpus, 3–5× simpler eval suite, cleaner ternary quantization, and ortho-positioning vs SEA-LION. Tier 3 follows as natural Phase F extension once Tier 1+2 are validated, not a co-dependency.

---

## 1. Production-readiness bar (~26 active items, v1.8 — Tier 3 items deferred to Phase F)

Borrowed pattern from Phase D §1, expanded for Phase E scope. **v1.1 added items 25–32** for serving targets, abort criteria, monthly literature sweep, security review. **v1.8 marks items 4, 11, 24-tax, 30-tax as `[DEFERRED — Phase F]`**: numbering preserved (audit-trail integrity per §6.8 R7), bar count drops 32 → ~26 active. Strikethrough'd items remain visible for forensic reference; they are NOT counted toward Phase E v1.8 ship gates.

### A. Benchmark claims (CLAUDE §6.9 R1, R6)

1. **Indonesian eval suite (EMPIRICALLY VERIFIED 2026-04-25, lm-eval v0.4.11, 94 strict-ID tasks present):**
   - **Core ID set (4 tasks):** `arc_id`, `belebele_ind_Latn`, `global_mmlu_id` (with 6 sub-categories: business / humanities / medical / other / social_sciences / stem), `m_mmlu_id`
   - **Extended ID set (~30+ subtasks):** `mmmlu_id_id_*` (Indonesian MMLU full subject breakdown — abstract_algebra, anatomy, astronomy, ..., college_*, high_school_*), `include_base_44_indonesian_*` (INCLUDE benchmark Indonesian, with few_shot_en/og variants)
   - Pin: lm-eval `0.4.11` (full 40-char SHA recorded in `eval/HARNESS_SHA` per Phase D pattern)
2. **English eval parity (canonical, same harness SHA):** MMLU (5-shot), HellaSwag, ARC-easy/challenge, PIQA, WinoGrande, BoolQ, OpenBookQA, lambada_openai, wikitext, triviaqa.
3. **Bilingual coherence gate:** ID and EN perplexity at Medium scale **within 1.3× of each other** (looser than v1.0's 1.2× because Indonesian web text noisier; calibrated empirically post-E3.1 results, recorded in `FJQ_PHASE_E_E3_BILINGUAL_GATE_CALIBRATION.md`).
4. **`[DEFERRED — Phase F]` Vertical eval (tax/legal Indonesian).** ~~**100-200**-prompt gold-labeled eval set per `FJQ_PHASE_E_TAXPRIME_DATASET_SPEC.md` v1.1...~~ Methodology preserved in §11 below + Phase F roadmap; not a Phase E v1.8 ship gate.
5. **Canonical baselines re-run on bilingual axis:** BitNet-2B4T, MMfreeLM-1.3B/2.7B, SEA-LION-8B-IT, Sahabat-AI-8B-Llama3-CPT, Komodo-7B-Instruct (all on ID core set + EN canonical set, same harness SHA). ~~tax-vertical set~~ deferred with item 4.

### B. Baseline parity (CLAUDE §6.9 R3)

6. **Outlier handling ported.** Hadamard rotation (à la QuaRot/SpinQuant) applied to attention output + LM head per literature consensus. Ablation with/without committed.
7. **Mixed-precision LM head + attention output.** FP8 (E4M3) for these layers, ternary for rest. Ablation result vs all-ternary.
8. **FP16 distillation during QAT.** Teacher = FP16 IntLLM Medium English baseline, student = bilingual ternary. Ablation vs no-distillation.
9. **Sub-byte tokenizer parity.** Mistral-v3 32k retained as primary tokenizer (decision deferred to E2 with empirical Indonesian compression measurement). Custom 24k optional in v2.

### C. Kernel deployment gates (research-OS bare-metal pattern)

10. **`make test-bilingual-kernel-path`** — FajarOS Nova boots, loads bilingual IntLLM Medium, runs both ID and EN inference inside `@kernel` context, 4 invariants (ID-PPL ≤ threshold, EN-PPL ≤ threshold, no `@device` calls in kernel path, security triple still on).
11. **`[DEFERRED — Phase F]` `make test-tax-vertical-kernel-path`.** ~~Same as item 10 but for vertical fine-tuned model, plus 1 invariant: pass@1 on TaxPrime smoke set inside `@kernel`.~~ Reactivated in Phase F after vertical FT lands.
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
24. **Patent prep filing draft** for kernel-context LLM execution model (TBD legal partner — external counsel; ~~TaxPrime IP team option~~ deferred with Phase F since Tier 3 was the natural TaxPrime IP-team engagement vector).

### G. Serving targets (Tier 1 quantitative differentiators — NEW v1.1 §13 spec)

25. **Tokens/sec on x86 in `@kernel` context:** ≥10 tok/s for Medium (~140M params) on RTX 4090 Laptop CPU (no GPU offload — kernel context can't dispatch to userspace CUDA). **Stretch on Stretch (~3B): ≥3 tok/s.** Measurement methodology in §13.
26. **First-token latency (TTFT):** ≤500 ms for Medium scale, ≤2 s for Stretch. Captured in QEMU + bare-metal Lenovo Legion Pro.
27. **Memory footprint at inference (peak RSS in `@kernel`):** Medium ≤512 MB, Stretch ≤4 GB. Critical for embedded ARM64 target (Radxa Q6A has 8 GB).
28. **Power budget (embedded target):** Medium ≤5 W average inference power on Radxa Q6A (ARM64). Measured via external watt-meter, methodology §13.

### H. Abort criteria (NEW v1.1 — §6.8 R6 mechanical FAIL paths)

29. **Per-phase abort gate:** every phase has explicit FAIL conditions (§5 table updated). Failing a gate triggers `FJQ_PHASE_E_E*_ABORT.md` decision doc + rollback / pivot path, NOT silent extension.
30. **Phase E global kill switch (v1.8):** if Q1 Indonesian corpus < 8B usable tokens AND bilingual coherence catastrophic at Mini scale → pause Phase E, write `FJQ_PHASE_E_PIVOT.md`, return to fundamental scope decision. ~~AND TaxPrime data fully blocked~~ removed v1.8: Tier 3 already deferred to Phase F, no longer a Phase E gating condition.

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

### v1.4 status update — ALL Q1–Q6 CLOSED 🎉

All E0 pre-flight questions resolved 2026-04-25 same-day. Detailed answers in respective decision docs (`FJQ_PHASE_E_E0_*.md`).

- Q1: How much high-quality Indonesian text is realistically obtainable from public sources? ✅ **CLOSED v1.4 (E0.0.1):** 25–54 B post-dedup ID tokens available across 9 sources. 3-7× over abort threshold. See `FJQ_PHASE_E_E0_FINDINGS.md` (forthcoming aggregation) + `paper/intllm/results/phase_e/e0_id_corpora_inventory.json`.
- Q2: ✅ **CLOSED v1.2 → DEFERRED Phase F per v1.8.** TaxPrime data ownership confirmed by Fajar (founder); dataset creation per `FJQ_PHASE_E_TAXPRIME_DATASET_SPEC.md` v1.2 (header marks Phase F roadmap). v1.8 strategic re-scope: Tier 3 entirely deferred; Q2 is no longer a Phase E gating concern.
- Q3: Tokenizer compression efficiency. ✅ **CLOSED v1.4 (E0.0.3):** Mistral-v3 32K kept. Indonesian bytes/token = 2.527 (≤ 3.5 efficient threshold). Important nuance: tokens/word ID = 2.948 vs EN 1.665 (+77% more tokens per word due to Indonesian morphology). Custom 24K deferred to v2. See `FJQ_PHASE_E_E0_TOKENIZER_DECISION.md`.
- Q4: Cloud GPU budget. ✅ **CLOSED v1.4 (E0.0.4):** **LAPTOP-ONLY RTX 4090** chosen by founder. Cost ~$0 marginal vs plan v1.3 ~$420-820. Trade: +6-8 weeks calendar. Stretch replanned to option (C) reduced ~1.5B × 15B tokens. See `FJQ_PHASE_E_HARDWARE_DECISION.md`.
- Q5: Indonesian eval suite freshness. ✅ **CLOSED v1.1:** empirically verified 2026-04-25 → 94 strict ID tasks in lm-eval v0.4.11 (see §1 item 1). E0.0.5 reduced to harness SHA pinning + canonical task list selection.
- Q5: Indonesian eval suite freshness. ✅ **CLOSED v1.4 (E0.0.5):** SHA pinned to `27988a293647d5853e48edea291640b4af54740c` (lm-eval v0.4.11). v1.8 canonical set (Tier 1+2): 4 ID core + 33+ extended + 11 EN canonical + bilingual coherence. ~~tax-vertical~~ deferred to Phase F per v1.8. See `FJQ_PHASE_E_E0_EVAL_CANONICAL.md`.
- Q6: Phase D §3.4 baseline-sweep status. ✅ **CLOSED v1.4 (E0.0.6):** Accept-partial path adopted same-day (vs 1-week window). 2 real baselines preserved (BitNet 2B4T ✅ + MMfreeLM-370M ✅). 5 deferred to paper appendix + Phase E side-task with §6.11-style hardening. See `FJQ_PHASE_E_E0_PHASE_D_DEPENDENCY.md`.
- Q7: **(NEW v1.4)** Per-source data licensing for Tier B commercial deployment. ✅ **CLOSED v1.4 (E0.0.7):** Two-tier scoping — Tier A (paper) all sources OK via Bartz fair-use; Tier B1 (commercial-clean) drops CC-100 + news + academic-restricted (~10-15B raw lost, still ~15-30B usable); Tier C Indonesian-market carve-out via UU Hak Cipta Pasal 43-44. See `FJQ_PHASE_E_E0_LICENSING.md`.

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
| ~~E0.0.2 TaxPrime data review~~ **[DEFERRED — Phase F per v1.8]** | Originally: `docs/FJQ_PHASE_E_TAXPRIME_DATA_REVIEW.md` checked in (privacy/NDA notes). Tier 3 deferred ⇒ no Phase E v1.8 closure required. | n/a | n/a — Phase F prerequisite |
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

### PHASE E1: Bilingual data curation (Weeks 2–4, ~30h human, ~5h GPU)

> **v1.8 status (2026-04-26 same-session as v1.8 plan revision):** E1.1 + E1.2 + E1.4 **CLOSED**. E1.3 + E1.5 + E1.6 status updated below. Detailed measurements + recipe pivots in `docs/FJQ_PHASE_E_E1_FINDINGS.md` v1.2.

#### E1.0 Pre-flight audit

Deliverable: `docs/FJQ_PHASE_E_E1_FINDINGS.md` — corpus stats per source, dedup rate, language-detection accuracy. ✅ **CLOSED v1.0 (2026-04-26).** Updated to v1.2 with E1.2 + E1.4 closure measurements.

#### E1.1 Indonesian corpus assembly — ✅ **CLOSED v1.8**

**Measured outcome (post-E1.4 dedup):** 10,665,621 docs / 38.92 GB / **15.40 B tokens** (Mistral v3 32K, 2.527 byte/tok). Sources actually used in v1 slice:

| Source | Docs | Raw text | ~Tokens | Status |
|---|---:|---:|---:|---|
| Wikipedia ID (`wikimedia/wikipedia/20231101.id`) | 665,621 | 1.06 GB | 0.42 B | ✅ FULL |
| FineWeb-2 ID (`HuggingFaceFW/fineweb-2/ind_Latn`) | 10,000,000 | 37.86 GB | ~15.0 B | ✅ SLICE 1 |
| **Subtotal v1 slice** | **10,665,621** | **38.92 GB** | **~15.4 B** | — |

**Original aspirational sources (CC-100, OSCAR, MADLAD, Indonesian news crawls, OCR books, academic) DEFERRED** — not required for §E1.1 budget per v1.8 (15.40 B easily clears >8 B abort threshold). See `FJQ_PHASE_E_E1_FINDINGS.md` v1.2 §3 for blocker status if user wants to revisit.

**Verification:** `python python/phase_e/scripts/build_id_corpus.py --source <name> --dry-run` + `python python/phase_e/scripts/audit_id_corpora.py`.

#### E1.2 English subset selection — ✅ **CLOSED v1.8**

**Decision:** Reuse Phase D's `intllm.data.slimpajama_stream` (no on-disk parquet for EN — laptop $0 storage budget). Bilingual mix locked at **60:40 ID:EN**, EN cap **10.27 B tokens**. Repo: `gmongaras/SlimPajama-627B_Reupload` (Stretch — EN cap exceeds 6 B threshold of `DKYoon/SlimPajama-6B`). Constants + helpers in `python/phase_e/intllm_en.py` (commit `f912054`).

**FineWeb-Edu add-on DEFERRED** — re-evaluate at E5/E6 if Mini-scale ablation shows EN coverage bottleneck.

**Verification:** `make test-intllm-en` (6/6 invariants, <1 s).

#### E1.3 Vertical fine-tune corpus (tax/legal Indonesian) — **`[DEFERRED — Phase F per v1.8]`**

Originally: `data/tax_id_corpus_v1.parquet` from TaxPrime archives (post-NDA review per E0.0.2). DEFERRED per v1.8 strategic re-scope (Tier 3 → Phase F).

| Source | Approx. size | Quality | v1.8 status |
|---|---|---|---|
| Indonesian tax law (UU, PMK, PER-DJP, SE) public PDFs | ~50–200 MB text | high (regulatory) | Phase F |
| TaxPrime internal memos / advisories (sanitized) | TBD per E0.0.2 | very high | Phase F |
| Indonesian tax court rulings (Pengadilan Pajak public decisions) | ~100–500 MB text | high | Phase F |
| Bilingual tax glossary | ~5 MB | very high | Phase F |

Companion spec `FJQ_PHASE_E_TAXPRIME_DATASET_SPEC.md` v1.1 retained as authoritative Phase F roadmap (header updated to reflect deferral).

#### E1.4 Dedup + language-detect + filter — ✅ **CLOSED v1.8 via exact-hash sha256**

**Recipe pivot (LSH → exact-hash):** original Falcon-Edge / RedPajama MinHash LSH (threshold 0.85, 128 perm, 5-gram word shingles) OOM-killed at 5.96M/10M docs (LSH index hit 30 GB on 31 GB box, 2026-04-26). Pivoted to exact-string sha256 over NFKC+lowercase+ws-collapse normalized text. Justification: LSH partial measured dedup_rate 0.0093% — at this rate, MinHash near-dup is overkill.

**Measured outcome:** 2,710 dupes / 10,665,621 docs = **0.0254%** (4.9× more catches per doc than LSH partial — Wikipedia stub-page templates that LSH refused to register). Memory ~1 GB constant. Wall time **20 min** (vs LSH 6+ h projected).

**Language-detect DEFERRED** — only needed if §E1.6 packaging audit flags non-ID content above threshold. Optional E1.4.x: `fasttext` lid.176 pass on 10.66M kept docs.

**Verification:** `make test-dedup-exact` (5/5) + `make dedup-corpus-id-exact-dryrun` (4 s) + per-source `_manifest.json` files.

#### E1.5 Synthetic pretrain augmentation (v1.3 — BeyondWeb recipe, REPHRASE-of-real)

Per §14 synthetic-data hygiene + R8: add 5–10 B (up to 50% of pretrain mix per BeyondWeb 50:50 SOTA) synthetic tokens via **REPHRASING real Indonesian/English text** (not generate-from-scratch). Open-source generator pinned by E0.0.10 legal review (v1.3 PRIMARY: **Gemma 4 Apache 2.0**; secondary: Mixtral-8x7B Apache 2.0).

**BeyondWeb-style rephrase strategies (canonical recipe per arxiv 2508.10975):**

| Task | Description | Volume target | Tool |
|---|---|---|---|
| E1.5.0 **(NEW v1.5)** Prompt-template engineering | **Claude Opus 4.7** designs the rephrase prompt templates for each task below (E1.5.1-4). Claude reads NO corpus chunks, only writes prompt template strings. Claude tokens never enter training. | one-shot design (~5K input, 5K output, ~$0.10) | Claude Opus 4.7 (Lapis 2.5 meta-design role) |
| E1.5.1 Cross-lingual rephrase | Take EN technical/educational seed, generate Indonesian-language paraphrase preserving facts | 2–4 B tokens | **Gemma 4** (Lapis 2 generator) |
| E1.5.2 Web-text cleanup rephrase | Rephrase low-quality CC-100 / OSCAR ID (typos, run-ons) into clean coherent Indonesian | 2–3 B tokens | Gemma 4 |
| E1.5.3 Textbook-style restructure | Take Indonesian Wikipedia seed paragraph, restructure as textbook explanation (not invent new facts) | 1–2 B tokens | Gemma 4 |
| E1.5.4 Multi-domain expansion | Rephrase technical/legal/medical seeds into Indonesian academic style preserving claim structure | 1–2 B tokens | Gemma 4 |
| E1.5.5 **(UPDATED v1.5)** Quality filter (LLM-as-judge) | **Claude Opus 4.7** rates each Gemma 4 output 1-10 on (fluency, factual-preservation, no-hallucination); drop bottom 20% per "How to Synthesize Without Model Collapse" recipe. Claude only outputs scores (~5 tokens per example), not new content. | enforced; ~$50-100 total (5-10B Gemma input × ~$5/MTok input + ~50 output tokens × $25/MTok = mostly input cost; prompt caching saves ~50%) | Claude Opus 4.7 (Lapis 2.5 filter role) |
| E1.5.6 **(NEW v1.5)** Quality audit pass | **Claude Opus 4.7** reads 5-10% random sample of top-quality synthetic batch, flags hallucination / political slant / factual-error / bias / Indonesian dialect issues; outputs structured issue report (NOT modifications). | ~$10-20 (sample-only, output-flagging only) | Claude Opus 4.7 (Lapis 2.5 audit role) |

**v1.3 explicit anti-pattern:** generate-from-scratch synthetic (no real seed) is **PROHIBITED** per §14 R8. Every synthetic example must trace to a `source_seed_id` from real corpus.

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

### PHASE E4: Vertical fine-tune (~~Weeks 19–22, ~20h human, ~30–60h GPU~~) — **`[DEFERRED — Phase F per v1.8]`**

> **v1.8 PIVOT (2026-04-26):** ENTIRE PHASE E4 deferred to Phase F. Phase E v1.8 ships Tier 1+2 (kernel + bilingual base) only; tax-vertical fine-tune (Tier 3) becomes Phase F deliverable, gated on Phase E success + founder-bandwidth re-check. Calendar saved: ~4 weeks human + 30-60 h GPU.
>
> **What's preserved as Phase F roadmap (NOT deleted):**
> - Companion spec `docs/FJQ_PHASE_E_TAXPRIME_DATASET_SPEC.md` v1.1 — authoritative dataset spec (header updated to reflect deferral; body unchanged).
> - §11 below — Tax-vertical eval methodology spec (DEFERRED header added; body preserved verbatim).
> - `docs/FJQ_PHASE_F_TAX_VERTICAL_ROADMAP.md` (NEW v1.8) — pointer document summarizing what Phase F will lift from this plan.
> - Original E4.0 / E4.1 / E4.2 sub-task tables: SEE `FJQ_PHASE_F_TAX_VERTICAL_ROADMAP.md` (lifted verbatim there to keep this plan focused).
>
> **What Phase F will deliver (summary, full plan in roadmap doc):**
> - E4.0 Pre-flight: dataset acceptance audit + PII residual scan + cross-contamination check + baseline pass@1 floor measurement
> - E4.1 Tax/legal Indonesian fine-tune (LoRA + full-tune comparison; Pass@1 ≥ 65% gate; hallucination check; citation-format compliance)
> - E4.2 Optional bilingual-instruct general-purpose ID+EN chat
> - Prevention layer: `verify_paper_tables.py` row for tax-vertical pass@1 (Phase F)
> - Gate F4: tax-vertical eval ≥65% + domain expert sign-off
>
> **Why deferred (audit-trail per §6.8 R6):** founder solo-execution + day-job + V31.E1 in flight. Tier 3 was 190-270 person-hours of TaxPrime archive ingest + dataset assembly. E1.3 archive ingest was founder-blocked on NDA review. Cleaner first paper without Tier 3 entanglement (all-public-corpus = stronger external validity).
>
> **E4.2 bilingual-instruct caveat:** general bilingual instruct (Alpaca-style ID+EN chat) is Tier 2-relevant, not Tier 3 — could in principle stay in Phase E v1.8. Deferred WITH E4.0+E4.1 because: (i) requires pretrain checkpoint that won't exist until E3 ships, (ii) instruct tuning is downstream of base-model paper claims, (iii) cleaner E5 packaging without instruct row.

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

#### E5.2 Tax-vertical kernel path — **`[DEFERRED — Phase F per v1.8]`**

> Originally: port tax-vertical FT checkpoint to `.fjm`, `make test-tax-vertical-kernel-path` 5-invariant gate including pass@1 inside `@kernel`, Nova shell demo `nova> tax-query "..."`. ENTIRE SUB-PHASE deferred with E4 — no tax-vertical FT checkpoint exists in Phase E v1.8 to port. Reactivated in Phase F after E4-equivalent fine-tune lands.

#### E5.3 White paper

Title: *"Kernel-Context Inference for Safety-Critical Embedded AI: A Bilingual Ternary LLM in an OS Kernel."*

> **v1.8 paper scope:** Tier 1 (kernel-context execution model) + Tier 2 (Indonesian + English bilingual ternary). Tax-vertical paper claims removed from this artifact; future Phase F paper extends with vertical fine-tune results. Title intentionally unchanged — bilingual base alone supports the title's claim.

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
| E5.4.1 Patent draft for kernel-context LLM execution model | drafted by legal counsel (~~TaxPrime team option~~ deferred with Phase F; external counsel for v1.8); committed under `docs/legal/PATENT_DRAFT_KERNEL_LLM_V1.md` (note: keep confidential pre-filing) |
| E5.4.2 Open-weights / proprietary-deployment licensing decision | `docs/LICENSING_PHASE_E.md` clarifies what's Apache 2.0 (FajarQuant + IntLLM + FajarOS userspace) vs commercial-licensed (FajarOS kernel-LLM integration if pursued) |
| E5.4.3 GitHub Releases for fajar-lang v32, fajaros-x86 v4.0, fajarquant v0.5 | releases cut, topics refreshed, descriptions updated per §6.8 R7 |

**Prevention layer:** `verify_phase_e_tables.py --strict` becomes a required pre-publication CI check. README badge added.

**Gate E5:** all of E5.1–E5.4 done AND paper draft ≥80% complete with all results tables AND v32/v4.0/v0.5 releases cut AND public-artifact sync verified per §6.8 R7.

---

## 4. Timeline + effort rollup

### v1.8 calendar (Tier 1+2 only; Tier 3 deferred to Phase F)

Realistic timeline for founder solo execution at ~10h/week part-time. Total external cash cost: **$0** (laptop-only, no DVC/S3, founder self-reviews legal).

| Phase | Calendar (solo, ~10h/wk) | Human (h) | GPU (h) | Cost | Status |
|---|---|---|---|---|---|
| E0 Pre-flight | ✅ DONE Week 1 (3.5h actual) | 3.5 | 0 | $0 | 10 decision docs (E0.0.2 deferred) |
| E1 Bilingual data (NO Lapis 2 per v1.7; NO E1.3 per v1.8) | ✅ E1.1+E1.2+E1.4 DONE Weeks 2-3 (~12h actual, exact-hash dedup pivot saved 4-6h) | ~12 actual | ~0.5 | $0 | corpus 15.40 B ID + EN streaming config |
| E1.6 packaging (bilingual base only) | Week 4 (~1 wk) | 5 | 0 | $0 | `FJQ_PHASE_E_BILINGUAL_CORPUS_V1.md` + `verify-bilingual-corpus` |
| E2 Algorithm catch-up | Weeks 5–12 (~8 wk) | 25 | ~30–60 | $0 | 5 ablation decision docs |
| E3 Bilingual pretrain Mini/Base/Medium | Weeks 13–30 (~18 wk, sequential laptop) | 35 | ~70–150 | $0 | 3 scale gates + canonical bench |
| E3.4 Reduced Stretch ~1.5B × 15B (optional) | Weeks 31–35 (~5 wk) | 8 | ~50–100 | $0 | gate <3.8; OPTIONAL — ship Medium-only paper if skipped |
| ~~E4 Tax-vertical~~ | **`[DEFERRED — Phase F]`** | — | — | — | Calendar saved: ~11 wk + 30-60 h GPU |
| E5 Kernel + paper + IP (Tier 1+2 scope) | Weeks 36–43 (~8 wk, simpler than v1.7 — no E5.2 tax kernel-path, no tax-vertical paper rows) | 28 | ~5 | $0 | 1 kernel-path gate (bilingual) + paper + releases |
| **Total v1.8 solo Tier 1+2** | **~43 weeks (~10 months)** | **~116.5h** | **~150–315h on laptop** | **$0** | **5 active gates (E0/E1/E2/E3/E5)** |

**Comparison to v1.7:**

| Metric | v1.7 (Tier 1+2+3) | v1.8 (Tier 1+2) | Δ |
|---|---|---|---|
| Total weeks | ~57 | ~43 | -14 wk (-25%) |
| Human hours | ~171.5 | ~116.5 | -55h (-32%) |
| GPU hours | ~190-380 | ~150-315 | -40-65h (-17 to -21%) |
| Active gates | 6 | 5 | -1 (E4 removed) |
| External cost | $0 | $0 | unchanged |
| Calendar (months solo, 10h/wk) | ~13 | **~10** | **-3 mo** |

**v1.8 takeaway:** Tier 3 deferral saves ~3 months calendar + ~32% human-hours. Phase F roadmap preserved as natural extension; E1.3+E4+E5.2 can re-activate in ~11 wk Phase F window (Weeks 44-54) once founder bandwidth + Phase E base success allow.

> **§6.8 R5 surprise budget:** all estimates carry +25% default. Solo-mode entries +30% (single-point-of-execution risk). Realistic worst-case v1.8: ~13 months if day-job intensifies. Phase F adds 3-4 months on top → ~16-18 months for full Tier 1+2+3.

> **Calendar elasticity (v1.8):** Phase F deferral is the primary safety valve. If E2/E3 GPU time overruns, Phase F slip-by-default doesn't impact Phase E paper artifact ship. Conversely, if Phase E lands faster than estimated, Phase F can be re-prioritized into the Q4 2026 / Q1 2027 window.

> **Phase D historical reference (still listed):**
> - v1.7: ~57 weeks ($0 with Tier 3 in scope)
> - v1.6: ~58 weeks ($300-700 with Tier 3)
> - v1.5: ~34 weeks (with-team, deprecated)

---

## 5. Decision gates (§6.8 R6 mechanical, **v1.1 with FAIL paths**)

Each gate produces a committed file checked by pre-commit / commit-msg hooks. Downstream blocked until gate file lands. **Every gate has both PASS and FAIL paths now (v1.1 gap #4 closed).**

| Phase | PASS produces | FAIL path | Blocks |
|---|---|---|---|
| E0 | `FJQ_PHASE_E_E0_FINDINGS.md` + `_HARDWARE_DECISION.md` + ~~`_TAXPRIME_DATA_REVIEW.md`~~ (deferred Phase F) + `_LICENSING.md` + `_DATA_VERSIONING.md` + `_CI_GPU_PLAN.md` + `_PHASE_D_DEPENDENCY.md` + `_EVAL_CANONICAL.md` (7 sub-decisions active in v1.8) | any sub-FAIL → corresponding `*_ABORT.md` or `_PIVOT.md` written, escalate to user | E1 corpus assembly |
| E1 | `FJQ_PHASE_E_BILINGUAL_CORPUS_V1.md` + `verify-bilingual-corpus` exit 0 | <8B tokens → `FJQ_PHASE_E_PIVOT.md` (synthetic translation OR scope cut) | E2 (need real corpus) |
| E2 | 5 `*_DECISION.md` (Hadamard, FP8, distillation, balanced calib, lang-cond) + combined ≥+0.10 nat | aggregate <0 nat → `FJQ_PHASE_E_E2_ABORT.md`, fall back to Phase D config + balanced calib only | E3 |
| E3.1 | `FJQ_PHASE_E_E3_MINI_GATE.md` (PASS) + bilingual coherence ≤1.5× | val_loss > calibrated threshold → `_E3_MINI_FAIL.md` + diagnostic; retry once with hyperparam tweak; second fail → escalate | E3.2 |
| E3.2 | `FJQ_PHASE_E_E3_BASE_GATE.md` (PASS) | same retry-once protocol | E3.3 |
| E3.3 | `FJQ_PHASE_E_E3_MEDIUM_GATE.md` (PASS) | same; second fail → ship Mini+Base only as paper artifact | E3.4 + E4 + E5 |
| E3.4 | `FJQ_PHASE_E_E3_STRETCH_GATE.md` (PASS) | optional anyway; FAIL → ship Medium-only paper | paper Table 2 Stretch row only |
| ~~E4~~ | **`[DEFERRED — Phase F per v1.8]`** | n/a — gate becomes Phase F entry condition (Phase E success first) | n/a |
| E5.0.5 | `FJQ_PHASE_E_SECURITY_REVIEW.md` + fuzz green + external review ack | any kernel panic in fuzz → block patent + paper publication, fix-and-retry | E5.3, E5.4 |
| E5.1 | `make test-bilingual-kernel-path` green + 4 serving-target gates green (~~E5.2 tax-vertical kernel-path deferred~~) | any serving target miss → soft FAIL: ship paper with measured numbers + acknowledge gap; do NOT relax target retroactively | E5.3 paper Tier 1 quantitative claims |
| E5.3 | `verify_phase_e_tables.py --strict` exit 0 | claim-row mismatch → block release announcement, fix script or claim, repeat | release announcements |

**Pre-commit hook additions (extends Phase D hook):**
- `scripts/git-hooks/check-phase-e-decisions` rejects commits touching `paper/intllm/results/training_intllm-bilingual-*.json` without corresponding `FJQ_PHASE_E_E3_*_GATE.md` existing.
- `scripts/git-hooks/check-phase-e-failure-docs` rejects PASS-doc commits when prior `*_ABORT.md` or `*_FAIL.md` exists for same phase without an explicit "supersedes" reference.
- `scripts/git-hooks/check-phase-e-monthly-sweep` (informational) warns if `docs/FJQ_PHASE_E_LITERATURE_LOG.md` not updated within last 35 days.

---

## 6. Risk register + mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Indonesian corpus quality insufficient** (Q1 returns < 10B usable tokens) | ~~Medium~~ ✅ **RESOLVED v1.8** | ~~High~~ | E1.4 measured 15.40 B post-dedup — 1.92× over §2 abort threshold. Risk closed via empirical verification. |
| ~~**TaxPrime data not usable** (NDA / privacy block in E0.0.2)~~ ✅ **RESOLVED v1.2 → DEFERRED v1.8** | ~~Medium~~ | ~~Medium~~ | v1.2: ownership confirmed. v1.8: Tier 3 deferred to Phase F; risk irrelevant for Phase E v1.8. |
| ~~**TaxPrime team unavailable** (parallel work blocked)~~ ✅ **RESOLVED v1.6 → DEFERRED v1.8** | ~~Med~~ | ~~Med~~ | v1.6: solo-mode adopted. v1.8: Tier 3 deferred entirely; not a Phase E risk anymore. |
| **Founder bandwidth burnout** (v1.6 → v1.7 mitigation chain shortened → v1.8 scope reduced) | ~~Medium~~ **Low–Medium (v1.8)** | High (Phase E lock + day-job impact) | (1) v1.8 calendar reduced ~13 mo → ~10 mo via Tier 3 deferral; (2) Phase F deferral is the burnout safety valve; (3) decision-point reviews at end of E1, E2, E3 — re-scope vs continue (E1 review scheduled `trig_01MdWsiy1MnM9PCkwsEZ9vdo`); (4) Phase E v1.8 ship target = first paper artifact (clean stopping point), Phase F is opt-in extension. |
| ~~**Single-rater eval validity** (NEW v1.6 Option A)~~ ✅ **DEFERRED v1.8 with Tier 3** | ~~Medium~~ | ~~Medium~~ | Tax-vertical eval methodology preserved in §11 + Phase F roadmap; single-rater concern only relevant when tax-eval activates in Phase F. Phase E v1.8 paper has no single-rater claim. |
| **Day-job conflict** (TaxPrime business demands during E3 multi-week training) | Medium | Medium | Track B 6-layer interruption-safety covers laptop suspend; sequential per-scale training allows pause-resume. v1.8: TaxPrime founder day-job conflict reduced to compute-time-availability only (no Tier 3 dataset assembly demand on top). |
| **`[NEW v1.8]` Tier 3 deferral cannibalizes paper differentiator** (vs SEA-LION, Sahabat-AI, Komodo on bilingual axis only) | Medium | Medium (paper less unique without tax wedge) | (1) Tier 1 (kernel-context LLM) remains unique differentiator — no other player runs LLM in `@kernel` with formal isolation. Title and methodology framing emphasize Tier 1 as primary contribution. (2) Tier 2 ternary + bilingual + embedded combo still novel (SEA-LION etc. are FP16, no embedded target). (3) Phase F roadmap visible in paper future-work — signals Tier 3 coming. (4) Worst case: paper accepted on Tier 1 alone; Tier 2 supports as secondary contribution. |
| **Synthetic generator legal blocker** (E1.5 + E4.2) | Low–Medium | Medium | E0.0.10 review pre-emptive; fallback to most-permissive open-source generator (Qwen2.5 Apache 2.0) or no-synthetic |
| **Anthropic ToS challenge** (regulator or Anthropic claims dataset uses Claude output) | Low | High (could force model retraction) | §14 explicit policy + annotator FAQ in TaxPrime spec §14 + monthly literature sweep monitors enforcement actions |
| **Bilingual catastrophic interference at ternary precision** (Phase E3 Mini fails coherence gate) | Medium | High | E2.4 combined ablation includes bilingual mini run; if interference shows, add language-conditioned LayerNorm or per-language adapter |
| ~~**Stretch GPU cost overrun** (cloud burst > $1000)~~ ✅ **RESOLVED v1.4** | ~~Low–Medium~~ | ~~Medium~~ | Laptop-only chosen; reduced Stretch (option C) ~50-100h on laptop; cost $0 marginal |
| **Laptop hardware SPOF** (v1.4 → v1.7 mitigation simplified) | Low | High (full Phase E lock) | (1) ~~S3 ckpt sync via DVC~~ REMOVED v1.7 (no DVC/S3); (2) ~~cloud burst rescue~~ REMOVED v1.7; (3) **founder accepts hardware-fail = 4-7 month restart cost**; (4) MITIGATION: optional external SSD weekly backup (founder's choice, not formal requirement); (5) public corpus + pinned generator weights + spec methodology = methodology-equivalent recreation possible from scratch in worst case |
| **Thermal throttling** during sustained laptop training (NEW v1.4) | Medium | Low–Medium | Cooling pad + cooler hours (22:00-06:00 WIB) + 6h breaks; nvidia-smi temp monitor |
| **CC-100 commercial license challenge** (NEW v1.4) | Low (BLOOM/mGPT precedent) | Medium | Tier B1 commercial-clean corpus subset (drops CC-100) prepared in parallel with paper Tier A |
| **Wikipedia CC-BY-SA share-alike commercial** (NEW v1.4) | Low (Wikimedia accepts ML fair use) | Low | Attribution + share-alike-equivalent in model release docs |
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

## 8. What this document commits to (v1.8 — Tier 1+2 only)

This plan, when accepted, commits the following:

1. **Strategic positioning (v1.8):** FajarQuant + FajarOS Nova + Fajar Lang as a unified blue-ocean play in kernel-context LLM + Indonesian-English bilingual ternary embedded. ~~+ tax-vertical~~ deferred to Phase F per v1.8. Phase E formally pivots away from "FajarQuant the algorithm" framing.
2. **Scope:** ID + EN bilingual ONLY (Tier 2). Pan-SEA explicitly excluded as red ocean. Tier 3 (tax-vertical) deferred to Phase F.
3. **Algorithm catch-up baseline:** 5 features (Hadamard outlier rotation + FP8 LM head + FP16 distillation + bilingual calibration balance + language-conditioned design) evaluated at Mini scale; adopted iff combined ablation ≥+0.10 nat AND coherence ratio ≤1.5×. Data decides; FAIL path = fall back to Phase D Mini config + balanced calibration only.
4. **Production bar (v1.8):** ~26 active items in §1 (was 32 in v1.7; 6 deferred to Phase F: items 4, 11, 24-tax, 30-tax + supporting). 9 active phase gates in §5 with explicit FAIL paths (E4 gate deferred). 25 self-check rules in §7 (HONEST: 18 YES + 3 SCHEDULED + 0 NO + 4 N/A). No "we'll harden later" exceptions — §6.8 R3 prevention layer required per phase.
5. **Reproducibility:** every ML claim backed by `verify_phase_e_tables.py --strict` (pre-commit + CI). Indonesian eval suite empirically verified (94 tasks in lm-eval v0.4.11 per §1 item 1). No protocol artifacts (per §6.9 lessons). v1.8 reproducibility advantage: all-public-corpus, no proprietary archive dependency.
6. **IP path:** patent draft + open-weights/proprietary-deployment split decided in E5.4 — explicit in `LICENSING_PHASE_E.md`. Pre-publication security review required (§1 item 32, E5.0.5 NEW).
7. **Kernel-LLM serving targets quantified (v1.1):** tok/s, TTFT, RSS, power — full spec in §13. These are the Tier 1 quantitative differentiators for paper.
8. **Continuous validation (v1.1):** monthly literature sweep cadence; competitive-blindside escalation rule (7-day SLA if Microsoft/AI-Singapore/etc. ships overlap).
9. **Phase F deferral commitment (NEW v1.8):** Tier 3 work preserved as Phase F roadmap (`FJQ_PHASE_F_TAX_VERTICAL_ROADMAP.md` + spec v1.2 + §11 + §F). Activation gated on Phase E success + bandwidth re-check; not a Phase E ship blocker.

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
| 15 | BitLoRA (2026, ScienceDirect) — federated on-device ternary LoRA | **`[Phase F ref]`** Interesting for Tier 3 vertical fine-tune via LoRA | Reference; consider for E4.1 LoRA recipe (Phase F deferral per v1.8) |

**Tier 1 prior-art claim defensibility (gap #2 closure):** searched arxiv + IEEE Xplore + Google Scholar for "kernel-mode LLM inference" / "kernel-context LLM" / "OS kernel large language model" — found tiny-ML in RTOS (Zephyr+Edge Impulse, but only CNN/RNN scale), unikernel-ML (paper 13, pre-LLM era), but **no published research on running LLM inference inside an OS kernel context with formal isolation guarantees as of 2026-04-25.** Claim defensible; framed honestly in paper as "we are not aware of prior work on..." rather than absolute "first."

**Monthly literature sweep cadence (§1 item 31):** automated check first Monday of each month for new papers/products in: ternary LLM, embedded LLM, kernel-mode ML, Indonesian LLM, FajarQuant/FajarOS direct competitors. Results appended to `docs/FJQ_PHASE_E_LITERATURE_LOG.md`.

---

## 11. Tax-vertical eval methodology spec (NEW v1.1 — gap #5 closure) — **`[DEFERRED — Phase F per v1.8]`**

> **STATUS (v1.8, 2026-04-26):** ENTIRE SECTION preserved verbatim as Phase F roadmap. Tier 3 tax-vertical work deferred from Phase E v1.8; the methodology below activates when Phase F (vertical fine-tune) kicks off, gated on Phase E base success + founder bandwidth re-check.
>
> **Why preserved (not deleted):** the spec is well-designed and forward-compatible. Future-Fajar can lift verbatim into Phase F plan. Per §6.8 R7 audit-trail, removing methodology that took effort to design = bad practice; freezing it as roadmap = good practice.
>
> **Companion docs (Phase F):**
> - `docs/FJQ_PHASE_E_TAXPRIME_DATASET_SPEC.md` v1.1 — authoritative dataset spec (header to be updated to "Phase F roadmap" per task #16 in this revision)
> - `docs/FJQ_PHASE_F_TAX_VERTICAL_ROADMAP.md` (NEW v1.8) — pointer doc tying together this §11 + the dataset spec + lifted E4 sub-task tables from former §3 E4
>
> **No Phase E v1.8 commitment** to any §11 sub-section. Reader should treat §11 as **future-state spec, not current-phase deliverable**. Body unchanged below.

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
| E0 Pre-flight | E0.0.1–9 all owned here (E0.0.2 deferred) | — | — |
| E1 Bilingual data | corpus + dedup + tokenizer measure (E1.3 deferred) | — | — |
| E2 Algorithm catch-up | all 5 ablations | — | — |
| E3 Bilingual pretrain | training + canonical bench | — | — |
| ~~E4 Vertical fine-tune~~ | **`[DEFERRED — Phase F]`** | — | — |
| E5.1 Bilingual kernel-path | .fjm export script + serving benches | maybe @kernel feature additions if FJM v10 spec needed | `kernel/compute/bilingual_medium.fjm` integration + Makefile gate |
| ~~E5.2 Tax-vertical kernel-path~~ | **`[DEFERRED — Phase F]`** | — | — |
| E5.3 Paper | results tables + verify script (Tier 1+2 only; no tax-vertical row) | — | kernel-path artifact + reproducibility |
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

| Lapis | Use case | Generator allowed (v1.7 simplified) | ToS-clean? | Why |
|---|---|---|---|---|
| **Lapis 1** | Pretrain core (E1+E3) | **NONE — real data only** | N/A | E0.0.1 confirmed 25–54 B post-dedup ID tokens available; no synthetic needed. **PRIMARY PATH v1.7.** |
| ~~**Lapis 2**~~ | ~~Pretrain augmentation (E1.5) — REPHRASE real text per BeyondWeb recipe~~ | **DEFERRED TO PHASE F per v1.7 Option B.** Real corpus alone is sufficient for Phase E (Phase D Mini/Base/Medium succeeded real-only). Phase F may revisit when resources allow. | N/A | Saves ~3 months calendar + complexity. Lose BeyondWeb 7.7× speedup (acceptable for solo). LAPIS2_TEMPLATES.md preserved as Phase F future-reference. |
| ~~**Lapis 3a**~~ | ~~Vertical FT — Tax (E4.1)~~ | **`[DEFERRED — Phase F per v1.8]`** TaxPrime real data only per spec v1.1 | N/A | Tier 3 wedge deferred; activates when Phase F begins. |
| ~~**Lapis 3b**~~ | ~~Vertical FT — Instruct (E4.2)~~ | **`[DEFERRED — Phase F per v1.8]`** Gemma 4 / Mixtral rephrase OR existing open ID-instruct datasets (Cendol) | N/A | Bilingual instruct deferred with E4 (downstream of pretrain checkpoint that won't ship until E3). |
| **Lapis 2.5 (NEW v1.5)** | Synthetic-data pipeline support: filter / audit / meta-design (E1.5.0 + E1.5.5 + E1.5.6) | **Claude Opus 4.7 ALLOWED** for: (a) quality filter — rate Gemma 4 output drop bottom-N%, (b) bias/hallucination audit — read sample, flag issues, (c) prompt-template engineering — design BeyondWeb rephrase prompts. **NEVER** generate, rewrite, expand, translate, or otherwise contribute tokens that enter training corpus. | ✅ yes — categorically judging/auditing/meta-design, NOT training-data creation | Claude likely +5-15% better Indonesian-fluency discrimination than Gemma 4 self-filter; ToS-clean per "judging ≠ training" principle |
| **Lapis 3c** | Eval / LLM-as-judge (§11) | Claude / GPT-4 / open — any judge is OK because **judging ≠ training** | ✅ yes (judging is not derivative-model creation) | standard practice in lm-eval ecosystem |

### 14.2 Hard rules

1. **R1 — Synthetic ratio cap:** Lapis 2 synthetic tokens ≤ 50% of pretrain mix per BeyondWeb 50:50 SOTA recipe (was ≤25% in v1.2 — **relaxed to match canonical empirical recipe**). Pre-commit hook `verify-synthetic-mix` enforces.
2. **R2 — No Claude API output as TRAINING data.** Confirmed by Anthropic Usage Policy update + active enforcement (VentureBeat 2026 — xAI/Cursor restricted; CNBC Feb 2026 — Anthropic publicly accused DeepSeek/Moonshot/MiniMax of distillation). Applies to E1.5, E4.1, E4.2 — exceptions: Lapis 2.5 (filter/audit/meta-design — NEW v1.5) and Lapis 3c (judge — original).
   - **R2 bright-line clarification (NEW v1.5):**
     - ✅ **Judging** Gemma 4 output (rate, score, classify, drop low-quality) — Claude OK
     - ✅ **Auditing** synthetic batch for hallucination/bias/factual-error — Claude OK (read-only role)
     - ✅ **Meta-designing** prompt templates that Gemma 4 then executes (Claude doesn't see/produce training tokens) — Claude OK
     - ❌ **Generating** synthetic from scratch — Claude BLOCKED
     - ❌ **Rewriting/improving/expanding** Gemma 4 output (Claude tokens enter training via backdoor) — Claude BLOCKED
     - ❌ **Translating** corpus chunks (translation = generation) — Claude BLOCKED
     - ❌ **Distillation** (Claude as teacher for IntLLM) — Claude BLOCKED
     - ⚠️ **Soft training signal via assigned scores baked into weights** — AVOID (gray zone; treat as blocked)
     - ⚠️ **Hybrid output partial-Claude-partial-Gemma** — AVOID (fraction of Claude tokens enter training)
   - Operational test: *"Does any Claude-API token ID end up directly in IntLLM's training data, or as a label/loss target derived from Claude content?"* If YES → BLOCKED. If NO → allowed.
3. **R3 — Generator pinning:** every synthetic token must be reproducible via committed generator-weights hash + prompt template. `FJQ_PHASE_E_E1_5_SYNTHETIC.md` records generator, version, prompt corpus.
4. **R4 — Quality filter mandatory:** LLM-as-judge filters bottom 20% of generated content per "How to Synthesize Without Model Collapse" (arxiv 2412.14689).
5. **R5 — No recursive bootstrapping:** never train Phase E IntLLM on Phase E IntLLM's own output. Per "The Curse of Recursion" (Shumailov+ 2023). Applies even at SFT.
6. **R6 — Cost ceiling:** synthetic generation cost ≤ 5% of total Phase E cloud GPU budget. Default: ≤ $40 of ~$800 total. Note v1.3: Gemma 4 self-host on H100 cloud burst is expected to satisfy this trivially (~$200–500 for 5–10 B tokens vs 1500–5000× more for Claude API path).
7. **R7 — Annotator FAQ enforced:** TaxPrime spec §14 already has "DO NOT use ChatGPT/Claude to draft training data" rule; reinforced here at plan level.
8. **R8 (NEW v1.3) — REPHRASE-of-real, not GENERATE-from-scratch.** Synthetic must be transformations of real Indonesian/English text (paraphrase, summarize, structure-extract, textbook-rewrite), seeded by real corpus chunks. Canonical recipe: BeyondWeb (DatologyAI 2026, arxiv 2508.10975) — empirically beats Cosmopedia/Nemotron-Synth pure-synthetic by 2.6–5.1 pp at same training budget. Generate-from-scratch synthetic loses authentic distribution and risks model collapse.

### 14.3 Per-use-case decision rules

```
For any new training-data ask:
  Q1: Is the dataset for pretrain or fine-tune of an LLM?
    → If yes, third-party Claude/GPT API output is FORBIDDEN as data source.
    → If no (eval, judge, classifier, retrieval), Claude/GPT is ALLOWED.
  Q2: Is the use case Tier 3 vertical (tax/legal)?
    → DEFERRED to Phase F per v1.8. When activated: TaxPrime real data only. Proprietary supremacy.
  Q3: Is open-source generator legally available for the corpus license target?
    → Check E0.0.10 matrix. Default Mixtral or Qwen2.5 (Apache 2.0).
  Q4: Will the synthetic mix exceed 25% of pretrain tokens?
    → REJECT. R1 violation.
```

### 14.4 Reference papers (cited in §10 literature review, expanded v1.3)

- *The Curse of Recursion* (Shumailov+ 2023) — model collapse foundational
- *How to Synthesize Without Model Collapse* (arxiv 2412.14689, 2024) — quality filter recipes
- *Demystifying Synthetic Data in LLM Pre-training* (arxiv 2510.01631, 2025) — scaling laws
- *Phi-4 Technical Report* (Microsoft 2024) — multi-agent + self-revision recipes (Microsoft owned generator, not subject to third-party ToS)
- *Cosmopedia* (HuggingFace 2024) — Mixtral-generated 25 B token open-source pretrain dataset (canonical Lapis 2 reference)
- ⭐ *BeyondWeb: Lessons from Scaling Synthetic Data for Trillion-scale Pretraining* (DatologyAI 2026, **arxiv 2508.10975**) — **Phase E §14.2 R8 canonical recipe**. 50:50 real:rephrase synthetic outperforms Cosmopedia by 5.1 pp + Nemotron-Synth by 2.6 pp. 7.7× faster training than open web; 2.7× faster than Nemotron-Synth. 3B model on 180B BeyondWeb tokens > 8B on Cosmopedia same budget.
- *Anthropic Usage Policy update* (anthropic.com/news/usage-policy-update) — ToS basis for R2
- ⭐ **Bartz v. Anthropic** (N.D. Cal. June 2025) — fair-use precedent for LLM training on copyrighted works ("quintessentially transformative"). Critical caveat: pirated source = no fair-use defense ($1.5 B settlement). Phase E uses only legally-acquired data so fair-use defense holds.
- *Anthropic v. DeepSeek/Moonshot/MiniMax distillation accusations* (CNBC Feb 2026) — public-accusation precedent that elevates ToS R2 from contractual concern to reputational risk
- *LLM License Types Guide 2025* (local-ai-zone) — Llama Community License 700M MAU output restriction; Qwen2.5 100M MAU output restriction; Apache 2.0 unrestricted derivative

### 14.5 Cost reality check (v1.7 — true $0 solo path)

**Phase E v1.7 total external cash cost: $0.**

| Component | Cost | Why $0 |
|---|---|---|
| Compute | $0 | Laptop-only RTX 4090 (own hardware) |
| Storage | $0 | Plain local filesystem (no DVC, no S3) |
| Synthetic generation (Lapis 2) | $0 | DEFERRED to Phase F per Option B |
| Lapis 2.5 Claude filter/audit/meta-design | $0 | Lapis 2 deferred → no synthetic to filter |
| Privacy counsel | $0 | Founder is qualified counsel (SH., MH.) — self-review |
| External bias auditor | $0 | Founder self-check (per spec v1.1 §11.2) |
| AWS account / cloud setup | $0 | Not needed (no cloud) |
| **Total Phase E external cash** | **$0** | |
| Founder opportunity cost (~171.5h × hourly rate) | informational only | not a budgeted item |

**Reproducibility surface (no DVC manifest needed):**
1. git commit SHA (this repo)
2. eval/HARNESS_SHA (lm-eval pin)
3. result JSONs committed to git
4. methodology spec (FJQ_PHASE_E_TAXPRIME_DATASET_SPEC v1.1+)

4-tuple replaces v1.6's 5-tuple (DVC manifest dropped).

### 14.5.1 Cost reality check (v1.3 reference — DEPRECATED, kept for plan-evolution audit)

| Approach | 30 B tokens | 100 B tokens (full ID+EN scope) | Reproducible? | ToS-clean? | Verdict |
|---|---|---|---|---|---|
| Claude Opus 4.7 API standard ($25/MTok output) | $750K | **$2.5M** | No (Opus 4.7→4.8 drift) | No (R2 violation) | ❌ catastrophic |
| Claude Opus 4.7 Batch API (50% off) | $375K | **$1.25M** | No | No | ❌ catastrophic |
| Claude Opus 4.7 worst case (+35% tokenizer bloat) | $1.0M | **$3.4M** | No | No | ❌ catastrophic |
| GPT-5 API | $400–600K | $1.3–2M | No | No (OpenAI ToS) | ❌ catastrophic |
| **Self-host Gemma 4 on cloud H100** ⭐ | $300–600 | **$1000–2000** | Yes (pin weights) | Yes (Apache 2.0 *unrestricted*) | ✅ **Lapis 2 PRIMARY default v1.3** |
| Self-host Mixtral-8x7B on cloud H100 | $300–600 | $1000–2000 | Yes | Yes (Apache 2.0) | ✅ Lapis 2 secondary |
| Self-host Qwen2.5-72B on cloud H100 | $500–1000 | $1700–3300 | Yes | ⚠️ output license <100M MAU restriction | ⚠️ check ToS first |
| Self-host Llama-3.3-70B on cloud H100 | $400–800 | $1400–2700 | Yes | ❌ **<700M MAU output restriction** | ❌ blocked |
| No synthetic (real corpus only) | $0 | $0 | Trivially | Yes | ✅ Lapis 1 default |

**Cost arithmetic confirms (v1.3 corrected):** open-source Gemma 4 self-host is **1500–5000× cheaper** than Claude API path at 100B token scope. Reproducibility + legal cleanliness + BeyondWeb SOTA recipe compatibility are bonuses on top.

### 14.5.1 Claude as quality filter / audit (NEW v1.5 — Lapis 2.5)

| Use case | Claude API cost (5-10 B Gemma 4 token batch) | Notes |
|---|---|---|
| **E1.5.0 prompt-template design** (Claude one-shot) | ~$0.10 | tiny; Claude reads no corpus, writes 4-6 prompt templates |
| **E1.5.5 quality filter** (Claude rates Gemma 4 output 1-10) | **~$50-100** | mostly Claude INPUT cost ($5/MTok × 5-10B = $25-50); short-output scores; prompt caching saves ~50% on identical rubric |
| **E1.5.6 audit pass** (Claude reads 5-10% sample, flags issues) | **~$10-20** | sample-only; structured issue report output |
| **TOTAL Lapis 2.5 Claude cost** | **~$60-120** | added to Lapis 2 Gemma 4 generation cost |
| **Combined Phase E synthetic-augmentation cost** | **~$1,060-2,120** | Gemma 4 generation (~$1000-2000) + Claude filter+audit (~$60-120) |

Still **dramatically cheaper** than full-Claude-synthetic path ($1.25-3.4M). Claude as filter/audit/meta-design adds quality discrimination edge AT cost of $60-120 (rounding error vs $0 budget).

**Methodology arithmetic (v1.3 NEW):** even ignoring cost + legal — Claude full-synthetic loses methodologically vs hybrid 50:50 real:rephrase. BeyondWeb proves 7.7× training efficiency from recipe choice, dwarfing the ~5–15% generator-quality gap between Claude vs Gemma 4/Mixtral. **Recipe > generator quality.**

---

## §F. Phase F — Tax-Vertical Fine-Tune (Deferred from Phase E v1.8)

> **Status (v1.8, 2026-04-26):** ROADMAP. No commitment to start. Activation gated on Phase E v1.8 success (all of E0+E1+E2+E3+E5 closed) AND founder bandwidth re-check.

### F.0 Why Phase F exists

Tier 3 (tax/legal Indonesian vertical AI with audit-trail-proof provenance) is the third blue-ocean wedge identified in §0 above. It was originally scoped into Phase E v1.7 as E1.3 (corpus) + E4 (vertical fine-tune) + E5.2 (kernel-path). The v1.8 pivot deferred all three to Phase F to:

1. **Unblock Phase E ship** — E1.3 was founder-blocked on TaxPrime NDA review; E4+E5.2 downstream of E1.3.
2. **Cleaner first paper** — bilingual base + kernel-context contributions stand alone for MLSys 2027; Tier 3 wedge becomes follow-up paper.
3. **Founder bandwidth honesty** — solo execution + day-job + V31.E1 in flight cannot sustainably absorb 190-270 person-hours of Tier 3 work concurrent with Phase E.

### F.1 What Phase F lifts from Phase E v1.8

All preserved verbatim — Phase F kickoff is mostly "lift these forward + add a new plan doc":

| Preserved artifact | Location | Reactivation note |
|---|---|---|
| Tax-vertical eval methodology spec | §11 of THIS plan (DEFERRED header) | Lift verbatim into Phase F plan §3 |
| TaxPrime dataset spec v1.1 | `docs/FJQ_PHASE_E_TAXPRIME_DATASET_SPEC.md` | Lift verbatim; bump to v1.2 with Phase F header |
| E4.0 Pre-flight (dataset audit, PII scan, cross-contamination check, baseline pass@1) | summary in §3 PHASE E4 above | Lift sub-task table into Phase F E4.0 |
| E4.1 SFT recipe (LoRA r=64 + full-tune, citation-format compliance) | summary in §3 PHASE E4 above | Lift into Phase F E4.1 |
| E4.2 bilingual instruct (general ID+EN chat) | summary in §3 PHASE E4 above | Lift into Phase F E4.2 |
| E5.2 tax-vertical kernel-path | summary in §3 PHASE E5 above | Lift into Phase F E5.2 |
| Lapis 3a tax-data policy | §14.1 above (DEFERRED row) | Lift; remains "TaxPrime real data only" |
| Single-rater eval methodology | spec v1.1 §11.2 | Lift; founder + self-consistency calibration |
| Patent prep "TaxPrime IP team option" | §1 item 24 above | Re-enable as Phase F option |

### F.2 Phase F prerequisites (entry conditions)

1. ✅ **Phase E v1.8 base model shipped** (Tier 1+2 paper artifact + verify_phase_e_tables.py --strict exit 0). The bilingual Medium checkpoint is the SFT base for Phase F E4.1.
2. ✅ **Founder bandwidth re-check** committed in `docs/FJQ_PHASE_F_BANDWIDTH_CHECK.md` — explicit yes/no on whether 190-270 person-hours over 11 weeks is sustainable given concurrent commitments.
3. ✅ **TaxPrime archive access cleared** — E0.0.2 deferred sub-task (`docs/FJQ_PHASE_E_TAXPRIME_DATA_REVIEW.md`) reactivated and committed before E1.3 ingest begins.
4. ✅ **Phase E paper accepted or arXiv-posted** (de-risks F as standalone follow-up paper rather than holding co-publication dependency on Phase E review cycles).

If any prerequisite fails, Phase F holds at "ROADMAP" status; no abort doc required (this is opt-in, not gated by FAIL paths).

### F.3 Estimated Phase F effort

| Sub-phase | Calendar (solo, ~10h/wk) | Human (h) | GPU (h) | Cost |
|---|---|---|---|---|
| F0 Pre-flight (TaxPrime data review reactivate + bandwidth check) | 1 wk | 5 | 0 | $0 |
| F1 Vertical corpus assembly (E1.3 lift) | 2 wk | 10 | ~1 | $0 |
| F2 SFT recipes + eval (E4.0 + E4.1) | 4 wk | 25 | ~30-60 | $0 |
| F2.x Bilingual instruct (E4.2) | 2 wk | 10 | ~10-20 | $0 |
| F3 Tax-vertical kernel-path (E5.2 lift) | 1 wk | 5 | ~1 | $0 |
| F4 Phase F paper + IP (vertical extension) | 3 wk | 15 | 0 | $0 |
| **Total Phase F solo** | **~13 weeks (~3 months)** | **~70 h** | **~42-82 h** | **$0** |

Phase E v1.8 (~10 mo) + Phase F (~3 mo) = ~13 mo total Tier 1+2+3 — unchanged from v1.7's ~13 mo estimate, but **distributed sequentially** with a clean first-paper landing pad in between.

### F.4 Companion docs (preserved for Phase F)

- `docs/FJQ_PHASE_E_TAXPRIME_DATASET_SPEC.md` v1.1 — authoritative dataset spec (header to be updated 2026-04-26 to mark Phase F deferral)
- `docs/FJQ_PHASE_F_TAX_VERTICAL_ROADMAP.md` (NEW v1.8) — pointer document with E4 sub-task tables lifted from former §3 E4
- §11 of this plan — eval methodology (DEFERRED header)

### F.5 Decision artifact

When Phase F kicks off (or is decided NOT to kick off), update this plan footer + create `docs/FJQ_PHASE_F_KICKOFF_DECISION.md` with the decision + rationale. Don't leave §F as ambiguous "someday" forever — schedule a yes/no review at Phase E paper acceptance + 2 weeks.

---

*Plan version: 1.8 (2026-04-26). Author: Claude Opus 4.7 + Fajar (PrimeCore.id).*
*v1.8 patch: Tier 3 tax-vertical DEFERRED to Phase F. Phase E v1.8 ships Tier 1 (kernel-context LLM) + Tier 2 (Indonesian + English bilingual ternary) only. Calendar reduced ~13 mo → ~10 mo. Total Phase E cash: $0.*
*Predecessor: FJQ_PHASE_D_PRODUCTION_PLAN.md v1.2. Companions: FJQ_PHASE_E_TAXPRIME_DATASET_SPEC.md v1.1 (Phase F roadmap) + FJQ_PHASE_F_TAX_VERTICAL_ROADMAP.md (NEW v1.8) + 10 E0 decision docs (E0.0.2 deferred).*
*v1.0→v1.1 closed 8 gaps via empirical verification. v1.1→v1.2 added TaxPrime data ownership + 3-lapis policy. v1.2→v1.3 corrected synthetic generator licensing + BeyondWeb recipe + Bartz fair-use precedent. v1.3→v1.4: Phase E E0 100% COMPLETE; laptop-only adopted. v1.4→v1.5 added Lapis 2.5 (Claude filter/audit/meta-design). v1.5→v1.6: founder solo-execution mode; Tier 3 scope-down. v1.6→v1.7: Option B $0 minimalism (Lapis 2 deferred to Phase F; DVC/S3 dropped). **v1.7→v1.8 (this version): Tier 3 (tax-vertical) entirely deferred to Phase F; Phase E ships clean bilingual base + kernel paper.***
*Cross-repo coordination: fajarquant (primary) + fajar-lang (compiler features for kernel-side tokenizer + IntLLM ops) + fajaros-x86 (deployment runtime + kernel-path Makefile gates).*
*Subject to revision per phase findings; major scope changes require new Plan version (v1.x → v2.0).*
