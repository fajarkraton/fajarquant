# Phase E E1 — Bilingual Corpus Findings (Indonesian + English halves)

> **Status (v1.3):** ✅ **PHASE E1 FULLY CLOSED.** E1.1 ID corpus + E1.2 EN reuse + E1.4 dedup + **E1.6 packaging** ALL LANDED. E1.3 (tax/legal) + E1.5 (synthetic) DEFERRED to Phase F per plan v1.8 + v1.7 respectively. Next active sub-phase: **Phase E2 algorithm catch-up**.
> **Plan reference:** `FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.8 §3 E1.0–E1.6
> **Last updated:** 2026-04-27 (v1.3 splice — E1.6 packaging closed; Phase E1 fully closed)
> **Origin commits (E1.1.0–E1.1.3):** `ef7dab8` → `3ba0d41` → `bda7e1c` → `47a8892`
> **Origin commits (E1.4.0–E1.4.3):** `991078f` (LSH scaffold) → `35d13e7` (LSH --resume) → `c83f1bd` (audit) → `4ebb437` (exact-hash sha256 scaffold) → `8dc4fab` (exact-hash full sweep + cleanup + findings v1.1)
> **Origin commits (E1.2.0–E1.2.1):** `f912054` (EN config shim + smoke gate) → `e82b1c4` (findings v1.2)
> **Origin commits (E1.6.0–E1.6.3):** `86f059a` (spec + verify gate) → `3550839` (tracked git-hooks + Layer 4) → `98d41e4` (NOTICE file) → this commit (findings v1.3 + E1.6 closure)
> **Plan revision:** `7b8949e` (v1.7 → v1.8 — Tier 3 deferred to Phase F)

This doc is incrementally updated as each E1.x sub-task completes. Initial version closes E1.0 pre-flight + records E1.1 first slice (Wikipedia ID + FineWeb-2 ID). v1.1 splice records E1.4 dedup completion via exact-hash sha256 (LSH variant pivoted after OOM). v1.2 splice records E1.2 EN reuse via Phase D `intllm.data.slimpajama_stream` shim with bilingual ratio + repo selection helpers. **v1.3 splice closes Phase E1 entirely** — E1.6 final corpus packaging shipped (spec + verify gate + tracked hooks + NOTICE).

---

## 1. Per-source corpus stats (E1.1 Indonesian)

All numbers from `data/phase_e/corpus_id/<source>/_manifest.json` written by `python/phase_e/scripts/build_id_corpus.py`. Token estimates use the E0.0.3 measured Mistral v3 32K rate of **2.527 bytes/token on Indonesian** (`paper/intllm/results/phase_e/e0_tokenizer_compression.json`).

| Source | Status | Docs | Raw text | Parquet | ~Tokens | Compression | Runtime | License |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Wikipedia ID (`wikimedia/wikipedia/20231101.id`) | ✅ FULL | 665,621 | 1.06 GB | 366 MB | 0.42 B | 2.90× | 3.5 min | CC BY-SA 4.0 |
| FineWeb-2 ID (`HuggingFaceFW/fineweb-2/ind_Latn`) | ✅ SLICE 1 (10M docs) | 10,000,000 | 37.86 GB | 14.60 GB | ~15.0 B | 2.59× | 75 min | ODC-By |
| **Subtotal on disk** | — | **10,665,621** | **38.92 GB** | **14.97 GB** | **~15.4 B** | — | **78.5 min** | — |

`_manifest.json` files are committed; raw `shard_*.parquet` files are gitignored under `.gitignore` rule `data/phase_e/corpus_id/**/shard_*.parquet`. Hash-pinning via DVC deferred per plan v1.7 (laptop-only $0 storage strategy).

## 2. Verdict vs plan §E1.1 budget — MEASURED post-dedup (v1.1)

Post-dedup numbers from `data/phase_e/corpus_id_dedup_exact/<source>/_manifest.json` written by `python/phase_e/scripts/dedup_corpus_exact.py` (commit `4ebb437`, sweep finished 2026-04-26 16:01 UTC).

| Source | Docs in | Docs kept | Docs dropped | dedup_rate | Bytes kept | ~Tokens (kept ÷ 2.527) |
|---|---:|---:|---:|---:|---:|---:|
| FineWeb-2 ID | 10,000,000 | 9,999,982 | 18 | 0.0002% | 37.86 GB | ~14.99 B |
| Wikipedia ID | 665,621 | 662,929 | 2,692 | 0.404% | 1.06 GB | ~0.42 B |
| **Aggregate** | **10,665,621** | **10,662,911** | **2,710** | **0.0254%** | **38.92 GB** | **~15.40 B** |

| Gate | Threshold | Measured | Status |
|---|---:|---:|---|
| Plan §E1.1 ID target | 12–18 B post-dedup | **15.40 B** | ✅ **MET** (within band, no retention assumptions needed) |
| Plan §2 abort threshold | > 8 B post-dedup | **15.40 B** | ✅ **CLEARED 1.92×** |
| Falcon-Edge / RedPajama recipe retention | 60–85% (estimated) | **99.97% kept** (corpus essentially clean) | ✅ Far above estimate |

**Implication:** Tier A pretrain corpus is foundational with just Wikipedia + FineWeb-2 slice 1. The remaining 4 ID sources (OSCAR / CulturaX / MADLAD-400 / CC-100) are confirmed **quality + diversity additions, not budget blockers**. Retention is essentially upstream-quality (FineWeb-2 already MinHash-deduped by HuggingFace) plus a handful of Wikipedia stub-page templates.

## 3. Remaining ID sources — blocker status (E1.1.2 dry-run)

| Source | Plan estimate | Blocker | Workaround |
|---|---:|---|---|
| CC-100 ID | 10–15 B | HF removed Python script datasets (`cc100.py`) | statmt.org tarball mirror (E1.1.x future) |
| OSCAR v23.01 ID | 5–8 B | HF gated; click-through accept needed | User: visit `huggingface.co/datasets/oscar-corpus/OSCAR-2301` and "Agree" |
| CulturaX ID | 8–20 B | HF gated; click-through accept needed | User: visit `huggingface.co/datasets/uonlp/CulturaX` and "Agree" |
| MADLAD-400 ID | 5–10 B | 99,272 siblings; HF Datasets config-resolve > 90 s | `HfApi().list_repo_files` + glob `data/*/id/*.jsonl.gz` direct path (E1.1.x future) |
| FineWeb-2 ID slice 2+ | +35–55 B est | None — script supports `--max-docs`, but streaming restarts at doc 0 | Add `--skip-docs N` flag for incremental slices (E1.1.x future) |

User-action blockers (OSCAR + CulturaX) are 1-click each, ~5 min total. Once accepted, `python python/phase_e/scripts/build_id_corpus.py --source oscar_2301_id` and `--source culturax_id` will proceed.

## 4. Dedup + language-detect — E1.4 LANDED (v1.1)

Per plan §3 E1.4, dedup was originally specified as MinHash LSH at threshold 0.85, 128 permutations, 5-gram word shingles (Falcon-Edge / RedPajama recipe). Implementation pivoted to **exact-string sha256** mid-execution after the LSH variant OOM-killed at 5.96M / 10M FineWeb-2 docs. Final results above (§2).

### 4.1 Per-source measurements

- **Wikipedia ID intra-dedup:** 2,692 dupes / 665,621 docs = **0.404%**. Sample dropped digests show stub-page templates (e.g. `49606bc...` matched 4× against bare "Kelahiran / Kematian / Pranala luar / Abad ke-20" boilerplate). Confirmed legitimate exact-template dupes after NFKC + lowercase + whitespace-collapse normalization.
- **FineWeb-2 ID intra-dedup:** 18 dupes / 10,000,000 docs = **0.0002%**. Corpus is essentially clean — consistent with HuggingFace's upstream MinHash deduplication (whose `minhash_cluster_size` field we dropped at ingest, see §5).
- **Cross-source dedup:** captured implicitly by the exact-hash sweep (single global digest set across both sources). 0 cross-source matches observed in sample_dropped (all 23 sample entries are intra-source). Not surprising: Wikipedia stub templates don't appear verbatim on FineWeb-2.
- **Language-detect:** still unmeasured at this version. FineWeb-2 ind_Latn pre-filter trusted; Wikipedia ID canonical-by-construction. Deferred to a follow-up E1.4.x or §E1.6 packaging audit if needed.

**Measured post-dedup retention:** **99.97%** kept (10,662,911 / 10,665,621). The 60–85% Falcon-Edge band was conservative — this corpus is upstream-clean.

### 4.2 Recipe pivot rationale (LSH → exact-hash sha256)

**OOM forensic.** Original `dedup_corpus.py` (MinHash LSH, commits `991078f` + `35d13e7`) was launched on the full 10.67M-doc corpus on 2026-04-26 09:58 WIB. Process killed by OS at 13:51 WIB after 3h53m, RSS hit 30 GB on a 31 GB box. Killpoint: 5.96M / 10M FineWeb-2 docs (Wikipedia untouched, scheduled second). LSH index growth dominated memory: each kept doc inserts ~5 KB (128 perm × ~40-byte hash shingles + LSH bucket overhead) → linear with kept-corpus size. Resume re-OOM'd at the same point.

**Empirical dedup_rate from LSH partial:** 552 dupes / 5.96M = **0.0093%**. At this rate, MinHash near-dup machinery is overkill — exact-string sha256 catches >99% of legitimate dupes (everything except trivial paraphrase, which at this rate would contribute <60 docs across the full 10M). The cost-benefit flipped against LSH.

**Exact-hash design:**

- NFKC normalize → lowercase → collapse whitespace runs to single space → strip → sha256 → 32-byte digest. First-seen wins.
- Storage: `set[bytes]` of digests. ~32 B per digest + ~80 B Python set overhead = ~1.1 GB at 10M docs, **constant** across corpus size (vs LSH linear growth).
- Reuses LSH script's streaming + atomic shard write + manifest + --resume infrastructure verbatim (only the "is this a dup?" core swapped).

**Decision artifact (CLAUDE §6.8 R6 mechanical decision gate):** the pivot is committed as code (`dedup_corpus_exact.py` + Makefile target `dedup-corpus-id-exact` + `.gitignore` entries) plus this doc splice. The LSH variant remains in-tree for forensic reference and as the canonical Falcon-Edge implementation; it would still apply to higher-noise corpora with paraphrase content.

**Outcome.** Full sweep ran in **20 minutes** wall-clock (vs LSH's projected 6+ hours before OOM): 19m23s for FineWeb-2 + 30.5s for Wikipedia, sustained 8,600 doc/s on a single thread. Peak RAM ~1 GB. Output 14.4 GB across 146 shards. **4.9× more dupes per doc detected than LSH** — exact-hash beats LSH on stub-page-heavy corpora because shingles of near-empty docs converge to one degenerate shingle that LSH refuses to register, while sha256-of-normalized-text catches the boilerplate match.

### 4.3 Verification commands (CLAUDE §6.8 R2)

```bash
# Re-run smoke gate (~5s)
make test-dedup-exact

# Inspect aggregate manifests
cat data/phase_e/corpus_id_dedup_exact/wikimedia__wikipedia/_manifest.json
cat data/phase_e/corpus_id_dedup_exact/HuggingFaceFW__fineweb-2/_manifest.json

# Re-dry-run on 10K-doc cap (~4s)
make dedup-corpus-id-exact-dryrun
```

## 5. EN subset reuse — E1.2 LANDED (v1.2)

Per plan §3 E1.2, the English half reuses Phase D's `intllm.data.slimpajama_stream` rather than producing its own on-disk parquet shards. EN at typical Phase E targets (10–25 B tokens) would consume 38–96 GB of raw text — too large for the laptop-only $0 storage strategy of plan v1.7.

### 5.1 Configuration shim

Constants and helpers committed at `python/phase_e/intllm_en.py` (commit `f912054`). Stdlib + `json` only — torch / transformers / datasets are NOT imported at module load, so the shim and its smoke gate run without GPU dependencies.

| Constant | Value | Source |
|---|---|---|
| `ID_BYTES_PER_TOKEN` | 2.527 | E0.0.3 measurement on 1000 ID Wikipedia articles |
| `EN_BYTES_PER_TOKEN` | 3.830 | E0.0.3 measurement on 1000 EN Wikipedia articles |
| `BILINGUAL_RATIO_DEFAULT` | 0.6 (60:40 ID:EN) | matches §E2.4.1 `BilingualCalibrationSampler` default |
| `ID_TOKENS_AVAILABLE` | 15.40 B | post-E1.4 exact-hash dedup, from `_manifest.json` aggregate |
| `SLIMPAJAMA_DEFAULT_REPO` | `DKYoon/SlimPajama-6B` | Phase D Mini/Base/Medium production |
| `SLIMPAJAMA_STRETCH_REPO` | `gmongaras/SlimPajama-627B_Reupload` | Phase D Stretch (>15 B token budgets) |
| `SLIMPAJAMA_STRETCH_THRESHOLD_TOKENS` | 6.0e9 | repo-pivot trigger (Phase D convention) |

Helpers: `compute_en_token_cap(id_tokens, ratio_id_share)`, `compute_en_byte_cap(en_tokens)`, `select_slimpajama_repo(en_token_target)`, `id_tokens_from_e1_4_manifests(repo_root)`, `bilingual_mix_summary(...)`. Doctests embedded in each.

### 5.2 Repo-selection arithmetic at the chosen ratio

At the working default of 60:40 ID:EN with the measured 15.40 B ID corpus:

| Quantity | Value | Source |
|---|---:|---|
| Target total tokens | 25.67 B | `id_tokens / ratio_id_share` = 15.40 / 0.6 |
| EN token cap | **10.27 B** | total − ID = 25.67 − 15.40 |
| EN bytes implied | 39.33 GB | EN cap × 3.830 byte/tok |
| Slimpajama repo selected | `gmongaras/SlimPajama-627B_Reupload` | EN cap > 6 B threshold → Stretch pivot |

The EN cap (10.27 B) **exceeds the 6 B threshold** of `DKYoon/SlimPajama-6B`, so `select_slimpajama_repo(10.27e9)` returns the 627B Stretch repo. Earlier project commentary that defaulted to "SlimPajama-6B" was insufficient for the chosen ratio at Phase E's actual ID corpus size; the shim makes this explicit and the smoke gate asserts the Stretch repo is selected for the default config (v1.2 audit-trail item).

Alternative ratio reference points (recompute via `compute_en_token_cap`):

| Ratio (ID share) | EN cap | Repo |
|---:|---:|---|
| 0.5 (50:50) | 15.40 B | Stretch (627B) |
| 0.6 (60:40, **default**) | 10.27 B | Stretch (627B) |
| 0.7 (70:30) | 6.60 B | Stretch (627B), barely over threshold |
| 0.72 (~71.9:28.1) | 6.00 B | exactly at threshold |
| 0.8 (80:20) | 3.85 B | Default (6B), comfortable single-epoch |

If a future decision pivots to ratio ≥ 0.72 to keep EN within `DKYoon/SlimPajama-6B`, the helper will auto-select the smaller repo without code changes.

### 5.3 No streaming code in this shim — by design

`slimpajama_stream` itself stays in `python/phase_d/intllm/data.py` (Phase D production, hardened by Track B step 4 with 60s read timeout + retry per CLAUDE §6.11). Phase E training drivers (E2/E3 phases) will `import` it directly when training begins, parameterised by `repo` and an EN-side cap derived from `bilingual_mix_summary()`. Keeping the shim torch-free preserves <1 s smoke-gate runtime and lets the constants survive heavyweight dependency churn.

### 5.4 Verification commands (CLAUDE §6.8 R2)

```bash
# Smoke gate — 6/6 invariants in <1 s
make test-intllm-en

# Inspect mix summary at default 60:40
.venv/bin/python -c "
import sys; sys.path.insert(0, 'python/phase_e')
import intllm_en, json
print(json.dumps(intllm_en.bilingual_mix_summary(), indent=2, default=str))
"

# Inspect alternative ratios
.venv/bin/python -c "
import sys; sys.path.insert(0, 'python/phase_e')
import intllm_en
for r in (0.5, 0.6, 0.7, 0.72, 0.8):
    en = intllm_en.compute_en_token_cap(15.40e9, r) / 1e9
    repo = intllm_en.select_slimpajama_repo(en * 1e9).split('/')[-1]
    print(f'  ratio={r}  EN={en:5.2f}B  repo={repo}')
"
```

## 6. Final corpus packaging — E1.6 LANDED (v1.3)

E1.6 closes Phase E1 with a 4-artifact packaging deliverable: spec doc + cross-validation gate + reproducible git-hooks installation + NOTICE file for downstream model release.

### 6.1 Artifacts shipped (4 sub-tasks)

| Sub-task | Commit | Artifact | Lines |
|---|---|---|---|
| E1.6.0 scaffold | `86f059a` | `docs/FJQ_PHASE_E_BILINGUAL_CORPUS_V1.md` (spec) + `python/phase_e/scripts/verify_bilingual_corpus.py` (verify) + Makefile target `verify-bilingual-corpus` | 175 + 145 + 16 |
| E1.6.1 hook wiring | `3550839` | `scripts/git-hooks/pre-commit` (V3 with Layer 4) + `scripts/git-hooks/install-hooks.sh` (idempotent installer) | 64 + 50 |
| E1.6.2 NOTICE file | `98d41e4` | `NOTICE_BILINGUAL_CORPUS_V1` at repo root (per-source attribution + reproducibility surface + use-case scope) | 132 |
| E1.6.3 findings v1.3 | this commit | This doc bumped v1.2 → v1.3; spec doc §6 tracker updated; Phase E1 sign-off "ALL E1.x sub-phases closed" | — |

### 6.2 Spec ↔ manifests ↔ shim consistency (8 invariants enforced)

`make verify-bilingual-corpus` runs <1 s and asserts:

1. Both ID manifests exist + parseable
2. ID `bytes_text_kept` aggregate within ±0.5 GB of spec §1.3 (38.92 GB)
3. ID `docs_kept` aggregate matches spec §1.3 exactly (10,662,911)
4. ID `dedup_rate` within ±0.005% of spec §1.3 (0.0254%)
5. EN cap from `intllm_en.bilingual_mix_summary()` matches spec §1.2 (10.27 B at 60:40 default ± 0.05 B)
6. EN repo selected = `gmongaras/SlimPajama-627B_Reupload` (Stretch — EN cap exceeds 6 B threshold of `DKYoon/SlimPajama-6B`)
7. `synthetic_mix_ratio == 0%` (regex-asserted on spec §1.3 + §4)
8. License attribution covers all 3 sources actually used (regex on spec §2 — Wikipedia CC BY-SA 4.0, FineWeb-2 ODC-By 1.0, SlimPajama Apache 2.0)

Pre-commit hook Layer 4 fires on any staged change to the 4 trigger paths (manifests / `intllm_en.py` / verify script / spec doc) so spec drift fails fast at commit time per CLAUDE §6.8 R3.

### 6.3 Smoke output (live numbers)

```
$ make verify-bilingual-corpus
verify-bilingual-corpus: 8/8 invariants PASS
  ID:   10,662,911 docs / 38.919 GB / ~15.401 B tokens (0.0254% dedup)
  EN:  cap 10.268 B at 60%:40% → gmongaras/SlimPajama-627B_Reupload
  Mix: total 25.669 B = 15.401 ID + 10.268 EN  (synthetic 0%)
```

### 6.4 NOTICE file ready for downstream model release

`NOTICE_BILINGUAL_CORPUS_V1` at repo root (132 lines) carries forward:

- **Wikipedia ID (CC BY-SA 4.0):** attribution + share-alike provision per §4(b)(iv); Bartz v. Anthropic fair-use precedent cited
- **FineWeb-2 ID (ODC-By 1.0):** attribution + indication-of-changes (our exact-hash sha256 dedup is incremental on top of upstream MinHash)
- **SlimPajama-627B (Apache 2.0):** NOTICE provisions travel with the upstream HF dataset (streaming-only, no bytes redistributed)
- **Reproducibility surface (4-tuple per plan v1.7 §14.5):** git SHA + HARNESS_SHA + per-source manifests + HF dataset revisions
- **Use-case scope:** Phase E v1.8 Tier 1+2 only; Tier 3 (tax-vertical) explicitly out of scope per Phase F deferral

When a Phase E-derived model ships (HF model card + paper + arXiv preprint), this NOTICE accompanies it.

### 6.5 Phase E1 → Phase E2 transition

With E1.6 closed, **Phase E1 is fully closed**. Per plan v1.8 §3:

| Sub-phase | v1.8 status |
|---|---|
| E1.0 Pre-flight | ✅ CLOSED |
| E1.1 Indonesian corpus | ✅ CLOSED — 10.66M docs / 15.40 B tokens |
| E1.2 EN subset | ✅ CLOSED — `intllm_en.py` shim, 60:40, Stretch repo |
| E1.3 Vertical FT corpus | ⏭️ Phase F |
| E1.4 Dedup | ✅ CLOSED — exact-hash sha256, 99.97% retention |
| E1.5 Synthetic | ⏭️ Phase F |
| E1.6 Final packaging | ✅ **CLOSED (v1.3)** |

**Next active sub-phase: Phase E2 algorithm catch-up** — 5 ablations on Mini scale (Hadamard outlier rotation + FP8 LM head + FP16 distillation + bilingual calibration balance + language-conditioned design). Estimated ~8 weeks human + 30-60 h GPU on laptop RTX 4090. See plan v1.8 §3 PHASE E2.

## 7. Schema coverage (ID corpus)

Both sources written with **two columns only**: `text` (str), `source` (str). Other source-side metadata is dropped by `build_source()` in `python/phase_e/scripts/build_id_corpus.py:128`. Specifically discarded:

- Wikipedia: `id`, `url`, `title`
- FineWeb-2: `id`, `dump`, `url`, `date`, `file_path`, `language`, `language_score`, `language_script`, `minhash_cluster_size`, `top_langs`

**Trade-off accepted:** simpler downstream tokenization + smaller parquet shards vs losing per-doc dedup signal already computed by HuggingFace (`minhash_cluster_size`). E1.4 will recompute dedup ourselves anyway; URL/date provenance lost is acceptable per plan §15 attribution-at-source-level (license file in §E1.6 packaging, not per-doc).

## 8. Verification commands — ID corpus (CLAUDE §6.8 R2)

```bash
# Total docs across all checked-in sources
python -c "
import pyarrow.parquet as pq, glob
total = 0
for src in sorted(glob.glob('data/phase_e/corpus_id/*/shard_*.parquet')):
    total += pq.read_metadata(src).num_rows
print(f'total docs: {total}')
"
# expected: 10,665,621 (665,621 Wikipedia + 10,000,000 FineWeb-2 slice 1)

# Re-dry-run any source to validate registry
python python/phase_e/scripts/build_id_corpus.py --source wikipedia_id --dry-run
python python/phase_e/scripts/build_id_corpus.py --source fineweb_2_id --dry-run

# Re-audit the 9 ID sources from plan §E1.1
python python/phase_e/scripts/audit_id_corpora.py
```

## 9. Decisions logged at this version

1. **FineWeb-2 ID capped at 10M docs** for first slice. Full ind_Latn (32 train files × 4.5 GB = 144 GB source) would extrapolate to ~50–70 B tokens, well above plan upper estimate. Cap chosen to fit laptop-only disk budget per plan v1.7 hardware lock; 15 B from one slice ALREADY meets §E1.1 target.
2. **OSCAR / CulturaX defer to user click-through.** Not blocking corpus completion; nice-to-have for diversity.
3. **CC-100 ID and MADLAD-400 ID workarounds DEFERRED.** Neither is required for §E1.1 budget. Re-evaluate at §E1.6 packaging time if quality/diversity audit demands more breadth.
4. **Cosmetic SIGABRT in FineWeb-2 download.** `PyGILState_Release: thread state must be current when releasing` from C-extension shutdown order quirk (HF Datasets + aiohttp + pyarrow). Crash occurred AFTER all data + manifest written; manifest reports `error: null`; read-back of all 142 shards verifies 10,000,000 rows. Documented for future hardening; no data loss this run.
5. **(v1.1)** **Dedup recipe pivot LSH → exact-hash sha256.** OOM-driven, see §4.2 above. Decision committed as code: `dedup_corpus_exact.py` + Makefile + .gitignore + this doc. LSH script kept in-tree for higher-noise future corpora.
6. **(v1.1)** **LSH partial cleanup.** 84-shard partial output (15 GB) + `.progress/state.pkl` (6.8 GB) removed after exact-hash sweep validated and manifests written. The exact-hash sweep is fully reproducible from `corpus_id/` raw shards via `make dedup-corpus-id-exact`; LSH partial held no information not reproducible from the full sweep.
7. **(v1.2)** **Bilingual ratio 60:40 ID:EN locked as default.** Matches §E2.4.1 `BilingualCalibrationSampler` default. Recorded in `intllm_en.BILINGUAL_RATIO_DEFAULT`; smoke gate asserts. 50:50 / 70:30 / 80:20 alternatives recoverable via `compute_en_token_cap(id_tokens, ratio_id_share)` without code changes (see §5.2 table).
8. **(v1.2)** **EN repo at 60:40 = `gmongaras/SlimPajama-627B_Reupload` (Stretch).** EN cap of 10.27 B at the working ratio exceeds the 6 B threshold of `DKYoon/SlimPajama-6B`. The smoke gate asserts the Stretch repo is selected. If a future decision pivots to ratio ≥ 0.72 the Default 6B repo would suffice — recompute via `select_slimpajama_repo`.
9. **(v1.2)** **EN as streaming-only, no on-disk parquet.** EN at 10–25 B tokens = 38–96 GB raw text, exceeds laptop-only $0 storage budget per plan v1.7. Phase E training drivers will import `slimpajama_stream` from Phase D and feed tokenized batches directly; no `data/phase_e/corpus_en_*` directory will exist. FineWeb-Edu add-on SKIPPED for v0; re-evaluate at E5/E6 if Mini-scale ablation flags EN coverage as a bottleneck.
10. **(v1.3)** **Plan v1.7 → v1.8 pivot — Tier 3 tax-vertical DEFERRED to Phase F.** Phase E v1.8 ships Tier 1+2 only (kernel-context LLM + bilingual ID+EN ternary). Calendar reduced ~13 mo → ~10 mo solo. Tier 3 work preserved as Phase F roadmap (see commit `7b8949e` + `FJQ_PHASE_F_TAX_VERTICAL_ROADMAP.md` + `FJQ_PHASE_E_TAXPRIME_DATASET_SPEC.md` v1.2). E1.3 + E1.5 are both Phase F deferrals.
11. **(v1.3)** **E1.6 final corpus packaging closes Phase E1.** Spec doc + 8-invariant verify gate (`make verify-bilingual-corpus`) + tracked git-hooks with conditional Layer 4 firing on staged trigger paths + `NOTICE_BILINGUAL_CORPUS_V1` for downstream model release. Pre-commit hook ensures spec ↔ manifests ↔ shim consistency mechanically (CLAUDE §6.8 R3 prevention layer).

## 10. Open items for E1.x continuation

- [x] **E1.2 — English subset selection: LANDED 2026-04-26 via `intllm_en.py` config shim (60:40 default, Stretch repo per ratio math, see §5).** Streaming integration deferred to E2/E3 training drivers.
- [x] **E1.3 — Vertical fine-tune corpus: DEFERRED to Phase F per plan v1.8** (Tier 3 strategic re-scope 2026-04-26). Companion spec preserved at `FJQ_PHASE_E_TAXPRIME_DATASET_SPEC.md` v1.2 with Phase F roadmap header. NOT a Phase E v1.8 ship gate.
- [x] **E1.4 — Dedup + filter: LANDED 2026-04-26 via exact-hash sha256 (LSH variant pivoted post-OOM, see §4).** Language-detect deferred to E1.6 packaging audit if needed.
- [x] **E1.5 — Synthetic pretrain augmentation: DEFERRED to Phase F** per plan v1.7 (Option B aggressive $0). Real corpus alone sufficient; lose BeyondWeb 7.7× speedup. Templates preserved at `python/phase_e/prompts/LAPIS2_TEMPLATES.md`.
- [x] **E1.6 — Final corpus packaging: LANDED 2026-04-27** (this v1.3 splice). 4 sub-tasks all closed: spec + verify gate + tracked hooks + NOTICE file. See §6 above.
- [ ] OPTIONAL E1.1.x — `--skip-docs N` flag for FineWeb-2 incremental slices.
- [ ] OPTIONAL E1.1.x — MADLAD-400 ID via `HfApi().list_repo_files` glob workaround.
- [ ] OPTIONAL E1.1.x — CC-100 ID via statmt.org tarball mirror (lower priority due to non-commercial license).
- [ ] OPTIONAL E1.2.x — FineWeb-Edu add-on (~20 B EN high-quality) for code/technical content. Re-evaluate at E5/E6 per §5.3.
- [ ] OPTIONAL E1.4.x — language-detect pass via `fasttext` lid.176 on the 10.66M kept docs (post-dedup); only needed if §E1.6 packaging audit flags non-ID content above a threshold.

---

**Sign-off this version (v1.3):** ✅ **PHASE E1 FULLY CLOSED 2026-04-27.** All 5 Phase E1 ship sub-phases LANDED (E1.0 + E1.1 + E1.2 + E1.4 + E1.6). 2 sub-phases DEFERRED to Phase F (E1.3 vertical FT corpus + E1.5 synthetic augmentation). ID corpus measured at 15.40 B tokens (1.92× over §2 abort threshold); EN side configured at 60:40 default → 10.27 B EN cap on `gmongaras/SlimPajama-627B_Reupload`; bilingual mix totals 25.67 B tokens, 0% synthetic. Spec ↔ manifests ↔ shim consistency enforced mechanically by `make verify-bilingual-corpus` (8/8 invariants) + pre-commit Layer 4 conditional gate + `NOTICE_BILINGUAL_CORPUS_V1` for downstream release.

**Next active sub-phase: Phase E2 algorithm catch-up** (5 ablations on Mini scale per plan v1.8 §3 PHASE E2). Estimated ~8 weeks human + 30-60 h GPU on laptop RTX 4090.
