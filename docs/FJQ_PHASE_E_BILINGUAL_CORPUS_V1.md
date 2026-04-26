# Phase E Bilingual Corpus v1.0 — Packaging Spec (Indonesian + English)

> **Status:** E1.6 packaging artifact for Phase E v1.8 (Tier 1+2 scope). Final close-out for Phase E1 sub-phase.
>
> **Plan reference:** `FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.8 §3 PHASE E1.6
>
> **Closure inputs:**
> - `docs/FJQ_PHASE_E_E1_FINDINGS.md` v1.2 (E1.0 + E1.1 + E1.2 + E1.4 measured outcomes)
> - `data/phase_e/corpus_id_dedup_exact/<source>/_manifest.json` (per-source ID dedup stats)
> - `python/phase_e/intllm_en.py` (EN config shim constants)
> - `paper/intllm/results/phase_e/e0_tokenizer_compression.json` (E0.0.3 measurement)
>
> **Verify (runnable, per CLAUDE §6.8 R2):**
> ```
> make verify-bilingual-corpus
> ```
>
> **Last updated:** 2026-04-27 (Phase E v1.8, E1.6.0 scaffold)

---

## 1. Source breakdown (Phase E v1.8 corpus composition)

### 1.1 Indonesian half — on-disk parquet shards

Both sources committed as gitignored parquet under `data/phase_e/corpus_id_dedup_exact/<source>/shard_NNNNN.parquet`. Per-source `_manifest.json` files committed (small, post-dedup stats verifiable from repo).

| Source | HF dataset | Revision | Docs (post-dedup) | Bytes kept | ~Tokens (Mistral v3 32K, 2.527 byte/tok) |
|---|---|---|---:|---:|---:|
| Wikipedia ID | `wikimedia/wikipedia` (config `20231101.id`) | snapshot `20231101` | 662,929 | 1.06 GB | 0.42 B |
| FineWeb-2 ID | `HuggingFaceFW/fineweb-2` (config `ind_Latn`, slice 1) | streaming default 2026-04-26 | 9,999,982 | 37.86 GB | ~14.99 B |
| **Subtotal ID (post-exact-hash dedup)** | — | — | **10,662,911** | **38.92 GB** | **~15.40 B** |

### 1.2 English half — streaming (no on-disk parquet)

EN reuses Phase D's `intllm.data.slimpajama_stream` per `python/phase_e/intllm_en.py`. No parquet shards on disk (laptop $0 storage budget per plan v1.7 / v1.8).

| Source | HF dataset | Revision | Token cap (60:40 ID:EN) |
|---|---|---|---:|
| SlimPajama (Stretch repo) | `gmongaras/SlimPajama-627B_Reupload` | streaming default 2026-04-27 | 10.27 B |
| (FineWeb-Edu add-on) | DEFERRED to E5/E6 per `intllm_en.py` §5.3 | — | — |

EN repo selected by `intllm_en.select_slimpajama_repo(10.27e9)` because the cap exceeds the 6 B threshold of `DKYoon/SlimPajama-6B`. See `FJQ_PHASE_E_E1_FINDINGS.md` v1.2 §5.2 for the alternative-ratio table.

### 1.3 Aggregate (ID + EN)

| Quantity | Value | Notes |
|---|---:|---|
| Total docs (ID side, post-dedup) | 10,662,911 | EN side has no doc count — packed-token streaming |
| Total bytes (ID side, post-dedup) | 38.92 GB | EN side capped by tokens, not bytes |
| Total tokens at 60:40 mix | **25.67 B** | 15.40 B ID + 10.27 B EN |
| Dedup rate (ID side, exact-hash sha256) | 0.0254% | 2,710 dupes / 10,665,621 input docs |
| Synthetic-mix-ratio | **0%** | E1.5 (Lapis 2 BeyondWeb-style augmentation) DEFERRED to Phase F per plan v1.7 |
| Bilingual ID:EN ratio | 60:40 | matches §E2.4.1 `BilingualCalibrationSampler` default; alt ratios via helper |

---

## 2. License attribution

| Source | License | Commercial use? | Attribution requirement | Source URL |
|---|---|---|---|---|
| Wikipedia ID | CC BY-SA 4.0 | Yes (with share-alike) | Attribution + share-alike-equivalent in derivative model docs | https://huggingface.co/datasets/wikimedia/wikipedia |
| FineWeb-2 ID | ODC-By 1.0 | Yes | Attribution + indication of changes | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 |
| SlimPajama (Stretch repo, EN) | Apache 2.0 (mirrors of original sources, redistribution permitted) | Yes | NOTICE-style attribution | https://huggingface.co/datasets/gmongaras/SlimPajama-627B_Reupload |

**Tier A (paper artifact)** — all 3 sources usable. Bartz v. Anthropic (N.D. Cal. June 2025) fair-use precedent applies for transformative LLM training (ML-training is "quintessentially transformative"). All sources legally acquired.

**Tier B1 (commercial-clean)** — same 3 sources; FineWeb-2 ODC-By and Apache 2.0 are commercial-friendly without further restriction. Wikipedia CC BY-SA share-alike means derivative model's training-data attribution must be retained, but Wikimedia accepts ML fair use per longstanding practice.

**Tier C (Indonesian-market commercial)** — defensible via UU Hak Cipta Pasal 43-44 fair-use-like provisions; not relevant for Phase E v1.8 since Tier 3 (tax-vertical) deferred to Phase F.

NOTICE file template (to be checked in alongside model release as `NOTICE_BILINGUAL_CORPUS_V1`):

```
This model was pretrained on a bilingual corpus comprising:

  Indonesian half:
    - Wikipedia (ID, snapshot 20231101) — CC BY-SA 4.0
    - FineWeb-2 (ind_Latn config, 10M-doc slice) — ODC-By 1.0

  English half (streaming):
    - SlimPajama-627B Reupload (gmongaras/SlimPajama-627B_Reupload)
      streamed at training time per Phase E v1.8 spec — Apache 2.0

Attribution preserved in NOTICE; share-alike provisions of Wikipedia
content carry forward to derivative training data per CC BY-SA 4.0
section 4(b)(iv).
```

---

## 3. Reproducibility surface

Per plan v1.7 §14.5 (4-tuple, DVC manifest dropped for $0 budget):

1. **git commit SHA** — this repo (`fajarkraton/fajarquant`)
2. **eval/HARNESS_SHA** — lm-eval pin `27988a293647d5853e48edea291640b4af54740c` (v0.4.11)
3. **Per-source `_manifest.json`** (committed) — `data/phase_e/corpus_id_dedup_exact/{HuggingFaceFW__fineweb-2,wikimedia__wikipedia}/_manifest.json` containing `bytes_text_kept`, `docs_kept`, `dedup_rate`, `sample_dropped`, normalization scheme (`norm=nfkc-lower-ws|alg=sha256`)
4. **HF dataset revisions** — pinned via streaming default at corpus-build time (Wikipedia 20231101 snapshot is stable; FineWeb-2 default is stable; SlimPajama-627B-Reupload is stable)

ID parquet shards regenerable end-to-end via:
```
make build-id-corpus-wikipedia-id   # ~5 min (665K articles)
make build-id-corpus-fineweb-2-id   # ~75 min (10M docs)
make dedup-corpus-id-exact          # ~20 min (sha256 over 10.67M docs)
```

EN streaming reproducible via `intllm_en.bilingual_mix_summary()` at training time — no separate corpus build step needed.

---

## 4. Synthetic-mix-ratio audit

Per plan v1.7 → v1.8 §14 R1: synthetic ratio cap is **≤50% of pretrain mix** (BeyondWeb canonical recipe). Phase E v1.8 ships with **0%** synthetic mix:

- **Lapis 1 (real corpus only):** 100% — 15.40 B ID + 10.27 B EN = 25.67 B real tokens
- **Lapis 2 (BeyondWeb rephrase):** 0% — DEFERRED to Phase F per plan v1.7 Option B
- **Lapis 3 (vertical FT):** 0% — DEFERRED to Phase F per plan v1.8 (Tier 3 → Phase F)

The `verify-bilingual-corpus` smoke gate asserts `synthetic_mix_ratio == 0.0` for v1.0. Future Phase F revision may bump to v2.0 with non-zero synthetic mix when E1.5 reactivates.

---

## 5. Verification gates (CLAUDE §6.8 R2 + R3)

### 5.1 R2 — runnable verification commands

```bash
make verify-bilingual-corpus     # 8/8 invariants in <1 s (R3 prevention layer)
make test-intllm-en              # 6/6 invariants for EN config shim
make test-dedup-exact            # 5/5 invariants for ID dedup script
make dedup-corpus-id-exact-dryrun  # 4 s sanity on 10K-doc cap
```

### 5.2 R3 — prevention layer

`make verify-bilingual-corpus` is the §E1.6 gate. Wired to pre-commit hook (follow-up E1.6.x sub-task) so any change to:
- `data/phase_e/corpus_id_dedup_exact/*/_manifest.json`
- `python/phase_e/intllm_en.py`
- This spec doc

triggers the verify script. Drift between spec and manifests = pre-commit fail.

### 5.3 What the verify script checks

| # | Invariant | Source of truth |
|---|---|---|
| 1 | Both ID manifests exist + parseable | `data/phase_e/corpus_id_dedup_exact/*/_manifest.json` |
| 2 | ID `bytes_text_kept` aggregate within ±0.5 GB of spec table §1.3 | manifests vs this doc §1.3 |
| 3 | ID `docs_kept` aggregate matches spec §1.3 exactly | manifests vs §1.3 |
| 4 | ID `dedup_rate` within ±0.005% of spec §1.3 | manifests vs §1.3 |
| 5 | EN cap from `intllm_en.bilingual_mix_summary()` matches spec §1.2 (10.27 B at 60:40) | shim vs §1.2 |
| 6 | EN repo selected = SLIMPAJAMA_STRETCH_REPO at 60:40 default | shim vs §1.2 |
| 7 | `synthetic_mix_ratio == 0.0` | this doc §4 |
| 8 | License attribution table covers all sources actually used (no missing rows) | this doc §2 vs manifests source list |

---

## 6. Phase E1 closure status (after E1.6 v1.0 scaffold)

| Sub-phase | Status (v1.8) | Closure artifact |
|---|---|---|
| E1.0 Pre-flight audit | ✅ CLOSED | `FJQ_PHASE_E_E1_FINDINGS.md` v1.2 |
| E1.1 Indonesian corpus assembly | ✅ CLOSED | First slice: Wikipedia + FineWeb-2; manifests in repo |
| E1.2 English subset selection | ✅ CLOSED | `python/phase_e/intllm_en.py` + smoke gate |
| E1.3 Vertical FT corpus | ⏭️ Phase F | `FJQ_PHASE_E_TAXPRIME_DATASET_SPEC.md` v1.2 (Phase F roadmap header) |
| E1.4 Dedup + filter | ✅ CLOSED | exact-hash sha256, 99.97% retention, manifests in repo |
| E1.5 Synthetic augmentation | ⏭️ Phase F (since v1.7) | `python/phase_e/prompts/LAPIS2_TEMPLATES.md` preserved |
| **E1.6 Final corpus packaging** | **🔄 IN PROGRESS (this doc v1.0 = scaffold)** | This doc + `verify-bilingual-corpus` Makefile target |

**E1.6 sub-task progress:**

- [x] **E1.6.0** — scaffold spec doc + `scripts/verify_bilingual_corpus.py` + Makefile target `verify-bilingual-corpus` (this commit)
- [ ] E1.6.1 — wire `verify-bilingual-corpus` into pre-commit hook (gates spec drift vs manifests)
- [ ] E1.6.2 — write `NOTICE_BILINGUAL_CORPUS_V1` file + add to repo root for future model release packaging
- [ ] E1.6.3 — Phase E1 closure commit + bump `FJQ_PHASE_E_E1_FINDINGS.md` to v1.3 with E1.6 closure block

When all 4 sub-tasks land, Phase E1 is fully closed and Phase E2 (algorithm catch-up) becomes the next active sub-phase.

---

*Document version: 1.0 (E1.6.0 scaffold). Author: Claude Opus 4.7 + Fajar (PrimeCore.id).*
*Phase E v1.8 Tier 1+2 scope. Tier 3 (tax-vertical) deferred to Phase F per plan v1.8.*
