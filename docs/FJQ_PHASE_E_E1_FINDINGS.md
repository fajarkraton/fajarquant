# Phase E E1 — Bilingual Corpus Findings (Indonesian half, first slice)

> **Status:** PARTIAL — E1.1 ID corpus assembly LANDED; E1.2 EN reuse pending; E1.3 tax/legal pending; E1.4 dedup pending; E1.5 synthetic DEFERRED to Phase F per plan v1.7.
> **Plan reference:** `FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.7 §3 E1.0–E1.6
> **Last updated:** 2026-04-26
> **Origin commits (E1.1.0–E1.1.3):** `ef7dab8` → `3ba0d41` → `bda7e1c` → `47a8892`

This doc is incrementally updated as each E1.x sub-task completes. Initial version closes E1.0 pre-flight + records E1.1 first slice (Wikipedia ID + FineWeb-2 ID).

---

## 1. Per-source corpus stats (E1.1 Indonesian)

All numbers from `data/phase_e/corpus_id/<source>/_manifest.json` written by `python/phase_e/scripts/build_id_corpus.py`. Token estimates use the E0.0.3 measured Mistral v3 32K rate of **2.527 bytes/token on Indonesian** (`paper/intllm/results/phase_e/e0_tokenizer_compression.json`).

| Source | Status | Docs | Raw text | Parquet | ~Tokens | Compression | Runtime | License |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Wikipedia ID (`wikimedia/wikipedia/20231101.id`) | ✅ FULL | 665,621 | 1.06 GB | 366 MB | 0.42 B | 2.90× | 3.5 min | CC BY-SA 4.0 |
| FineWeb-2 ID (`HuggingFaceFW/fineweb-2/ind_Latn`) | ✅ SLICE 1 (10M docs) | 10,000,000 | 37.86 GB | 14.60 GB | ~15.0 B | 2.59× | 75 min | ODC-By |
| **Subtotal on disk** | — | **10,665,621** | **38.92 GB** | **14.97 GB** | **~15.4 B** | — | **78.5 min** | — |

`_manifest.json` files are committed; raw `shard_*.parquet` files are gitignored under `.gitignore` rule `data/phase_e/corpus_id/**/shard_*.parquet`. Hash-pinning via DVC deferred per plan v1.7 (laptop-only $0 storage strategy).

## 2. Verdict vs plan §E1.1 budget

| Gate | Threshold | Current | Status |
|---|---:|---:|---|
| Plan §E1.1 ID target | 12–18 B post-dedup | ~15.4 B raw → ~9.2 B at 60% retention worst case → ~13.1 B at typical 85% | ✅ **MET** at typical retention; tight at worst case |
| Plan §2 abort threshold | > 8 B post-dedup | 9.2–13.1 B post-dedup band | ✅ **CLEARED COMFORTABLY** |

**Implication:** Tier A pretrain corpus is foundational with just Wikipedia + FineWeb-2 slice 1. The remaining 4 ID sources (OSCAR / CulturaX / MADLAD-400 / CC-100) are now **quality + diversity additions, not budget blockers**.

## 3. Remaining ID sources — blocker status (E1.1.2 dry-run)

| Source | Plan estimate | Blocker | Workaround |
|---|---:|---|---|
| CC-100 ID | 10–15 B | HF removed Python script datasets (`cc100.py`) | statmt.org tarball mirror (E1.1.x future) |
| OSCAR v23.01 ID | 5–8 B | HF gated; click-through accept needed | User: visit `huggingface.co/datasets/oscar-corpus/OSCAR-2301` and "Agree" |
| CulturaX ID | 8–20 B | HF gated; click-through accept needed | User: visit `huggingface.co/datasets/uonlp/CulturaX` and "Agree" |
| MADLAD-400 ID | 5–10 B | 99,272 siblings; HF Datasets config-resolve > 90 s | `HfApi().list_repo_files` + glob `data/*/id/*.jsonl.gz` direct path (E1.1.x future) |
| FineWeb-2 ID slice 2+ | +35–55 B est | None — script supports `--max-docs`, but streaming restarts at doc 0 | Add `--skip-docs N` flag for incremental slices (E1.1.x future) |

User-action blockers (OSCAR + CulturaX) are 1-click each, ~5 min total. Once accepted, `python python/phase_e/scripts/build_id_corpus.py --source oscar_2301_id` and `--source culturax_id` will proceed.

## 4. Dedup + language-detect (DEFERRED to E1.4)

Per plan §3 E1.4, dedup runs MinHash LSH at threshold 0.85 across all ID sources jointly, after ingestion. Currently NOT measured:

- **Per-source intra-dedup rate:** unknown — Wikipedia is canonical (low expected); FineWeb-2 already deduplicated upstream by HuggingFace (`minhash_cluster_size` field present in source schema, dropped by our script — see §6 below).
- **Cross-source dedup rate:** unknown until E1.4 runs.
- **Language-detect accuracy:** unmeasured. FineWeb-2 ind_Latn is pre-filtered to ID Latin script (`language_score` p50 ~0.99 from dry-run sample). Wikipedia ID is canonical-by-construction.

**Estimated post-dedup retention:** 60–85% based on typical web corpus dedup rates (Falcon-Edge / RedPajama recipe). Lower bound chosen for budget verdict in §2.

## 5. Schema coverage

Both sources written with **two columns only**: `text` (str), `source` (str). Other source-side metadata is dropped by `build_source()` in `python/phase_e/scripts/build_id_corpus.py:128`. Specifically discarded:

- Wikipedia: `id`, `url`, `title`
- FineWeb-2: `id`, `dump`, `url`, `date`, `file_path`, `language`, `language_score`, `language_script`, `minhash_cluster_size`, `top_langs`

**Trade-off accepted:** simpler downstream tokenization + smaller parquet shards vs losing per-doc dedup signal already computed by HuggingFace (`minhash_cluster_size`). E1.4 will recompute dedup ourselves anyway; URL/date provenance lost is acceptable per plan §15 attribution-at-source-level (license file in §E1.6 packaging, not per-doc).

## 6. Verification commands (CLAUDE §6.8 R2)

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

## 7. Decisions logged at this version

1. **FineWeb-2 ID capped at 10M docs** for first slice. Full ind_Latn (32 train files × 4.5 GB = 144 GB source) would extrapolate to ~50–70 B tokens, well above plan upper estimate. Cap chosen to fit laptop-only disk budget per plan v1.7 hardware lock; 15 B from one slice ALREADY meets §E1.1 target.
2. **OSCAR / CulturaX defer to user click-through.** Not blocking corpus completion; nice-to-have for diversity.
3. **CC-100 ID and MADLAD-400 ID workarounds DEFERRED.** Neither is required for §E1.1 budget. Re-evaluate at §E1.6 packaging time if quality/diversity audit demands more breadth.
4. **Cosmetic SIGABRT in FineWeb-2 download.** `PyGILState_Release: thread state must be current when releasing` from C-extension shutdown order quirk (HF Datasets + aiohttp + pyarrow). Crash occurred AFTER all data + manifest written; manifest reports `error: null`; read-back of all 142 shards verifies 10,000,000 rows. Documented for future hardening; no data loss this run.

## 8. Open items for E1.x continuation

- [ ] E1.2 — English subset selection: reuse Phase D `intllm.data.slimpajama_stream` (~50 B available). Pure registry/config work, no new download.
- [ ] E1.3 — Vertical fine-tune corpus (tax/legal Indonesian): requires TaxPrime archive access via founder; small total volume (~150–700 MB).
- [ ] E1.4 — Dedup + language-detect + filter: MinHash LSH script not yet written. Operates on the 10.67M docs currently checked in.
- [ ] E1.5 — Synthetic pretrain augmentation: **DEFERRED TO PHASE F** per plan v1.7 (Option B aggressive $0). Real corpus alone sufficient; lose BeyondWeb 7.7× speedup.
- [ ] E1.6 — Final corpus packaging: `FJQ_PHASE_E_BILINGUAL_CORPUS_V1.md` with attribution + license file. Pre-commit hook `verify-bilingual-corpus` to write.
- [ ] OPTIONAL E1.1.x — `--skip-docs N` flag for FineWeb-2 incremental slices.
- [ ] OPTIONAL E1.1.x — MADLAD-400 ID via `HfApi().list_repo_files` glob workaround.
- [ ] OPTIONAL E1.1.x — CC-100 ID via statmt.org tarball mirror (lower priority due to non-commercial license).

---

**Sign-off this version:** Phase E E1.0 pre-flight CLOSED. E1.1 first slice LANDED with budget cleared. Continue with E1.2 (registry only) or E1.4 (dedup) at next session per founder decision.
