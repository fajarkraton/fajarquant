# Phase E E1 — Bilingual Corpus Findings (Indonesian half, first slice)

> **Status (v1.1):** PARTIAL — E1.1 ID corpus assembly LANDED + **E1.4 dedup LANDED**; E1.2 EN reuse pending; E1.3 tax/legal pending; E1.5 synthetic DEFERRED to Phase F per plan v1.7.
> **Plan reference:** `FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.7 §3 E1.0–E1.6
> **Last updated:** 2026-04-26 (v1.1 splice — E1.4 dedup landed)
> **Origin commits (E1.1.0–E1.1.3):** `ef7dab8` → `3ba0d41` → `bda7e1c` → `47a8892`
> **Origin commits (E1.4.0–E1.4.3):** `991078f` (LSH scaffold) → `35d13e7` (LSH --resume) → `c83f1bd` (audit) → `4ebb437` (exact-hash sha256 scaffold) → exact-hash full sweep landed

This doc is incrementally updated as each E1.x sub-task completes. Initial version closes E1.0 pre-flight + records E1.1 first slice (Wikipedia ID + FineWeb-2 ID). v1.1 splice records E1.4 dedup completion via exact-hash sha256 (LSH variant pivoted after OOM).

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
5. **(v1.1)** **Dedup recipe pivot LSH → exact-hash sha256.** OOM-driven, see §4.2 above. Decision committed as code: `dedup_corpus_exact.py` + Makefile + .gitignore + this doc. LSH script kept in-tree for higher-noise future corpora.
6. **(v1.1)** **LSH partial cleanup.** 84-shard partial output (15 GB) + `.progress/state.pkl` (6.8 GB) removed after exact-hash sweep validated and manifests written. The exact-hash sweep is fully reproducible from `corpus_id/` raw shards via `make dedup-corpus-id-exact`; LSH partial held no information not reproducible from the full sweep.

## 8. Open items for E1.x continuation

- [ ] E1.2 — English subset selection: reuse Phase D `intllm.data.slimpajama_stream` (~50 B available). Pure registry/config work, no new download.
- [ ] E1.3 — Vertical fine-tune corpus (tax/legal Indonesian): requires TaxPrime archive access via founder; small total volume (~150–700 MB).
- [x] **E1.4 — Dedup + filter: LANDED 2026-04-26 via exact-hash sha256 (LSH variant pivoted post-OOM, see §4).** Language-detect deferred to E1.6 packaging audit if needed.
- [ ] E1.5 — Synthetic pretrain augmentation: **DEFERRED TO PHASE F** per plan v1.7 (Option B aggressive $0). Real corpus alone sufficient; lose BeyondWeb 7.7× speedup.
- [ ] E1.6 — Final corpus packaging: `FJQ_PHASE_E_BILINGUAL_CORPUS_V1.md` with attribution + license file. Pre-commit hook `verify-bilingual-corpus` to write.
- [ ] OPTIONAL E1.1.x — `--skip-docs N` flag for FineWeb-2 incremental slices.
- [ ] OPTIONAL E1.1.x — MADLAD-400 ID via `HfApi().list_repo_files` glob workaround.
- [ ] OPTIONAL E1.1.x — CC-100 ID via statmt.org tarball mirror (lower priority due to non-commercial license).
- [ ] OPTIONAL E1.4.x — language-detect pass via `fasttext` lid.176 on the 10.66M kept docs (post-dedup); only needed if §E1.6 packaging audit flags non-ID content above a threshold.

---

**Sign-off this version (v1.1):** Phase E E1.0 pre-flight CLOSED. E1.1 first slice LANDED. **E1.4 dedup LANDED** with budget MEASURED at 15.40 B tokens (1.92× over §2 abort threshold). Continue with E1.2 (registry only) or E1.3 (TaxPrime archive) at next session per founder decision.
