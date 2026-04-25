# Phase E E0.0.8 — Dataset Versioning Strategy

> **Status:** PASS — DVC + S3 (AP-Southeast) chosen as primary; HF Datasets private as paper-public-sample sidecar.
> **Decision date:** 2026-04-25
> **Plan reference:** `FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.3 §3 E0.0.8 + §10.2 storage strategy

---

## 1. Volume estimate (Phase E datasets)

| Dataset | Raw token estimate | Compressed size | Storage class needed |
|---|---|---|---|
| Bilingual pretrain corpus (E1) — real ID + EN | 50–80 B tokens | **~120–200 GB** parquet | versioned, large, deduped |
| Bilingual pretrain synthetic (E1.5) — Gemma 4 rephrase | 5–10 B tokens | ~15–30 GB | versioned, generator-pinned |
| Tax-vertical corpus (E4 — TaxPrime) | ~10 M–100 M tokens | ~100 MB–1 GB | versioned, **PRIVATE**, UU PDP residency |
| Tax-vertical eval set (§11) | 500 prompts | ~5 MB | versioned, private, gold-labeled |
| Phase D English baseline (already cached) | ~50 B tokens | ~80 GB | already streaming, no re-storage |
| **TOTAL Phase E unique data** | **~75–105 B tokens** | **~135–230 GB** | mixed public/private |

## 2. Three options evaluated

### A. HF Datasets (private repos)

**Pros:**
- Native integration with `datasets.load_dataset()` — zero-friction loading
- Streaming support out-of-box (proven in Phase D `intllm.data`)
- Versioning via revision SHA / branch
- Free hosting up to repo size limits
- Gated dataset mode for private + access control

**Cons:**
- ⚠️ **Hosted on US infrastructure** — TaxPrime data residency (UU PDP Art. 56) requires Indonesia/AP-Southeast → blocks tax-corpus path
- Repo size limits (~hundreds of GB OK but >TB requires negotiation)
- Anthropic-style ToS concerns for HF as data steward — minimal but exists
- Re-pinning to specific revision requires SHA tracking

### B. DVC (Data Version Control) + S3/GCS backend

**Pros:**
- Works **on top of any blob storage** — choose AWS S3 ap-southeast-1 (Singapore) or ap-southeast-3 (Jakarta) for UU PDP residency
- Full content-addressable manifest in git (no LFS bloat)
- Standard git workflow + cheap blob storage
- Reproducibility: every commit pins exact data SHA
- No vendor lock-in (DVC files portable across S3/GCS/Azure)

**Cons:**
- Requires `dvc pull` step before training (vs HF auto-stream)
- Additional CI complexity (S3 credentials in CI)
- Slightly steeper learning curve than HF

### C. git-lfs

**Pros:**
- Simplest setup
- GitHub-native

**Cons:**
- ❌ **Cost prohibitive at scale** — GitHub LFS bandwidth $0.04/GB, storage $0.07/GB-month for >1GB packs
- ❌ **File size limits** — GitHub LFS hard 5GB per-file limit; many parquet shards needed
- ❌ **Public-by-default** — private LFS requires GitHub Pro/Enterprise
- ❌ **No data residency control** — GitHub infrastructure US/EU
- ❌ **Inferior reproducibility** — LFS smudge can drift if not pinned

## 3. Decision: **DVC + S3 (AP-Southeast) PRIMARY** + HF Datasets private SIDECAR

### Primary path: DVC + S3 ap-southeast (Singapore region)

For **all Phase E primary datasets:**
- Bilingual pretrain corpus (E1)
- Synthetic augmentation (E1.5)
- TaxPrime tax-vertical corpus (E4)

**Rationale:**
- Data residency UU PDP compliance (Singapore AWS region acceptable per Indonesian regulators for SEA-region; Jakarta region preferred for sensitive TaxPrime data)
- Full reproducibility — git-pinned SHA per dataset version
- Cost ~$0.023/GB-month S3 standard ≈ **~$5–10/month for 200 GB** (negligible)
- DVC ergonomics: `dvc pull <data>` before training, manifest in git for version-pin

### Sidecar: HF Datasets private (paper-public-sample only)

For **paper reproducibility artifact** (≤20 representative examples per dataset):
- Public-safe redacted samples
- HF Datasets gated repo
- Linked from paper §4 Reproducibility Appendix
- Allows reviewers to verify schema + format without accessing full proprietary data

## 4. Implementation plan

### 4.1 Repository structure

```
~/Documents/fajarquant/
├── data/                       # DVC-tracked
│   ├── bilingual_corpus_v1/    # E1 deliverable
│   ├── synthetic_v1/           # E1.5 deliverable
│   └── tax_id_corpus_v1/       # E4 deliverable (private)
├── .dvc/                       # DVC config
│   └── config                  # S3 ap-southeast-1 endpoint
├── data.dvc / *.dvc            # git-tracked manifest pointers
└── docs/REPRODUCIBILITY.md     # how to dvc pull from public/private
```

### 4.2 S3 bucket structure

```
s3://fajarkraton-phase-e/  (ap-southeast-1, Singapore)
├── public/
│   └── bilingual_corpus_v1/    # public-safe (CC-100, OSCAR, Wikipedia, MADLAD-400, FineWeb-2)
├── private/
│   ├── synthetic_v1/           # private-eyes-only (license-restricted source mix)
│   └── tax_id_corpus_v1/       # TaxPrime proprietary
└── samples/
    ├── bilingual_sample/       # ≤20-line samples for HF mirror
    └── tax_sample/             # redacted sample for HF mirror
```

**Access control:**
- `public/` — read-public, write-FajarQuant-team
- `private/` — read-FajarQuant-team-only, IAM-policy-gated
- `samples/` — public-readable, mirrored to HF Datasets

### 4.3 CI integration

`.github/workflows/phase_e_eval.yml` (planned for E5.3):

```yaml
- uses: iterative/setup-dvc@v1
- run: dvc pull data/bilingual_corpus_v1
  env:
    AWS_ACCESS_KEY_ID: ${{ secrets.S3_AP_SOUTHEAST_READ }}
    AWS_SECRET_ACCESS_KEY: ${{ secrets.S3_AP_SOUTHEAST_SECRET }}
    AWS_DEFAULT_REGION: ap-southeast-1
```

### 4.4 Cost estimate (annual)

| Component | Volume | Cost/year |
|---|---|---|
| S3 ap-southeast-1 storage standard | ~200 GB × 12 mo × $0.025/GB-mo | **~$60** |
| S3 egress to dev machine (occasional) | ~500 GB/year × $0.09/GB | **~$45** |
| S3 egress to cloud GPU (E1.5 generator) | ~30 GB × $0/GB (intra-AWS) | **$0** |
| DVC software | open source | **$0** |
| HF Datasets sidecar (samples) | ~50 MB free tier | **$0** |
| **Total Phase E annual data infrastructure** | — | **~$105/year** |

Negligible vs Phase E ~$420–820 total budget per plan v1.3 §4.

## 5. Reproducibility commitment

Every Phase E paper claim derived from datasets traces to:

```
1. git commit SHA (this repo)             — code + paper version
2. eval/HARNESS_SHA                        — lm-eval pin (E0.0.5)
3. data/<dataset>_v1.dvc manifest          — exact data SHA at use
4. .dvc/config                             — S3 bucket reference
5. paper/intllm/results/phase_e/*.json     — measurement output
6. samples/ on HF Datasets                  — public-verifiable sample
```

5-tuple traceability — paper reviewer can verify each claim against pinned data.

## 6. Open licensing decisions (deferred to E0.0.7)

Specific per-source license verification (CC-100 terms, OSCAR CC0, Wikipedia CC-BY-SA, news crawl licensing, etc.) and commercial-vs-research split is the scope of **E0.0.7** (next sub-task). E0.0.8 here decides the **storage + versioning mechanism**, not the legal licensing.

## 7. Acceptance criteria

- [x] One versioning approach picked (DVC + S3 PRIMARY, HF private sidecar)
- [x] Data residency UU PDP compliance addressed (ap-southeast-1 Singapore region)
- [x] Repository + bucket structure defined
- [x] CI integration sketched
- [x] Cost estimated (~$105/year — negligible)
- [x] Reproducibility 5-tuple specified
- [x] Public/private split clear (public corpus + private synthetic + private TaxPrime + public samples)

**Phase E E0.0.8 gate: PASS** (mechanical decision file landed per §6.8 R6).

7 of 10 E0 sub-tasks now closed.

---

*Decision committed 2026-04-25. Author: Claude Opus 4.7 + Fajar.*
*Implementation triggers: DVC `dvc init` + S3 bucket creation as part of E1.0 pre-flight (E1 phase work).*
*Companion: E0.0.7 will cover per-source data licensing on top of this storage decision.*
