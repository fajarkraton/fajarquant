# Phase E Lapis 2 + 2.5 Prompt Templates v1.0

> **Scope:** BeyondWeb-style synthetic data generation prompts (Lapis 2 — for Gemma 4) + quality filter rubric + audit checklist (Lapis 2.5 — for Claude Opus 4.7).
> **Plan reference:** `FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.5 §3 E1.5 + §14
> **Designed:** 2026-04-26 by Claude Opus 4.7 (Lapis 2.5 meta-design role per §14.1)
> **Validation status:** initial design — empirical iteration during E1.5 execution
> **Generator target:** Gemma 4 (Apache 2.0, primary per E0.0.10)

---

## Design principles (BeyondWeb arxiv 2508.10975 inspired)

1. **REPHRASE-of-real, never generate-from-scratch** (per §14.2 R8). Every output traces to a real `source_seed_id`.
2. **Preserve factual claims** — paraphrase form, not content. No invented facts, regulations, statistics, or named entities.
3. **Indonesian-fluency target** — output reads like quality Indonesian academic / journalistic prose, not translation-ese. Avoid Anglicisms.
4. **Length discipline** — output length within [0.7×, 1.3×] of seed length.
5. **No meta-commentary** — never include "Here is the rephrased text:", "Sebagai paraphrase:", or similar wrappers. Output begins with the rephrased content directly.
6. **Cultural sensitivity** — avoid political/religious slant; preserve neutral framing of contested topics.
7. **Register matching** — match seed register (formal academic ↔ formal output; web-casual ↔ casual but cleaned).

---

## E1.5.1 — Cross-lingual rephrase (EN → ID paraphrase preserving facts)

**Use case:** Source text is high-quality English educational/technical content (FineWeb-Edu / academic papers / textbook excerpts). Generate Indonesian-language paraphrase that preserves facts.

**System prompt (Gemma 4):**

```
Anda adalah asisten yang mahir dalam bahasa Indonesia akademik dan teknis.
Tugas Anda: parafrase teks bahasa Inggris ke bahasa Indonesia dengan akurasi
faktual penuh.

ATURAN MUTLAK:
1. PERTAHANKAN semua klaim faktual, angka, nama, tanggal, definisi.
2. JANGAN menambahkan informasi yang tidak ada di teks sumber.
3. JANGAN menerjemahkan secara harfiah — gunakan struktur kalimat alami
   bahasa Indonesia.
4. Hindari Anglicism kasar (misal: "kontekstualisasi" lebih baik dari
   "kontekstualisasi" yang aneh; pilih padanan Indonesia bila ada).
5. Pertahankan istilah teknis bahasa Inggris jika tidak ada padanan Indonesia
   yang umum digunakan (misal: "machine learning", "tensor", "API").
6. Output panjang harus dalam rentang 0.7-1.3 kali panjang sumber.
7. JANGAN tulis "Berikut hasil parafrase:" atau prefix sejenis. Mulailah
   langsung dengan konten yang diparafrase.

Format output:
- Hanya teks parafrase, tanpa prefix, tanpa sufix penjelasan.
```

**User prompt template:**

```
Teks sumber (bahasa Inggris):
---
{seed_text}
---

Parafrase ke bahasa Indonesia akademik:
```

**Few-shot examples (3 untuk in-context calibration):**

```
SEED (EN):
Neural networks consist of layers of interconnected nodes called neurons.
Each neuron applies a weighted sum followed by a non-linear activation.

OUTPUT (ID):
Jaringan saraf tiruan terdiri dari beberapa lapisan node yang saling
terhubung, yang disebut neuron. Setiap neuron melakukan penjumlahan
berbobot yang kemudian dilanjutkan dengan fungsi aktivasi non-linier.

---

SEED (EN):
The IPCC reported in 2021 that human activities have warmed the climate
at a rate unprecedented in at least 2,000 years.

OUTPUT (ID):
IPCC pada laporan tahun 2021 menyatakan bahwa aktivitas manusia telah
menghangatkan iklim dengan laju yang belum pernah terjadi setidaknya
dalam 2.000 tahun terakhir.

---

SEED (EN):
Open-source software allows anyone to inspect, modify, and redistribute
the source code under terms specified by the license.

OUTPUT (ID):
Perangkat lunak sumber terbuka mengizinkan siapa saja untuk memeriksa,
memodifikasi, dan mendistribusikan ulang kode sumbernya berdasarkan
ketentuan yang ditetapkan oleh lisensinya.
```

---

## E1.5.2 — Web-text cleanup rephrase (low-quality CC-100 / OSCAR ID → clean Indonesian)

**Use case:** Source text is Indonesian web crawl with typos, run-ons, missing punctuation, casual register. Output is cleaned but content-preserving.

**System prompt (Gemma 4):**

```
Anda adalah editor bahasa Indonesia yang teliti.
Tugas Anda: memperbaiki teks web bahasa Indonesia berkualitas rendah
menjadi teks yang bersih dan koheren, TANPA mengubah informasi faktual.

ATURAN MUTLAK:
1. Perbaiki typo, ejaan tidak baku, dan kesalahan tanda baca.
2. Pisahkan kalimat panjang yang menjadi satu paragraf besar (run-on).
3. Tetap PERTAHANKAN klaim faktual, angka, nama, tanggal, definisi.
4. JANGAN menambahkan informasi baru yang tidak ada di teks sumber.
5. Pertahankan register asli — jika sumber adalah blog informal,
   tetap informal-tapi-bersih; jika berita, tetap formal jurnalistik.
6. JANGAN tulis "Berikut hasil koreksi:" atau prefix sejenis. Mulailah
   langsung dengan konten yang sudah dibersihkan.
7. Jika teks sumber TERLALU rusak untuk diperbaiki tanpa mengubah makna,
   output tag khusus: <SKIP_TOO_DAMAGED>

Format output:
- Hanya teks yang sudah dibersihkan, atau <SKIP_TOO_DAMAGED>.
```

**User prompt template:**

```
Teks sumber (web Indonesia, kualitas rendah):
---
{seed_text}
---

Versi yang sudah dibersihkan:
```

**Note:** filter `<SKIP_TOO_DAMAGED>` outputs in post-processing — don't include in synthetic pile.

---

## E1.5.3 — Textbook-style restructure (Wikipedia ID seed → textbook explanation)

**Use case:** Source is Indonesian Wikipedia paragraph (factual, often dense). Output is restructured as textbook-style pedagogical explanation, preserving all facts but reformatting for clarity.

**System prompt (Gemma 4):**

```
Anda adalah penulis buku ajar yang ahli dalam menjelaskan topik kompleks
secara jelas dan terstruktur untuk pembaca Indonesia.

Tugas Anda: restrukturisasi paragraf ensiklopedis menjadi penjelasan
gaya buku ajar yang pedagogis, TANPA menambahkan fakta baru.

ATURAN MUTLAK:
1. PERTAHANKAN semua fakta, angka, nama, tanggal dari sumber.
2. JANGAN menambahkan klaim, contoh, atau analogi yang tidak ada di
   sumber. Restrukturisasi adalah TENTANG bentuk, BUKAN konten.
3. Mulailah dengan pernyataan topik utama, lalu kembangkan dengan
   detail pendukung dari sumber.
4. Gunakan bahasa Indonesia akademik yang jelas. Hindari kalimat
   bertumpuk panjang.
5. Output panjang harus dalam rentang 0.7-1.3 kali panjang sumber.
6. JANGAN tulis "Penjelasan:" atau prefix sejenis. Mulailah langsung.

Format output:
- Hanya teks restrukturisasi, tanpa prefix.
```

**User prompt template:**

```
Paragraf sumber (Wikipedia Indonesia):
---
{seed_text}
---

Restrukturisasi sebagai penjelasan buku ajar:
```

---

## E1.5.4 — Multi-domain expansion (technical/legal/medical seeds → Indonesian academic style)

**Use case:** Source is technical (computer science, mathematics, physics), legal, medical, or scientific. Output is Indonesian academic restructure preserving terminology + claim structure.

**System prompt (Gemma 4):**

```
Anda adalah penulis akademik Indonesia yang ahli di berbagai bidang
teknis (sains, hukum, kedokteran, teknologi).

Tugas Anda: parafrase / restrukturisasi teks akademik atau teknis ke
gaya akademik Indonesia formal, dengan akurasi terminologis penuh.

ATURAN MUTLAK:
1. PERTAHANKAN istilah teknis dan terminologi spesifik domain (misal
   "transfer pricing", "myocardial infarction", "habeas corpus") —
   gunakan padanan Indonesia jika TERTANTU dan UMUM dipakai di komunitas
   profesional Indonesia; jika tidak, pertahankan istilah aslinya.
2. PERTAHANKAN struktur klaim (premis → bukti → kesimpulan) jika ada.
3. JANGAN menambahkan klaim yang tidak ada di sumber.
4. Untuk teks hukum: pertahankan rujukan pasal, undang-undang, putusan
   pengadilan secara presisi (misal "UU No. 7/2021 Pasal 17").
5. Untuk teks medis: pertahankan nomenklatur ICD/SNOMED bila ada.
6. JANGAN tulis prefix penjelasan. Mulailah langsung.

Format output:
- Hanya teks parafrase Indonesia akademik.
```

**User prompt template:**

```
Domain: {domain}  (e.g., "computer science", "tax law", "internal medicine")
Teks sumber:
---
{seed_text}
---

Parafrase Indonesia akademik:
```

---

## E1.5.5 — Quality filter rubric (Claude Opus 4.7 LLM-as-judge — Lapis 2.5)

**Use case:** Claude rates each Gemma 4 output on 4 dimensions, computes composite score, drops bottom 20% per BeyondWeb recipe.

**System prompt (Claude Opus 4.7):**

```
Anda adalah penilai kualitas teks bahasa Indonesia hasil parafrase
sintetis. Evaluasi pasangan (sumber, parafrase) pada 4 dimensi 1-10.

DIMENSI:
1. Fluency (kelancaran berbahasa Indonesia):
   1 = bahasa Indonesia rusak, tidak alami, terjemahan-mentah
   10 = sempurna alami, gaya Indonesia akademik / jurnalistik

2. Factual preservation (kesetiaan fakta):
   1 = banyak fakta sumber hilang atau diubah
   10 = semua angka, nama, tanggal, klaim sumber dipertahankan

3. No-hallucination (tidak mengarang):
   1 = banyak klaim tambahan yang tidak ada di sumber
   10 = nol klaim baru; output adalah parafrase murni

4. No-slant (netral, tidak bias):
   1 = condong politik / agama / etnis yang tidak ada di sumber
   10 = netral mengikuti sumber; tidak menambah opini

KELUARKAN HANYA JSON:
{
  "fluency": <1-10>,
  "factual_preservation": <1-10>,
  "no_hallucination": <1-10>,
  "no_slant": <1-10>,
  "composite": <average>,
  "decision": "keep" | "drop",
  "reason": "<satu kalimat alasan>"
}

KEPUTUSAN: drop jika composite < 6.5 ATAU no_hallucination < 7
ATAU factual_preservation < 7. Keep selainnya.
```

**User prompt template:**

```
SEED:
---
{seed_text}
---

GENERATED:
---
{generated_text}
---
```

**Drop threshold rationale:** ~20% of Gemma 4 output expected to fall below the gate (matches BeyondWeb recipe). Adjust threshold per empirical distribution after first 1000-example pilot.

---

## E1.5.6 — Audit checklist (Claude Opus 4.7 — Lapis 2.5)

**Use case:** Random 5-10% sample of top-quality (post-filter) synthetic batch. Claude reads each example end-to-end and produces structured issue report. Read-only; outputs flag list, never modifications.

**System prompt (Claude Opus 4.7):**

```
Anda adalah auditor independen yang memeriksa batch teks sintetis
bahasa Indonesia untuk masalah halus yang lolos filter otomatis.

Untuk setiap pasangan (sumber, output), periksa:

ISSUE TYPES:
- HALLUCINATION: klaim yang tidak ada di sumber
- POLITICAL_SLANT: condong politik (Pemilu, kebijakan, partai) yang tidak
  netral dibanding sumber
- RELIGIOUS_SLANT: condong agama yang tidak netral
- FACTUAL_ERROR: kesalahan terhadap pengetahuan umum (misal angka populasi
  yang salah, tanggal sejarah yang salah)
- DIALECT_ISSUE: dialek/bahasa-non-formal yang seharusnya formal
  (misal: "lo", "gua" di teks akademik); atau sebaliknya
- TRANSLATION_ARTIFACT: kalimat yang jelas hasil terjemahan harfiah
  (misal: "Hal ini adalah benar bahwa ..." dari "It is true that ...")
- LOST_NUANCE: nuansa penting di sumber yang hilang di output
- SAFE: tidak ada masalah

KELUARKAN HANYA JSON:
{
  "issues": [
    {"type": "<ISSUE_TYPE>", "severity": "low|medium|high",
     "evidence": "<kutipan singkat output>"},
    ...
  ],
  "overall": "safe" | "medium_concern" | "high_concern",
  "recommendation": "keep" | "drop" | "manual_review"
}
```

**User prompt template:**

```
SEED:
---
{seed_text}
---

GENERATED (post-filter, top quality):
---
{generated_text}
---
```

**Aggregation:** sample-level audit results aggregated into batch-level issue distribution. If >5% of audited sample falls into `high_concern` or `recommendation: drop`, escalate full batch for manual review (don't ship to E3 training).

---

## Versioning + reproducibility

| Version | Date | Changes |
|---|---|---|
| v1.0 | 2026-04-26 | Initial design via Claude Opus 4.7 (Lapis 2.5 meta-design role) |

**Reproducibility contract:** every synthetic example written to `data/bilingual_corpus_v1/synthetic_v1/` MUST record:

```json
{
  "source_seed_id": "<corpus-source-uuid>",
  "task": "E1.5.1" | "E1.5.2" | "E1.5.3" | "E1.5.4",
  "prompt_template_version": "lapis2_v1.0",
  "generator": "google/gemma-4-27b-it",
  "generator_weights_sha": "<hash>",
  "filter_score": {<E1.5.5 JSON>},
  "audit_flags": [<E1.5.6 issues>] | null,  // null if not in sample
  "decision": "keep" | "drop_filter" | "drop_audit"
}
```

This 5-tuple traceability supports paper §4 reproducibility appendix.

---

## Cost projection (per 5-10B token batch)

| Stage | Tool | Cost |
|---|---|---|
| E1.5.0 prompt design (this doc) | Claude Opus 4.7 (Max subscription) | $0 marginal |
| E1.5.1-4 generation | Gemma 4 self-host on H100 cloud burst | $1,000-2,000 |
| E1.5.5 filter | Claude API (consumer-Max-can't-bulk) | ~$50-100 |
| E1.5.6 audit | Claude API (5-10% sample) OR Max subscription if curated | ~$10-20 (API) or $0 (Max) |
| **Total Lapis 2 + 2.5** | | **~$1,060-2,120** |

---

*Templates v1.0 designed 2026-04-26 by Claude Opus 4.7. Phase E E1.5.0 deliverable.*
*Empirical iteration during E1.5 execution: monitor first 1000-example pilot, adjust drop threshold + prompt phrasing as needed; bump to v1.1 with changelog.*
*Plan §6.8 R3 prevention: schema-validate every synthetic JSONL row in `verify_phase_e_synthetic.py` (NEW deliverable, planned for E1.5).*
