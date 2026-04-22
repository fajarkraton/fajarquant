# IntLLM — MLSys Artifact Appendix

> Phase 5.1 of `FJQ_PHASE_D_PRODUCTION_PLAN.md`. Follows ACM Artifact
> Evaluation template v1.1. Companion to `docs/REPRODUCIBILITY.md`
> (which covers software / hardware / dataset versions in detail);
> this document is the AE-format summary.

---

## A.1 Abstract

IntLLM is a 1.58-bit ternary-weight, MatMul-Free language model
trained from scratch on SlimPajama-6B and deployed bit-exactly inside
the FajarOS Nova kernel. The artifact contains: the trained
checkpoints (Mini, Base, Medium, Stretch), the QAT training pipeline,
the FajarOS kernel build, the lm-evaluation-harness benchmarks, and
end-to-end gates that mechanically verify every numeric claim made
in the paper.

**Headline result:** first published per-layer fp16-vs-ternary parity
distribution for a MatMul-Free LM (Phase 3.5 differentiator vs BitNet
b1.58 2B4T which did not ship a parity gate).

---

## A.2 Artifact check-list (meta-information)

| Item | Value |
|---|---|
| **Algorithm** | IntLLM (HGRNBit + intllm.qat outlier handling) |
| **Program** | Python 3.11+, PyTorch ≥2.3, FajarOS Nova kernel |
| **Compilation** | LLVM-19 (Fajar Lang), GCC-13 (kernel runtime helpers) |
| **Transformations** | Ternary 1.58-bit weight quant + 8-bit activation quant via STE |
| **Binary** | `fajaros-llvm.elf` (1.45 MB) + `fajaros-llvm.iso` (boot CD) |
| **Model** | HGRNBit Mini (21.5M) / Base (46.4M) / Medium (71.3M) / Stretch (369M) |
| **Data set** | SlimPajama-6B streaming + WikiText-2 + LM-Eval Harness suite (8 zero-shot + mmlu + triviaqa) |
| **Run-time environment** | Linux ≥6.5 (Ubuntu 24.04 LTS tested), CUDA 12.1+ |
| **Hardware** | NVIDIA RTX 4090 (16 GB; Laptop GPU sufficient); 32 GB system RAM; 200 GB SSD |
| **Run-time state** | Deterministic per fixed prompt set (`python/phase_d/data/prompts.json`) |
| **Execution** | `bash run_smoke.sh` (≤2 min functional); `make verify-intllm-tables --strict` (≤1 s) |
| **Metrics** | val_loss / Wikitext-103 PPL / HellaSwag acc_norm / MMLU 5-shot / kernel tok/s + Watts |
| **Output** | JSON in `paper/intllm/results/`; serial logs in `build/test-*.log` |
| **Experiments** | Phase 2 training (Mini-Stretch); Phase 3 benchmarks; Phase 4.2 ablations; Phase 5 kernel deploy |
| **How much disk space required (approximately)?** | 60 GB for full Stretch checkpoint + dataset cache; 5 GB for Mini-only smoke |
| **How much time is needed to prepare workflow (approximately)?** | 30 min (clone + venv + pip + tokenizer download) |
| **How much time is needed to complete experiments (approximately)?** | Smoke: 2 min. Full Mini retrain: 2 h. Full Stretch retrain: 17 days. |
| **Publicly available?** | Yes — `github.com/fajarkraton/fajarquant` + `github.com/fajarkraton/fajaros-x86` |
| **Code licenses (if publicly available)?** | Apache-2.0 (paper artifacts); MIT (kernel); upstream MMfreeLM is Apache-2.0 |
| **Workflow framework used?** | Make targets + bash + pytest |
| **Archived?** | Will be Zenodo-DOI'd post-acceptance |

---

## A.3 Description

### A.3.1 How to access

```bash
git clone https://github.com/fajarkraton/fajarquant.git
git clone https://github.com/fajarkraton/fajaros-x86.git
```

Pinned commit SHAs live in:
- `fajarquant`: tag `v0.1.0-intllm-phase-d` (set at submission)
- `fajaros-x86`: tag `v3.8.0` (set 2026-04-21)

### A.3.2 Hardware dependencies

- **GPU:** NVIDIA Ampere or newer with ≥16 GB VRAM. Tested on RTX 4090
  Laptop (16 GB GDDR6X). Minimum: RTX 3060 12 GB (Mini retrain only).
- **CPU:** x86-64 with AVX2. Required for FajarOS Nova boot.
- **RAM:** 32 GB system memory recommended (16 GB minimum).
- **Disk:** 60 GB free for full retrain. 5 GB for smoke + Mini-only.
- **Network:** Required during initial setup (HuggingFace tokenizer +
  optional dataset download). Not required at run-time.

### A.3.3 Software dependencies

See `docs/REPRODUCIBILITY.md` §1 for exact pin versions. Quick form:

```
python      = 3.12.x  (≥3.11)
torch       = 2.10.x  (≥2.3, <3.0)
transformers= 4.46.x  (≥4.44, <5.0 — DynamicCache incompat)
triton      = 3.x.x   (auto via torch[cuda])
lm-eval     = v0.4.11 (PINNED, see eval/HARNESS_SHA)
mistral     = mistralai/Mistral-7B-v0.3 tokenizer (32K vocab)
QEMU        = 8.x.x   (kernel deploy gate)
LLVM        = 19      (Fajar Lang codegen — fajaros-x86 build)
```

### A.3.4 Data sets

- **Training:** `DKYoon/SlimPajama-6B` (Mini/Base/Medium) +
  `gmongaras/SlimPajama-627B_Reupload` (Stretch fallback). Streamed
  via `intllm.data.slimpajama_stream`; no local copy required.
- **Eval (canonical):** `wikitext-2-raw-v1` + 7 lm-eval-harness 0.4.11
  zero-shot tasks. ~50 MB after first download, cached locally.
- **Eval (knowledge):** `mmlu` 5-shot + `triviaqa` 5-shot + `boolq`.
  ~5 GB after first download.
- **Smoke prompt set:** `python/phase_d/data/prompts.json` —
  100 fixed prompts, CC0, committed in repo.

### A.3.5 Models

| Tag | Params | Tokens | val_loss | Status |
|---|---|---|---|---|
| `intllm-mini`     | 21.5M  | 491M  | 4.38   | trained, gate PASS |
| `intllm-base-c.2` | 46.4M  | 491M  | 4.13   | trained, gate PASS (sub-Chinchilla) |
| `intllm-base-c.1` | 46.4M  | 982M  | TBD    | training in flight 2026-04-22, gate <4.2 expected |
| `intllm-medium`   | 71.3M  | 2.0B  | TBD    | pending Base c.1 PASS |
| `intllm-stretch`  | 369M   | 15B   | TBD    | pending Medium PASS |

Checkpoints will be released via GitHub Releases under
`fajarkraton/fajarquant/releases/tag/v0.1.0-intllm-phase-d` at paper
submission time. ~85 MB Mini, ~180 MB Base c.1.

---

## A.4 Installation

```bash
cd fajarquant
python3 -m venv .venv
source .venv/bin/activate
cd python/phase_d
pip install -e .
cd -

# Optional: download tokenizer (one-off, ~250 MB)
python -c "from intllm.tokenizer import get_tokenizer; get_tokenizer()"
```

For the kernel-deploy gate:
```bash
cd ../fajaros-x86
make build-llvm                           # ~3 min
make test-intllm-kernel-path              # ~45 s, 4 invariants
```

---

## A.5 Experiment workflow

### A.5.1 Functional smoke (≤2 min)

```bash
bash run_smoke.sh
```

Asserts: env OK + 56 unit tests + 13 paper-claim verifier (--strict)
+ fp16-parity scaffold + Mini inference. Produces no per-paper
numbers — purely functional.

### A.5.2 Reproduce Phase 2 training (~2 h Mini, ~17 days Stretch)

```bash
cd python/phase_d
PYTHONPATH=. ../../.venv/bin/python scripts/train_mini.py
PYTHONPATH=. ../../.venv/bin/python scripts/train_base.py    # ~12-17 h
PYTHONPATH=. ../../.venv/bin/python scripts/train_medium.py  # ~11 h
PYTHONPATH=. ../../.venv/bin/python scripts/train_stretch.py # ~17 d
```

Each emits a `train_*_full_trace.json` + `checkpoints/<config>/<config>_final.pt`
+ stdout PASS/FAIL against the calibrated gate.

### A.5.3 Reproduce Phase 3 benchmarks (~10 h)

```bash
make bench-canonical    # Phase 3.2 — Wikitext + 7 zero-shot
make bench-knowledge    # Phase 3.3 — MMLU + TriviaQA + BoolQ
make bench-baselines    # Phase 3.4 — re-run BitNet, MMfreeLM, SmolLM2
make test-intllm-fp16-parity   # Phase 3.5 — IntLLM differentiator
```

(As of submission: Phase 3.2/3.3/3.4 are TODO stubs. Phase 3.5 is
shipped; threshold gate to be activated post-Base-c.1.)

### A.5.4 Reproduce Phase 5 kernel deploy

```bash
cd fajaros-x86
make test-intllm-kernel-path     # 4 invariants
make test-intllm-kernel-e2e      # full boot + ask "hello" (TBD)
```

---

## A.6 Evaluation and expected results

`make verify-intllm-tables --strict` exit 0 ⇔ every numeric claim in
the paper PDF matches the JSON artifact in `paper/intllm/results/`
within ±0.01 absolute. Coverage today (2026-04-22): 6 PASS / 7 PEND
out of 13 registered claims.

Reviewer expectation: at submission this will be 13/13 PASS. PEND
state is permitted only DURING active development.

---

## A.7 Experiment customization

- Train for fewer steps: `--n-steps 10000` on any `train_*.py`
- Skip val eval: `--proof-of-life` (also caps to 200 steps)
- Different batch: `--batch-size 4` (Base c.2 reproducer)
- Override device: `--device cpu` (will error in HGRNBit triton path)

---

## A.8 Methodology

Per CLAUDE.md §6.9 (Research Integrity Rules):
- All canonical benchmarks use lm-evaluation-harness v0.4.11 SHA
  pinned in `eval/HARNESS_SHA`.
- All quantitative paper claims gated by `verify_intllm_tables.py`.
- Outlier handling = top-K channel preservation + adaptive bit
  allocation + periodic γ_x re-cal (intllm/qat.py).
- Pre-publication audit: `verify-intllm-tables --strict` is a CI
  required check before any tag.

---

## A.9 Notes

- **GPU memory:** Stretch (369M) fits in 16 GB at batch=4 seq=2048.
  At seq=4096 you'll need 24 GB.
- **Disk pressure:** SlimPajama-627B subset for Stretch is ~60 GB.
  `scripts/cleanup_hf_cache.py --check-free --threshold-gb 30`
  guards against full-disk crashes.
- **Triton requires CUDA at import time** — there is no CPU fallback
  for the upstream HGRNBit forward path. CPU-only reviewers can run
  steps 2-4 of `run_smoke.sh` (env check, unit tests, verify gate)
  but will skip step 5 (Mini inference).
