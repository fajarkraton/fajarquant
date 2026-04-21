# IntLLM — Reproducibility Guide

> Phase 3.1 of `FJQ_PHASE_D_PRODUCTION_PLAN.md`. Lives in `docs/` (IntLLM
> paper) to avoid collision with `paper/REPRODUCIBILITY.md` (V26 FajarQuant
> paper — separate artifact).

This document specifies the exact software + hardware + data versions
needed to reproduce every numeric claim in the IntLLM paper. Drift from
any pinned version = reviewer-reproducibility failure; §6.9 R7 verify
script enforces a pre-publish check.

---

## 1. Software

### 1.1 Python environment

```bash
git clone https://github.com/fajarkraton/fajarquant.git
cd fajarquant
python -m venv .venv
source .venv/bin/activate
cd python/phase_d
pip install -e .          # installs intllm + runtime deps
```

Tested on Python 3.12.3. Minimum 3.11.

### 1.2 PyTorch / CUDA

```
torch >= 2.3.0, < 3.0
transformers >= 4.44, < 5.0   # DynamicCache incompat with 5.x; see C.P2.0 findings
```

CUDA 12.1+ for RTX 4090 (Phase D training hardware).

### 1.3 lm-evaluation-harness (pinned)

```bash
pip install "lm-evaluation-harness @ git+https://github.com/EleutherAI/lm-evaluation-harness.git@$(cat ../eval/HARNESS_SHA | grep '^ref:' | awk '{print $2}')"
```

Current pin: **v0.4.11** (see `eval/HARNESS_SHA`). Tag-based — will be
replaced with a full SHA in a future commit once CI resolves it.

Why pinned: lm-eval has a known 0.3→0.4 ARC-E normalization drift
(evidence in `docs/FJQ_PHASE_D_P2_1_GATE.md`). All Phase D numbers +
baseline re-runs use exactly this version for apples-to-apples.

### 1.4 Fajar Lang + FajarOS (kernel path)

For the `make test-intllm-kernel-path` regression gate:

```bash
git clone https://github.com/fajarkraton/fajar-lang.git
cd fajar-lang
cargo build --release --features "llvm,native"
# Binary at target/release/fj
```

```bash
git clone https://github.com/fajarkraton/fajaros-x86.git
cd fajaros-x86
make build-llvm               # produces build/fajaros-llvm.elf
make test-intllm-kernel-path  # 4-invariant gate in QEMU; <60 s
```

Pinned commits:
- fajar-lang: see `deps/FAJAR_LANG_SHA`   (TBD in Phase 5.1)
- fajaros-x86: see `deps/FAJAROS_SHA`      (TBD in Phase 5.1)

### 1.5 System dependencies

```
qemu-system-x86_64 >= 9.0
grub-mkrescue (xorriso + grub-common + grub-pc-bin)
build-essential (gcc, make)
e2fsprogs + dosfstools (for FS-adjacent test targets only — not required
  for IntLLM gate)
```

Debian/Ubuntu:

```bash
sudo apt install qemu-system-x86 xorriso grub-common grub-pc-bin \
                 build-essential python3-venv python3-pip
```

---

## 2. Hardware

### 2.1 Training

RTX 4090 (24 GB VRAM) is the reference training hardware. All Phase D
timing numbers assume this GPU. Per-config estimates from
`FJQ_PHASE_D_CONFIG.md` §3:

| Config  | Params | Tokens | RTX 4090 hours |
|---------|--------|--------|----------------|
| Mini    | 21.5M  | 500M   | ~1.8 h         |
| Base    | 46.4M  | 982M   | ~11.0 h (Chinchilla-optimal) |
| Medium  | 71.3M  | 2B     | ~11.0 h        |
| Stretch | 369.1M | 15B    | ~17 days       |

Other GPUs: expect ≤2× slowdown on RTX 3090, ≤4× on RTX 3080 10 GB (due
to gradient-checkpoint activation memory). A100 40 GB is roughly
equivalent to RTX 4090 for these sizes.

### 2.2 Inference / benchmarks

Any CUDA GPU with ≥8 GB VRAM for Medium/Stretch inference.
CPU-only inference works for Mini/Base at acceptable throughput (~20-40
tok/s on Ryzen 7950X3D).

### 2.3 Kernel-path regression gate

Any x86_64 machine with:
- `/dev/kvm` access (put user in `kvm` group) for ~5× speedup, OR
- QEMU TCG fallback (auto, no KVM needed; ~5× slower)

4 GB RAM + 10 GB disk sufficient.

---

## 3. Data

### 3.1 Training corpus

Mistral v3 tokenizer (Apache-2.0, 32K vocab) — `mistralai/Mistral-7B-v0.3`
from HuggingFace. Tokenizer only, not model weights. See
`FJQ_PHASE_D_P3_FINDINGS.md` for rationale.

Training data (streaming, no disk staging required):
- **Mini / Base / Medium:** `DKYoon/SlimPajama-6B` (≈6B tokens total).
- **Stretch:** `gmongaras/SlimPajama-627B_Reupload` (a re-upload of the
  full SlimPajama corpus; original `cerebras/SlimPajama-627B` is 404 as
  of 2026-04-21). ~627B tokens; Phase D consumes 15B.

Rationale: SlimPajama is the de-deduplicated RedPajama v2 — the same
data BitNet 2B4T was trained on. Enables apples-to-apples scaling-law
comparisons.

### 3.2 Evaluation corpora

lm-evaluation-harness auto-downloads these on first invocation:
- `wikitext` (PPL)
- `lambada_openai`
- `hellaswag`, `piqa`, `winogrande`
- `arc_easy`, `arc_challenge`, `openbookqa`
- `mmlu` (5-shot), `triviaqa` (5-shot), `boolq` (0-shot)

Cached at `~/.cache/huggingface/datasets/`.

### 3.3 Paper sample dataset (reviewer-facing)

The MLSys AE submission ships a ≤50 MB sample dataset in
`paper/intllm/sample_data/`:
- WikiText-2 slice (2 MB)
- 1-layer IntLLM weight (~30 MB)
- 100-token prompt set (10 KB)

Reviewers run `paper/intllm/run_smoke.sh` (Phase 5.3, TBD) to produce
one Table 2 row without downloading the full 60 GB SlimPajama.

---

## 4. Quick-start: reproduce the headline claims

### 4.1 Kernel path (no GPU, ~5 min)

```bash
cd fajaros-x86
make test-intllm-kernel-path
```

Expected: 4 invariants PASS, "Output (tokens): ..." visible in
`build/test-intllm-kernel-path.log`.

### 4.2 Mini model proof-of-life (RTX 4090, ~2 min)

```bash
cd fajarquant/python/phase_d
python scripts/train_mini.py --proof-of-life
```

Expected: loss 10.42 → ~6.26 (Δ ≈ 4.16 nat) in 200 steps (~62 s).

### 4.3 Full Mini training (RTX 4090, ~2 h)

```bash
python scripts/train_mini.py
```

Expected: val_loss ≤ 4.5 (Mini gate from `FJQ_PHASE_D_GATE_CALIBRATION.md`).

### 4.4 Canonical benchmarks (TBD Phase 3)

```bash
make bench-canonical MODEL=intllm-base
```

Produces `paper/intllm/results/bench_canonical_intllm-base.json` + a
human-readable report. Gated by `verify_intllm_tables.py --strict`.

---

## 5. Troubleshooting

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| `[ERR] Load failed: E2` during `model-load` | Bad version field in .fjm | Regenerate with current `intllm.export.export_fjm_v9` (V31.C.P1.5 fix) |
| `[ERR] Load failed: E3` | Wrong header offset in .fjm | Same fix |
| Mini loss plateaus at ~10.4 | Random-token input (no convergence possible) | Use `--overfit-batch` flag instead of default streaming |
| `OOM` at Stretch training | Gradient checkpoint disabled | Ensure `config.use_checkpoint=True`; lower batch; check `scripts/cleanup_hf_cache.py` hasn't run during training |
| `model-load nvme 0` hangs in QEMU | NVMe driver init race | Add `sleep 8` before `model-load` command in test scripts |

---

## 6. Changelog

| Date | Commit | Change |
|---|---|---|
| 2026-04-22 | (this commit) | Phase 3.1 initial — pinned lm-eval v0.4.11 + Mini/Base/Medium/Stretch hardware table + 4-step quick-start |
