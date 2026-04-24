# IntLLM Baseline Comparisons — Unported-Features Disclosure

> **Purpose:** Make every baseline → IntLLM comparison in the paper auditable per CLAUDE.md §6.9 R3 (no strawman baselines — published features must be ported, otherwise the divergence must be disclosed and caveated). This doc is the **public-facing ledger** read by referees, artifact evaluators, and future readers.
>
> **Rule citations:** CLAUDE.md §6.9 R3 (baseline parity), R1 (canonical protocol from ≥2 refs), R7 (pre-publication audit gate)
>
> **Scope:** §3.4 of `FJQ_PHASE_D_PRODUCTION_PLAN.md` (`make bench-baselines` + `scripts/bench_baselines.py`) across 7 baseline repos. Audit trail in `docs/FJQ_PHASE_D_3_4_B0_FINDINGS.md`.
>
> **Invariant (enforced by `verify_intllm_tables.py --strict`, Phase 4):** Any paper cell comparing an IntLLM metric to a baseline in this doc MUST cite the relevant row — either by structural footnote flag (BitNet's `sft_lift_caveat`) or by naming the tokenizer column (BPB instead of PPL). Violation ⇒ pre-commit hook rejects the commit.

---

## 1. Baselines in scope

7 HuggingFace repos, all confirmed accessible 2026-04-24 (see FJQ_PHASE_D_3_4_B0_FINDINGS.md §2):

| Repo | Params | HF arch class | Training regime | Tokenizer family | Default batch |
|---|---|---|---|---|---|
| `microsoft/bitnet-b1.58-2B-4T` | 2.0 B | `bitnet` (native tfm ≥ 4.54) | **pretrain + SFT + DPO** | Llama-3, 128,256 | 2 |
| `ridger/MMfreeLM-370M` | 370 M | `hgrn_bit` (trust_remote_code) | pretrain-only | Mistral v3, 32,768 | 4 |
| `ridger/MMfreeLM-1.3B` | 1.3 B | `hgrn_bit` (trust_remote_code) | pretrain-only | Mistral v3, 32,768 | 4 |
| `ridger/MMfreeLM-2.7B` | 2.7 B | `hgrn_bit` (trust_remote_code) | pretrain-only | Mistral v3, 32,768 | 2 |
| `HuggingFaceTB/SmolLM2-135M` | 135 M | `llama` (native) | pretrain-only (base, not `-Instruct`) | SmolLM2, 49,152 | 4 |
| `HuggingFaceTB/SmolLM2-360M` | 360 M | `llama` (native) | pretrain-only (base, not `-Instruct`) | SmolLM2, 49,152 | 4 |
| `EleutherAI/pythia-160m` | ~160 M | `gpt_neox` (native) | pretrain-only | gpt-neox, 50,304 | 4 |

IntLLM (ours) is **pretrain-only**, Mistral v3 tokenizer (32,768). These values drive the two cross-cutting caveats in §2.

---

## 2. Cross-cutting caveats

### 2.1 The SFT/DPO lift (affects BitNet 2B4T row only)

- **Fact:** No pretrain-only BitNet 2B4T checkpoint exists on HuggingFace. Probed `microsoft/bitnet-b1.58-2B-pretrain`, `…-base`, `…-4T-base` on 2026-04-24 — all 401/not found. The available variants (`microsoft/bitnet-b1.58-2B-4T`, `…-bf16`, `…-gguf`) all ship weights **after** the BitNet 2B4T Technical Report's 3-stage pipeline (pretrain → SFT → DPO, per §2 of arXiv 2504.12285).
- **Consequence:** Direct zero-shot-accuracy comparison of IntLLM (pretrain-only) vs BitNet 2B4T (pretrain + SFT + DPO) is **not** apples-to-apples on any task that SFT/DPO lifts — notably CommonsenseQA, TruthfulQA, GSM8K, HumanEval+, MMLU, MT-bench, IFEval. Continuation LM loss (PPL / BPB on WikiText, LAMBADA) is only mildly affected by instruction tuning and remains fair.
- **Disclosure enforcement:** `scripts/bench_baselines.py` records `sft_lift_caveat: true` on the BitNet row. Paper Table 2 MUST carry a footnote identifying the BitNet row as "post-SFT+DPO" when reporting zero-shot accuracies; the Phase 4 `verify_intllm_tables.py --strict` hook rejects any Table 2 commit that reports BitNet zero-shot accuracy without the footnote.

### 2.2 The tokenizer-family mismatch (affects PPL, not accuracy)

Five distinct tokenizers are in play:

| Tokenizer family | Vocab | Used by |
|---|---|---|
| Mistral v3 | 32,768 | IntLLM (ours), MMfreeLM (all sizes) |
| Llama-3 | 128,256 | BitNet 2B4T |
| SmolLM2 (custom BPE) | 49,152 | SmolLM2-135M, SmolLM2-360M |
| gpt-neox | 50,304 | Pythia-160m |

Absolute PPL is not directly comparable across tokenizer families (PPL ~ exp(loss), and loss scales with the tokenizer's bits-per-subword). The lm-evaluation-harness `bits_per_byte` metric on WikiText is tokenizer-invariant and is the correct headline measure.

**Disclosure enforcement:**
- `scripts/bench_baselines.py` records `tokenizer_family` in the JSON artifact.
- Paper Table 2 MUST use **bits_per_byte (BPB)** as the primary text-modeling column and list PPL only in a supplementary column grouped by tokenizer family (MMfreeLM vs IntLLM can share a PPL column; the others cannot).
- `verify_intllm_tables.py --strict` checks that every Table 2 PPL cell is in the same `tokenizer_family` as the column header; mixed PPL ⇒ reject.

---

## 3. Per-baseline unported-features ledger

### 3.1 BitNet 2B4T (`microsoft/bitnet-b1.58-2B-4T`)

| Feature | Ported for comparison? | Detail |
|---|---|---|
| W1.58 ternary weight forward | ✓ | HF native `BitNetForCausalLM` unpacks U8-packed ternary weights on-the-fly in forward. |
| 8-bit absmax activation quantization | ✓ | HF native. |
| squared-ReLU FFN, subln norm, RoPE | ✓ | HF native. |
| Pretrain + SFT + DPO training pipeline | partial (no pretrain-only variant exists) | See §2.1. |
| **`bitnet.cpp` optimized CPU kernels** | ✗ | The repo README explicitly warns: *"Running via transformers will likely result in inference speeds and energy usage comparable to, or potentially worse than, standard full-precision models."* Our benchmark uses the HFLM/transformers path on GPU. |
| Published tok/s (29 ms CPU decode) and energy (0.028 J) | ✗ | Produced by `bitnet.cpp`, not replicable through our path. |
| 4 T-token training corpus | ✗ | Not released; not needed for eval. |

**Paper consequence:**
- Table 2 quality column (BPB, zero-shot accuracy): BitNet row **included**, with the §2.1 SFT-lift footnote.
- Table 4 throughput/energy column: BitNet row **stays "N/A — requires bitnet.cpp port, future work"**. Reporting our transformers-GPU tok/s for BitNet would be a §6.9 R3 violation because we'd be benchmarking an implementation artifact, not the method.

### 3.2 MMfreeLM 370M / 1.3B / 2.7B (`ridger/MMfreeLM-*`)

| Feature | Ported? | Detail |
|---|---|---|
| BitLinear ternary forward | ✓ | `mmfreelm.ops.bitnet.BitLinear`, vendored at `python/phase_d/_upstream/` (commit `f24cfe5`, see `UPSTREAM_PIN.md`). |
| fusedbitnet Triton kernel | ✓ | `mmfreelm.ops.fusedbitnet` from vendored snapshot. |
| MLGRU time-mixing (softmax-free attention surrogate) | ✓ | Native to the model class. |
| HGRN chunk-parallel kernel | ✓ | Vendored. |
| Pretrain-only checkpoints (no SFT/DPO) | ✓ | The HF repos are published as pretrain checkpoints per upstream README. |
| Upstream-published reference numbers | ✓ for 370M (see `FJQ_PHASE_D_P2_1_GATE.md` for the lm-eval 0.3→0.4 drift analysis: 5/6 tasks within ±1.53) | Only the 370M variant has Table 1 ground truth. 1.3B and 2.7B have no upstream Table 1 — our measured numbers stand as first-party publication with no cross-check column. |

**Paper consequence:** MMfreeLM is the cleanest apples-to-apples baseline in the set. Same training regime (pretrain-only), same tokenizer family (Mistral v3 32K), same HFLM eval plumbing. **Zero caveats on the 370M row; the 1.3B and 2.7B rows should be footnoted as "first-party eval, no upstream-published Table 1 to cross-check against."**

### 3.3 SmolLM2 135M / 360M (`HuggingFaceTB/SmolLM2-*`)

| Feature | Ported? | Detail |
|---|---|---|
| Llama-class arch via native transformers | ✓ | |
| Pretrain-only base checkpoints | ✓ | Explicitly use `SmolLM2-135M` and `SmolLM2-360M`, **not** `-Instruct` variants. |
| Custom 49,152-vocab tokenizer | tokenizer-local (not portable) | §2.2 caveat applies to PPL; zero-shot accuracy remains fair. |

**Paper consequence:** BPB-based comparison is fair and requires no footnote. PPL comparison is grouped.

### 3.4 Pythia-160m (`EleutherAI/pythia-160m`)

| Feature | Ported? | Detail |
|---|---|---|
| gpt_neox arch via native transformers | ✓ | |
| Pile pretrain, no SFT confound | ✓ | |
| gpt-neox 50,304-vocab tokenizer | tokenizer-local | §2.2 caveat applies to PPL only. |

**Paper consequence:** Clean baseline on accuracy + BPB. PPL grouped.

---

## 4. Harness parity (shared across all baselines)

Every baseline in this doc is evaluated via the **same** lm-evaluation-harness plumbing as IntLLM itself:

- `lm_eval.models.huggingface.HFLM(pretrained=<repo>, …)` is the single model-load path.
- `lm_eval.simple_evaluate(..., bootstrap_iters=100)` is the single scoring path.
- Harness version is pinned to `eval/HARNESS_SHA` (currently `ref: v0.4.11`; full 40-char SHA to be captured after the first real run, per §3.1 of the production plan).

**Consequence of this parity:** the only axes of variance between an IntLLM row and a baseline row are (a) model weights, (b) tokenizer, and (c) compute kernel — which is exactly what the paper sets out to measure. Silent disagreement from wrapper / formatter / sampling behaviors is eliminated by construction.

---

## 5. Paper Table 2 footnote requirements (machine-checked by Phase 4)

Minimum footnotes Table 2 must carry for §6.9 R3 compliance:

1. **BitNet row, zero-shot accuracy columns:** footnote "Post-SFT+DPO; IntLLM rows are pretrain-only. No pretrain-only BitNet 2B4T checkpoint is publicly available (see BASELINE_UNPORTED_FEATURES.md §2.1)."
2. **BitNet row in any tok/s or watts column:** cell must be literal `N/A` or a marker like `—` with footnote "requires `bitnet.cpp` port; not implemented (§3.1 of BASELINE_UNPORTED_FEATURES.md)".
3. **Any cross-tokenizer PPL column:** footnote "Absolute PPL not comparable across tokenizer families; see the BPB column (§2.2)."
4. **MMfreeLM 1.3B / 2.7B rows:** footnote "First-party evaluation; no upstream-published Table 1 exists for this parameter size."

The pre-commit hook added in Phase 4 (item 10 of FJQ_PHASE_D_3_4_B0_FINDINGS.md §7) matches these structural requirements against a marker file `paper/intllm/tables/table2_footnote_markers.json` so a paper edit that forgets a footnote fails commit.

---

## 6. Reproducibility

Each `bench_baselines_<tag>.json` artifact under `paper/intllm/results/` carries:

- `repo` (full HF id), `tag` (filesystem-safe derivative)
- `arch`, `params_approx`, `training_regime`, `tokenizer_family`, `sft_lift_caveat`, `kernel_note`
- `harness_version` (from `eval/HARNESS_SHA`)
- `status` ∈ {`scaffold`, `real`, `real-partial`}
- `tasks`, `num_fewshot_map`, `limit`, `batch_size`, `device`, `elapsed_seconds`
- `results` nested by task → metric → value

Rerun a single baseline:

```bash
cd ~/Documents/fajarquant/python/phase_d
PYTHONPATH=_upstream:. ../../.venv/bin/python scripts/bench_baselines.py \
    --repo ridger/MMfreeLM-370M
```

Rerun all 7 baselines (scaffold — fast, no GPU):

```bash
make bench-baselines
```

Rerun all 7 real (Phase 3.4 commit — ~13 GPU-hours on RTX 4090 16 GB):

```bash
make bench-baselines-real
```

---

## 7. Changelog

| Date | Change | Driver |
|---|---|---|
| 2026-04-24 | Initial commit — 7 baselines, 2 cross-cutting caveats, 4 footnote requirements | §3.4 B0 scaffold (FJQ_PHASE_D_3_4_B0_FINDINGS.md) |

Updates to this file require re-running `scripts/verify_intllm_tables.py --strict` before commit; any structural change (adding/removing a baseline, flipping a `sft_lift_caveat` flag, shifting a tokenizer_family label) must be matched by an update to `scripts/bench_baselines.py::BASELINE_REGISTRY` in the same commit.
