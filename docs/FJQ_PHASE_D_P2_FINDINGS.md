# FajarQuant Phase D — C.P2 Pre-Flight Findings

> **Date:** 2026-04-21 | **Rule:** §6.8 Rule 1 (pre-flight audit mandatory) | **Depends on:** `FJQ_PHASE_D_ARCH.md` + `FJQ_PHASE_D_OPS.md` + `FJQ_PHASE_D_PAPER_OUTLINE.md` + `FJQ_PHASE_D_CONFIG.md`

## 0. Why this pre-flight exists

Per §6.8 Rule 1: every Phase starts with a hands-on audit before
downstream subphases can begin. C.P2 is a ≥3-4d PyTorch implementation
commitment per subpart; entering it with a wrong assumption about
upstream repo, checkpoint availability, or local GPU would waste days.

This doc runs pre-flight and commits the findings before C.P2.1 starts.

## 1. Upstream repo audit — `ridgerchu/matmulfreellm`

- **License:** Apache-2.0 ✓ (consistent with C.P0 sweep claim)
- **Stars:** 3.1k (up from ~3.05k in C.P0 sweep)
- **Dependencies:** PyTorch ≥ 2.0, Triton ≥ 2.2, einops — no exact pins
- **HF checkpoints confirmed pullable:**
  - `ridger/MMfreeLM-370M`
  - `ridger/MMfreeLM-1.3B`
  - `ridger/MMfreeLM-2.7B`
- **Model class:** `HGRNBitConfig` + `AutoModel.from_config()` +
  standard `AutoModelForCausalLM.from_pretrained()` pattern (HF
  Transformers integration)

**Surprise #1 (+25% surprise budget tag per §6.8 R5):** **Upstream is
inference-only — no training code in the repo.** The C.P0 survey
report implied training code was released; the actual README confirms
only a minimal inference quickstart. Training code exists instead in
the sibling project `fla-org/flash-linear-attention` (FLA), which
hosts QAT-compatible implementations of MLGRU / HGRN / BitLinear.

**Implication for C.P2 scope:**
- Fork `ridgerchu/matmulfreellm` for the model class (forward-pass arch)
- Pull FLA for training-compatible layers + optionally its trainer
- Build Phase D's QAT-specific training loop from scratch (paper §3.3 contribution)
- Verification target at C.P2.1 shifts from "repro training Table 1" to
  "repro **inference** on pretrained 370M/1.3B/2.7B checkpoints, then
  build QAT from there"

## 2. Local environment audit

| Component | Status | Notes |
|---|---|---|
| Python | ✅ 3.12.3 | System python; `fajarquant/.venv` has same |
| Torch | ✅ **2.6.0+cu124, CUDA=True** | Inside `fajarquant/.venv/` |
| GPU | ✅ **NVIDIA RTX 4090 Laptop 16 GB**, driver 590.48.01 | Matches V24 inventory |
| RAM | ⚠️ 31 GiB total, 20 GiB free | Enough for 2.7B inference; tight for heavy training alongside OS |
| Disk | ⚠️ **136 GB free / 937 GB (85% full)** | Flag: 17d Stretch run may accumulate logs; clean checkpoint cadence needed |
| HF cache | ✅ Initialized at `~/.cache/huggingface/` | datasets + hub + auth token present |
| UV / conda | ❌ Not installed | Using system pip inside `.venv` |

**Surprise #2:** Disk is already 85% full. Per §6.8 R3, this spawns a
prevention layer: add a `python scripts/cleanup_hf_cache.py` utility
(C.P2.4 deliverable) and document retention policy in the Stretch
training gate (`FJQ_PHASE_D_STRETCH_GATE.md`). If `df` shows <30 GB
free during training, runs pause and surface to stderr.

## 3. Repository layout decision

**Decision: Option C — Phase D Python code lives at
`fajarquant/python/phase_d/` as a subdir.**

Alternatives considered:
- Option A (`fajarquant/src/phase_d/`): mixes Python training with
  Rust crate source tree — confusing for crate users
- Option B (new repo `fajarkraton/intllm`): scatters Phase D across
  repos, complicates paper reproducibility
- **Option C (chosen):** clean separation by language subdir, Phase D
  stays co-located with the algorithm + paper assets, single
  `reproduce.sh` can orchestrate both the Rust KV-cache work (Phase
  A-C) and Python LM work (Phase D)

Planned layout:
```
fajarquant/
├── Cargo.toml             # existing Rust crate (Phase A-C)
├── src/                   # existing Rust src (KV cache quant)
├── paper/                 # existing — shared with Phase D
├── reproduce.sh           # existing — extend to include Phase D modes
└── python/
    └── phase_d/
        ├── pyproject.toml         # new — PEP 621 Python package spec
        ├── intllm/                # package name = intllm
        │   ├── __init__.py
        │   ├── model.py           # MLGRU + GLU + BitLinear (from upstream)
        │   ├── quant.py           # FajarQuant QAT recipe (§3.3 paper)
        │   ├── train.py           # training loop + optimizer
        │   ├── data.py            # SlimPajama streaming loader
        │   ├── eval.py            # Wikitext-103, LAMBADA, MMLU harness
        │   └── export.py          # .fjm v9 exporter for FajarOS Nova
        ├── tests/
        │   ├── test_forward.py    # bit-exact vs upstream on 370M
        │   ├── test_qat.py        # QAT gradient checks
        │   └── test_export.py     # .fjm v9 round-trip
        ├── configs/               # hparam YAML per Mini/Base/Medium/Stretch
        └── scripts/
            ├── repro_upstream.py  # C.P2.1 baseline repro
            ├── train_mini.py      # wraps intllm.train for Mini config
            └── cleanup_hf_cache.py  # disk hygiene per §6.8 R3
```

## 4. C.P2 subphase gating

Per V31 Master Plan §4.3, updated with C.P2.0 findings:

| Sub | Task | Effort | Gate |
|---|---|---|---|
| **C.P2.0** | ⭐ This findings doc (pre-flight audit) | ~0.5h actual | ✅ Committed |
| **C.P2.1** | Fork upstream + verify inference reproduces 370M HF checkpoint (PPL ± 0.1 on Wikitext-103) | 1.5-2d (down from 3-4d: inference-only simplifies) | Table 1 row repro within ± 1 avg |
| **C.P2.2** | Port int-RMSNorm + σ LUT + SiLU approximation as PyTorch modules with gradient | 2d | Forward numerically matches FP reference ±1e-3 |
| **C.P2.3** | Wire up full training forward pass, loss decreasing on Mini synthetic corpus | 1-2d | `loss < 4.0` after 1h of Mini config |
| **C.P2.4** | Build QAT harness: FP shadow + fake-quantize forward + STE backward + periodic γ_x re-calibration | 2d | Mini QAT converges within 10% of FP shadow |

Total C.P2 effort: **6.5-8d** (down from plan-book 9-10d after C.P2.0
findings simplify the repro target).

## 5. Novel contributions that ship through C.P2

Per `FJQ_PHASE_D_PAPER_OUTLINE.md` §3.3, the three QAT recipe
contributions all live in `intllm/quant.py`:

1. **Per-coord adaptive bit allocation** (`BitAllocator` class) —
   tracks per-channel ‖x_·,j‖_∞ over calibration corpus; top-5%
   channels get 10-bit activations, rest stay 8-bit. Surfaces as a
   `forward_quant_bits: int[d]` buffer on each BitLinear.
2. **Ternary-aware channel reordering** (`ChannelPermuter` class) —
   permutes output channels so that high-β rows cluster together;
   enables block-sparse `W̃_ij = 0` compute-skip during inference.
3. **Periodic γ_x re-calibration** (`AbsmaxRecalibrator` hook) —
   re-computes γ_x every K=1000 training steps during QAT phase, not
   held fixed at initialization. After QAT freeze, γ_x is baked
   into `.fjm` v9.

All three are opt-in via `train.py` config flags; ablation runs in
C.P4 toggle them individually for §5.1 Table 4 of the paper.

## 6. Risk register update for C.P2

| Risk | Status | Mitigation |
|---|---|---|
| Upstream repo goes unmaintained during Phase D | Monitored | Fork `ridgerchu/matmulfreellm` into `fajarkraton/matmulfreellm-fork` as a frozen-dep snapshot (C.P2.1 step 1) |
| `ridger/MMfreeLM-*` HF checkpoints get pulled | Unlikely | Download all 3 ckpts to `~/.cache/huggingface/hub` during C.P2.1, then archive to `~/Documents/fajarquant/data/checkpoints/` as backup |
| Triton version incompat with torch 2.6 | Unknown | Pin `triton==3.1.0` or whatever ships with torch 2.6.0+cu124 in C.P2.1 requirements |
| Disk exhaustion during Stretch training | **New (§2 surprise)** | `scripts/cleanup_hf_cache.py` + monitoring during 17d run; rotate checkpoints every 500 steps; stop if <30 GB free |
| RTX 4090 Laptop vs desktop thermals during 17d run | Unknown | Medium config (~11h) as thermal stress test before committing Stretch |

## 7. §6.8 Plan Hygiene self-check for C.P2.0

- [x] R1 — Pre-flight audit (this doc) completed before C.P2.1 starts
- [x] R2 — C.P2 verification columns are runnable commands (§4 gates)
- [x] R3 — Prevention layer: `scripts/cleanup_hf_cache.py` addresses
  disk hygiene; archive step in §6 addresses ckpt availability
- [x] R4 — Upstream findings cross-checked by direct WebFetch, not
  relying only on C.P0 agent report (R1 requirement)
- [x] R5 — Variance tagged in commit: 0.5h actual vs 0.25d est (+100%:
  disk + upstream-inference-only surprises consumed the extra time)
- [x] R6 — Decision gate: repo layout decision (§3) committed as file
- [x] R7 — No public artifact implications (internal design doc)
- [x] R8 — Multi-repo state confirmed pre-commit:
  fajarquant ahead 0 / fajar-lang ahead 0 / fajaros-x86 ahead 0

## 8. Unblocks

C.P2.1 ("Fork upstream + verify inference reproduces") can now start.
First step: `git clone` upstream into `python/phase_d/_upstream/`
(vendored snapshot), pull `ridger/MMfreeLM-370M` checkpoint via HF
hub, run `scripts/repro_upstream.py` on Wikitext-103 → capture PPL
and compare to Zhu et al. Table 1.

Expected wall-clock for C.P2.1 smoke test: ~10 minutes (eval only, no
training).
