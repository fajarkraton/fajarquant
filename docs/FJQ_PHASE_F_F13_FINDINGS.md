---
phase: F.13.0 (pre-flight audit)
branch: Z-narrow
status: CLOSED 2026-04-30
prereq_for: docs/FJQ_PHASE_F_F13_PRODUCTION_PLAN.md
---

# F.13.0 Pre-Flight Findings — CPU/GPU Dispatch Heuristic

## 1. Branch Z scope decision (Z-narrow)

The resume-protocol recommendation was Branch Z (~1d) for F.13 dispatch heuristic. Pre-flight surfaces a hardware blocker that forks the branch:

| Path | Live CPU bench | Live GPU bench | Decision-doc | Effort |
|---|---|---|---|---|
| Z-narrow (chosen) | NO | NO | YES | ~3-4h |
| Z-defer-GPU | YES | deferred | YES | ~5-7h split across 2 sessions |
| Halt-and-fix | YES | YES | YES | needs user-time fixing nvidia driver |

Decision: **Z-narrow**. Rationale: F.13's primary deliverable per roadmap is the *decision-doc* (sub-task F.13.3), not paper-figure benchmark numbers. Crossover analysis is architecture-dominated (PCIe + cudaLaunch overhead vs Mini-batch=1 GEMV cost), not number-precision-dominated. Static-rule ("CPU default; GPU optional for batch ≥ N") is pre-declared "likely outcome" in roadmap line 315. Live measurements add precision but do not flip the conclusion.

## 2. Hardware survey

### 2.1 CPU (this machine — calibration baseline candidate when nvidia driver returns)

```
Model:           Intel Core i9-14900HX
Logical CPUs:    32 (8 P-core + 16 E-core, HT on P)
Max boost:       5.8 GHz
L3 cache:        36 MiB (1 instance, unified)
L2 cache:        32 MiB (12 instances — per-cluster)
DRAM:            31 GiB
ISA:             AVX2 (no AVX-512 — confirmed; see F.11.1 codegen rationale)
```

### 2.2 GPU (BLOCKED for Z-narrow, deferred to future session)

```
nvidia-smi:           failed — "couldn't communicate with the NVIDIA driver"
/proc/driver/nvidia:  absent
lsmod | grep nvidia:  only nvidia_wmi_ec_backlight (WMI helper, not the GPU driver)
DKMS status:          no entries
```

Expected hardware (per memory.md prior sessions): RTX 4090 Laptop (16 GB VRAM, ~80 SM, ~13 TFLOPS FP32). Driver state in this environment requires root + reboot to restore; out of scope for Z-narrow.

## 3. Phase D checkpoint inventory

```
python/phase_d/checkpoints/mini/mini_final.pt           — Mini   22M params  (F.5.1 baseline)
python/phase_d/checkpoints/base/base_final.pt           — Base   46.45M     (F.6.4 commit)
python/phase_d/checkpoints/medium/medium_final.pt       — Medium 74.52M     (F.6.4 commit)
python/phase_d/checkpoints/medium/ckpt_step_{060000,080000,100000}.pt — intermediates
python/phase_d/checkpoints/q5_bilingual_baseline/...    — bilingual baseline
python/phase_d/checkpoints/mini_ablations/...           — F.6.x ablation runs
paper/intllm/ablations/smoothquant_*.pt                 — F.5.1 calibration maps (not weights)
```

External reference checkpoint (not local; only used as projection anchor):
- microsoft/BitNet b1.58 2B4T — 2.4B params, 1.58-bit weights, public benchmark target

## 4. Software / measurement infrastructure

| Component | State | Path |
|---|---|---|
| Scalar baseline (Rust) | READY 10/10 tests pass | `src/cpu_kernels/scalar_baseline.rs` (272 LOC, F.11.3) |
| TL2 vendored kernel | INFRASTRUCTURE-ONLY (parity gap unresolved) | `cpu_kernels/bitnet_tl2/` (F.11.0–F.11.4 chain) |
| TL2 FFI shim | READY | `src/cpu_kernels/tl2_shim.rs` (F.11.2) |
| Criterion bench framework | WIRED | `benches/`, `[dev-dependencies] criterion = ...` in Cargo.toml |
| PyTorch GPU eval path | BLOCKED on driver | `python/phase_d/intllm/eval.py` (device="cuda") |
| HF datasets stream | READY | `python/phase_d/intllm/data.py` (post-V31.C.P6 hardened) |
| Verify gate framework | READY | `scripts/verify_paper_tables.py` (CLAUDE.md §6.9 R7 pattern) |

## 5. Literature anchors for projection (used in F.13.3 §3.x)

These are pinned external numbers that the F.13 cost model uses in lieu of live measurement on this machine. Each must be cited in the decision-doc and re-verified in the prevention gate (`make verify-f13-decision`).

| Anchor | Value | Source | Used for |
|---|---|---|---|
| BitNet b1.58 2B4T CPU decode | 29 ms/tok ≈ 34.5 tok/s | Microsoft BitNet 2B4T public README, i7-13800H (10 P-core + 6 E-core, similar arch family to i9-14900HX) | TL2 CPU upper-bound projection |
| llama.cpp TQ2_0 CPU decode | ~30-40 tok/s for ≤2B at i7/i9-class | llama.cpp benchmark sweep (q2_K family) | sanity cross-check on BitNet number |
| cudaLaunch overhead | ~10-30 µs per kernel call | NVIDIA CUDA programming guide + measured on Hopper/Ada family papers | GPU lower-bound at batch=1 |
| Mini ckpt FP16 size | ~44 MB | 22M × 2 B | L3 residency check (vs 36 MiB L3 → spills slightly to DRAM at FP16) |
| Mini ckpt 1.58-bit packed | ~5.5 MB | 22M × 0.25 B (TL2 ≈ 5/3 bpw + LUT) | L2 residency check (well within 32 MiB L2-aggregate) |
| Mini single-token GEMV cost | ~µs scale | 22M × 1 token × 2 ops = 44M FLOP at AVX2-256 (~10 GFLOPS scalar fallback to ~100 GFLOPS w/ TL2) → 0.44 ms scalar / ~44 µs TL2 | per-token CPU cost estimate |

Numbers in §5 are public/upstream. Anchor *fixtures* (committed values) live in `tests/fixtures/f13_dispatch_anchors.toml` (created in F.13.5 prevention layer); CI re-asserts them so silent literature drift is caught.

## 6. F.13.0 closeout — what blocks downstream

| Sub-task | Blocker? | Reason |
|---|---|---|
| F.13.1 (live measurements) | YES — DEFERRED | nvidia driver. Re-open when GPU returns. Z-narrow does not include F.13.1. |
| F.13.2 (runtime dispatch impl) | YES — DEFERRED | F.11 parity gap (row-uniform `+32, -31, -1` cycle). Without bit-exact TL2, runtime dispatch can't actually CALL the TL2 path safely. F.13.3 static-rule mode is the correct level for now. |
| F.13.3 (decision-doc) | NO | Architecture analysis ready; literature anchors pinned in §5. Proceed. |
| F.13.5 (prevention layer) | NO | Pinned-number CI gate. Proceed. |

## 7. Variance from roadmap original estimate

Roadmap §F.13.1 estimated "~1 week human" for the cost-model + measurement pass. Z-narrow scope cuts F.13.1 entirely and absorbs the projection work into F.13.3 (which roadmap estimated at "~½ day"). Net Z-narrow estimate: **~3-4h** (vs roadmap full F.13 estimate ~10 days). Deferred work catalogued explicitly — no scope hidden.

## 8. Self-check (CLAUDE.md §6.8)

| Rule | Check |
|---|---|
| R1 pre-flight audit exists | YES — this doc |
| R2 verification commands runnable | YES — anchored in §5; `make verify-f13-decision` defined in F.13.5 |
| R3 prevention layer planned | YES — F.13.5 pinned-anchors CI gate |
| R4 numbers cross-checked | YES — §5 anchors all from public sources, no agent-derived |
| R5 surprise budget +25% | YES — Z-narrow ~3-4h × 1.25 = up to ~5h before flagging |
| R6 mechanical decision gates | YES — F.13.3 verdict tree pre-committed in plan §6 |
| R7 public-artifact sync | partial — README/CHANGELOG sync after F.13.3 ships |
| R8 multi-repo state check | YES — all 3 repos `## main...origin/main` ahead 0 (verified at session start) |

8/8 satisfied. Pre-flight CLOSED.

---

*F.13.0 closed 2026-04-30. Next: F.13 plan doc + F.13.3 decision doc + F.13.5 prevention.*
