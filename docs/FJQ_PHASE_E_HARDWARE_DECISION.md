# Phase E E0.0.4 — Hardware Decision

> **Status:** PASS — laptop-only GPU strategy adopted; Stretch scale REPLANNED.
> **Decision date:** 2026-04-25
> **Plan reference:** `FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md` v1.3 §3 E0.0.4 + §2 Q4
> **Decision authority:** Fajar (founder)

---

## 1. Decision

**LAPTOP-ONLY** — all Phase E GPU work runs on Lenovo Legion Pro RTX 4090 Laptop. No cloud burst budgeted.

**Confirmed by founder 2026-04-25:** *"Kita hanya akan pakai GPU yang ada di laptop ini."*

## 2. Hardware specifications

| Component | Spec | Phase E implication |
|---|---|---|
| GPU | NVIDIA RTX 4090 Laptop GPU | 16 GB VRAM (vs desktop 4090 24 GB) |
| CPU | Intel i9-14900HX (24c / 32t) | strong CPU for data pipeline |
| RAM | 31 GB DDR5 | sufficient for 100-200 GB streamed corpus |
| Storage | local SSD (free space TBD) | DVC pull staging required |
| Network | dual-wifi (4GHz + 5GHz observed) | E1.5 generator + dataset downloads |
| Power | mains + battery | sustained training requires mains |
| Thermal | laptop chassis | sustained training = throttling concern |

## 3. Cascading implications

### 3.1 Stretch scale (~3 B params, ~30 B tokens) — REPLANNED

**Plan v1.3 §3 E3.4 originally:** "Cloud-burst launch... ~$300-600 for Stretch ID+EN"

**Laptop-only revised projection:**
- 3B bilingual model + bf16 + grad checkpointing: ~12-14 GB VRAM (tight on 16 GB)
- Throughput: ~10K-20K tokens/sec on RTX 4090 Laptop sustained (vs ~50K on H100)
- 30 B tokens / 10K tok/s = **~833 hours ≈ 35 days continuous**
- 30 B tokens / 20K tok/s = **~417 hours ≈ 17 days continuous**

**Realistic outcomes:**
- (A) **Defer Stretch entirely** to Phase F — ship Medium (~140M) as Phase E paper headline
- (B) **Run Stretch over 4-6 weeks elapsed time** — laptop dedicated to training, with intermittent breaks (thermal cooldown, user activity, etc.). Pro: fits laptop-only constraint. Con: laptop unavailable for other work for 4-6 weeks.
- (C) **Run reduced Stretch** (~1.5B × 15 B tokens) — ~7-10 days laptop, more realistic. Cuts paper Table 2 Stretch row claim from 3B to 1.5B but proves scaling chain.

**Recommended: option (C) reduced Stretch** — preserves Phase E narrative integrity (Mini → Base → Medium → Stretch monotonic widening proven) while staying within laptop-feasible elapsed time.

### 3.2 Calendar extension

| Phase | Plan v1.3 calendar | Laptop-only revised |
|---|---|---|
| E0 Pre-flight | Week 1 | Week 1 (no change, all CPU-feasible) |
| E1 Bilingual data | Weeks 2–4 | Weeks 2–4 (no change, ~5h GPU OK on laptop) |
| E2 Algorithm catch-up | Weeks 5–7 | **Weeks 5–9** (+2 weeks; ablations sequential, no parallel cloud) |
| E3 Bilingual pretrain Mini/Base/Medium | Weeks 8–18 | **Weeks 10–22** (+4 weeks; sequential laptop runs) |
| E3.4 Stretch | (cloud burst, ~17 days) | **Weeks 23–30 reduced Stretch (option C) OR DEFERRED** |
| E4 Vertical fine-tune | Weeks 19–22 | **Weeks 23–28** |
| E5 Kernel + paper + IP | Weeks 23–28 | **Weeks 29–34** |
| **Total** | **~28 weeks (6.5 months)** | **~34 weeks (8 months) with reduced Stretch** OR **~28 weeks Medium-only** |

### 3.3 GPU-hour budget revisited

| Phase | Plan v1.3 GPU-h | Laptop-only revised |
|---|---|---|
| E1 (data tokenization) | ~5h | ~5h (laptop) |
| E2 (mini ablations) | 30–60h | 30–60h (laptop, ~3-5 days elapsed) |
| E3 (Mini/Base/Medium) | 70–150h | 70–150h (laptop, ~10-21 days elapsed) |
| E3.4 (Stretch) | 80–150h cloud (or 8-17 days) | reduced Stretch ~50-100h laptop (~5-10 days) OR defer |
| E4 (tax FT) | 30–60h | 30–60h (laptop, ~3-5 days) |
| E5 (serving benches) | ~5h | ~5h (laptop) |
| **Total** | **~220–430h GPU** | **~190–380h GPU on laptop** (Stretch reduced) |

Laptop GPU-hours similar to plan; **calendar elapsed time stretches 1.4× due to no parallelism**.

### 3.4 Cost revision

| Plan v1.3 | Laptop-only |
|---|---|
| ~$420–820 cloud | **$0 cloud cost** |
| Power: ~$20-40/month sustained | (already amortized, no marginal) |
| **TOTAL** | **~$0 marginal cost** |

Trade-off: $420–820 saved, but +6-8 weeks calendar time + laptop unavailable for non-Phase-E work during training windows.

## 4. Mitigations for laptop-only constraints

### 4.1 Thermal management

- Run training during cooler hours (late evening to early morning, ~22:00–06:00 WIB)
- Use cooling pad / external fans during sustained training
- Monitor `nvidia-smi --query-gpu=temperature.gpu` — stop if sustained >85°C
- Schedule breaks every 6h for 30-min cooldown

### 4.2 Power supply

- Always plugged in during training (battery-only training is an interruption-safety risk per §6.11 — already covered by Track B)
- UPS recommended for power-blip resilience (~$50 single 600VA unit)
- Track B watchdog already SIGTERMs if step-idle (covers inadvertent suspend)

### 4.3 Disk space

- DVC pull staging needs ~135-230 GB Phase E data (per E0.0.8) + checkpoints (~20-50 GB Mini/Base/Medium)
- Recommend: 500 GB free on training drive
- Older Phase D checkpoints → archive to external SSD or cloud cold storage

### 4.4 Laptop availability

- During E3 training (~10-21 days), laptop heavily occupied
- Plan: light work (docs, planning) acceptable in foreground, heavy work blocked
- Batched training schedule: nights + weekends for non-disruption

### 4.5 Single-point-of-failure

- Laptop hardware failure = full Phase E lock
- Mitigation 1: regular checkpoint sync to S3 (per E0.0.8 DVC config)
- Mitigation 2: borrow alternate machine in emergency (1-2 GPU acquaintances)
- Mitigation 3: cloud burst fallback — defer cloud decision but DON'T fully foreclose; if laptop hardware fails, can fall back to ~$300-500 cloud rescue

## 5. Update to plan v1.3 (next: v1.4 patch)

When v1.4 plan patch is written:

- §2 Q4 closed: laptop-only confirmed
- §3 E0.0.4 status: PASS
- §3 E3.4 Stretch: option (C) reduced Stretch ~1.5B × 15B tokens, OR option (A) deferred
- §4 calendar revised: 28 weeks → 34 weeks (with reduced Stretch) OR 28 weeks (Medium-only)
- §6 risk register: "GPU cost overrun" downgraded; "laptop hardware failure" upgraded with mitigation chain

Plan v1.3 §6 Stretch GPU cost-overrun risk:
> "Low–Medium likelihood, Medium impact. E0.0.4 sets explicit cap"

→ **v1.4: Resolved via laptop-only — risk class shifts to laptop-failure (Low likelihood, High impact)**

## 6. Decision: PASS

- [x] Hardware path chosen (laptop-only)
- [x] Cascading implications mapped (Stretch replan, calendar extension, GPU-hour budget, cost)
- [x] Stretch handling 3 options documented (defer / extended / reduced); recommended option (C)
- [x] Calendar revision quantified (28 → 34 weeks with reduced Stretch)
- [x] Cost revision quantified ($420-820 → $0 marginal)
- [x] Mitigations for laptop constraints written (thermal, power, disk, availability, SPOF)
- [x] Plan v1.4 patch scope identified

**Phase E E0.0.4 gate: PASS** (mechanical decision file landed per §6.8 R6).

**9 of 10 E0 sub-tasks now closed** (only E0.0.7 data licensing per source remains).

---

*Decision committed 2026-04-25. Author: Claude Opus 4.7 + Fajar.*
*Authority: Fajar (founder, hardware-budget owner).*
*Trigger v1.4 plan patch when E0.0.7 lands — bundle both updates.*
*Companion: Track B 6-layer interruption-safety (per CLAUDE §6.11) covers laptop suspend/blip recovery; this doc adds hardware-specific mitigations on top.*
