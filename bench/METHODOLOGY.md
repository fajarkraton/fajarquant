# FajarQuant Benchmark Methodology

> **V26 Phase C2.0.1 deliverable.** This document is the canonical
> methodology for every `cargo bench` run that produces a number cited
> in the FajarQuant paper or supplementary materials. Any benchmark
> result captured without the protocol below is **not paper-grade**
> and must not be cited.
>
> **Why this document exists:** V26 Phase A1.3 found 14 unit tests in
> the fajar-lang compiler that asserted `elapsed < N_ms` and flaked
> ~20% of the time under parallel test load. The same statistical
> noise will corrupt paper benchmarks unless we lock the methodology
> *before* collecting numbers. CLAUDE.md §6.7 already forbids
> wall-clock assertions in unit tests; this document establishes the
> equivalent rules for benchmarks.

---

## 1. Tooling

### Required: `criterion` (Rust)

All micro-benchmarks must use the [`criterion`](https://docs.rs/criterion)
crate from `benches/*.rs`. Criterion handles:

- Statistical noise via repeated sampling
- Warmup separation
- Outlier detection (Tukey fences)
- Median + confidence intervals
- Regression detection across runs (`target/criterion/`)

### Forbidden: `std::time::Instant::now()` for assertions

Per CLAUDE.md §6.7 (Test Hygiene Rules), `assert!(elapsed < N_ms)` is
**forbidden in unit tests** because scheduler jitter parks threads for
100s of ms under parallel load. The same rule applies to benchmark
"sanity checks" — if you need to verify a benchmark didn't regress,
use `criterion`'s built-in regression detection, not a hand-rolled
timer assertion.

### Forbidden: Python `time.perf_counter()` for paper numbers

Python timing is acceptable for *exploration* (e.g., `scripts/run_comparison.py`
prints `total_seconds`) but never for *paper* numbers. The paper-grade
performance section (V26 §C2.5) cites only `criterion` outputs.

---

## 2. Sample Protocol

| Parameter | Required value | Why |
|---|---|---|
| Sample count | **100 samples** per measurement | Criterion's default; balances precision against runtime. Smaller sample counts (e.g., 30) inflate CI width 1.8×. |
| **Warmup** runs | **10 warmup** iterations before sampling starts | Drains CPU caches from prior tests, lets HWP find a stable frequency, lets the JIT (if any) emit hot code paths. |
| Statistic reported | **Median + 95% CI** (NOT mean) | Median is robust to outliers from preemption / IRQ storms. 95% CI tells reviewers how stable the number is. Reporting mean alone hides instability. |
| Outlier handling | Criterion's default Tukey fences | Marks samples > Q3 + 3·IQR or < Q1 − 3·IQR as severe outliers. |
| Confidence level | 95% (default) | Standard for papers. |

The minimum reportable timing is the **noise floor** (see §6 below).
Anything smaller than that is statistical noise and must be reported
as such.

---

## 3. CPU Configuration

The Lenovo Legion Pro test machine (i9-14900HX, 24c/32t — see
`bench/hardware_snapshot.txt`) is unstable for benchmarks at default
settings. Before any `cargo bench` run:

### 3.1 Governor pinning

```bash
sudo cpupower frequency-set --governor performance
# or:
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

Default governor on Ubuntu 24.04 is `powersave` (verified in
`hardware_snapshot.txt`). Powersave drops CPU frequency under low
load, which adds 100-300% timing variance. **Never benchmark under
powersave.**

### 3.2 HWP / `intel_pstate`

Even with `governor=performance`, Hardware P-states (HWP) can throttle
the CPU based on package thermal and power limits. For paper-grade
stability:

```bash
# In /etc/default/grub: add intel_pstate=passive
GRUB_CMDLINE_LINUX="intel_pstate=passive"
sudo update-grub
sudo reboot
```

`intel_pstate=passive` puts the kernel scheduler in control instead
of HWP, which gives the userspace governor full authority. Without
this, HWP can override `performance` and re-introduce variance.

### 3.3 Turbo boost: disable for paper runs

Turbo boost is great for exploration but bad for reproducibility — it
depends on instantaneous package temperature, which fluctuates. For
paper-grade runs:

```bash
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
```

This pins frequency to the rated max (5.6 GHz on i9-14900HX, NOT the
Turbo 5.8 GHz). The number you report is the steady-state number,
not the burst number.

### 3.4 Single-threaded evaluation

```bash
cargo bench -- --test-threads=1
```

Or set in `.cargo/config.toml`:
```toml
[env]
CARGO_BENCH_THREADS = "1"
```

Parallel benchmark threads share L3 cache and contend for memory
bandwidth, which adds noise even with criterion's outlier filtering.
**Single-thread is mandatory** for paper-grade runs.

### 3.5 CPU pinning to even-numbered cores

The i9-14900HX has Hyperthreading (24 cores × 2 threads = 32 logical).
HT siblings share L1 + L2 cache, so two benchmark runs on threads
0+1 will contend. Pin to even-numbered logical CPUs only:

```bash
taskset -c 0,2,4,6,8,10,12,14,16,18,20,22 cargo bench
```

The full `bench/setup_perf.sh` script (V26 §C2.0.4 deliverable,
pending) wraps all of §3.1-§3.5 into one command.

---

## 4. Memory Configuration

### 4.1 Drop caches before each run

```bash
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches
```

This forces all cached file data to be re-read from disk. Important
for memory-bound benchmarks (matmul, KV cache loading) where the
first run is faster simply because pages are warm.

### 4.2 No swap pressure

```bash
swapoff -a   # before benchmark
# ... cargo bench ...
swapon -a    # after benchmark
```

If the machine starts paging during a benchmark, timings are useless.
On the 31 GiB Legion Pro with 8 GiB swap, ensure no other workload is
above 20 GiB resident before starting. The `hardware_snapshot.txt`
captured during the C2.0.2 run noted "swap used 4.6 GiB" because
Mistral 7B was downloading — that snapshot is for documentation,
not benchmark provenance.

### 4.3 NUMA: not applicable

i9-14900HX is single-socket / single-NUMA node. No NUMA pinning
needed. If FajarQuant is ever benchmarked on a multi-socket EPYC or
Xeon, this section must be revised.

---

## 5. GPU Configuration (when CUDA paths are benchmarked)

The CUDA matmul kernels in `fajarquant/src/` (and the kernel-LLM
work in `fajaros-x86/`) are benchmarked separately:

### 5.1 GPU clock pinning

```bash
sudo nvidia-smi -lgc 2520,2520    # lock graphics clock to base
sudo nvidia-smi -lmc 9001,9001    # lock memory clock to max
```

After the bench run:
```bash
sudo nvidia-smi -rgc
sudo nvidia-smi -rmc
```

### 5.2 Persistence mode

```bash
sudo nvidia-smi -pm 1
```

Without persistence mode, the driver tears down between processes,
which adds ~500ms variance to short benchmark runs.

### 5.3 Use CUDA events, not host wall-clock

Time GPU kernels with `cudaEventRecord` / `cudaEventElapsedTime`,
NOT `Instant::now()` on the host. Host timing includes kernel launch
latency + sync overhead, which is unrelated to kernel runtime.

---

## 6. Noise Floor

Before publishing any benchmark number, run a **no-op baseline** to
establish what counts as "statistical noise" on this machine:

```rust
// benches/noise_floor.rs
use criterion::{black_box, Criterion, criterion_group, criterion_main};

fn noop_bench(c: &mut Criterion) {
    c.bench_function("noise_floor", |b| {
        b.iter(|| black_box(0u64));
    });
}

criterion_group!(benches, noop_bench);
criterion_main!(benches);
```

Run this 5× consecutively (no other workload). Take the **maximum
95% CI half-width** across the 5 runs. That value is the noise floor.

**Reporting rule:** any benchmark improvement smaller than 2× the noise
floor must be reported as "below noise floor" instead of as a number.

The C2.0.3 deliverable (`bench/results/noise_floor.json`, pending)
captures this baseline.

---

## 7. What Invalidates a Benchmark Run

Any of the following invalidates the run and requires a re-do:

1. **Background load**: any process > 1% CPU during the bench window
   (check with `top -d 1 -n 5` before starting)
2. **Network activity**: any process downloading or uploading > 1 MB/s
   (Mistral 7B download was running during C2.0.2 capture — that's
   fine for the snapshot, NOT fine for actual benchmark runs)
3. **Thermal throttling**: `nvidia-smi` shows temperature > 85°C, or
   `sensors` shows package temp > 95°C
4. **CPU frequency drift > 5%** during the run (criterion may flag this
   as "Performance has improved/regressed" — investigate before citing)
5. **Governor not `performance`**: re-check
   `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` after
   the run completed
6. **Hardware drift from `bench/hardware_snapshot.txt`**: `lscpu`,
   `nvidia-smi`, `uname -a` outputs differ from the captured snapshot
   (e.g., kernel update, NVIDIA driver update). Re-capture and re-run.
7. **Anyone else logged in via SSH or running on the same machine**:
   benchmarks are single-tenant.

When in doubt, re-run. Re-running is cheap; citing a noisy number in
a paper is not.

---

## 8. Per-Benchmark Templates

### 8.1 Latency benchmark (`benches/quant_latency.rs`)

Measures: time to quantize one KV cache layer (single function call).

```rust
use criterion::{Criterion, criterion_group, criterion_main};
use fajarquant::FajarQuant;

fn quantize_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize_layer");
    group.sample_size(100);                    // §2 requirement
    group.warm_up_time(std::time::Duration::from_secs(3));   // §2 warmup
    group.measurement_time(std::time::Duration::from_secs(10));
    group.confidence_level(0.95);              // §2 95% CI

    let kv = load_test_layer();                // 1 layer × 8 KV heads × seq_len × 128
    for &bits in &[2, 3, 4] {
        group.bench_function(format!("fajarquant_{bits}bit"), |b| {
            b.iter(|| FajarQuant::quantize_kv(&kv, bits));
        });
    }
    group.finish();
}

criterion_group!(benches, quantize_layer);
criterion_main!(benches);
```

### 8.2 Throughput benchmark (`benches/throughput.rs`)

Measures: tokens/sec for KV-quantized inference vs FP16 baseline.

Use `cargo bench --bench throughput -- --model llama2_7b` form. The
`--model` arg picks the input data set from `data/kv_cache/<model>/`.

### 8.3 Memory benchmark (`benches/memory_profile.rs`)

Measures: peak RSS for each algorithm at 16K context.

Use `valgrind --tool=massif` for ground truth, then a `criterion`
group that records `peak_rss_kb` via `getrusage(RUSAGE_SELF)`.

---

## 9. Wiring This Methodology Into the Paper

When reporting numbers in `paper/fajarquant.tex`, every benchmark
table must include a footnote citing this file:

> *All wall-clock numbers measured per `bench/METHODOLOGY.md`
> on the hardware described in `bench/hardware_snapshot.txt`.*

The supplementary materials (V26 §C3.4 deliverable) must include a
copy of both files alongside the raw `target/criterion/` directory
(or at least the JSON exports) so reviewers can verify reproducibility.

---

## 10. Cross-References

| File | Role |
|---|---|
| `bench/hardware_snapshot.txt` | C2.0.2 — captured hardware state |
| `bench/METHODOLOGY.md` | C2.0.1 — this file |
| `bench/results/noise_floor.json` | C2.0.3 — noise floor baseline (pending) |
| `bench/setup_perf.sh` | C2.0.4 — wraps §3.1-§3.5 governor + HT + freq lock (pending) |
| `fajar-lang/CLAUDE.md` §6.7 | Test hygiene rules — no wall-clock asserts in unit tests |
| `fajar-lang/docs/V26_PRODUCTION_PLAN.md` §C2.0 | The plan section that gates this work |

---

## Verification

```bash
cd ~/Documents/fajarquant && test -f bench/METHODOLOGY.md && \
  grep -c 'criterion\|95% CI\|warmup\|governor' bench/METHODOLOGY.md
```

Expected: **≥ 6** matches per V26 plan §C2.0.1 task. This file has
many more (every section references at least one of the four terms).

## Sign-Off

C2.0.1 completed 2026-04-11 by Claude Code session continuation.
**No benchmark in V26 Phase C2.1+ may run without §3-§5 satisfied.**
The companion `bench/setup_perf.sh` script (C2.0.4, pending) will
mechanize §3.1-§3.5 so future benchmark runs are one-command setup.
