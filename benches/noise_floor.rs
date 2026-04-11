// V26 Phase C2.0.3 deliverable — benches/noise_floor.rs
//
// Establishes the criterion measurement noise floor on the test machine.
// Per bench/METHODOLOGY.md §6, ANY benchmark improvement smaller than
// 2× the noise floor must be reported as "below noise floor", never
// as a number.
//
// What this bench does: nothing. It's a no-op `black_box(0u64)` so the
// only thing being measured is criterion's own measurement overhead +
// scheduler jitter on the current machine. The 95% CI half-width of
// the median for this no-op is the noise floor.
//
// How to use:
//   cargo bench --bench noise_floor              # one run
//   for i in 1 2 3 4 5; do                       # 5 runs per METHODOLOGY §6
//       cargo bench --bench noise_floor
//   done
//   # Then capture the max 95% CI half-width across the 5 runs into
//   # bench/results/noise_floor.json
//
// criterion stores raw timings + estimates in target/criterion/<bench>/
// where <bench> = "noise_floor". The estimates.json file there contains
// median + confidence_interval data.
//
// Note: this bench should be re-run any time bench/setup_perf.sh state
// changes (governor / HT / turbo) to refresh the noise floor under
// paper-grade conditions.

use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn noop_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("noise_floor");

    // METHODOLOGY.md §2 sample protocol
    group.sample_size(100);
    group.warm_up_time(std::time::Duration::from_secs(3));
    group.measurement_time(std::time::Duration::from_secs(10));
    group.confidence_level(0.95);

    group.bench_function("noop_u64", |b| {
        b.iter(|| black_box(0u64));
    });

    group.bench_function("noop_arith", |b| {
        b.iter(|| {
            // Slightly more than a no-op so the optimizer can't fold to
            // a constant. Still <1ns of real work.
            let x = black_box(7u64);
            let y = black_box(13u64);
            black_box(x.wrapping_mul(y))
        });
    });

    group.finish();
}

criterion_group!(benches, noop_bench);
criterion_main!(benches);
