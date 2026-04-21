#!/usr/bin/env python3
"""V31.C.P3.3 throughput benchmark — measures tokens/sec for the
SlimPajama streaming loader.

Per V31 Master Plan §4.4 C.P3.3 gate: ≥ 10K tok/s. This is a soft
ceiling — tokeniser alone can do >100K tok/s on CPU; the bottleneck
is HF dataset streaming network throughput, not tokenizer compute.

Run:
    cd fajarquant/python/phase_d
    PYTHONPATH=. ../../.venv/bin/python scripts/bench_stream.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intllm.data import slimpajama_stream  # noqa: E402
from intllm.tokenizer import get_tokenizer  # noqa: E402


WARMUP = 5      # cold-start batches excluded from steady-state measurement
N_BATCHES = 30
BATCH_SIZE = 8
SEQ_LEN = 512
GATE_TOK_PER_SEC = 10_000.0


def main() -> int:
    tok = get_tokenizer()
    total = N_BATCHES * BATCH_SIZE * SEQ_LEN
    measured = (N_BATCHES - WARMUP) * BATCH_SIZE * SEQ_LEN
    print(
        f"benchmark: {N_BATCHES} batches × {BATCH_SIZE} × {SEQ_LEN} = "
        f"{total:,} tokens total ({WARMUP} warmup batches discarded)"
    )

    stream = slimpajama_stream(
        tokenizer=tok,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        device="cpu",
        seed=0,
    )

    cold_start_t0 = time.time()
    measured_t0: float | None = None
    n_tok = 0
    for i, batch in enumerate(stream):
        if i == WARMUP:
            measured_t0 = time.time()
            n_tok = 0  # reset post-warmup counter
        n_tok += batch.numel()
        if i + 1 >= N_BATCHES:
            break
    measured_elapsed = time.time() - (measured_t0 or cold_start_t0)
    cold_elapsed = time.time() - cold_start_t0

    cold_tps = total / cold_elapsed
    steady_tps = (n_tok / measured_elapsed) if measured_t0 is not None else cold_tps

    print(f"cold-start total: {cold_elapsed:.1f}s, throughput {cold_tps:,.0f} tok/s")
    print(f"steady-state    : {measured_elapsed:.1f}s, throughput {steady_tps:,.0f} tok/s")
    print(f"gate (steady)   : ≥ {GATE_TOK_PER_SEC:,.0f} tok/s -> "
          f"{'PASS ✓' if steady_tps >= GATE_TOK_PER_SEC else 'FAIL ✗'}")
    return 0 if steady_tps >= GATE_TOK_PER_SEC else 1


if __name__ == "__main__":
    raise SystemExit(main())
