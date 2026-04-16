"""Per-op JSONL trace capture for the kernel_sim forward pass.

Plan ref: `docs/V30_SIM_PLAN.md` §P2.2. This module provides a single
TraceWriter class that emits one JSON record per op boundary during
a forward pass. The same schema is produced by three independent
sources so they can be diffed directly:

  * Python-int sim       (this file, P2.2)          — kernel mirror
  * Kernel-actual        (FJTRACE=1 mode, P2.3)     — ground truth under-test
  * HuggingFace float    (hf_reference.py, P2.5)    — float reference

A single schema across the three is the whole point of P2.2 — it
makes `diff.py` (P3.1) trivial: find the first `step` at which any
field diverges beyond the tolerance and you have the first-divergence
op for free.

── Schema (v1) ──────────────────────────────────────────────────────

Every line of a trace file is a compact JSON object with these keys:

  schema_version : int   — always 1 for now; bumping this invalidates
                           cached traces
  step           : int   — monotonic record index within the trace
                           (0, 1, 2, …); disambiguates repeats
  op             : str   — one of OP_NAMES below. Pinning the set of
                           allowed names here prevents P2.3 and P2.5
                           from drifting into spelling variants
  token_idx      : int   — 0-based position in the prompt
  layer          : int   — layer index (0..n_layers-1) or -1 for
                           ops that are not inside a transformer block
                           (embed, final_rmsnorm, argmax)
  shape          : list[int] — tensor shape as a flat JSON array
  dtype          : str   — one of "i64", "i32", "i16", "u8", "f32",
                           "f16"; kernel/sim use i64, HF uses f32/f16
  min            : int|float — minimum value across the tensor
  max            : int|float — maximum value across the tensor
  mean           : int|float — integer-truncated mean for int dtypes
                               (sum // n, matching kernel arithmetic);
                               true float mean for float dtypes
  nnz            : int   — count of non-zero entries (signal for
                           sparsity / pad-collapse detection)
  top5_abs       : list[[idx:int, val:int|float]] — the 5 values with
                           largest absolute magnitude, as (flat_idx,
                           signed_value) pairs. Specified in the plan
                           §P2.2 as a mandatory field for divergence
                           attribution
  hash           : str   — FNV-1a 64-bit hex string ("0x..."). Hashing
                           bit-identical byte streams gives identical
                           hashes across Python and a hypothetical
                           kernel implementation, so a single
                           hash-equality check certifies 0-ULP parity.
                           For float tensors the hash covers the IEEE754
                           bit pattern
  extra          : dict  (optional) — op-specific metadata, e.g.
                           {"winning_token": 42, "best_score": 123456}
                           for argmax records

OP_NAMES (canonical, pinned):

    "embed_lookup"          — per-token embedding dequant
    "pre_attn_rmsnorm"      — per-layer, first norm
    "q_proj"                — Q projection output
    "k_proj"                — K projection output
    "v_proj"                — V projection output
    "attn_out"              — attention block output (before post-norm)
    "post_attn_rmsnorm"     — Gemma-3 post-attn norm (skipped in Llama)
    "attn_residual"         — x + attn_out
    "pre_ffn_rmsnorm"       — per-layer, second norm
    "gate_proj"             — FFN gate output
    "up_proj"               — FFN up output
    "ffn_hidden"            — act(gate) * up / scale
    "down_proj"             — FFN down output
    "post_ffn_rmsnorm"      — Gemma-3 post-ffn norm (skipped in Llama)
    "ffn_residual"          — x + ffn_out
    "final_rmsnorm"         — after all layers
    "argmax"                — LM head argmax; extra has winning token

── Why FNV-1a ───────────────────────────────────────────────────────

FNV-1a 64-bit is trivially implementable in Fajar Lang (6 lines: XOR
+ multiply + wrap). That matters for P2.3, where the kernel must emit
the same hash over the serial line with no dependency on any library.
A tensor hash gives cheap zero-ULP equality without having to dump
every value.

── Overhead ─────────────────────────────────────────────────────────

Tracing is opt-in (`enabled=False` default). When off, `.record()`
returns immediately before computing any stats. When on, each record
costs O(n) for min/max/mean/nnz/top5/hash — fine for per-op granularity
at ~1M values/op.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import IO, Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


SCHEMA_VERSION = 1


# Pinned set of op names — P2.3 and P2.5 MUST emit only these.
# Kept sorted by forward-pass order for readability.
OP_NAMES = frozenset({
    "embed_lookup",
    "pre_attn_rmsnorm",
    "q_proj",
    "k_proj",
    "v_proj",
    "attn_out",
    "post_attn_rmsnorm",
    "attn_residual",
    "pre_ffn_rmsnorm",
    "gate_proj",
    "up_proj",
    "ffn_hidden",
    "down_proj",
    "post_ffn_rmsnorm",
    "ffn_residual",
    "final_rmsnorm",
    "argmax",
})

LAYER_SCOPED_OPS = frozenset(OP_NAMES - {
    "embed_lookup", "final_rmsnorm", "argmax",
})


# FNV-1a 64-bit constants. Fixed by the algorithm spec; do not change
# without bumping SCHEMA_VERSION (traces become incomparable).
FNV_OFFSET_64 = 0xCBF29CE484222325
FNV_PRIME_64 = 0x100000001B3
_U64_MASK = (1 << 64) - 1


def fnv1a_u64_bytes(data: bytes) -> int:
    """FNV-1a 64-bit over a byte sequence. Deterministic and
    reproducible in pure Fajar Lang for P2.3 kernel parity."""
    h = FNV_OFFSET_64
    for b in data:
        h ^= b
        h = (h * FNV_PRIME_64) & _U64_MASK
    return h


def _i64_to_bytes_le(values: Iterable[int]) -> bytes:
    """Serialize int64 tensor values as 8-byte little-endian two's
    complement. Matches how the kernel lays out int64 in memory on
    x86_64, so hash(python bytes) == hash(kernel bytes) is exact."""
    out = bytearray()
    for v in values:
        # Mask to u64 (two's complement wrap identical on both sides)
        out.extend((v & _U64_MASK).to_bytes(8, "little", signed=False))
    return bytes(out)


def _f32_to_bytes_le(values: Iterable[float]) -> bytes:
    import struct
    return b"".join(struct.pack("<f", float(v)) for v in values)


def _tensor_stats_int(
    values: Sequence[int],
) -> Tuple[int, int, int, int, List[Tuple[int, int]], str]:
    """Compute (min, max, mean, nnz, top5_abs, hash) for an int tensor."""
    if not values:
        return 0, 0, 0, 0, [], "0x" + format(FNV_OFFSET_64, "016x")
    vmin = min(values)
    vmax = max(values)
    # Integer-truncated mean (matches kernel sum/n arithmetic)
    n = len(values)
    total = sum(values)
    # Python `//` rounds toward -inf; kernel trunc-divs toward 0.
    # For the mean we want kernel semantics.
    if total >= 0:
        vmean = total // n
    else:
        vmean = -(-total // n)
    nnz = sum(1 for v in values if v != 0)
    # top5_abs: (flat_idx, signed_val), descending by |val|
    # Ties broken by lower flat_idx (deterministic).
    indexed = [(i, v) for i, v in enumerate(values)]
    indexed.sort(key=lambda p: (-abs(p[1]), p[0]))
    top5 = [(i, v) for i, v in indexed[:5]]
    h = fnv1a_u64_bytes(_i64_to_bytes_le(values))
    return vmin, vmax, vmean, nnz, top5, "0x" + format(h, "016x")


def _tensor_stats_float(
    values: Sequence[float],
) -> Tuple[float, float, float, int, List[Tuple[int, float]], str]:
    if not values:
        return 0.0, 0.0, 0.0, 0, [], "0x" + format(FNV_OFFSET_64, "016x")
    vmin = min(values)
    vmax = max(values)
    vmean = sum(values) / len(values)
    nnz = sum(1 for v in values if v != 0.0)
    indexed = [(i, float(v)) for i, v in enumerate(values)]
    indexed.sort(key=lambda p: (-abs(p[1]), p[0]))
    top5 = [(i, v) for i, v in indexed[:5]]
    h = fnv1a_u64_bytes(_f32_to_bytes_le(values))
    return vmin, vmax, vmean, nnz, top5, "0x" + format(h, "016x")


@dataclass
class TraceWriter:
    """Emit one JSON record per op boundary to a stream.

    Use as a context manager or construct directly; call `.record(...)`
    at each op boundary. Disabled by default — construct with
    `enabled=True` or via `TraceWriter.from_env()` which honors the
    `FJ_SIM_TRACE` environment variable (set to a file path to enable).

    Example:

        with TraceWriter(path="/tmp/sim_trace.jsonl", enabled=True) as tw:
            result = forward(tokens, ..., tracer=tw)

    The writer is intentionally simple — no buffering beyond the OS
    default for the file handle, no compression. At ~500 records per
    prompt that costs <5 ms to write, negligible next to the forward
    itself.
    """

    path: Optional[str] = None
    enabled: bool = False
    _fh: Optional[IO[str]] = None
    _step: int = 0
    _closed: bool = False

    @classmethod
    def from_env(cls, default_path: str = "/tmp/sim_trace.jsonl") -> "TraceWriter":
        """Enable tracing iff FJ_SIM_TRACE is set. Value of the env var,
        if non-empty, overrides `default_path`."""
        val = os.environ.get("FJ_SIM_TRACE", "")
        if not val:
            return cls(path=None, enabled=False)
        path = val if val != "1" else default_path
        return cls(path=path, enabled=True)

    def __enter__(self) -> "TraceWriter":
        self._open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _open(self) -> None:
        if self.enabled and self._fh is None and self.path is not None:
            self._fh = open(self.path, "w", encoding="utf-8")

    def close(self) -> None:
        if self._fh is not None and not self._closed:
            self._fh.close()
            self._closed = True

    def record(
        self,
        op: str,
        values: Sequence[Union[int, float]],
        *,
        token_idx: int,
        layer: int,
        shape: Sequence[int],
        dtype: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit one JSONL record. No-op when `enabled=False`."""
        if not self.enabled:
            return
        if op not in OP_NAMES:
            raise ValueError(
                f"unknown op {op!r}; extend OP_NAMES in trace.py if adding a new boundary"
            )
        if op in LAYER_SCOPED_OPS:
            if layer < 0:
                raise ValueError(
                    f"op {op!r} is layer-scoped; pass layer >= 0"
                )
        else:
            if layer != -1:
                raise ValueError(
                    f"op {op!r} is not layer-scoped; pass layer == -1"
                )
        n_expected = 1
        for d in shape:
            n_expected *= d
        if n_expected != len(values):
            raise ValueError(
                f"shape {list(shape)} implies {n_expected} values, got {len(values)}"
            )

        if dtype in ("f32", "f16"):
            vmin, vmax, vmean, nnz, top5, h = _tensor_stats_float(values)
        else:
            vmin, vmax, vmean, nnz, top5, h = _tensor_stats_int(values)

        record: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "step": self._step,
            "op": op,
            "token_idx": token_idx,
            "layer": layer,
            "shape": list(shape),
            "dtype": dtype,
            "min": vmin,
            "max": vmax,
            "mean": vmean,
            "nnz": nnz,
            "top5_abs": [list(p) for p in top5],
            "hash": h,
        }
        if extra:
            record["extra"] = dict(extra)

        self._step += 1
        if self._fh is not None:
            self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")

    @property
    def step(self) -> int:
        return self._step


# ── Noop default ─────────────────────────────────────────────────────

class NoopTracer:
    """Stand-in tracer for call sites where passing None would require
    an `if tracer is not None` check at every op. Accepts the same
    `.record()` signature and discards everything."""

    enabled = False
    step = 0

    def record(self, *args: Any, **kwargs: Any) -> None:
        return

    def __enter__(self) -> "NoopTracer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return

    def close(self) -> None:
        return


NOOP_TRACER = NoopTracer()
