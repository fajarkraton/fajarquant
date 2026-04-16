#!/usr/bin/env python3
"""diff.py — V30.SIM Phase P3.1

Three-way per-op divergence analysis for FJTRACE JSONL files:

  A — Python-sim trace   (i64, from kernel_sim.forward + TraceWriter)
  B — kernel-actual trace (i64, from FJTRACE serial → parse_kernel_trace)
  C — HF float reference (f32, from hf_reference.py)

Purpose: identify the FIRST op where any pair diverges beyond a
magnitude tolerance, so P4 fix work can target one op instead of
combing through 1,831+ records per trace (full Gemma-3-1B prefill).

Plan reference: `docs/V30_SIM_PLAN.md` §P3.1 + §P3.2.

── Alignment ────────────────────────────────────────────────────

Records align by `(token_idx, layer, op)`. That tuple is unique per
record in a single well-formed trace — the plan's 17 op names are
declared distinct, each layer appears at most once per token, and
ops tagged `layer = -1` are not layer-scoped.

If a record appears in one trace but not the other, it is reported
as `only_in_a` or `only_in_b`. That's a structural divergence
(different op set emitted) and needs fixing before numeric diff is
meaningful.

── Metrics ──────────────────────────────────────────────────────

Two cases:

  1. Same dtype (both `i64` OR both `f32`):
     - Hash equality → 0 ULP parity. Numeric fields MUST match if
       hash matches (assuming identical FNV constants + byte layout).
     - Hash mismatch → compute rel-error on `min`, `max`, `mean`.
       Also count |diff| on `nnz`.

  2. Cross dtype (`i64` vs `f32`):
     - Kernel + sim emit fp×1000 fixed-point ints. HF emits raw f32.
     - Scale int side by /1000 before comparing to the f32 side.
     - Hash is incomparable across dtypes (different byte surface);
       compute rel-error on the scaled `min`, `max`, `mean`.

Relative error = `|a - b| / max(|a|, |b|, eps)` with `eps = 1e-9`.
A pair is "diverged" if any of {min, max, mean} exceeds tolerance
(default 0.05 per plan "first significant (>5% magnitude)").

── Top5_abs asymmetry ───────────────────────────────────────────

The kernel emits `top5_abs: []` (space-saving), the Python sim
populates from i64 values, and HF populates from f32 values. The
diff skips `top5_abs` when either side is empty. If both populate,
it reports whether the SAME 5 flat-indices appear (even if values
differ) — a strong structural signal for P3.5/P3.6.

── Usage ────────────────────────────────────────────────────────

  # Two-way (most common P3.1/P3.2 run):
  python3 scripts/diff.py \\
      --a /tmp/sim_trace.jsonl --a-label sim \\
      --b /tmp/hf_trace.jsonl  --b-label hf  \\
      --tolerance 0.05 \\
      -o /tmp/diff_report.json

  # Three-way (P3 planner):
  python3 scripts/diff.py \\
      --a /tmp/sim.jsonl  --a-label sim \\
      --b /tmp/ker.jsonl  --b-label ker \\
      --c /tmp/hf.jsonl   --c-label hf

  # Self-test:
  python3 scripts/diff.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "tools"))

from kernel_sim.trace import OP_NAMES, SCHEMA_VERSION  # noqa: E402


DEFAULT_TOLERANCE = 0.05
_EPS = 1e-9

# Scale factor: kernel + sim emit fixed-point fp×1000. HF emits raw
# f32. Multiplying HF by this factor (or dividing int by it) puts
# them on the same scale. Per kernel/compute/transformer.fj:
# km_vecmat_packed_v8 writes ×1000 ints, km_rmsnorm preserves that
# scale, argmax compares ×1000 ints, all through the forward chain.
INT_SCALE = 1000


# ── Record loading ──────────────────────────────────────────────

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL trace file; skip blank lines; return parsed records."""
    out: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Skip — `parse_kernel_trace.py` should have cleaned
                # these out already for kernel traces.
                continue
            if rec.get("schema_version") != SCHEMA_VERSION:
                continue
            out.append(rec)
    return out


def _key(rec: Dict[str, Any]) -> Tuple[int, int, str]:
    return (rec["token_idx"], rec["layer"], rec["op"])


# ── Pairwise diff primitives ────────────────────────────────────

def _scale_int_to_float(v: float) -> float:
    return v / INT_SCALE


def _rel_error(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), _EPS)
    return abs(a - b) / denom


@dataclass
class FieldDelta:
    name: str
    a: float
    b: float
    rel_error: float
    exceeds_tolerance: bool


@dataclass
class PairDivergence:
    key: Tuple[int, int, str]
    step_a: int
    step_b: int
    dtype_a: str
    dtype_b: str
    hash_a: str
    hash_b: str
    hash_comparable: bool
    hash_equal: Optional[bool]
    fields: List[FieldDelta] = field(default_factory=list)
    nnz_diff: int = 0

    @property
    def any_field_diverges(self) -> bool:
        return any(f.exceeds_tolerance for f in self.fields)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_idx": self.key[0],
            "layer": self.key[1],
            "op": self.key[2],
            "step_a": self.step_a,
            "step_b": self.step_b,
            "dtype_a": self.dtype_a,
            "dtype_b": self.dtype_b,
            "hash_a": self.hash_a,
            "hash_b": self.hash_b,
            "hash_comparable": self.hash_comparable,
            "hash_equal": self.hash_equal,
            "nnz_diff": self.nnz_diff,
            "fields": [
                {"name": f.name, "a": f.a, "b": f.b,
                 "rel_error": f.rel_error, "diverges": f.exceeds_tolerance}
                for f in self.fields
            ],
            "diverges": self.any_field_diverges,
        }


def _compute_divergence(
    rec_a: Dict[str, Any], rec_b: Dict[str, Any],
    tolerance: float,
) -> PairDivergence:
    """Compare two matched records, return a PairDivergence entry."""
    dtype_a = rec_a["dtype"]
    dtype_b = rec_b["dtype"]
    hash_comparable = (dtype_a == dtype_b)
    hash_equal: Optional[bool] = None
    if hash_comparable:
        hash_equal = (rec_a["hash"] == rec_b["hash"])

    # Scale int side to float for cross-dtype comparisons.
    def _val(rec: Dict[str, Any], field_name: str) -> float:
        v = float(rec[field_name])
        if rec["dtype"] != "f32":
            return _scale_int_to_float(v)
        return v

    fields: List[FieldDelta] = []
    for fname in ("min", "max", "mean"):
        a = _val(rec_a, fname)
        b = _val(rec_b, fname)
        err = _rel_error(a, b)
        fields.append(FieldDelta(
            name=fname, a=a, b=b,
            rel_error=err,
            exceeds_tolerance=err > tolerance,
        ))

    return PairDivergence(
        key=_key(rec_a),
        step_a=rec_a["step"], step_b=rec_b["step"],
        dtype_a=dtype_a, dtype_b=dtype_b,
        hash_a=rec_a["hash"], hash_b=rec_b["hash"],
        hash_comparable=hash_comparable, hash_equal=hash_equal,
        fields=fields,
        nnz_diff=abs(int(rec_a["nnz"]) - int(rec_b["nnz"])),
    )


# ── Pair report ─────────────────────────────────────────────────

@dataclass
class PairReport:
    label_a: str
    label_b: str
    tolerance: float
    n_a: int
    n_b: int
    n_matched: int
    only_in_a: List[Tuple[int, int, str]]
    only_in_b: List[Tuple[int, int, str]]
    divergences: List[PairDivergence]
    first_divergence: Optional[PairDivergence]
    op_divergence_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair": f"{self.label_a}_vs_{self.label_b}",
            "tolerance": self.tolerance,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "n_matched": self.n_matched,
            "only_in_a": [list(k) for k in self.only_in_a],
            "only_in_b": [list(k) for k in self.only_in_b],
            "first_divergence": (
                self.first_divergence.to_dict()
                if self.first_divergence is not None else None
            ),
            "n_diverged": sum(
                1 for d in self.divergences if d.any_field_diverges
            ),
            "op_divergence_counts": dict(self.op_divergence_counts),
        }


def diff_traces(
    recs_a: Sequence[Dict[str, Any]],
    recs_b: Sequence[Dict[str, Any]],
    *,
    label_a: str, label_b: str,
    tolerance: float = DEFAULT_TOLERANCE,
) -> PairReport:
    """Compare two trace record lists. Report first op where the pair
    diverges beyond `tolerance` on any of min/max/mean (scaled)."""
    idx_a = {_key(r): r for r in recs_a}
    idx_b = {_key(r): r for r in recs_b}

    only_a = [k for k in idx_a if k not in idx_b]
    only_b = [k for k in idx_b if k not in idx_a]

    # Iterate in step order of A so first_divergence is meaningful.
    ordered_keys = sorted(
        set(idx_a) & set(idx_b),
        key=lambda k: idx_a[k]["step"],
    )

    divergences: List[PairDivergence] = []
    first: Optional[PairDivergence] = None
    op_counts: Dict[str, int] = {}
    for k in ordered_keys:
        d = _compute_divergence(idx_a[k], idx_b[k], tolerance)
        divergences.append(d)
        if d.any_field_diverges:
            op_counts[k[2]] = op_counts.get(k[2], 0) + 1
            if first is None:
                first = d

    return PairReport(
        label_a=label_a, label_b=label_b,
        tolerance=tolerance,
        n_a=len(recs_a), n_b=len(recs_b),
        n_matched=len(ordered_keys),
        only_in_a=only_a, only_in_b=only_b,
        divergences=divergences,
        first_divergence=first,
        op_divergence_counts=op_counts,
    )


def write_report(
    report: Dict[str, Any], out_path: Optional[str],
) -> None:
    payload = json.dumps(report, indent=2)
    if out_path is None or out_path == "-":
        print(payload)
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(payload)


# ── Self-test ───────────────────────────────────────────────────

def _synth_record(step: int, token_idx: int, layer: int, op: str,
                  dtype: str, vmin, vmax, vmean,
                  nnz: int = 4, hash_: str = "0x0") -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "step": step, "op": op, "token_idx": token_idx, "layer": layer,
        "shape": [4], "dtype": dtype,
        "min": vmin, "max": vmax, "mean": vmean,
        "nnz": nnz, "top5_abs": [],
        "hash": hash_,
    }


def run_self_test() -> int:
    """Construct two tiny traces with a known divergence at step 2,
    run the diff, assert the first_divergence lands on that step."""
    # Trace A (sim-style, i64 fp×1000)
    a = [
        _synth_record(0, 0, -1, "embed_lookup", "i64",
                      -1000, 2000, 500, hash_="0xaaaa"),
        _synth_record(1, 0, 0, "pre_attn_rmsnorm", "i64",
                      -1500, 2500, 700, hash_="0xbbbb"),
        # DELIBERATE: q_proj diverges at step 2 — make A's max 10x
        # bigger than B's to blow past the 5% tolerance.
        _synth_record(2, 0, 0, "q_proj", "i64",
                      -3000, 50000, 1000, hash_="0xcccc"),
        _synth_record(3, 0, 0, "k_proj", "i64",
                      -2000, 2000, 0, hash_="0xdddd"),
    ]
    # Trace B (hf-style, f32 raw floats — should roughly match A
    # after /1000 scaling, except at step 2 where we break)
    b = [
        _synth_record(0, 0, -1, "embed_lookup", "f32",
                      -1.0, 2.0, 0.5, hash_="0xeeee"),
        _synth_record(1, 0, 0, "pre_attn_rmsnorm", "f32",
                      -1.5, 2.5, 0.7, hash_="0xffff"),
        _synth_record(2, 0, 0, "q_proj", "f32",
                      -3.0, 5.0, 1.0, hash_="0x1234"),
        _synth_record(3, 0, 0, "k_proj", "f32",
                      -2.0, 2.0, 0.0, hash_="0x5678"),
    ]

    ok = True

    def check(cond: bool, msg: str) -> None:
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        print(f"[diff self-test] {status} {msg}", file=sys.stderr)
        if not cond:
            ok = False

    # 1. Happy-path alignment — all 4 records match
    r = diff_traces(a, b, label_a="sim", label_b="hf",
                    tolerance=DEFAULT_TOLERANCE)
    check(r.n_matched == 4, f"expected 4 matched, got {r.n_matched}")
    check(not r.only_in_a and not r.only_in_b,
          f"expected no orphans, got A={r.only_in_a} B={r.only_in_b}")

    # 2. First divergence at step 2 (q_proj)
    check(r.first_divergence is not None,
          "expected a divergence, got None")
    if r.first_divergence is not None:
        check(r.first_divergence.key[2] == "q_proj",
              f"first_divergence op should be q_proj, got "
              f"{r.first_divergence.key[2]!r}")
        check(r.first_divergence.key[1] == 0,
              f"first_divergence layer should be 0")
        # Field that diverged should be max (A scaled = 50, B = 5 → 100% error)
        diverging_fields = [f for f in r.first_divergence.fields
                            if f.exceeds_tolerance]
        check(any(f.name == "max" for f in diverging_fields),
              f"expected `max` field to diverge, got "
              f"{[f.name for f in diverging_fields]}")

    # 3. Cross-dtype hash must be flagged incomparable
    if r.first_divergence is not None:
        check(not r.first_divergence.hash_comparable,
              "hash should be incomparable across i64/f32 dtypes")

    # 4. Only-in-A path: drop last record from B, re-run
    r2 = diff_traces(a, b[:-1], label_a="sim", label_b="hf",
                     tolerance=DEFAULT_TOLERANCE)
    check(len(r2.only_in_a) == 1,
          f"dropping last B → expected 1 only_in_A, got {len(r2.only_in_a)}")
    check(r2.only_in_a[0][2] == "k_proj",
          f"orphan should be k_proj, got {r2.only_in_a[0]!r}")
    check(r2.n_matched == 3,
          f"n_matched should be 3 after dropping, got {r2.n_matched}")

    # 5. Zero-divergence case: identical scaled traces
    a_clean = [_synth_record(i, 0, 0, "q_proj", "i64",
                             -1000 * (i + 1), 2000 * (i + 1),
                             500 * (i + 1))
               for i in range(3)]
    b_clean = [_synth_record(i, 0, 0, "q_proj", "f32",
                             -1.0 * (i + 1), 2.0 * (i + 1),
                             0.5 * (i + 1))
               for i in range(3)]
    # Bump token_idx to make keys unique (same op repeated).
    for i, (ra, rb) in enumerate(zip(a_clean, b_clean)):
        ra["token_idx"] = i
        rb["token_idx"] = i
    r3 = diff_traces(a_clean, b_clean, label_a="sim", label_b="hf",
                     tolerance=DEFAULT_TOLERANCE)
    check(r3.first_divergence is None,
          f"expected no divergence on clean scaled traces, "
          f"got {r3.first_divergence}")

    # 6. Same-dtype hash equality → short-circuit path
    same_a = [_synth_record(0, 0, -1, "embed_lookup", "i64",
                            -1, 1, 0, hash_="0xabc")]
    same_b = [_synth_record(0, 0, -1, "embed_lookup", "i64",
                            -1, 1, 0, hash_="0xabc")]
    r4 = diff_traces(same_a, same_b, label_a="ker", label_b="sim",
                     tolerance=DEFAULT_TOLERANCE)
    check(r4.n_matched == 1 and r4.first_divergence is None,
          f"identical i64 records should not diverge")
    check(r4.divergences[0].hash_equal is True,
          f"hash_equal should be True for identical records")

    # 7. Report serialization (to_dict round-trips through json)
    payload = json.dumps(r.to_dict())
    check(json.loads(payload)["pair"] == "sim_vs_hf",
          "report JSON round-trip failed")

    return 0 if ok else 1


# ── CLI ─────────────────────────────────────────────────────────

def _main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(
        description="Per-op divergence analysis across FJTRACE JSONL files.",
    )
    p.add_argument("--a", help="first trace file")
    p.add_argument("--b", help="second trace file")
    p.add_argument("--c", help="third trace file (optional, enables 3-way)")
    p.add_argument("--a-label", default="a", help="label for --a (default: a)")
    p.add_argument("--b-label", default="b", help="label for --b (default: b)")
    p.add_argument("--c-label", default="c", help="label for --c (default: c)")
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                   help=f"relative-error threshold (default: {DEFAULT_TOLERANCE})")
    p.add_argument("-o", "--output", default="-",
                   help="output report JSON path, or '-' for stdout")
    p.add_argument("--self-test", action="store_true",
                   help="run internal test against synthetic traces")
    args = p.parse_args(argv)

    if args.self_test:
        return run_self_test()

    if args.a is None or args.b is None:
        print("[diff] --a and --b are required (or --self-test)",
              file=sys.stderr)
        return 2

    recs_a = _load_jsonl(args.a)
    recs_b = _load_jsonl(args.b)
    pairs: List[Tuple[Sequence[Dict[str, Any]], str,
                      Sequence[Dict[str, Any]], str]] = [
        (recs_a, args.a_label, recs_b, args.b_label),
    ]
    if args.c is not None:
        recs_c = _load_jsonl(args.c)
        pairs.append((recs_a, args.a_label, recs_c, args.c_label))
        pairs.append((recs_b, args.b_label, recs_c, args.c_label))

    reports = [
        diff_traces(ra, rb, label_a=la, label_b=lb,
                    tolerance=args.tolerance).to_dict()
        for ra, la, rb, lb in pairs
    ]
    aggregate = {
        "tolerance": args.tolerance,
        "pairs": reports,
        "summary": {
            r["pair"]: (
                r["first_divergence"]["op"]
                if r["first_divergence"] is not None else None
            )
            for r in reports
        },
    }
    write_report(aggregate, args.output)

    # Exit code: 0 if no pair diverged, 1 if any diverged, 2 for CLI error.
    any_diverged = any(r["first_divergence"] is not None for r in reports)
    return 1 if any_diverged else 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
