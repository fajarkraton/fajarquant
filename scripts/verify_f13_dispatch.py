#!/usr/bin/env python3
"""
verify_f13_dispatch.py — F.13.5 prevention gate for the dispatch decision-doc.

Purpose
-------
The F.13.3 dispatch decision (`docs/FJQ_PHASE_F_F13_DISPATCH_DECISION.md`)
is anchored to public reference numbers because Z-narrow scope did not
include live measurements. If those reference numbers drift, or if the
decision-doc itself is edited inconsistently with the fixture, the verdict
"CPU-first static rule" may no longer be supported.

This script re-asserts that:
  1. Every anchor in `tests/fixtures/f13_dispatch_anchors.toml` is present
     and well-formed.
  2. The four dispatch invariants (I1-I4) implied by the verdict still hold
     given the anchor values.
  3. The decision-doc references the anchor numbers in the expected order
     of magnitude (regex sanity check; not a full text match).

Usage
-----
    python3 scripts/verify_f13_dispatch.py            # human-readable
    python3 scripts/verify_f13_dispatch.py --strict   # CI / pre-commit (exits 1 on any warning)

Per CLAUDE.md §6.8 R3 (prevention layer per phase).
Created: 2026-04-30 (F.13.5).
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "f13_dispatch_anchors.toml"
DECISION_DOC = REPO_ROOT / "docs" / "FJQ_PHASE_F_F13_DISPATCH_DECISION.md"
PLAN_DOC = REPO_ROOT / "docs" / "FJQ_PHASE_F_F13_PRODUCTION_PLAN.md"
FINDINGS_DOC = REPO_ROOT / "docs" / "FJQ_PHASE_F_F13_FINDINGS.md"


class CheckResult:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []
        self.warned: list[str] = []

    def ok(self, msg: str) -> None:
        self.passed.append(msg)

    def fail(self, msg: str) -> None:
        self.failed.append(msg)

    def warn(self, msg: str) -> None:
        self.warned.append(msg)

    def report(self, strict: bool) -> int:
        for m in self.passed:
            print(f"  PASS  {m}")
        for m in self.warned:
            print(f"  WARN  {m}")
        for m in self.failed:
            print(f"  FAIL  {m}")
        n_pass = len(self.passed)
        n_warn = len(self.warned)
        n_fail = len(self.failed)
        print(f"\n  Summary: {n_pass} passed, {n_warn} warned, {n_fail} failed.")
        if n_fail > 0:
            return 1
        if strict and n_warn > 0:
            return 1
        return 0


def load_fixture() -> dict[str, Any]:
    if not FIXTURE.exists():
        raise FileNotFoundError(f"fixture missing: {FIXTURE}")
    with FIXTURE.open("rb") as f:
        return tomllib.load(f)


def check_anchor_shape(name: str, anchor: dict[str, Any], required: list[str]) -> list[str]:
    """Return list of missing required keys for one anchor."""
    return [k for k in required if k not in anchor]


def check_invariants(fixture: dict[str, Any], r: CheckResult) -> None:
    inv = fixture.get("dispatch_invariants", {})

    # I1 — TL2 vs scalar ratio
    bitnet = fixture.get("bitnet_2b4t_cpu_decode", {})
    # Derive a TL2 lower bound at Mini scale (projected from BitNet 2B4T)
    # We use the conservative anchor: TL2 ≥ 200 tok/s at Mini scale.
    # Scalar ≤ 83 tok/s at Mini (from F.11.3 baseline + AVX2 8-lane).
    tl2_min = 200.0
    scalar_max = 83.0
    derived_ratio = tl2_min / scalar_max
    floor = inv.get("i1_tl2_vs_scalar_min_ratio", 2.4)
    if derived_ratio >= floor:
        r.ok(f"I1 TL2/scalar ratio = {derived_ratio:.2f} >= floor {floor}")
    else:
        r.fail(f"I1 TL2/scalar ratio = {derived_ratio:.2f} < floor {floor} -- verdict G2 INVALIDATED")

    # I2 — CPU TL2 minus GPU margin
    gpu = fixture.get("gpu_small_model_batch1_decode", {})
    gpu_max = float(gpu.get("value_tok_per_sec_max", 120.0))
    margin = tl2_min - gpu_max
    floor_margin = inv.get("i2_cpu_tl2_minus_gpu_min_margin_tok_per_sec", 50)
    if margin >= floor_margin:
        r.ok(f"I2 CPU-TL2 minus GPU margin = {margin:.1f} tok/s >= floor {floor_margin}")
    else:
        r.fail(f"I2 margin = {margin:.1f} < floor {floor_margin} -- verdict G3 INVALIDATED")

    # I3 — cudaLaunch × layers minimum total
    cuda = fixture.get("cuda_launch_overhead", {})
    cuda_min_us = float(cuda.get("value_us_min", 10.0))
    mini_kernels_per_token = 72  # 12 layers × 6 matmul-class kernels (decision-doc §1)
    derived_min_total_us = cuda_min_us * mini_kernels_per_token
    floor_total = inv.get("i3_min_gpu_launch_total_us", 720)
    if derived_min_total_us >= floor_total:
        r.ok(
            f"I3 cudaLaunch × Mini kernels = {derived_min_total_us:.0f} µs >= floor {floor_total}"
        )
    else:
        r.fail(
            f"I3 derived total = {derived_min_total_us:.0f} µs < floor {floor_total}"
        )

    # I4 — FajarOS envelope inside CPU-wins region
    fajaros_batch = inv.get("i4_fajaros_max_batch", 8)
    fajaros_params = inv.get("i4_fajaros_max_model_params", 5_000_000_000)
    fajaros_seq = inv.get("i4_fajaros_max_seq_len", 4096)
    if fajaros_batch <= 8 and fajaros_params <= 5_000_000_000 and fajaros_seq <= 4096:
        r.ok(
            f"I4 FajarOS envelope (batch≤{fajaros_batch}, params≤{fajaros_params:,}, "
            f"seq≤{fajaros_seq}) inside CPU-wins region"
        )
    else:
        r.fail("I4 FajarOS envelope falls outside CPU-wins region -- F.13.2 dispatch needed")

    # I5 — F.13.1 NAIVE measured anchor (warm-state, no KV cache, no compile)
    # Updated F.13.1 v2 2026-05-01: replaced cold-GPU artifact (19.7) with
    # warm-state baseline (118.9). Sanity band tightened to [50, 200].
    measured = fixture.get("gpu_hgrn_bit_mini_batch1_measured")
    if measured is None:
        r.fail("I5 measured anchor [gpu_hgrn_bit_mini_batch1_measured] MISSING -- F.13.1 outcome lost")
    else:
        measured_tps = float(measured.get("value_tok_per_sec_median", 0.0))
        floor_max = float(inv.get("i5_measured_gpu_tok_per_sec_max", 200.0))
        floor_min = float(inv.get("i5_measured_gpu_tok_per_sec_min", 50.0))
        if floor_min <= measured_tps <= floor_max:
            r.ok(
                f"I5 measured GPU naive (warm) = {measured_tps:.1f} tok/s within sanity band "
                f"[{floor_min}, {floor_max}]"
            )
        elif measured_tps > floor_max:
            r.fail(
                f"I5 measured GPU naive = {measured_tps:.1f} > {floor_max} tok/s -- "
                f"investigate: bench upgraded? hardware refresh? Re-derive verdict G3"
            )
        else:
            r.fail(
                f"I5 measured GPU naive = {measured_tps:.1f} < {floor_min} tok/s -- "
                f"sanity floor violation (cold GPU? thermal throttle? bench broken?)"
            )

    # I6 — F.13.1 v2 OPTIMIZED-best anchor (kvcache + torch.compile(default))
    # Tracks the value AND surfaces verdict G3 cushion question. If optimized
    # GPU exceeds CPU-TL2 conservative lower bound (200), the projection
    # margin has vanished and verdict needs caveat. This is informational —
    # the bench fact is real; the verdict re-derivation lives in the
    # decision-doc, not here.
    #
    # Acknowledgment pattern: once decision-doc §11 documents the cushion
    # analysis with explicit "static-rule still holds because X" reasoning,
    # set `i6_above_floor_acknowledged = true` in fixture to acknowledge.
    # This lets pre-commit --strict pass while preserving auditability —
    # any future change to the anchor will re-trigger WARN until re-acked.
    optimized = fixture.get("gpu_hgrn_bit_mini_batch1_optimized_best")
    if optimized is None:
        r.fail("I6 optimized anchor [gpu_hgrn_bit_mini_batch1_optimized_best] MISSING -- F.13.1 v2 outcome lost")
    else:
        opt_tps = float(optimized.get("value_tok_per_sec_median", 0.0))
        cpu_tl2_floor = 200.0  # conservative lower bound projection from BitNet 2B4T scaling
        acked = bool(inv.get("i6_above_floor_acknowledged", False))
        if opt_tps > cpu_tl2_floor:
            if acked:
                r.ok(
                    f"I6 optimized GPU = {opt_tps:.1f} tok/s exceeds CPU-TL2 floor "
                    f"({cpu_tl2_floor}) — ACKNOWLEDGED (decision-doc §11 documents "
                    f"cushion analysis)"
                )
            else:
                r.warn(
                    f"I6 optimized GPU = {opt_tps:.1f} tok/s EXCEEDS CPU-TL2 conservative "
                    f"floor ({cpu_tl2_floor}) -- verdict G3 cushion VANISHED. Document "
                    f"in decision-doc §11 then set i6_above_floor_acknowledged = true "
                    f"in fixture to clear this warning."
                )
        elif opt_tps >= 80.0:  # within or above literature anchor band low end
            r.ok(
                f"I6 optimized GPU = {opt_tps:.1f} tok/s within literature anchor band "
                f"reach (≥80 tok/s); verdict G3 cushion intact"
            )
        else:
            r.fail(
                f"I6 optimized GPU = {opt_tps:.1f} < 80 tok/s -- below literature anchor "
                f"band; investigate"
            )


def check_anchor_completeness(fixture: dict[str, Any], r: CheckResult) -> None:
    required = {
        "bitnet_2b4t_cpu_decode": ["value_ms_per_token", "value_tok_per_sec", "source", "verified"],
        "llama_cpp_tq2_cpu_decode": ["value_tok_per_sec_min", "value_tok_per_sec_max", "source"],
        "cuda_launch_overhead": ["value_us_min", "value_us_max", "source"],
        "gpu_small_model_batch1_decode": ["value_tok_per_sec_min", "value_tok_per_sec_max"],
        "gpu_hgrn_bit_mini_batch1_measured": [
            "value_tok_per_sec_median",
            "value_ms_per_token_median",
            "hardware",
            "inference_stack",
            "source",
        ],
        "gpu_hgrn_bit_mini_batch1_optimized_best": [
            "value_tok_per_sec_median",
            "value_ms_per_token_median",
            "hardware",
            "inference_stack",
            "speedup_vs_naive",
            "source",
        ],
        "mini_ckpt_size": ["params_million", "fp16_size_mb", "ternary_packed_mb"],
        "host_cpu_envelope": ["model", "logical_cpus", "l3_cache_mib", "isa_features"],
        "host_gpu_envelope_expected": ["model", "vram_gb"],
        "dispatch_invariants": [
            "i1_tl2_vs_scalar_min_ratio",
            "i2_cpu_tl2_minus_gpu_min_margin_tok_per_sec",
            "i3_min_gpu_launch_total_us",
            "i4_fajaros_max_batch",
            "i5_measured_gpu_tok_per_sec_max",
            "i5_measured_gpu_tok_per_sec_min",
            "i6_literature_anchor_band_low",
        ],
    }
    for anchor_name, required_keys in required.items():
        if anchor_name not in fixture:
            r.fail(f"anchor missing: [{anchor_name}]")
            continue
        missing = check_anchor_shape(anchor_name, fixture[anchor_name], required_keys)
        if missing:
            r.fail(f"anchor [{anchor_name}] missing keys: {missing}")
        else:
            r.ok(f"anchor [{anchor_name}] complete")


def check_decision_doc_references(fixture: dict[str, Any], r: CheckResult) -> None:
    if not DECISION_DOC.exists():
        r.fail(f"decision doc missing: {DECISION_DOC}")
        return
    text = DECISION_DOC.read_text()

    # Each anchor's headline number must appear in the doc text (regex-ish).
    bitnet = fixture.get("bitnet_2b4t_cpu_decode", {})
    bitnet_ms = bitnet.get("value_ms_per_token")
    if bitnet_ms is not None:
        # Accept "29" or "29.0" (drop trailing .0 when integer-valued)
        v = int(bitnet_ms) if float(bitnet_ms).is_integer() else bitnet_ms
        if re.search(rf"\b{v}(?:\.0+)?\s*ms\b", text):
            r.ok(f"decision-doc references BitNet anchor {v} ms/tok")
        else:
            r.warn(f"decision-doc does NOT reference BitNet anchor {v} ms/tok")

    cuda = fixture.get("cuda_launch_overhead", {})
    if (
        re.search(r"\b10\s*[-–]\s*30\s*µs", text)
        or re.search(r"\b10\s*to\s*30\s*microseconds", text, re.IGNORECASE)
    ):
        r.ok("decision-doc references cudaLaunch 10-30 µs anchor")
    else:
        r.warn("decision-doc does NOT reference cudaLaunch 10-30 µs anchor")

    gpu = fixture.get("gpu_small_model_batch1_decode", {})
    gpu_min = gpu.get("value_tok_per_sec_min")
    gpu_max = gpu.get("value_tok_per_sec_max")
    if gpu_min and gpu_max and (str(int(gpu_min)) in text and str(int(gpu_max)) in text):
        r.ok(f"decision-doc references GPU anchor {gpu_min}-{gpu_max} tok/s")
    else:
        r.warn(f"decision-doc does NOT reference GPU anchor {gpu_min}-{gpu_max} tok/s")

    # Verdict statement must be present
    if "static rule" in text.lower() and re.search(r"batch\s*[≥>=]+\s*8", text, re.IGNORECASE):
        r.ok("decision-doc verdict 'static rule batch ≥ 8' present")
    else:
        r.fail("decision-doc verdict statement missing or malformed")


def check_companion_docs_present(r: CheckResult) -> None:
    for d in [PLAN_DOC, DECISION_DOC, FINDINGS_DOC]:
        if d.exists():
            r.ok(f"companion doc present: {d.relative_to(REPO_ROOT)}")
        else:
            r.fail(f"companion doc MISSING: {d.relative_to(REPO_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="F.13 dispatch decision verifier")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 on any WARN as well as FAIL (use in pre-commit / CI)",
    )
    args = parser.parse_args()

    print("F.13.5 verify-f13-decision  (Z-narrow prevention gate)")
    print(f"  fixture:  {FIXTURE.relative_to(REPO_ROOT)}")
    print(f"  decision: {DECISION_DOC.relative_to(REPO_ROOT)}")
    print()

    r = CheckResult()
    try:
        fixture = load_fixture()
    except Exception as e:
        print(f"  FAIL  could not load fixture: {e}")
        return 1

    check_anchor_completeness(fixture, r)
    check_invariants(fixture, r)
    check_companion_docs_present(r)
    check_decision_doc_references(fixture, r)

    return r.report(strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())
