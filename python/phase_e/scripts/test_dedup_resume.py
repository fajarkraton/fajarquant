#!/usr/bin/env python3
"""
Phase E E1.4.1 — Smoke gate for `dedup_corpus.py --resume`.

Per CLAUDE §6.8 R3 (prevention layer per phase) + §6.11 R5 spirit
(Makefile regression gate for long-running interruption-safe pipelines):
this gate is wired to `make test-dedup-resume` and asserted by hand on
every dedup_corpus.py change before launching a multi-hour full sweep.

What it covers (must stay green):
    1. State save at shard boundary writes a non-empty state.pkl
    2. Resume reloads LSH, counters, and skips completed shards
    3. Cross-run dedup detection works (run-2 doc matches run-1 doc via
       reloaded LSH index)
    4. Reference (non-interrupted) run produces same docs_in / docs_kept /
       docs_dropped as the resume run on identical input

Runtime: < 5 seconds on 11 synthetic docs.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
SCRIPT = REPO_ROOT / "python" / "phase_e" / "scripts" / "dedup_corpus.py"


def _make_corpus(src_dir: Path) -> None:
    src_dir.mkdir(parents=True, exist_ok=True)
    shard0 = [
        {"text": f"doc {i} kalimat indonesia panjang tujuh kata berbeda", "source": "t"}
        for i in range(5)
    ]
    shard1 = [
        {"text": f"shard satu doc {i} text uniknya delta epsilon", "source": "t"}
        for i in range(5)
    ]
    # last row of shard1 is an exact duplicate of shard0[0] — must be caught
    # cross-run (i.e., the LSH index reloaded from state.pkl needs to recognize
    # this match even though d0 was inserted in a previous Python process).
    shard1.append(
        {
            "text": "doc 0 kalimat indonesia panjang tujuh kata berbeda",
            "source": "t",
        }
    )
    pq.write_table(
        pa.Table.from_pylist(shard0),
        src_dir / "shard_00000.parquet",
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pylist(shard1),
        src_dir / "shard_00001.parquet",
        compression="zstd",
    )


def _run_dedup(input_root: Path, output_root: Path, *flags: str) -> dict | None:
    """Run dedup_corpus.py and return the per-source manifest JSON. Manifest
    is only written in non-dry-run mode; returns None for dry-run."""
    cmd = [
        str(PYTHON), str(SCRIPT),
        "--input-root", str(input_root),
        "--output-root", str(output_root),
        "--source", "test_src",
        "--progress-every", "0",
        *flags,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        sys.stderr.write(res.stdout)
        sys.stderr.write(res.stderr)
        raise RuntimeError(f"dedup_corpus.py failed (rc={res.returncode})")
    if "--dry-run" in flags:
        return None
    manifest_path = output_root / "test_src" / "_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"manifest missing: {manifest_path}\nstdout:\n{res.stdout}")
    import json as _json
    return _json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="dedup_resume_smoke_"))
    try:
        src_root = tmp / "src"
        out_resume = tmp / "out_resume"
        out_ref = tmp / "out_ref"
        _make_corpus(src_root / "test_src")

        # Step 1: process shard 0 only (max-docs=5 lands at end-of-shard).
        r1 = _run_dedup(src_root, out_resume, "--max-docs", "5", "--resume")
        assert r1["docs_in"] == 5, f"step1 docs_in: {r1['docs_in']} != 5"
        assert r1["docs_kept"] == 5, f"step1 docs_kept: {r1['docs_kept']} != 5"
        assert r1["docs_dropped"] == 0, f"step1 docs_dropped: {r1['docs_dropped']} != 0"
        state_path = out_resume / ".progress" / "test_src" / "state.pkl"
        assert state_path.is_file(), f"state.pkl missing at {state_path}"
        assert state_path.stat().st_size > 100, f"state.pkl too small: {state_path.stat().st_size}"

        # Step 2: --resume completes the run; cross-run dup must be caught.
        r2 = _run_dedup(src_root, out_resume, "--resume")
        assert r2["docs_in"] == 11, f"resume docs_in: {r2['docs_in']} != 11"
        assert r2["docs_kept"] == 10, f"resume docs_kept: {r2['docs_kept']} != 10"
        assert r2["docs_dropped"] == 1, f"resume docs_dropped: {r2['docs_dropped']} != 1"
        sd = r2.get("sample_dropped", [])
        assert sd and sd[0]["key"] == "d10" and sd[0]["matched"] == "d0", (
            f"resume cross-run dedup failed: sample_dropped={sd}"
        )

        # Step 3: reference (non-interrupted) run on a fresh out dir.
        r3 = _run_dedup(src_root, out_ref)
        assert r3["docs_in"] == r2["docs_in"], (
            f"ref/resume docs_in mismatch: {r3['docs_in']} vs {r2['docs_in']}"
        )
        assert r3["docs_kept"] == r2["docs_kept"], (
            f"ref/resume docs_kept mismatch: {r3['docs_kept']} vs {r2['docs_kept']}"
        )
        assert r3["docs_dropped"] == r2["docs_dropped"], (
            f"ref/resume docs_dropped mismatch: {r3['docs_dropped']} vs {r2['docs_dropped']}"
        )

        # Step 4: dry-run remains side-effect-free (no .progress/, no out shards).
        out_dryrun = tmp / "out_dryrun"
        _ = _run_dedup(src_root, out_dryrun, "--max-docs", "5", "--dry-run", "--resume")
        # --dry-run takes precedence over --resume; no state file should exist,
        # and no manifest is written either.
        assert not (out_dryrun / ".progress").exists(), (
            "dry-run + --resume must not write .progress/"
        )
        assert not (out_dryrun / "test_src" / "_manifest.json").is_file(), (
            "dry-run must not write manifest"
        )

        print("test-dedup-resume: 4/4 invariants PASS")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
