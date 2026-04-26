#!/usr/bin/env python3
"""
Phase E E1.4.3 — Smoke gate for `dedup_corpus_exact.py --resume`.

Mirror of `test_dedup_resume.py` for the exact-hash variant. Per CLAUDE
§6.8 R3 (prevention layer per phase) + §6.11 R5 spirit (Makefile
regression gate for long-running interruption-safe pipelines): wired to
`make test-dedup-exact` and asserted by hand on every dedup_corpus_exact.py
change before launching a multi-hour full sweep.

What it covers (must stay green):
    1. State save at shard boundary writes a non-empty state.pkl
    2. Resume reloads digest set, counters, and skips completed shards
    3. Cross-run dedup detection works (run-2 doc matches run-1 doc via
       reloaded digest set)
    4. Reference (non-interrupted) run produces same docs_in / docs_kept /
       docs_dropped as the resume run on identical input
    5. NFKC + lowercase + whitespace-collapse normalization catches
       trivial whitespace/case duplicates that byte-equal hashing would miss

Runtime: < 5 seconds on 12 synthetic docs.
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
SCRIPT = REPO_ROOT / "python" / "phase_e" / "scripts" / "dedup_corpus_exact.py"


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
    # Last row of shard1: byte-exact duplicate of shard0[0] — must be caught
    # cross-run via the digest set reloaded from state.pkl.
    shard1.append(
        {
            "text": "doc 0 kalimat indonesia panjang tujuh kata berbeda",
            "source": "t",
        }
    )
    # Extra row in shard1 differs from shard0[1] only by case + double space.
    # Normalization (NFKC + lowercase + whitespace collapse) must reduce
    # it to the same digest as shard0[1] — guards against regressions in
    # the normalizer.
    shard1.append(
        {
            "text": "DOC 1 kalimat   indonesia panjang tujuh KATA berbeda",
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
    """Run dedup_corpus_exact.py and return the per-source manifest JSON.
    Manifest is only written in non-dry-run mode; returns None for dry-run."""
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
        raise RuntimeError(f"dedup_corpus_exact.py failed (rc={res.returncode})")
    if "--dry-run" in flags:
        return None
    manifest_path = output_root / "test_src" / "_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"manifest missing: {manifest_path}\nstdout:\n{res.stdout}")
    import json as _json
    return _json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="dedup_exact_smoke_"))
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

        # Step 2: --resume completes; cross-run byte-exact dup AND case+whitespace
        # variant must both be caught by the reloaded digest set.
        r2 = _run_dedup(src_root, out_resume, "--resume")
        assert r2["docs_in"] == 12, f"resume docs_in: {r2['docs_in']} != 12"
        assert r2["docs_kept"] == 10, f"resume docs_kept: {r2['docs_kept']} != 10"
        assert r2["docs_dropped"] == 2, f"resume docs_dropped: {r2['docs_dropped']} != 2"
        sd = r2.get("sample_dropped", [])
        assert len(sd) == 2, f"sample_dropped len: {len(sd)} != 2"
        # Both dups should report doc_index in the post-shard0 range (5..11).
        for entry in sd:
            assert "digest_hex" in entry and len(entry["digest_hex"]) == 64, (
                f"bad digest_hex in sample: {entry}"
            )
            assert entry["doc_index"] >= 5, f"unexpected dup at index {entry}"

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

        # Step 4: dry-run remains side-effect-free.
        out_dryrun = tmp / "out_dryrun"
        _ = _run_dedup(src_root, out_dryrun, "--max-docs", "5", "--dry-run", "--resume")
        assert not (out_dryrun / ".progress").exists(), (
            "dry-run + --resume must not write .progress/"
        )
        assert not (out_dryrun / "test_src" / "_manifest.json").is_file(), (
            "dry-run must not write manifest"
        )

        # Step 5: normalization catches case+whitespace variants —
        # already implicit in step 2 (12 in, 10 kept = 2 dups: byte-exact +
        # case/ws variant). Make it explicit with a focused assert.
        # docs_in=12, byte-exact dup of shard0[0] = 1 drop, ws+case variant of
        # shard0[1] = 1 drop. If normalization were missing, only the byte-
        # exact dup would be caught (1 drop, 11 kept) — assertion above
        # would have failed already.
        assert r2["norm_scheme"] == "nfkc-lower-ws", (
            f"unexpected norm_scheme: {r2.get('norm_scheme')!r}"
        )
        assert r2["hash_alg"] == "sha256", (
            f"unexpected hash_alg: {r2.get('hash_alg')!r}"
        )

        print("test-dedup-exact: 5/5 invariants PASS")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
