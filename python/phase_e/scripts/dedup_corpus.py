#!/usr/bin/env python3
"""
Phase E E1.4 — MinHash LSH near-duplicate dedup on the bilingual corpus.

Per FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md §E1.4 (Falcon-Edge / RedPajama
recipe): 5-gram word shingles, 128 MinHash permutations, Jaccard threshold 0.85,
first-seen wins.

V31.E1.4.0 first slice: ID corpus only (10.67M docs, 38.92 GB raw text spread
across 146 shards under data/phase_e/corpus_id/). Bilingual extension (EN
subset registry from Phase D slimpajama_stream) lands in a follow-up E1.4.x
sub-task once §E1.2 is wired.

Layout:
    data/phase_e/corpus_id/<source>/shard_NNNNN.parquet   (input)
    data/phase_e/corpus_id_dedup/<source>/shard_NNNNN.parquet (output)
    data/phase_e/corpus_id_dedup/<source>/_manifest.json   (per-source dedup stats)

Verify (runnable, per CLAUDE §6.8 R2):
    .venv/bin/python python/phase_e/scripts/dedup_corpus.py \
        --source wikipedia_id --max-docs 10000 --dry-run

Notes on §6.11 cross-repo training-script rule (resilience):
    Long full-corpus runs (10.67M docs, ~hours) should checkpoint dedup state
    every K docs to allow --resume. V31.E1.4.0 ships the dry-run path only;
    full-run resilience layers (atomic per-shard output + .progress journal +
    --resume-auto) are scoped to E1.4.1 once dry-run is validated.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "phase_e" / "corpus_id"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "phase_e" / "corpus_id_dedup"
SHARD_TARGET_BYTES = 256 * 1024 * 1024  # match build_id_corpus.py

# RedPajama / Falcon-Edge recipe defaults.
DEFAULT_NUM_PERM = 128
DEFAULT_THRESHOLD = 0.85
DEFAULT_SHINGLE_SIZE = 5  # word 5-grams

# Whitespace tokenization with Unicode-aware splitting. Lowercase upstream of
# shingle extraction so casing differences don't inflate Jaccard distance.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass
class DedupReport:
    source: str
    started_at: str
    finished_at: str | None = None
    elapsed_s: float = 0.0
    docs_in: int = 0
    docs_kept: int = 0
    docs_dropped: int = 0
    bytes_text_in: int = 0
    bytes_text_kept: int = 0
    shards_in: int = 0
    shards_out: int = 0
    threshold: float = DEFAULT_THRESHOLD
    num_perm: int = DEFAULT_NUM_PERM
    shingle_size: int = DEFAULT_SHINGLE_SIZE
    dry_run: bool = False
    error: str | None = None
    sample_dropped: list[dict] = field(default_factory=list)

    @property
    def dedup_rate(self) -> float:
        return (self.docs_dropped / self.docs_in) if self.docs_in else 0.0

    def to_json(self) -> str:
        out = dict(self.__dict__)
        out["dedup_rate"] = round(self.dedup_rate, 6)
        return json.dumps(out, indent=2, ensure_ascii=False)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _shingles(text: str, size: int) -> set[str]:
    """5-gram word shingles, lowercased. Returns empty set for under-size docs."""
    tokens = _TOKEN_RE.findall(text.lower())
    if len(tokens) < size:
        # Short docs: degenerate to a single shingle of all tokens so the
        # MinHash isn't empty (datasketch crashes on empty update set).
        return {" ".join(tokens)} if tokens else {""}
    return {" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def _iter_source_shards(source_dir: Path) -> Iterator[Path]:
    yield from sorted(source_dir.glob("shard_*.parquet"))


def _iter_docs_in_shard(shard_path: Path) -> Iterator[tuple[str, str]]:
    """Yield (text, source) tuples streaming row-by-row from a parquet shard."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(shard_path)
    for batch in pf.iter_batches(batch_size=1024, columns=["text", "source"]):
        col_text = batch.column("text").to_pylist()
        col_source = batch.column("source").to_pylist()
        for t, s in zip(col_text, col_source):
            yield t, s


def _list_source_dirs(input_root: Path, source_filter: str | None) -> list[Path]:
    if not input_root.is_dir():
        return []
    dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])
    if source_filter == "all" or source_filter is None:
        return dirs
    # Match by directory name == "<owner>__<dataset>" or by suffix substring.
    out = [p for p in dirs if source_filter in p.name]
    return out


def _write_shard_atomic(out_dir: Path, shard_index: int, rows: list[dict]) -> tuple[Path, int]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        return out_dir / f"shard_{shard_index:05d}.parquet", 0
    table = pa.Table.from_pylist(rows)
    final_path = out_dir / f"shard_{shard_index:05d}.parquet"
    tmp_path = final_path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp_path, compression="zstd")
    size = tmp_path.stat().st_size
    os.replace(tmp_path, final_path)
    return final_path, size


def dedup_source(
    source_dir: Path,
    output_root: Path,
    max_docs: int | None,
    threshold: float,
    num_perm: int,
    shingle_size: int,
    dry_run: bool,
    progress_every: int,
) -> DedupReport:
    from datasketch import MinHash, MinHashLSH

    report = DedupReport(
        source=source_dir.name,
        started_at=_now_iso(),
        threshold=threshold,
        num_perm=num_perm,
        shingle_size=shingle_size,
        dry_run=dry_run,
    )
    out_dir = output_root / source_dir.name
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    try:
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        shards = list(_iter_source_shards(source_dir))
        report.shards_in = len(shards)

        buffer: list[dict] = []
        buffer_bytes = 0
        shard_index = 0

        for shard in shards:
            for text, source_name in _iter_docs_in_shard(shard):
                report.docs_in += 1
                report.bytes_text_in += len(text.encode("utf-8"))

                shingles = _shingles(text, shingle_size)
                m = MinHash(num_perm=num_perm)
                for s in shingles:
                    m.update(s.encode("utf-8"))

                key = f"d{report.docs_in - 1}"
                neighbors = lsh.query(m)
                if neighbors:
                    report.docs_dropped += 1
                    if len(report.sample_dropped) < 5:
                        report.sample_dropped.append(
                            {
                                "key": key,
                                "matched": neighbors[0],
                                "preview": text[:160],
                            }
                        )
                    continue

                lsh.insert(key, m)
                report.docs_kept += 1
                report.bytes_text_kept += len(text.encode("utf-8"))

                if not dry_run:
                    row = {"text": text, "source": source_name}
                    buffer.append(row)
                    buffer_bytes += len(text.encode("utf-8"))
                    if buffer_bytes >= SHARD_TARGET_BYTES:
                        _, _ = _write_shard_atomic(out_dir, shard_index, buffer)
                        report.shards_out += 1
                        shard_index += 1
                        buffer = []
                        buffer_bytes = 0

                if progress_every and report.docs_in % progress_every == 0:
                    elapsed = time.time() - t_start
                    rate = report.docs_in / elapsed if elapsed > 0 else 0.0
                    print(
                        f"  ... {report.docs_in:>10d} docs "
                        f"({report.docs_dropped:>8d} dup, "
                        f"{report.dedup_rate * 100:5.2f}% rate, "
                        f"{rate:>6.0f} doc/s)",
                        flush=True,
                    )

                if max_docs is not None and report.docs_in >= max_docs:
                    break
            if max_docs is not None and report.docs_in >= max_docs:
                break

        if not dry_run and buffer:
            _, _ = _write_shard_atomic(out_dir, shard_index, buffer)
            report.shards_out += 1

    except Exception as exc:  # noqa: BLE001 — top-level orchestration
        report.error = f"{type(exc).__name__}: {exc}"

    report.finished_at = _now_iso()
    report.elapsed_s = round(time.time() - t_start, 3)

    if not dry_run:
        manifest_path = out_dir / "_manifest.json"
        manifest_path.write_text(report.to_json(), encoding="utf-8")

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MinHash LSH dedup on Phase E corpus shards (E1.4).",
    )
    parser.add_argument(
        "--source",
        default="all",
        help="Source directory name filter (substring match) or 'all'. "
        "Default: all subdirs under --input-root.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Input parquet root. Default: {DEFAULT_INPUT_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output parquet root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Cap input docs (smoke / dry-run). Default: no cap.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Jaccard threshold. Default: {DEFAULT_THRESHOLD}")
    parser.add_argument("--num-perm", type=int, default=DEFAULT_NUM_PERM,
                        help=f"MinHash permutations. Default: {DEFAULT_NUM_PERM}")
    parser.add_argument("--shingle-size", type=int, default=DEFAULT_SHINGLE_SIZE,
                        help=f"Word n-gram shingle size. Default: {DEFAULT_SHINGLE_SIZE}")
    parser.add_argument("--dry-run", action="store_true",
                        help="Do not write output shards; report stats only.")
    parser.add_argument("--progress-every", type=int, default=1000,
                        help="Print progress every N docs. 0 disables. Default: 1000")
    args = parser.parse_args(argv)

    sources = _list_source_dirs(args.input_root, args.source)
    if not sources:
        print(
            f"error: no source dirs found under {args.input_root} "
            f"matching filter {args.source!r}",
            file=sys.stderr,
        )
        return 2

    overall_ok = True
    aggregate = {
        "docs_in": 0, "docs_kept": 0, "docs_dropped": 0,
        "bytes_text_in": 0, "bytes_text_kept": 0,
    }
    for src_dir in sources:
        print(f"=== dedup {src_dir.name} (input={src_dir}) ===")
        report = dedup_source(
            source_dir=src_dir,
            output_root=args.output_root,
            max_docs=args.max_docs,
            threshold=args.threshold,
            num_perm=args.num_perm,
            shingle_size=args.shingle_size,
            dry_run=args.dry_run,
            progress_every=args.progress_every,
        )
        print(report.to_json())
        if report.error:
            overall_ok = False
        for k in aggregate:
            aggregate[k] += getattr(report, k)

    if len(sources) > 1:
        agg_rate = (
            aggregate["docs_dropped"] / aggregate["docs_in"]
            if aggregate["docs_in"]
            else 0.0
        )
        print("=== aggregate ===")
        print(json.dumps({**aggregate, "dedup_rate": round(agg_rate, 6)}, indent=2))

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
