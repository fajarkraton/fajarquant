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

Resilience (V31.E1.4.1, per CLAUDE §6.11 spirit):
    --resume enables shard-boundary checkpoints written atomically to
    `<output_root>/.progress/<source>/state.pkl`. State carries pickled
    MinHashLSH index + report counters + output-shard buffer. On the next
    `--resume` invocation, completed input shards are skipped and the LSH
    index is reloaded, so a SIGTERM mid-shard loses at most one input
    shard's compute (~10 minutes at full-sweep throughput).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "phase_e" / "corpus_id"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "phase_e" / "corpus_id_dedup"
SHARD_TARGET_BYTES = 256 * 1024 * 1024  # match build_id_corpus.py

# RedPajama / Falcon-Edge recipe defaults.
DEFAULT_NUM_PERM = 128
DEFAULT_THRESHOLD = 0.85
DEFAULT_SHINGLE_SIZE = 5  # word 5-grams

# Resume state layout: <output_root>/.progress/<source_dir_name>/state.pkl
PROGRESS_DIR_NAME = ".progress"
STATE_FILE_NAME = "state.pkl"

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


def _config_fingerprint(threshold: float, num_perm: int, shingle_size: int) -> str:
    """Stable string identity for the dedup config — guards resume across changes."""
    return f"thr={threshold}|np={num_perm}|sh={shingle_size}"


def _state_path(output_root: Path, source_dir_name: str) -> Path:
    return output_root / PROGRESS_DIR_NAME / source_dir_name / STATE_FILE_NAME


@dataclass
class _DedupState:
    """On-disk checkpoint for `--resume`. Pickled atomically at shard boundaries."""

    config_fingerprint: str
    saved_at: str
    completed_input_shards: int
    shard_index_out: int
    buffer: list[dict]
    buffer_bytes: int
    report_dict: dict
    lsh_pickle: bytes


def _save_state(
    state_path: Path,
    lsh,
    report: "DedupReport",
    buffer: list[dict],
    buffer_bytes: int,
    shard_index_out: int,
    completed_input_shards: int,
    config_fingerprint: str,
) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = _DedupState(
        config_fingerprint=config_fingerprint,
        saved_at=_now_iso(),
        completed_input_shards=completed_input_shards,
        shard_index_out=shard_index_out,
        buffer=list(buffer),
        buffer_bytes=buffer_bytes,
        report_dict=dict(report.__dict__),
        lsh_pickle=pickle.dumps(lsh),
    )
    tmp = state_path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, state_path)


def _load_state(state_path: Path) -> _DedupState | None:
    if not state_path.is_file():
        return None
    with open(state_path, "rb") as f:
        return pickle.load(f)


def _shard_row_count(shard_path: Path) -> int:
    """O(1) row count from parquet footer — distinguishes max-docs-mid-shard
    from max-docs-exactly-at-end-of-shard for the checkpoint decision."""
    import pyarrow.parquet as pq

    return pq.ParquetFile(shard_path).metadata.num_rows


def _list_source_dirs(input_root: Path, source_filter: str | None) -> list[Path]:
    if not input_root.is_dir():
        return []
    dirs = sorted(
        [
            p
            for p in input_root.iterdir()
            if p.is_dir() and not p.name.startswith(".")  # skip .progress/
        ]
    )
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
    resume: bool = False,
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

    config_fp = _config_fingerprint(threshold, num_perm, shingle_size)
    state_path = _state_path(output_root, source_dir.name)
    can_checkpoint = resume and not dry_run

    t_start = time.time()
    try:
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        shards = list(_iter_source_shards(source_dir))
        report.shards_in = len(shards)

        buffer: list[dict] = []
        buffer_bytes = 0
        shard_index = 0
        skip_n = 0

        # Resume state load (real-mode only).
        if can_checkpoint:
            loaded = _load_state(state_path)
            if loaded is not None:
                if loaded.config_fingerprint != config_fp:
                    raise RuntimeError(
                        f"resume config mismatch at {state_path}: "
                        f"saved={loaded.config_fingerprint!r} vs "
                        f"current={config_fp!r}. Delete the .progress dir to "
                        f"start fresh, or rerun with the same config."
                    )
                # Restore counters into report (preserve started_at as-is).
                for k, v in loaded.report_dict.items():
                    if k in {"started_at", "finished_at", "elapsed_s"}:
                        continue
                    if hasattr(report, k):
                        setattr(report, k, v)
                lsh = pickle.loads(loaded.lsh_pickle)
                buffer = list(loaded.buffer)
                buffer_bytes = loaded.buffer_bytes
                shard_index = loaded.shard_index_out
                skip_n = loaded.completed_input_shards
                print(
                    f"  [resume] {state_path.name}: "
                    f"{report.docs_in} docs in / {report.docs_kept} kept, "
                    f"skipping {skip_n}/{report.shards_in} input shards",
                    flush=True,
                )

        completed_input_shards = skip_n
        for shard_idx in range(skip_n, len(shards)):
            shard = shards[shard_idx]
            shard_total = _shard_row_count(shard)
            shard_processed = 0

            for text, source_name in _iter_docs_in_shard(shard):
                shard_processed += 1
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
                else:
                    lsh.insert(key, m)
                    report.docs_kept += 1
                    report.bytes_text_kept += len(text.encode("utf-8"))

                    if not dry_run:
                        buffer.append({"text": text, "source": source_name})
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

            shard_complete = shard_processed >= shard_total
            if shard_complete:
                completed_input_shards = shard_idx + 1
                if can_checkpoint:
                    _save_state(
                        state_path, lsh, report, buffer, buffer_bytes,
                        shard_index, completed_input_shards, config_fp,
                    )
                # Shard ended cleanly (possibly because max-docs landed at
                # end-of-shard). Stop the outer loop if the cap is hit too.
                if max_docs is not None and report.docs_in >= max_docs:
                    break
            else:
                # Partial shard (max_docs hit mid-shard) — do NOT checkpoint;
                # next --resume re-processes this shard from scratch, and the
                # in-memory partial-shard inserts to LSH are discarded with
                # the process. Acceptable: at most one shard's worth of work.
                break

        if not dry_run and buffer:
            _, _ = _write_shard_atomic(out_dir, shard_index, buffer)
            report.shards_out += 1
            shard_index += 1
            buffer = []
            buffer_bytes = 0
        # Final state save reflects the trailing-buffer flush.
        if can_checkpoint:
            _save_state(
                state_path, lsh, report, buffer, buffer_bytes,
                shard_index, completed_input_shards, config_fp,
            )

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
    parser.add_argument("--resume", action="store_true",
                        help="Save shard-boundary state to <output_root>/.progress/"
                        "<source>/state.pkl and load existing state if present. "
                        "No effect with --dry-run.")
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
            resume=args.resume,
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
