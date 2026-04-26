#!/usr/bin/env python3
"""
Phase E E1.4.3 — Exact-string sha256 dedup on the bilingual corpus.

Per `memory/project_phase_e_e1_4_oom_remediation.md` Option A: drop in for
`dedup_corpus.py` (MinHash LSH) after the LSH variant OOM-killed at 5.96M /
10M FineWeb-2 ID docs (RSS 30 GB on 31 GB box, 2026-04-26 13:51 WIB). The
LSH partial measured an empirical dedup_rate of 0.0093% (552 dupes in
5.96M docs), so MinHash near-dup machinery is overkill at this rate —
exact-string sha256 catches >99% at constant ~1 GB memory.

Core algorithm:
    1. NFKC-normalize text → lowercase → collapse whitespace runs to single
       space → strip → sha256 → 32-byte digest.
    2. Maintain `set[bytes]` of seen digests. First-seen wins.
    3. Memory: ~32 bytes per digest + ~80 bytes Python set overhead ≈ 1.1
       GB at 10M docs. Constant across the entire ID corpus, unlike LSH
       whose index grew linearly with kept docs (~5 KB per doc).

Layout (mirrors dedup_corpus.py to keep pipelines parallel):
    data/phase_e/corpus_id/<source>/shard_NNNNN.parquet         (input)
    data/phase_e/corpus_id_dedup_exact/<source>/shard_NNNNN.parquet (output)
    data/phase_e/corpus_id_dedup_exact/<source>/_manifest.json   (per-source stats)
    data/phase_e/corpus_id_dedup_exact/.progress/<source>/state.pkl (--resume)

Verify (runnable, per CLAUDE §6.8 R2):
    .venv/bin/python python/phase_e/scripts/dedup_corpus_exact.py \\
        --source wikipedia_id --max-docs 10000 --dry-run

Resilience (per CLAUDE §6.11 spirit):
    --resume saves shard-boundary state to <output_root>/.progress/<source>/
    state.pkl. State carries the digest set + report counters + output-shard
    buffer. On the next --resume invocation, completed input shards are
    skipped and the digest set is reloaded; SIGTERM mid-shard loses at most
    one input shard's compute.

Why a separate output dir (`corpus_id_dedup_exact` vs `corpus_id_dedup`):
    The 84-shard partial LSH output stays put for forensics + future
    bilingual rates if needed. Cleanup happens after this exact-hash run
    is verified per the resume protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "phase_e" / "corpus_id"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "phase_e" / "corpus_id_dedup_exact"
SHARD_TARGET_BYTES = 256 * 1024 * 1024  # match build_id_corpus.py + dedup_corpus.py

# Normalization + hash scheme — bumping any of these MUST change the
# fingerprint string below so --resume rejects mismatched state.
NORM_SCHEME = "nfkc-lower-ws"
HASH_ALG = "sha256"
DIGEST_BYTES = 32  # sha256

PROGRESS_DIR_NAME = ".progress"
STATE_FILE_NAME = "state.pkl"

_WS_RE = re.compile(r"\s+", re.UNICODE)


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
    norm_scheme: str = NORM_SCHEME
    hash_alg: str = HASH_ALG
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


def _normalize(text: str) -> str:
    """NFKC → lowercase → collapse whitespace runs → strip. Stable across
    Python builds and not locale-dependent."""
    nfkc = unicodedata.normalize("NFKC", text)
    return _WS_RE.sub(" ", nfkc.lower()).strip()


def _digest(text: str) -> bytes:
    return hashlib.sha256(_normalize(text).encode("utf-8")).digest()


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


def _config_fingerprint() -> str:
    """Stable string identity for the dedup config — guards resume across changes."""
    return f"norm={NORM_SCHEME}|alg={HASH_ALG}"


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
    seen: set[bytes]


def _save_state(
    state_path: Path,
    seen: set[bytes],
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
        seen=seen,
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
    dry_run: bool,
    progress_every: int,
    resume: bool = False,
) -> DedupReport:
    report = DedupReport(
        source=source_dir.name,
        started_at=_now_iso(),
        dry_run=dry_run,
    )
    out_dir = output_root / source_dir.name
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    config_fp = _config_fingerprint()
    state_path = _state_path(output_root, source_dir.name)
    can_checkpoint = resume and not dry_run

    t_start = time.time()
    try:
        seen: set[bytes] = set()
        shards = list(_iter_source_shards(source_dir))
        report.shards_in = len(shards)

        buffer: list[dict] = []
        buffer_bytes = 0
        shard_index = 0
        skip_n = 0

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
                for k, v in loaded.report_dict.items():
                    if k in {"started_at", "finished_at", "elapsed_s"}:
                        continue
                    if hasattr(report, k):
                        setattr(report, k, v)
                seen = loaded.seen
                buffer = list(loaded.buffer)
                buffer_bytes = loaded.buffer_bytes
                shard_index = loaded.shard_index_out
                skip_n = loaded.completed_input_shards
                print(
                    f"  [resume] {state_path.name}: "
                    f"{report.docs_in} docs in / {report.docs_kept} kept, "
                    f"|seen|={len(seen)}, "
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

                d = _digest(text)
                if d in seen:
                    report.docs_dropped += 1
                    if len(report.sample_dropped) < 5:
                        report.sample_dropped.append(
                            {
                                "doc_index": report.docs_in - 1,
                                "digest_hex": d.hex(),
                                "preview": text[:160],
                            }
                        )
                else:
                    seen.add(d)
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
                        f"|seen|={len(seen)}, "
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
                        state_path, seen, report, buffer, buffer_bytes,
                        shard_index, completed_input_shards, config_fp,
                    )
                if max_docs is not None and report.docs_in >= max_docs:
                    break
            else:
                # Partial shard (max_docs hit mid-shard) — do NOT checkpoint.
                # Next --resume re-processes this shard from scratch; in-memory
                # partial-shard digest inserts are discarded.
                break

        if not dry_run and buffer:
            _, _ = _write_shard_atomic(out_dir, shard_index, buffer)
            report.shards_out += 1
            shard_index += 1
            buffer = []
            buffer_bytes = 0
        if can_checkpoint:
            _save_state(
                state_path, seen, report, buffer, buffer_bytes,
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
        description="Exact-string sha256 dedup on Phase E corpus shards (E1.4.3).",
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
    parser.add_argument("--dry-run", action="store_true",
                        help="Do not write output shards; report stats only.")
    parser.add_argument("--resume", action="store_true",
                        help="Save shard-boundary state to <output_root>/.progress/"
                        "<source>/state.pkl and load existing state if present. "
                        "No effect with --dry-run.")
    parser.add_argument("--progress-every", type=int, default=10000,
                        help="Print progress every N docs. 0 disables. Default: 10000")
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
        print(f"=== dedup-exact {src_dir.name} (input={src_dir}) ===")
        report = dedup_source(
            source_dir=src_dir,
            output_root=args.output_root,
            max_docs=args.max_docs,
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
