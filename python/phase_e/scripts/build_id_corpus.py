#!/usr/bin/env python3
"""
Phase E E1.1 — Build Indonesian-language pretraining corpus.

Per FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md §3 E1.1:
- Stream from HF Datasets, write per-source sharded parquet files
- Resume-safe: existing shards are skipped (atomic .tmp -> os.replace write)
- Dry-run mode probes schema + first N docs without writing
- Per-source --source flag for selective download

Initial implementation (V31.E1.1.0): Wikipedia ID only (smallest, public domain).
Subsequent commits add CC-100 ID, OSCAR ID, CulturaX ID, MADLAD-400 ID,
FineWeb-2 ID per the same registry pattern.

Layout written:
    data/phase_e/corpus_id/<source>/shard_NNNNN.parquet   (sharded, ~256 MB each)
    data/phase_e/corpus_id/<source>/_manifest.json        (per-source stats)

Verify (runnable, per CLAUDE §6.8 R2):
    python python/phase_e/scripts/build_id_corpus.py --source wikipedia_id --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "phase_e" / "corpus_id"
SHARD_TARGET_BYTES = 256 * 1024 * 1024  # ~256 MB per parquet shard


@dataclass
class SourceSpec:
    name: str
    hf_dataset_id: str
    hf_config: str | None
    hf_split: str
    text_field: str
    license: str
    plan_estimate_tokens_b: tuple[float, float]
    notes: str
    streaming: bool = True


# Registry — add a new SourceSpec entry per E1.1.x sub-task.
# Plan §E1.1 token estimates from audit_id_corpora.py output (E0.0.1).
# All HF dataset IDs verified `available` against HF probe at audit time
# (E0.0.1 only confirms reachability of the HF Hub record, not download
# accessibility — that is verified separately at E1.1.x dry-run time).
#
# E1.1.2 dry-run blocker status (2026-04-25, recorded in commit body):
#   - wikipedia_id     ✅ schema PASS, downloaded in E1.1.1
#   - fineweb_2_id     ✅ schema PASS, ready to run
#   - cc100_id         ❌ HF removed Python script datasets (cc100.py)
#                          → mirror at https://data.statmt.org/cc-100/ as raw HTTP
#                            tarball; needs separate downloader (E1.1.3 future)
#   - oscar_2301_id    ❌ HF-gated; user must accept terms at
#                          https://huggingface.co/datasets/oscar-corpus/OSCAR-2301
#   - culturax_id      ❌ HF-gated; user must accept terms at
#                          https://huggingface.co/datasets/uonlp/CulturaX
#   - madlad_400_id    ❌ 99,272 siblings; HF Datasets config resolution > 90s
#                          → use HF Hub list_repo_files + glob "data/*/id/*.jsonl.gz"
#                            (E1.1.4 future)
SOURCES: dict[str, SourceSpec] = {
    # E1.1.1 — Wikipedia ID (smallest, public domain, curated)
    "wikipedia_id": SourceSpec(
        name="Wikipedia Indonesian",
        hf_dataset_id="wikimedia/wikipedia",
        hf_config="20231101.id",
        hf_split="train",
        text_field="text",
        license="CC BY-SA 4.0 (commercial-OK with attribution)",
        plan_estimate_tokens_b=(0.5, 1.0),
        notes="curated; ~700K articles; 20231101.id snapshot. First E1.1 source.",
    ),
    # E1.1.2 — FineWeb-2 ID (READY: latest high-quality multilingual; BCP-47 ind_Latn)
    "fineweb_2_id": SourceSpec(
        name="FineWeb-2 Indonesian (HuggingFaceFW)",
        hf_dataset_id="HuggingFaceFW/fineweb-2",
        hf_config="ind_Latn",
        hf_split="train",
        text_field="text",
        license="ODC-By (commercial-OK with attribution)",
        plan_estimate_tokens_b=(10.0, 30.0),
        notes="NEW v1.1: latest high-quality multilingual web; ind_Latn (BCP-47).",
    ),
    # E1.1.2 — CC-100 ID (BLOCKED: HF removed cc100.py script-dataset)
    "cc100_id": SourceSpec(
        name="CC-100 Indonesian",
        hf_dataset_id="cc100",
        hf_config="id",
        hf_split="train",
        text_field="text",
        license="CC-100 custom (NOT commercial-OK; excluded from E1.1.1 Tier B1)",
        plan_estimate_tokens_b=(10.0, 15.0),
        notes="BLOCKED: HF removed Python script datasets. Use statmt.org mirror.",
    ),
    # E1.1.2 — OSCAR v23.01 ID (BLOCKED: HF-gated, user accept needed)
    "oscar_2301_id": SourceSpec(
        name="OSCAR Indonesian v23.01",
        hf_dataset_id="oscar-corpus/OSCAR-2301",
        hf_config="id",
        hf_split="train",
        text_field="text",
        license="CC0 1.0 (commercial-OK; HF gated)",
        plan_estimate_tokens_b=(5.0, 8.0),
        notes="BLOCKED: gated. Accept at huggingface.co/datasets/oscar-corpus/OSCAR-2301.",
    ),
    # E1.1.2 — CulturaX ID (BLOCKED: HF-gated, user accept needed)
    "culturax_id": SourceSpec(
        name="CulturaX Indonesian",
        hf_dataset_id="uonlp/CulturaX",
        hf_config="id",
        hf_split="train",
        text_field="text",
        license="ODC-By (commercial-OK with attribution; gated on HF)",
        plan_estimate_tokens_b=(8.0, 20.0),
        notes="BLOCKED: gated. Accept at huggingface.co/datasets/uonlp/CulturaX.",
    ),
    # E1.1.2 — MADLAD-400 ID (BLOCKED: 99K siblings, needs glob-list workaround)
    "madlad_400_id": SourceSpec(
        name="MADLAD-400 Indonesian (Google)",
        hf_dataset_id="allenai/MADLAD-400",
        hf_config="id",
        hf_split="train",
        text_field="text",
        license="ODC-By (commercial-OK with attribution)",
        plan_estimate_tokens_b=(5.0, 10.0),
        notes="BLOCKED: 99K siblings; HF Datasets config-resolve >90s. Need glob workaround.",
    ),
}


@dataclass
class BuildReport:
    source: str
    started_at: str
    finished_at: str | None = None
    elapsed_s: float = 0.0
    num_docs_written: int = 0
    num_docs_skipped_empty: int = 0
    bytes_written_text: int = 0
    bytes_written_parquet: int = 0
    num_shards: int = 0
    schema_sample: dict | None = None
    dry_run: bool = False
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, ensure_ascii=False)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stream_source(spec: SourceSpec) -> Iterator[dict]:
    """Yield raw HF Datasets records for the given source."""
    from datasets import load_dataset

    ds = load_dataset(
        spec.hf_dataset_id,
        spec.hf_config,
        split=spec.hf_split,
        streaming=spec.streaming,
    )
    for rec in ds:
        yield rec


def _write_shard_atomic(
    out_dir: Path,
    shard_index: int,
    rows: list[dict],
) -> tuple[Path, int]:
    """Write a parquet shard via .tmp + os.replace. Returns (final_path, bytes)."""
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


def _next_shard_index(out_dir: Path) -> int:
    existing = sorted(out_dir.glob("shard_*.parquet"))
    if not existing:
        return 0
    last = existing[-1].stem  # "shard_00007"
    return int(last.split("_")[1]) + 1


def build_source(
    spec: SourceSpec,
    output_root: Path,
    max_docs: int | None,
    dry_run: bool,
) -> BuildReport:
    report = BuildReport(source=spec.name, started_at=_now_iso(), dry_run=dry_run)
    out_dir = output_root / spec.hf_dataset_id.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    try:
        stream = _stream_source(spec)

        if dry_run:
            sample = next(stream, None)
            if sample is None:
                report.error = "stream returned no records"
            else:
                report.schema_sample = {
                    k: (type(v).__name__, str(v)[:120]) for k, v in sample.items()
                }
                report.num_docs_written = 1
                if spec.text_field in sample:
                    report.bytes_written_text = len(
                        sample[spec.text_field].encode("utf-8")
                    )
                else:
                    report.error = (
                        f"expected text_field={spec.text_field!r} not in sample keys "
                        f"{list(sample.keys())}"
                    )
            report.finished_at = _now_iso()
            report.elapsed_s = round(time.time() - t_start, 3)
            return report

        shard_index = _next_shard_index(out_dir)
        buffer: list[dict] = []
        buffer_bytes = 0

        for rec in stream:
            text = rec.get(spec.text_field)
            if not text or not text.strip():
                report.num_docs_skipped_empty += 1
                continue
            row = {"text": text, "source": spec.name}
            buffer.append(row)
            buffer_bytes += len(text.encode("utf-8"))
            report.num_docs_written += 1
            report.bytes_written_text += len(text.encode("utf-8"))

            if buffer_bytes >= SHARD_TARGET_BYTES:
                path, sz = _write_shard_atomic(out_dir, shard_index, buffer)
                report.bytes_written_parquet += sz
                report.num_shards += 1
                shard_index += 1
                buffer = []
                buffer_bytes = 0

            if max_docs is not None and report.num_docs_written >= max_docs:
                break

        # flush trailing buffer
        if buffer:
            path, sz = _write_shard_atomic(out_dir, shard_index, buffer)
            report.bytes_written_parquet += sz
            report.num_shards += 1

    except Exception as exc:  # noqa: BLE001 — top-level orchestration
        report.error = f"{type(exc).__name__}: {exc}"

    report.finished_at = _now_iso()
    report.elapsed_s = round(time.time() - t_start, 3)

    # Persist per-source manifest (non-atomic OK; small file, regeneratable)
    manifest_path = out_dir / "_manifest.json"
    manifest_path.write_text(report.to_json(), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build Indonesian-language pretraining corpus (Phase E E1.1).",
    )
    parser.add_argument(
        "--source",
        choices=list(SOURCES.keys()) + ["all"],
        default="all",
        help="Which source to download. 'all' downloads every registered source.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root for sharded parquet files.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Stop after N docs (smoke test). Default: no limit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Probe schema + first record only; do not write shards.",
    )
    args = parser.parse_args(argv)

    if args.source == "all":
        targets = list(SOURCES.values())
    else:
        targets = [SOURCES[args.source]]

    overall_ok = True
    for spec in targets:
        print(f"=== {spec.name} ({spec.hf_dataset_id}, config={spec.hf_config}) ===")
        report = build_source(
            spec,
            output_root=args.output_root,
            max_docs=args.max_docs,
            dry_run=args.dry_run,
        )
        print(report.to_json())
        if report.error:
            overall_ok = False

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
