#!/usr/bin/env python3
"""Hugging Face cache hygiene — prevention layer per CLAUDE.md §6.8 R3.

Surfaced by V31.C.P2.0 pre-flight: local disk at 85% full (136 GB
free on /dev/nvme0n1p2). The 17-day Phase D Stretch training run
will accumulate HF dataset snapshots + model checkpoints; this
utility prevents disk exhaustion during that run.

Usage:
    # Dry run — show what would be removed
    ./scripts/cleanup_hf_cache.py

    # Actually remove items older than 14 days
    ./scripts/cleanup_hf_cache.py --delete --age-days 14

    # Pre-training guard — exit 1 if less than THRESHOLD_GB free
    ./scripts/cleanup_hf_cache.py --check-free --threshold-gb 30

The --check-free mode is wired into the Stretch training config
(FJQ_PHASE_D_STRETCH_GATE.md) as a gate that stops training if the
disk drops below 30 GB free.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

DEFAULT_HF_HOME = Path.home() / ".cache" / "huggingface" / "hub"
DEFAULT_THRESHOLD_GB = 30.0


def _bytes_to_gb(b: int) -> float:
    return b / (1024**3)


def _disk_free_gb(path: Path) -> float:
    stat = shutil.disk_usage(path if path.is_dir() else path.parent)
    return _bytes_to_gb(stat.free)


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue
    return total


def _iter_cache_entries(cache_dir: Path):
    """Yield (entry_path, last_access_seconds_ago, size_bytes) for each
    top-level model/dataset dir inside the HF hub cache."""
    if not cache_dir.is_dir():
        return
    now = time.time()
    for entry in cache_dir.iterdir():
        if not entry.is_dir():
            continue
        # Oldest atime across all files; falls back to entry mtime.
        atime = entry.stat().st_atime
        for f in entry.rglob("*"):
            try:
                if f.is_file():
                    atime = max(atime, f.stat().st_atime)
            except OSError:
                continue
        size = _dir_size_bytes(entry)
        yield entry, (now - atime) / 86400.0, size


def cmd_check_free(args: argparse.Namespace) -> int:
    free_gb = _disk_free_gb(args.hf_home)
    threshold = args.threshold_gb
    status = "OK" if free_gb >= threshold else "BELOW THRESHOLD"
    print(f"disk free: {free_gb:.1f} GB, threshold: {threshold:.1f} GB — {status}")
    return 0 if free_gb >= threshold else 1


def cmd_list_or_delete(args: argparse.Namespace) -> int:
    cache_dir = args.hf_home
    if not cache_dir.is_dir():
        print(f"no cache found at {cache_dir}")
        return 0

    entries = sorted(_iter_cache_entries(cache_dir), key=lambda e: -e[2])
    total_bytes = sum(e[2] for e in entries)
    print(f"cache root: {cache_dir}")
    print(f"total size: {_bytes_to_gb(total_bytes):.2f} GB across {len(entries)} entries\n")
    print(f"{'age (d)':>8}  {'size (GB)':>10}  entry")
    print(f"{'-' * 8}  {'-' * 10}  {'-' * 40}")

    freed = 0
    to_delete = []
    for entry, age_days, size in entries:
        marker = " [DELETE]" if args.delete and age_days > args.age_days else ""
        print(f"{age_days:>8.1f}  {_bytes_to_gb(size):>10.2f}  {entry.name}{marker}")
        if args.delete and age_days > args.age_days:
            to_delete.append(entry)
            freed += size

    if args.delete and to_delete:
        print(f"\nDeleting {len(to_delete)} entries ({_bytes_to_gb(freed):.2f} GB) ...")
        for entry in to_delete:
            shutil.rmtree(entry, ignore_errors=True)
        free_after_gb = _disk_free_gb(args.hf_home)
        print(f"done. disk free: {free_after_gb:.1f} GB")
    elif args.delete:
        print(f"\nNo entries older than {args.age_days} days — nothing to delete.")
    else:
        print(f"\n(dry run — re-run with --delete --age-days {args.age_days} to remove)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--hf-home",
        type=Path,
        default=DEFAULT_HF_HOME,
        help=f"HF hub cache root (default: {DEFAULT_HF_HOME})",
    )
    p.add_argument("--delete", action="store_true", help="actually remove entries")
    p.add_argument(
        "--age-days",
        type=float,
        default=14,
        help="delete entries whose newest file is older than N days (default: 14)",
    )
    p.add_argument(
        "--check-free",
        action="store_true",
        help="exit 1 if disk-free drops below --threshold-gb",
    )
    p.add_argument(
        "--threshold-gb",
        type=float,
        default=DEFAULT_THRESHOLD_GB,
        help=f"free-space threshold in GB (default: {DEFAULT_THRESHOLD_GB})",
    )
    args = p.parse_args()

    if args.check_free:
        return cmd_check_free(args)
    return cmd_list_or_delete(args)


if __name__ == "__main__":
    sys.exit(main())
