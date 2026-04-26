#!/usr/bin/env python3
"""
Phase E E1.4 — post-sweep audit on `make dedup-corpus-id`.

Idempotent helper that reports the state of the background dedup process
and (when finished) summarizes the outputs: manifest stats, disk usage,
post-dedup token estimate, and §E1.1 budget verification.

Read-only by design: NEVER kills the process, NEVER modifies findings docs
or commits anything. Prints a unified diff suggestion for
`docs/FJQ_PHASE_E_E1_FINDINGS.md` for the human to review.

Usage:
    make audit-dedup-sweep
    # or directly:
    .venv/bin/python python/phase_e/scripts/audit_dedup_sweep.py

Exit codes:
    0 — sweep finished cleanly OR sweep still running and healthy
    1 — sweep exited with error / output corrupt / abort threshold tripped
    2 — bad arguments / missing artifacts (PID file, log, etc.)

Pairs with one-shot cron job ee2d4601 (session-only) for opportunistic
auto-audit when the user is in Claude Code at fire time. This script is
the durable fallback that survives session restarts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PID_FILE = Path("/tmp/dedup_corpus_id.pid")
DEFAULT_LOG_GLOB = "/tmp/fajarquant_logs/dedup_corpus_id_*.log"
CORPUS_DEDUP_ROOT = REPO_ROOT / "data" / "phase_e" / "corpus_id_dedup"
FINDINGS_DOC = REPO_ROOT / "docs" / "FJQ_PHASE_E_E1_FINDINGS.md"

# Mistral v3 tokenizer: 2.527 bytes/token on the ID corpus per E1.0 finding.
BYTES_PER_TOKEN = 2.527
# §2 abort threshold from FJQ_PHASE_E_BILINGUAL_KERNEL_PRODUCTION_PLAN.md v1.4:
# < 8 B post-dedup ID tokens → trigger Pivot. ≥ 8 B → continue Lapis 1 path.
ABORT_THRESHOLD_TOKENS = 8e9


def _read_pid() -> int | None:
    if not PID_FILE.is_file():
        return None
    try:
        return int(PID_FILE.read_text().strip())
    except ValueError:
        return None


def _process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _ps_snapshot(pid: int) -> str:
    res = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid,etime,rss,pcpu,cmd", "--no-headers"],
        capture_output=True, text=True, check=False,
    )
    return res.stdout.strip() or "<no ps output>"


def _latest_log() -> Path | None:
    import glob
    matches = sorted(glob.glob(DEFAULT_LOG_GLOB))
    return Path(matches[-1]) if matches else None


def _tail(path: Path, n: int) -> str:
    if not path.is_file():
        return f"<log missing: {path}>"
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return "".join(f.readlines()[-n:])


def _read_manifests() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not CORPUS_DEDUP_ROOT.is_dir():
        return out
    for sub in sorted(CORPUS_DEDUP_ROOT.iterdir()):
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        m = sub / "_manifest.json"
        if m.is_file():
            try:
                out[sub.name] = json.loads(m.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                out[sub.name] = {"error": f"manifest parse: {e}"}
    return out


def _du_human(path: Path) -> str:
    if not path.is_dir():
        return "<missing>"
    res = subprocess.run(
        ["du", "-sh", str(path)], capture_output=True, text=True, check=False,
    )
    return res.stdout.split()[0] if res.stdout else "<du failed>"


def _emit_running_report(pid: int, log_path: Path | None) -> int:
    print(f"[audit] PID {pid}: STILL RUNNING")
    print(f"[audit] {_ps_snapshot(pid)}")
    if log_path is not None:
        print(f"[audit] log: {log_path}")
        print("[audit] last 20 log lines:")
        print(_tail(log_path, 20))
    print("[audit] action: do nothing — sweep is healthy. Re-run audit when finished.")
    return 0


def _propose_findings_diff(manifests: dict[str, dict]) -> str:
    """Produce a human-readable summary block to manually splice into v1.1.
    Does NOT modify the file — caller must review and apply."""
    lines = []
    lines.append("# ─── proposed FJQ_PHASE_E_E1_FINDINGS.md v1.1 splice ───")
    lines.append("")
    lines.append("## E1.4 Post-dedup figures (measured 2026-04-26)")
    lines.append("")
    total_in = total_kept = total_dropped = 0
    total_bytes_kept = 0
    for src, m in manifests.items():
        if "error" in m:
            lines.append(f"- {src}: MANIFEST ERROR — {m['error']}")
            continue
        di = m.get("docs_in", 0)
        dk = m.get("docs_kept", 0)
        dd = m.get("docs_dropped", 0)
        bk = m.get("bytes_text_kept", 0)
        rate = m.get("dedup_rate", 0.0)
        elapsed = m.get("elapsed_s", 0.0)
        ns = m.get("num_shards", 0)
        total_in += di
        total_kept += dk
        total_dropped += dd
        total_bytes_kept += bk
        tok_b = bk / BYTES_PER_TOKEN / 1e9
        lines.append(
            f"- **{src}**: {di:,} in → {dk:,} kept ({dd:,} dup, "
            f"{rate * 100:.3f}%); {bk:,} kept-bytes ≈ {tok_b:.2f} B tokens; "
            f"{ns} output shards; elapsed {elapsed / 3600:.2f} h"
        )
    if total_in:
        agg_rate = total_dropped / total_in
        agg_tok_b = total_bytes_kept / BYTES_PER_TOKEN / 1e9
        lines.append("")
        lines.append(
            f"- **AGGREGATE**: {total_in:,} in → {total_kept:,} kept "
            f"({total_dropped:,} dup, {agg_rate * 100:.3f}%); "
            f"{total_bytes_kept:,} kept-bytes ≈ {agg_tok_b:.2f} B tokens"
        )
        if total_bytes_kept / BYTES_PER_TOKEN >= ABORT_THRESHOLD_TOKENS:
            lines.append(f"- §E1.1 §2 abort threshold (>{ABORT_THRESHOLD_TOKENS / 1e9:.0f} B): **PASS**")
        else:
            lines.append(f"- §E1.1 §2 abort threshold (>{ABORT_THRESHOLD_TOKENS / 1e9:.0f} B): **FAIL — TRIGGER PIVOT**")
    return "\n".join(lines)


def _emit_finished_report(log_path: Path | None) -> int:
    print(f"[audit] PID exited.")
    if log_path is not None:
        print(f"[audit] log: {log_path}")
        print("[audit] last 15 log lines:")
        print(_tail(log_path, 15))

    manifests = _read_manifests()
    if not manifests:
        print(f"[audit] FAIL: no manifests found under {CORPUS_DEDUP_ROOT}")
        return 1

    print(f"\n[audit] manifests found: {list(manifests.keys())}")
    has_error = False
    for src, m in manifests.items():
        if "error" in m:
            print(f"[audit] FAIL: {src} manifest broken — {m['error']}")
            has_error = True
            continue
        print(
            f"[audit]   {src}: docs_in={m.get('docs_in', '?'):,}, "
            f"kept={m.get('docs_kept', '?'):,}, "
            f"dropped={m.get('docs_dropped', '?'):,}, "
            f"rate={m.get('dedup_rate', 0) * 100:.3f}%, "
            f"shards={m.get('num_shards', '?')}, "
            f"err={m.get('error', None)}"
        )

    print("\n[audit] disk usage (du -sh per source):")
    for sub in sorted(CORPUS_DEDUP_ROOT.iterdir()):
        if sub.is_dir() and not sub.name.startswith("."):
            print(f"[audit]   {sub.name}: {_du_human(sub)}")

    print("")
    print(_propose_findings_diff(manifests))
    print("")
    print(f"[audit] findings doc: {FINDINGS_DOC}  ← splice the block above; do not auto-modify.")
    print("[audit] commit nothing without user nod (per `feedback_lanjutkan_rekomendasi.md`).")

    return 1 if has_error else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Phase E dedup sweep state.")
    parser.add_argument(
        "--log", type=Path, default=None,
        help="Path to dedup log. Default: latest matching " + DEFAULT_LOG_GLOB,
    )
    args = parser.parse_args(argv)

    pid = _read_pid()
    log_path = args.log or _latest_log()

    if pid is None:
        print(f"[audit] FAIL: no PID file at {PID_FILE}", file=sys.stderr)
        return 2

    if _process_running(pid):
        return _emit_running_report(pid, log_path)
    return _emit_finished_report(log_path)


if __name__ == "__main__":
    sys.exit(main())
