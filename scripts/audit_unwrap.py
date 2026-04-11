#!/usr/bin/env python3
"""
audit_unwrap.py — V26 Phase A2.1

Inventory every production .unwrap() in src/, splitting at the
#[cfg(test)] mod tests boundary. Outputs CSV with file:line:snippet
for downstream categorization.

Usage:
    python3 scripts/audit_unwrap.py > audit/unwrap_inventory.csv
    python3 scripts/audit_unwrap.py --summary
    python3 scripts/audit_unwrap.py --by-file

Why this exists: V17 audit found "43 production unwraps". V26 audit
revealed the real count is 174 — V17 used a different methodology that
missed many. CLAUDE.md §6.3 says "NEVER use .unwrap() in src/" but this
rule has drifted. A2.1 builds the inventory; A2.2 categorizes; A2.3-A2.4
replaces or justifies each.
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import List, NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

# Match an inline test module: #[cfg(test)]\n mod tests {
INLINE_TEST_MOD_RE = re.compile(
    r"#\[cfg\(test\)\]\s*\n\s*(?:pub\s+)?mod\s+\w+\s*\{",
    re.MULTILINE,
)

# Match a file-level test module declaration in a parent mod file:
#   #[cfg(test)]\n mod <name>;
# (semicolon, not opening brace — this declares an external file)
FILE_LEVEL_TEST_MOD_RE = re.compile(
    r"#\[cfg\(test\)\]\s*\n\s*(?:pub\s+)?mod\s+(\w+)\s*;",
    re.MULTILINE,
)

# Match .unwrap() but skip .unwrap_or(, .unwrap_or_else(, .unwrap_err()
UNWRAP_RE = re.compile(r"\.unwrap\(\)")

# Detect lines that are doc comments or regular comments — these contain
# `.unwrap()` only as documentation, not as code.
COMMENT_RE = re.compile(r"^\s*(?://|///|//!|/\*|\*)")


def is_test_only_file(path: Path) -> bool:
    """Return True if `path` is declared as #[cfg(test)] in its parent mod file.

    For src/foo/bar.rs the parent mod file is src/foo/mod.rs (or src/foo.rs).
    For src/foo.rs the parent mod file is src/lib.rs (or src/main.rs).
    """
    parent_dir = path.parent
    basename = path.stem  # without .rs

    # Candidate parent mod files in priority order
    candidates = [
        parent_dir / "mod.rs",
        parent_dir.parent / f"{parent_dir.name}.rs",  # sibling .rs file
    ]
    # If we're at src/, the parent is lib.rs / main.rs
    if parent_dir == SRC_DIR:
        candidates = [SRC_DIR / "lib.rs", SRC_DIR / "main.rs"]

    for parent in candidates:
        if not parent.exists() or parent == path:
            continue
        try:
            content = parent.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        for m in FILE_LEVEL_TEST_MOD_RE.finditer(content):
            if m.group(1) == basename:
                return True
    return False


class UnwrapHit(NamedTuple):
    file: str  # path relative to src/
    line: int  # 1-based line number
    snippet: str  # the line itself, trimmed
    in_function: str  # nearest enclosing fn or impl block (best-effort)


def find_test_module_start(content: str) -> int:
    """Return byte offset of the first inline #[cfg(test)] mod tests block,
    or len(content) if not found."""
    m = INLINE_TEST_MOD_RE.search(content)
    if m:
        return m.start()
    return len(content)


def find_enclosing_fn(lines: List[str], lineno: int) -> str:
    """Walk backwards from lineno to find the nearest fn/impl/struct."""
    fn_re = re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)")
    impl_re = re.compile(r"^\s*impl[\s<]")
    for i in range(lineno - 1, max(-1, lineno - 200), -1):
        line = lines[i]
        m = fn_re.match(line)
        if m:
            return f"fn {m.group(1)}"
        if impl_re.match(line):
            # Try to extract a meaningful identifier from impl line
            m2 = re.search(r"impl(?:<[^>]*>)?\s+(?:[\w<>:'\s,]+\s+for\s+)?(\w+)", line)
            return f"impl {m2.group(1) if m2 else '?'}"
    return "<top-level>"


def scan_file(path: Path) -> List[UnwrapHit]:
    """Return all production .unwrap() hits in a file.

    Excludes:
      - Files declared as #[cfg(test)] mod <name>; in a parent mod file
      - Code inside #[cfg(test)] mod tests { ... } blocks within a file
    """
    # If the entire file is test-gated by its parent module, skip it.
    if is_test_only_file(path):
        return []

    try:
        content = path.read_text()
    except (OSError, UnicodeDecodeError):
        return []

    test_start = find_test_module_start(content)
    prod_content = content[:test_start]
    lines = prod_content.splitlines()

    hits = []
    rel = str(path.relative_to(SRC_DIR))
    for lineno_zero, line in enumerate(lines):
        if not UNWRAP_RE.search(line):
            continue
        # Filter false positives:
        # 1. Doc comments and regular comments (//, ///, //!, /*, *)
        if COMMENT_RE.match(line):
            continue
        # 2. .unwrap() inside a string literal — find each match and check
        #    whether it sits between an unescaped pair of double quotes.
        if all(_in_string_literal(line, m.start()) for m in UNWRAP_RE.finditer(line)):
            continue

        lineno = lineno_zero + 1
        snippet = line.strip()
        fn = find_enclosing_fn(lines, lineno)
        hits.append(UnwrapHit(rel, lineno, snippet, fn))
    return hits


def _in_string_literal(line: str, pos: int) -> bool:
    """Best-effort: is `pos` inside a "..." string literal on this line?

    Counts unescaped double quotes before `pos`. Odd count → inside string.
    Doesn't handle raw strings (r"..."), multi-line strings, or escaped
    quotes inside char literals — but good enough for the typical cases
    in this codebase (rule definitions and macro patterns).
    """
    quotes = 0
    i = 0
    while i < pos:
        c = line[i]
        if c == "\\":
            i += 2  # skip escaped char
            continue
        if c == '"':
            quotes += 1
        i += 1
    return quotes % 2 == 1


def collect_all() -> List[UnwrapHit]:
    """Walk src/ and return all production .unwrap() hits."""
    hits = []
    for root, _, files in os.walk(SRC_DIR):
        for fn in files:
            if fn.endswith(".rs"):
                hits.extend(scan_file(Path(root) / fn))
    return hits


def emit_csv(hits: List[UnwrapHit], stream) -> None:
    w = csv.writer(stream)
    w.writerow(["file", "line", "function", "snippet"])
    for h in hits:
        w.writerow([h.file, h.line, h.in_function, h.snippet])


def emit_summary(hits: List[UnwrapHit]) -> None:
    by_file = {}
    for h in hits:
        by_file.setdefault(h.file, 0)
        by_file[h.file] += 1
    print(f"Total production .unwrap() hits: {len(hits)}")
    print(f"Files containing production unwraps: {len(by_file)}")
    print()
    print("Top 20 files:")
    for f, n in sorted(by_file.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {n:>4}  {f}")


def emit_by_file(hits: List[UnwrapHit]) -> None:
    by_file = {}
    for h in hits:
        by_file.setdefault(h.file, []).append(h)
    for f in sorted(by_file.keys()):
        print(f"\n=== {f} ({len(by_file[f])}) ===")
        for h in by_file[f]:
            print(f"  L{h.line:>4}  [{h.in_function}]  {h.snippet[:100]}")


def main():
    p = argparse.ArgumentParser(description="V26 A2.1 — production .unwrap() inventory")
    p.add_argument("--summary", action="store_true", help="print summary by file")
    p.add_argument("--by-file", action="store_true", help="print all hits grouped by file")
    args = p.parse_args()

    hits = collect_all()

    if args.summary:
        emit_summary(hits)
    elif args.by_file:
        emit_by_file(hits)
    else:
        emit_csv(hits, sys.stdout)


if __name__ == "__main__":
    main()
