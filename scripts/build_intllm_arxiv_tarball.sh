#!/usr/bin/env bash
# build_intllm_arxiv_tarball.sh — assemble arXiv submission tarball.
#
# Produces paper/intllm/intllm-arxiv.tar.gz containing:
#   intllm.tex
#   refs.bib
#   figures/*.pdf  (5 figures)
#
# Verifies the tarball compiles standalone by extracting to a tmp
# directory and running pdflatex + bibtex + pdflatex × 2 cycle. If
# the standalone build fails, the tarball is NOT shipped (script
# exits 1 and removes the partial tarball).
#
# Path A Week 4 polish — founder runs this before arxiv.org upload.
# Re-runnable; idempotent. Safe to run on every paper-source edit.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
PAPER_DIR="$REPO_ROOT/paper/intllm"
OUT_TARBALL="$PAPER_DIR/intllm-arxiv.tar.gz"
TMP_VERIFY=$(mktemp -d -t intllm-arxiv-verify-XXXXXX)
trap 'rm -rf "$TMP_VERIFY"' EXIT

cd "$PAPER_DIR"

# ─── Step 1: assemble tarball ──────────────────────────────────────
echo "[1/3] assembling tarball at $OUT_TARBALL"
tar -czf "$OUT_TARBALL" \
    --transform 's,^,intllm-arxiv/,' \
    intllm.tex refs.bib figures/

ls -la "$OUT_TARBALL"

# ─── Step 2: extract to tmp dir for standalone verification ────────
echo "[2/3] extracting to $TMP_VERIFY"
tar -xzf "$OUT_TARBALL" -C "$TMP_VERIFY"
cd "$TMP_VERIFY/intllm-arxiv"

# ─── Step 3: standalone-build cycle ────────────────────────────────
echo "[3/3] standalone build cycle (pdflatex + bibtex + pdflatex × 2)"
pdflatex -interaction=nonstopmode -halt-on-error intllm.tex > /tmp/build1.log 2>&1 \
    || { echo "FAIL: first pdflatex pass; see /tmp/build1.log"; \
         tail -20 /tmp/build1.log; rm -f "$OUT_TARBALL"; exit 1; }
bibtex intllm > /tmp/bibtex.log 2>&1 \
    || { echo "FAIL: bibtex; see /tmp/bibtex.log"; \
         tail -10 /tmp/bibtex.log; rm -f "$OUT_TARBALL"; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error intllm.tex > /tmp/build2.log 2>&1 \
    || { echo "FAIL: second pdflatex pass; see /tmp/build2.log"; \
         tail -20 /tmp/build2.log; rm -f "$OUT_TARBALL"; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error intllm.tex > /tmp/build3.log 2>&1 \
    || { echo "FAIL: third pdflatex pass; see /tmp/build3.log"; \
         tail -20 /tmp/build3.log; rm -f "$OUT_TARBALL"; exit 1; }

# Audit: zero undefined-citation warnings on final pass
if grep -qE "undefined" /tmp/build3.log; then
    echo "FAIL: undefined-reference warnings on final pass:"
    grep -E "undefined" /tmp/build3.log
    rm -f "$OUT_TARBALL"
    exit 1
fi

PAGES=$(pdfinfo intllm.pdf 2>/dev/null | awk '/Pages:/ {print $2}')
SIZE=$(ls -l intllm.pdf | awk '{print $5}')
echo ""
echo "  ── standalone-build verification PASS ──"
echo "  Pages:    $PAGES"
echo "  PDF size: $SIZE bytes"
echo "  Tarball:  $OUT_TARBALL"
ls -la "$OUT_TARBALL"
echo ""
echo "  Ready for arXiv upload. See docs/ARXIV_SUBMISSION.md for next steps."
