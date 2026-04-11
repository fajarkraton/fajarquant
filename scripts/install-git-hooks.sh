#!/usr/bin/env bash
# Install git hooks for fajarquant repo.
# Mirror of fajar-lang's pattern (commits 6775e44 + 0fdf477 + audit_unwrap.py).
#
# Installs:
#   - pre-commit  (fmt + clippy + production-unwrap audit)
#   - commit-msg  (V26 Plan Hygiene Rule 6 mechanical gates — currently C3.2)
#
# Run from repo root: bash scripts/install-git-hooks.sh

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK_DIR="$REPO_ROOT/.git/hooks"
HOOK_FILE="$HOOK_DIR/pre-commit"
COMMIT_MSG_FILE="$HOOK_DIR/commit-msg"

mkdir -p "$HOOK_DIR"

cat > "$HOOK_FILE" <<'HOOK'
#!/usr/bin/env bash
# fajarquant pre-commit hook (v2 — uses audit_unwrap.py 3-layer filter)
# Mirror of fajar-lang pattern (commits 6775e44 + 0fdf477 + audit_unwrap.py).
#
# v1 used naive `grep -B 20` which had false positives when the
# #[cfg(test)] marker is more than 20 lines above the .unwrap() call.
# v2 uses scripts/audit_unwrap.py which has a proper 3-layer filter:
#   1. file-level #[cfg(test)] mod foo; declarations
#   2. inline #[cfg(test)] mod tests blocks
#   3. .unwrap() inside doc comments and string literals

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Layer 1: cargo fmt --check
if ! cargo fmt -- --check 2>&1; then
    echo ""
    echo "❌ fmt drift detected. Run 'cargo fmt' and re-stage."
    exit 1
fi

# Layer 2: cargo clippy --lib -- -D warnings
if ! cargo clippy --lib -- -D warnings 2>&1; then
    echo ""
    echo "❌ clippy warnings. Fix before committing."
    exit 1
fi

# Layer 3: production .unwrap() audit via 3-layer filter
if [ -x scripts/audit_unwrap.py ]; then
    UNWRAP_SUMMARY=$(python3 scripts/audit_unwrap.py --summary 2>&1)
    UNWRAP_COUNT=$(echo "$UNWRAP_SUMMARY" | grep "Total production" | awk '{print $5}')
    if [ -n "$UNWRAP_COUNT" ] && [ "$UNWRAP_COUNT" -gt 0 ]; then
        echo "❌ $UNWRAP_COUNT production .unwrap() detected. Use .expect(\"reason\") or ?"
        echo "   See V26 Plan Hygiene Rule 3 (prevention layer)."
        echo "$UNWRAP_SUMMARY"
        exit 1
    fi
fi

echo "✅ pre-commit OK (fmt, clippy, 0 production unwraps)"
HOOK

chmod +x "$HOOK_FILE"

# ─────────────────────────────────────────────────────────────────
# commit-msg hook — V26 Plan Hygiene Rule 6 mechanical gates
# ─────────────────────────────────────────────────────────────────
cat > "$COMMIT_MSG_FILE" <<'COMMITMSG'
#!/usr/bin/env bash
# fajarquant commit-msg hook — V26 Plan Hygiene Rule 6 mechanical gates
#
# Currently enforces:
#   - C3.2 (paper venue decision): block any commit whose message contains
#     the 'v26-c3' scope token if paper/SUBMISSION.md is missing on disk
#     on or after 2026-04-25.
#
# Rationale: Plan Hygiene Rule 6 — decisions must be committed files,
# not prose paragraphs. Without a mechanical gate, the C3.2 deadline
# would be a soft prose deadline that gets skipped under execution
# pressure (V26 Phase A1.4 lesson).

set -e

MSG_FILE="$1"
MSG="$(cat "$MSG_FILE")"
REPO_ROOT="$(git rev-parse --show-toplevel)"

# C3.2 mechanical gate
if printf '%s' "$MSG" | grep -qE 'v26-c3'; then
    DEADLINE="2026-04-25"
    TODAY="$(date +%Y-%m-%d)"
    if [ "$TODAY" \> "$DEADLINE" ] || [ "$TODAY" = "$DEADLINE" ]; then
        if [ ! -f "$REPO_ROOT/paper/SUBMISSION.md" ]; then
            echo "❌ V26 C3.2 mechanical gate: paper/SUBMISSION.md is missing"
            echo "   and the 2026-04-25 deadline has passed."
            echo ""
            echo "   Plan Hygiene Rule 6 requires venue decisions to live in"
            echo "   committed files. Restore paper/SUBMISSION.md or remove"
            echo "   'v26-c3' from this commit message."
            exit 1
        fi
    fi
fi

exit 0
COMMITMSG

chmod +x "$COMMIT_MSG_FILE"

echo "✅ pre-commit hook (v2) installed at $HOOK_FILE"
echo "✅ commit-msg hook (v1, V26 C3.2 gate) installed at $COMMIT_MSG_FILE"
echo "   Test pre-commit with: bash $HOOK_FILE"
echo ""
echo "   pre-commit v2 uses scripts/audit_unwrap.py for accurate 3-layer filtering."
echo "   Make sure that script is committed and executable."
echo ""
echo "   commit-msg v1 enforces V26 Plan Hygiene Rule 6 mechanical gates."
echo "   Currently enforces C3.2 (paper/SUBMISSION.md required after 2026-04-25"
echo "   for any commit message containing 'v26-c3')."
