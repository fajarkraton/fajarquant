#!/usr/bin/env bash
# Install pre-commit hook for fajarquant repo.
# Mirror of fajar-lang's pattern (commits 6775e44 + 0fdf477 + audit_unwrap.py).
#
# Run from repo root: bash scripts/install-git-hooks.sh

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK_DIR="$REPO_ROOT/.git/hooks"
HOOK_FILE="$HOOK_DIR/pre-commit"

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

echo "✅ pre-commit hook (v2) installed at $HOOK_FILE"
echo "   Test it with: bash $HOOK_FILE"
echo ""
echo "   v2 uses scripts/audit_unwrap.py for accurate 3-layer filtering."
echo "   Make sure that script is committed and executable."
