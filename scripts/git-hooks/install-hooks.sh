#!/usr/bin/env bash
# fajarquant — install git hooks from `scripts/git-hooks/` into `.git/hooks/`.
#
# Idempotent: backs up any existing `.git/hooks/<hook>` to
# `.git/hooks/<hook>.bak.<timestamp>` before overwriting, then symlinks
# the tracked source-of-truth into place.
#
# Usage:
#   ./scripts/git-hooks/install-hooks.sh
#
# Hooks installed:
#   - pre-commit  (V4 — fmt + clippy + audit_unwrap + Phase E1.6 corpus gate + Phase F.13 dispatch gate)

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

HOOKS_SRC="scripts/git-hooks"
HOOKS_DST=".git/hooks"
TS="$(date +%Y%m%d-%H%M%S)"

if [ ! -d "$HOOKS_SRC" ]; then
    echo "error: $HOOKS_SRC not found. Run from fajarquant repo root."
    exit 2
fi

mkdir -p "$HOOKS_DST"

for src in "$HOOKS_SRC"/*; do
    name="$(basename "$src")"
    case "$name" in
        install-hooks.sh|*.md|*.bak.*)
            continue  # skip the installer itself + docs + backups
            ;;
    esac

    dst="$HOOKS_DST/$name"

    # Back up existing hook if it differs from the tracked source.
    if [ -e "$dst" ]; then
        if ! cmp -s "$src" "$dst"; then
            cp -p "$dst" "$dst.bak.$TS"
            echo "  backed up: $dst → $dst.bak.$TS"
        fi
    fi

    # Copy (not symlink — git hooks must be regular files for portability
    # across worktrees + cross-platform compat).
    cp -p "$src" "$dst"
    chmod +x "$dst"
    echo "  installed: $src → $dst"
done

echo ""
echo "✅ git hooks installed from $HOOKS_SRC/"
echo "   re-run after pulling new hook versions."
