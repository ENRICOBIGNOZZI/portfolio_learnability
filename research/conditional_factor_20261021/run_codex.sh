#!/usr/bin/env bash
# Run on an authorized machine with an existing authenticated Codex installation.
# This script does not configure Tailscale/SSH or bypass Codex sandbox controls.
set -euo pipefail
REPO_INPUT="${1:-$PWD}"
SOURCE_BRANCH="codex/conditional-factor-audit-20261021"
PACKAGE="research/conditional_factor_20261021"
for cmd in git codex python3; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "Required command unavailable: $cmd" >&2; exit 2; }
done
ROOT="$(git -C "$REPO_INPUT" rev-parse --show-toplevel)"
ORIGIN="$(git -C "$ROOT" remote get-url origin)"
case "$ORIGIN" in
  https://github.com/ENRICOBIGNOZZI/portfolio_learnability|https://github.com/ENRICOBIGNOZZI/portfolio_learnability.git|git@github.com:ENRICOBIGNOZZI/portfolio_learnability.git) ;;
  *) echo 'Wrong repository; refusing to run outside portfolio_learnability.' >&2; exit 2 ;;
esac
codex --version
if ! codex login status >/dev/null 2>&1; then
  echo 'Codex authentication is unavailable. Authenticate locally; do not copy credentials into this package.' >&2
  exit 2
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
BRANCH="codex/conditional-factor-run-$STAMP"
WORKTREE="$(dirname "$ROOT")/portfolio-learnability-factor-$STAMP"
# Fetches a known branch, without pull, reset, clean, or changing the active branch.
git -C "$ROOT" fetch --no-tags origin "$SOURCE_BRANCH"
BASE="$(git -C "$ROOT" rev-parse FETCH_HEAD)"
git -C "$ROOT" worktree add -b "$BRANCH" "$WORKTREE" "$BASE"
PROMPT="$WORKTREE/$PACKAGE/CODEX_PROMPT.md"
AUDIT="$WORKTREE/$PACKAGE/AUDIT.md"
test -f "$PROMPT" && test -f "$AUDIT"
RECEIPTS="$WORKTREE/$PACKAGE/execution/$STAMP"
mkdir -p "$RECEIPTS"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 - "$RECEIPTS/launch.json" "$WORKTREE" "$BRANCH" "$BASE" <<'PY'
import datetime, json, pathlib, sys
p, root, branch, sha = sys.argv[1:]
pathlib.Path(p).write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'worktree':root,'branch':branch,'base_commit':sha,
    'status':'prepared; execution completion requires exit.json'},indent=2)+'\n')
PY
cd "$WORKTREE"
printf 'Worktree: %s\nBranch: %s\nLogs: %s\n' "$WORKTREE" "$BRANCH" "$RECEIPTS"
set +e
codex exec --sandbox workspace-write -o "$RECEIPTS/final.md" - \
  < "$PROMPT" > "$RECEIPTS/stdout.log" 2> "$RECEIPTS/stderr.log"
CODE=$?
set -e
python3 - "$RECEIPTS/exit.json" "$CODE" <<'PY'
import datetime, json, pathlib, sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'codex_exit_code':int(sys.argv[2]),'scope':'CLI process exit, not scientific acceptance'},indent=2)+'\n')
PY
if test -f "$RECEIPTS/final.md"; then cat "$RECEIPTS/final.md"; fi
printf '\nCodex exit code: %s\nReceipts: %s\n' "$CODE" "$RECEIPTS"
exit "$CODE"
