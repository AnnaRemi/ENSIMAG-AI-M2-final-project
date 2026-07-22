#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIX_ROOT="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$FIX_ROOT/.." && pwd)"
AKER_HOST="${AKER_HOST:-remizova@aker.imag.fr}"
AKER_ROOT="${AKER_ROOT:-/home/daisy/remizova/lab_m2_benchmarks}"

AKER_HOST="$AKER_HOST" AKER_ROOT="$AKER_ROOT" bash "$REPO_ROOT/benchmarks/sync_to_aker.sh"
ssh "$AKER_HOST" "mkdir -p '$AKER_ROOT/fix'"
remote_path() { printf "%s:%q" "$AKER_HOST" "$1"; }
rsync -av --delete --exclude outputs/ --exclude logs/ --exclude jobs/ \
  --exclude __pycache__/ --exclude .mplconfig/ \
  "$FIX_ROOT/" "$(remote_path "$AKER_ROOT/fix/")"
ssh "$AKER_HOST" "chmod +x '$AKER_ROOT/fix/benchmarks/_aker_worker.sh'"
echo "Synced fixed implementations and runner to $AKER_HOST:$AKER_ROOT/fix"
