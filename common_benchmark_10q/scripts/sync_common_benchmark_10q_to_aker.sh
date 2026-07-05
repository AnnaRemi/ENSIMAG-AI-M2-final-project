#!/usr/bin/env bash
# Run from the local Mac.

set -Eeuo pipefail

AKER_HOST="${AKER_HOST:-remizova@aker.imag.fr}"
AKER_ROOT="${AKER_ROOT:-/home/daisy/remizova/common_benchmark_10q_workspace}"
LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMMON="$LAB_ROOT/common_benchmark_10q"
SUQL_BASELINE="$LAB_ROOT/project SUQL/src_baseline"
TRUMMER_ROOT="$LAB_ROOT/project Trummer"

for path in \
  "$COMMON/manifest.json" \
  "$COMMON/scripts/run_all.py" \
  "$COMMON/scripts/run_method.py" \
  "$COMMON/scripts/run_aker_common_benchmark_10q.sh" \
  "$SUQL_BASELINE/suql_engine.py" \
  "$TRUMMER_ROOT/heterogen_v2_3/trummer_join/cascade.py" \
  "$TRUMMER_ROOT/heterogen_v3/trummer_join/cascade.py" \
  "$TRUMMER_ROOT/heterogen_v3/trummer_join/structured_filter.py"
do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing $path" >&2
    echo "Run: python3 common_benchmark_10q/scripts/build_datasets.py" >&2
    exit 1
  fi
done

ssh "$AKER_HOST" \
  "mkdir -p '$AKER_ROOT/common_benchmark_10q' \
    '$AKER_ROOT/project SUQL/src_baseline' \
    '$AKER_ROOT/project Trummer/heterogen_v2_2' \
    '$AKER_ROOT/project Trummer/heterogen_v2_3' \
    '$AKER_ROOT/project Trummer/heterogen_v3'"

RSYNC=(rsync -av)
remote_path() {
  printf "%s:%q" "$AKER_HOST" "$1"
}

"${RSYNC[@]}" \
  --exclude outputs/ --exclude logs/ --exclude jobs/ --exclude .mplconfig/ --exclude __pycache__/ \
  "$COMMON/" "$(remote_path "$AKER_ROOT/common_benchmark_10q/")"
"${RSYNC[@]}" \
  --exclude __pycache__/ \
  "$SUQL_BASELINE/" "$(remote_path "$AKER_ROOT/project SUQL/src_baseline/")"
"${RSYNC[@]}" \
  --exclude outputs/ --exclude __pycache__/ \
  "$TRUMMER_ROOT/heterogen_v2_2/" "$(remote_path "$AKER_ROOT/project Trummer/heterogen_v2_2/")"
"${RSYNC[@]}" \
  --exclude outputs/ --exclude __pycache__/ \
  "$TRUMMER_ROOT/heterogen_v2_3/" "$(remote_path "$AKER_ROOT/project Trummer/heterogen_v2_3/")"
"${RSYNC[@]}" \
  --exclude outputs/ --exclude __pycache__/ \
  "$TRUMMER_ROOT/heterogen_v3/" "$(remote_path "$AKER_ROOT/project Trummer/heterogen_v3/")"

ssh "$AKER_HOST" "chmod +x '$AKER_ROOT/common_benchmark_10q/scripts/'*.sh"
ssh "$AKER_HOST" \
  "test -f '$AKER_ROOT/project SUQL/src_baseline/suql_engine.py' \
    && test -f '$AKER_ROOT/project Trummer/heterogen_v2_3/trummer_join/cascade.py' \
    && test -f '$AKER_ROOT/project Trummer/heterogen_v3/trummer_join/cascade.py' \
    && test -f '$AKER_ROOT/project Trummer/heterogen_v3/trummer_join/structured_filter.py'"

echo "Sync complete."
echo "Aker:"
echo "  cd '$AKER_ROOT'"
echo "  PULL_MODELS=1 bash common_benchmark_10q/scripts/submit_aker_common_benchmark_10q.sh"
