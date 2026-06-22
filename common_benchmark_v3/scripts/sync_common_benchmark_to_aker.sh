#!/usr/bin/env bash
# Run from the local Mac.

set -Eeuo pipefail

AKER_HOST="${AKER_HOST:-remizova@aker.imag.fr}"
AKER_ROOT="${AKER_ROOT:-/home/daisy/remizova/common_benchmark_v3_workspace}"
LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMMON="$LAB_ROOT/common_benchmark_v3"
V1="$LAB_ROOT/project Trummer/heterogen_v1"
V2="$LAB_ROOT/project Trummer/heterogen_v2"

for path in \
  "$COMMON/benchmark.json" \
  "$COMMON/data/imdb_structured_joined.csv" \
  "$COMMON/data/imdb_reviews.csv" \
  "$V1/trummer_join/operators.py" \
  "$V2/trummer_join/cascade.py"
do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing $path" >&2
    exit 1
  fi
done

ssh "$AKER_HOST" \
  "mkdir -p '$AKER_ROOT/common_benchmark_v3' \
    '$AKER_ROOT/project_Trummer/heterogen_v1/trummer_join' \
    '$AKER_ROOT/project_Trummer/heterogen_v2/trummer_join'"

rsync -av \
  --exclude outputs/ --exclude logs/ --exclude jobs/ --exclude .mplconfig/ --exclude __pycache__/ \
  "$COMMON/" "$AKER_HOST:$AKER_ROOT/common_benchmark_v3/"
rsync -av \
  "$V1/trummer_join/__init__.py" "$V1/trummer_join/client.py" "$V1/trummer_join/operators.py" \
  "$AKER_HOST:$AKER_ROOT/project_Trummer/heterogen_v1/trummer_join/"
rsync -av \
  "$V2/trummer_join/__init__.py" "$V2/trummer_join/cascade.py" \
  "$AKER_HOST:$AKER_ROOT/project_Trummer/heterogen_v2/trummer_join/"

echo "Sync complete."
echo "Aker: cd '$AKER_ROOT' && bash common_benchmark_v3/scripts/submit_aker_common_benchmark.sh"

