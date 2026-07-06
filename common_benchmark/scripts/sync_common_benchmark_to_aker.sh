#!/usr/bin/env bash
# Run this script from the local Mac. It uploads only the source and fixed
# common-benchmark data needed by the remote OAR job.

set -Eeuo pipefail

AKER_HOST="${AKER_HOST:-remizova@aker.imag.fr}"
AKER_ROOT="${AKER_ROOT:-/home/daisy/remizova/common_benchmark_workspace}"

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMMON_ROOT="$LAB_ROOT/common_benchmark"
SUQL_ENGINE="$LAB_ROOT/project SUQL/src_baseline"
TRUMMER_ROOT="$LAB_ROOT/project Trummer/heterogen_v1"

for path in \
  "$COMMON_ROOT/benchmark.json" \
  "$COMMON_ROOT/requirements.txt" \
  "$COMMON_ROOT/data/imdb_joined.csv" \
  "$COMMON_ROOT/data/imdb_structured_joined.csv" \
  "$COMMON_ROOT/data/imdb_reviews.csv" \
  "$SUQL_ENGINE/suql_engine.py" \
  "$TRUMMER_ROOT/trummer_join/client.py" \
  "$TRUMMER_ROOT/trummer_join/data.py" \
  "$TRUMMER_ROOT/trummer_join/operators.py"
do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required local file is missing: $path" >&2
    exit 1
  fi
done

echo "Creating remote workspace: $AKER_HOST:$AKER_ROOT"
ssh "$AKER_HOST" \
  "mkdir -p '$AKER_ROOT/common_benchmark/data' \
    '$AKER_ROOT/common_benchmark/scripts' \
    '$AKER_ROOT/common_benchmark/outputs' \
    '$AKER_ROOT/common_benchmark/logs' \
    '$AKER_ROOT/common_benchmark/jobs' \
    '$AKER_ROOT/project_SUQL/src_baseline' \
    '$AKER_ROOT/project_Trummer/heterogen_v1/trummer_join'"

echo "Syncing common benchmark."
rsync -av \
  --exclude outputs/ \
  --exclude logs/ \
  --exclude jobs/ \
  --exclude .mplconfig/ \
  --exclude __pycache__/ \
  "$COMMON_ROOT/" \
  "$AKER_HOST:$AKER_ROOT/common_benchmark/"

echo "Syncing minimal SUQL baseline engine."
rsync -av \
  "$SUQL_ENGINE/suql_engine.py" \
  "$AKER_HOST:$AKER_ROOT/project_SUQL/src_baseline/"

echo "Syncing minimal Trummer implementation."
rsync -av \
  "$TRUMMER_ROOT/trummer_join/__init__.py" \
  "$TRUMMER_ROOT/trummer_join/client.py" \
  "$TRUMMER_ROOT/trummer_join/data.py" \
  "$TRUMMER_ROOT/trummer_join/operators.py" \
  "$AKER_HOST:$AKER_ROOT/project_Trummer/heterogen_v1/trummer_join/"

echo
echo "Sync complete."
echo "Next, connect to the Aker login node:"
echo "  ssh $AKER_HOST"
echo "Then submit one or more models:"
echo "  cd '$AKER_ROOT'"
echo "  MODELS='gemma4:e4b' PULL_MODELS=1 \\"
echo "    bash common_benchmark/scripts/runner.sh submit-aker"
