#!/usr/bin/env bash
# Run from the local Mac after the OAR job completes.

set -Eeuo pipefail

AKER_HOST="${AKER_HOST:-remizova@aker.imag.fr}"
AKER_ROOT="${AKER_ROOT:-/home/daisy/remizova/common_benchmark_v3_workspace}"
LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL="$LAB_ROOT/common_benchmark_v3"
OUTPUT_NAME="${OUTPUT_NAME:-}"

mkdir -p "$LOCAL/outputs" "$LOCAL/aker_logs"
if [[ -n "$OUTPUT_NAME" ]]; then
  mkdir -p "$LOCAL/outputs/$OUTPUT_NAME"
  rsync -av \
    "$AKER_HOST:$AKER_ROOT/common_benchmark_v3/outputs/$OUTPUT_NAME/" \
    "$LOCAL/outputs/$OUTPUT_NAME/"
else
  rsync -av \
    "$AKER_HOST:$AKER_ROOT/common_benchmark_v3/outputs/all_heterogen_*/" \
    "$LOCAL/outputs/"
fi
rsync -av \
  "$AKER_HOST:$AKER_ROOT/common_benchmark_v3/logs/all_heterogen_*" \
  "$LOCAL/aker_logs/" || true

echo "Outputs: $LOCAL/outputs"
echo "Logs: $LOCAL/aker_logs"
