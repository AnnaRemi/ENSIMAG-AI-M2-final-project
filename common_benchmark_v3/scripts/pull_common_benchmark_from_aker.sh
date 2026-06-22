#!/usr/bin/env bash
# Run from the local Mac after completion.

set -Eeuo pipefail

AKER_HOST="${AKER_HOST:-remizova@aker.imag.fr}"
AKER_ROOT="${AKER_ROOT:-/home/daisy/remizova/common_benchmark_v3_workspace}"
LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL="$LAB_ROOT/common_benchmark_v3"
mkdir -p "$LOCAL/outputs" "$LOCAL/aker_logs"
rsync -av "$AKER_HOST:$AKER_ROOT/common_benchmark_v3/outputs/" "$LOCAL/outputs/"
rsync -av "$AKER_HOST:$AKER_ROOT/common_benchmark_v3/logs/" "$LOCAL/aker_logs/"
echo "Outputs: $LOCAL/outputs"
echo "Logs: $LOCAL/aker_logs"

