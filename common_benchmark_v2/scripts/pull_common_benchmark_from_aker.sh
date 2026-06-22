#!/usr/bin/env bash
# Run this script from the local Mac after the OAR job finishes.

set -Eeuo pipefail

AKER_HOST="${AKER_HOST:-remizova@aker.imag.fr}"
AKER_ROOT="${AKER_ROOT:-/home/daisy/remizova/common_benchmark_v2_workspace}"

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_COMMON="$LAB_ROOT/common_benchmark_v2"

mkdir -p "$LOCAL_COMMON/outputs" "$LOCAL_COMMON/aker_logs"

echo "Pulling model experiment directories."
rsync -av \
  "$AKER_HOST:$AKER_ROOT/common_benchmark_v2/outputs/" \
  "$LOCAL_COMMON/outputs/"

echo "Pulling OAR, Ollama, and console logs."
rsync -av \
  "$AKER_HOST:$AKER_ROOT/common_benchmark_v2/logs/" \
  "$LOCAL_COMMON/aker_logs/"

echo "Results are now under: $LOCAL_COMMON/outputs"
echo "Remote logs are under: $LOCAL_COMMON/aker_logs"
