#!/usr/bin/env bash
# Run from the local Mac.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/sync_common_benchmark_to_aker.sh"

AKER_ROOT="${AKER_ROOT:-/home/daisy/remizova/common_benchmark_v3_workspace}"
echo "Aker login node:"
echo "  cd '$AKER_ROOT'"
echo "  PULL_MODELS=1 bash common_benchmark_v3/scripts/runner.sh submit-aker"
