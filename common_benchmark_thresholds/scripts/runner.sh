#!/usr/bin/env bash
# Single local entry point for common_benchmark_thresholds.

set -Eeuo pipefail

MODE="${1:-local}"
if [[ "$MODE" == "local" ]]; then
  shift || true
else
  MODE="local"
fi

CHEAP_MODEL="${CHEAP_MODEL:-gemma4:e2b}"
EXPENSIVE_MODEL="${EXPENSIVE_MODEL:-gemma4:e4b}"
METHODS="${METHODS:-v2 v2_3}"
THRESHOLDS="${THRESHOLDS:-0.0,0.5,1.0,1.5,2.0,2.5,3.0}"
REPETITIONS="${REPETITIONS:-3}"
OUTPUT_NAME="${OUTPUT_NAME:-}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
API_BASE="${API_BASE:-http://127.0.0.1:11434}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"
CASCADE_TARGET="${CASCADE_TARGET:-0.9}"
CALIBRATION_BUDGET="${CALIBRATION_BUDGET:-20}"
CHEAP_BATCH_SIZE="${CHEAP_BATCH_SIZE:-8}"
EXPENSIVE_BATCH_SIZE="${EXPENSIVE_BATCH_SIZE:-8}"
V2_3_EXPENSIVE_BATCH_SIZE="${V2_3_EXPENSIVE_BATCH_SIZE:-32}"
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage:
  scripts/runner.sh local [options]

Options:
  --cheap-model MODEL
  --expensive-model MODEL
  --methods "v2 v2_3"
  --thresholds CSV
  --repetitions N
  --output-name NAME
  --outputs-dir DIR
  --api-base URL
  --python PATH
  --request-timeout SEC
  --cascade-target FLOAT
  --calibration-budget N
  --dry-run
USAGE
}

OUTPUTS_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cheap-model) CHEAP_MODEL="$2"; shift 2 ;;
    --expensive-model) EXPENSIVE_MODEL="$2"; shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    --thresholds) THRESHOLDS="$2"; shift 2 ;;
    --repetitions) REPETITIONS="$2"; shift 2 ;;
    --output-name) OUTPUT_NAME="$2"; shift 2 ;;
    --outputs-dir) OUTPUTS_DIR="$2"; shift 2 ;;
    --api-base) API_BASE="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --request-timeout) REQUEST_TIMEOUT="$2"; shift 2 ;;
    --cascade-target) CASCADE_TARGET="$2"; shift 2 ;;
    --calibration-budget) CALIBRATION_BUDGET="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option $1" >&2; usage >&2; exit 1 ;;
  esac
done

slug() { printf "%s" "$1" | sed 's/^ollama\///; s/:latest$//; s#[/:]#_#g'; }
if [[ -z "$OUTPUT_NAME" ]]; then
  OUTPUT_NAME="$(printf '%s_%s_thresholds_%sreps_%s' "$(slug "$CHEAP_MODEL")" "$(slug "$EXPENSIVE_MODEL")" "$REPETITIONS" "$RUN_STAMP")"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -n "$OUTPUTS_DIR" ]] || OUTPUTS_DIR="$SCRIPT_DIR/../outputs/$OUTPUT_NAME"

command=(
  "$PYTHON_BIN" "$SCRIPT_DIR/run_threshold_sweep.py"
  --python "$PYTHON_BIN"
  --api-base "$API_BASE"
  --cheap-model "ollama/${CHEAP_MODEL#ollama/}"
  --expensive-model "ollama/${EXPENSIVE_MODEL#ollama/}"
  --thresholds "$THRESHOLDS"
  --repetitions "$REPETITIONS"
  --request-timeout "$REQUEST_TIMEOUT"
  --calibration-budget "$CALIBRATION_BUDGET"
  --cascade-target "$CASCADE_TARGET"
  --cheap-batch-size "$CHEAP_BATCH_SIZE"
  --expensive-batch-size "$EXPENSIVE_BATCH_SIZE"
  --v2-3-expensive-batch-size "$V2_3_EXPENSIVE_BATCH_SIZE"
  --outputs-dir "$OUTPUTS_DIR"
)

[[ " $METHODS " != *" v2 "* ]] && command+=(--skip-v2)
[[ " $METHODS " != *" v2_3 "* ]] && command+=(--skip-v2-3)
[[ "$DRY_RUN" == "1" ]] && command+=(--dry-run)

exec "${command[@]}"
