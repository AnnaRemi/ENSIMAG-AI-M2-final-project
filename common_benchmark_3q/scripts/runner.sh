#!/usr/bin/env bash
# Single local entry point for common_benchmark_3q.

set -Eeuo pipefail

MODE="${1:-local}"
if [[ "$MODE" == "local" ]]; then
  shift || true
else
  MODE="local"
fi

CHEAP_MODEL="${CHEAP_MODEL:-gemma4:e2b}"
EXPENSIVE_MODEL="${EXPENSIVE_MODEL:-gemma4:e4b}"
STRUCTURED_PARSER_MODEL="${STRUCTURED_PARSER_MODEL:-}"
METHODS="${METHODS:-suql v2_2 v2_3 v3 v3_2}"
REPETITIONS="${REPETITIONS:-11}"
OUTPUT_NAME="${OUTPUT_NAME:-}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
API_BASE="${API_BASE:-http://127.0.0.1:11434}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CASCADE_TARGET="${CASCADE_TARGET:-0.9}"
CALIBRATION_BUDGET="${CALIBRATION_BUDGET:-20}"
MANUAL_CONFIDENCE_THRESHOLD="${MANUAL_CONFIDENCE_THRESHOLD:-}"
CHEAP_ACCEPT_THRESHOLD="${CHEAP_ACCEPT_THRESHOLD:-}"
CHEAP_REJECT_THRESHOLD="${CHEAP_REJECT_THRESHOLD:-}"
CHEAP_BATCH_SIZE="${CHEAP_BATCH_SIZE:-8}"
EXPENSIVE_BATCH_SIZE="${EXPENSIVE_BATCH_SIZE:-8}"
V2_3_EXPENSIVE_BATCH_SIZE="${V2_3_EXPENSIVE_BATCH_SIZE:-32}"
MAX_EXPENSIVE_CALLS="${MAX_EXPENSIVE_CALLS:-4}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/runner.sh local [options]

Options:
  --cheap-model MODEL
  --expensive-model MODEL
  --structured-parser-model MODEL
  --disable-llm-structured-parser
  --methods "suql v2_2 v2_3 v3 v3_2"
  --repetitions N
  --output-name NAME
  --output-dir DIR
  --api-base URL
  --python PATH
  --request-timeout SEC
  --cascade-target FLOAT
  --calibration-budget N
  --manual-confidence-threshold FLOAT
  --cheap-accept-threshold FLOAT
  --cheap-reject-threshold FLOAT
USAGE
}

OUTPUT_DIR=""
DISABLE_LLM_STRUCTURED_PARSER=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cheap-model) CHEAP_MODEL="$2"; shift 2 ;;
    --expensive-model) EXPENSIVE_MODEL="$2"; shift 2 ;;
    --structured-parser-model) STRUCTURED_PARSER_MODEL="$2"; shift 2 ;;
    --disable-llm-structured-parser) DISABLE_LLM_STRUCTURED_PARSER=1; shift ;;
    --methods) METHODS="$2"; shift 2 ;;
    --repetitions) REPETITIONS="$2"; shift 2 ;;
    --output-name) OUTPUT_NAME="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --api-base) API_BASE="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --request-timeout) REQUEST_TIMEOUT="$2"; shift 2 ;;
    --cascade-target) CASCADE_TARGET="$2"; shift 2 ;;
    --calibration-budget) CALIBRATION_BUDGET="$2"; shift 2 ;;
    --manual-confidence-threshold) MANUAL_CONFIDENCE_THRESHOLD="$2"; shift 2 ;;
    --cheap-accept-threshold) CHEAP_ACCEPT_THRESHOLD="$2"; shift 2 ;;
    --cheap-reject-threshold) CHEAP_REJECT_THRESHOLD="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option $1" >&2; usage >&2; exit 1 ;;
  esac
done

slug() { printf "%s" "$1" | sed 's/^ollama\///; s/:latest$//; s#[/:]#_#g'; }
if [[ -z "$OUTPUT_NAME" ]]; then
  OUTPUT_NAME="$(printf '%s_%s_3q_%sreps_%s' "$(slug "$CHEAP_MODEL")" "$(slug "$EXPENSIVE_MODEL")" "$REPETITIONS" "$RUN_STAMP")"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -n "$OUTPUT_DIR" ]] || OUTPUT_DIR="$SCRIPT_DIR/../outputs/$OUTPUT_NAME"

command=(
  "$PYTHON_BIN" "$SCRIPT_DIR/run_all.py"
  --python "$PYTHON_BIN"
  --api-base "$API_BASE"
  --cheap-model "ollama/${CHEAP_MODEL#ollama/}"
  --expensive-model "ollama/${EXPENSIVE_MODEL#ollama/}"
  --cascade-target "$CASCADE_TARGET"
  --calibration-budget "$CALIBRATION_BUDGET"
  --cheap-batch-size "$CHEAP_BATCH_SIZE"
  --expensive-batch-size "$EXPENSIVE_BATCH_SIZE"
  --v2-3-expensive-batch-size "$V2_3_EXPENSIVE_BATCH_SIZE"
  --max-expensive-calls "$MAX_EXPENSIVE_CALLS"
  --request-timeout "$REQUEST_TIMEOUT"
  --repetitions "$REPETITIONS"
  --methods $METHODS
  --output-dir "$OUTPUT_DIR"
)

[[ -n "$STRUCTURED_PARSER_MODEL" ]] && command+=(--structured-parser-model "ollama/${STRUCTURED_PARSER_MODEL#ollama/}")
[[ "$DISABLE_LLM_STRUCTURED_PARSER" == "1" ]] && command+=(--disable-llm-structured-parser)
[[ -n "$MANUAL_CONFIDENCE_THRESHOLD" ]] && command+=(--manual-confidence-threshold "$MANUAL_CONFIDENCE_THRESHOLD")
[[ -n "$CHEAP_ACCEPT_THRESHOLD" ]] && command+=(--cheap-accept-threshold "$CHEAP_ACCEPT_THRESHOLD")
[[ -n "$CHEAP_REJECT_THRESHOLD" ]] && command+=(--cheap-reject-threshold "$CHEAP_REJECT_THRESHOLD")

exec "${command[@]}"
