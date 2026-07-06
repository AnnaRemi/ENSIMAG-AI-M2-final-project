#!/usr/bin/env bash
# Single entry point for common_benchmark_v3 Heterogen comparisons.

set -Eeuo pipefail

MODE="${1:-local}"
if [[ "$MODE" == "local" || "$MODE" == "submit-aker" || "$MODE" == "aker-worker" ]]; then
  shift || true
else
  MODE="local"
fi

AKER_ROOT="${AKER_ROOT:-/home/daisy/remizova/common_benchmark_v3_workspace}"
CHEAP_MODEL="${CHEAP_MODEL:-gemma4:e2b}"
EXPENSIVE_MODEL="${EXPENSIVE_MODEL:-gemma4:e4b}"
METHODS="${METHODS:-v1 v2 v2_2 v2_3 v3}"
REPETITIONS="${REPETITIONS:-9}"
OUTPUT_NAME="${OUTPUT_NAME:-all_heterogen_$(date +%Y%m%d_%H%M%S)}"
PULL_MODELS="${PULL_MODELS:-0}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-4}"
WALLTIME="${WALLTIME:-08:00:00}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
API_BASE="${API_BASE:-http://127.0.0.1:11434}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CASCADE_TARGET="${CASCADE_TARGET:-0.9}"
CALIBRATION_BUDGET="${CALIBRATION_BUDGET:-20}"
V2_MANUAL_CONFIDENCE_THRESHOLD="${V2_MANUAL_CONFIDENCE_THRESHOLD:-}"
CHEAP_BATCH_SIZE="${CHEAP_BATCH_SIZE:-8}"
EXPENSIVE_BATCH_SIZE="${EXPENSIVE_BATCH_SIZE:-8}"
V2_3_EXPENSIVE_BATCH_SIZE="${V2_3_EXPENSIVE_BATCH_SIZE:-32}"
MAX_EXPENSIVE_CALLS="${MAX_EXPENSIVE_CALLS:-4}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"
TOKEN_THRESHOLD="${TOKEN_THRESHOLD:-4096}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-512}"
MAX_MOVIE_BLOCK_SIZE="${MAX_MOVIE_BLOCK_SIZE:-25}"
MAX_REVIEW_BLOCK_SIZE="${MAX_REVIEW_BLOCK_SIZE:-8}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/runner.sh local [options]
  scripts/runner.sh submit-aker [options]

Options:
  --cheap-model MODEL
  --expensive-model MODEL
  --methods "v1 v2 v2_2 v2_3 v3"
  --repetitions N
  --output-name NAME
  --output-dir DIR              Local mode only.
  --api-base URL
  --python PATH
  --pull-models
  --require-gpu 0|1
  --parallel-workers N
  --walltime HH:MM:SS
  --request-timeout SEC
USAGE
}

OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cheap-model) CHEAP_MODEL="$2"; shift 2 ;;
    --expensive-model) EXPENSIVE_MODEL="$2"; shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    --repetitions) REPETITIONS="$2"; shift 2 ;;
    --output-name) OUTPUT_NAME="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --api-base) API_BASE="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --pull-models) PULL_MODELS=1; shift ;;
    --require-gpu) REQUIRE_GPU="$2"; shift 2 ;;
    --parallel-workers) PARALLEL_WORKERS="$2"; shift 2 ;;
    --walltime) WALLTIME="$2"; shift 2 ;;
    --request-timeout) REQUEST_TIMEOUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option $1" >&2; usage >&2; exit 1 ;;
  esac
done

skip_args=()
for method in v1 v2 v2_2 v2_3 v3; do
  if [[ " $METHODS " != *" $method "* ]]; then
    skip_args+=("--skip-${method//_/-}")
  fi
done

if [[ "$MODE" == "local" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  [[ -n "$OUTPUT_DIR" ]] || OUTPUT_DIR="$SCRIPT_DIR/../outputs/$OUTPUT_NAME"
  command=(
    "$PYTHON_BIN" "$SCRIPT_DIR/run_all_heterogen.py"
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
    --parallel-workers "$PARALLEL_WORKERS"
    --request-timeout "$REQUEST_TIMEOUT"
    --token-threshold "$TOKEN_THRESHOLD"
    --max-completion-tokens "$MAX_COMPLETION_TOKENS"
    --max-movie-block-size "$MAX_MOVIE_BLOCK_SIZE"
    --max-review-block-size "$MAX_REVIEW_BLOCK_SIZE"
    --repetitions "$REPETITIONS"
    --outputs-dir "$OUTPUT_DIR"
    "${skip_args[@]}"
  )
  [[ -n "$V2_MANUAL_CONFIDENCE_THRESHOLD" ]] && command+=(--v2-manual-confidence-threshold "$V2_MANUAL_CONFIDENCE_THRESHOLD")
  exec "${command[@]}"
fi

if [[ "$MODE" == "submit-aker" ]]; then
  WORKER="$AKER_ROOT/common_benchmark_v3/scripts/_aker_worker.sh"
  JOBS="$AKER_ROOT/common_benchmark_v3/jobs"
  LOGS="$AKER_ROOT/common_benchmark_v3/logs"
  WRAPPER="$JOBS/common_benchmark_v3_${RUN_STAMP}.sh"
  [[ -x "$WORKER" ]] || { echo "ERROR: missing worker $WORKER; sync first." >&2; exit 1; }
  mkdir -p "$JOBS" "$LOGS"
  shell_quote() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }
  {
    echo "#!/usr/bin/env bash"
    echo "set -Eeuo pipefail"
    for name in \
      AKER_ROOT CHEAP_MODEL EXPENSIVE_MODEL PULL_MODELS CASCADE_TARGET CALIBRATION_BUDGET \
      CHEAP_BATCH_SIZE EXPENSIVE_BATCH_SIZE V2_3_EXPENSIVE_BATCH_SIZE MAX_EXPENSIVE_CALLS \
      PARALLEL_WORKERS REQUEST_TIMEOUT TOKEN_THRESHOLD MAX_COMPLETION_TOKENS \
      MAX_MOVIE_BLOCK_SIZE MAX_REVIEW_BLOCK_SIZE REPETITIONS REQUIRE_GPU RUN_STAMP OUTPUT_NAME
    do
      printf "export %s=%s\n" "$name" "$(shell_quote "${!name}")"
    done
    for method in v1 v2 v2_2 v2_3 v3; do
      var="SKIP_${method^^}"
      [[ " $METHODS " == *" $method "* ]] && value=0 || value=1
      printf "export %s=%s\n" "$var" "$value"
    done
    [[ -n "$V2_MANUAL_CONFIDENCE_THRESHOLD" ]] && printf "export MANUAL_V2_CONFIDENCE_THRESHOLD=%s\n" "$(shell_quote "$V2_MANUAL_CONFIDENCE_THRESHOLD")"
    printf "exec bash %s\n" "$(shell_quote "$WORKER")"
  } >"$WRAPPER"
  chmod 700 "$WRAPPER"
  oarsub -n common-benchmark-v3 -l "/nodes=1/gpu=1,walltime=$WALLTIME" \
    -O "$LOGS/oar_%jobid%.out" -E "$LOGS/oar_%jobid%.err" -S "$WRAPPER"
  echo "Output name: $OUTPUT_NAME"
  echo "Status: oarstat -u \$USER"
  echo "Logs: $LOGS"
  exit 0
fi

exec bash "$AKER_ROOT/common_benchmark_v3/scripts/_aker_worker.sh"
