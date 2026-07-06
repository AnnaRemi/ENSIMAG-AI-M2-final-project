#!/usr/bin/env bash
# Single entry point for common_benchmark.

set -Eeuo pipefail

MODE="${1:-local}"
if [[ "$MODE" == "local" || "$MODE" == "submit-aker" || "$MODE" == "aker-worker" ]]; then
  shift || true
else
  MODE="local"
fi

AKER_ROOT="${AKER_ROOT:-/home/daisy/remizova/common_benchmark_workspace}"
MODELS="${MODELS:-gemma4:e4b}"
MODEL="${MODEL:-gemma4:e4b}"
METHODS="${METHODS:-all}"
PULL_MODELS="${PULL_MODELS:-0}"
TRUMMER_REQUEST_TIMEOUT="${TRUMMER_REQUEST_TIMEOUT:-3600}"
REPETITIONS="${REPETITIONS:-9}"
WALLTIME="${WALLTIME:-04:00:00}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
API_BASE="${API_BASE:-http://127.0.0.1:11434}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/runner.sh local [options]
  scripts/runner.sh submit-aker [options]
  scripts/runner.sh aker-worker [options]

Options:
  --model MODEL                  Local single model, e.g. ollama/gemma4:e4b.
  --models "M1 M2"               Aker model list. Each model is run separately.
  --methods all|suql|trummer     Implementations to run.
  --repetitions N
  --api-base URL
  --python PATH
  --pull-models
  --dry-run
  --walltime HH:MM:SS
  --trummer-request-timeout SEC
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; MODELS="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    --repetitions) REPETITIONS="$2"; shift 2 ;;
    --api-base) API_BASE="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --pull-models) PULL_MODELS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --walltime) WALLTIME="$2"; shift 2 ;;
    --trummer-request-timeout) TRUMMER_REQUEST_TIMEOUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option $1" >&2; usage >&2; exit 1 ;;
  esac
done

method_args=()
case "$METHODS" in
  all) ;;
  suql) method_args+=(--skip-trummer) ;;
  trummer) method_args+=(--skip-suql-baseline) ;;
  *) echo "ERROR: --methods must be all, suql, or trummer" >&2; exit 1 ;;
esac
[[ "$DRY_RUN" == "1" ]] && method_args+=(--dry-run)

if [[ "$MODE" == "local" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  exec "$PYTHON_BIN" "$SCRIPT_DIR/run_all.py" \
    --api-base "$API_BASE" \
    --model "$MODEL" \
    --python "$PYTHON_BIN" \
    --trummer-request-timeout "$TRUMMER_REQUEST_TIMEOUT" \
    --repetitions "$REPETITIONS" \
    "${method_args[@]}"
fi

if [[ "$MODE" == "submit-aker" ]]; then
  WORKER="$AKER_ROOT/common_benchmark/scripts/_aker_worker.sh"
  JOBS_DIR="$AKER_ROOT/common_benchmark/jobs"
  LOGS_DIR="$AKER_ROOT/common_benchmark/logs"
  WRAPPER="$JOBS_DIR/common_benchmark_${RUN_STAMP}.sh"
  [[ -x "$WORKER" ]] || { echo "ERROR: missing worker $WORKER; sync first." >&2; exit 1; }
  mkdir -p "$JOBS_DIR" "$LOGS_DIR"
  shell_quote() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }
  {
    echo "#!/usr/bin/env bash"
    echo "set -Eeuo pipefail"
    for name in AKER_ROOT MODELS PULL_MODELS TRUMMER_REQUEST_TIMEOUT REPETITIONS RUN_STAMP; do
      printf "export %s=%s\n" "$name" "$(shell_quote "${!name}")"
    done
    case "$METHODS" in
      suql) echo "export SKIP_TRUMMER=1"; echo "export SKIP_SUQL=0" ;;
      trummer) echo "export SKIP_SUQL=1"; echo "export SKIP_TRUMMER=0" ;;
      *) echo "export SKIP_SUQL=0"; echo "export SKIP_TRUMMER=0" ;;
    esac
    printf "exec bash %s\n" "$(shell_quote "$WORKER")"
  } >"$WRAPPER"
  chmod 700 "$WRAPPER"
  oarsub -n common-benchmark -l "/nodes=1/gpu=1,walltime=$WALLTIME" \
    -O "$LOGS_DIR/oar_%jobid%.out" -E "$LOGS_DIR/oar_%jobid%.err" -S "$WRAPPER"
  echo "Status: oarstat -u \$USER"
  echo "Logs: $LOGS_DIR"
  exit 0
fi

exec bash "$AKER_ROOT/common_benchmark/scripts/_aker_worker.sh"
