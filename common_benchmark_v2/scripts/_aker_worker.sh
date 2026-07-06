#!/usr/bin/env bash
# OAR worker. This runs entirely on the allocated Aker GPU node.

set -Eeuo pipefail

AKER_ROOT="${AKER_ROOT:-/home/daisy/remizova/common_benchmark_v2_workspace}"
COMMON_ROOT="$AKER_ROOT/common_benchmark_v2"
MODELS="${MODELS:-gemma4:e4b}"
PULL_MODELS="${PULL_MODELS:-0}"
SKIP_SUQL="${SKIP_SUQL:-0}"
SKIP_TRUMMER="${SKIP_TRUMMER:-0}"
TRUMMER_REQUEST_TIMEOUT="${TRUMMER_REQUEST_TIMEOUT:-3600}"
TRUMMER_TOKEN_THRESHOLD="${TRUMMER_TOKEN_THRESHOLD:-4096}"
TRUMMER_MAX_COMPLETION_TOKENS="${TRUMMER_MAX_COMPLETION_TOKENS:-256}"
TRUMMER_MAX_MOVIE_BLOCK_SIZE="${TRUMMER_MAX_MOVIE_BLOCK_SIZE:-25}"
TRUMMER_MAX_REVIEW_BLOCK_SIZE="${TRUMMER_MAX_REVIEW_BLOCK_SIZE:-8}"
REPETITIONS="${REPETITIONS:-9}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OLLAMA_BIN="${OLLAMA_BIN:-}"
OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}"

cd "$AKER_ROOT"
mkdir -p "$COMMON_ROOT/logs" "$COMMON_ROOT/outputs"

echo "Aker common benchmark v2"
echo "OAR job id: ${OAR_JOB_ID:-missing}"
echo "Host: $(hostname)"
echo "Models: $MODELS"
echo "Skip SUQL: $SKIP_SUQL"
echo "Skip Trummer: $SKIP_TRUMMER"
echo "Trummer request timeout: $TRUMMER_REQUEST_TIMEOUT seconds"
echo "Trummer token threshold: $TRUMMER_TOKEN_THRESHOLD"
echo "Trummer max completion tokens: $TRUMMER_MAX_COMPLETION_TOKENS"
echo "Trummer max block sizes: movies=$TRUMMER_MAX_MOVIE_BLOCK_SIZE reviews=$TRUMMER_MAX_REVIEW_BLOCK_SIZE"
echo "Repetitions: $REPETITIONS"
echo "Run stamp: $RUN_STAMP"

if [[ -z "${OAR_JOB_ID:-}" ]]; then
  echo "ERROR: this worker must run inside a non-interactive OAR allocation." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: no NVIDIA GPU is visible in the OAR job." >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if [[ ! -x "$AKER_ROOT/.venv/bin/python" ]]; then
  echo "Creating remote virtual environment: $AKER_ROOT/.venv"
  python3 -m venv "$AKER_ROOT/.venv"
fi
PYTHON_BIN="$AKER_ROOT/.venv/bin/python"
"$PYTHON_BIN" -m pip install -r "$COMMON_ROOT/requirements.txt"

find_ollama() {
  if [[ -n "$OLLAMA_BIN" && -x "$OLLAMA_BIN" ]]; then
    printf '%s\n' "$OLLAMA_BIN"
    return 0
  fi
  if command -v ollama >/dev/null 2>&1; then
    command -v ollama
    return 0
  fi
  for candidate in \
    "$HOME/.local/ollama/bin/ollama" \
    "$HOME/.local/bin/ollama" \
    "$HOME/bin/ollama" \
    "/usr/local/bin/ollama" \
    "/usr/bin/ollama" \
    "/opt/ollama/bin/ollama"
  do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

if ! OLLAMA_BIN="$(find_ollama)"; then
  if command -v module >/dev/null 2>&1; then
    set +u
    module load ollama >/dev/null 2>&1 || true
    module load ollama/latest >/dev/null 2>&1 || true
    set -u
    OLLAMA_BIN="$(find_ollama || true)"
  fi
fi
if [[ -z "$OLLAMA_BIN" ]]; then
  echo "ERROR: Ollama was not found on the GPU node." >&2
  echo "Set OLLAMA_BIN before submission if it is installed at a custom path." >&2
  exit 1
fi
echo "Ollama binary: $OLLAMA_BIN"

# Use a job-specific port so simultaneous jobs do not share or stop each
# other's model server.
OLLAMA_PORT="${OLLAMA_PORT:-$((12000 + OAR_JOB_ID % 1000))}"
export OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"
API_BASE="http://$OLLAMA_HOST"
OLLAMA_LOG="$COMMON_ROOT/logs/ollama_${OAR_JOB_ID}_${RUN_STAMP}.log"

ollama_pid=""
cleanup() {
  if [[ -n "$ollama_pid" ]] && kill -0 "$ollama_pid" >/dev/null 2>&1; then
    kill "$ollama_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "Starting job-owned Ollama server at $API_BASE"
OLLAMA_HOST="$OLLAMA_HOST" \
OLLAMA_CONTEXT_LENGTH="$OLLAMA_CONTEXT_LENGTH" \
nohup "$OLLAMA_BIN" serve > "$OLLAMA_LOG" 2>&1 &
ollama_pid="$!"

waited=0
until "$PYTHON_BIN" - "$API_BASE" <<'PY'
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1] + "/api/tags", timeout=2) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
do
  if (( waited >= 180 )); then
    echo "ERROR: Ollama did not become ready. See $OLLAMA_LOG" >&2
    exit 1
  fi
  sleep 2
  waited=$((waited + 2))
done

read -r -a model_array <<< "$MODELS"
if (( ${#model_array[@]} == 0 )); then
  echo "ERROR: MODELS is empty." >&2
  exit 1
fi

for plain_model in "${model_array[@]}"; do
  plain_model="${plain_model#ollama/}"
  if [[ "$PULL_MODELS" == "1" ]]; then
    echo "Pulling model: $plain_model"
    OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" pull "$plain_model"
  elif ! OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" show "$plain_model" >/dev/null 2>&1; then
    echo "ERROR: model '$plain_model' is not installed." >&2
    echo "Resubmit with PULL_MODELS=1 or choose an installed model." >&2
    exit 1
  fi

  model="ollama/$plain_model"
  model_slug="$(printf '%s' "$plain_model" | sed 's/:latest$//; s#[/:]#_#g')"
  console_log="$COMMON_ROOT/logs/${model_slug}_${OAR_JOB_ID}_${RUN_STAMP}.console.log"
  echo
  echo "Running model: $model"
  echo "Console log: $console_log"

  export COMMON_BENCHMARK_SUQL_ENGINE_DIR="$AKER_ROOT/project_SUQL/src_baseline"
  export COMMON_BENCHMARK_TRUMMER_ROOT="$AKER_ROOT/project_Trummer/heterogen_v1"

  implementation_args=()
  if [[ "$SKIP_SUQL" == "1" ]]; then
    implementation_args+=(--skip-suql-baseline)
  fi
  if [[ "$SKIP_TRUMMER" == "1" ]]; then
    implementation_args+=(--skip-trummer)
  fi

  MPLBACKEND=Agg \
  MPLCONFIGDIR="$COMMON_ROOT/.mplconfig" \
  "$PYTHON_BIN" -u "$COMMON_ROOT/scripts/run_all.py" \
    --api-base "$API_BASE" \
    --model "$model" \
    --python "$PYTHON_BIN" \
    --skip-build-dataset \
    --trummer-request-timeout "$TRUMMER_REQUEST_TIMEOUT" \
    --trummer-token-threshold "$TRUMMER_TOKEN_THRESHOLD" \
    --trummer-max-completion-tokens "$TRUMMER_MAX_COMPLETION_TOKENS" \
    --trummer-max-movie-block-size "$TRUMMER_MAX_MOVIE_BLOCK_SIZE" \
    --trummer-max-review-block-size "$TRUMMER_MAX_REVIEW_BLOCK_SIZE" \
    --repetitions "$REPETITIONS" \
    "${implementation_args[@]}" \
    2>&1 | tee "$console_log"
done

echo
echo "All requested models completed."
echo "Remote outputs: $COMMON_ROOT/outputs/"
echo "Remote logs: $COMMON_ROOT/logs/"
