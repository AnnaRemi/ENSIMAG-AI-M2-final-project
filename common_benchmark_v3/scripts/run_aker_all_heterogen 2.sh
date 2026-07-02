#!/usr/bin/env bash
# OAR worker: run Heterogen versions on one GPU and one Ollama server.

set -Eeuo pipefail

AKER_ROOT="${AKER_ROOT:-/home/daisy/remizova/common_benchmark_v3_workspace}"
COMMON_ROOT="$AKER_ROOT/common_benchmark_v3"
CHEAP_MODEL="${CHEAP_MODEL:-gemma4:e2b}"
EXPENSIVE_MODEL="${EXPENSIVE_MODEL:-gemma4:e4b}"
PULL_MODELS="${PULL_MODELS:-0}"
CASCADE_TARGET="${CASCADE_TARGET:-0.9}"
CALIBRATION_BUDGET="${CALIBRATION_BUDGET:-20}"
MANUAL_V2_CONFIDENCE_THRESHOLD="${MANUAL_V2_CONFIDENCE_THRESHOLD:-}"
CHEAP_BATCH_SIZE="${CHEAP_BATCH_SIZE:-8}"
EXPENSIVE_BATCH_SIZE="${EXPENSIVE_BATCH_SIZE:-8}"
V2_3_EXPENSIVE_BATCH_SIZE="${V2_3_EXPENSIVE_BATCH_SIZE:-32}"
MAX_EXPENSIVE_CALLS="${MAX_EXPENSIVE_CALLS:-4}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-4}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"
TOKEN_THRESHOLD="${TOKEN_THRESHOLD:-4096}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-512}"
MAX_MOVIE_BLOCK_SIZE="${MAX_MOVIE_BLOCK_SIZE:-25}"
MAX_REVIEW_BLOCK_SIZE="${MAX_REVIEW_BLOCK_SIZE:-8}"
REPETITIONS="${REPETITIONS:-9}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_NAME="${OUTPUT_NAME:-all_heterogen_${RUN_STAMP}}"
SKIP_V1="${SKIP_V1:-0}"
SKIP_V2="${SKIP_V2:-0}"
SKIP_V2_2="${SKIP_V2_2:-0}"
SKIP_V2_3="${SKIP_V2_3:-0}"
SKIP_V3="${SKIP_V3:-0}"
OLLAMA_BIN="${OLLAMA_BIN:-}"

cd "$AKER_ROOT"
mkdir -p "$COMMON_ROOT/logs" "$COMMON_ROOT/outputs/$OUTPUT_NAME"
if [[ -z "${OAR_JOB_ID:-}" ]]; then
  echo "ERROR: this worker requires an OAR allocation." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: no NVIDIA GPU is visible." >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if [[ ! -x "$AKER_ROOT/.venv/bin/python" ]]; then
  python3 -m venv "$AKER_ROOT/.venv"
fi
PYTHON="$AKER_ROOT/.venv/bin/python"
"$PYTHON" -m pip install -r "$COMMON_ROOT/requirements.txt"

find_ollama() {
  if [[ -n "$OLLAMA_BIN" && -x "$OLLAMA_BIN" ]]; then printf '%s\n' "$OLLAMA_BIN"; return; fi
  if command -v ollama >/dev/null 2>&1; then command -v ollama; return; fi
  for candidate in "$HOME/.local/ollama/bin/ollama" "$HOME/.local/bin/ollama" "$HOME/bin/ollama" /usr/local/bin/ollama /usr/bin/ollama; do
    if [[ -x "$candidate" ]]; then printf '%s\n' "$candidate"; return; fi
  done
  return 1
}

OLLAMA_BIN="$(find_ollama || true)"
if [[ -z "$OLLAMA_BIN" ]]; then
  echo "ERROR: Ollama not found; set OLLAMA_BIN." >&2
  exit 1
fi

OLLAMA_PORT="${OLLAMA_PORT:-$((12000 + OAR_JOB_ID % 1000))}"
export OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"
API_BASE="http://$OLLAMA_HOST"
OLLAMA_LOG="$COMMON_ROOT/logs/all_heterogen_ollama_${OAR_JOB_ID}_${RUN_STAMP}.log"
CONSOLE_LOG="$COMMON_ROOT/logs/all_heterogen_${OAR_JOB_ID}_${RUN_STAMP}.console.log"
ollama_pid=""
cleanup() {
  if [[ -n "$ollama_pid" ]] && kill -0 "$ollama_pid" >/dev/null 2>&1; then
    kill "$ollama_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

OLLAMA_HOST="$OLLAMA_HOST" OLLAMA_NUM_PARALLEL="$PARALLEL_WORKERS" \
  nohup "$OLLAMA_BIN" serve >"$OLLAMA_LOG" 2>&1 &
ollama_pid="$!"

ready=0
for _ in $(seq 1 90); do
  if "$PYTHON" - "$API_BASE" <<'PY'
import sys, urllib.request
try:
    urllib.request.urlopen(sys.argv[1] + "/api/tags", timeout=2)
except Exception:
    raise SystemExit(1)
PY
  then
    ready=1
    break
  fi
  sleep 2
done
if [[ "$ready" != "1" ]]; then
  echo "ERROR: Ollama did not become ready. See $OLLAMA_LOG" >&2
  exit 1
fi
if [[ "$REQUIRE_GPU" == "1" ]]; then
  if grep -qi "NVIDIA driver too old" "$OLLAMA_LOG"; then
    echo "ERROR: Ollama cannot use the allocated NVIDIA GPU." >&2
    echo "The Ollama log reports an NVIDIA driver that is too old." >&2
    echo "This job would run on CPU and be extremely slow. See: $OLLAMA_LOG" >&2
    exit 1
  fi
fi

for model in "$CHEAP_MODEL" "$EXPENSIVE_MODEL"; do
  model="${model#ollama/}"
  echo "Checking Ollama model: $model"
  if [[ "$PULL_MODELS" == "1" ]]; then
    if ! OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" pull "$model"; then
      echo "ERROR: failed to pull Ollama model '$model'." >&2
      exit 1
    fi
  elif ! OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" show "$model" >/dev/null 2>&1; then
    echo "ERROR: missing model '$model'; resubmit with PULL_MODELS=1." >&2
    exit 1
  fi
done

export COMMON_BENCHMARK_V3_HETEROGEN_V1_ROOT="$AKER_ROOT/project_Trummer/heterogen_v1"
export COMMON_BENCHMARK_V3_HETEROGEN_V2_ROOT="$AKER_ROOT/project_Trummer/heterogen_v2"
export COMMON_BENCHMARK_V3_HETEROGEN_V2_2_ROOT="$AKER_ROOT/project_Trummer/heterogen_v2_2"
export COMMON_BENCHMARK_V3_HETEROGEN_V2_3_ROOT="$AKER_ROOT/project_Trummer/heterogen_v2_3"
export COMMON_BENCHMARK_V3_HETEROGEN_V3_ROOT="$AKER_ROOT/project_Trummer/heterogen_v3"

command=(
  "$PYTHON" -u "$COMMON_ROOT/scripts/run_all_heterogen.py"
  --python "$PYTHON"
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
  --outputs-dir "$COMMON_ROOT/outputs/$OUTPUT_NAME"
)
if [[ -n "$MANUAL_V2_CONFIDENCE_THRESHOLD" ]]; then
  command+=(--v2-manual-confidence-threshold "$MANUAL_V2_CONFIDENCE_THRESHOLD")
fi
[[ "$SKIP_V1" == "1" ]] && command+=(--skip-v1)
[[ "$SKIP_V2" == "1" ]] && command+=(--skip-v2)
[[ "$SKIP_V2_2" == "1" ]] && command+=(--skip-v2-2)
[[ "$SKIP_V2_3" == "1" ]] && command+=(--skip-v2-3)
[[ "$SKIP_V3" == "1" ]] && command+=(--skip-v3)

echo "Output directory: $COMMON_ROOT/outputs/$OUTPUT_NAME"
echo "Console log: $CONSOLE_LOG"
echo "Require GPU: $REQUIRE_GPU"
MPLBACKEND=Agg MPLCONFIGDIR="$COMMON_ROOT/.mplconfig" \
  "${command[@]}" 2>&1 | tee "$CONSOLE_LOG"

echo "Completed all-Heterogen experiment."
echo "Outputs: $COMMON_ROOT/outputs/$OUTPUT_NAME"
echo "Console log: $CONSOLE_LOG"
echo "Ollama log: $OLLAMA_LOG"
