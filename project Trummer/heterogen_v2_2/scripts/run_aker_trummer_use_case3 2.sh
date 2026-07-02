#!/usr/bin/env bash
# Run the Trummer-style IMDb semantic join on Aker.
#
# This script is meant to be launched on Aker. If launched from a login node,
# it submits itself as an OAR GPU job.
#
# Submit/run examples on Aker:
#   bash scripts/run_aker_trummer_use_case3.sh
#   oarsub -S scripts/run_aker_trummer_use_case3.sh
#
#OAR -n trummer-use-case3
#OAR -l /nodes=1/gpu=1,walltime=04:00:00
#OAR -O /home/daisy/remizova/project_Trummer/Heterogen_v1/logs/oar_%jobid%.out
#OAR -E /home/daisy/remizova/project_Trummer/Heterogen_v1/logs/oar_%jobid%.err

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/daisy/remizova/project_Trummer/Heterogen_v1}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
MODEL="${MODEL:-gemma4:e4b}"
PREDICATE="${PREDICATE:-the review chunk is about the same movie row, based on movie_id/tconst, and the review expresses a negative, critical, or strongly unfavorable opinion about the movie}"
API_BASE="${API_BASE:-http://127.0.0.1:11434}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
OLLAMA_BIN="${OLLAMA_BIN:-}"
PULL_MODEL="${PULL_MODEL:-0}"
MAX_MOVIES="${MAX_MOVIES:-100}"
MAX_REVIEWS="${MAX_REVIEWS:-1000}"
SAMPLE_MOVIES_BEFORE_YEAR_FILTER="${SAMPLE_MOVIES_BEFORE_YEAR_FILTER:-0}"
PREFILTER_REVIEWS_BY_MOVIE_ID="${PREFILTER_REVIEWS_BY_MOVIE_ID:-1}"
MOVIE_BLOCK_SIZE="${MOVIE_BLOCK_SIZE:-4}"
REVIEW_BLOCK_SIZE="${REVIEW_BLOCK_SIZE:-8}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
AUTO_OAR_SUBMIT="${AUTO_OAR_SUBMIT:-1}"

cd "$PROJECT_ROOT"
mkdir -p logs outputs

if [[ -z "${OAR_JOB_ID:-}" && "$AUTO_OAR_SUBMIT" == "1" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    if ! command -v oarsub >/dev/null 2>&1; then
      echo "ERROR: no GPU is visible and oarsub is not available on this host." >&2
      exit 1
    fi

    echo "No GPU is visible on this host; submitting this script as an OAR GPU job."
    echo "Project root: $PROJECT_ROOT"
    echo "Watch status with: oarstat -u \$USER"
    exec oarsub -S "$PROJECT_ROOT/scripts/run_aker_trummer_use_case3.sh"
  fi
fi

echo "Project root: $PROJECT_ROOT"
echo "Data dir: $DATA_DIR"
echo "Model: $MODEL"
echo "Predicate: $PREDICATE"
echo "API base: $API_BASE"
echo "Movie source rows: $MAX_MOVIES (sample before year filter: $SAMPLE_MOVIES_BEFORE_YEAR_FILTER)"
echo "Review source rows: $MAX_REVIEWS"
echo "Prefilter reviews by movie_id: $PREFILTER_REVIEWS_BY_MOVIE_ID"
echo "Run stamp: $RUN_STAMP"
echo "OAR job id: ${OAR_JOB_ID:-not-running-under-oar}"

if [[ ! -f "$DATA_DIR/imdb_structured_joined.csv" ]]; then
  echo "ERROR: missing $DATA_DIR/imdb_structured_joined.csv" >&2
  exit 1
fi
if [[ ! -f "$DATA_DIR/imdb_reviews.csv" ]]; then
  echo "ERROR: missing $DATA_DIR/imdb_reviews.csv" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: no NVIDIA GPU is visible. Run inside a GPU allocation." >&2
  exit 1
fi

echo "Visible GPU(s):"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

find_ollama() {
  if [[ -n "$OLLAMA_BIN" && -x "$OLLAMA_BIN" ]]; then
    echo "$OLLAMA_BIN"
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
      echo "$candidate"
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
  echo "ERROR: ollama command not found on this GPU host." >&2
  echo "Try: module avail ollama; find \$HOME -name ollama -type f 2>/dev/null" >&2
  exit 1
fi
echo "Ollama binary: $OLLAMA_BIN"

ollama_pid=""
cleanup() {
  if [[ -n "$ollama_pid" ]] && kill -0 "$ollama_pid" >/dev/null 2>&1; then
    echo "Stopping Ollama server started by this script: pid=$ollama_pid"
    kill "$ollama_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

wait_for_ollama() {
  local max_wait_seconds="${1:-180}"
  local waited=0
  until python3 - <<'PY'
import os
import sys
import urllib.request

url = "http://" + os.environ.get("OLLAMA_HOST", "127.0.0.1:11434") + "/api/tags"
try:
    with urllib.request.urlopen(url, timeout=2) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
  do
    if (( waited >= max_wait_seconds )); then
      echo "ERROR: Ollama did not become ready after ${max_wait_seconds}s." >&2
      exit 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
}

if python3 - <<'PY'
import os
import sys
import urllib.request

url = "http://" + os.environ.get("OLLAMA_HOST", "127.0.0.1:11434") + "/api/tags"
try:
    with urllib.request.urlopen(url, timeout=2) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
then
  echo "Using existing Ollama server at http://$OLLAMA_HOST"
else
  echo "Starting Ollama server on http://$OLLAMA_HOST"
  OLLAMA_HOST="$OLLAMA_HOST" nohup "$OLLAMA_BIN" serve > "logs/ollama_trummer_${RUN_STAMP}.log" 2>&1 &
  ollama_pid="$!"
  wait_for_ollama 180
fi

if [[ "$PULL_MODEL" == "1" ]]; then
  models_to_pull=("$MODEL")
  declare -A pulled_models=()
  for model_to_pull in "${models_to_pull[@]}"; do
    if [[ -z "${pulled_models[$model_to_pull]:-}" ]]; then
      echo "Pulling Ollama model if needed: $model_to_pull"
      "$OLLAMA_BIN" pull "$model_to_pull"
      pulled_models["$model_to_pull"]=1
    fi
  done
fi

RUN_OUT="$PROJECT_ROOT/outputs/aker_trummer_${RUN_STAMP}"
CONSOLE_LOG="$PROJECT_ROOT/outputs/aker_trummer_${RUN_STAMP}.console.log"
mkdir -p "$RUN_OUT"

echo "Output dir: $RUN_OUT"
echo "Console log: $CONSOLE_LOG"

movie_sampling_args=()
review_prefilter_args=()
if [[ "$SAMPLE_MOVIES_BEFORE_YEAR_FILTER" == "1" ]]; then
  movie_sampling_args+=(--sample-movies-before-year-filter)
fi
if [[ "$PREFILTER_REVIEWS_BY_MOVIE_ID" == "1" ]]; then
  review_prefilter_args+=(--prefilter-reviews-by-movie-id)
fi
python3 -u run_use_case3_light.py \
  --data-dir "$DATA_DIR" \
  --api-base "$API_BASE" \
  --model "$MODEL" \
  --predicate "$PREDICATE" \
  --max-movies "$MAX_MOVIES" \
  --max-reviews "$MAX_REVIEWS" \
  --movie-block-size "$MOVIE_BLOCK_SIZE" \
  --review-block-size "$REVIEW_BLOCK_SIZE" \
  "${movie_sampling_args[@]}" \
  "${review_prefilter_args[@]}" \
  --output-dir "$RUN_OUT" \
  2>&1 | tee "$CONSOLE_LOG"

echo
echo "Run finished."
echo "Outputs:"
echo "  $RUN_OUT/use_case3_join_stats.csv"
echo "  $RUN_OUT/use_case3_joined_pairs.csv"
echo "  $RUN_OUT/use_case3_final_movies.csv"
echo "Console log:"
echo "  $CONSOLE_LOG"
