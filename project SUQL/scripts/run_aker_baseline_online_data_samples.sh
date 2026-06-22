#!/usr/bin/env bash
# Run baseline vs online-join benchmarks on Aker using data_samples/data_sample_*
# directories. This script is meant to run as an OAR job on Aker/GPU host
# itself, not from a laptop through an SSH tunnel.
#
# Submit/run examples on Aker:
#   bash scripts/run_aker_baseline_online_data_samples.sh
#   oarsub -S scripts/run_aker_baseline_online_data_samples.sh
#
#OAR -n suql-baseline-online
#OAR -l /nodes=1/gpu=1,walltime=12:00:00
#OAR -O /home/daisy/remizova/project/benchmarks/oar_%jobid%.out
#OAR -E /home/daisy/remizova/project/benchmarks/oar_%jobid%.err

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/daisy/remizova/project}"
SIZES=(${SIZES:-200})
SUQL_MODEL="${SUQL_MODEL:-ollama/gemma2:2b}"
SUQL_API_BASE="${SUQL_API_BASE:-http://127.0.0.1:11434}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
OLLAMA_BIN="${OLLAMA_BIN:-}"
PULL_MODELS="${PULL_MODELS:-0}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
AUTO_OAR_SUBMIT="${AUTO_OAR_SUBMIT:-1}"
ALLOW_PHI4="${ALLOW_PHI4:-0}"

cd "$PROJECT_ROOT"
mkdir -p benchmarks logs

model_lc="$(printf '%s' "$SUQL_MODEL" | tr '[:upper:]' '[:lower:]')"
if [[ "$ALLOW_PHI4" != "1" && "$model_lc" == *phi4* ]]; then
  echo "ERROR: SUQL_MODEL is set to '$SUQL_MODEL', but this Aker run is configured for a non-phi4 Ollama model." >&2
  echo "Set SUQL_MODEL to another Ollama model, for example: SUQL_MODEL=ollama/gemma2:2b" >&2
  exit 1
fi

if [[ -z "${OAR_JOB_ID:-}" && "$AUTO_OAR_SUBMIT" == "1" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    if ! command -v oarsub >/dev/null 2>&1; then
      echo "ERROR: no GPU is visible and oarsub is not available on this host." >&2
      exit 1
    fi

    echo "No GPU is visible on this host; submitting this script as an OAR GPU job."
    echo "Project root: $PROJECT_ROOT"
    echo "Watch status with: oarstat -u \$USER"
    echo "Watch logs with: ls -lh $PROJECT_ROOT/benchmarks/oar_*.out"
    exec oarsub -S "$PROJECT_ROOT/scripts/run_aker_baseline_online_data_samples.sh"
  fi
fi

echo "Project root: $PROJECT_ROOT"
echo "Sample sizes: ${SIZES[*]}"
echo "Model: $SUQL_MODEL"
echo "API base: $SUQL_API_BASE"
echo "Run stamp: $RUN_STAMP"
echo "OAR job id: ${OAR_JOB_ID:-not-running-under-oar}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is not available. Run this script on an allocated GPU host." >&2
  exit 1
fi

if ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: no NVIDIA GPU is visible. Run this script inside a GPU allocation." >&2
  exit 1
fi

echo "Visible GPU(s):"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if [[ ! -x .venv/bin/python ]]; then
  echo "Creating Python virtual environment at $PROJECT_ROOT/.venv"
  python3 -m venv .venv
fi

PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
"$PYTHON_BIN" -m pip install -r requirements.txt

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
  echo "Set OLLAMA_BIN=/path/to/ollama or load the Aker Ollama module before submitting." >&2
  echo "Quick checks on Aker: module avail ollama; find \$HOME -name ollama -type f 2>/dev/null" >&2
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
  local max_wait_seconds="${1:-120}"
  local waited=0

  until "$PYTHON_BIN" - <<'PY'
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

if "$PYTHON_BIN" - <<'PY'
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
  OLLAMA_HOST="$OLLAMA_HOST" nohup "$OLLAMA_BIN" serve > "logs/ollama_${RUN_STAMP}.log" 2>&1 &
  ollama_pid="$!"
  wait_for_ollama 180
fi

if [[ "$PULL_MODELS" == "1" ]]; then
  plain_model="${SUQL_MODEL#ollama/}"
  echo "Pulling Ollama model if needed: $plain_model"
  OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" pull "$plain_model"
else
  plain_model="${SUQL_MODEL#ollama/}"
  if ! OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" list | awk 'NR > 1 {print $1}' | grep -Fx "$plain_model" >/dev/null; then
    echo "ERROR: Ollama model '$plain_model' is not installed on this host." >&2
    echo "Install it first or rerun with PULL_MODELS=1." >&2
    exit 1
  fi
fi

prepare_sample_dir() {
  local size="$1"
  local src_dir="$PROJECT_ROOT/data_samples/data_sample_${size}"
  local bench_dir="$PROJECT_ROOT/benchmarks/data_sample_${size}"

  if [[ ! -f "$src_dir/imdb_joined.csv" ]]; then
    echo "ERROR: missing $src_dir/imdb_joined.csv" >&2
    exit 1
  fi

  echo "Preparing data sample $size from $src_dir"
  "$PYTHON_BIN" - "$src_dir" <<'PY'
import sys
from pathlib import Path

import pandas as pd

sample_dir = Path(sys.argv[1])
joined_path = sample_dir / "imdb_joined.csv"
structured_path = sample_dir / "imdb_structured_joined.csv"
reviews_path = sample_dir / "imdb_reviews.csv"

joined = pd.read_csv(joined_path)
required = {"movie_id", "title", "year", "runtime", "director", "genres", "review"}
missing = sorted(required - set(joined.columns))
if missing:
    raise SystemExit(f"{joined_path} is missing required columns: {missing}")

structured_cols = ["movie_id", "title", "year", "runtime", "director", "genres"]
joined[structured_cols].drop_duplicates("movie_id").to_csv(structured_path, index=False)
joined[["movie_id", "review"]].to_csv(reviews_path, index=False)
print(f"rows={len(joined)} structured_rows={joined['movie_id'].nunique()}")
PY

  mkdir -p "$bench_dir"
  cp "$src_dir/imdb_joined.csv" "$bench_dir/imdb_joined.csv"
  cp "$src_dir/imdb_structured_joined.csv" "$bench_dir/imdb_structured_joined.csv"
  cp "$src_dir/imdb_reviews.csv" "$bench_dir/imdb_reviews.csv"
}

for size in "${SIZES[@]}"; do
  prepare_sample_dir "$size"

  run_name="aker_data_sample_${size}_${RUN_STAMP}"
  console_log="$PROJECT_ROOT/benchmarks/${run_name}.console.log"

  echo
  echo "Running baseline vs online_join for data_sample_${size}"
  echo "Console log: $console_log"

  export SUQL_API_BASE
  export SUQL_MODEL
  export OLLAMA_HOST

  "$PYTHON_BIN" -u benchmark_compare.py \
    --sample-size "$size" \
    --data-sample-dir "$PROJECT_ROOT/data_samples/data_sample_${size}" \
    --api-base "$SUQL_API_BASE" \
    --model "$SUQL_MODEL" \
    --python "$PYTHON_BIN" \
    --run-name "$run_name" \
    2>&1 | tee "$console_log"

  echo "Finished sample $size"
  echo "Metrics: $PROJECT_ROOT/benchmarks/$run_name/metrics.csv"
done

echo
echo "Aggregating scaling summary."
"$PYTHON_BIN" scripts/aggregate_online_join_scaling_svg.py \
  --sizes "${SIZES[@]}" \
  --run-prefix "aker_data_sample" \
  --output-dir "$PROJECT_ROOT/benchmarks/baseline_vs_online_join_data_samples_${RUN_STAMP}"

echo
echo "All runs finished."
echo "Metrics files:"
for size in "${SIZES[@]}"; do
  echo "  $PROJECT_ROOT/benchmarks/aker_data_sample_${size}_${RUN_STAMP}/metrics.csv"
done
echo "Scaling output:"
echo "  $PROJECT_ROOT/benchmarks/baseline_vs_online_join_data_samples_${RUN_STAMP}/metrics_vs_sample_size.svg"
