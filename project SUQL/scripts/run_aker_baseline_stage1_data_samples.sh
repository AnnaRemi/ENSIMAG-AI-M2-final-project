#!/usr/bin/env bash
# Run baseline vs Stage 1 benchmarks on Aker using data_samples/data_sample_*
# directories. This script is meant to run as an OAR job on Aker/GPU host
# itself, not from a laptop through an SSH tunnel.
#
# Submit/run examples on Aker:
#   bash scripts/run_aker_baseline_stage1_data_samples.sh
#   oarsub -S scripts/run_aker_baseline_stage1_data_samples.sh
#
#OAR -n suql-baseline-stage1
#OAR -l /nodes=1/gpu=1,walltime=12:00:00
#OAR -O /home/daisy/remizova/project/Stage_1/benchmarks/oar_%jobid%.out
#OAR -E /home/daisy/remizova/project/Stage_1/benchmarks/oar_%jobid%.err

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/daisy/remizova/project}"
SIZES=(${SIZES:-100 200 500 1000 1500})
SUQL_MODEL="${SUQL_MODEL:-ollama/phi4-mini}"
SUQL_API_BASE="${SUQL_API_BASE:-http://127.0.0.1:11434}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
OLLAMA_BIN="${OLLAMA_BIN:-}"
PULL_MODELS="${PULL_MODELS:-0}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
AUTO_OAR_SUBMIT="${AUTO_OAR_SUBMIT:-1}"

cd "$PROJECT_ROOT"
mkdir -p Stage_1/benchmarks logs

if [[ -z "${OAR_JOB_ID:-}" && "$AUTO_OAR_SUBMIT" == "1" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    if ! command -v oarsub >/dev/null 2>&1; then
      echo "ERROR: no GPU is visible and oarsub is not available on this host." >&2
      exit 1
    fi

    echo "No GPU is visible on this host; submitting this script as an OAR GPU job."
    echo "Project root: $PROJECT_ROOT"
    echo "Watch status with: oarstat -u \$USER"
    echo "Watch logs with: ls -lh $PROJECT_ROOT/Stage_1/benchmarks/oar_*.out"
    exec oarsub -S "$PROJECT_ROOT/scripts/run_aker_baseline_stage1_data_samples.sh"
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
"$PYTHON_BIN" -m pip install -r requirements.txt -r Stage_1/requirements.txt matplotlib

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
  OLLAMA_HOST="$OLLAMA_HOST" nohup "$OLLAMA_BIN" serve > "logs/ollama_stage1_${RUN_STAMP}.log" 2>&1 &
  ollama_pid="$!"
  wait_for_ollama 180
fi

if [[ "$PULL_MODELS" == "1" ]]; then
  plain_model="${SUQL_MODEL#ollama/}"
  echo "Pulling Ollama model if needed: $plain_model"
  "$OLLAMA_BIN" pull "$plain_model"
fi

for size in "${SIZES[@]}"; do
  data_path="$PROJECT_ROOT/data_samples/data_sample_${size}/imdb_joined.csv"
  if [[ ! -f "$data_path" ]]; then
    echo "ERROR: missing $data_path" >&2
    exit 1
  fi

  run_name="aker_baseline_stage1_data_sample_${size}_${RUN_STAMP}"
  console_log="$PROJECT_ROOT/Stage_1/benchmarks/${run_name}.console.log"

  echo
  echo "Running baseline vs stage1 for data_sample_${size}"
  echo "Data: $data_path"
  echo "Console log: $console_log"

  export SUQL_API_BASE
  export SUQL_MODEL
  export OLLAMA_HOST

  "$PYTHON_BIN" -u Stage_1/benchmark_stage1.py \
    --sample-size "$size" \
    --data-path "$data_path" \
    --api-base "$SUQL_API_BASE" \
    --model "$SUQL_MODEL" \
    --python "$PYTHON_BIN" \
    --run-name "$run_name" \
    2>&1 | tee "$console_log"

  echo "Finished sample $size"
  echo "Metrics: $PROJECT_ROOT/Stage_1/benchmarks/$run_name/metrics.csv"
done

echo
echo "Aggregating scaling plot."
"$PYTHON_BIN" scripts/aggregate_stage1_scaling_svg.py \
  --sizes "${SIZES[@]}" \
  --run-prefix "aker_baseline_stage1_data_sample" \
  --output-dir "$PROJECT_ROOT/Stage_1/benchmarks/baseline_vs_stage1_data_samples_${RUN_STAMP}"

echo
echo "All runs finished."
echo "Metrics files:"
for size in "${SIZES[@]}"; do
  echo "  $PROJECT_ROOT/Stage_1/benchmarks/aker_baseline_stage1_data_sample_${size}_${RUN_STAMP}/metrics.csv"
done
echo "Scaling output:"
echo "  $PROJECT_ROOT/Stage_1/benchmarks/baseline_vs_stage1_data_samples_${RUN_STAMP}/metrics_vs_sample_size.svg"
