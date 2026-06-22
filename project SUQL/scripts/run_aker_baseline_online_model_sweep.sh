#!/usr/bin/env bash
# Run baseline vs online-join model sweep on Aker for data_samples/data_sample_200.
# This script is meant to run as an OAR job on an Aker GPU host. If launched
# from a login node, it submits itself to OAR.
#
# Submit/run examples on Aker:
#   bash scripts/run_aker_baseline_online_model_sweep.sh
#   oarsub -S scripts/run_aker_baseline_online_model_sweep.sh
#
#OAR -n suql-model-sweep
#OAR -l /nodes=1/gpu=1,walltime=24:00:00
#OAR -O /home/daisy/remizova/project/benchmarks/baseline_vs_join/sample_200_model_sweeps/oar_%jobid%.out
#OAR -E /home/daisy/remizova/project/benchmarks/baseline_vs_join/sample_200_model_sweeps/oar_%jobid%.err

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/daisy/remizova/project}"
MODELS=(${MODELS:-gemma2:2b llama3.2:1b smollm2:360m smollm2:1.7b tinyllama:1.1b qwen2.5:0.5b qwen2.5:1.5b llama3.2:3b})
SUQL_API_BASE="${SUQL_API_BASE:-http://127.0.0.1:11434}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
OLLAMA_BIN="${OLLAMA_BIN:-}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
AUTO_OAR_SUBMIT="${AUTO_OAR_SUBMIT:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/benchmarks/baseline_vs_join/sample_200_model_sweeps}"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_ROOT" logs

if [[ -z "${OAR_JOB_ID:-}" && "$AUTO_OAR_SUBMIT" == "1" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    if ! command -v oarsub >/dev/null 2>&1; then
      echo "ERROR: no GPU is visible and oarsub is not available on this host." >&2
      exit 1
    fi

    echo "No GPU is visible on this host; submitting this script as an OAR GPU job."
    echo "Project root: $PROJECT_ROOT"
    echo "Models: ${MODELS[*]}"
    echo "Watch status with: oarstat -u \$USER"
    echo "Watch logs with: ls -lh $OUTPUT_ROOT/oar_*.out"
    exec oarsub -S "$PROJECT_ROOT/scripts/run_aker_baseline_online_model_sweep.sh"
  fi
fi

echo "Project root: $PROJECT_ROOT"
echo "Models: ${MODELS[*]}"
echo "API base: $SUQL_API_BASE"
echo "Ollama host: $OLLAMA_HOST"
echo "Output root: $OUTPUT_ROOT"
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
  OLLAMA_HOST="$OLLAMA_HOST" nohup "$OLLAMA_BIN" serve > "logs/ollama_model_sweep_${RUN_STAMP}.log" 2>&1 &
  ollama_pid="$!"
  wait_for_ollama 180
fi

export SUQL_API_BASE
export OLLAMA_HOST

"$PYTHON_BIN" -u scripts/run_baseline_online_model_sweep.py \
  --pull-models \
  --api-base "$SUQL_API_BASE" \
  --ollama-bin "$OLLAMA_BIN" \
  --python "$PYTHON_BIN" \
  --output-root "$OUTPUT_ROOT" \
  --models "${MODELS[@]}" \
  2>&1 | tee "$OUTPUT_ROOT/model_sweep_${RUN_STAMP}.console.log"

echo
echo "Model sweep finished."
echo "Summary: $OUTPUT_ROOT/sweep_summary.csv"
echo "Best model: $OUTPUT_ROOT/best_model.txt"
echo "Per-model outputs: $OUTPUT_ROOT/<model_name>/"
