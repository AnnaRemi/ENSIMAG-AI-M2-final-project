#!/usr/bin/env bash
# OAR worker: runs v1 and v2 on the same GPU node and Ollama server.

set -Eeuo pipefail

AKER_ROOT="${AKER_ROOT:-/home/daisy/remizova/common_benchmark_v3_workspace}"
COMMON_ROOT="$AKER_ROOT/common_benchmark_v3"
CHEAP_MODEL="${CHEAP_MODEL:-gemma2:2b}"
EXPENSIVE_MODELS="${EXPENSIVE_MODELS:-qwen2.5:3b}"
PULL_MODELS="${PULL_MODELS:-0}"
CHEAP_ACCEPT_THRESHOLD="${CHEAP_ACCEPT_THRESHOLD:-3.0}"
CHEAP_REJECT_THRESHOLD="${CHEAP_REJECT_THRESHOLD:--1.5}"
EXPENSIVE_BATCH_SIZE="${EXPENSIVE_BATCH_SIZE:-8}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OLLAMA_BIN="${OLLAMA_BIN:-}"

cd "$AKER_ROOT"
mkdir -p "$COMMON_ROOT/logs" "$COMMON_ROOT/outputs"
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
OLLAMA_LOG="$COMMON_ROOT/logs/ollama_${OAR_JOB_ID}_${RUN_STAMP}.log"
ollama_pid=""
cleanup() {
  if [[ -n "$ollama_pid" ]] && kill -0 "$ollama_pid" >/dev/null 2>&1; then
    kill "$ollama_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
OLLAMA_HOST="$OLLAMA_HOST" nohup "$OLLAMA_BIN" serve >"$OLLAMA_LOG" 2>&1 &
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

models="$CHEAP_MODEL $EXPENSIVE_MODELS"
declare -A checked=()
for model in $models; do
  model="${model#ollama/}"
  if [[ -n "${checked[$model]:-}" ]]; then continue; fi
  checked["$model"]=1
  if [[ "$PULL_MODELS" == "1" ]]; then
    OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" pull "$model"
  elif ! OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" show "$model" >/dev/null 2>&1; then
    echo "ERROR: missing model '$model'; resubmit with PULL_MODELS=1." >&2
    exit 1
  fi
done

export COMMON_BENCHMARK_V3_HETEROGEN_V1_ROOT="$AKER_ROOT/project_Trummer/heterogen_v1"
export COMMON_BENCHMARK_V3_HETEROGEN_V2_ROOT="$AKER_ROOT/project_Trummer/heterogen_v2"
for expensive_model in $EXPENSIVE_MODELS; do
  console_slug="$(printf '%s__%s' "$CHEAP_MODEL" "$expensive_model" | sed 's#[/:]#_#g')"
  console_log="$COMMON_ROOT/logs/${console_slug}_${OAR_JOB_ID}_${RUN_STAMP}.console.log"
  echo "Running cheap=$CHEAP_MODEL expensive=$expensive_model"
  echo "Console log: $console_log"
  MPLBACKEND=Agg MPLCONFIGDIR="$COMMON_ROOT/.mplconfig" \
    "$PYTHON" -u "$COMMON_ROOT/scripts/run_all.py" \
      --python "$PYTHON" \
      --api-base "$API_BASE" \
      --cheap-model "ollama/${CHEAP_MODEL#ollama/}" \
      --expensive-model "ollama/${expensive_model#ollama/}" \
      --cheap-accept-threshold "$CHEAP_ACCEPT_THRESHOLD" \
      --cheap-reject-threshold "$CHEAP_REJECT_THRESHOLD" \
      --expensive-batch-size "$EXPENSIVE_BATCH_SIZE" \
      --request-timeout "$REQUEST_TIMEOUT" \
      2>&1 | tee "$console_log"
done

echo "Completed. Outputs: $COMMON_ROOT/outputs"
