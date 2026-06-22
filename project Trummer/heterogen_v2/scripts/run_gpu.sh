#!/usr/bin/env bash
# Run heterogen_v2 on an Aker GPU node. From a login node, it submits itself.
#OAR -n trummer-heterogen-v2
#OAR -l /nodes=1/gpu=1,walltime=04:00:00
#OAR -O /home/daisy/remizova/project_Trummer/heterogen_v2/logs/oar_%jobid%.out
#OAR -E /home/daisy/remizova/project_Trummer/heterogen_v2/logs/oar_%jobid%.err

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/daisy/remizova/project_Trummer/heterogen_v2}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
CHEAP_MODEL="${CHEAP_MODEL:-gemma2:2b}"
EXPENSIVE_MODEL="${EXPENSIVE_MODEL:-qwen2.5:3b}"
CHEAP_ACCEPT_THRESHOLD="${CHEAP_ACCEPT_THRESHOLD:-3.0}"
CHEAP_REJECT_THRESHOLD="${CHEAP_REJECT_THRESHOLD:--1.5}"
EXPENSIVE_BATCH_SIZE="${EXPENSIVE_BATCH_SIZE:-8}"
PULL_MODELS="${PULL_MODELS:-0}"
OLLAMA_BIN="${OLLAMA_BIN:-}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

cd "$PROJECT_ROOT"
mkdir -p logs outputs

if [[ ! -f "$DATA_DIR/imdb_structured_joined.csv" || ! -f "$DATA_DIR/imdb_reviews.csv" ]]; then
  echo "ERROR: missing IMDb input CSVs under $DATA_DIR." >&2
  exit 1
fi

if [[ -z "${OAR_JOB_ID:-}" ]]; then
  if ! command -v oarsub >/dev/null 2>&1; then
    echo "ERROR: run inside an OAR GPU allocation or on an Aker login node with oarsub." >&2
    exit 1
  fi
  echo "Submitting heterogen_v2 as an OAR GPU job."
  exec oarsub -S "$PROJECT_ROOT/scripts/run_gpu.sh"
fi

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: no NVIDIA GPU is visible in OAR job ${OAR_JOB_ID}." >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

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
  echo "ERROR: Ollama was not found. Set OLLAMA_BIN to its absolute path." >&2
  exit 1
fi

OLLAMA_PORT="${OLLAMA_PORT:-$((12000 + OAR_JOB_ID % 1000))}"
export OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"
API_BASE="http://$OLLAMA_HOST"
OLLAMA_LOG="$PROJECT_ROOT/logs/ollama_${OAR_JOB_ID}_${RUN_STAMP}.log"
CONSOLE_LOG="$PROJECT_ROOT/logs/cascade_${OAR_JOB_ID}_${RUN_STAMP}.console.log"
OUTPUT_DIR="$PROJECT_ROOT/outputs/cascade_${RUN_STAMP}"

ollama_pid=""
cleanup() {
  if [[ -n "$ollama_pid" ]] && kill -0 "$ollama_pid" >/dev/null 2>&1; then
    kill "$ollama_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

OLLAMA_HOST="$OLLAMA_HOST" nohup "$OLLAMA_BIN" serve >"$OLLAMA_LOG" 2>&1 &
ollama_pid="$!"
ollama_ready=0
for _ in $(seq 1 90); do
  if python3 - "$API_BASE" <<'PY'
import sys, urllib.request
try:
    urllib.request.urlopen(sys.argv[1] + "/api/tags", timeout=2)
except Exception:
    raise SystemExit(1)
PY
  then
    ollama_ready=1
    break
  fi
  sleep 2
done
if [[ "$ollama_ready" != "1" ]]; then
  echo "ERROR: Ollama did not become ready. See $OLLAMA_LOG" >&2
  exit 1
fi

for model in "$CHEAP_MODEL" "$EXPENSIVE_MODEL"; do
  if [[ "$PULL_MODELS" == "1" ]]; then
    OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" pull "$model"
  elif ! OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" show "$model" >/dev/null 2>&1; then
    echo "ERROR: model '$model' is missing. Resubmit with PULL_MODELS=1." >&2
    exit 1
  fi
done

python3 -u run_use_case3.py \
  --data-dir "$DATA_DIR" \
  --api-base "$API_BASE" \
  --cheap-model "$CHEAP_MODEL" \
  --expensive-model "$EXPENSIVE_MODEL" \
  --cheap-accept-threshold "$CHEAP_ACCEPT_THRESHOLD" \
  --cheap-reject-threshold "$CHEAP_REJECT_THRESHOLD" \
  --expensive-batch-size "$EXPENSIVE_BATCH_SIZE" \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$CONSOLE_LOG"

echo "Outputs: $OUTPUT_DIR"
echo "Console log: $CONSOLE_LOG"
echo "Ollama log: $OLLAMA_LOG"
