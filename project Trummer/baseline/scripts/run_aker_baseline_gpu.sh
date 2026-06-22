#!/usr/bin/env bash
#OAR -n trummer-baseline-gpu
#OAR -l /nodes=1/gpu=1,walltime=08:00:00
#OAR -O /home/daisy/remizova/project_Trummer/baseline/logs/oar_%jobid%.out
#OAR -E /home/daisy/remizova/project_Trummer/baseline/logs/oar_%jobid%.err

set -Eeuo pipefail

ROOT="${ROOT:-/home/daisy/remizova/project_Trummer/baseline}"
PY="${PY:-/home/daisy/remizova/project/.venv/bin/python}"
OLLAMA_BIN="${OLLAMA_BIN:-/home/daisy/remizova/.local/ollama/bin/ollama}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11435}"
MODEL="${MODEL:-gemma2:2b}"
TOKEN_LIMIT="${TOKEN_LIMIT:-1000}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
AUTO_OAR_SUBMIT="${AUTO_OAR_SUBMIT:-1}"

cd "$ROOT"
mkdir -p logs outputs

if [[ -z "${OAR_JOB_ID:-}" && "$AUTO_OAR_SUBMIT" == "1" ]]; then
  echo "Submitting baseline as an OAR GPU job."
  exec oarsub -S "$ROOT/scripts/run_aker_baseline_gpu.sh"
fi

if ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: no NVIDIA GPU is visible in this job." >&2
  exit 1
fi
if [[ ! -x "$OLLAMA_BIN" ]]; then
  echo "ERROR: Ollama not found at $OLLAMA_BIN" >&2
  exit 1
fi
if [[ ! -x "$PY" ]]; then
  echo "ERROR: Python not found at $PY" >&2
  exit 1
fi

echo "OAR job: ${OAR_JOB_ID:-unknown}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

export OLLAMA_HOST
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OLLAMA_CONTEXT_LENGTH=2048
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1

OLLAMA_LOG="$ROOT/logs/ollama_baseline_${RUN_STAMP}.log"
RUN_DIR="$ROOT/outputs/block_gemma2_gpu_${RUN_STAMP}"
CONSOLE_LOG="$ROOT/outputs/block_gemma2_gpu_${RUN_STAMP}.console.log"
mkdir -p "$RUN_DIR"

"$OLLAMA_BIN" serve >"$OLLAMA_LOG" 2>&1 &
ollama_pid=$!
cleanup() {
  kill "$ollama_pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 90); do
  if "$PY" -c "import urllib.request; urllib.request.urlopen('http://$OLLAMA_HOST/api/tags', timeout=2)" \
      >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if ! "$PY" -c "import urllib.request; urllib.request.urlopen('http://$OLLAMA_HOST/api/tags', timeout=2)" \
    >/dev/null 2>&1; then
  echo "ERROR: Ollama did not start. See $OLLAMA_LOG" >&2
  exit 1
fi

"$OLLAMA_BIN" pull "$MODEL"

# Force model loading before checking its processor placement.
"$PY" - <<PY
import json
import urllib.request

payload = {
    "model": "$MODEL",
    "prompt": "Reply with OK.",
    "stream": False,
    "keep_alive": "30m",
    "options": {"num_predict": 2, "num_ctx": 2048},
}
request = urllib.request.Request(
    "http://$OLLAMA_HOST/api/generate",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=300) as response:
    print(response.read().decode())
PY

echo "Loaded model placement:"
"$OLLAMA_BIN" ps

model_status=$("$OLLAMA_BIN" ps | awk -v model="$MODEL" '$1 == model {print}')
if [[ "$model_status" == *"100% GPU"* ]]; then
  echo "Verified: $MODEL is fully loaded on GPU."
else
  echo "WARNING: $MODEL is not fully on GPU (status='$model_status')." >&2
  echo "Continuing like Heterogen_v1; execution may be slower on CPU." >&2
fi
nvidia-smi

"$PY" -u run_experiment.py \
  --operator block \
  --backend llm \
  --api-base "http://$OLLAMA_HOST" \
  --model "ollama/$MODEL" \
  --token-limit "$TOKEN_LIMIT" \
  --timeout 900 \
  --output-dir "$RUN_DIR" \
  2>&1 | tee "$CONSOLE_LOG"

echo "Results: $RUN_DIR"
echo "Console: $CONSOLE_LOG"
echo "Ollama: $OLLAMA_LOG"
