#!/usr/bin/env bash
# Run any canonical suite on a local Ollama server after a fresh repository clone.

set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
SUITE="1q"
REPETITIONS=1
METHODS="suql_baseline suql_v1 trummer_baseline trummer_v1"
CHEAP_MODEL="gemma4:e2b"
EXPENSIVE_MODEL="gemma4:e4b"
API_BASE="${OLLAMA_API_BASE:-http://127.0.0.1:11434}"
OUTPUT_NAME=""
PULL_MODELS=0
PARALLEL_WORKERS="${PARALLEL_WORKERS:-1}"

usage() {
  cat <<'EOF'
Usage: benchmarks/run_local.sh [options]

Options:
  --suite {1q|3q|5q|10q}       Benchmark suite (default: 1q)
  --repetitions N              Runs per question and method (default: 1)
  --methods "METHOD ..."        Methods to run (default: all four)
  --cheap-model MODEL          Ollama cascade model (default: gemma4:e2b)
  --expensive-model MODEL      Ollama baseline/fallback model (default: gemma4:e4b)
  --api-base URL               Existing Ollama API (default: http://127.0.0.1:11434)
  --parallel-workers N         Concurrent repetitions (default: 1)
  --output-name NAME           Directory name below benchmarks/SUITE/outputs
  --pull-models                Pull missing Ollama models before running
  -h, --help                   Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite) SUITE="$2"; shift 2 ;;
    --repetitions) REPETITIONS="$2"; shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    --cheap-model) CHEAP_MODEL="$2"; shift 2 ;;
    --expensive-model) EXPENSIVE_MODEL="$2"; shift 2 ;;
    --api-base) API_BASE="$2"; shift 2 ;;
    --parallel-workers) PARALLEL_WORKERS="$2"; shift 2 ;;
    --output-name) OUTPUT_NAME="$2"; shift 2 ;;
    --pull-models) PULL_MODELS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$SUITE" in 1q|3q|5q|10q) ;; *) echo "ERROR: invalid suite '$SUITE'" >&2; exit 2 ;; esac
[[ "$REPETITIONS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: repetitions must be positive" >&2; exit 2; }
[[ "$PARALLEL_WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: parallel-workers must be positive" >&2; exit 2; }

# Keep dependencies isolated and make repeated invocations idempotent.
if [[ ! -x "$REPO_ROOT/.venv/bin/python" ]]; then
  python3 -m venv "$REPO_ROOT/.venv"
fi
PYTHON="$REPO_ROOT/.venv/bin/python"
"$PYTHON" -m pip install -r "$HERE/requirements.txt"

# run_all accepts OpenAI-style model names, while the local server stores bare tags.
if [[ "$PULL_MODELS" == "1" ]]; then
  command -v ollama >/dev/null 2>&1 || { echo "ERROR: Ollama is not installed" >&2; exit 1; }
  ollama pull "${CHEAP_MODEL#ollama/}"
  ollama pull "${EXPENSIVE_MODEL#ollama/}"
fi

"$PYTHON" - "$API_BASE" "${CHEAP_MODEL#ollama/}" "${EXPENSIVE_MODEL#ollama/}" <<'PY'
import json
import sys
import urllib.error
import urllib.request

api_base, cheap, expensive = sys.argv[1:]
try:
    with urllib.request.urlopen(api_base.rstrip("/") + "/api/tags", timeout=5) as response:
        models = json.load(response).get("models", [])
except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
    raise SystemExit(f"ERROR: Ollama is not reachable at {api_base}: {exc}")

available = {item.get("name", "") for item in models} | {item.get("model", "") for item in models}
missing = [model for model in (cheap, expensive) if model not in available]
if missing:
    raise SystemExit("ERROR: missing Ollama model(s): " + ", ".join(missing) + "; rerun with --pull-models")
PY

if [[ -z "$OUTPUT_NAME" ]]; then
  OUTPUT_NAME="local_${SUITE}_${REPETITIONS}reps_$(date +%Y%m%d_%H%M%S)"
fi
SUITE_ROOT="$HERE/$SUITE"
export BENCHMARK_SUITE_ROOT="$SUITE_ROOT"
export PARALLEL_WORKERS
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# shellcheck disable=SC2206 # METHODS intentionally expands a documented space-separated list.
METHOD_ARGS=($METHODS)
exec "$PYTHON" -u "$HERE/shared/scripts/run_all.py" \
  --python "$PYTHON" \
  --api-base "$API_BASE" \
  --cheap-model "ollama/${CHEAP_MODEL#ollama/}" \
  --expensive-model "ollama/${EXPENSIVE_MODEL#ollama/}" \
  --repetitions "$REPETITIONS" \
  --methods "${METHOD_ARGS[@]}" \
  --output-dir "$SUITE_ROOT/outputs/$OUTPUT_NAME"
