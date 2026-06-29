#!/usr/bin/env bash
# Run this script on the Aker login node. It submits a non-interactive OAR
# batch job and returns immediately after printing the job id.

set -Eeuo pipefail

AKER_ROOT="${AKER_ROOT:-/home/daisy/remizova/common_benchmark_v2_workspace}"
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
WALLTIME="${WALLTIME:-12:00:00}"
OLLAMA_BIN="${OLLAMA_BIN:-}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

WORKER="$AKER_ROOT/common_benchmark_v2/scripts/run_aker_common_benchmark.sh"
JOBS_DIR="$AKER_ROOT/common_benchmark_v2/jobs"
LOGS_DIR="$AKER_ROOT/common_benchmark_v2/logs"
WRAPPER="$JOBS_DIR/common_benchmark_v2_${RUN_STAMP}.sh"

if ! command -v oarsub >/dev/null 2>&1; then
  echo "ERROR: oarsub is unavailable. Run this script on the Aker login node." >&2
  exit 1
fi
if [[ ! -x "$WORKER" ]]; then
  echo "ERROR: missing or non-executable worker: $WORKER" >&2
  echo "Run common_benchmark_v2/scripts/sync_common_benchmark_to_aker.sh again from the local Mac." >&2
  exit 1
fi

mkdir -p "$JOBS_DIR" "$LOGS_DIR"

shell_quote() {
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

{
  echo "#!/usr/bin/env bash"
  echo "set -Eeuo pipefail"
  printf "export AKER_ROOT=%s\n" "$(shell_quote "$AKER_ROOT")"
  printf "export MODELS=%s\n" "$(shell_quote "$MODELS")"
  printf "export PULL_MODELS=%s\n" "$(shell_quote "$PULL_MODELS")"
  printf "export SKIP_SUQL=%s\n" "$(shell_quote "$SKIP_SUQL")"
  printf "export SKIP_TRUMMER=%s\n" "$(shell_quote "$SKIP_TRUMMER")"
  printf "export TRUMMER_REQUEST_TIMEOUT=%s\n" "$(shell_quote "$TRUMMER_REQUEST_TIMEOUT")"
  printf "export TRUMMER_TOKEN_THRESHOLD=%s\n" "$(shell_quote "$TRUMMER_TOKEN_THRESHOLD")"
  printf "export TRUMMER_MAX_COMPLETION_TOKENS=%s\n" "$(shell_quote "$TRUMMER_MAX_COMPLETION_TOKENS")"
  printf "export TRUMMER_MAX_MOVIE_BLOCK_SIZE=%s\n" "$(shell_quote "$TRUMMER_MAX_MOVIE_BLOCK_SIZE")"
  printf "export TRUMMER_MAX_REVIEW_BLOCK_SIZE=%s\n" "$(shell_quote "$TRUMMER_MAX_REVIEW_BLOCK_SIZE")"
  printf "export REPETITIONS=%s\n" "$(shell_quote "$REPETITIONS")"
  printf "export RUN_STAMP=%s\n" "$(shell_quote "$RUN_STAMP")"
  printf "export OLLAMA_BIN=%s\n" "$(shell_quote "$OLLAMA_BIN")"
  printf "exec bash %s\n" "$(shell_quote "$WORKER")"
} > "$WRAPPER"
chmod 700 "$WRAPPER"

echo "Submitting non-interactive GPU batch job."
echo "Models: $MODELS"
echo "Skip SUQL: $SKIP_SUQL"
echo "Skip Trummer: $SKIP_TRUMMER"
echo "Trummer request timeout: $TRUMMER_REQUEST_TIMEOUT seconds"
echo "Trummer token threshold: $TRUMMER_TOKEN_THRESHOLD"
echo "Trummer max completion tokens: $TRUMMER_MAX_COMPLETION_TOKENS"
echo "Trummer max block sizes: movies=$TRUMMER_MAX_MOVIE_BLOCK_SIZE reviews=$TRUMMER_MAX_REVIEW_BLOCK_SIZE"
echo "Repetitions: $REPETITIONS"
echo "Walltime: $WALLTIME"
echo "Persistent wrapper: $WRAPPER"

oarsub \
  -n "common-benchmark-v2" \
  -l "/nodes=1/gpu=1,walltime=$WALLTIME" \
  -O "$LOGS_DIR/oar_%jobid%.out" \
  -E "$LOGS_DIR/oar_%jobid%.err" \
  -S "$WRAPPER"

echo
echo "The job is now managed by OAR; closing this terminal will not stop it."
echo "Status: oarstat -u \$USER"
echo "For exact log paths: oarstat -f -j <jobid>"
echo "Live stdout: tail -F $LOGS_DIR/oar_<jobid>.out"
echo "Live stderr: tail -F $LOGS_DIR/oar_<jobid>.err"
