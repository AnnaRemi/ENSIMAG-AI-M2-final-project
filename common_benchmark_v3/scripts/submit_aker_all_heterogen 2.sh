#!/usr/bin/env bash
# Run on the Aker login node after syncing from the local Mac.

set -Eeuo pipefail

AKER_ROOT="${AKER_ROOT:-/home/daisy/remizova/common_benchmark_v3_workspace}"
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
WALLTIME="${WALLTIME:-08:00:00}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_NAME="${OUTPUT_NAME:-all_heterogen_${RUN_STAMP}}"
SKIP_V1="${SKIP_V1:-0}"
SKIP_V2="${SKIP_V2:-0}"
SKIP_V2_2="${SKIP_V2_2:-0}"
SKIP_V2_3="${SKIP_V2_3:-0}"
SKIP_V3="${SKIP_V3:-0}"

WORKER="$AKER_ROOT/common_benchmark_v3/scripts/run_aker_all_heterogen.sh"
JOBS="$AKER_ROOT/common_benchmark_v3/jobs"
LOGS="$AKER_ROOT/common_benchmark_v3/logs"
WRAPPER="$JOBS/all_heterogen_${RUN_STAMP}.sh"

if [[ ! -x "$WORKER" ]]; then
  echo "ERROR: missing worker $WORKER; sync from the local Mac first." >&2
  exit 1
fi
mkdir -p "$JOBS" "$LOGS"
shell_quote() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }
{
  echo "#!/usr/bin/env bash"
  echo "set -Eeuo pipefail"
  for name in \
    AKER_ROOT CHEAP_MODEL EXPENSIVE_MODEL PULL_MODELS \
    CASCADE_TARGET CALIBRATION_BUDGET MANUAL_V2_CONFIDENCE_THRESHOLD CHEAP_BATCH_SIZE \
    EXPENSIVE_BATCH_SIZE V2_3_EXPENSIVE_BATCH_SIZE MAX_EXPENSIVE_CALLS \
    PARALLEL_WORKERS REQUEST_TIMEOUT TOKEN_THRESHOLD MAX_COMPLETION_TOKENS \
    MAX_MOVIE_BLOCK_SIZE MAX_REVIEW_BLOCK_SIZE REPETITIONS REQUIRE_GPU RUN_STAMP OUTPUT_NAME \
    SKIP_V1 SKIP_V2 SKIP_V2_2 SKIP_V2_3 SKIP_V3
  do
    printf "export %s=%s\n" "$name" "$(shell_quote "${!name}")"
  done
  printf "exec bash %s\n" "$(shell_quote "$WORKER")"
} >"$WRAPPER"
chmod 700 "$WRAPPER"

oarsub \
  -n all-heterogen-one-question \
  -l "/nodes=1/gpu=1,walltime=$WALLTIME" \
  -O "$LOGS/all_heterogen_oar_%jobid%.out" \
  -E "$LOGS/all_heterogen_oar_%jobid%.err" \
  -S "$WRAPPER"

echo "Output name: $OUTPUT_NAME"
echo "Status: oarstat -u \$USER"
echo "Exact paths: oarstat -f -j <jobid>"
echo "Live stdout: tail -F $LOGS/all_heterogen_oar_<jobid>.out"
