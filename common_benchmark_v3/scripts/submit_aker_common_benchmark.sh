#!/usr/bin/env bash
# Run on the Aker login node.

set -Eeuo pipefail

AKER_ROOT="${AKER_ROOT:-/home/daisy/remizova/common_benchmark_v3_workspace}"
CHEAP_MODEL="${CHEAP_MODEL:-gemma2:2b}"
EXPENSIVE_MODELS="${EXPENSIVE_MODELS:-qwen2.5:3b}"
PULL_MODELS="${PULL_MODELS:-0}"
CHEAP_ACCEPT_THRESHOLD="${CHEAP_ACCEPT_THRESHOLD:-3.0}"
CHEAP_REJECT_THRESHOLD="${CHEAP_REJECT_THRESHOLD:--1.5}"
EXPENSIVE_BATCH_SIZE="${EXPENSIVE_BATCH_SIZE:-8}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"
WALLTIME="${WALLTIME:-08:00:00}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
WORKER="$AKER_ROOT/common_benchmark_v3/scripts/run_aker_common_benchmark.sh"
JOBS="$AKER_ROOT/common_benchmark_v3/jobs"
LOGS="$AKER_ROOT/common_benchmark_v3/logs"
WRAPPER="$JOBS/common_benchmark_v3_${RUN_STAMP}.sh"

if [[ ! -x "$WORKER" ]]; then
  echo "ERROR: missing worker $WORKER; sync from the local Mac first." >&2
  exit 1
fi
mkdir -p "$JOBS" "$LOGS"
shell_quote() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }
{
  echo "#!/usr/bin/env bash"
  echo "set -Eeuo pipefail"
  for name in AKER_ROOT CHEAP_MODEL EXPENSIVE_MODELS PULL_MODELS CHEAP_ACCEPT_THRESHOLD CHEAP_REJECT_THRESHOLD EXPENSIVE_BATCH_SIZE REQUEST_TIMEOUT RUN_STAMP; do
    printf "export %s=%s\n" "$name" "$(shell_quote "${!name}")"
  done
  printf "exec bash %s\n" "$(shell_quote "$WORKER")"
} >"$WRAPPER"
chmod 700 "$WRAPPER"

oarsub \
  -n common-benchmark-v3 \
  -l "/nodes=1/gpu=1,walltime=$WALLTIME" \
  -O "$LOGS/oar_%jobid%.out" \
  -E "$LOGS/oar_%jobid%.err" \
  -S "$WRAPPER"

echo "Status: oarstat -u \$USER"
echo "Exact logs: oarstat -f -j <jobid>"
echo "Live stdout: tail -F $LOGS/oar_<jobid>.out"
