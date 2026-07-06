#!/usr/bin/env bash
# Sync the Stage 1 sample-size experiment code and required data samples to Aker.
#
# Usage:
#   AKER_HOST="your-login@aker-host" bash scripts/sync_stage1_experiment_to_aker.sh
#
# Optional:
#   AKER_PROJECT_ROOT="/home/daisy/remizova/project" \
#   SIZES="100 200 500 1000 1500" \
#   AKER_HOST="your-login@aker-host" \
#   bash scripts/sync_stage1_experiment_to_aker.sh

set -Eeuo pipefail

AKER_HOST="${AKER_HOST:?Set AKER_HOST, for example: AKER_HOST='username@aker-host'}"
AKER_PROJECT_ROOT="${AKER_PROJECT_ROOT:-/home/daisy/remizova/project}"
SIZES=(${SIZES:-100 200 500 1000 1500})

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "Sync target: $AKER_HOST:$AKER_PROJECT_ROOT"
echo "Sample sizes: ${SIZES[*]}"

rsync -avR \
  Stage_1/benchmark_stage1.py \
  Stage_1/README.md \
  Stage_1/thresholds.json \
  requirements.txt \
  src_baseline/ \
  src_baseline_stage1/ \
  scripts/run_aker_baseline_stage1_data_samples.sh \
  scripts/plot_benchmarks.py \
  scripts/run_suql.py \
  "$AKER_HOST:$AKER_PROJECT_ROOT/"

for size in "${SIZES[@]}"; do
  sample_path="data_samples/data_sample_${size}/imdb_joined.csv"
  if [[ ! -f "$sample_path" ]]; then
    echo "ERROR: missing local $sample_path" >&2
    exit 1
  fi

  rsync -avR "$sample_path" "$AKER_HOST:$AKER_PROJECT_ROOT/"
done

echo
echo "Synced Stage 1 experiment files."
echo "On Aker, run:"
echo "  cd $AKER_PROJECT_ROOT"
echo "  bash scripts/run_aker_baseline_stage1_data_samples.sh"
