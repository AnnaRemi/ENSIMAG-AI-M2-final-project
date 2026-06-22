#!/usr/bin/env bash
# Sync the baseline vs online-join experiment code and required data samples to Aker.
#
# Usage:
#   AKER_HOST="your-login@aker-host" bash scripts/sync_baseline_online_experiment_to_aker.sh
#
# Optional:
#   AKER_PROJECT_ROOT="/home/daisy/remizova/project" \
#   SIZES="200" \
#   AKER_HOST="your-login@aker-host" \
#   bash scripts/sync_baseline_online_experiment_to_aker.sh

set -Eeuo pipefail

AKER_HOST="${AKER_HOST:?Set AKER_HOST, for example: AKER_HOST='username@aker-host'}"
AKER_PROJECT_ROOT="${AKER_PROJECT_ROOT:-/home/daisy/remizova/project}"
SIZES=(${SIZES:-200})

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "Sync target: $AKER_HOST:$AKER_PROJECT_ROOT"
echo "Sample sizes: ${SIZES[*]}"

rsync -avR \
  benchmark_compare.py \
  requirements.txt \
  src_baseline/ \
  src_online_join/ \
  scripts/run_aker_baseline_online_data_samples.sh \
  scripts/run_aker_baseline_online_model_sweep.sh \
  scripts/aggregate_online_join_scaling_svg.py \
  scripts/plot_baseline_online_question_metrics_svg.py \
  scripts/run_baseline_online_model_sweep.py \
  "$AKER_HOST:$AKER_PROJECT_ROOT/"

for size in "${SIZES[@]}"; do
  sample_dir="data_samples/data_sample_${size}"
  if [[ ! -f "$sample_dir/imdb_joined.csv" ]]; then
    echo "ERROR: missing local $sample_dir/imdb_joined.csv" >&2
    exit 1
  fi

  rsync -avR "$sample_dir/" "$AKER_HOST:$AKER_PROJECT_ROOT/"
done

echo
echo "Synced baseline vs online-join experiment files."
echo "On Aker, run:"
echo "  cd $AKER_PROJECT_ROOT"
echo "  SUQL_MODEL=ollama/gemma2:2b bash scripts/run_aker_baseline_online_data_samples.sh"
