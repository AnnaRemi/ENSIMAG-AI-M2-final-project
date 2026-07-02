#!/usr/bin/env bash
# Run on the Aker login node after syncing from the local Mac.
# Reruns only row-wise V2 with a manual confidence threshold, then re-evaluates.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTPUT_NAME="${OUTPUT_NAME:-heterogen_comparing}"
MANUAL_V2_CONFIDENCE_THRESHOLD="${MANUAL_V2_CONFIDENCE_THRESHOLD:-2.0}"
CALIBRATION_BUDGET="${CALIBRATION_BUDGET:-0}"
REPETITIONS="${REPETITIONS:-9}"
PULL_MODELS="${PULL_MODELS:-0}"
SKIP_V1=1
SKIP_V2=0
SKIP_V2_2=1
SKIP_V2_3=1
SKIP_V3=1

export OUTPUT_NAME
export MANUAL_V2_CONFIDENCE_THRESHOLD
export CALIBRATION_BUDGET
export REPETITIONS
export PULL_MODELS
export SKIP_V1 SKIP_V2 SKIP_V2_2 SKIP_V2_3 SKIP_V3

exec bash "$SCRIPT_DIR/submit_aker_all_heterogen.sh"
