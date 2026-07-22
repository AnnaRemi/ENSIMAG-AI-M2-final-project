#!/usr/bin/env bash
# Convenience entry point for the three-question local suite.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/../run_local.sh" --suite 3q "$@"
