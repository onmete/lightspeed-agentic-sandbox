#!/bin/bash
# Run evals against live container servers — one per provider.
# Usage: bash evals/run.sh [pytest args...]
#
# Thin wrapper around scripts/start-containers.sh for backward compatibility.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${SCRIPT_DIR}/scripts/start-containers.sh" evals/ -v "$@"
