#!/usr/bin/env bash
# Run sandbox BDD integration tests for a single LLM provider.
#
# Called by the Konflux sandbox-integration-test-pipeline after
# cloning the repo at the SNAPSHOT revision.
#
# Usage:
#   bash .tekton/integration-tests/scripts/run-sandbox-integration-tests.sh <provider>
#
# Provider ids:
#   anthropic-vertex-deepagents | gemini-vertex-adk | openai-agents
#
# Expects:
#   - Provider credentials mounted at /var/run/credentials/token
#   - ARTIFACT_DIR set (for junit XML output)
#   - Working directory is the sandbox repo root

set -euo pipefail
trap 'echo "error: $0 line $LINENO: command \"$BASH_COMMAND\" exited with status $?" >&2' ERR

PROVIDER="${1:?Usage: $0 <provider> (anthropic-vertex-deepagents|gemini-vertex-adk|openai-agents)}"
CRED_PATH="/var/run/credentials/token"

if [ ! -f "${CRED_PATH}" ]; then
    echo "error: credential file not found at ${CRED_PATH}" >&2
    exit 1
fi

# --- Set up provider credentials ---
case "${PROVIDER}" in
  anthropic-vertex-deepagents)
    mkdir -p "${HOME}/.config/gcloud"
    cp "${CRED_PATH}" "${HOME}/.config/gcloud/application_default_credentials.json"
    export GOOGLE_APPLICATION_CREDENTIALS="${CRED_PATH}"
    export CLAUDE_CODE_USE_VERTEX=1
    ANTHROPIC_VERTEX_PROJECT_ID=$(python3 -c "import json; print(json.load(open('${CRED_PATH}'))['project_id'])")
    export ANTHROPIC_VERTEX_PROJECT_ID
    ;;
  gemini-vertex-adk)
    mkdir -p "${HOME}/.config/gcloud"
    cp "${CRED_PATH}" "${HOME}/.config/gcloud/application_default_credentials.json"
    export GOOGLE_APPLICATION_CREDENTIALS="${CRED_PATH}"
    GOOGLE_CLOUD_PROJECT=$(python3 -c "import json; print(json.load(open('${CRED_PATH}'))['project_id'])")
    export GOOGLE_CLOUD_PROJECT
    ;;
  openai-agents)
    OPENAI_API_KEY=$(cat "${CRED_PATH}")
    export OPENAI_API_KEY
    ;;
  *)
    echo "error: unknown provider: ${PROVIDER}" >&2
    exit 1
    ;;
esac

# --- Install uv if not present ---
UV_VERSION="0.11.19"
if ! command -v uv >/dev/null 2>&1; then
    max_attempts=3
    delay=5
    for attempt in $(seq 1 "${max_attempts}"); do
        if pip install --quiet "uv==${UV_VERSION}"; then
            break
        fi
        if [ "${attempt}" -eq "${max_attempts}" ]; then
            echo "error: failed to install uv==${UV_VERSION} after ${max_attempts} attempts" >&2
            exit 1
        fi
        echo "warning: pip install attempt ${attempt}/${max_attempts} failed, retrying in ${delay}s..." >&2
        sleep "${delay}"
        delay=$((delay * 2))
    done
fi
# --- Run BDD tests in host mode ---
ARTIFACT_DIR="${ARTIFACT_DIR:-/workspace/artifacts}"
mkdir -p "${ARTIFACT_DIR}"
export E2E_ARGS="--junitxml=${ARTIFACT_DIR}/junit_e2e.xml --tb=short"
echo "runner: credentials configured for ${PROVIDER}, starting e2e"
E2E_PROW_HOST=1 bash scripts/e2e-containers.sh "${PROVIDER}"
