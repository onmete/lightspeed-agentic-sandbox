#!/bin/bash
# Start one container per provider, wait for /health, run a command, tear down.
#
# Usage: bash scripts/start-containers.sh <test-dir> [pytest args...]
#
# Exports EVAL_SERVER_URLS and EVAL_WORKSPACES for pytest fixtures.
# The first positional arg is the test directory; the rest are forwarded to pytest.

set -euo pipefail

TEST_DIR="${1:?Usage: start-containers.sh <test-dir> [pytest args...]}"
shift

# Source config.env from the test directory if present (sets model defaults etc.)
# Existing env vars take precedence — config.env only fills in unset vars.
CONFIG_ENV="$(pwd)/${TEST_DIR}/config.env"
if [ -f "$CONFIG_ENV" ]; then
    while IFS='=' read -r key value; do
        key="${key%%#*}"
        key="${key// /}"
        [ -z "$key" ] && continue
        value="${value%%#*}"
        value="${value// /}"
        if [ -z "${!key:-}" ]; then
            export "$key=$value"
        fi
    done < "$CONFIG_ENV"
fi

IMAGE="${IMAGE:-lightspeed-agentic-sandbox:latest}"
RUNTIME="${CONTAINER_RUNTIME:-$(command -v podman 2>/dev/null || command -v docker 2>/dev/null)}"
BASE_PORT=18080

GCLOUD_ADC="$HOME/.config/gcloud/application_default_credentials.json"
GCLOUD_MOUNT=""
if [ -f "$GCLOUD_ADC" ]; then
    GCLOUD_MOUNT="-v $GCLOUD_ADC:/tmp/gcloud-adc.json:ro,Z -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcloud-adc.json"
fi

PROVIDERS=("claude" "gemini" "openai" "deepagents" "deepagents-gemini" "deepagents-openai")
CONTAINERS=()
WORKDIRS=()
OUTDIRS=()

WORKSPACE_SRC=""
if [ -d "$(pwd)/evals/workspace" ]; then
    WORKSPACE_SRC="$(pwd)/evals/workspace"
fi
if [ -d "$(pwd)/tests/e2e/workspace" ] && [[ "$TEST_DIR" == *e2e* ]]; then
    WORKSPACE_SRC="$(pwd)/tests/e2e/workspace"
fi
if [ -z "$WORKSPACE_SRC" ]; then
    WORKSPACE_SRC="$(pwd)/evals/workspace"
fi

cleanup() {
    for i in "${!PROVIDERS[@]}"; do
        name="${PROVIDERS[$i]}"
        outdir="$(pwd)/.eval-workspaces/output-${name}"
        $RUNTIME logs "eval-${name}" > "${outdir}/container.log" 2>&1 || true
        $RUNTIME stop "eval-${name}" 2>/dev/null || true
        $RUNTIME rm -f "eval-${name}" 2>/dev/null || true
    done
    for d in "${WORKDIRS[@]}"; do
        rm -rf "$d" 2>/dev/null || true
    done
}
trap cleanup EXIT

provider_env() {
    case "$1" in
        deepagents-gemini|deepagents-openai) echo "deepagents" ;;
        *) echo "$1" ;;
    esac
}

model_env() {
    case "$1" in
        claude)            echo "-e ANTHROPIC_MODEL=${ANTHROPIC_MODEL:-claude-sonnet-4-6}" ;;
        gemini)            echo "-e GEMINI_MODEL=${GEMINI_MODEL:-gemini-3.1-pro-preview}" ;;
        openai)            echo "-e OPENAI_MODEL=${OPENAI_MODEL:-gpt-5.4}" ;;
        deepagents)        echo "-e DEEPAGENTS_MODEL=${DEEPAGENTS_MODEL:-claude-opus-4-6}" ;;
        deepagents-gemini) echo "-e DEEPAGENTS_MODEL=${DEEPAGENTS_GEMINI_MODEL:-gemini-3.1-pro-preview}" ;;
        deepagents-openai) echo "-e DEEPAGENTS_MODEL=${DEEPAGENTS_OPENAI_MODEL:-gpt-5.4}" ;;
    esac
}

echo "Starting provider containers..."

mkdir -p "$(pwd)/.eval-workspaces"

for i in "${!PROVIDERS[@]}"; do
    name="${PROVIDERS[$i]}"
    port=$((BASE_PORT + i))
    agent_provider=$(provider_env "$name")
    workdir=$(mktemp -d "$(pwd)/.eval-workspaces/eval-${name}-XXXXXX")
    outdir="$(pwd)/.eval-workspaces/output-${name}"
    mkdir -p "$outdir"
    WORKDIRS+=("$workdir")
    OUTDIRS+=("$outdir")
    cp -r "${WORKSPACE_SRC}/skills" "$workdir/skills"
    cp -r "${WORKSPACE_SRC}/tools" "$workdir/tools"
    mkdir -p "$workdir/.claude"
    cp -r "${WORKSPACE_SRC}/skills" "$workdir/.claude/skills"
    chmod -R 777 "$workdir" "$outdir"

    cid=$($RUNTIME run -d --rm \
        --name "eval-${name}" \
        -p "${port}:8080" \
        -v "${workdir}:/app/workspace:Z" \
        -v "${outdir}:/app/eval-output:Z" \
        -e EVAL_OUTPUT_DIR="/app/eval-output" \
        $GCLOUD_MOUNT \
        -e LIGHTSPEED_AGENT_PROVIDER="$agent_provider" \
        -e LIGHTSPEED_SKILLS_DIR="/app/workspace" \
        -e ANTHROPIC_API_KEY \
        -e CLAUDE_CODE_USE_VERTEX \
        -e ANTHROPIC_VERTEX_PROJECT_ID \
        -e CLOUD_ML_REGION \
        -e GOOGLE_API_KEY \
        -e GEMINI_API_KEY \
        -e OPENAI_API_KEY \
        -e OPENAI_BASE_URL \
        -e AWS_ACCESS_KEY_ID \
        -e AWS_SECRET_ACCESS_KEY \
        -e AWS_REGION \
        $(model_env "$name") \
        "$IMAGE")

    CONTAINERS+=("$cid")
    echo "  ${name}: port ${port} (container ${cid:0:12})"
done

echo "Waiting for servers..."
WAIT_PIDS=()
for i in "${!PROVIDERS[@]}"; do
    name="${PROVIDERS[$i]}"
    port=$((BASE_PORT + i))
    (
        for attempt in $(seq 1 30); do
            if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
                echo "  ${name}: ready"
                exit 0
            fi
            sleep 1
        done
        echo "  ${name}: FAILED to start (timeout)"
        $RUNTIME logs "eval-${name}" 2>&1 | tail -10
        exit 1
    ) &
    WAIT_PIDS+=($!)
done
for pid in "${WAIT_PIDS[@]}"; do
    wait "$pid" || exit 1
done

SERVER_URLS=""
WORKSPACE_MAP=""
for i in "${!PROVIDERS[@]}"; do
    name="${PROVIDERS[$i]}"
    port=$((BASE_PORT + i))
    SERVER_URLS="${SERVER_URLS}${name}=http://localhost:${port},"
    WORKSPACE_MAP="${WORKSPACE_MAP}${name}=${OUTDIRS[$i]},"
done

echo ""
echo "Running tests from ${TEST_DIR}..."
PYTEST="${PYTEST:-python3 -m pytest}"

export EVAL_SERVER_URLS="$SERVER_URLS"
export EVAL_WORKSPACES="$WORKSPACE_MAP"
$PYTEST "$TEST_DIR" "$@"
