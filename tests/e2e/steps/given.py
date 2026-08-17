"""Given steps — service, schemas, prompts."""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Any

from pytest_bdd import given

from credentials import require_credentials
from schemas_contract import (
    CONTEXT_APPROVED_OPTION_ECHO_SCHEMA,
    CONTEXT_NAMESPACES_ECHO_SCHEMA,
    CONTEXT_PREVIOUS_ATTEMPTS_ECHO_SCHEMA,
    ECHO_TOKEN_SCHEMA,
    FLAT_OUTPUT_SCHEMA,
    MCP_TOOL_OUTPUT_SCHEMA,
    NESTED_OUTPUT_SCHEMA,
    STRICT_CONFLICT_SCHEMA,
)


@given("provider credentials are configured")
def provider_credentials_configured(provider_name: str) -> None:
    require_credentials(provider_name)


@given("the sandbox service is running")
def sandbox_running(server_url: str) -> None:
    assert server_url.startswith("http"), f"unexpected server URL: {server_url!r}"


@given("the sandbox service is running with skills")
def sandbox_running_with_skills(server_url: str, e2e_output_dir: Path | None) -> None:
    assert server_url.startswith("http"), f"unexpected server URL: {server_url!r}"
    assert e2e_output_dir is not None, (
        "E2E_OUTPUT_DIR not set — skills not mounted (run via scripts/e2e-containers.sh)"
    )


@given("a simple non-skill query has been prepared")
def prepare_simple_non_skill(bdd_context: dict[str, Any]) -> None:
    bdd_context["query"] = (
        "In one sentence, name any primary color. "
        "Answer with plain text only. Do not call any tools."
    )
    bdd_context["output_schema"] = None


@given("a query that will exceed the timeout has been prepared")
def prepare_timeout_query(bdd_context: dict[str, Any], provider_name: str) -> None:
    """Force multi-step tool work that guarantees multiple LLM roundtrips.

    Known issue: the OpenAI agents SDK swallows asyncio.CancelledError internally,
    so asyncio.wait_for cannot enforce sub-second timeouts. Gemini's ADK has a
    similar behavior. Skip for affected providers until the timeout mechanism is
    reworked (tracked separately).
    """
    import pytest

    broken_timeout_providers = {"openai-agents", "gemini-vertex-adk"}
    if provider_name in broken_timeout_providers:
        pytest.skip(
            f"{provider_name} SDK does not propagate asyncio cancellation — "
            "timeout_ms cannot be tested via asyncio.wait_for"
        )

    bdd_context["query"] = (
        "You MUST use tools to complete this task. "
        "Step 1: Use apply_patch to create a file /tmp/timeout_test.txt with content 'hello'. "
        "Step 2: Read the file back to verify it. "
        "Step 3: Use apply_patch to append ' world' to it. "
        "Step 4: Read it again and report the final content. "
        "Do NOT skip any steps. Each step requires a separate tool call."
    )


@given("a context with target namespaces and an echo output schema have been prepared")
def prepare_context_namespaces_echo(bdd_context: dict[str, Any]) -> None:
    # Unguessable per-run values: the echo can only match if context was truly
    # injected — real namespaces like "default"/"kube-system" a model could guess.
    nonce = secrets.token_hex(3)
    target_namespaces = [f"ns-{nonce}-alpha", f"ns-{nonce}-bravo"]
    bdd_context["context"] = {"targetNamespaces": target_namespaces}
    bdd_context["expected_namespaces"] = ", ".join(target_namespaces)
    bdd_context["output_schema"] = CONTEXT_NAMESPACES_ECHO_SCHEMA
    bdd_context["query"] = (
        "The user message contains a [context] block with Target namespaces. "
        "Return a single JSON object only (no markdown). "
        "Set success=true, summary='context-echo-ok', and set namespaces to the "
        "comma-separated namespace values from the 'Target namespaces:' line "
        "(values only, not the label)."
    )


@given("a context with previous attempts and an echo output schema have been prepared")
def prepare_context_previous_attempts_echo(bdd_context: dict[str, Any]) -> None:
    # Unguessable per-run failure reason: a generic value like "timeout" a model
    # could produce without any injected context.
    nonce = secrets.token_hex(3)
    first_failure_reason = f"probe-fault-{nonce}"
    bdd_context["context"] = {
        "previousAttempts": [
            {"attempt": 1, "failureReason": first_failure_reason},
            {"attempt": 2},
        ]
    }
    bdd_context["expected_first_failure_reason"] = first_failure_reason
    bdd_context["output_schema"] = CONTEXT_PREVIOUS_ATTEMPTS_ECHO_SCHEMA
    bdd_context["query"] = (
        "The user message contains a [context] block with a Previous attempts section. "
        "Return a single JSON object only (no markdown). "
        "Set success=true, summary='context-echo-ok', and set firstFailureReason to the "
        "failure reason on Attempt 1 (value only, not the label or attempt number)."
    )


_CONTEXT_APPROVED_OPTION = {
    "title": "Restart deployment",
    "diagnosis": {"rootCause": "CrashLoopBackOff"},
    "remediationPlan": {
        "description": "Roll out restart",
        "actions": [
            {
                "command": "kubectl rollout restart deploy/web -n prod",
                "type": "mutation",
                "description": "Restart the deployment",
            },
        ],
        "risk": "low",
        "reversible": True,
    },
}


@given("a context with approved option and an echo output schema have been prepared")
def prepare_context_approved_option_echo(bdd_context: dict[str, Any]) -> None:
    bdd_context["context"] = {"approvedOption": _CONTEXT_APPROVED_OPTION}
    bdd_context["expected_approved_title"] = _CONTEXT_APPROVED_OPTION["title"]
    bdd_context["expected_root_cause"] = _CONTEXT_APPROVED_OPTION["diagnosis"]["rootCause"]
    first_action = _CONTEXT_APPROVED_OPTION["remediationPlan"]["actions"][0]
    bdd_context["expected_first_command"] = first_action["command"]
    bdd_context["output_schema"] = CONTEXT_APPROVED_OPTION_ECHO_SCHEMA
    bdd_context["query"] = (
        "The user message contains a [context] block with an approved remediation section. "
        "Return a single JSON object only (no markdown). "
        "Set success=true, summary='context-echo-ok', approvedTitle to the remediation "
        "Title value, rootCause to the Diagnosis root cause value, and firstCommand to "
        "the command field of the first action in the remediation plan (values only, not labels)."
    )


@given("the echo-token skill query has been prepared")
def prepare_echo_token(bdd_context: dict[str, Any]) -> None:
    bdd_context["system_prompt"] = (
        "You are an agent with shell access. When a skill provides a script, "
        "you must run that script via the shell tool and use its stdout JSON "
        "before producing your final answer. Never invent or placeholder token values."
    )
    bdd_context["query"] = (
        "Use the echo-token skill end-to-end:\n"
        "1. Load the echo-token skill.\n"
        "2. From the skill directory, run: bash scripts/echo-token.sh\n"
        "3. Parse the JSON printed to stdout.\n"
        "4. Reply with a single JSON object only (no markdown): success=true, "
        "summary containing the token verbatim, token equal to the script token field, "
        "and status equal to the script status field.\n"
        "Do not reply until step 2 exits 0 and prints JSON."
    )
    bdd_context["output_schema"] = ECHO_TOKEN_SCHEMA


@given("a flat output schema with required fields has been prepared")
def prepare_flat(bdd_context: dict[str, Any]) -> None:
    bdd_context["output_schema"] = FLAT_OUTPUT_SCHEMA
    bdd_context["query"] = (
        "Respond with a single JSON object only (no markdown). "
        'Fields: success=true, summary="e2e-flat-ok", ticketId="E2E-STRUCT-001".'
    )


@given("a nested output schema has been prepared")
def prepare_nested(bdd_context: dict[str, Any]) -> None:
    bdd_context["output_schema"] = NESTED_OUTPUT_SCHEMA
    bdd_context["query"] = (
        "Respond with a single JSON object only (no markdown). "
        'success=true, summary="e2e-nested-ok", '
        'items=[{"name":"widget","count":1},{"name":"gadget","count":2}].'
    )


@given("no output schema will be sent")
def prepare_no_schema(bdd_context: dict[str, Any]) -> None:
    bdd_context["output_schema"] = None
    bdd_context["query"] = (
        "In one short sentence, name any primary color. "
        "Answer with plain text only. Do not return JSON. Do not call any tools."
    )


@given("an adversarial output schema and prompt have been prepared")
def prepare_adversarial(bdd_context: dict[str, Any]) -> None:
    bdd_context["output_schema"] = STRICT_CONFLICT_SCHEMA
    bdd_context["query"] = (
        "Reply with exactly the single word hello in plain text. "
        "Do not use JSON. Do not use markdown."
    )


# --- MCP scenarios ---


@given("the sandbox service is running with reasoning configured")
def sandbox_running_with_reasoning(server_url: str) -> None:
    import json as _json

    import pytest

    assert server_url.startswith("http"), f"unexpected server URL: {server_url!r}"
    raw = os.environ.get("LIGHTSPEED_REASONING_CONFIG", "").strip()
    if not raw:
        pytest.skip("LIGHTSPEED_REASONING_CONFIG not set — reasoning tests skipped")
    parsed = _json.loads(raw)
    assert isinstance(parsed, dict), (
        f"LIGHTSPEED_REASONING_CONFIG must be a JSON object, got: {raw!r}"
    )


@given("the sandbox service is running with MCP servers configured")
def sandbox_running_with_mcp(server_url: str) -> None:
    import json as _json

    import pytest

    assert server_url.startswith("http"), f"unexpected server URL: {server_url!r}"
    raw = os.environ.get("LIGHTSPEED_MCP_SERVERS", "").strip()
    if not raw:
        pytest.skip("LIGHTSPEED_MCP_SERVERS not set — MCP tests skipped")
    parsed = _json.loads(raw)
    assert isinstance(parsed, list), f"LIGHTSPEED_MCP_SERVERS must be a JSON array, got: {raw!r}"
    assert len(parsed) > 0, "LIGHTSPEED_MCP_SERVERS is an empty array"
    assert all("name" in s and "url" in s for s in parsed), (
        f"Each MCP server entry must have 'name' and 'url': {parsed!r}"
    )


@given("an MCP tool invocation query has been prepared")
def prepare_mcp_tool_invocation(bdd_context: dict[str, Any]) -> None:
    bdd_context["output_schema"] = MCP_TOOL_OUTPUT_SCHEMA
    bdd_context["query"] = (
        "Call the 'list_namespaces' tool from the MCP server named 'mock-ocp-mcp'. "
        "Return a single JSON object only (no markdown). "
        "Fields: success=true, summary=<the exact text output from the tool>."
    )


@given("an MCP query targeting a nonexistent tool has been prepared")
def prepare_mcp_nonexistent_tool(bdd_context: dict[str, Any]) -> None:
    bdd_context["output_schema"] = MCP_TOOL_OUTPUT_SCHEMA
    bdd_context["query"] = (
        "Actually attempt to call a tool named 'nonexistent_tool_xyz_999' from the MCP "
        "server named 'mock-ocp-mcp'. Do not assume the outcome — make the call. "
        "Return a single JSON object only (no markdown). Report the real outcome: "
        "set success to true only if the tool call actually succeeded, false otherwise, "
        "and put a short description of what happened (including the tool name) in summary."
    )
