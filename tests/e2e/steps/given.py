"""Given steps — service, schemas, prompts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pytest_bdd import given

from schemas_contract import (
    ECHO_TOKEN_SCHEMA,
    FLAT_OUTPUT_SCHEMA,
    NESTED_OUTPUT_SCHEMA,
    STRICT_CONFLICT_SCHEMA,
)


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
    bdd_context["query"] = "In one sentence, name any primary color."
    bdd_context["output_schema"] = None


@given("the echo-token skill query has been prepared")
def prepare_echo_token(bdd_context: dict[str, Any]) -> None:
    bdd_context["query"] = (
        "Use the 'echo-token' skill to generate a verification token. "
        "The skill instructions tell you to run: bash echo-token/tools/echo-token.sh "
        "Return the token and status values from the script output."
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
        "In one short sentence, name any primary color. Do not return JSON; plain text is fine."
    )


@given("an adversarial output schema and prompt have been prepared")
def prepare_adversarial(bdd_context: dict[str, Any]) -> None:
    bdd_context["output_schema"] = STRICT_CONFLICT_SCHEMA
    bdd_context["query"] = (
        "Reply with exactly the single word hello in plain text. "
        "Do not use JSON. Do not use markdown."
    )
