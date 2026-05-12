"""Given steps — service, schemas, prompts."""

from __future__ import annotations

from typing import Any

from pytest_bdd import given

from schemas_contract import (
    FLAT_OUTPUT_SCHEMA,
    NESTED_OUTPUT_SCHEMA,
    STRICT_CONFLICT_SCHEMA,
)


@given("the sandbox service is running")
def sandbox_running(server_url: str) -> None:
    assert server_url.startswith("http"), f"unexpected server URL: {server_url!r}"


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
