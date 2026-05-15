"""Then steps — HTTP and JSON assertions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jsonschema
from pytest_bdd import then

from runner import RunHttpResult


@then("the HTTP response status code is 200")
def assert_status_200(bdd_context: dict[str, Any]) -> None:
    res: RunHttpResult = bdd_context["http_result"]
    assert res.error is None, f"transport error: {res.error}"
    assert res.status_code == 200, f"expected 200, got {res.status_code}: {res.raw_text[:500]}"


@then("the response includes success summary and ticketId fields")
def assert_flat_fields(bdd_context: dict[str, Any]) -> None:
    body = bdd_context["response_body"]
    assert "success" in body
    assert "summary" in body
    assert isinstance(body["summary"], str)
    assert body.get("ticketId"), f"missing ticketId in {body!r}"


@then("the response JSON validates against the output schema")
def assert_jsonschema(bdd_context: dict[str, Any]) -> None:
    schema = bdd_context["output_schema"]
    body = bdd_context["response_body"]
    jsonschema.validate(instance=body, schema=schema)


@then("the response has a non-empty summary")
def assert_nonempty_summary(bdd_context: dict[str, Any]) -> None:
    body = bdd_context["response_body"]
    summary = body.get("summary", "")
    assert isinstance(summary, str), f"summary not a string: {body!r}"
    assert summary.strip(), f"summary missing/empty: {body!r}"


@then("success is true")
def assert_success_true(bdd_context: dict[str, Any]) -> None:
    body = bdd_context["response_body"]
    assert body.get("success") is True, body


@then("the skill script wrote a token file to disk")
def assert_token_file(e2e_output_dir: Path | None, bdd_context: dict[str, Any]) -> None:
    assert e2e_output_dir is not None, "E2E_OUTPUT_DIR not set"
    token_path = e2e_output_dir / ".e2e_token"
    assert token_path.exists(), f"token file not found at {token_path}"
    token = token_path.read_text().strip()
    assert token, "token file is empty"
    bdd_context["token"] = token


@then("the response contains the generated token")
def assert_token_in_response(bdd_context: dict[str, Any]) -> None:
    body = bdd_context["response_body"]
    token = bdd_context["token"]
    response_token = body.get("token", "")
    summary = body.get("summary", "")
    assert token in response_token or token in summary, (
        f"token {token!r} not found in response token={response_token!r} or summary={summary!r}"
    )


@then("the HTTP response status code is 200 and the envelope has success and summary")
def assert_200_envelope(bdd_context: dict[str, Any]) -> None:
    res: RunHttpResult = bdd_context["http_result"]
    assert res.error is None, f"transport error: {res.error}"
    assert res.status_code == 200, f"expected 200, got {res.status_code}: {res.raw_text[:500]}"
    body = bdd_context["response_body"]
    assert "success" in body, body
    assert "summary" in body, body
    assert isinstance(body["summary"], str), body
