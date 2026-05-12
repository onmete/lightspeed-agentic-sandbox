"""Then steps — HTTP and JSON assertions."""

from __future__ import annotations

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


@then("the HTTP response status code is 200 and the envelope has success and summary")
def assert_200_envelope(bdd_context: dict[str, Any]) -> None:
    res: RunHttpResult = bdd_context["http_result"]
    assert res.error is None, f"transport error: {res.error}"
    assert res.status_code == 200, f"expected 200, got {res.status_code}: {res.raw_text[:500]}"
    body = bdd_context["response_body"]
    assert "success" in body, body
    assert "summary" in body, body
    assert isinstance(body["summary"], str), body
