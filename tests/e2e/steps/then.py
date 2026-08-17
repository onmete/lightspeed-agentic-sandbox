"""Then steps — HTTP and JSON assertions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import jsonschema
from pytest_bdd import then

from tests.e2e.runner import RunHttpResult

# SHA-256 of empty string — models sometimes fabricate this instead of running echo-token.sh
_EMPTY_STRING_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


@then("the response body status is ok")
def assert_body_status_ok(bdd_context: dict[str, Any]) -> None:
    """Assert probe JSON body has ``status: ok``."""
    body = bdd_context["response_body"]
    assert body.get("status") == "ok", body


@then("the HTTP response status code is 200")
def assert_status_200(bdd_context: dict[str, Any]) -> None:
    """Assert HTTP 200 with no transport error."""
    res: RunHttpResult = bdd_context["http_result"]
    assert res.error is None, f"transport error: {res.error}"
    assert res.status_code == 200, f"expected 200, got {res.status_code}: {res.raw_text[:500]}"


@then("the response body contains gen_ai histogram metrics")
def assert_gen_ai_histograms(bdd_context: dict[str, Any]) -> None:
    raw = bdd_context["http_result"].raw_text
    for name in (
        "gen_ai_client_token_usage",
        "gen_ai_client_operation_duration_seconds",
        "gen_ai_execute_tool_duration_seconds",
    ):
        assert name in raw, f"metric {name!r} not found in /metrics output"


def _histogram_count(raw: str, metric: str) -> float:
    """Sum the ``<metric>_count`` samples across all label sets in Prometheus text."""
    total = 0.0
    prefix = f"{metric}_count"
    for line in raw.splitlines():
        if line.startswith("#") or not line.startswith(prefix):
            continue
        # Format: gen_ai_..._count{labels} <value>   (or without the {labels} block)
        char_after = line[len(prefix) : len(prefix) + 1]
        if char_after not in ("{", " "):
            continue
        total += float(line.rsplit(" ", 1)[1])
    return total


@then("the gen_ai token and duration metrics have recorded samples")
def assert_gen_ai_metrics_recorded(bdd_context: dict[str, Any]) -> None:
    """Assert the histograms actually observed data, not just that they are registered.

    Only ``token_usage`` and ``operation_duration`` are recorded by the run
    endpoint; ``execute_tool_duration`` is registered but never observed, so it is
    intentionally excluded here.
    """
    raw = bdd_context["http_result"].raw_text
    for metric in (
        "gen_ai_client_token_usage",
        "gen_ai_client_operation_duration_seconds",
    ):
        count = _histogram_count(raw, metric)
        assert count > 0, f"metric {metric!r} has no recorded samples (count={count})"


@then("the response includes success summary and ticketId fields")
def assert_flat_fields(bdd_context: dict[str, Any]) -> None:
    """Assert structured output includes success, summary, and ticketId."""
    body = bdd_context["response_body"]
    assert "success" in body
    assert "summary" in body
    assert isinstance(body["summary"], str)
    assert body.get("ticketId"), f"missing ticketId in {body!r}"


@then("the response JSON validates against the output schema")
def assert_jsonschema(bdd_context: dict[str, Any]) -> None:
    """Validate response body against the prepared output schema."""
    schema = bdd_context["output_schema"]
    body = bdd_context["response_body"]
    response_token = body.get("token", "")
    if isinstance(response_token, str) and response_token == _EMPTY_STRING_SHA256:
        raise AssertionError(
            "response token looks fabricated (empty-string SHA-256); "
            "run bash scripts/echo-token.sh and use its stdout JSON"
        )
    jsonschema.validate(instance=body, schema=schema)


@then("the response has a non-empty summary")
def assert_nonempty_summary(bdd_context: dict[str, Any]) -> None:
    """Assert summary is a non-empty string."""
    body = bdd_context["response_body"]
    summary = body.get("summary", "")
    assert isinstance(summary, str), f"summary not a string: {body!r}"
    assert summary.strip(), f"summary missing/empty: {body!r}"


@then("success is true")
def assert_success_true(bdd_context: dict[str, Any]) -> None:
    """Assert RunResponse ``success`` is true."""
    body = bdd_context["response_body"]
    assert body.get("success") is True, body


@then("success is false")
def assert_success_false(bdd_context: dict[str, Any]) -> None:
    """Assert RunResponse ``success`` is false."""
    body = bdd_context["response_body"]
    assert body.get("success") is False, body


@then("the response summary indicates a timeout")
def assert_summary_indicates_timeout(bdd_context: dict[str, Any]) -> None:
    """Assert the failure was a timeout, not any other error.

    The endpoint returns ``success=false`` for timeouts, generic errors, and
    empty responses alike; this distinguishes the timeout path via its summary
    (``Agent timed out after {N}ms``).
    """
    body = bdd_context["response_body"]
    summary = body.get("summary", "").lower()
    assert "timed out" in summary or "timeout" in summary, (
        f"summary does not indicate a timeout: {body!r}"
    )


@then("the response summary contains the reasoning answer")
def assert_summary_contains_reasoning_answer(bdd_context: dict[str, Any]) -> None:
    """Assert the model produced the correct answer to 17 * 23 (391)."""
    body = bdd_context["response_body"]
    summary = body.get("summary", "")
    assert re.search(r"\b391\b", summary), f"summary missing correct answer 391: {body!r}"


@then("the response namespaces field matches the prepared context")
def assert_namespaces_match_context(bdd_context: dict[str, Any]) -> None:
    """Assert echoed namespaces match targetNamespaces from prepared context."""
    body = bdd_context["response_body"]
    expected = bdd_context["expected_namespaces"]
    actual = body.get("namespaces", "")

    def _ns_parts(value: str) -> list[str]:
        return [part.strip() for part in value.split(",") if part.strip()]

    assert _ns_parts(actual) == _ns_parts(expected), (
        f"expected namespaces {expected!r}, got {actual!r} in {body!r}"
    )


@then("the response first failure reason matches the prepared context")
def assert_first_failure_reason_matches_context(bdd_context: dict[str, Any]) -> None:
    """Assert echoed firstFailureReason matches previousAttempts from prepared context."""
    body = bdd_context["response_body"]
    expected = bdd_context["expected_first_failure_reason"]
    actual = body.get("firstFailureReason", "")
    assert actual == expected, (
        f"expected firstFailureReason {expected!r}, got {actual!r} in {body!r}"
    )


@then("the response approved option fields match the prepared context")
def assert_approved_option_matches_context(bdd_context: dict[str, Any]) -> None:
    """Assert echoed approvedTitle, rootCause, and firstCommand match context."""
    body = bdd_context["response_body"]
    expected_title = bdd_context["expected_approved_title"]
    expected_root_cause = bdd_context["expected_root_cause"]
    expected_command = bdd_context["expected_first_command"]
    actual_title = body.get("approvedTitle", "")
    actual_root_cause = body.get("rootCause", "")
    actual_command = body.get("firstCommand", "")
    assert actual_title == expected_title, (
        f"expected approvedTitle {expected_title!r}, got {actual_title!r} in {body!r}"
    )
    assert actual_root_cause == expected_root_cause, (
        f"expected rootCause {expected_root_cause!r}, got {actual_root_cause!r} in {body!r}"
    )
    assert actual_command == expected_command, (
        f"expected firstCommand {expected_command!r}, got {actual_command!r} in {body!r}"
    )


@then("the skill script wrote a token file to disk")
def assert_token_file(e2e_output_dir: Path | None, bdd_context: dict[str, Any]) -> None:
    """Assert the echo-token skill wrote ``.e2e_token`` under E2E_OUTPUT_DIR."""
    assert e2e_output_dir is not None, "E2E_OUTPUT_DIR not set"
    token_path = e2e_output_dir / ".e2e_token"
    assert token_path.exists(), (
        f"token file not found at {token_path}; "
        "the agent must run bash scripts/echo-token.sh from the skill directory"
    )
    token = token_path.read_text().strip()
    assert token, "token file is empty"
    bdd_context["token"] = token


@then("the response contains the generated token")
def assert_token_in_response(bdd_context: dict[str, Any]) -> None:
    """Assert the response body or summary includes the token from disk."""
    body = bdd_context["response_body"]
    token = bdd_context["token"]
    response_token = body.get("token", "")
    summary = body.get("summary", "")
    assert token in response_token or token in summary, (
        f"token {token!r} not found in response token={response_token!r} or summary={summary!r}"
    )


@then("the HTTP response status code is 200 and the envelope has success and summary")
def assert_200_envelope(bdd_context: dict[str, Any]) -> None:
    """Assert HTTP 200 and RunResponse envelope fields without schema validation."""
    res: RunHttpResult = bdd_context["http_result"]
    assert res.error is None, f"transport error: {res.error}"
    assert res.status_code == 200, f"expected 200, got {res.status_code}: {res.raw_text[:500]}"
    body = bdd_context["response_body"]
    assert "success" in body, body
    assert "summary" in body, body
    assert isinstance(body["summary"], str), body


# --- MCP assertions ---

# Unguessable marker returned only by the mock's list_namespaces tool — proves the
# agent actually invoked the tool. Keep in sync with mock_mcp_server.MOCK_NAMESPACES.
_MOCK_SENTINEL_NAMESPACE = "e2e-sentinel-ns-7f3a9"


@then("the response summary contains the sentinel namespace from the tool")
def assert_summary_contains_namespace_output(bdd_context: dict[str, Any]) -> None:
    """Assert the summary contains the unguessable sentinel namespace.

    A model cannot fabricate the sentinel, so its presence proves the agent
    actually invoked the mock MCP ``list_namespaces`` tool and used its output.
    """
    body = bdd_context["response_body"]
    summary = body.get("summary", "").lower()
    assert _MOCK_SENTINEL_NAMESPACE in summary, (
        f"summary does not contain sentinel namespace {_MOCK_SENTINEL_NAMESPACE!r} "
        f"(tool was likely not invoked): {summary!r}"
    )
