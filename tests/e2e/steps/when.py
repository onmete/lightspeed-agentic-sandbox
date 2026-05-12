"""When steps — POST /v1/agent/run."""

from __future__ import annotations

from typing import Any

from pytest_bdd import when

from runner import RunHttpResult


@when("I POST run with the prepared schema and query")
def post_with_schema(bdd_context: dict[str, Any], run_runner: Any) -> None:
    schema = bdd_context.get("output_schema")
    query = bdd_context["query"]
    res: RunHttpResult = run_runner(query, output_schema=schema)
    bdd_context["http_result"] = res
    bdd_context["response_body"] = res.body


@when("I POST run with the prepared query and no output schema")
def post_without_schema(bdd_context: dict[str, Any], run_runner: Any) -> None:
    query = bdd_context["query"]
    res: RunHttpResult = run_runner(query, output_schema=None)
    bdd_context["http_result"] = res
    bdd_context["response_body"] = res.body
