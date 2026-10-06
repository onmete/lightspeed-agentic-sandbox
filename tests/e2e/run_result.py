"""E2E run result envelope for batch Job BDD steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tests.e2e.batch_runner import RunBatchResult


@dataclass
class E2ERunResult:
    """Run outcome exposed to BDD steps after a batch Job completes."""

    body: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    error: str | None = None
    batch: RunBatchResult | None = None


def batch_to_run_result(batch: RunBatchResult) -> E2ERunResult:
    """Map a batch Job result to the shared E2E assertion envelope."""
    return E2ERunResult(
        body=batch.body,
        raw_text=batch.pod_logs,
        error=batch.error,
        batch=batch,
    )


def store_run_result(bdd_context: dict[str, Any], result: E2ERunResult) -> None:
    """Store run result on the BDD context."""
    bdd_context["run_result"] = result
    bdd_context["response_body"] = result.body
    if result.batch is not None:
        bdd_context["run_uid"] = result.batch.run_uid
        bdd_context["run_step"] = result.batch.step
        bdd_context["batch_job_name"] = result.batch.job_name
        if result.batch.tool_token:
            bdd_context["tool_token"] = result.batch.tool_token
