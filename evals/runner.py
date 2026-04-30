"""Eval runner — POSTs to /v1/agent/run (or deprecated /v1/agent/analyze)."""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx


@dataclass
class RunResult:
    provider: str = ""
    success: bool = False
    summary: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    latency_seconds: float = 0.0
    error: str | None = None


AnalyzeResult = RunResult


async def run_query(
    server_url: str,
    query: str,
    system_prompt: str = "You are a helpful assistant.",
    output_schema: dict | None = None,
    timeout_ms: int | None = None,
) -> RunResult:
    """POST to /v1/agent/run — the primary eval entry point."""
    result = RunResult()
    start = time.monotonic()

    body: dict[str, Any] = {
        "query": query,
        "systemPrompt": system_prompt,
    }
    if output_schema:
        body["outputSchema"] = output_schema
    if timeout_ms is not None:
        body["timeout_ms"] = timeout_ms

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(f"{server_url}/v1/agent/run", json=body)
            resp.raise_for_status()
            data = resp.json()

        result.success = data.get("success", False)
        result.summary = data.get("summary", "")
        result.raw = data
    except Exception as e:
        result.error = str(e)

    result.latency_seconds = time.monotonic() - start
    return result


async def run_analyze(
    server_url: str,
    query: str,
    system_prompt: str = "You are a helpful assistant.",
    output_schema: dict | None = None,
) -> RunResult:
    """Deprecated — use run_query() instead."""
    warnings.warn("run_analyze is deprecated, use run_query", DeprecationWarning, stacklevel=2)
    return await run_query(server_url, query, system_prompt, output_schema)


def assert_tool_token(
    eval_workspace: Path,
    token_file_name: str,
    result: AnalyzeResult,
    provider_name: str,
    script_name: str,
) -> None:
    matches = list(eval_workspace.rglob(token_file_name))
    if not matches:
        raise AssertionError(
            f"{provider_name} did not run {script_name} "
            f"(no {token_file_name} found in {eval_workspace})"
        )

    response_text = json.dumps(result.raw)
    tokens = matches[0].read_text().strip().splitlines()
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        assert token in response_text, (
            f"{provider_name} did not report verification token {token} from {script_name}. "
            f"response={result.summary[:200]}"
        )
