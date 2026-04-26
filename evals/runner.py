"""Eval execution wrapper — times queries and captures all events."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from lightspeed_agentic.types import (
    AgentProvider,
    ProviderEvent,
    ProviderQueryOptions,
    ResultEvent,
    ToolCallEvent,
    ToolResultEvent,
)


@dataclass
class EvalResult:
    provider: str
    result_text: str = ""
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    tool_calls: list[ToolCallEvent] = field(default_factory=list)
    tool_results: list[ToolResultEvent] = field(default_factory=list)
    events: list[ProviderEvent] = field(default_factory=list)
    error: str | None = None


async def run_eval(
    provider: AgentProvider,
    options: ProviderQueryOptions,
) -> EvalResult:
    result = EvalResult(provider=provider.name)
    start = time.monotonic()

    try:
        async for event in provider.query(options):
            result.events.append(event)

            if isinstance(event, ToolCallEvent):
                result.tool_calls.append(event)
            elif isinstance(event, ToolResultEvent):
                result.tool_results.append(event)
            elif isinstance(event, ResultEvent):
                result.result_text = event.text
                result.cost_usd = event.cost_usd
                result.input_tokens = event.input_tokens
                result.output_tokens = event.output_tokens
    except ImportError as e:
        import pytest
        pytest.skip(f"SDK not installed: {e}")
    except Exception as e:
        result.error = str(e)

    result.latency_seconds = time.monotonic() - start
    return result


def assert_tool_token(
    eval_workspace: Path,
    token_file_name: str,
    result: EvalResult,
    provider_name: str,
    script_name: str,
) -> None:
    token_file = eval_workspace / token_file_name
    try:
        expected_token = token_file.read_text().strip()
    except FileNotFoundError:
        raise AssertionError(
            f"{provider_name} did not run {script_name} (no {token_file_name} file)"
        )

    tool_outputs = [e.output for e in result.tool_results]
    all_text = " ".join(tool_outputs) + " " + result.result_text
    assert expected_token in all_text, (
        f"{provider_name} did not report the verification token from {script_name}. "
        f"expected={expected_token}, result={result.result_text[:200]}"
    )
