"""Tool usage tests — model invokes bash scripts and uses their output."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import jsonschema
import pytest

from lightspeed_agentic.types import ProviderQueryOptions

from .runner import EvalResult
from .schemas import TOOL_USAGE_SCHEMA


@pytest.mark.eval
@pytest.mark.asyncio
async def test_greet_tool(
    provider_name: str, default_model: str, eval_workspace: Path,
    eval_runner: Callable[..., EvalResult],
) -> None:
    """Provider invokes a bash script and uses its output."""
    result = await eval_runner(ProviderQueryOptions(
        prompt=(
            "Run the greeting tool with the name 'Alice' by executing: "
            "bash tools/greet.sh Alice\n"
            "Then tell me what the greeting said."
        ),
        system_prompt=(
            "You are an assistant. Use bash tools to accomplish tasks. "
            "Execute commands in the workspace directory."
        ),
        model=default_model,
        max_turns=10,
        max_budget_usd=1.0,
        allowed_tools=["Bash"],
        cwd=str(eval_workspace),
    ))

    assert result.error is None, f"{provider_name} errored: {result.error}"

    tool_outputs = [e.output for e in result.tool_results]
    has_tool_evidence = (
        len(result.tool_calls) > 0
        or any("Alice" in o for o in tool_outputs)
        or "Alice" in result.result_text
    )
    assert has_tool_evidence, (
        f"{provider_name} did not invoke greet tool or mention Alice. "
        f"tool_calls={len(result.tool_calls)}, result={result.result_text[:200]}"
    )


@pytest.mark.eval
@pytest.mark.asyncio
async def test_compute_tool_with_structured_output(
    provider_name: str, default_model: str, eval_workspace: Path,
    eval_runner: Callable[..., EvalResult],
) -> None:
    """Provider runs compute.sh and returns structured output."""
    result = await eval_runner(ProviderQueryOptions(
        prompt=(
            "Run the compute tool to calculate 42 * 3 by executing: "
            "bash tools/compute.sh '42 * 3'\n"
            "Report which tools you used and the output."
        ),
        system_prompt="You are an assistant. Execute bash commands to accomplish tasks.",
        model=default_model,
        max_turns=10,
        max_budget_usd=1.0,
        allowed_tools=["Bash"],
        cwd=str(eval_workspace),
        output_schema=TOOL_USAGE_SCHEMA,
    ))

    assert result.error is None, f"{provider_name} errored: {result.error}"

    if provider_name == "gemini" and not result.result_text:
        pytest.xfail("Gemini ADK output_schema disables tools (google/adk-python#5054)")

    assert result.result_text, f"{provider_name} returned empty result"
    parsed = json.loads(result.result_text)
    jsonschema.validate(parsed, TOOL_USAGE_SCHEMA)
    assert parsed["success"] is True


@pytest.mark.eval
@pytest.mark.asyncio
async def test_lookup_data_tool(
    provider_name: str, default_model: str, eval_workspace: Path,
    eval_runner: Callable[..., EvalResult],
) -> None:
    """Provider queries lookup-data.sh for version information."""
    result = await eval_runner(ProviderQueryOptions(
        prompt=(
            "Run: bash tools/lookup-data.sh version\n"
            "Tell me the version number from the output."
        ),
        system_prompt="You are an assistant. Execute bash commands to retrieve data.",
        model=default_model,
        max_turns=10,
        max_budget_usd=1.0,
        allowed_tools=["Bash"],
        cwd=str(eval_workspace),
    ))

    assert result.error is None, f"{provider_name} errored: {result.error}"

    tool_outputs = [e.output for e in result.tool_results]
    has_version = (
        any("2.1.0" in o for o in tool_outputs)
        or "2.1.0" in result.result_text
    )
    assert has_version, (
        f"{provider_name} did not find version 2.1.0. "
        f"tool_outputs={tool_outputs}, result={result.result_text[:200]}"
    )
