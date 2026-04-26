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

    token_file = eval_workspace / ".greet_token"
    assert token_file.exists(), f"{provider_name} did not run greet.sh (no .greet_token file)"
    expected_token = token_file.read_text().strip()

    tool_outputs = [e.output for e in result.tool_results]
    all_text = " ".join(tool_outputs) + " " + result.result_text
    assert expected_token in all_text, (
        f"{provider_name} did not report the verification token from greet.sh. "
        f"expected={expected_token}, result={result.result_text[:200]}"
    )
    assert "Alice" in result.result_text, (
        f"{provider_name} did not mention Alice. result={result.result_text[:200]}"
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
            "Run: bash tools/compute.sh '42 * 3'\n"
            "Report which tools you used, the output, and the verification token."
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
    assert result.result_text, f"{provider_name} returned empty result"

    token_file = eval_workspace / ".compute_token"
    assert token_file.exists(), f"{provider_name} did not run compute.sh (no .compute_token file)"
    expected_token = token_file.read_text().strip()

    tool_outputs = [e.output for e in result.tool_results]
    all_text = " ".join(tool_outputs) + " " + result.result_text
    assert expected_token in all_text, (
        f"{provider_name} did not report the verification token from compute.sh. "
        f"expected={expected_token}, result={result.result_text[:200]}"
    )

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
            "Tell me the version number and verification token from the output."
        ),
        system_prompt="You are an assistant. Execute bash commands to retrieve data.",
        model=default_model,
        max_turns=10,
        max_budget_usd=1.0,
        allowed_tools=["Bash"],
        cwd=str(eval_workspace),
    ))

    assert result.error is None, f"{provider_name} errored: {result.error}"

    token_file = eval_workspace / ".lookup_token"
    assert token_file.exists(), f"{provider_name} did not run lookup-data.sh (no .lookup_token file)"
    expected_token = token_file.read_text().strip()

    tool_outputs = [e.output for e in result.tool_results]
    all_text = " ".join(tool_outputs) + " " + result.result_text
    assert expected_token in all_text, (
        f"{provider_name} did not report the verification token from lookup-data.sh. "
        f"expected={expected_token}, result={result.result_text[:200]}"
    )
    assert "2.1.0" in result.result_text, (
        f"{provider_name} did not report version 2.1.0. "
        f"result={result.result_text[:200]}"
    )
