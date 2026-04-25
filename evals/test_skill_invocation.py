"""Skill invocation tests — model discovers and uses dummy skills."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from lightspeed_agentic.types import ProviderQueryOptions

from .runner import EvalResult
from .schemas import CALCULATION_SCHEMA


@pytest.mark.eval
@pytest.mark.asyncio
async def test_calculator_skill(
    provider_name: str, default_model: str, eval_workspace: Path,
    eval_runner: Callable[..., EvalResult],
) -> None:
    """Provider discovers and uses the calculator skill."""
    result = await eval_runner(ProviderQueryOptions(
        prompt=(
            "Use the calculator skill to compute (15 * 7) + 23. "
            "You must use the skill, not calculate mentally."
        ),
        system_prompt=(
            "You are an assistant with access to skills. "
            "Use the calculator skill to perform calculations. "
            "The skill lets you run: python3 -c \"print(eval('EXPRESSION'))\""
        ),
        model=default_model,
        max_turns=15,
        max_budget_usd=1.0,
        allowed_tools=["Bash", "Skill"],
        cwd=str(eval_workspace),
        output_schema=CALCULATION_SCHEMA,
    ))

    assert result.error is None, f"{provider_name} errored: {result.error}"
    assert result.result_text, f"{provider_name} returned empty result"

    parsed = json.loads(result.result_text)
    assert parsed["result"] == 128 or abs(parsed["result"] - 128) < 0.01, (
        f"Expected 128, got {parsed['result']}"
    )


@pytest.mark.eval
@pytest.mark.asyncio
async def test_lookup_skill(
    provider_name: str, default_model: str, eval_workspace: Path,
    eval_runner: Callable[..., EvalResult],
) -> None:
    """Provider discovers and uses the lookup skill to query data."""
    result = await eval_runner(ProviderQueryOptions(
        prompt=(
            "Use the lookup skill to find the current system status. "
            "Run: bash tools/lookup-data.sh status\n"
            "Tell me the system health status."
        ),
        system_prompt=(
            "You are an assistant with access to skills. "
            "Use the lookup skill to query data. "
            "The skill provides bash tools/lookup-data.sh with keys: status, version, config."
        ),
        model=default_model,
        max_turns=15,
        max_budget_usd=1.0,
        allowed_tools=["Bash", "Skill"],
        cwd=str(eval_workspace),
    ))

    assert result.error is None, f"{provider_name} errored: {result.error}"

    tool_outputs = [e.output for e in result.tool_results]
    has_health = (
        any("healthy" in o for o in tool_outputs)
        or "healthy" in result.result_text.lower()
    )
    assert has_health, (
        f"{provider_name} did not find 'healthy' in results. "
        f"tool_outputs={tool_outputs}, result={result.result_text[:200]}"
    )
