"""Skill invocation tests — model discovers and uses dummy skills."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from lightspeed_agentic.types import ProviderQueryOptions

from .runner import EvalResult


@pytest.mark.eval
@pytest.mark.asyncio
async def test_calculator_skill(
    provider_name: str, default_model: str, eval_workspace: Path,
    eval_runner: Callable[..., EvalResult],
) -> None:
    """Provider discovers and uses the calculator skill."""
    result = await eval_runner(ProviderQueryOptions(
        prompt=(
            "Run: bash tools/calc.sh '(15 * 7) + 23'\n"
            "What is the result and verification token?"
        ),
        system_prompt="You are an assistant. Execute commands to get results.",
        model=default_model,
        max_turns=15,
        max_budget_usd=1.0,
        allowed_tools=["Bash", "Skill"],
        cwd=str(eval_workspace),
    ))

    assert result.error is None, f"{provider_name} errored: {result.error}"

    token_file = eval_workspace / ".calc_token"
    assert token_file.exists(), f"{provider_name} did not run calc.sh (no .calc_token file)"
    expected_token = token_file.read_text().strip()

    tool_outputs = [e.output for e in result.tool_results]
    all_text = " ".join(tool_outputs) + " " + result.result_text
    assert expected_token in all_text, (
        f"{provider_name} did not report the verification token from calc.sh. "
        f"expected={expected_token}, result={result.result_text[:200]}"
    )
    assert "128" in result.result_text, (
        f"{provider_name} did not report 128. result={result.result_text[:200]}"
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
            "Run: bash tools/lookup-data.sh version\n"
            "What is the version number and verification token?"
        ),
        system_prompt="You are an assistant. Execute commands to get results.",
        model=default_model,
        max_turns=15,
        max_budget_usd=1.0,
        allowed_tools=["Bash", "Skill"],
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
