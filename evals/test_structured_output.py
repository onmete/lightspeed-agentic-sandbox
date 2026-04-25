"""Structured output tests — JSON schema enforcement via native provider mechanisms."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import jsonschema
import pytest

from lightspeed_agentic.types import ProviderQueryOptions

from .runner import EvalResult
from .schemas import ANALYSIS_SCHEMA, CALCULATION_SCHEMA


@pytest.mark.eval
@pytest.mark.asyncio
async def test_analysis_schema(
    provider_name: str, default_model: str, eval_workspace: Path,
    eval_runner: Callable[..., EvalResult],
) -> None:
    """Provider returns valid JSON conforming to the analysis schema."""
    result = await eval_runner(ProviderQueryOptions(
        prompt=(
            "Analyze the health of a Kubernetes cluster that has 3 nodes, "
            "all reporting Ready status, with 12 pods running across 4 namespaces. "
            "One pod in the 'monitoring' namespace has been restarting frequently."
        ),
        system_prompt="You are a Kubernetes cluster analyst. Provide structured analysis.",
        model=default_model,
        max_turns=10,
        max_budget_usd=1.0,
        allowed_tools=["Bash"],
        cwd=str(eval_workspace),
        output_schema=ANALYSIS_SCHEMA,
    ))

    assert result.error is None, f"{provider_name} errored: {result.error}"
    assert result.result_text, f"{provider_name} returned empty text"

    parsed = json.loads(result.result_text)
    jsonschema.validate(parsed, ANALYSIS_SCHEMA)

    assert isinstance(parsed["success"], bool)
    assert parsed["confidence"] in ("high", "medium", "low")
    assert len(parsed["findings"]) > 0
    for finding in parsed["findings"]:
        assert finding["category"]
        assert finding["description"]


@pytest.mark.eval
@pytest.mark.asyncio
async def test_calculation_schema(
    provider_name: str, default_model: str, eval_workspace: Path,
    eval_runner: Callable[..., EvalResult],
) -> None:
    """Provider returns a calculation result conforming to the schema."""
    result = await eval_runner(ProviderQueryOptions(
        prompt="Calculate (15 * 7) + 23. Return the expression and numeric result.",
        system_prompt="You are a math assistant.",
        model=default_model,
        max_turns=5,
        max_budget_usd=0.50,
        allowed_tools=[],
        cwd=str(eval_workspace),
        output_schema=CALCULATION_SCHEMA,
    ))

    assert result.error is None, f"{provider_name} errored: {result.error}"
    assert result.result_text

    parsed = json.loads(result.result_text)
    jsonschema.validate(parsed, CALCULATION_SCHEMA)

    assert parsed["result"] == 128 or abs(parsed["result"] - 128) < 0.01


@pytest.mark.eval
@pytest.mark.asyncio
async def test_schema_with_enum(
    provider_name: str, default_model: str, eval_workspace: Path,
    eval_runner: Callable[..., EvalResult],
) -> None:
    """Provider respects enum constraints in the schema."""
    result = await eval_runner(ProviderQueryOptions(
        prompt=(
            "A production cluster has 5 nodes, all healthy, running 50 pods with "
            "zero restarts in the last 24 hours. Provide your analysis."
        ),
        system_prompt="You are a cluster health analyst.",
        model=default_model,
        max_turns=5,
        max_budget_usd=0.50,
        allowed_tools=[],
        cwd=str(eval_workspace),
        output_schema=ANALYSIS_SCHEMA,
    ))

    assert result.error is None
    assert result.result_text

    parsed = json.loads(result.result_text)
    jsonschema.validate(parsed, ANALYSIS_SCHEMA)
    assert parsed["confidence"] in ("high", "medium", "low")
