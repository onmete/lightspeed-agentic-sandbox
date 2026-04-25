"""Smoke tests — basic prompt/response sanity across providers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from lightspeed_agentic.types import ProviderQueryOptions

from .runner import EvalResult


@pytest.mark.eval
@pytest.mark.asyncio
async def test_basic_response(
    provider_name: str, default_model: str, eval_workspace: Path,
    eval_runner: Callable[..., EvalResult],
) -> None:
    """Provider returns a coherent text response to a simple question."""
    result = await eval_runner(ProviderQueryOptions(
        prompt="What is 2 + 2? Answer with just the number.",
        system_prompt="You are a helpful assistant. Be concise.",
        model=default_model,
        max_turns=5,
        max_budget_usd=0.50,
        allowed_tools=[],
        cwd=str(eval_workspace),
    ))

    assert result.error is None, f"{provider_name} errored: {result.error}"
    assert result.result_text, f"{provider_name} returned empty text"
    assert "4" in result.result_text, f"Expected '4' in: {result.result_text}"
    assert result.latency_seconds < 60, f"Took too long: {result.latency_seconds:.1f}s"


@pytest.mark.eval
@pytest.mark.asyncio
async def test_cost_tracking(
    provider_name: str, default_model: str, eval_workspace: Path,
    eval_runner: Callable[..., EvalResult],
) -> None:
    """Provider reports token usage (cost may be 0 for some providers)."""
    result = await eval_runner(ProviderQueryOptions(
        prompt="Say hello in one word.",
        system_prompt="Be concise.",
        model=default_model,
        max_turns=3,
        max_budget_usd=0.50,
        allowed_tools=[],
        cwd=str(eval_workspace),
    ))

    assert result.error is None
    assert result.result_text
    assert result.input_tokens >= 0
    assert result.output_tokens >= 0
