"""Skill invocation test.

Model discovers the skill, executes the tool, and returns complex structured
output.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import jsonschema
import pytest

from .runner import RunResult, assert_tool_token
from .schemas import ANALYSIS_WITH_COMPONENTS_SCHEMA


@pytest.mark.eval
@pytest.mark.asyncio
async def test_find_token_skill(
    provider_name: str,
    eval_workspace: Path,
    eval_runner: Callable[..., RunResult],
) -> None:
    """Provider discovers find-token skill, executes it, returns analysis with components."""
    result = await eval_runner(
        query="Find the hidden token using the 'find-token' skill.",
        system_prompt="You are an assistant. Use your available skills to accomplish tasks.",
        output_schema=ANALYSIS_WITH_COMPONENTS_SCHEMA,
    )

    assert result.error is None, f"{provider_name} errored: {result.error}"

    jsonschema.validate(result.raw, ANALYSIS_WITH_COMPONENTS_SCHEMA)

    assert_tool_token(eval_workspace, ".hidden_token", result, provider_name, "find-token.sh")

    option = result.raw["options"][0]
    assert option["diagnosis"]["confidence"] in ("low", "medium", "high")
    assert option["proposal"]["risk"] in ("low", "medium", "high", "critical")
    assert isinstance(option["proposal"]["reversible"], bool)
    assert len(option["proposal"]["actions"]) > 0
    assert len(option["components"]) > 0

    comp = option["components"][0]
    assert comp["tokens"]["primary"]["valid"] is True
    assert comp["tokens"]["secondary"]["valid"] is True
    assert comp["audit"]["outcome"] in ("pass", "fail", "partial")
    assert len(comp["audit"]["findings"]) > 0
    for finding in comp["audit"]["findings"]:
        assert finding["severity"] in ("info", "warning", "critical")
