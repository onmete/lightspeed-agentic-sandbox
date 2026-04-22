"""Tests for normalized event logging."""

import logging

from lightspeed_agentic.logging import log_provider_event
from lightspeed_agentic.types import (
    ContentBlockStopEvent,
    ResultEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)


def test_log_thinking_delta(caplog):
    with caplog.at_level(logging.INFO, logger="lightspeed_agentic"):
        log_provider_event("analysis", ThinkingDeltaEvent(thinking="reasoning about issue"))
    assert "thinking" in caplog.text
    assert "reasoning about issue" in caplog.text


def test_log_tool_call(caplog):
    with caplog.at_level(logging.INFO, logger="lightspeed_agentic"):
        log_provider_event("execution", ToolCallEvent(name="bash", input='{"command": "ls"}'))
    assert "tool_use" in caplog.text
    assert "bash" in caplog.text


def test_log_tool_result(caplog):
    with caplog.at_level(logging.INFO, logger="lightspeed_agentic"):
        log_provider_event("analysis", ToolResultEvent(output="file1.txt"))
    assert "tool_result" in caplog.text


def test_log_result(caplog):
    with caplog.at_level(logging.INFO, logger="lightspeed_agentic"):
        log_provider_event("analysis", ResultEvent(text="done", cost_usd=0.05, input_tokens=100, output_tokens=50))
    assert "cost=$0.0500" in caplog.text
    assert "tokens=150" in caplog.text


def test_log_text_delta_is_silent(caplog):
    with caplog.at_level(logging.INFO, logger="lightspeed_agentic"):
        log_provider_event("chat", TextDeltaEvent(text="hello"))
    assert caplog.text == ""


def test_log_content_block_stop_is_silent(caplog):
    with caplog.at_level(logging.INFO, logger="lightspeed_agentic"):
        log_provider_event("chat", ContentBlockStopEvent())
    assert caplog.text == ""
