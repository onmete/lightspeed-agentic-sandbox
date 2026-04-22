"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from lightspeed_agentic.types import (
    AgentProvider,
    ProviderEvent,
    ProviderQueryOptions,
    ResultEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)


class MockProvider(AgentProvider):
    """Provider that yields a configurable sequence of events."""

    def __init__(self, events: list[ProviderEvent] | None = None) -> None:
        self._events = events or [
            ResultEvent(text='{"success": true, "summary": "mock result"}', cost_usd=0.01, input_tokens=100, output_tokens=50),
        ]

    @property
    def name(self) -> str:
        return "mock"

    async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
        for event in self._events:
            yield event


class StreamingMockProvider(AgentProvider):
    """Provider that simulates a streaming response with tool calls."""

    @property
    def name(self) -> str:
        return "streaming_mock"

    async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
        yield TextDeltaEvent(text="Hello ")
        yield TextDeltaEvent(text="world")
        yield ToolCallEvent(name="bash", input='{"command": "ls"}')
        yield ToolResultEvent(output="file1.txt\nfile2.txt")
        yield TextDeltaEvent(text="\nDone.")
        yield ResultEvent(text="Hello world\nDone.", cost_usd=0.05, input_tokens=200, output_tokens=100)


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()


@pytest.fixture
def streaming_provider() -> StreamingMockProvider:
    return StreamingMockProvider()
