"""Tests for FastAPI routes using mock providers."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from lightspeed_agentic.routes import build_router
from lightspeed_agentic.types import ResultEvent, TextDeltaEvent, ThinkingDeltaEvent

from .conftest import MockProvider, StreamingMockProvider


def _ag_ui_events(body: str) -> list[dict]:
    """Parse AG-UI JSON payloads from an SSE body (`data:` lines only)."""
    events = []
    for line in body.split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            events.append(json.loads(line.removeprefix("data: ")))
    return events


def _make_app(provider) -> FastAPI:
    app = FastAPI()
    router = build_router(provider, skills_dir="/workspace", model="test-model")
    app.include_router(router, prefix="/v1/agent")
    return app


@pytest.mark.asyncio
async def test_analyze_endpoint():
    app = _make_app(MockProvider())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/agent/analyze",
            json={
                "query": "Diagnose the issue",
                "systemPrompt": "You are an SRE agent.",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "mock result" in data["summary"]


@pytest.mark.asyncio
async def test_execute_endpoint():
    app = _make_app(MockProvider())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/agent/execute",
            json={
                "query": "Apply the fix",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True


@pytest.mark.asyncio
async def test_verify_endpoint():
    app = _make_app(MockProvider())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/agent/verify",
            json={
                "query": "Check the fix worked",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True


@pytest.mark.asyncio
async def test_analyze_with_context():
    app = _make_app(MockProvider())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/agent/analyze",
            json={
                "query": "Diagnose the issue",
                "context": {
                    "targetNamespaces": ["default", "kube-system"],
                    "attempt": 2,
                    "previousAttempts": [{"attempt": 1, "failureReason": "timeout"}],
                },
            },
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True


@pytest.mark.asyncio
async def test_analyze_with_output_schema():
    app = _make_app(MockProvider())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/agent/analyze",
            json={
                "query": "Diagnose",
                "outputSchema": {
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                },
            },
        )
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_analyze_empty_response():
    provider = MockProvider(events=[ResultEvent(text="")])
    app = _make_app(provider)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/agent/analyze", json={"query": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "empty" in data["summary"].lower()


@pytest.mark.asyncio
async def test_analyze_text_response():
    provider = MockProvider(events=[ResultEvent(text="Just plain text, not JSON")])
    app = _make_app(provider)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/agent/analyze", json={"query": "test"})
        data = resp.json()
        assert data["success"] is True
        assert data["summary"] == "Just plain text, not JSON"


@pytest.mark.asyncio
async def test_chat_endpoint_sse():
    app = _make_app(StreamingMockProvider())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/agent/chat",
            json={
                "message": "What's happening?",
                "context": {
                    "remediation": {"name": "test-rem", "namespace": "default"},
                    "alert": {"name": "TestAlert", "status": "firing", "severity": "warning"},
                },
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"

        evs = _ag_ui_events(resp.text)
        assert evs[0]["type"] == "RUN_STARTED"
        assert "threadId" in evs[0]
        assert "runId" in evs[0]
        assert evs[-1]["type"] == "RUN_FINISHED"
        types = {e["type"] for e in evs}
        assert "TEXT_MESSAGE_START" in types
        assert "TEXT_MESSAGE_CONTENT" in types
        assert "TOOL_CALL_START" in types
        assert "TOOL_CALL_ARGS" in types
        assert "TOOL_CALL_END" in types
        assert "TOOL_CALL_RESULT" in types


@pytest.mark.asyncio
async def test_chat_conversation_continuity():
    app = _make_app(StreamingMockProvider())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp1 = await client.post(
            "/v1/agent/chat",
            json={
                "message": "First message",
                "context": {
                    "remediation": {"name": "r", "namespace": "ns"},
                    "alert": {"name": "a", "status": "firing", "severity": "low"},
                },
            },
        )
        evs1 = _ag_ui_events(resp1.text)
        assert evs1[0]["type"] == "RUN_STARTED"
        thread_id = evs1[0]["threadId"]

        resp2 = await client.post(
            "/v1/agent/chat",
            json={
                "message": "Follow-up",
                "conversationId": thread_id,
                "context": {
                    "remediation": {"name": "r", "namespace": "ns"},
                    "alert": {"name": "a", "status": "firing", "severity": "low"},
                },
            },
        )
        assert resp2.status_code == 200
        evs2 = _ag_ui_events(resp2.text)
        assert evs2[0]["type"] == "RUN_STARTED"
        assert evs2[0]["threadId"] == thread_id


@pytest.mark.asyncio
async def test_chat_ui_fence_parsing():
    """Verify that ```ui:type``` fences are emitted as ui_component events."""
    fenced_text = 'Before ```ui:visualization\n{"title": "Test", "queries": ["up"]}\n```After'

    provider = MockProvider(
        events=[
            TextDeltaEvent(text=fenced_text),
            ResultEvent(text=fenced_text),
        ]
    )
    app = _make_app(provider)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/agent/chat",
            json={
                "message": "Show metrics",
                "context": {
                    "remediation": {"name": "r", "namespace": "ns"},
                    "alert": {"name": "a", "status": "firing", "severity": "low"},
                },
            },
        )
        evs = _ag_ui_events(resp.text)
        custom = [e for e in evs if e.get("type") == "CUSTOM" and e.get("name") == "ui_component"]
        assert custom
        assert custom[0]["value"]["type"] == "visualization"
        assert custom[0]["value"]["props"]["title"] == "Test"


@pytest.mark.asyncio
async def test_chat_reasoning_ag_ui():
    """Thinking deltas emit AG-UI reasoning lifecycle events."""
    provider = MockProvider(
        events=[
            ThinkingDeltaEvent(thinking="step one "),
            ThinkingDeltaEvent(thinking="step two"),
            TextDeltaEvent(text="Answer."),
            ResultEvent(text="Answer.", cost_usd=0.0, input_tokens=1, output_tokens=1),
        ]
    )
    app = _make_app(provider)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/agent/chat",
            json={
                "message": "Think",
                "context": {
                    "remediation": {"name": "r", "namespace": "ns"},
                    "alert": {"name": "a", "status": "firing", "severity": "low"},
                },
            },
        )
        assert resp.status_code == 200
        evs = _ag_ui_events(resp.text)
        types = [e["type"] for e in evs]
        assert "REASONING_START" in types
        assert "REASONING_MESSAGE_START" in types
        assert "REASONING_MESSAGE_CONTENT" in types
        assert "REASONING_MESSAGE_END" in types
        assert "REASONING_END" in types
