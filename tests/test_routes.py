"""Tests for FastAPI routes using mock providers."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from lightspeed_agentic.routes import build_router
from lightspeed_agentic.types import ResultEvent, TextDeltaEvent

from .conftest import MockProvider, StreamingMockProvider


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
async def test_run_endpoint():
    app = _make_app(MockProvider())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/agent/run",
            json={"query": "Diagnose the issue"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "mock result" in data["summary"]


@pytest.mark.asyncio
async def test_run_with_system_prompt():
    app = _make_app(MockProvider())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/agent/run",
            json={
                "query": "Diagnose the issue",
                "systemPrompt": "You are an SRE agent.",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True


@pytest.mark.asyncio
async def test_run_with_context():
    app = _make_app(MockProvider())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/agent/run",
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
async def test_run_with_output_schema():
    app = _make_app(MockProvider())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/agent/run",
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
async def test_run_with_timeout():
    app = _make_app(MockProvider())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/agent/run",
            json={
                "query": "Diagnose",
                "timeout_ms": 60000,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True


@pytest.mark.asyncio
async def test_run_empty_response():
    provider = MockProvider(events=[ResultEvent(text="")])
    app = _make_app(provider)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/agent/run", json={"query": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "empty" in data["summary"].lower()


@pytest.mark.asyncio
async def test_run_text_response():
    provider = MockProvider(events=[ResultEvent(text="Just plain text, not JSON")])
    app = _make_app(provider)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/agent/run", json={"query": "test"})
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

        body = resp.text
        assert "event: status" in body
        assert "event: text" in body
        assert "event: tool_call" in body
        assert "event: tool_result" in body
        assert "event: done" in body


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
        lines = resp1.text.strip().split("\n")
        done_line = [line for line in lines if '"conversationId"' in line]
        assert done_line
        conv_id = json.loads(done_line[0].replace("data: ", ""))["conversationId"]

        resp2 = await client.post(
            "/v1/agent/chat",
            json={
                "message": "Follow-up",
                "conversationId": conv_id,
                "context": {
                    "remediation": {"name": "r", "namespace": "ns"},
                    "alert": {"name": "a", "status": "firing", "severity": "low"},
                },
            },
        )
        assert resp2.status_code == 200
        assert conv_id in resp2.text


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
        body = resp.text
        assert "event: ui_component" in body
        assert "visualization" in body
