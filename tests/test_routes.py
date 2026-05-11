"""Tests for FastAPI routes using mock providers."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from lightspeed_agentic.routes import build_router
from lightspeed_agentic.types import ResultEvent

from .conftest import MockProvider


def _make_app(provider) -> FastAPI:
    app = FastAPI()
    router = build_router(provider, skills_dir="/workspace", model="test-model")
    app.include_router(router, prefix="/v1/agent")
    return app


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
async def test_run_with_timeout_applied():
    """Verify timeout_ms is actually used: a slow provider exceeds a 1ms timeout."""
    import asyncio

    class SlowProvider(MockProvider):
        async def query(self, _options):
            await asyncio.sleep(0.1)
            async for event in super().query(_options):
                yield event

    app = _make_app(SlowProvider())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/agent/run",
            json={"query": "test", "timeout_ms": 1},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "timed out" in data["summary"].lower()


@pytest.mark.asyncio
async def test_run_with_timeout_default():
    """Without timeout_ms the server default applies and the fast mock succeeds."""
    app = _make_app(MockProvider())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/agent/run",
            json={"query": "Diagnose"},
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
