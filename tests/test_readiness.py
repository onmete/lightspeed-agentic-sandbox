"""Tests for GET /readiness (OLS-3060)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from lightspeed_agentic.app import app
from lightspeed_agentic.readiness import (
    check_mcp_servers,
    check_provider_credentials,
    clear_readiness_cache,
    run_readiness_checks,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_readiness_cache()
    yield
    clear_readiness_cache()


def test_health_unchanged() -> None:
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}


def test_readiness_ok(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "demo-skill.md").write_text("# skill", encoding="utf-8")
    monkeypatch.setenv("LIGHTSPEED_SKILLS_DIR", str(skills))
    monkeypatch.setenv("LIGHTSPEED_AGENT_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("LIGHTSPEED_MCP_SERVERS", raising=False)
    monkeypatch.setenv("LIGHTSPEED_READINESS_CACHE_TTL_SECONDS", "0")

    client = TestClient(app)
    resp = client.get("/readiness")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["checks"]["skills_dir"] == "ok"
    assert data["checks"]["provider_credentials"] == "ok"
    assert data["checks"]["mcp_servers"] == "skipped: not configured"


def test_readiness_fails_when_skills_missing(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = tmp_path / "empty-skills"
    empty.mkdir()
    monkeypatch.setenv("LIGHTSPEED_SKILLS_DIR", str(empty))
    monkeypatch.setenv("LIGHTSPEED_AGENT_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LIGHTSPEED_READINESS_CACHE_TTL_SECONDS", "0")

    client = TestClient(app)
    resp = client.get("/readiness")
    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "error"
    assert data["checks"]["skills_dir"].startswith("error:")


def test_readiness_fails_without_credentials(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_AGENT_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LIGHTSPEED_READINESS_CACHE_TTL_SECONDS", "0")

    report = run_readiness_checks(str(tmp_path))
    assert report.checks["provider_credentials"].startswith("error:")


def test_readiness_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_READINESS_CACHE_TTL_SECONDS", "60")
    with patch(
        "lightspeed_agentic.readiness.check_skills_dir",
        side_effect=["ok", "error: should be cached"],
    ):
        first = run_readiness_checks("/app/skills")
        second = run_readiness_checks("/app/skills")
    assert first.checks["skills_dir"] == "ok"
    assert second.checks["skills_dir"] == "ok"


def test_check_mcp_skipped_when_unconfigured() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("LIGHTSPEED_MCP_SERVERS", None)
        assert check_mcp_servers() == "skipped: not configured"


def test_check_mcp_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_MCP_SERVERS", "http://127.0.0.1:1")
    result = check_mcp_servers()
    assert result.startswith("error:")


def test_claude_vertex_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_CODE_USE_VERTEX", "1")
    monkeypatch.delenv("ANTHROPIC_VERTEX_PROJECT_ID", raising=False)
    assert check_provider_credentials("claude").startswith("error:")
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "my-project")
    assert check_provider_credentials("claude") == "ok"
