"""E2E fixtures — provider parametrization, server URLs, workspace paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from .credentials import PROVIDER_NAMES, detect_all
from .runner import ChatResult, QueryResult
from .runner import run_analyze as _run_analyze
from .runner import run_chat as _run_chat
from .runner import run_execute as _run_execute
from .runner import run_verify as _run_verify


def _parse_env_map(var: str, _cache: dict[str, dict[str, str]] | None = None) -> dict[str, str]:
    if _cache is None:
        _cache = {}
    if var not in _cache:
        raw = os.environ.get(var, "")
        result: dict[str, str] = {}
        for entry in raw.split(","):
            entry = entry.strip()
            if "=" in entry:
                name, value = entry.split("=", 1)
                result[name.strip()] = value.strip()
        _cache[var] = result
    return _cache[var]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "provider" not in metafunc.fixturenames:
        return

    server_urls = _parse_env_map("EVAL_SERVER_URLS")
    creds = detect_all()
    params: list[Any] = []
    for name in PROVIDER_NAMES:
        if name not in server_urls:
            params.append(
                pytest.param(
                    name,
                    id=name,
                    marks=pytest.mark.skip(reason=f"No server for {name}"),
                )
            )
            continue
        status = creds[name]
        if not status.available:
            params.append(
                pytest.param(
                    name,
                    id=name,
                    marks=pytest.mark.skip(reason=status.reason),
                )
            )
            continue
        params.append(pytest.param(name, id=name))
    metafunc.parametrize("provider", params)


@pytest.fixture
def server_url(provider: str) -> str:
    return _parse_env_map("EVAL_SERVER_URLS")[provider]


@pytest.fixture
def eval_workspace(provider: str) -> Path:
    return Path(_parse_env_map("EVAL_WORKSPACES")[provider])


@pytest.fixture
def analyze_runner(server_url: str, provider: str):
    """Async callable that POSTs to /v1/agent/analyze."""

    async def _run(
        query: str,
        system_prompt: str = "You are a helpful assistant.",
        output_schema: dict | None = None,
    ) -> QueryResult:
        result = await _run_analyze(server_url, query, system_prompt, output_schema)
        result.provider = provider
        return result

    return _run


@pytest.fixture
def execute_runner(server_url: str, provider: str):
    """Async callable that POSTs to /v1/agent/execute."""

    async def _run(
        query: str,
        system_prompt: str = "You are a helpful assistant.",
        output_schema: dict | None = None,
        context: dict | None = None,
    ) -> QueryResult:
        result = await _run_execute(server_url, query, system_prompt, output_schema, context)
        result.provider = provider
        return result

    return _run


@pytest.fixture
def verify_runner(server_url: str, provider: str):
    """Async callable that POSTs to /v1/agent/verify."""

    async def _run(
        query: str,
        system_prompt: str = "You are a helpful assistant.",
        output_schema: dict | None = None,
        context: dict | None = None,
    ) -> QueryResult:
        result = await _run_verify(server_url, query, system_prompt, output_schema, context)
        result.provider = provider
        return result

    return _run


@pytest.fixture
def chat_runner(server_url: str, provider: str):
    """Async callable that POSTs to /v1/agent/chat."""

    async def _run(
        message: str,
        conversation_id: str | None = None,
    ) -> ChatResult:
        result = await _run_chat(server_url, message, conversation_id)
        result.provider = provider
        return result

    return _run
