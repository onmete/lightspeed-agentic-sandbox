"""Eval fixtures — provider parametrize, workspace, model defaults."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lightspeed_agentic.types import AgentProvider, ProviderQueryOptions

from .credentials import PROVIDER_NAMES, detect_all, detect_credentials
from .report import pytest_addoption, pytest_configure, store_eval_result  # noqa: F401
from .runner import EvalResult, run_eval as _run_eval

_DEFAULT_MODELS: dict[str, str] = {
    "claude": "claude-sonnet-4-6",
    "gemini": "gemini-3.1-flash-lite-preview",
    "openai": "gpt-5.4",
    "deepagents": "claude-opus-4-6",
}

_MODEL_ENV_VARS: dict[str, str] = {
    "claude": "ANTHROPIC_MODEL",
    "gemini": "GEMINI_MODEL",
    "openai": "OPENAI_MODEL",
    "deepagents": "DEEPAGENTS_MODEL",
}


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "provider_name" not in metafunc.fixturenames:
        return

    creds = detect_all()
    params = []
    for name in PROVIDER_NAMES:
        status = creds[name]
        if status.available:
            params.append(pytest.param(name, id=name))
        else:
            params.append(pytest.param(
                name, id=name,
                marks=pytest.mark.skip(reason=status.reason),
            ))
    metafunc.parametrize("provider_name", params)


@pytest.fixture
def provider(provider_name: str) -> AgentProvider:
    from lightspeed_agentic.factory import create_provider

    # Inject provider-specific env vars (e.g., GEMINI_API_KEY → GOOGLE_API_KEY)
    status = detect_credentials(provider_name)
    for key, value in status.env_vars.items():
        os.environ.setdefault(key, value)

    try:
        return create_provider(provider_name)
    except ImportError as e:
        pytest.skip(f"{provider_name} SDK not installed: {e}")


@pytest.fixture
def eval_workspace() -> Path:
    return Path(__file__).parent / "workspace"


@pytest.fixture
def default_model(provider_name: str) -> str:
    env_var = _MODEL_ENV_VARS.get(provider_name, "")
    return os.environ.get(env_var, _DEFAULT_MODELS.get(provider_name, ""))


@pytest.fixture
def eval_runner(provider: AgentProvider, request: pytest.FixtureRequest):
    """Returns an async callable that runs an eval and stores the result for reporting."""

    async def _run(options: ProviderQueryOptions) -> EvalResult:
        result = await _run_eval(provider, options)
        store_eval_result(request.node, result)
        return result

    return _run
