"""FastAPI routers — mount into any FastAPI app.

Usage:
    from lightspeed_agentic.routes import build_router
    app.include_router(build_router(provider), prefix="/v1/agent")
"""

from __future__ import annotations

import os

from fastapi import APIRouter

from lightspeed_agentic.routes.query import register_query_routes
from lightspeed_agentic.types import DEFAULT_MODEL, AgentProvider


def build_router(
    provider: AgentProvider,
    *,
    skills_dir: str = "/app/skills",
    model: str | None = None,
    max_turns: int = 200,
    default_timeout_ms: int = 300_000,
) -> APIRouter:
    model_env_vars = {
        "claude": "ANTHROPIC_MODEL",
        "gemini": "GEMINI_MODEL",
        "openai": "OPENAI_MODEL",
        "deepagents": "DEEPAGENTS_MODEL",
    }
    env_var = model_env_vars.get(provider.name, "ANTHROPIC_MODEL")
    resolved_model = model or os.environ.get(env_var, DEFAULT_MODEL)

    router = APIRouter()
    register_query_routes(
        router,
        provider=provider,
        skills_dir=skills_dir,
        model=resolved_model,
        max_turns=max_turns,
        default_timeout_ms=default_timeout_ms,
    )
    return router
