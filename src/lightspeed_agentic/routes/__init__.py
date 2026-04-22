"""FastAPI routers — mount into any FastAPI app.

Maps to lightspeed-agent/src/agent.ts (query endpoints) and chat.ts (SSE chat).

Usage in lightspeed-service:
    from lightspeed_agentic.routes import build_router
    app.include_router(build_router(provider), prefix="/v1/agent")
"""

from __future__ import annotations

from fastapi import APIRouter

from lightspeed_agentic.routes.chat import register_chat_routes
from lightspeed_agentic.routes.query import register_query_routes
from lightspeed_agentic.types import AgentProvider


def build_router(
    provider: AgentProvider,
    *,
    skills_dir: str = "/app/skills",
    model: str | None = None,
    max_turns: int = 200,
    analysis_timeout_ms: int = 300_000,
    execution_timeout_ms: int = 600_000,
    chat_timeout_ms: int = 120_000,
    chat_max_turns: int = 30,
    chat_max_budget_usd: float = 1.0,
) -> APIRouter:
    router = APIRouter()

    register_query_routes(
        router,
        provider=provider,
        skills_dir=skills_dir,
        model=model,
        max_turns=max_turns,
        analysis_timeout_ms=analysis_timeout_ms,
        execution_timeout_ms=execution_timeout_ms,
    )

    register_chat_routes(
        router,
        provider=provider,
        skills_dir=skills_dir,
        model=model or "claude-opus-4-6",
        max_turns=chat_max_turns,
        timeout_ms=chat_timeout_ms,
        max_budget_usd=chat_max_budget_usd,
    )

    return router
