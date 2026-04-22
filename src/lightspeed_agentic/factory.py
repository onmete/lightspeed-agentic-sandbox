"""Provider factory — maps to lightspeed-agent/src/providers/factory.ts."""

from __future__ import annotations

import os
from typing import Literal

from lightspeed_agentic.types import AgentProvider

ProviderName = Literal["claude", "gemini", "openai"]


def create_provider(name: str | None = None) -> AgentProvider:
    provider_name: ProviderName = (
        name or os.environ.get("LIGHTSPEED_AGENT_PROVIDER", "claude")
    )  # type: ignore[assignment]

    match provider_name:
        case "claude":
            from lightspeed_agentic.providers.claude import ClaudeProvider
            return ClaudeProvider()
        case "gemini":
            from lightspeed_agentic.providers.gemini import GeminiProvider
            return GeminiProvider()
        case "openai":
            from lightspeed_agentic.providers.openai import OpenAIProvider
            return OpenAIProvider()
        case _:
            raise ValueError(
                f"Unknown provider: {provider_name}. Supported: claude, gemini, openai"
            )
