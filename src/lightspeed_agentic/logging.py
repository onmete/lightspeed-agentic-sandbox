"""Normalized provider event logging — maps to lightspeed-agent/src/providers/logging.ts."""

from __future__ import annotations

import logging

from lightspeed_agentic.types import ProviderEvent

logger = logging.getLogger("lightspeed_agentic")


def log_provider_event(phase: str, event: ProviderEvent) -> None:
    match event.type:
        case "thinking_delta":
            logger.info("[provider:%s] thinking: %s", phase, event.thinking[:500])
        case "tool_call":
            logger.info("[provider:%s] tool_use: %s(%s)", phase, event.name, event.input)
        case "tool_result":
            logger.info("[provider:%s] tool_result: %s", phase, event.output)
        case "result":
            logger.info(
                "[provider:%s] result: cost=$%.4f, tokens=%d",
                phase,
                event.cost_usd,
                event.input_tokens + event.output_tokens,
            )
            if event.text:
                logger.info("[provider:%s] output: %s", phase, event.text[:500])
