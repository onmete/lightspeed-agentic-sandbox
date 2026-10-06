"""Capped developer diagnostics for normalized provider events."""

from __future__ import annotations

import logging

from lightspeed_agentic.types import ProviderEvent

logger = logging.getLogger("lightspeed_agentic")

MAX_THINKING_LOG = 2_000
MAX_TOOL_INPUT_LOG = 500
MAX_TOOL_OUTPUT_LOG = 1_000
MAX_RESULT_LOG = 500
THINKING_BUF_FLUSH = 50_000


class EventLogger:
    """Log bounded provider diagnostics and token counts from provider events."""

    def __init__(self, phase: str) -> None:
        self._phase = phase
        self._thinking: list[str] = []
        self._thinking_len = 0

    def _flush_thinking(self) -> None:
        if not self._thinking_len:
            return

        thinking = "".join(self._thinking).strip()[:MAX_THINKING_LOG]
        self._thinking.clear()
        self._thinking_len = 0
        if thinking:
            logger.info("[provider:%s] thinking: %r", self._phase, thinking)

    def log(self, event: ProviderEvent) -> None:
        match event.type:
            case "thinking_delta":
                if event.thinking:
                    self._thinking.append(event.thinking)
                    self._thinking_len += len(event.thinking)
                    if self._thinking_len >= THINKING_BUF_FLUSH:
                        self._flush_thinking()
            case "content_block_stop":
                self._flush_thinking()
            case "tool_call":
                self._flush_thinking()
                logger.info(
                    "[provider:%s] tool_use: %r args=%r",
                    self._phase,
                    event.name,
                    event.input[:MAX_TOOL_INPUT_LOG],
                )
            case "tool_result":
                logger.info(
                    "[provider:%s] tool_result: %r",
                    self._phase,
                    event.output[:MAX_TOOL_OUTPUT_LOG],
                )
            case "result":
                self._flush_thinking()
                logger.info(
                    "[provider:%s] result: tokens=%d",
                    self._phase,
                    event.input_tokens + event.output_tokens,
                )
                output = event.text.strip()[:MAX_RESULT_LOG]
                if output:
                    logger.info("[provider:%s] output: %r", self._phase, output)
