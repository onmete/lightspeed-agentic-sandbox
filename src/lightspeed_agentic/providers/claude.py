"""Claude provider — wraps claude-agent-sdk.

Maps to lightspeed-agent/src/providers/claude.ts.

The SDK handles everything natively:
  - Tools: built-in (Bash, Read, Glob, Grep, Skill)
  - Skills: auto-discovered via skills parameter
  - Structured output: output_format → ResultMessage.structured_output
  - Streaming: include_partial_messages → StreamEvent with raw API deltas
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from lightspeed_agentic.types import (
    AgentProvider,
    ContentBlockStopEvent,
    ProviderEvent,
    ProviderQueryOptions,
    ResultEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)


class ClaudeProvider(AgentProvider):
    @property
    def name(self) -> str:
        return "claude"

    async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            StreamEvent,
            query,
        )

        sdk_options = ClaudeAgentOptions(
            model=options.model,
            max_turns=options.max_turns,
            max_budget_usd=options.max_budget_usd,
            system_prompt=options.system_prompt,
            allowed_tools=options.allowed_tools,
            permission_mode="bypassPermissions",
            cwd=options.cwd,
            skills="all",
            include_partial_messages=options.stream,
            **({"output_format": {
                "type": "json_schema",
                "schema": options.output_schema,
            }} if options.output_schema else {}),
        )

        async for msg in query(prompt=options.prompt, options=sdk_options):
            if isinstance(msg, StreamEvent):
                event = msg.event
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        yield TextDeltaEvent(text=delta["text"])
                    elif delta.get("type") == "thinking_delta" and delta.get("thinking"):
                        yield ThinkingDeltaEvent(thinking=delta["thinking"])
                elif event.get("type") == "content_block_stop":
                    yield ContentBlockStopEvent()
                continue

            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if getattr(block, "type", None) == "tool_use":
                        yield ToolCallEvent(
                            name=getattr(block, "name", ""),
                            input=json.dumps(getattr(block, "input", {}))[:300],
                        )

            if getattr(msg, "type", None) == "tool":
                for block in getattr(msg, "content", []):
                    if getattr(block, "type", None) == "tool_result":
                        content = getattr(block, "content", "")
                        output = content if isinstance(content, str) else json.dumps(content)
                        yield ToolResultEvent(output=output[:500])

            if isinstance(msg, ResultMessage):
                text = ""
                structured = getattr(msg, "structured_output", None)
                if structured is not None:
                    text = structured if isinstance(structured, str) else json.dumps(structured)
                elif getattr(msg, "result", None):
                    text = msg.result

                yield ResultEvent(
                    text=text,
                    cost_usd=getattr(msg, "total_cost_usd", 0) or 0,
                    input_tokens=getattr(getattr(msg, "usage", None), "input_tokens", 0),
                    output_tokens=getattr(getattr(msg, "usage", None), "output_tokens", 0),
                )
