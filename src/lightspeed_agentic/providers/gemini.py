"""Gemini provider — wraps google-adk.

Maps to lightspeed-agent/src/providers/gemini.ts.
"""

from __future__ import annotations

import json
import pathlib
import time
from collections.abc import AsyncIterator
from typing import Any

from lightspeed_agentic.tools import build_gemini_tools, resolve_skills_dir
from lightspeed_agentic.types import (
    TOOL_INPUT_MAX_CHARS,
    TOOL_OUTPUT_MAX_CHARS,
    AgentProvider,
    ContentBlockStopEvent,
    ProviderEvent,
    ProviderQueryOptions,
    ResultEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    stringify,
)


def _load_skills_toolset(skills_dir: str) -> Any:
    try:
        from google.adk.skills import list_skills_in_dir
        from google.adk.tools.skill_toolset import SkillToolset

        target = pathlib.Path(resolve_skills_dir(skills_dir))
        skills = list_skills_in_dir(target)
        if skills:
            return SkillToolset(skills=skills)
    except Exception:
        pass
    return None


class GeminiProvider(AgentProvider):
    @property
    def name(self) -> str:
        return "gemini"

    async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
        from google.adk.agents import Agent, RunConfig
        from google.adk.agents.run_config import StreamingMode
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types

        tool_functions = build_gemini_tools(options.allowed_tools, options.cwd)
        tools: list[Any] = list(tool_functions)

        if "Skill" in options.allowed_tools:
            skill_toolset = _load_skills_toolset(options.cwd)
            if skill_toolset is not None:
                tools.append(skill_toolset)

        agent_kwargs: dict[str, Any] = {
            "name": "lightspeed",
            "model": options.model,
            "instruction": options.system_prompt,
            "tools": tools,
        }

        if options.output_schema:
            agent_kwargs["output_schema"] = options.output_schema

        agent = Agent(**agent_kwargs)

        session_service = InMemorySessionService()
        runner = Runner(
            app_name="lightspeed",
            agent=agent,
            session_service=session_service,
        )

        user_id = f"agent-{int(time.time())}"
        session = await session_service.create_session(
            app_name="lightspeed", user_id=user_id
        )

        streaming_mode = StreamingMode.SSE if options.stream else StreamingMode.NONE
        run_config = RunConfig(
            streaming_mode=streaming_mode,
            max_llm_calls=options.max_turns,
        )

        result_text = ""
        total_input_tokens = 0
        total_output_tokens = 0

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=options.prompt)],
            ),
            run_config=run_config,
        ):
            if not event.content or not event.content.parts:
                continue

            is_partial = getattr(event, "partial", False)

            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    if options.stream and is_partial:
                        yield TextDeltaEvent(text=part.text)
                    if not is_partial and not event.get_function_calls():
                        result_text = part.text

                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    yield ToolCallEvent(
                        name=fc.name,
                        input=json.dumps(dict(fc.args) if fc.args else {})[:TOOL_INPUT_MAX_CHARS],
                    )

                if hasattr(part, "function_response") and part.function_response:
                    fr = part.function_response
                    yield ToolResultEvent(output=stringify(fr.response)[:TOOL_OUTPUT_MAX_CHARS])

            usage = getattr(event, "usage_metadata", None)
            if usage:
                total_input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                total_output_tokens = getattr(usage, "candidates_token_count", 0) or 0

        if options.stream:
            yield ContentBlockStopEvent()

        yield ResultEvent(
            text=result_text,
            cost_usd=0,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )
