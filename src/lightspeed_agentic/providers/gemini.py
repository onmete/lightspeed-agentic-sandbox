"""Gemini provider — wraps google-adk.

Maps to lightspeed-agent/src/providers/gemini.ts.

Key differences from the TS version:
  - Tools: plain Python callables (ADK auto-wraps via type hints)
  - Skills: native SkillToolset with load_skill_from_dir()
  - Streaming: RunConfig(streaming_mode=StreamingMode.SSE) → event.partial
  - output_schema disables tools in ADK — we embed schema in prompt instead
"""

from __future__ import annotations

import json
import pathlib
import time
from collections.abc import AsyncIterator

from lightspeed_agentic.tools import augment_system_prompt, build_gemini_tools
from lightspeed_agentic.types import (
    AgentProvider,
    ContentBlockStopEvent,
    ProviderEvent,
    ProviderQueryOptions,
    ResultEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)


def _load_skills_toolset(skills_dir: str) -> object | None:
    try:
        from google.adk.skills import list_skills_in_dir
        from google.adk.tools.skill_toolset import SkillToolset

        skills_path = pathlib.Path(skills_dir)
        skills_subdir = skills_path / "skills"
        target = skills_subdir if skills_subdir.is_dir() else skills_path

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
        tools: list[object] = list(tool_functions)

        if "Skill" in options.allowed_tools:
            skill_toolset = _load_skills_toolset(options.cwd)
            if skill_toolset is not None:
                tools.append(skill_toolset)

        system_prompt = augment_system_prompt(options.system_prompt, options.cwd)

        # ADK's output_schema disables tools — embed schema in prompt instead
        if options.output_schema:
            system_prompt += (
                "\n\nYou MUST respond with valid JSON matching this schema:\n"
                f"```json\n{json.dumps(options.output_schema, indent=2)}\n```\n"
                "Output ONLY the JSON object, no other text."
            )

        agent = Agent(
            name="lightspeed",
            model=options.model,
            instruction=system_prompt,
            tools=tools,
        )

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
                parts=[types.Part.from_text(options.prompt)],
            ),
            run_config=run_config,
        ):
            if not event.content or not event.content.parts:
                continue

            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    if options.stream and getattr(event, "partial", False):
                        yield TextDeltaEvent(text=part.text)
                    result_text = part.text if not getattr(event, "partial", False) else result_text

                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    yield ToolCallEvent(
                        name=fc.name,
                        input=json.dumps(dict(fc.args) if fc.args else {})[:300],
                    )

                if hasattr(part, "function_response") and part.function_response:
                    fr = part.function_response
                    response = fr.response
                    output = response if isinstance(response, str) else json.dumps(response)
                    yield ToolResultEvent(output=output[:500])

            usage = getattr(event, "usage_metadata", None)
            if usage:
                total_input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                total_output_tokens = getattr(usage, "candidates_token_count", 0) or 0

            # Accumulate final text from non-partial events
            if not getattr(event, "partial", False) and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text and not event.get_function_calls():
                        result_text = part.text

        if options.stream:
            yield ContentBlockStopEvent()

        yield ResultEvent(
            text=result_text,
            cost_usd=0,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )
