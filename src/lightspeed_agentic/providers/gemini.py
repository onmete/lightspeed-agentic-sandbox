"""Gemini provider — wraps google-adk.

Uses native ExecuteBashTool for shell execution and SkillToolset for
skill discovery. The SDK handles tool registration and command execution.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import shlex
import time
from collections.abc import AsyncGenerator, AsyncIterator
from copy import deepcopy
from typing import Any
from uuid import uuid4

from lightspeed_agentic.types import (
    MAX_TOOL_RETURN_CHARS,
    TOOL_RETURN_PREVIEW_CHARS,
    AgentProvider,
    ContentBlockStopEvent,
    ProviderEvent,
    ProviderQueryOptions,
    ResultEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    stringify,
)

logger = logging.getLogger(__name__)


def _trim_tool_response(
    tool: Any,
    args: dict[str, Any],
    tool_context: Any,
    tool_response: Any,
) -> Any:
    """Replace oversized Gemini tool results with a bounded preview."""
    _ = tool, args, tool_context
    serialized = stringify(tool_response)
    if len(serialized) <= MAX_TOOL_RETURN_CHARS:
        return None

    return {
        "status": "truncated",
        "preview": serialized[:TOOL_RETURN_PREVIEW_CHARS],
        "original_size": len(serialized),
        "message": "Tool output was truncated; request a narrower result if needed.",
    }


def _load_skills_toolset(skills_dir: str) -> Any:
    try:
        from google.adk.code_executors.unsafe_local_code_executor import (
            UnsafeLocalCodeExecutor,
        )
        from google.adk.skills import list_skills_in_dir, load_skill_from_dir
        from google.adk.tools.skill_toolset import SkillToolset

        target = pathlib.Path(skills_dir)
        skill_entries = list_skills_in_dir(target)
        skills = [
            load_skill_from_dir(target / skill_id)
            for skill_id in skill_entries
            if (target / skill_id).is_dir()
        ]
        if skills:
            return SkillToolset(
                skills=skills,
                code_executor=UnsafeLocalCodeExecutor(),  # type: ignore[no-untyped-call]
            )
    except Exception as e:
        logger.debug("Failed to load skills toolset from %s: %s", skills_dir, e)
    return None


def _messages(contents: Any) -> list[dict[str, Any]]:
    """Convert the ordered Gemini content stream without flattening its parts."""
    messages: list[dict[str, Any]] = []
    for content in contents or []:
        role = {"model": "assistant", "user": "user"}.get(content.role, content.role or "user")
        parts: list[dict[str, Any]] = []
        for part in content.parts or []:
            if part.text is not None:
                parts.append(
                    {
                        "type": "reasoning" if part.thought else "text",
                        "content": part.text,
                    }
                )
            if part.function_call is not None:
                call = part.function_call
                if not call.id:
                    call.id = f"call-{uuid4()}"
                parts.append(
                    {
                        "type": "tool_call",
                        "id": call.id,
                        "name": call.name or "",
                        "arguments": dict(call.args) if call.args is not None else {},
                    }
                )
            if part.function_response is not None:
                response = part.function_response
                parts.append(
                    {
                        "type": "tool_call_response",
                        "id": response.id or "",
                        "response": response.response,
                    }
                )
        if parts:
            message_role = (
                "tool" if any(part["type"] == "tool_call_response" for part in parts) else role
            )
            messages.append({"role": message_role, "parts": parts})
    return messages


def _finish_reason(response: Any, has_calls: bool, observed_reason: Any = None) -> str:
    reason = getattr(response, "finish_reason", None)
    if reason is None:
        reason = observed_reason
    reason = getattr(reason, "name", reason)
    reason = str(reason).upper() if reason is not None else ""
    if getattr(response, "error_code", None) is not None or getattr(response, "interrupted", False):
        return "error"
    if reason in ("MAX_TOKENS", "MAX_OUTPUT_TOKENS"):
        return "length"
    if reason in (
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
        "MODEL_ARMOR",
    ):
        return "content_filter"
    if reason in ("ERROR", "OTHER", "MALFORMED_FUNCTION_CALL"):
        return "error"
    if reason in ("TOOL_CALL", "FUNCTION_CALL") or has_calls:
        return "tool_call"
    if reason == "STOP":
        return "stop"
    # Required by the v1.41 output schema; do not invent normal completion.
    return "unknown"


def _tool_definitions(config: Any) -> list[dict[str, Any]] | None:
    definitions = []
    for tool in getattr(config, "tools", None) or []:
        for declaration in getattr(tool, "function_declarations", None) or []:
            values = declaration.model_dump(mode="json", exclude_none=True)
            definitions.append({"type": "function", **values})
    return definitions or None


class GeminiProvider(AgentProvider):
    def __init__(self) -> None:
        self._cached_skills: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "gemini"

    async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
        from google.adk.agents import Agent, RunConfig
        from google.adk.agents.run_config import StreamingMode
        from google.adk.models import Gemini
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.adk.telemetry.context import ContentCapturingMode, TelemetryConfig
        from google.adk.tools import (  # type: ignore[attr-defined]
            exit_loop,
            google_search,
            url_context,
        )
        from google.adk.tools.bash_tool import ExecuteBashTool
        from google.adk.tools.tool_confirmation import ToolConfirmation
        from google.genai import types
        from pydantic import PrivateAttr

        from lightspeed_agentic.tls import get_ssl_context

        workspace = pathlib.Path(options.cwd)

        bash = ExecuteBashTool(workspace=workspace)
        _orig_run = bash.run_async

        async def _auto_confirm_run(*, args: Any, tool_context: Any) -> Any:
            tool_context.tool_confirmation = ToolConfirmation(confirmed=True)
            # ExecuteBashTool uses subprocess_exec (no shell), so wrap through
            # bash -c to support shell builtins, PATH lookups, and pipes.
            if "command" in args:
                args = {**args, "command": f"bash -c {shlex.quote(args['command'])}"}
            return await _orig_run(args=args, tool_context=tool_context)

        bash.run_async = _auto_confirm_run  # type: ignore[method-assign]

        # TODO: investigate more ADK built-in tools:
        # load_artifacts, load_memory, computer_use, file_search, mcp_servers
        is_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").upper() == "TRUE"
        tools: list[Any] = [bash]
        # Vertex AI rejects mixing search tools (google_search, url_context)
        # with non-search tools like bash in the same request.
        if not is_vertex:
            tools.extend([google_search, url_context])

        if options.cwd not in self._cached_skills:
            self._cached_skills[options.cwd] = _load_skills_toolset(options.cwd)
        skill_toolset = self._cached_skills[options.cwd]
        if skill_toolset is not None:
            tools.append(skill_toolset)

        mcp_toolsets: list[Any] = []
        if options.mcp_servers:
            from lightspeed_agentic.mcp import to_gemini_mcp_toolsets

            mcp_toolsets = to_gemini_mcp_toolsets(options.mcp_servers)
            try:
                for toolset in mcp_toolsets:
                    tools.extend(await toolset.get_tools())
            except BaseException:
                for toolset in mcp_toolsets:
                    await toolset.close()
                raise

        if not options.output_schema:
            tools.append(exit_loop)

        tool_config_kwargs: dict[str, Any] = {}
        if not is_vertex:
            tool_config_kwargs["include_server_side_tool_invocations"] = True

        gen_content_kwargs: dict[str, Any] = {
            "tool_config": types.ToolConfig(**tool_config_kwargs),
        }

        if options.reasoning_config:
            gen_content_kwargs["thinking_config"] = types.ThinkingConfig(**options.reasoning_config)

        telemetry = options.telemetry

        class ObservedGemini(Gemini):
            """Observe the request after ADK's Gemini-specific preprocessing."""

            _observed_handle: Any = PrivateAttr(default=None)

            def _maybe_append_user_content(self, llm_request: Any) -> None:
                super()._maybe_append_user_content(llm_request)
                if telemetry is None:
                    return
                instruction = getattr(llm_request.config, "system_instruction", None)
                if isinstance(instruction, str):
                    system = [{"type": "text", "content": instruction}]
                elif instruction is not None:
                    system = [
                        part for message in _messages([instruction]) for part in message["parts"]
                    ] or None
                else:
                    system = None
                self._observed_handle = telemetry.start_model(
                    _messages(llm_request.contents),
                    system,
                    llm_request.model,
                    operation_name="generate_content",
                    tool_definitions=_tool_definitions(llm_request.config),
                )

            async def generate_content_async(
                self, llm_request: Any, stream: bool = False
            ) -> AsyncGenerator[Any, None]:
                self._observed_handle = None
                final = None
                observed_usage: dict[str, int] = {}
                observed_model = None
                observed_reason = None
                try:
                    async for response in super().generate_content_async(
                        llm_request, stream=stream
                    ):
                        usage = getattr(response, "usage_metadata", None)
                        for key, field in (
                            ("input_tokens", "prompt_token_count"),
                            ("output_tokens", "candidates_token_count"),
                            ("reasoning_tokens", "thoughts_token_count"),
                        ):
                            value = getattr(usage, field, None)
                            if value is not None:
                                observed_usage[key] = value
                        observed_model = getattr(response, "model_version", None) or observed_model
                        observed_reason = (
                            getattr(response, "finish_reason", None) or observed_reason
                        )
                        if not response.partial:
                            final = response
                            # Assign missing IDs before ADK clones/finalizes this response.
                            if response.content:
                                _messages([response.content])
                        yield response
                except BaseException as error:
                    if telemetry is not None and self._observed_handle is not None:
                        telemetry.end_model(self._observed_handle, None, None, {}, error)
                        self._observed_handle = None
                    raise
                else:
                    if telemetry is not None and self._observed_handle is not None:
                        output = _messages([final.content]) if final and final.content else None
                        if output is not None:
                            for message in output:
                                message["finish_reason"] = _finish_reason(
                                    final,
                                    any(part["type"] == "tool_call" for part in message["parts"]),
                                    observed_reason,
                                )
                        failure = None
                        if final is None:
                            failure = RuntimeError("incomplete model response")
                        elif getattr(final, "error_code", None) is not None:
                            failure = RuntimeError(str(final.error_code))
                        elif getattr(final, "interrupted", False):
                            failure = RuntimeError("model response interrupted")
                        telemetry.end_model(
                            self._observed_handle,
                            output,
                            observed_model,
                            observed_usage,
                            failure,
                        )
                        self._observed_handle = None

        gemini_model = ObservedGemini(
            model=options.model,
            client_kwargs={
                "http_options": types.HttpOptions(
                    async_client_args={"verify": get_ssl_context()},
                ),
            },
        )
        active_tools: dict[str, Any] = {}

        def _before_tool(tool: Any, args: dict[str, Any], tool_context: Any) -> None:
            if telemetry is not None:
                call_id = tool_context.function_call_id
                active_tools[call_id] = telemetry.start_tool(tool.name, call_id, deepcopy(args))

        def _after_tool(
            tool: Any, args: dict[str, Any], tool_context: Any, tool_response: Any
        ) -> None:
            _ = tool, args
            if telemetry is None:
                return
            handle = active_tools.pop(tool_context.function_call_id, None)
            if handle is None:
                return
            error = None
            if isinstance(tool_response, dict):
                if "error" in tool_response:
                    error = RuntimeError(str(tool_response["error"]))
                elif tool_response.get("status") in ("error", "failed"):
                    error = RuntimeError(str(tool_response["status"]))
                elif tool_response.get("isError") is True:
                    error = RuntimeError("MCP tool call failed")
            telemetry.end_tool(handle, tool_response if error is None else None, error)

        def _on_tool_error(
            tool: Any, args: dict[str, Any], tool_context: Any, error: Exception
        ) -> None:
            _ = tool, args
            if telemetry is not None:
                handle = active_tools.pop(tool_context.function_call_id, None)
                if handle is not None:
                    telemetry.end_tool(handle, None, error)

        agent_kwargs: dict[str, Any] = {
            "name": "lightspeed",
            "model": gemini_model,
            "instruction": options.system_prompt,
            "tools": tools,
            "after_tool_callback": [_after_tool, _trim_tool_response],
            "generate_content_config": types.GenerateContentConfig(**gen_content_kwargs),
        }
        if telemetry is not None:
            agent_kwargs["before_tool_callback"] = _before_tool
            agent_kwargs["on_tool_error_callback"] = _on_tool_error

        agent = Agent(**agent_kwargs)

        if options.output_schema:
            # Bypass ADK's output_schema (routes through broken SetModelResponseTool)
            # and use Gemini's native response_schema directly.
            gen_cfg = agent.generate_content_config
            if gen_cfg is not None:
                gen_cfg.response_mime_type = "application/json"
                gen_cfg.response_schema = options.output_schema

        session_service = InMemorySessionService()  # type: ignore[no-untyped-call]
        runner = Runner(
            app_name="lightspeed",
            agent=agent,
            session_service=session_service,
        )

        try:
            user_id = f"agent-{int(time.time())}"
        except (OSError, OverflowError, ValueError):
            user_id = "agent"
        session = await session_service.create_session(app_name="lightspeed", user_id=user_id)

        streaming_mode = StreamingMode.SSE if options.stream else StreamingMode.NONE
        run_config = RunConfig(
            streaming_mode=streaming_mode,
            max_llm_calls=options.max_turns,
            telemetry=TelemetryConfig(
                capture_message_content=ContentCapturingMode.NO_CONTENT,
                genai_semconv_stability_opt_in="stable",
            ),
        )

        result_text = ""
        total_input_tokens = 0
        total_output_tokens = 0

        try:
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
                    if (
                        hasattr(part, "thought")
                        and part.thought
                        and hasattr(part, "text")
                        and part.text
                    ):
                        yield ThinkingDeltaEvent(thinking=part.text)
                        continue

                    if hasattr(part, "text") and part.text:
                        if options.stream and is_partial:
                            yield TextDeltaEvent(text=part.text)
                        if not is_partial and not event.get_function_calls():
                            result_text = part.text

                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        yield ToolCallEvent(
                            name=fc.name or "",
                            input=json.dumps(dict(fc.args) if fc.args else {}),
                            call_id=getattr(fc, "id", "") or "",
                        )

                    if hasattr(part, "function_response") and part.function_response:
                        fr = part.function_response
                        yield ToolResultEvent(
                            output=stringify(fr.response),
                            call_id=getattr(fr, "id", "") or "",
                        )

                usage = getattr(event, "usage_metadata", None)
                if usage:
                    total_input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                    total_output_tokens = getattr(usage, "candidates_token_count", 0) or 0
        finally:
            for toolset in mcp_toolsets:
                await toolset.close()

        yield ContentBlockStopEvent()

        yield ResultEvent(
            text=result_text,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            response_model="",
        )
