"""OpenAI provider — wraps openai-agents SDK.

Maps to lightspeed-agent/src/providers/openai.ts.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from typing import Any

from lightspeed_agentic.tools import (
    discover_openai_skills,
    parse_bash_restrictions,
    validate_bash_command,
)
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


def _extract_tokens(usage: Any) -> tuple[int, int]:
    if hasattr(usage, "input_tokens"):
        return getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0)
    if isinstance(usage, dict):
        return usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    return 0, 0


def _build_result(result: Any) -> ResultEvent:
    usage = getattr(result, "usage", None) or {}
    input_tokens, output_tokens = _extract_tokens(usage)
    return ResultEvent(
        text=stringify(result.final_output),
        cost_usd=0,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _build_shell_executor(cwd: str, patterns: list[str] | None) -> Any:
    from agents import ShellCallOutcome, ShellCommandOutput, ShellCommandRequest, ShellResult

    async def executor(request: ShellCommandRequest) -> ShellResult:
        action = request.data.action
        outputs: list[ShellCommandOutput] = []

        for command in action.commands:
            if not validate_bash_command(command, patterns):
                outputs.append(ShellCommandOutput(
                    command=command,
                    stdout="",
                    stderr=f"Command not allowed. Permitted prefixes: {', '.join(patterns or [])}",
                    outcome=ShellCallOutcome(type="exit", exit_code=1),
                ))
                continue

            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            timed_out = False
            try:
                timeout = (action.timeout_ms or 0) / 1000 or None
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                stdout_bytes, stderr_bytes = await proc.communicate()
                timed_out = True

            outputs.append(ShellCommandOutput(
                command=command,
                stdout=stdout_bytes.decode("utf-8", errors="ignore"),
                stderr=stderr_bytes.decode("utf-8", errors="ignore"),
                outcome=ShellCallOutcome(
                    type="timeout" if timed_out else "exit",
                    exit_code=getattr(proc, "returncode", None),
                ),
            ))

            if timed_out:
                break

        return ShellResult(
            output=outputs,
            provider_data={"working_directory": cwd},
        )

    return executor


def _ensure_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """OpenAI strict mode requires additionalProperties:false on every object."""
    schema = dict(schema)
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
        if "properties" in schema:
            schema["properties"] = {
                k: _ensure_strict_schema(v) for k, v in schema["properties"].items()
            }
    if schema.get("type") == "array" and "items" in schema:
        schema["items"] = _ensure_strict_schema(schema["items"])
    return schema


def _build_output_schema(schema: dict[str, Any]) -> Any:
    """Wrap raw JSON Schema for the OpenAI SDK's AgentOutputSchemaBase."""
    from agents.agent_output import AgentOutputSchemaBase
    from agents.exceptions import ModelBehaviorError

    strict_schema = _ensure_strict_schema(schema)

    class RawJsonSchemaOutput(AgentOutputSchemaBase):
        def __init__(self, json_schema: dict[str, Any]) -> None:
            self._schema = json_schema

        def is_plain_text(self) -> bool:
            return False

        def name(self) -> str:
            return "agent_output"

        def json_schema(self) -> dict[str, Any]:
            return self._schema

        def is_strict_json_schema(self) -> bool:
            return True

        def validate_json(self, json_str: str) -> Any:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                raise ModelBehaviorError(f"Invalid JSON output: {e}") from e

    return RawJsonSchemaOutput(strict_schema)


class OpenAIProvider(AgentProvider):
    @property
    def name(self) -> str:
        return "openai"

    async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
        from agents import Agent, Runner, ShellTool

        bash_allowed, patterns = parse_bash_restrictions(options.allowed_tools)

        tools: list[Any] = []
        if bash_allowed or "Skill" in options.allowed_tools:
            executor = _build_shell_executor(options.cwd, patterns)
            skills = (
                discover_openai_skills(options.cwd)
                if "Skill" in options.allowed_tools
                else []
            )
            environment: dict[str, Any] = {"type": "local"}
            if skills:
                environment["skills"] = skills

            tools.append(ShellTool(
                executor=executor,
                environment=environment,
                needs_approval=False,
            ))

        output_schema = _build_output_schema(options.output_schema) if options.output_schema else None

        agent = Agent(
            name="lightspeed",
            instructions=options.system_prompt,
            model=options.model,
            tools=tools,
            **({"output_type": output_schema} if output_schema else {}),
        )

        if options.stream:
            result = Runner.run_streamed(
                agent, options.prompt, max_turns=options.max_turns
            )

            async for event in result.stream_events():
                if event.type == "raw_response_event":
                    data = event.data
                    if (
                        hasattr(data, "type")
                        and data.type == "response.output_text.delta"
                        and hasattr(data, "delta")
                    ):
                        yield TextDeltaEvent(text=data.delta)

                elif event.type == "run_item_stream_event":
                    item = event.item
                    item_type = getattr(item, "type", "")

                    if item_type == "tool_call_item":
                        yield ToolCallEvent(
                            name=getattr(item, "name", "unknown"),
                            input=json.dumps(
                                getattr(item, "arguments", {}) or {}
                            )[:TOOL_INPUT_MAX_CHARS],
                        )
                    elif item_type == "tool_call_output_item":
                        yield ToolResultEvent(
                            output=(getattr(item, "output", "") or "")[:TOOL_OUTPUT_MAX_CHARS]
                        )

            yield ContentBlockStopEvent()
            yield _build_result(result)

        else:
            result = await Runner.run(
                agent, options.prompt, max_turns=options.max_turns
            )
            yield _build_result(result)
