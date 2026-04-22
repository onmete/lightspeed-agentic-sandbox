"""OpenAI provider — wraps openai-agents SDK.

Maps to lightspeed-agent/src/providers/openai.ts.

Key differences from the TS version:
  - Shell: ShellTool with ShellExecutor callable (same pattern, Python native)
  - Skills: ShellToolLocalSkill dicts on environment.skills (native)
  - Structured output: output_type=PydanticModel (native, works with tools)
  - Streaming: Runner.run_streamed() → stream_events() async iterator
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path

from lightspeed_agentic.tools import (
    augment_system_prompt,
    discover_openai_skills,
    parse_bash_restrictions,
    validate_bash_command,
)
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


def _build_shell_executor(cwd: str, patterns: list[str] | None) -> object:
    """Build a ShellExecutor callable for OpenAI's ShellTool."""
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
                env=os.environ.copy(),
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


def _build_output_type(schema: dict) -> type | None:
    """Convert a JSON Schema dict to a Pydantic model for structured output."""
    try:
        from pydantic import create_model

        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        field_definitions: dict = {}

        type_map = {"string": str, "integer": int, "number": float, "boolean": bool}

        for field_name, field_schema in properties.items():
            field_type = type_map.get(field_schema.get("type", "string"), str)
            if field_schema.get("type") == "array":
                field_type = list
            elif field_schema.get("type") == "object":
                field_type = dict

            if field_name in required:
                field_definitions[field_name] = (field_type, ...)
            else:
                field_definitions[field_name] = (field_type | None, None)

        return create_model("AgentOutput", **field_definitions)
    except Exception:
        return None


class OpenAIProvider(AgentProvider):
    @property
    def name(self) -> str:
        return "openai"

    async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
        from agents import Agent, Runner, ShellTool

        bash_allowed, patterns = parse_bash_restrictions(options.allowed_tools)

        tools: list[object] = []
        if bash_allowed or "Skill" in options.allowed_tools:
            executor = _build_shell_executor(options.cwd, patterns)
            skills = (
                discover_openai_skills(options.cwd)
                if "Skill" in options.allowed_tools
                else []
            )
            environment: dict = {"type": "local"}
            if skills:
                environment["skills"] = skills

            tools.append(ShellTool(
                executor=executor,
                environment=environment,
                needs_approval=False,
            ))

        system_prompt = augment_system_prompt(options.system_prompt, options.cwd)

        output_type = None
        if options.output_schema:
            output_type = _build_output_type(options.output_schema)

        agent = Agent(
            name="lightspeed",
            instructions=system_prompt,
            model=options.model,
            tools=tools,
            **({"output_type": output_type} if output_type else {}),
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
                            )[:300],
                        )
                    elif item_type == "tool_call_output_item":
                        yield ToolResultEvent(
                            output=(getattr(item, "output", "") or "")[:500]
                        )

            yield ContentBlockStopEvent()

            final_output = result.final_output
            text = (
                final_output
                if isinstance(final_output, str)
                else json.dumps(final_output) if final_output else ""
            )
            usage = getattr(result, "usage", None) or {}

            yield ResultEvent(
                text=text,
                cost_usd=0,
                input_tokens=getattr(usage, "input_tokens", 0) if hasattr(usage, "input_tokens") else usage.get("input_tokens", 0) if isinstance(usage, dict) else 0,
                output_tokens=getattr(usage, "output_tokens", 0) if hasattr(usage, "output_tokens") else usage.get("output_tokens", 0) if isinstance(usage, dict) else 0,
            )

        else:
            result = await Runner.run(
                agent, options.prompt, max_turns=options.max_turns
            )

            final_output = result.final_output
            text = (
                final_output
                if isinstance(final_output, str)
                else json.dumps(final_output) if final_output else ""
            )
            usage = getattr(result, "usage", None) or {}

            yield ResultEvent(
                text=text,
                cost_usd=0,
                input_tokens=getattr(usage, "input_tokens", 0) if hasattr(usage, "input_tokens") else usage.get("input_tokens", 0) if isinstance(usage, dict) else 0,
                output_tokens=getattr(usage, "output_tokens", 0) if hasattr(usage, "output_tokens") else usage.get("output_tokens", 0) if isinstance(usage, dict) else 0,
            )
