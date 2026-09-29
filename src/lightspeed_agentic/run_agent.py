"""Shared agent execution for the batch entrypoint."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import SpanKind, StatusCode

from lightspeed_agentic.audit import GenAIRecorder
from lightspeed_agentic.genai_messages import encode_messages
from lightspeed_agentic.inspection.errors import ToolResultSafetyInspectionFailed
from lightspeed_agentic.logging import EventLogger
from lightspeed_agentic.mcp import AdmittedMCPProviderServer
from lightspeed_agentic.metrics import operation_duration, token_usage
from lightspeed_agentic.tools import DEFAULT_ALLOWED_TOOLS
from lightspeed_agentic.tracing import get_tracer, parse_traceparent
from lightspeed_agentic.types import (
    AgentProvider,
    ProviderQueryOptions,
    ToolCallEvent,
    ToolResultEvent,
)

logger = logging.getLogger("lightspeed_agentic")


@dataclass
class AgentResult:
    """Wraps agent output dict with token counts for Result CR publishing."""

    output: dict[str, Any] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    timed_out: bool = False


class ContextFormatError(ValueError):
    """``context`` JSON is present but missing fields required for prefix formatting."""


def _developer_log_event(provider_name: str, event: Any) -> Any:
    """Remove DeepAgents tool payloads from developer logs, retaining safe metadata."""
    if provider_name != "deepagents":
        return event
    if event.type == "tool_call":
        return ToolCallEvent(name=event.name, call_id=event.call_id)
    if event.type == "tool_result":
        return ToolResultEvent(call_id=event.call_id)
    return event


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContextFormatError(f"Invalid context: {path} must be a JSON object")
    return value


def _require_non_empty_str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContextFormatError(f"Invalid context: {path} must be a non-empty string")
    return value


def format_context_prefix(context: dict[str, Any]) -> str:
    """Format context fields as a prefix block prepended to the query text."""
    if not isinstance(context, dict):
        raise ContextFormatError("Invalid context: must be a JSON object")

    lines: list[str] = ["[context]"]

    namespaces = context.get("targetNamespaces")
    if namespaces:
        if not isinstance(namespaces, list):
            raise ContextFormatError("Invalid context: targetNamespaces must be a list")
        lines.append(f"Target namespaces: {', '.join(str(ns) for ns in namespaces)}")

    if (attempt := context.get("attempt")) is not None:
        lines.append(f"Attempt: {attempt} of max")

    prev = context.get("previousAttempts")
    if prev:
        if not isinstance(prev, list):
            raise ContextFormatError("Invalid context: previousAttempts must be a list")
        lines.append("Previous attempts:")
        for i, entry in enumerate(prev):
            if not isinstance(entry, dict):
                raise ContextFormatError(
                    f"Invalid context: previousAttempts[{i}] must be a JSON object"
                )
            attempt_no = entry.get("attempt")
            if attempt_no is None:
                raise ContextFormatError(f"Invalid context: previousAttempts[{i}] missing attempt")
            reason = f": {entry['failureReason']}" if entry.get("failureReason") else ""
            lines.append(f"  Attempt {attempt_no}{reason}")

    opt = context.get("approvedOption")
    if opt is not None:
        opt = _require_mapping(opt, "approvedOption")
        title = _require_non_empty_str(opt.get("title"), "approvedOption.title")
        diagnosis = _require_mapping(opt.get("diagnosis"), "approvedOption.diagnosis")
        root_cause = _require_non_empty_str(
            diagnosis.get("rootCause"),
            "approvedOption.diagnosis.rootCause",
        )
        plan = _require_mapping(opt.get("remediationPlan"), "approvedOption.remediationPlan")
        plan_description = _require_non_empty_str(
            plan.get("description"),
            "approvedOption.remediationPlan.description",
        )
        lines.append("")
        lines.append("=== APPROVED REMEDIATION (execute ONLY these actions) ===")
        lines.append(f"Title: {title}")
        lines.append(f"Diagnosis: {root_cause}")
        lines.append(f"Plan: {plan_description}")
        lines.append(f"Reversible: {plan.get('reversible', 'unknown')}")
        actions = plan.get("actions")
        if actions:
            if not isinstance(actions, list):
                raise ContextFormatError(
                    "Invalid context: approvedOption.remediationPlan.actions must be a list"
                )
            lines.append("Actions to execute:")
            for j, action in enumerate(actions):
                if not isinstance(action, dict):
                    raise ContextFormatError(
                        f"Invalid context: approvedOption.remediationPlan.actions[{j}] "
                        "must be a JSON object"
                    )
                action_type = _require_non_empty_str(
                    action.get("type"),
                    f"approvedOption.remediationPlan.actions[{j}].type",
                )
                action_description = _require_non_empty_str(
                    action.get("description"),
                    f"approvedOption.remediationPlan.actions[{j}].description",
                )
                if cmd := action.get("command"):
                    lines.append(f"  - [{action_type}] {cmd} — {action_description}")
                else:
                    lines.append(f"  - [{action_type}] {action_description}")
        lines.append("=== DO NOT perform any actions beyond what is listed above ===")
        lines.append("")

    lines.append("[/context]")
    return "\n".join(lines)


async def run_agent_query(
    provider: AgentProvider,
    *,
    prompt: str,
    system_prompt: str,
    output_schema: dict[str, Any] | None,
    context: dict[str, Any] | None,
    skills_dir: str,
    model: str,
    max_turns: int,
    timeout_seconds: int,
    mcp_servers: list[AdmittedMCPProviderServer] | None = None,
    reasoning_config: dict[str, Any] | None = None,
    tool_output_inspection_enabled: bool = True,
    agenticrun_uid: str = "",
    traceparent: str | None = None,
    step: str = "",
) -> AgentResult:
    """Run the provider agent and return structured output for Result CR publishing.

    The agent span is a child of the operator's W3C phase span when supplied.
    An absent or invalid traceparent starts a fresh trace; observed model and
    tool operations are sibling children of the agent span.
    """
    if context:
        try:
            prefix = format_context_prefix(context)
        except ContextFormatError as exc:
            return AgentResult(output={"success": False, "summary": str(exc)})
        prompt = f"{prefix}\n\n{prompt}"

    trace_id, trace_ctx = parse_traceparent(traceparent)
    tracer = get_tracer()
    route = os.environ.get("LIGHTSPEED_PROVIDER", "").strip().lower()
    endpoint_names = {
        "vertex": "gcp.vertex_ai",
        "azure": "azure.ai.openai",
        "bedrock": "aws.bedrock",
    }
    sdk_names = {"deepagents": "anthropic", "gemini": "gcp.gen_ai"}
    provider_name = endpoint_names.get(route, route or sdk_names.get(provider.name, provider.name))
    span_attrs: dict[str, Any] = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": "lightspeed",
        "gen_ai.provider.name": provider_name,
    }
    if model:
        span_attrs["gen_ai.request.model"] = model
    if step:
        span_attrs["agenticrun.phase"] = step
    if agenticrun_uid:
        span_attrs["agenticrun.uid"] = agenticrun_uid
    if output_schema is not None:
        span_attrs["gen_ai.output.type"] = "json"
    agent_span = tracer.start_span(
        "invoke_agent lightspeed",
        kind=SpanKind.INTERNAL,
        context=trace_ctx,
        attributes=span_attrs,
    )
    if agent_span.is_recording():
        agent_span.set_attribute(
            "gen_ai.input.messages",
            encode_messages([{"role": "user", "parts": [{"type": "text", "content": prompt}]}]),
        )
        if system_prompt:
            agent_span.set_attribute(
                "gen_ai.system_instructions",
                encode_messages([{"type": "text", "content": system_prompt}]),
            )
    span_ctx = trace.set_span_in_context(agent_span, trace_ctx)
    recorder = GenAIRecorder(
        phase=step,
        provider=provider_name,
        capture_content=agent_span.is_recording(),
        agenticrun_uid=agenticrun_uid,
        parent_context=span_ctx,
        output_type="json" if output_schema is not None else None,
    )
    logger.info(
        "[agent] Starting query (model=%s, provider=%s, trace_id=%s)",
        model,
        provider.name,
        trace_id,
    )
    start_time = time.monotonic()
    text = ""
    input_tokens = 0
    output_tokens = 0
    terminal = False

    def _record_metrics(*, elapsed: float) -> None:
        if input_tokens:
            token_usage.labels(
                gen_ai_token_type="input",  # noqa: S106
                gen_ai_request_model=model,
                gen_ai_provider_name=provider.name,
                gen_ai_operation_name="invoke_agent",
            ).observe(input_tokens)
        if output_tokens:
            token_usage.labels(
                gen_ai_token_type="output",  # noqa: S106
                gen_ai_request_model=model,
                gen_ai_provider_name=provider.name,
                gen_ai_operation_name="invoke_agent",
            ).observe(output_tokens)
        operation_duration.labels(
            gen_ai_request_model=model,
            gen_ai_provider_name=provider.name,
            gen_ai_operation_name="invoke_agent",
        ).observe(elapsed)

    try:

        async def run() -> None:
            nonlocal text, input_tokens, output_tokens, terminal
            token = otel_context.attach(span_ctx)
            try:
                result = provider.query(
                    ProviderQueryOptions(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=model,
                        max_turns=max_turns,
                        allowed_tools=DEFAULT_ALLOWED_TOOLS,
                        cwd=skills_dir,
                        output_schema=output_schema,
                        mcp_servers=mcp_servers or [],
                        reasoning_config=reasoning_config,
                        tool_output_inspection_enabled=tool_output_inspection_enabled,
                        deadline=time.monotonic() + timeout_seconds,
                        telemetry=recorder,
                    )
                )
                event_logger = EventLogger("run")
                async for event in result:
                    event_logger.log(_developer_log_event(provider.name, event))
                    if event.type == "result":
                        text = event.text
                        input_tokens = event.input_tokens
                        output_tokens = event.output_tokens
                        terminal = True
                        break
            finally:
                otel_context.detach(token)

        await asyncio.wait_for(run(), timeout=timeout_seconds)

        if not terminal or not text:
            # No usable provider result: no terminal agent output was produced.
            agent_span.set_attribute("error.type", "empty_response")
            agent_span.set_status(StatusCode.ERROR)
            return AgentResult(
                output={"success": False, "summary": "Agent returned empty response"},
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        try:
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise TypeError("expected dict")
            success = parsed.get("success", True)
        except (json.JSONDecodeError, TypeError):
            parsed = None
            success = True

        if parsed is not None:
            logger.info("[agent] query complete: success=%s", success)
            result = AgentResult(
                output={
                    "success": success,
                    "summary": parsed.get("summary", text),
                    **{k: v for k, v in parsed.items() if k not in ("success", "summary")},
                },
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        else:
            logger.info("[agent] query complete (text response)")
            result = AgentResult(
                output={"success": True, "summary": text},
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        if agent_span.is_recording():
            # "stop" describes the completed agent lifecycle, not a provider finish reason.
            agent_span.set_attribute(
                "gen_ai.output.messages",
                encode_messages(
                    [
                        {
                            "role": "assistant",
                            "parts": [
                                {
                                    "type": "text",
                                    "content": json.dumps(result.output, ensure_ascii=False),
                                }
                            ],
                            "finish_reason": "stop",
                        }
                    ]
                ),
            )
        return result
    except TimeoutError:
        agent_span.set_attribute("error.type", "TimeoutError")
        agent_span.set_status(StatusCode.ERROR)
        elapsed = time.monotonic() - start_time
        return AgentResult(
            output={
                "success": False,
                "summary": (
                    f"Agent invocation exceeded timeout of {timeout_seconds}s after {elapsed:.1f}s"
                ),
            },
            timed_out=True,
        )
    except ToolResultSafetyInspectionFailed as exc:
        agent_span.set_attribute("error.type", type(exc).__name__)
        agent_span.set_status(StatusCode.ERROR)
        raise
    except Exception as exc:
        agent_span.set_attribute("error.type", type(exc).__name__)
        agent_span.set_status(StatusCode.ERROR)
        logger.exception("[agent] query error")
        return AgentResult(output={"success": False, "summary": f"Agent error: {exc}"})
    finally:
        recorder.close()
        agent_span.end()
        _record_metrics(elapsed=time.monotonic() - start_time)
