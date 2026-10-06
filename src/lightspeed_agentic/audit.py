"""Shared OTel recorder for GenAI inference and tool lifecycle spans."""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import NonRecordingSpan, Span, SpanKind, StatusCode

from lightspeed_agentic.metrics import operation_duration, token_usage, tool_duration
from lightspeed_agentic.tracing import emit_audit_log_records, get_tracer
from lightspeed_agentic.types import ProviderEvent

_METRIC_ERROR_TYPES = frozenset(
    {
        "TimeoutError",
        "CancelledError",
        "operation_cancelled",
        "empty_response",
        "error",
        "response.error",
        "response.failed",
        "response.incomplete",
        "response_incomplete",
    }
)


@dataclass
class _OpenSpan:
    span: Span
    kind: Literal["inference", "tool"]
    started_monotonic_ns: int
    start_time: int | None
    model: str = ""
    operation: str = ""
    tool_name: str = ""


def resolve_provider_name(sdk_name: str) -> str:
    """Resolve an SDK/framework name to the routed GenAI endpoint provider."""
    sdk = sdk_name.lower()
    if sdk == "deepagents":
        if os.environ.get("CLAUDE_CODE_USE_VERTEX") == "1":
            return "gcp.vertex_ai"
        if os.environ.get("CLAUDE_CODE_USE_BEDROCK") == "1":
            return "aws.bedrock"
        return "anthropic"
    if sdk == "gemini":
        if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").upper() == "TRUE":
            return "gcp.vertex_ai"
        return "gcp.gemini"
    if sdk == "openai":
        configured_provider = os.environ.get("LIGHTSPEED_PROVIDER", "").strip().lower()
        if configured_provider == "azure" or (
            not configured_provider and os.environ.get("AZURE_OPENAI_ENDPOINT")
        ):
            return "azure.ai.openai"
        if configured_provider == "vertex":
            return "gcp.vertex_ai"
        return "openai"
    return sdk_name


def _json_default(value: Any) -> Any:
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, Enum):
        return value.value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_attribute(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, default=_json_default)


def _error_type(error: BaseException | str) -> str:
    return error if isinstance(error, str) else type(error).__name__


def _metric_error_type(error: BaseException | str) -> str:
    error_type = _error_type(error)
    return error_type if error_type in _METRIC_ERROR_TYPES else "_OTHER"


def _unique_span_handle(span: Span) -> Span:
    """Give invalid no-op spans unique handles without inventing span IDs."""
    span_context = span.get_span_context()
    if not span_context.is_valid:
        return NonRecordingSpan(span_context)
    return span


class AuditLogger:
    """Record native GenAI spans and buffer legacy choice-event logs."""

    def __init__(
        self,
        *,
        phase: str,
        model: str,
        provider: str,
        enabled: bool = True,
        capture_content: bool = False,
        agenticrun_uid: str = "",
    ) -> None:
        """Configure native-span metadata and optional legacy choice-event logs.

        ``enabled`` and ``capture_content`` control only normalized-event logs;
        source-span recording remains controlled by the tracer provider.
        """
        self._phase = phase
        self._model = model
        self._provider = provider
        self._enabled = enabled
        self._capture_content = capture_content
        self._agenticrun_uid = agenticrun_uid
        self._tracer = get_tracer()
        self._parent_context: Context = Context()
        self._event_log_context: Context | None = None
        self._event_log_recording = False
        self._event_log_records: list[dict[str, str]] = []
        self._text_buffer: list[str] = []
        self._thinking_buffer: list[str] = []
        self._open_spans: dict[int, _OpenSpan] = {}

    def set_parent_context(self, ctx: Context) -> None:
        """Set the agent context used for child spans and buffered audit logs."""
        self._parent_context = ctx
        self._event_log_context = ctx
        self._event_log_recording = trace.get_current_span(ctx).is_recording()

    def process_event(self, event: ProviderEvent) -> None:
        """Buffer normalized completion/reasoning for legacy audit log records."""
        if not self._enabled:
            return
        match event.type:
            case "text_delta":
                self._text_buffer.append(event.text)
            case "thinking_delta":
                self._thinking_buffer.append(event.thinking)
            case "content_block_stop" | "tool_call" | "result":
                self._flush_event_buffers()

    def _flush_event_buffers(self) -> None:
        """Queue completion before reasoning, matching the legacy choice flush."""
        if not self._enabled:
            self._text_buffer.clear()
            self._thinking_buffer.clear()
            return
        if self._event_log_context is None:
            span = trace.get_current_span()
            self._event_log_context = trace.set_span_in_context(span)
            self._event_log_recording = span.is_recording()
        if not self._event_log_recording:
            self._text_buffer.clear()
            self._thinking_buffer.clear()
            return
        if self._text_buffer:
            text = "".join(self._text_buffer)
            self._text_buffer.clear()
            if text:
                self._event_log_records.append(
                    {"gen_ai.completion": text} if self._capture_content else {}
                )
        if self._thinking_buffer:
            thinking = "".join(self._thinking_buffer)
            self._thinking_buffer.clear()
            if thinking:
                self._event_log_records.append(
                    {"gen_ai.reasoning_content": thinking} if self._capture_content else {}
                )

    def flush_event_logs(self) -> None:
        """Emit queued choice records after the enclosing agent span has ended."""
        context = self._event_log_context
        records = self._event_log_records
        recording = self._event_log_recording
        self._event_log_records = []
        self._event_log_context = None
        self._event_log_recording = False
        if not records or not self._enabled or not recording or context is None:
            return
        emit_audit_log_records(
            context,
            (("gen_ai.choice", attributes) for attributes in records),
        )

    def start_inference(
        self,
        *,
        model: str,
        operation: str,
        input_messages: list[dict[str, Any]] | None,
        system_instructions: list[dict[str, Any]] | None = None,
        tool_definitions: list[dict[str, Any]] | None = None,
        output_type: str | None = None,
        start_time: int | None = None,
    ) -> Span:
        """Start one CLIENT span at the actual SDK model-request boundary."""
        request_model = model or self._model
        attributes: dict[str, Any] = {
            "gen_ai.operation.name": operation,
            "gen_ai.provider.name": self._provider,
        }
        if request_model:
            attributes["gen_ai.request.model"] = request_model
        if output_type:
            attributes["gen_ai.output.type"] = output_type
        self._add_correlation(attributes)
        started_monotonic_ns = time.monotonic_ns()
        span = self._tracer.start_span(
            f"{operation} {request_model}" if request_model else operation,
            kind=SpanKind.CLIENT,
            context=self._parent_context,
            attributes=attributes,
            start_time=start_time,
        )
        span = _unique_span_handle(span)
        self._open_spans[id(span)] = _OpenSpan(
            span=span,
            kind="inference",
            started_monotonic_ns=started_monotonic_ns,
            start_time=start_time,
            model=request_model,
            operation=operation,
        )
        if span.is_recording():
            if input_messages is not None:
                span.set_attribute("gen_ai.input.messages", _json_attribute(input_messages))
            if system_instructions is not None:
                span.set_attribute(
                    "gen_ai.system_instructions", _json_attribute(system_instructions)
                )
            if tool_definitions is not None:
                span.set_attribute("gen_ai.tool.definitions", _json_attribute(tool_definitions))
        return span

    def end_inference(
        self,
        span: Span,
        *,
        output_messages: list[dict[str, Any]] | None = None,
        response_model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        finish_reasons: list[str] | None = None,
        error: BaseException | str | None = None,
        end_time: int | None = None,
    ) -> None:
        """End one inference span and record only observations supplied by its SDK."""
        end_monotonic_ns = time.monotonic_ns()
        state = self._open_spans.pop(id(span), None)
        if span.is_recording():
            if output_messages is not None:
                span.set_attribute("gen_ai.output.messages", _json_attribute(output_messages))
            if response_model:
                span.set_attribute("gen_ai.response.model", response_model)
            if input_tokens is not None:
                span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
            if output_tokens is not None:
                span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
            if reasoning_tokens is not None:
                span.set_attribute("gen_ai.usage.reasoning.output_tokens", reasoning_tokens)
            if finish_reasons is not None:
                span.set_attribute("gen_ai.response.finish_reasons", tuple(finish_reasons))
            if error is not None:
                self._set_error(span, error)
        if state is not None:
            duration = self._duration_seconds(state, end_time, end_monotonic_ns)
            if input_tokens is not None:
                token_usage.labels(
                    "input",
                    state.model,
                    self._provider,
                    state.operation,
                ).observe(input_tokens)
            if output_tokens is not None:
                token_usage.labels(
                    "output",
                    state.model,
                    self._provider,
                    state.operation,
                ).observe(output_tokens)
            operation_duration.labels(
                gen_ai_request_model=state.model,
                gen_ai_provider_name=self._provider,
                gen_ai_operation_name=state.operation,
                error_type=_metric_error_type(error) if error is not None else "",
            ).observe(duration)
        span.end(end_time=end_time)

    def start_tool(
        self,
        *,
        name: str,
        call_id: str = "",
        arguments: Any = None,
        tool_type: str = "function",
        start_time: int | None = None,
    ) -> Span:
        """Start one INTERNAL span at the actual SDK tool-execution boundary."""
        attributes: dict[str, Any] = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": name,
            "gen_ai.tool.type": tool_type,
        }
        if call_id:
            attributes["gen_ai.tool.call.id"] = call_id
        self._add_correlation(attributes)
        started_monotonic_ns = time.monotonic_ns()
        span = self._tracer.start_span(
            f"execute_tool {name}",
            kind=SpanKind.INTERNAL,
            context=self._parent_context,
            attributes=attributes,
            start_time=start_time,
        )
        span = _unique_span_handle(span)
        self._open_spans[id(span)] = _OpenSpan(
            span=span,
            kind="tool",
            started_monotonic_ns=started_monotonic_ns,
            start_time=start_time,
            tool_name=name,
        )
        if arguments is not None and span.is_recording():
            span.set_attribute("gen_ai.tool.call.arguments", _json_attribute(arguments))
        return span

    def end_tool(
        self,
        span: Span,
        *,
        result: Any = None,
        error: BaseException | str | None = None,
        end_time: int | None = None,
    ) -> None:
        """End one tool span; failed executions never expose a result attribute."""
        end_monotonic_ns = time.monotonic_ns()
        state = self._open_spans.pop(id(span), None)
        if span.is_recording():
            if error is None and result is not None:
                span.set_attribute("gen_ai.tool.call.result", _json_attribute(result))
            if error is not None:
                self._set_error(span, error)
        if state is not None:
            tool_duration.labels(gen_ai_tool_name=state.tool_name).observe(
                self._duration_seconds(state, end_time, end_monotonic_ns)
            )
        span.end(end_time=end_time)

    def close(self, error: BaseException | str = "operation_cancelled") -> None:
        """Flush partial log buffers and end only outstanding child spans."""
        self._flush_event_buffers()
        end_monotonic_ns = time.monotonic_ns()
        for state in tuple(self._open_spans.values()):
            self._open_spans.pop(id(state.span), None)
            if state.span.is_recording():
                self._set_error(state.span, error)
            if state.kind == "inference":
                operation_duration.labels(
                    gen_ai_request_model=state.model,
                    gen_ai_provider_name=self._provider,
                    gen_ai_operation_name=state.operation,
                    error_type=_metric_error_type(error),
                ).observe(self._duration_seconds(state, None, end_monotonic_ns))
            else:
                tool_duration.labels(gen_ai_tool_name=state.tool_name).observe(
                    self._duration_seconds(state, None, end_monotonic_ns)
                )
            state.span.end()

    def _add_correlation(self, attributes: dict[str, Any]) -> None:
        if self._agenticrun_uid:
            attributes["agenticrun.uid"] = self._agenticrun_uid
        if self._phase:
            attributes["agenticrun.phase"] = self._phase

    def correlation_attributes(self) -> dict[str, str]:
        attributes: dict[str, str] = {}
        self._add_correlation(attributes)
        return attributes

    @staticmethod
    def _duration_seconds(
        state: _OpenSpan,
        end_time: int | None,
        end_monotonic_ns: int,
    ) -> float:
        if state.start_time is not None and end_time is not None:
            return (end_time - state.start_time) / 1_000_000_000
        return (end_monotonic_ns - state.started_monotonic_ns) / 1_000_000_000

    @staticmethod
    def _set_error(span: Span, error: BaseException | str) -> None:
        span.set_attribute("error.type", _error_type(error))
        span.set_status(StatusCode.ERROR)
