"""Provider operation spans using OTel GenAI semantic conventions."""

from __future__ import annotations

import json
import time
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.context import Context
from opentelemetry.trace import Span, SpanKind, StatusCode

from lightspeed_agentic.genai_messages import encode_messages, encode_tool_object
from lightspeed_agentic.metrics import tool_duration
from lightspeed_agentic.tracing import get_tracer


class GenAIRecorder:
    """Record model and tool operations as siblings beneath one run context."""

    def __init__(
        self,
        *,
        phase: str,
        provider: str,
        capture_content: bool = False,
        agenticrun_uid: str = "",
        parent_context: Context | None = None,
        output_type: str | None = None,
        server_address: str | None = None,
    ) -> None:
        self._phase = phase
        self._provider = provider
        self._capture_content = capture_content
        self._agenticrun_uid = agenticrun_uid
        self._parent_context = (
            parent_context if parent_context is not None else otel_context.get_current()
        )
        self._output_type = output_type
        self._server_address = server_address
        self._tracer = get_tracer()
        self._model_spans: set[Span] = set()
        self._tool_spans: dict[Span, tuple[str, float]] = {}

    def _correlation(self) -> dict[str, str]:
        attributes = {}
        if self._agenticrun_uid:
            attributes["agenticrun.uid"] = self._agenticrun_uid
        if self._phase:
            attributes["agenticrun.phase"] = self._phase
        return attributes

    def start_model(
        self,
        input_messages: list[dict[str, Any]],
        system_instructions: list[dict[str, Any]] | None,
        request_model: str,
        *,
        operation_name: str = "chat",
        tool_definitions: list[dict[str, Any]] | None = None,
    ) -> Span:
        attributes: dict[str, str] = {
            "gen_ai.operation.name": operation_name,
            "gen_ai.provider.name": self._provider,
            "gen_ai.request.model": request_model,
            **self._correlation(),
        }
        if self._output_type:
            attributes["gen_ai.output.type"] = self._output_type
        if self._server_address:
            attributes["server.address"] = self._server_address
        if self._capture_content:
            attributes["gen_ai.input.messages"] = encode_messages(input_messages)
            if system_instructions is not None:
                attributes["gen_ai.system_instructions"] = encode_messages(system_instructions)
            if tool_definitions is not None:
                attributes["gen_ai.tool.definitions"] = encode_messages(tool_definitions)
        span = self._tracer.start_span(
            f"{operation_name} {request_model}",
            kind=SpanKind.CLIENT,
            context=self._parent_context,
            attributes=attributes,
        )
        self._model_spans.add(span)
        return span

    def end_model(
        self,
        handle: Span,
        output_messages: list[dict[str, Any]] | None,
        response_model: str | None,
        usage: dict[str, int],
        error: BaseException | None,
    ) -> None:
        span = handle
        self._model_spans.remove(span)
        try:
            if self._capture_content and output_messages is not None:
                span.set_attribute("gen_ai.output.messages", encode_messages(output_messages))
            if response_model:
                span.set_attribute("gen_ai.response.model", response_model)
            if "input_tokens" in usage:
                span.set_attribute("gen_ai.usage.input_tokens", usage["input_tokens"])
            if "output_tokens" in usage:
                span.set_attribute("gen_ai.usage.output_tokens", usage["output_tokens"])
            if "reasoning_tokens" in usage:
                span.set_attribute(
                    "gen_ai.usage.reasoning.output_tokens", usage["reasoning_tokens"]
                )
            if error is not None:
                span.set_attribute("error.type", type(error).__name__)
                span.set_status(StatusCode.ERROR)
        except BaseException as exc:
            span.set_attribute("error.type", type(exc).__name__)
            span.set_status(StatusCode.ERROR)
            raise
        finally:
            span.end()

    def start_tool(self, name: str, call_id: str, arguments: Any) -> Span:
        attributes = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": name,
            "gen_ai.tool.call.id": call_id,
            "gen_ai.tool.type": "function",
            **self._correlation(),
        }
        if self._capture_content:
            attributes["gen_ai.tool.call.arguments"] = json.dumps(
                encode_tool_object(arguments), ensure_ascii=False, separators=(",", ":")
            )
        span = self._tracer.start_span(
            f"execute_tool {name}",
            kind=SpanKind.INTERNAL,
            context=self._parent_context,
            attributes=attributes,
        )
        self._tool_spans[span] = (name, time.monotonic())
        return span

    def end_tool(self, handle: Span, result: Any, error: BaseException | None) -> None:
        span = handle
        name, start = self._tool_spans.pop(span)
        try:
            if error is not None:
                span.set_attribute("error.type", type(error).__name__)
                span.set_status(StatusCode.ERROR)
            else:
                if self._capture_content:
                    span.set_attribute(
                        "gen_ai.tool.call.result",
                        json.dumps(
                            encode_tool_object(result), ensure_ascii=False, separators=(",", ":")
                        ),
                    )
        except BaseException as exc:
            span.set_attribute("error.type", type(exc).__name__)
            span.set_status(StatusCode.ERROR)
            raise
        finally:
            tool_duration.labels(gen_ai_tool_name=name).observe(time.monotonic() - start)
            span.end()

    def close(self) -> None:
        """End any operations that the provider left unfinished."""
        for span in self._model_spans:
            span.set_attribute("error.type", "incomplete")
            span.set_status(StatusCode.ERROR)
            span.end()
        self._model_spans.clear()
        for span, (name, start) in self._tool_spans.items():
            span.set_attribute("error.type", "incomplete")
            span.set_status(StatusCode.ERROR)
            tool_duration.labels(gen_ai_tool_name=name).observe(time.monotonic() - start)
            span.end()
        self._tool_spans.clear()
