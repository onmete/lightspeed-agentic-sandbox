"""OpenAI Agents model and tool telemetry adapters.

Model spans wrap the SDK's actual request methods; tool spans use the SDK's native
RunHooks callbacks and narrowly wrap FunctionTool failures because RunHooks has
no failure callback.
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agents.models.interface import Model
    from agents.tool import FunctionTool

from lightspeed_agentic.audit import AuditLogger

_MISSING = object()
_UNKNOWN_FINISH_REASON = "unknown"

_TOOL_CALL_NAMES = {
    "apply_patch_call": "apply_patch",
    "code_interpreter_call": "code_interpreter",
    "computer_call": "computer_use_preview",
    "file_search_call": "file_search",
    "image_generation_call": "image_generation",
    "local_shell_call": "local_shell",
    "mcp_call": "mcp",
    "shell_call": "shell",
    "tool_search_call": "tool_search",
    "web_search_call": "web_search",
}

_TOOL_RESPONSE_TYPES = {
    "file_search_call_output",
    "image_generation_call_output",
    "tool_search_call_output",
    "web_search_call_output",
    "apply_patch_call_output",
    "code_interpreter_call_output",
    "computer_call_output",
    "custom_tool_call_output",
    "function_call_output",
    "local_shell_call_output",
    "mcp_call_output",
    "shell_call_output",
}


@dataclass
class _OpenToolSpan:
    span: Any | None
    tool: Any
    context_id: int
    call_id: str
    name: str
    tool_type: str
    start_time: int
    arguments: Any
    raw_arguments: str | None
    error: BaseException | str | None = None


@dataclass
class _FunctionToolPatch:
    tool: FunctionTool
    original_invoke: Any
    wrapped_invoke: Any
    original_failure: Any
    wrapped_failure: Any
    uses_default_failure: bool
    active_calls: int = 0


@dataclass
class _ExecutorPatch:
    session: Any
    methods: dict[str, tuple[Any, Any]]


@dataclass
class _MCPExtractorPatch:
    server: Any
    instance_value: Any
    wrapped: Any


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(exclude_none=True, by_alias=True)
        if isinstance(dumped, Mapping):
            return dumped
    return {}


def _sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _kind(value: Any) -> str | None:
    kind = _field(value, "type")
    if isinstance(kind, str):
        return kind
    enum_value = getattr(kind, "value", None)
    return enum_value if isinstance(enum_value, str) else None


def _json_arguments(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _tool_call_id(value: Any) -> str:
    call_id = _field(value, "tool_call_id")
    return call_id if isinstance(call_id, str) else ""


def _tool_name(tool: Any, context: Any) -> str | None:
    for candidate in (_field(context, "tool_name"), _field(tool, "name")):
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _tool_arguments(context: Any) -> tuple[Any, str | None]:
    raw = _field(context, "tool_arguments", _MISSING)
    if raw is _MISSING:
        return None, None
    if isinstance(raw, str):
        return _json_arguments(raw), raw
    return raw, None


def _text_part(text: Any) -> dict[str, Any] | None:
    if isinstance(text, str):
        return {"type": "text", "content": text}
    return None


def _generic_part(value: Any) -> dict[str, Any] | None:
    raw = _mapping(value)
    kind = _kind(value)
    if kind is None:
        return None
    part = dict(raw)
    part["type"] = kind
    return part


def _content_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "content": content}]
    if not _sequence(content):
        return []

    parts: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            parts.append({"type": "text", "content": item})
            continue
        kind = _kind(item)
        if kind in {"input_text", "output_text", "text"}:
            part = _text_part(_field(item, "text", _field(item, "content")))
            if part is not None:
                annotations = _field(item, "annotations", _MISSING)
                if annotations is not _MISSING:
                    part["annotations"] = annotations
                parts.append(part)
        elif kind == "refusal":
            part = _text_part(_field(item, "refusal", _field(item, "text")))
            if part is not None:
                parts.append(part)
        else:
            part = _generic_part(item)
            if part is not None:
                parts.append(part)
    return parts


def _reasoning_parts(item: Any) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for field_name in ("content", "summary"):
        segments = _field(item, field_name)
        if isinstance(segments, str):
            segments = [segments]
        if not _sequence(segments):
            continue
        for segment in segments:
            content = _field(segment, "text")
            if content is None and isinstance(segment, str):
                content = segment
            if isinstance(content, str):
                parts.append({"type": "reasoning", "content": content})
    return parts


def _tool_call_part(item: Any, *, nested_function: bool = False) -> dict[str, Any] | None:
    function = _field(item, "function") if nested_function else item
    name = _field(function, "name")
    kind = _kind(item)
    if not isinstance(name, str) or not name:
        name = _TOOL_CALL_NAMES.get(kind or "")
    if not isinstance(name, str) or not name:
        return _generic_part(item)

    part: dict[str, Any] = {"type": "tool_call", "name": name}
    call_id = _field(item, "call_id")
    if not isinstance(call_id, str) or not call_id:
        call_id = _field(item, "id")
    if isinstance(call_id, str) and call_id:
        part["id"] = call_id

    arguments = _field(function, "arguments", _MISSING)
    if arguments is _MISSING:
        arguments = _field(item, "input", _MISSING)
    if arguments is _MISSING:
        arguments = {
            key: value
            for key, value in _mapping(item).items()
            if key not in {"type", "id", "call_id", "name", "status", "server_label"}
        }
        if not arguments:
            arguments = _MISSING
    if arguments is not _MISSING:
        part["arguments"] = _json_arguments(arguments)
    return part


def _tool_response_part(item: Any) -> dict[str, Any]:
    part: dict[str, Any] = {"type": "tool_call_response"}
    call_id = _field(item, "call_id")
    if not isinstance(call_id, str) or not call_id:
        call_id = _field(item, "id")
    if isinstance(call_id, str) and call_id:
        part["id"] = call_id
    response = _field(item, "output", _MISSING)
    if response is _MISSING:
        response = _field(item, "result", _MISSING)
    if response is _MISSING:
        fallback = {
            key: value
            for key, value in _mapping(item).items()
            if key not in {"type", "id", "call_id"}
        }
        response = fallback if fallback else _MISSING
    if response is not _MISSING:
        part["response"] = response
    return part


def _input_item_part(item: Any) -> dict[str, Any] | None:
    kind = _kind(item)
    if kind in {"function_call", "custom_tool_call"} or kind in _TOOL_CALL_NAMES:
        return _tool_call_part(item)
    if kind in _TOOL_RESPONSE_TYPES:
        return _tool_response_part(item)
    return _generic_part(item)


def _input_messages(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        return [{"role": "user", "parts": [{"type": "text", "content": value}]}]
    if isinstance(value, Mapping):
        items: Sequence[Any] = [value]
    elif _sequence(value):
        items = value
    else:
        return []

    messages: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            messages.append({"role": "user", "parts": [{"type": "text", "content": item}]})
            continue
        kind = _kind(item)
        if kind == "message" or _field(item, "role") is not None:
            role = _field(item, "role", "user")
            message: dict[str, Any] = {
                "role": role if isinstance(role, str) else "user",
                "parts": _content_parts(_field(item, "content")),
            }
            name = _field(item, "name")
            if isinstance(name, str) and name:
                message["name"] = name
            messages.append(message)
            continue
        if kind == "reasoning":
            parts = _reasoning_parts(item)
            if parts:
                messages.append({"role": "assistant", "parts": parts})
            continue

        part = _input_item_part(item)
        if part is None:
            continue
        role = "tool" if kind in _TOOL_RESPONSE_TYPES else "assistant"
        messages.append({"role": role, "parts": [part]})
    return messages


def _system_instruction_parts(value: Any) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [{"type": "text", "content": value}]
    parts = _content_parts(value)
    return parts


def _function_tool_definition(tool: FunctionTool) -> dict[str, Any]:
    definition: dict[str, Any] = {
        "type": "function",
        "name": tool.name,
        "parameters": tool.params_json_schema,
    }
    if tool.description:
        definition["description"] = tool.description
    strict = getattr(tool, "strict_json_schema", None)
    if isinstance(strict, bool):
        definition["strict"] = strict
    return definition


def _tool_definition(tool: Any) -> dict[str, Any] | None:
    from agents.tool import FunctionTool

    if isinstance(tool, FunctionTool):
        return _function_tool_definition(tool)

    config = _field(tool, "tool_config")
    if config is not None:
        definition = _mapping(config)
        if definition:
            return dict(definition)

    name = _field(tool, "name")
    tool_type = _field(tool, "type")
    if not isinstance(tool_type, str) or not tool_type:
        tool_type = name if isinstance(name, str) else None
    if not isinstance(tool_type, str) or not tool_type:
        return None

    definition = {"type": tool_type}
    if isinstance(name, str) and name:
        definition["name"] = name
    for field_name in (
        "description",
        "parameters",
        "params_json_schema",
        "strict_json_schema",
        "format",
        "server_label",
        "vector_store_ids",
        "max_num_results",
        "search_context_size",
        "user_location",
        "filters",
        "environment",
        "needs_approval",
    ):
        field_value = _field(tool, field_name, _MISSING)
        if field_value is not _MISSING and field_value is not None:
            definition[field_name] = field_value
    return definition


def _handoff_definition(handoff: Any) -> dict[str, Any] | None:
    name = _field(handoff, "tool_name")
    if not isinstance(name, str) or not name:
        return None
    definition: dict[str, Any] = {"type": "function", "name": name}
    for source, target in (
        ("tool_description", "description"),
        ("input_json_schema", "parameters"),
        ("strict_json_schema", "strict"),
    ):
        value = _field(handoff, source, _MISSING)
        if value is not _MISSING and value is not None:
            definition[target] = value
    return definition


def _tool_definitions(tools: Any, handoffs: Any) -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = []
    if _sequence(tools):
        for tool in tools:
            definition = _tool_definition(tool)
            if definition is not None:
                definitions.append(definition)
    if _sequence(handoffs):
        for handoff in handoffs:
            definition = _handoff_definition(handoff)
            if definition is not None:
                definitions.append(definition)
    return definitions


def _output_type(output_schema: Any) -> str | None:
    if output_schema is None:
        return None
    is_plain_text = getattr(output_schema, "is_plain_text", None)
    if callable(is_plain_text):
        return "text" if is_plain_text() else "json"
    return None


def _chat_tool_call_part(tool_call: Any) -> dict[str, Any] | None:
    function = _field(tool_call, "function")
    if function is None:
        return _tool_call_part(tool_call)
    return _tool_call_part(tool_call, nested_function=True)


def _finish_reason(value: Any) -> str:
    reason = _field(value, "finish_reason")
    if isinstance(reason, str) and reason:
        return reason
    reason_value = getattr(reason, "value", None)
    if isinstance(reason_value, str) and reason_value:
        return reason_value
    return _UNKNOWN_FINISH_REASON


def _choice_message(choice: Any) -> dict[str, Any]:
    message = _field(choice, "message", _field(choice, "delta"))
    role = _field(message, "role", "assistant")
    parts = _content_parts(_field(message, "content"))
    for tool_call in _field(message, "tool_calls", ()) or ():
        part = _chat_tool_call_part(tool_call)
        if part is not None:
            parts.append(part)
    reasoning = _field(message, "reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        parts.append({"type": "reasoning", "content": reasoning})
    refusal = _field(message, "refusal")
    if isinstance(refusal, str):
        parts.append({"type": "text", "content": refusal})
    return {
        "role": role if isinstance(role, str) else "assistant",
        "parts": parts,
        "finish_reason": _finish_reason(choice),
    }


def _response_item_parts(item: Any) -> list[dict[str, Any]]:
    kind = _kind(item)
    if kind == "message":
        return _content_parts(_field(item, "content"))
    if kind == "reasoning":
        return _reasoning_parts(item)
    if kind in {"function_call", "custom_tool_call"}:
        part = _tool_call_part(item)
    elif kind in _TOOL_RESPONSE_TYPES:
        part = _tool_response_part(item)
    elif kind in _TOOL_CALL_NAMES:
        part = _tool_call_part(item)
    else:
        part = _generic_part(item)
    return [part] if part is not None else []


def _output_messages(response: Any) -> list[dict[str, Any]] | None:
    choices = _field(response, "choices", _MISSING)
    if _sequence(choices):
        return [_choice_message(choice) for choice in choices]

    output = response if _sequence(response) else _field(response, "output", _MISSING)
    if output is _MISSING or not _sequence(output):
        return None

    parts: list[dict[str, Any]] = []
    role = "assistant"
    for item in output:
        item_role = _field(item, "role")
        if isinstance(item_role, str):
            role = item_role
        parts.extend(_response_item_parts(item))
    if not output:
        return []
    return [
        {
            "role": role,
            "parts": parts,
            "finish_reason": _finish_reason(response),
        }
    ]


def _integer(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _field_was_observed(value: Any, name: str) -> bool:
    if isinstance(value, Mapping):
        return name in value
    fields_set = getattr(value, "model_fields_set", None)
    if isinstance(fields_set, (set, frozenset)):
        return name in fields_set
    return getattr(value, name, _MISSING) is not _MISSING


def _normalized_usage(
    response: Any, *, native_responses: bool
) -> tuple[int | None, int | None, int | None]:
    usage = _field(response, "usage")
    if usage is None:
        return None, None, None
    requests = _integer(_field(usage, "requests"))
    if requests == 0:
        return None, None, None

    input_tokens = _integer(_field(usage, "input_tokens"))
    output_tokens = _integer(_field(usage, "output_tokens"))
    details = _field(usage, "output_tokens_details")
    reasoning_tokens = (
        _integer(_field(details, "reasoning_tokens")) if details is not None else None
    )
    if reasoning_tokens == 0 and not (
        native_responses
        and details is not None
        and _field_was_observed(details, "reasoning_tokens")
    ):
        reasoning_tokens = None
    return input_tokens, output_tokens, reasoning_tokens


def _response_usage(
    response: Any, *, native_responses: bool
) -> tuple[int | None, int | None, int | None]:
    usage = _field(response, "usage")
    if usage is None:
        return None, None, None
    input_tokens = _integer(_field(usage, "input_tokens"))
    output_tokens = _integer(_field(usage, "output_tokens"))
    details = _field(usage, "output_tokens_details")
    reasoning_tokens = None
    if details is not None:
        observed = _field(details, "reasoning_tokens", _MISSING)
        if native_responses and _field_was_observed(details, "reasoning_tokens"):
            reasoning_tokens = _integer(observed)
        elif not native_responses:
            reasoning_tokens = _integer(observed)
            if reasoning_tokens == 0:
                reasoning_tokens = None
    return input_tokens, output_tokens, reasoning_tokens


def _response_model(response: Any, *, native_responses: bool) -> str | None:
    if not native_responses:
        return None
    model = _field(response, "model")
    return model if isinstance(model, str) and model else None


def _finish_reasons(messages: list[dict[str, Any]] | None) -> list[str] | None:
    if not messages:
        return None
    return [
        reason if isinstance(reason, str) and reason else _UNKNOWN_FINISH_REASON
        for message in messages
        if (reason := message.get("finish_reason")) is not None
    ]


def _response_observations(
    response: Any,
    *,
    native_responses: bool,
    normalized: bool,
) -> dict[str, Any]:
    messages = _output_messages(response)
    if normalized:
        input_tokens, output_tokens, reasoning_tokens = _normalized_usage(
            response, native_responses=native_responses
        )
    else:
        input_tokens, output_tokens, reasoning_tokens = _response_usage(
            response, native_responses=native_responses
        )
    return {
        "output_messages": messages,
        "response_model": _response_model(response, native_responses=native_responses),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "finish_reasons": _finish_reasons(messages),
    }


def _completed_response(event: Any) -> Any:
    if _kind(event) != "response.completed":
        return None
    return _field(event, "response")


def _stream_error(event: Any) -> str | None:
    kind = _kind(event)
    if kind in {"error", "response.error", "response.failed", "response.incomplete"}:
        return kind
    return None


def create_model_proxy(
    model: Model,
    audit_logger: AuditLogger,
    *,
    request_model: str,
    native_responses: bool,
) -> Model:
    """Create a native Model proxy only when the OpenAI extra is used."""
    from agents.models.interface import Model

    class OpenAIModelProxy(Model):
        """Delegate native Model methods while recording each real request and response."""

        def __init__(
            self,
            model: Model,
            audit_logger: AuditLogger,
            *,
            request_model: str,
            native_responses: bool,
        ) -> None:
            self._delegate = model
            self._audit_logger = audit_logger
            self._request_model = request_model
            self._native_responses = native_responses

        def __getattr__(self, name: str) -> Any:
            return getattr(object.__getattribute__(self, "_delegate"), name)

        async def _cleanup_on_run_end(self, owner: object) -> None:
            await self._delegate._cleanup_on_run_end(owner)

        async def close(self) -> None:
            await self._delegate.close()

        def get_retry_advice(self, request: Any) -> Any:
            return self._delegate.get_retry_advice(request)

        def _start_inference(
            self,
            system_instructions: Any,
            model_input: Any,
            tools: Any,
            output_schema: Any,
            handoffs: Any,
        ) -> Any:
            return self._audit_logger.start_inference(
                model=self._request_model,
                operation="chat",
                input_messages=_input_messages(model_input),
                system_instructions=_system_instruction_parts(system_instructions),
                tool_definitions=_tool_definitions(tools, handoffs),
                output_type=_output_type(output_schema),
                start_time=time.time_ns(),
            )

        async def get_response(
            self,
            system_instructions: str | None,
            input: str | list[Any],  # noqa: A002 - OpenAI Agents Model API keyword.
            model_settings: Any,
            tools: list[Any],
            output_schema: Any,
            handoffs: list[Any],
            tracing: Any,
            *,
            previous_response_id: str | None = None,
            conversation_id: str | None = None,
            prompt: Any = None,
        ) -> Any:
            span = self._start_inference(system_instructions, input, tools, output_schema, handoffs)
            try:
                response = await self._delegate.get_response(
                    system_instructions,
                    input,
                    model_settings,
                    tools,
                    output_schema,
                    handoffs,
                    tracing,
                    previous_response_id=previous_response_id,
                    conversation_id=conversation_id,
                    prompt=prompt,
                )
            except BaseException as error:
                self._audit_logger.end_inference(
                    span,
                    error=error,
                    end_time=time.time_ns(),
                )
                raise

            end_time = time.time_ns()
            observations = _response_observations(
                response,
                native_responses=self._native_responses,
                normalized=True,
            )
            self._audit_logger.end_inference(span, end_time=end_time, **observations)
            return response

        def stream_response(
            self,
            system_instructions: str | None,
            input: str | list[Any],  # noqa: A002 - OpenAI Agents Model API keyword.
            model_settings: Any,
            tools: list[Any],
            output_schema: Any,
            handoffs: list[Any],
            tracing: Any,
            *,
            previous_response_id: str | None = None,
            conversation_id: str | None = None,
            prompt: Any = None,
        ) -> AsyncIterator[Any]:
            return self._stream_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                tracing,
                previous_response_id=previous_response_id,
                conversation_id=conversation_id,
                prompt=prompt,
            )

        async def _stream_response(
            self,
            system_instructions: str | None,
            model_input: str | list[Any],
            model_settings: Any,
            tools: list[Any],
            output_schema: Any,
            handoffs: list[Any],
            tracing: Any,
            *,
            previous_response_id: str | None,
            conversation_id: str | None,
            prompt: Any,
        ) -> AsyncIterator[Any]:
            span = self._start_inference(
                system_instructions, model_input, tools, output_schema, handoffs
            )
            stream: Any = None
            exhausted = False
            ended = False
            request_error: BaseException | None = None
            try:
                stream = self._delegate.stream_response(
                    system_instructions,
                    model_input,
                    model_settings,
                    tools,
                    output_schema,
                    handoffs,
                    tracing,
                    previous_response_id=previous_response_id,
                    conversation_id=conversation_id,
                    prompt=prompt,
                )
                async for event in stream:
                    if not ended and _kind(event) == "response.completed":
                        response = _completed_response(event)
                        observations = (
                            _response_observations(
                                response,
                                native_responses=self._native_responses,
                                normalized=not self._native_responses,
                            )
                            if response is not None
                            else {}
                        )
                        self._audit_logger.end_inference(
                            span, end_time=time.time_ns(), **observations
                        )
                        ended = True
                    elif not ended:
                        error_type = _stream_error(event)
                        if error_type is not None:
                            terminal_response = (
                                _field(event, "response")
                                if error_type in {"response.incomplete", "response.failed"}
                                else None
                            )
                            observations = (
                                _response_observations(
                                    terminal_response,
                                    native_responses=self._native_responses,
                                    normalized=not self._native_responses,
                                )
                                if terminal_response is not None
                                else {}
                            )
                            self._audit_logger.end_inference(
                                span,
                                error=error_type,
                                end_time=time.time_ns(),
                                **observations,
                            )
                            ended = True
                    yield event
                exhausted = True
                if not ended:
                    self._audit_logger.end_inference(
                        span,
                        error="response_incomplete",
                        end_time=time.time_ns(),
                    )
                    ended = True
            except BaseException as error:
                request_error = error
                if not ended:
                    self._audit_logger.end_inference(
                        span,
                        error=error,
                        end_time=time.time_ns(),
                    )
                    ended = True
                raise
            finally:
                if not exhausted and stream is not None:
                    close = getattr(stream, "aclose", None)
                    if callable(close):
                        try:
                            await close()
                        except BaseException as close_error:
                            if not ended:
                                self._audit_logger.end_inference(
                                    span,
                                    error=close_error,
                                    end_time=time.time_ns(),
                                )
                                ended = True
                            if request_error is None:
                                raise

    return OpenAIModelProxy(
        model, audit_logger, request_model=request_model, native_responses=native_responses
    )


def _is_native_view_image_tool(tool: Any) -> bool:
    """Return whether this is the SDK filesystem tool that reports errors as text."""
    try:
        from agents.sandbox.capabilities.tools.view_image import ViewImageTool
    except ImportError:
        return False
    return isinstance(tool, ViewImageTool)


def create_tool_hooks(audit_logger: AuditLogger) -> Any:
    """Create native RunHooks only when the OpenAI extra is used."""
    from agents.lifecycle import RunHooks
    from agents.sandbox import ExecTimeoutError
    from agents.sandbox.errors import PtySessionNotFoundError
    from agents.tool import (
        FunctionTool,
        resolve_function_tool_failure_error_function,
        set_function_tool_failure_error_function,
    )

    class OpenAIToolRunHooks(RunHooks[Any]):
        """Record SDK-native local tool starts/ends and exceptions hidden by tool wrappers."""

        def __init__(self, audit_logger: AuditLogger) -> None:
            self._audit_logger = audit_logger
            self._pending: list[_OpenToolSpan] = []
            self._function_tool_patches: dict[int, _FunctionToolPatch] = {}
            self._executor_patches: dict[int, _ExecutorPatch] = {}
            self._mcp_extractor_patches: dict[int, _MCPExtractorPatch] = {}
            self._active_tool_span: ContextVar[_OpenToolSpan | None] = ContextVar(
                f"openai_tool_span_{id(self)}", default=None
            )
            self._deferred_apply_patch_tools: set[int] = set()

        def configure_shell_tools(self, toolset: Any) -> None:
            """Observe native shell/PTY failures before tools format them as text."""
            tool = _field(toolset, "exec_command")
            session = _field(tool, "session")
            if tool is None or session is None or id(session) in self._executor_patches:
                return

            patch = _ExecutorPatch(session=session, methods={})
            self._executor_patches[id(session)] = patch
            instance_attributes = getattr(session, "__dict__", {})
            try:
                for method_name in ("exec", "pty_exec_start", "pty_write_stdin"):
                    original = getattr(session, method_name, None)
                    if not callable(original):
                        continue

                    @wraps(original)
                    async def instrumented(
                        *args: Any,
                        _original: Any = original,
                        _method_name: str = method_name,
                        **kwargs: Any,
                    ) -> Any:
                        try:
                            result = _original(*args, **kwargs)
                            if inspect.isawaitable(result):
                                result = await result
                            return result
                        except (
                            ExecTimeoutError,
                            TimeoutError,
                            PtySessionNotFoundError,
                            RuntimeError,
                        ) as error:
                            pending = self._active_tool_span.get()
                            if pending is not None and (
                                (
                                    pending.name == "exec_command"
                                    and _method_name in {"exec", "pty_exec_start"}
                                    and isinstance(error, (ExecTimeoutError, TimeoutError))
                                )
                                or (
                                    pending.name == "write_stdin"
                                    and _method_name == "pty_write_stdin"
                                    and isinstance(error, (PtySessionNotFoundError, RuntimeError))
                                )
                            ):
                                pending.error = error
                            raise

                    instance_value = instance_attributes.get(method_name, _MISSING)
                    try:
                        setattr(session, method_name, instrumented)
                    except (AttributeError, TypeError):
                        continue
                    patch.methods[method_name] = (instance_value, instrumented)
            except BaseException:
                self._restore_executor_patch(patch)
                self._executor_patches.pop(id(session), None)
                raise

            if not patch.methods:
                self._executor_patches.pop(id(session), None)

        def configure_mcp_servers(self, servers: Sequence[Any]) -> None:
            """Observe MCP isError flags through the SDK's custom-data extension."""
            for server in servers:
                server_id = id(server)
                if server_id in self._mcp_extractor_patches:
                    continue

                original = _field(server, "custom_data_extractor")

                async def extract_custom_data(
                    context: Any,
                    _original: Any = original,
                    _server: Any = server,
                ) -> Any:
                    custom_data = None
                    if callable(_original):
                        custom_data = _original(context)
                        if inspect.isawaitable(custom_data):
                            custom_data = await custom_data

                    pending = self._active_tool_span.get()
                    if (
                        _field(context, "is_error") is True
                        and pending is not None
                        and pending.name == _field(context, "tool_display_name")
                        and _field(context, "server_name") == _field(_server, "name")
                    ):
                        pending.error = "mcp_tool_error"
                    return custom_data

                instance_value = getattr(server, "__dict__", {}).get(
                    "custom_data_extractor", _MISSING
                )
                try:
                    server.custom_data_extractor = extract_custom_data
                except (AttributeError, TypeError):
                    continue
                self._mcp_extractor_patches[server_id] = _MCPExtractorPatch(
                    server=server,
                    instance_value=instance_value,
                    wrapped=extract_custom_data,
                )

        def configure_filesystem_tools(self, toolset: Any) -> None:
            """Capture apply_patch editor operations absent from its RunHooks context."""
            tool = _field(toolset, "apply_patch")
            editor = _field(tool, "editor")
            if tool is None or editor is None:
                return

            self._deferred_apply_patch_tools.add(id(tool))
            for method_name in ("create_file", "update_file", "delete_file"):
                original = getattr(editor, method_name, None)
                if not callable(original):
                    continue

                @wraps(original)
                async def instrumented(
                    operation: Any,
                    _original: Any = original,
                    _tool: Any = tool,
                ) -> Any:
                    context = _field(operation, "ctx_wrapper")
                    pending = self._find_pending(context, _tool)
                    if pending is not None:
                        if pending.arguments is None:
                            pending.arguments = []
                        pending.arguments.append(
                            {
                                field_name: _field(operation, field_name)
                                for field_name in ("type", "path", "diff", "move_to")
                            }
                        )
                    try:
                        result = _original(operation)
                        if inspect.isawaitable(result):
                            result = await result
                    except BaseException as error:
                        if pending is not None:
                            pending.error = error
                        raise
                    if pending is not None and _field(result, "status") == "failed":
                        pending.error = "apply_patch_failed"
                    return result

                setattr(editor, method_name, instrumented)

        async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
            del agent
            name = _tool_name(tool, context)
            if name is None:
                return

            arguments, raw_arguments = _tool_arguments(context)
            call_id = _tool_call_id(context)
            tool_type = _field(tool, "type")
            if not isinstance(tool_type, str) or not tool_type:
                tool_type = "function"
            start_time = time.time_ns()
            if (
                id(tool) in self._deferred_apply_patch_tools
                and _field(context, "tool_arguments", _MISSING) is _MISSING
            ):
                # ApplyPatchAction omits request data from RunHooks; retain its native start
                # and collect parsed editor operations before writing span content.
                self._pending.append(
                    _OpenToolSpan(
                        span=None,
                        tool=tool,
                        context_id=id(context),
                        call_id=call_id,
                        name=name,
                        tool_type=tool_type,
                        start_time=start_time,
                        arguments=None,
                        raw_arguments=raw_arguments,
                    )
                )
                return

            patch: _FunctionToolPatch | None = None
            if isinstance(tool, FunctionTool):
                patch = self._patch_function_tool(tool)
                patch.active_calls += 1

            try:
                span = self._audit_logger.start_tool(
                    name=name,
                    call_id=call_id,
                    arguments=arguments,
                    tool_type=tool_type,
                    start_time=start_time,
                )
            except BaseException:
                if patch is not None:
                    self._release_function_tool(tool)
                raise

            self._pending.append(
                _OpenToolSpan(
                    span=span,
                    tool=tool,
                    context_id=id(context),
                    call_id=call_id,
                    name=name,
                    tool_type=tool_type,
                    start_time=start_time,
                    arguments=arguments,
                    raw_arguments=raw_arguments,
                )
            )

        async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
            del agent
            pending = self._take_pending(context, tool)
            if pending is not None:
                if (
                    pending.name == "view_image"
                    and isinstance(result, str)
                    and _is_native_view_image_tool(pending.tool)
                ):
                    pending.error = "view_image_failed"
                self._finish_pending(pending, result=result)

        def close(self, error: BaseException | str = "operation_cancelled") -> None:
            """Close only still-open local tool spans; completed spans are left untouched."""
            pending_spans, self._pending = self._pending, []
            for pending in pending_spans:
                self._finish_pending(pending, error=error)
            for patch in list(self._function_tool_patches.values()):
                patch.active_calls = 0
                self._restore_function_tool(patch)
            for executor_patch in list(self._executor_patches.values()):
                self._restore_executor_patch(executor_patch)
            self._executor_patches.clear()

            for extractor_patch in list(self._mcp_extractor_patches.values()):
                if (
                    getattr(extractor_patch.server, "custom_data_extractor", None)
                    is extractor_patch.wrapped
                ):
                    if extractor_patch.instance_value is _MISSING:
                        with suppress(AttributeError):
                            delattr(extractor_patch.server, "custom_data_extractor")
                    else:
                        extractor_patch.server.custom_data_extractor = (
                            extractor_patch.instance_value
                        )
            self._mcp_extractor_patches.clear()

        def _patch_function_tool(self, tool: FunctionTool) -> _FunctionToolPatch:
            existing = self._function_tool_patches.get(id(tool))
            if existing is not None:
                return existing

            original_invoke = tool.on_invoke_tool
            uses_default_failure = bool(getattr(tool, "_use_default_failure_error_function", False))
            original_failure = getattr(tool, "_failure_error_function", None)
            effective_failure = resolve_function_tool_failure_error_function(tool)

            async def wrapped_invoke(context: Any, raw_arguments: str) -> Any:
                token = self._active_tool_span.set(self._find_pending(context, tool))
                try:
                    result = original_invoke(context, raw_arguments)
                    if inspect.isawaitable(result):
                        return await result
                    return result
                except BaseException as error:
                    self._finish_error(tool, context, error, raw_arguments=raw_arguments)
                    raise
                finally:
                    self._active_tool_span.reset(token)

            wrapped_failure: Any = None
            if effective_failure is not None:

                @wraps(effective_failure)
                async def handle_failure(context: Any, error: Exception) -> Any:
                    self._finish_error(tool, context, error)
                    result = effective_failure(context, error)
                    if inspect.isawaitable(result):
                        return await result
                    return result

                wrapped_failure = handle_failure

            patch = _FunctionToolPatch(
                tool=tool,
                original_invoke=original_invoke,
                wrapped_invoke=wrapped_invoke,
                original_failure=original_failure,
                wrapped_failure=wrapped_failure,
                uses_default_failure=uses_default_failure,
            )
            self._function_tool_patches[id(tool)] = patch
            tool.on_invoke_tool = wrapped_invoke
            if wrapped_failure is not None:
                set_function_tool_failure_error_function(tool, wrapped_failure)
            return patch

        def _restore_executor_patch(self, patch: _ExecutorPatch) -> None:
            for method_name, (instance_value, wrapped) in patch.methods.items():
                if getattr(patch.session, method_name, None) is not wrapped:
                    continue
                if instance_value is _MISSING:
                    with suppress(AttributeError):
                        delattr(patch.session, method_name)
                else:
                    setattr(patch.session, method_name, instance_value)
            patch.methods.clear()

        def _finish_error(
            self,
            tool: FunctionTool,
            context: Any,
            error: BaseException,
            *,
            raw_arguments: str | None = None,
        ) -> None:
            pending = self._take_pending(context, tool, raw_arguments=raw_arguments)
            if pending is not None:
                self._finish_pending(pending, error=error)

        def _find_pending(self, context: Any, tool: Any) -> _OpenToolSpan | None:
            call_id = _tool_call_id(context)
            if call_id:
                matches = [
                    pending
                    for pending in self._pending
                    if pending.tool is tool and pending.call_id == call_id
                ]
                if len(matches) == 1:
                    return matches[0]
                return None

            context_id = id(context)
            matches = [
                pending
                for pending in self._pending
                if pending.tool is tool and pending.context_id == context_id
            ]
            if len(matches) == 1:
                return matches[0]

            raw_arguments = _field(context, "tool_arguments", _MISSING)
            if raw_arguments is not _MISSING:
                matches = [
                    pending
                    for pending in self._pending
                    if pending.tool is tool and pending.raw_arguments == raw_arguments
                ]
                if len(matches) == 1:
                    return matches[0]

            tool_input = _field(context, "tool_input", _MISSING)
            if tool_input is not _MISSING:
                matches = [
                    pending
                    for pending in self._pending
                    if pending.tool is tool and pending.arguments == tool_input
                ]
                if len(matches) == 1:
                    return matches[0]

            matches = [pending for pending in self._pending if pending.tool is tool]
            return matches[0] if len(matches) == 1 else None

        def _take_pending(
            self,
            context: Any,
            tool: Any,
            *,
            raw_arguments: str | None = None,
        ) -> _OpenToolSpan | None:
            call_id = _tool_call_id(context)
            if call_id:
                matches = [
                    pending
                    for pending in self._pending
                    if pending.tool is tool and pending.call_id == call_id
                ]
                if len(matches) == 1:
                    self._pending.remove(matches[0])
                    return matches[0]
                return None

            for index, pending in enumerate(self._pending):
                if pending.tool is tool and pending.context_id == id(context):
                    return self._pending.pop(index)

            if raw_arguments is not None:
                matches = [
                    pending
                    for pending in self._pending
                    if pending.tool is tool and pending.raw_arguments == raw_arguments
                ]
                if len(matches) == 1:
                    self._pending.remove(matches[0])
                    return matches[0]

            tool_input = _field(context, "tool_input", _MISSING)
            if tool_input is not _MISSING:
                matches = [
                    pending
                    for pending in self._pending
                    if pending.tool is tool and pending.arguments == tool_input
                ]
                if len(matches) == 1:
                    self._pending.remove(matches[0])
                    return matches[0]

            matches = [pending for pending in self._pending if pending.tool is tool]
            if len(matches) == 1:
                self._pending.remove(matches[0])
                return matches[0]
            return None

        def _finish_pending(
            self,
            pending: _OpenToolSpan,
            *,
            result: Any = None,
            error: BaseException | str | None = None,
        ) -> None:
            effective_error = pending.error if pending.error is not None else error
            end_time = time.time_ns()
            try:
                span = pending.span
                if span is None:
                    span = self._audit_logger.start_tool(
                        name=pending.name,
                        call_id=pending.call_id,
                        arguments=pending.arguments,
                        tool_type=pending.tool_type,
                        start_time=pending.start_time,
                    )
                self._audit_logger.end_tool(
                    span,
                    result=result if effective_error is None else None,
                    error=effective_error,
                    end_time=end_time,
                )
            finally:
                if isinstance(pending.tool, FunctionTool):
                    self._release_function_tool(pending.tool)

        def _release_function_tool(self, tool: FunctionTool) -> None:
            patch = self._function_tool_patches.get(id(tool))
            if patch is None:
                return
            patch.active_calls = max(0, patch.active_calls - 1)
            if patch.active_calls == 0:
                self._restore_function_tool(patch)

        def _restore_function_tool(self, patch: _FunctionToolPatch) -> None:
            tool = patch.tool
            if tool.on_invoke_tool is patch.wrapped_invoke:
                tool.on_invoke_tool = patch.original_invoke
            if (
                patch.wrapped_failure is not None
                and getattr(tool, "_failure_error_function", None) is patch.wrapped_failure
            ):
                if patch.uses_default_failure:
                    set_function_tool_failure_error_function(tool)
                else:
                    set_function_tool_failure_error_function(tool, patch.original_failure)
            self._function_tool_patches.pop(id(tool), None)

    return OpenAIToolRunHooks(audit_logger)
