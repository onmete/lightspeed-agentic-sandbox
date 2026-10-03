"""OpenAI model and tool telemetry adapter tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest


class _AuditRecorder:
    def __init__(self) -> None:
        self.inference_starts: list[tuple[object, dict[str, Any]]] = []
        self.inference_ends: list[tuple[object, dict[str, Any]]] = []
        self.tool_starts: list[tuple[object, dict[str, Any]]] = []
        self.tool_ends: list[tuple[object, dict[str, Any]]] = []

    def start_inference(self, **attributes: Any) -> object:
        span = object()
        self.inference_starts.append((span, attributes))
        return span

    def end_inference(self, span: object, **attributes: Any) -> None:
        self.inference_ends.append((span, attributes))

    def start_tool(self, **attributes: Any) -> object:
        span = object()
        self.tool_starts.append((span, attributes))
        return span

    def end_tool(self, span: object, **attributes: Any) -> None:
        self.tool_ends.append((span, attributes))


class _FakeModel:
    def __init__(
        self,
        *,
        response: Any = None,
        stream_events: list[Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.response = response
        self.stream_events = stream_events or []
        self.error = error

    async def get_response(self, *_args: Any, **_kwargs: Any) -> Any:
        if self.error is not None:
            raise self.error
        return self.response

    def stream_response(self, *_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[Any]:
        for event in self.stream_events:
            if isinstance(event, BaseException):
                raise event
            yield event


class _OutputSchema:
    def is_plain_text(self) -> bool:
        return False


def _function_tool(callback: Any) -> Any:
    from agents.tool import FunctionTool

    return FunctionTool(
        name="exec_command",
        description="Run a shell command.",
        params_json_schema={
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        },
        on_invoke_tool=callback,
        strict_json_schema=False,
    )


def _message(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="message",
        role="assistant",
        content=[SimpleNamespace(type="output_text", text=text)],
    )


@pytest.mark.asyncio
async def test_model_proxy_records_actual_request_and_response() -> None:
    from lightspeed_agentic.providers.openai_telemetry import create_model_proxy

    audit = _AuditRecorder()
    input_items = [
        {"role": "user", "content": [{"type": "input_text", "text": "run pwd"}]},
        {"type": "function_call_output", "call_id": "call-old", "output": "done"},
    ]

    async def invoke(_context: Any, raw_arguments: str) -> str:
        return raw_arguments

    tool = _function_tool(invoke)
    response = SimpleNamespace(
        output=[
            _message("The command succeeded."),
            SimpleNamespace(
                type="reasoning",
                summary=[SimpleNamespace(type="summary_text", text="Inspecting the result.")],
            ),
            SimpleNamespace(
                type="function_call",
                call_id="call-new",
                name="exec_command",
                arguments='{"cmd":"pwd"}',
            ),
        ],
        usage=SimpleNamespace(
            requests=1,
            input_tokens=12,
            output_tokens=8,
            output_tokens_details=SimpleNamespace(reasoning_tokens=3),
        ),
    )
    delegate: Any = _FakeModel(response=response)
    proxy = create_model_proxy(delegate, audit, request_model="gpt-4.1-mini", native_responses=True)

    await proxy.get_response(
        "system prompt",
        input_items,
        object(),
        [tool],
        _OutputSchema(),
        [],
        object(),
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    )

    _, start = audit.inference_starts[0]
    assert start["operation"] == "chat"
    assert start["model"] == "gpt-4.1-mini"
    assert start["system_instructions"] == [{"type": "text", "content": "system prompt"}]
    assert start["output_type"] == "json"
    assert start["input_messages"] == [
        {"role": "user", "parts": [{"type": "text", "content": "run pwd"}]},
        {
            "role": "tool",
            "parts": [
                {
                    "type": "tool_call_response",
                    "id": "call-old",
                    "response": "done",
                }
            ],
        },
    ]
    definition = start["tool_definitions"][0]
    assert definition["type"] == "function"
    assert definition["name"] == "exec_command"
    assert definition["parameters"] == tool.params_json_schema
    assert definition["strict"] is False

    _, ended = audit.inference_ends[0]
    assert ended["output_messages"] == [
        {
            "role": "assistant",
            "parts": [
                {"type": "text", "content": "The command succeeded."},
                {"type": "reasoning", "content": "Inspecting the result."},
                {
                    "type": "tool_call",
                    "id": "call-new",
                    "name": "exec_command",
                    "arguments": {"cmd": "pwd"},
                },
            ],
            "finish_reason": "unknown",
        }
    ]
    assert ended["response_model"] is None
    assert ended["input_tokens"] == 12
    assert ended["output_tokens"] == 8
    assert ended["reasoning_tokens"] == 3
    assert ended["finish_reasons"] == ["unknown"]


@pytest.mark.asyncio
async def test_chat_proxy_uses_normalized_response_without_fabricated_metadata() -> None:
    from lightspeed_agentic.providers.openai_telemetry import create_model_proxy

    audit = _AuditRecorder()
    response = SimpleNamespace(
        output=[_message("done")],
        usage=SimpleNamespace(
            requests=1,
            input_tokens=4,
            output_tokens=2,
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
    )
    proxy = create_model_proxy(
        _FakeModel(response=response),
        audit,
        request_model="offline-model",
        native_responses=False,
    )

    await proxy.get_response(None, "hello", object(), [], None, [], object())

    _, started = audit.inference_starts[0]
    assert started["operation"] == "chat"
    _, ended = audit.inference_ends[0]
    assert ended["output_messages"] == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "done"}],
            "finish_reason": "unknown",
        }
    ]
    assert ended["response_model"] is None
    assert ended["input_tokens"] == 4
    assert ended["output_tokens"] == 2
    assert ended["reasoning_tokens"] is None
    assert ended["finish_reasons"] == ["unknown"]


@pytest.mark.asyncio
async def test_chat_stream_omits_unavailable_model_and_default_usage() -> None:
    from agents.usage import Usage

    from lightspeed_agentic.providers.openai_telemetry import create_model_proxy

    audit = _AuditRecorder()
    completed = SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(output=[_message("done")], usage=Usage()),
    )
    proxy = create_model_proxy(
        _FakeModel(stream_events=[completed]),
        audit,
        request_model="offline-model",
        native_responses=False,
    )
    stream = proxy.stream_response(None, "hello", object(), [], None, [], object())
    async for _event in stream:
        pass
    _, ended = audit.inference_ends[0]
    assert ended["response_model"] is None
    assert ended["input_tokens"] is None
    assert ended["output_tokens"] is None
    assert ended["reasoning_tokens"] is None
    assert ended["finish_reasons"] == ["unknown"]


@pytest.mark.asyncio
async def test_stream_proxy_records_native_response_metadata_and_zero_usage() -> None:
    from lightspeed_agentic.providers.openai_telemetry import create_model_proxy

    audit = _AuditRecorder()
    delta = SimpleNamespace(type="response.output_text.delta", delta="done")
    response = SimpleNamespace(
        model="gpt-4.1-2025-04-14",
        output=[_message("done")],
        usage=SimpleNamespace(
            input_tokens=4,
            output_tokens=0,
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
    )
    completed = SimpleNamespace(type="response.completed", response=response)
    proxy = create_model_proxy(
        _FakeModel(stream_events=[delta, completed]),
        audit,
        request_model="gpt-4.1",
        native_responses=True,
    )
    async for _event in proxy.stream_response(None, "hello", object(), [], None, [], object()):
        pass
    _, ended = audit.inference_ends[0]
    assert ended["response_model"] == "gpt-4.1-2025-04-14"
    assert ended["input_tokens"] == 4
    assert ended["output_tokens"] == 0
    assert ended["reasoning_tokens"] == 0
    assert ended["finish_reasons"] == ["unknown"]


@pytest.mark.asyncio
async def test_non_streaming_native_response_preserves_observed_zero_reasoning_tokens(
    span_exporter,
) -> None:
    from agents.usage import Usage
    from openai.types.responses.response_usage import OutputTokensDetails
    from opentelemetry.trace import StatusCode
    from prometheus_client import REGISTRY

    from lightspeed_agentic.audit import AuditLogger
    from lightspeed_agentic.providers.openai_telemetry import create_model_proxy

    model = "openai-nonstream-zero"
    usage = Usage(
        requests=1,
        input_tokens=4,
        output_tokens=2,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )
    assert "reasoning_tokens" in usage.output_tokens_details.model_fields_set
    input_labels = {
        "gen_ai_token_type": "input",
        "gen_ai_request_model": model,
        "gen_ai_provider_name": "openai",
        "gen_ai_operation_name": "chat",
    }
    output_labels = {**input_labels, "gen_ai_token_type": "output"}
    input_count = REGISTRY.get_sample_value("gen_ai_client_token_usage_count", input_labels) or 0
    output_count = REGISTRY.get_sample_value("gen_ai_client_token_usage_count", output_labels) or 0

    audit = AuditLogger(phase="analysis", model=model, provider="openai")
    proxy = create_model_proxy(
        _FakeModel(response=SimpleNamespace(output=[_message("done")], usage=usage)),
        audit,
        request_model=model,
        native_responses=True,
    )
    await proxy.get_response(None, "hello", object(), [], None, [], object())

    span = next(span for span in span_exporter.get_finished_spans() if span.name == f"chat {model}")
    attributes = dict(span.attributes)
    assert attributes["gen_ai.usage.input_tokens"] == 4
    assert attributes["gen_ai.usage.output_tokens"] == 2
    assert attributes["gen_ai.usage.reasoning.output_tokens"] == 0
    assert span.status.status_code == StatusCode.UNSET
    assert (
        REGISTRY.get_sample_value("gen_ai_client_token_usage_count", input_labels)
        == input_count + 1
    )
    assert (
        REGISTRY.get_sample_value("gen_ai_client_token_usage_count", output_labels)
        == output_count + 1
    )


@pytest.mark.asyncio
async def test_non_streaming_native_response_omits_default_usage_and_token_metrics(span_exporter):
    from agents.usage import Usage
    from opentelemetry.trace import StatusCode
    from prometheus_client import REGISTRY

    from lightspeed_agentic.audit import AuditLogger
    from lightspeed_agentic.providers.openai_telemetry import create_model_proxy

    model = "openai-nonstream-missing-usage"
    usage = Usage()
    input_labels = {
        "gen_ai_token_type": "input",
        "gen_ai_request_model": model,
        "gen_ai_provider_name": "openai",
        "gen_ai_operation_name": "chat",
    }
    output_labels = {**input_labels, "gen_ai_token_type": "output"}
    input_count = REGISTRY.get_sample_value("gen_ai_client_token_usage_count", input_labels)
    output_count = REGISTRY.get_sample_value("gen_ai_client_token_usage_count", output_labels)

    audit = AuditLogger(phase="analysis", model=model, provider="openai")
    proxy = create_model_proxy(
        _FakeModel(response=SimpleNamespace(output=[_message("done")], usage=usage)),
        audit,
        request_model=model,
        native_responses=True,
    )
    await proxy.get_response(None, "hello", object(), [], None, [], object())

    span = next(span for span in span_exporter.get_finished_spans() if span.name == f"chat {model}")
    attributes = dict(span.attributes)
    assert "gen_ai.response.model" not in attributes
    assert "gen_ai.output.messages" in attributes
    assert "gen_ai.usage.input_tokens" not in attributes
    assert "gen_ai.usage.output_tokens" not in attributes
    assert "gen_ai.usage.reasoning.output_tokens" not in attributes
    assert span.status.status_code == StatusCode.UNSET
    assert REGISTRY.get_sample_value("gen_ai_client_token_usage_count", input_labels) == input_count
    assert (
        REGISTRY.get_sample_value("gen_ai_client_token_usage_count", output_labels) == output_count
    )


@pytest.mark.asyncio
async def test_model_proxy_records_non_streaming_error_without_response_data() -> None:
    from lightspeed_agentic.providers.openai_telemetry import create_model_proxy

    audit = _AuditRecorder()
    error = RuntimeError("request failed")
    delegate: Any = _FakeModel(error=error)
    proxy = create_model_proxy(delegate, audit, request_model="gpt-4.1", native_responses=False)

    with pytest.raises(RuntimeError):
        await proxy.get_response(None, "hello", object(), [], None, [], object())
    _, ended = audit.inference_ends[0]
    assert isinstance(ended["error"], RuntimeError)
    assert "output_messages" not in ended
    assert "input_tokens" not in ended


@pytest.mark.asyncio
async def test_model_proxy_records_streaming_error_without_partial_output() -> None:
    from lightspeed_agentic.providers.openai_telemetry import create_model_proxy

    audit = _AuditRecorder()
    partial = SimpleNamespace(type="response.output_text.delta", delta="partial")
    error = RuntimeError("stream failed")
    delegate: Any = _FakeModel(stream_events=[partial, error])
    proxy = create_model_proxy(
        delegate,
        audit,
        request_model="gpt-4.1",
        native_responses=False,
    )

    async def consume_stream() -> None:
        async for _event in proxy.stream_response(None, "hello", object(), [], None, [], object()):
            pass

    with pytest.raises(RuntimeError):
        await consume_stream()
    _, ended = audit.inference_ends[0]
    assert isinstance(ended["error"], RuntimeError)
    assert "output_messages" not in ended


@pytest.mark.asyncio
async def test_stream_proxy_finishes_success_before_completed_event_is_yielded() -> None:
    from lightspeed_agentic.providers.openai_telemetry import create_model_proxy

    audit = _AuditRecorder()
    response = SimpleNamespace(output=[_message("done")], usage=None)
    completed = SimpleNamespace(type="response.completed", response=response)
    proxy = create_model_proxy(
        _FakeModel(stream_events=[completed]),
        audit,
        request_model="gpt-4.1",
        native_responses=True,
    )
    stream = proxy.stream_response(None, "hello", object(), [], None, [], object())

    await stream.__anext__()
    assert len(audit.inference_ends) == 1
    assert "error" not in audit.inference_ends[0][1]

    await stream.aclose()
    assert len(audit.inference_ends) == 1


@pytest.mark.asyncio
async def test_stream_proxy_records_terminal_provider_failure_before_yield(span_exporter) -> None:
    from opentelemetry.trace import StatusCode

    from lightspeed_agentic.audit import AuditLogger
    from lightspeed_agentic.providers.openai_telemetry import create_model_proxy

    request_model = "openai-stream-failed-no-response"
    audit = AuditLogger(phase="analysis", model=request_model, provider="openai")
    failed = SimpleNamespace(type="response.failed")
    proxy = create_model_proxy(
        _FakeModel(stream_events=[failed]),
        audit,
        request_model=request_model,
        native_responses=True,
    )
    stream = proxy.stream_response(None, "hello", object(), [], None, [], object())

    await stream.__anext__()
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR
    assert spans[0].attributes["error.type"] == "response.failed"
    assert (
        not {
            "gen_ai.response.model",
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.output_tokens",
            "gen_ai.usage.reasoning.output_tokens",
            "gen_ai.output.messages",
        }
        & spans[0].attributes.keys()
    )
    await stream.aclose()
    assert len(span_exporter.get_finished_spans()) == 1


@pytest.mark.parametrize(
    ("request_model", "event_type", "native_responses", "usage", "expected_tokens"),
    [
        pytest.param(
            "openai-stream-incomplete-observed",
            "response.incomplete",
            True,
            SimpleNamespace(
                input_tokens=4,
                output_tokens=0,
                output_tokens_details=SimpleNamespace(reasoning_tokens=0),
            ),
            (4, 0, 0),
            id="incomplete-native-observed-zero-usage",
        ),
        pytest.param(
            "openai-stream-failed-observed",
            "response.failed",
            True,
            SimpleNamespace(
                input_tokens=3,
                output_tokens=2,
                output_tokens_details=SimpleNamespace(reasoning_tokens=1),
            ),
            (3, 2, 1),
            id="failed-native-observed-usage",
        ),
        pytest.param(
            "openai-stream-incomplete-missing-usage",
            "response.incomplete",
            True,
            None,
            (None, None, None),
            id="incomplete-native-missing-usage",
        ),
        pytest.param(
            "openai-stream-chat-default-zeros",
            "response.incomplete",
            False,
            SimpleNamespace(
                requests=0,
                input_tokens=0,
                output_tokens=0,
                output_tokens_details=SimpleNamespace(reasoning_tokens=0),
            ),
            (None, None, None),
            id="chat-completions-default-zeros",
        ),
    ],
)
@pytest.mark.asyncio
async def test_stream_proxy_records_terminal_response_observations_with_error_status(
    span_exporter,
    request_model: str,
    event_type: str,
    native_responses: bool,
    usage: Any,
    expected_tokens: tuple[int | None, int | None, int | None],
) -> None:
    from opentelemetry.trace import StatusCode
    from prometheus_client import REGISTRY

    from lightspeed_agentic.audit import AuditLogger
    from lightspeed_agentic.providers.openai_telemetry import create_model_proxy

    response_model = "gpt-4.1-2025-04-14"
    response = SimpleNamespace(model=response_model, usage=usage)
    event = SimpleNamespace(type=event_type, response=response)
    input_labels = {
        "gen_ai_token_type": "input",
        "gen_ai_request_model": request_model,
        "gen_ai_provider_name": "openai",
        "gen_ai_operation_name": "chat",
    }
    output_labels = {**input_labels, "gen_ai_token_type": "output"}
    input_count = REGISTRY.get_sample_value("gen_ai_client_token_usage_count", input_labels) or 0
    output_count = REGISTRY.get_sample_value("gen_ai_client_token_usage_count", output_labels) or 0

    audit = AuditLogger(phase="analysis", model=request_model, provider="openai")
    proxy = create_model_proxy(
        _FakeModel(stream_events=[event]),
        audit,
        request_model=request_model,
        native_responses=native_responses,
    )
    async for _event in proxy.stream_response(None, "hello", object(), [], None, [], object()):
        pass

    span = next(
        span for span in span_exporter.get_finished_spans() if span.name == f"chat {request_model}"
    )
    attributes = dict(span.attributes)
    if native_responses:
        assert attributes["gen_ai.response.model"] == response_model
    else:
        assert "gen_ai.response.model" not in attributes
    for name, expected in (
        ("gen_ai.usage.input_tokens", expected_tokens[0]),
        ("gen_ai.usage.output_tokens", expected_tokens[1]),
        ("gen_ai.usage.reasoning.output_tokens", expected_tokens[2]),
    ):
        if expected is None:
            assert name not in attributes
        else:
            assert attributes[name] == expected
    assert "gen_ai.output.messages" not in attributes
    assert attributes["error.type"] == event_type
    assert span.status.status_code == StatusCode.ERROR
    assert (REGISTRY.get_sample_value("gen_ai_client_token_usage_count", input_labels) or 0) == (
        input_count + (expected_tokens[0] is not None)
    )
    assert (REGISTRY.get_sample_value("gen_ai_client_token_usage_count", output_labels) or 0) == (
        output_count + (expected_tokens[1] is not None)
    )


@pytest.mark.asyncio
async def test_stream_proxy_marks_missing_completion_as_error() -> None:
    from lightspeed_agentic.providers.openai_telemetry import create_model_proxy

    audit = _AuditRecorder()
    partial = SimpleNamespace(type="response.output_text.delta", delta="partial")
    proxy = create_model_proxy(
        _FakeModel(stream_events=[partial]),
        audit,
        request_model="gpt-4.1",
        native_responses=True,
    )

    async for _event in proxy.stream_response(None, "hello", object(), [], None, [], object()):
        pass
    assert audit.inference_ends[0][1]["error"] == "response_incomplete"


@pytest.mark.asyncio
async def test_tool_hooks_record_actual_arguments_result_and_failures_once() -> None:
    from lightspeed_agentic.providers.openai_telemetry import create_tool_hooks

    audit = _AuditRecorder()
    hooks = create_tool_hooks(audit)
    raw_arguments = '{"cmd":"pwd"}'
    context = SimpleNamespace(
        tool_name="exec_command",
        tool_call_id="provider-call-1",
        tool_arguments=raw_arguments,
    )

    async def succeed(_context: Any, _arguments: str) -> dict[str, int]:
        return {"exit_code": 0}

    tool = _function_tool(succeed)
    await hooks.on_tool_start(context, object(), tool)
    result = await tool.on_invoke_tool(context, raw_arguments)
    await hooks.on_tool_end(context, object(), tool, result)

    _, started = audit.tool_starts[0]
    assert started["name"] == "exec_command"
    assert started["call_id"] == "provider-call-1"
    assert started["arguments"] == {"cmd": "pwd"}
    _, ended = audit.tool_ends[0]
    assert ended["result"] == {"exit_code": 0}
    assert ended.get("error") is None

    class _ToolFailureError(Exception):
        pass

    async def fail(_context: Any, _arguments: str) -> str:
        raise _ToolFailureError("sensitive tool failure")

    failing_tool = _function_tool(fail)
    failure_context = SimpleNamespace(
        tool_name="exec_command",
        tool_call_id="provider-call-2",
        tool_arguments=raw_arguments,
    )
    await hooks.on_tool_start(failure_context, object(), failing_tool)
    failing_invoke = failing_tool.on_invoke_tool
    with pytest.raises(_ToolFailureError):
        await failing_invoke(failure_context, raw_arguments)

    await hooks.on_tool_end(failure_context, object(), failing_tool, "not a successful result")
    assert len(audit.tool_ends) == 2
    _, failed = audit.tool_ends[1]
    assert isinstance(failed["error"], _ToolFailureError)
    assert failed["result"] is None
    hooks.close()


@pytest.mark.asyncio
async def test_overlapping_tool_failure_cannot_finish_its_sibling() -> None:
    from agents import function_tool

    from lightspeed_agentic.providers.openai_telemetry import create_tool_hooks

    audit = _AuditRecorder()
    hooks = create_tool_hooks(audit)
    sibling_started = asyncio.Event()
    release_sibling = asyncio.Event()
    calls = 0

    @function_tool(
        name_override="exec_command",
        failure_error_function=lambda _context, _error: "handled failure",
    )
    async def execute(cmd: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            await sibling_started.wait()
            raise RuntimeError("native execution failed")
        sibling_started.set()
        await release_sibling.wait()
        return f"completed {cmd}"

    raw_arguments = '{"cmd":"pwd"}'
    first = SimpleNamespace(
        tool_name="exec_command", tool_call_id="A", tool_arguments=raw_arguments, run_config=None
    )
    sibling = SimpleNamespace(
        tool_name="exec_command", tool_call_id="B", tool_arguments=raw_arguments, run_config=None
    )
    await hooks.on_tool_start(first, object(), execute)
    await hooks.on_tool_start(sibling, object(), execute)
    first_task = asyncio.create_task(execute.on_invoke_tool(first, raw_arguments))
    sibling_task = asyncio.create_task(execute.on_invoke_tool(sibling, raw_arguments))
    try:
        first_result = await asyncio.wait_for(first_task, timeout=5)
        assert first_result == "handled failure"
        first_span, first_end = audit.tool_ends[0]
        assert first_span is audit.tool_starts[0][0]
        assert isinstance(first_end["error"], RuntimeError)
        assert first_end["result"] is None

        await hooks.on_tool_end(first, object(), execute, first_result)
        assert len(audit.tool_ends) == 1
        assert hooks._find_pending(first, execute) is None

        release_sibling.set()
        sibling_result = await asyncio.wait_for(sibling_task, timeout=5)
        await hooks.on_tool_end(sibling, object(), execute, sibling_result)
        sibling_span, sibling_end = audit.tool_ends[1]
        assert sibling_span is audit.tool_starts[1][0]
        assert sibling_end["result"] == "completed pwd"
        assert sibling_end["error"] is None
        assert sibling_end["end_time"] >= first_end["end_time"]
    finally:
        release_sibling.set()
        for task in (first_task, sibling_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(first_task, sibling_task, return_exceptions=True)
        hooks.close()


@pytest.mark.asyncio
async def test_duplicate_call_ids_do_not_select_an_arbitrary_pending_span() -> None:
    from lightspeed_agentic.providers.openai_telemetry import create_tool_hooks

    audit = _AuditRecorder()
    hooks = create_tool_hooks(audit)
    tool = SimpleNamespace(name="exec_command")
    contexts = [
        SimpleNamespace(tool_call_id="duplicate", tool_arguments='{"cmd":"pwd"}') for _ in range(2)
    ]
    for context in contexts:
        await hooks.on_tool_start(context, object(), tool)
    try:
        assert hooks._find_pending(contexts[0], tool) is None
        await hooks.on_tool_end(contexts[0], object(), tool, "ambiguous result")
        assert audit.tool_ends == []
    finally:
        hooks.close()


@pytest.mark.asyncio
async def test_custom_function_tool_named_view_image_can_return_text_successfully() -> None:
    from agents.tool import FunctionTool

    from lightspeed_agentic.providers.openai_telemetry import create_tool_hooks

    async def succeed(_context: Any, _arguments: str) -> str:
        return "valid custom tool result"

    tool = FunctionTool(
        name="view_image",
        description="A custom admitted tool with the same public name.",
        params_json_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        on_invoke_tool=succeed,
        strict_json_schema=False,
    )
    raw_arguments = '{"path":"report.txt"}'
    context = SimpleNamespace(
        tool_name="view_image",
        tool_call_id="custom-view-call",
        tool_arguments=raw_arguments,
    )
    audit = _AuditRecorder()
    hooks = create_tool_hooks(audit)

    await hooks.on_tool_start(context, object(), tool)
    result = await tool.on_invoke_tool(context, raw_arguments)
    await hooks.on_tool_end(context, object(), tool, result)

    _, ended = audit.tool_ends[0]
    assert ended["result"] == "valid custom tool result"
    assert ended.get("error") is None
    hooks.close()


@pytest.mark.asyncio
async def test_registered_skill_load_uses_native_tool_hooks() -> None:
    from pathlib import Path

    from agents.sandbox.capabilities.skills import (
        LocalDirLazySkillSource,
        Skills,
    )
    from agents.sandbox.entries import LocalDir

    from lightspeed_agentic.providers.openai_telemetry import create_tool_hooks

    class _Skills(Skills):
        async def load_skill(self, skill_name: str) -> dict[str, str]:
            return {"status": "loaded", "skill_name": skill_name}

    skills = _Skills(lazy_from=LocalDirLazySkillSource(source=LocalDir(src=Path.cwd())))
    object.__setattr__(skills, "session", object())
    tool: Any = skills.tools()[0]
    raw_arguments = '{"skill_name":"guide"}'
    context = SimpleNamespace(
        tool_name="load_skill",
        tool_call_id="skill-call",
        tool_arguments=raw_arguments,
    )

    audit = _AuditRecorder()
    hooks = create_tool_hooks(audit)
    await hooks.on_tool_start(context, object(), tool)
    result = await tool.on_invoke_tool(context, raw_arguments)
    await hooks.on_tool_end(context, object(), tool, result)

    _, started = audit.tool_starts[0]
    assert started["name"] == "load_skill"
    assert started["call_id"] == "skill-call"
    assert started["arguments"] == {"skill_name": "guide"}
    _, ended = audit.tool_ends[0]
    assert ended["result"] == {"status": "loaded", "skill_name": "guide"}
    assert ended.get("error") is None
    hooks.close()


@pytest.mark.parametrize("supports_pty", [False, True])
@pytest.mark.asyncio
async def test_exec_command_typed_timeout_is_error_before_sdk_formats_result(
    supports_pty: bool,
) -> None:
    from agents.sandbox import ExecTimeoutError
    from agents.sandbox.capabilities.tools.shell_tool import ExecCommandTool

    from lightspeed_agentic.providers.openai_telemetry import create_tool_hooks

    class _TimedOutSession:
        def supports_pty(self) -> bool:
            return supports_pty

        async def exec(self, *command: Any, **kwargs: Any) -> Any:
            raise ExecTimeoutError(command=command, timeout_s=kwargs["timeout"])

        async def pty_exec_start(self, *command: Any, **kwargs: Any) -> Any:
            raise ExecTimeoutError(
                command=command,
                timeout_s=kwargs.get("yield_time_s"),
            )

    audit = _AuditRecorder()
    hooks = create_tool_hooks(audit)
    session = _TimedOutSession()
    tool = ExecCommandTool(session=session)
    hooks.configure_shell_tools(SimpleNamespace(exec_command=tool))

    raw_arguments = '{"cmd":"sleep 1","yield_time_ms":1}'
    context = SimpleNamespace(
        tool_name="exec_command",
        tool_call_id="timeout-call",
        tool_arguments=raw_arguments,
    )
    await hooks.on_tool_start(context, object(), tool)
    invoke_context = SimpleNamespace(
        tool_name="exec_command",
        tool_call_id="timeout-call",
        tool_arguments=raw_arguments,
    )
    result = await tool.on_invoke_tool(invoke_context, raw_arguments)
    await hooks.on_tool_end(context, object(), tool, result)

    assert isinstance(result, str)
    _, ended = audit.tool_ends[0]
    assert isinstance(ended["error"], ExecTimeoutError)
    assert ended["result"] is None

    hooks.close()


@pytest.mark.parametrize("error_kind", ["session_missing", "stdin_unavailable"])
@pytest.mark.asyncio
async def test_write_stdin_sdk_handled_failures_record_error_without_result(
    error_kind: str,
) -> None:
    from agents.sandbox.capabilities.tools.shell_tool import (
        ExecCommandTool,
        WriteStdinTool,
    )
    from agents.sandbox.errors import PtySessionNotFoundError

    from lightspeed_agentic.providers.openai_telemetry import create_tool_hooks

    failure: BaseException
    if error_kind == "session_missing":
        failure = PtySessionNotFoundError(session_id=987654321)
    else:
        failure = RuntimeError("stdin is not available for this process")

    class _Session:
        def __init__(self, error: BaseException) -> None:
            self.error = error

        def supports_pty(self) -> bool:
            return True

        async def exec(self, *_args: Any, **_kwargs: Any) -> Any:
            raise NotImplementedError

        async def pty_exec_start(self, *_args: Any, **_kwargs: Any) -> Any:
            raise NotImplementedError

        async def pty_write_stdin(self, **_kwargs: Any) -> Any:
            raise self.error

    audit = _AuditRecorder()
    hooks = create_tool_hooks(audit)
    session = _Session(failure)
    exec_tool = ExecCommandTool(session=session)
    tool = WriteStdinTool(session=session)
    hooks.configure_shell_tools(SimpleNamespace(exec_command=exec_tool, write_stdin=tool))

    raw_arguments = '{"session_id":987654321,"chars":"","yield_time_ms":0}'
    context = SimpleNamespace(
        tool_name="write_stdin",
        tool_call_id="stdin-call",
        tool_arguments=raw_arguments,
    )
    await hooks.on_tool_start(context, object(), tool)
    invoke_context = SimpleNamespace(
        tool_name="write_stdin",
        tool_call_id="stdin-call",
        tool_arguments=raw_arguments,
    )
    result = await tool.on_invoke_tool(invoke_context, raw_arguments)
    await hooks.on_tool_end(context, object(), tool, result)

    assert isinstance(result, str)
    _, ended = audit.tool_ends[0]
    assert isinstance(ended["error"], type(failure))
    assert ended["result"] is None
    hooks.close()


@pytest.mark.asyncio
async def test_view_image_error_result_type_records_error_without_result() -> None:
    import io

    from agents.sandbox.capabilities.tools.view_image import ViewImageTool

    from lightspeed_agentic.providers.openai_telemetry import create_tool_hooks

    class _PathPolicy:
        def absolute_workspace_path(self, path: Any) -> Any:
            return path

        def relative_path(self, path: Any) -> Any:
            return path

    class _Session:
        def _workspace_path_policy(self) -> _PathPolicy:
            return _PathPolicy()

        async def read(self, _path: Any, **_kwargs: Any) -> io.BytesIO:
            return io.BytesIO(b"not an image")

    audit = _AuditRecorder()
    hooks = create_tool_hooks(audit)
    tool = ViewImageTool(session=_Session())
    raw_arguments = '{"path":"unsupported.bin"}'
    context = SimpleNamespace(
        tool_name="view_image",
        tool_call_id="image-call",
        tool_arguments=raw_arguments,
    )
    await hooks.on_tool_start(context, object(), tool)
    result = await tool.on_invoke_tool(context, raw_arguments)
    await hooks.on_tool_end(context, object(), tool, result)

    assert isinstance(result, str)
    _, ended = audit.tool_ends[0]
    assert ended["error"] == "view_image_failed"
    assert ended["result"] is None
    hooks.close()


@pytest.mark.asyncio
async def test_view_image_typed_success_records_successful_result() -> None:
    import base64
    import io

    from agents.sandbox.capabilities.tools.view_image import ViewImageTool
    from agents.tool import ToolOutputImage

    from lightspeed_agentic.providers.openai_telemetry import create_tool_hooks

    class _PathPolicy:
        def absolute_workspace_path(self, path: Any) -> Any:
            return path

        def relative_path(self, path: Any) -> Any:
            return path

    class _Session:
        def _workspace_path_policy(self) -> _PathPolicy:
            return _PathPolicy()

        async def read(self, _path: Any, **_kwargs: Any) -> io.BytesIO:
            payload = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
                "x8AAwMCAO+jU1kAAAAASUVORK5CYII="
            )
            return io.BytesIO(payload)

    audit = _AuditRecorder()
    hooks = create_tool_hooks(audit)
    tool = ViewImageTool(session=_Session())
    raw_arguments = '{"path":"valid.png"}'
    context = SimpleNamespace(
        tool_name="view_image",
        tool_call_id="image-call",
        tool_arguments=raw_arguments,
    )
    await hooks.on_tool_start(context, object(), tool)
    result = await tool.on_invoke_tool(context, raw_arguments)
    await hooks.on_tool_end(context, object(), tool, result)

    assert isinstance(result, ToolOutputImage)
    _, ended = audit.tool_ends[0]
    assert ended.get("error") is None
    assert isinstance(ended["result"], ToolOutputImage)
    hooks.close()


@pytest.mark.asyncio
async def test_mcp_is_error_result_records_error_without_result() -> None:
    from agents.mcp.server import MCPServer
    from agents.mcp.util import MCPUtil
    from mcp.types import CallToolResult, TextContent
    from mcp.types import Tool as MCPTool

    from lightspeed_agentic.providers.openai_telemetry import create_tool_hooks

    tool_definition = MCPTool(
        name="fail_tool",
        description="Test MCP tool",
        inputSchema={"type": "object", "properties": {}},
    )
    call_result = CallToolResult(
        content=[TextContent(type="text", text="controlled MCP error")],
        isError=True,
    )
    extractor_calls: list[bool | None] = []

    def custom_data_extractor(context: Any) -> None:
        extractor_calls.append(context.is_error)

    class _Server(MCPServer):
        def __init__(self) -> None:
            super().__init__(custom_data_extractor=custom_data_extractor)

        @property
        def name(self) -> str:
            return "offline"

        async def connect(self) -> None:
            return None

        async def cleanup(self) -> None:
            return None

        async def list_tools(self, _run_context: Any = None, _agent: Any = None) -> list[Any]:
            return [tool_definition]

        async def call_tool(
            self,
            tool_name: str,
            arguments: dict[str, Any] | None,
            meta: dict[str, Any] | None = None,
        ) -> CallToolResult:
            assert tool_name == tool_definition.name
            assert arguments == {}
            assert meta is None
            return call_result

        async def list_prompts(self) -> Any:
            raise NotImplementedError

        async def get_prompt(self, _name: str, _arguments: dict[str, Any] | None = None) -> Any:
            raise NotImplementedError

    audit = _AuditRecorder()
    hooks = create_tool_hooks(audit)
    server = _Server()
    hooks.configure_mcp_servers([server])
    tool = MCPUtil.to_function_tool(
        tool_definition,
        server,
        convert_schemas_to_strict=False,
    )
    context = SimpleNamespace(
        tool_name=tool.name,
        tool_call_id="mcp-call",
        tool_arguments="{}",
    )

    await hooks.on_tool_start(context, object(), tool)
    result = await tool.on_invoke_tool(context, "{}")
    await hooks.on_tool_end(context, object(), tool, result)

    assert result is not None
    assert extractor_calls == [True]
    _, ended = audit.tool_ends[0]
    assert ended["error"] == "mcp_tool_error"
    assert ended["result"] is None
    hooks.close()


@pytest.mark.asyncio
async def test_filesystem_apply_patch_hooks_capture_editor_arguments_and_errors() -> None:
    from lightspeed_agentic.providers.openai_telemetry import create_tool_hooks

    class _Editor:
        async def create_file(self, _operation: Any) -> SimpleNamespace:
            return SimpleNamespace(status="completed", output="created")

        async def update_file(self, _operation: Any) -> SimpleNamespace:
            raise _EditorFailureError("editor failed")

        async def delete_file(self, _operation: Any) -> SimpleNamespace:
            return SimpleNamespace(status="completed", output="deleted")

    class _EditorFailureError(Exception):
        pass

    audit = _AuditRecorder()
    hooks = create_tool_hooks(audit)
    tool = SimpleNamespace(name="apply_patch", type="custom", editor=_Editor())
    hooks.configure_filesystem_tools(SimpleNamespace(apply_patch=tool))

    context = object()
    await hooks.on_tool_start(context, object(), tool)
    operation = SimpleNamespace(
        type="create_file",
        path="new.txt",
        diff="contents",
        move_to=None,
        ctx_wrapper=context,
    )
    await tool.editor.create_file(operation)
    await hooks.on_tool_end(context, object(), tool, "created")

    _, started = audit.tool_starts[0]
    assert started["arguments"] == [
        {"type": "create_file", "path": "new.txt", "diff": "contents", "move_to": None}
    ]
    assert started["call_id"] == ""
    assert audit.tool_ends[0][1]["result"] == "created"

    failure_context = object()
    await hooks.on_tool_start(failure_context, object(), tool)
    failed_operation = SimpleNamespace(
        type="update_file",
        path="existing.txt",
        diff="change",
        move_to=None,
        ctx_wrapper=failure_context,
    )
    with pytest.raises(_EditorFailureError):
        await tool.editor.update_file(failed_operation)
    await hooks.on_tool_end(failure_context, object(), tool, "formatted error")

    failed = audit.tool_ends[1][1]
    assert isinstance(failed["error"], _EditorFailureError)
    assert failed["result"] is None
    hooks.close()
