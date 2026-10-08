"""OpenAI generation spans from completed SDK responses."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import SpanKind, StatusCode

from lightspeed_agentic.types import ProviderQueryOptions


class _FakeAgent:
    pass


def _options(cwd: Path) -> ProviderQueryOptions:
    return ProviderQueryOptions(
        prompt="user prompt",
        system_prompt="system instructions",
        model="gpt-4.1-mini",
        max_turns=1,
        allowed_tools=[],
        cwd=str(cwd),
    )


def _model_response(
    output: list[Any],
    *,
    response_id: str | None = None,
    request_id: str | None = None,
    requests: int = 1,
    input_tokens: int = 0,
    output_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> Any:
    from agents.items import ModelResponse
    from agents.usage import Usage
    from openai.types.responses.response_usage import OutputTokensDetails

    return ModelResponse(
        output=output,
        usage=Usage(
            requests=requests,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=reasoning_tokens),
        ),
        response_id=response_id,
        request_id=request_id,
    )


def _invocation_span() -> Any:
    return trace.get_tracer("openai-generation-tests").start_span(
        "invoke_agent",
        attributes={"agenticrun.uid": "run-123", "agenticrun.phase": "task"},
    )


def _generation_span(exporter: Any) -> Any:
    return next(
        span
        for span in exporter.get_finished_spans()
        if span.attributes.get("gen_ai.operation.name") == "chat"
    )


class _FakeStreamingResult:
    def __init__(
        self,
        agent: Any,
        hooks: Any,
        *,
        response: Any,
        events_before_end: list[Any],
        events_after_end: list[Any],
        failure: BaseException | None,
        other_response: Any | None,
    ) -> None:
        self._agent = agent
        self._hooks = hooks
        self._response = response
        self._events_before_end = events_before_end
        self._events_after_end = events_after_end
        self._failure = failure
        self._other_response = other_response
        self.context_wrapper = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=6,
                output_tokens=7,
                output_tokens_details=SimpleNamespace(reasoning_tokens=0),
            ),
            model=None,
        )
        self.final_output = "terminal answer"

    async def stream_events(self):
        await self._hooks.on_llm_start(None, self._agent, None, [])
        if self._other_response is not None:
            other_agent = _FakeAgent()
            await self._hooks.on_llm_start(None, other_agent, None, [])
            await self._hooks.on_llm_end(None, other_agent, self._other_response)
        for event in self._events_before_end:
            yield event
        if self._failure is not None:
            raise self._failure
        if self._response is not None:
            await self._hooks.on_llm_end(None, self._agent, self._response)
        for event in self._events_after_end:
            yield event


def _install_provider_query(
    monkeypatch: pytest.MonkeyPatch,
    cwd: Path,
    *,
    provider_type: str = "openai",
    base_url: str | None = None,
    azure_api_version: str | None = None,
    response: Any | None = None,
    events_before_end: list[Any] | None = None,
    events_after_end: list[Any] | Callable[[Any], list[Any]] | None = None,
    failure: BaseException | None = None,
    other_response: Any | None = None,
) -> tuple[Any, ProviderQueryOptions]:
    import agents.models.openai_chatcompletions as chatcompletions
    import agents.models.openai_responses as responses
    import agents.sandbox as agents_sandbox
    from agents import Runner

    import lightspeed_agentic.providers.openai as openai_provider

    main_agent = _FakeAgent()
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", provider_type)
    if base_url is None:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    if azure_api_version is None:
        monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
    else:
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", azure_api_version)

    monkeypatch.setattr(openai_provider, "_ensure_openai_init", lambda: None)
    monkeypatch.setattr(openai_provider, "_build_manifest", lambda _cwd: None)
    monkeypatch.setattr(agents_sandbox, "SandboxAgent", lambda **_kwargs: main_agent)
    monkeypatch.setattr(responses, "OpenAIResponsesModel", lambda **_kwargs: object())
    monkeypatch.setattr(chatcompletions, "OpenAIChatCompletionsModel", lambda **_kwargs: object())

    provider = openai_provider.OpenAIProvider()
    provider._client = object()
    if provider_type == "azure":
        monkeypatch.setattr(provider, "_build_azure_model", lambda *_args: object())

    model_response = response if response is not None else _model_response([])

    def run_streamed(agent: Any, _prompt: str, *, hooks: Any, **_kwargs: Any) -> Any:
        after_end = (
            events_after_end(agent) if callable(events_after_end) else events_after_end or []
        )
        return _FakeStreamingResult(
            agent,
            hooks,
            response=model_response,
            events_before_end=events_before_end or [],
            events_after_end=after_end,
            failure=failure,
            other_response=other_response,
        )

    monkeypatch.setattr(Runner, "run_streamed", staticmethod(run_streamed))
    return provider, _options(cwd)


@pytest.mark.asyncio
async def test_responses_generation_maps_typed_output_and_metadata(
    span_exporter: Any, tmp_path: Path
) -> None:
    from openai.types.responses.response_custom_tool_call import (
        ResponseCustomToolCall,
    )
    from openai.types.responses.response_function_tool_call import (
        ResponseFunctionToolCall,
    )
    from openai.types.responses.response_output_message import ResponseOutputMessage
    from openai.types.responses.response_output_refusal import ResponseOutputRefusal
    from openai.types.responses.response_output_text import ResponseOutputText
    from openai.types.responses.response_reasoning_item import (
        Content as ReasoningContent,
    )
    from openai.types.responses.response_reasoning_item import (
        ResponseReasoningItem,
        Summary,
    )

    from lightspeed_agentic.providers.openai import _create_generation_hooks

    patch_input = '*** Begin Patch\n*** Update File: "雪 file.py"\n+quoted "line"\n*** End Patch'
    response = _model_response(
        [
            ResponseReasoningItem(
                id="rs_1",
                content=[ReasoningContent(text="think-first", type="reasoning_text")],
                summary=[Summary(text="summary-next", type="summary_text")],
                type="reasoning",
            ),
            ResponseFunctionToolCall(
                arguments='{"pod": "pod-a"}',
                call_id="call-lookup",
                name="lookup",
                type="function_call",
            ),
            ResponseFunctionToolCall(
                arguments="{not-json",
                call_id="call-raw",
                name="raw_tool",
                type="function_call",
            ),
            ResponseFunctionToolCall(
                arguments="NaN",
                call_id="call-nan",
                name="raw_nan",
                type="function_call",
            ),
            ResponseFunctionToolCall(
                arguments='{"too_large": 1e999}',
                call_id="call-overflow",
                name="raw_overflow",
                type="function_call",
            ),
            ResponseCustomToolCall(
                input=patch_input,
                call_id="call-patch",
                name="apply_patch",
                type="custom_tool_call",
            ),
            ResponseOutputMessage(
                id="msg_1",
                content=[
                    ResponseOutputText(
                        text="say-after-tools",
                        type="output_text",
                        annotations=[],
                        logprobs=[],
                    ),
                    ResponseOutputRefusal(refusal="refusal text", type="refusal"),
                    ResponseOutputText(
                        text="",
                        type="output_text",
                        annotations=[],
                        logprobs=[],
                    ),
                ],
                role="assistant",
                status="completed",
                type="message",
            ),
        ],
        response_id="resp_actual",
        request_id="transport_request_only",
        input_tokens=0,
        output_tokens=9,
        reasoning_tokens=4,
    )
    options = _options(tmp_path)
    main_agent = _FakeAgent()
    invocation = _invocation_span()

    with trace.use_span(invocation, end_on_exit=False):
        parent_context = otel_context.get_current()
        hooks = _create_generation_hooks(main_agent, options, parent_context, api_type="responses")
        await hooks.on_llm_start(None, main_agent, "ignored", [{"role": "user"}])
        assert (
            trace.get_current_span().get_span_context().span_id
            == invocation.get_span_context().span_id
        )
        await hooks.on_llm_end(None, main_agent, response)
        assert (
            trace.get_current_span().get_span_context().span_id
            == invocation.get_span_context().span_id
        )
    invocation.end()

    span = _generation_span(span_exporter)
    assert span.name == "chat gpt-4.1-mini"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET
    assert span.parent.span_id == invocation.get_span_context().span_id
    assert span.attributes["gen_ai.operation.name"] == "chat"
    assert span.attributes["gen_ai.request.model"] == options.model
    assert span.attributes["gen_ai.provider.name"] == "openai"
    assert span.attributes["openai.api.type"] == "responses"
    assert span.attributes["agenticrun.uid"] == "run-123"
    assert span.attributes["agenticrun.phase"] == "task"
    assert span.attributes["gen_ai.response.id"] == "resp_actual"
    assert span.attributes["gen_ai.usage.input_tokens"] == 0
    assert span.attributes["gen_ai.usage.output_tokens"] == 9
    assert span.attributes["gen_ai.usage.reasoning.output_tokens"] == 4
    assert "gen_ai.response.model" not in span.attributes
    assert "gen_ai.response.finish_reasons" not in span.attributes
    assert "gen_ai.input.messages" not in span.attributes
    assert "gen_ai.system_instructions" not in span.attributes
    assert json.loads(span.attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "parts": [
                {"type": "reasoning", "content": "think-first"},
                {"type": "reasoning", "content": "summary-next"},
                {
                    "type": "tool_call",
                    "id": "call-lookup",
                    "name": "lookup",
                    "arguments": {"pod": "pod-a"},
                },
                {
                    "type": "tool_call",
                    "id": "call-raw",
                    "name": "raw_tool",
                    "arguments": "{not-json",
                },
                {
                    "type": "tool_call",
                    "id": "call-nan",
                    "name": "raw_nan",
                    "arguments": "NaN",
                },
                {
                    "type": "tool_call",
                    "id": "call-overflow",
                    "name": "raw_overflow",
                    "arguments": '{"too_large": 1e999}',
                },
                {
                    "type": "tool_call",
                    "id": "call-patch",
                    "name": "apply_patch",
                    "arguments": patch_input,
                },
                {"type": "text", "content": "say-after-tools"},
                {"type": "refusal", "content": "refusal text"},
                {"type": "text", "content": ""},
            ],
        }
    ]


@pytest.mark.asyncio
async def test_chat_converter_output_uses_consistent_completion_id_not_requested_model(
    span_exporter: Any, tmp_path: Path
) -> None:
    from agents.models.chatcmpl_converter import Converter

    from lightspeed_agentic.providers.openai import _create_generation_hooks

    tool_call = SimpleNamespace(
        type="function",
        id="call-chat-1",
        function=SimpleNamespace(name="lookup", arguments='{"pod": "pod-b"}'),
        extra_content=None,
    )
    message = SimpleNamespace(
        role="assistant",
        reasoning_content="think-first",
        thinking_blocks=None,
        content="say-second",
        refusal=None,
        audio=None,
        tool_calls=[tool_call],
    )
    output_items = Converter.message_to_output_items(
        message,
        provider_data={
            "model": "requested-model-copy",
            "response_id": "chatcmpl_actual",
        },
    )
    response = _model_response(
        output_items,
        response_id=None,
        request_id="transport_request_only",
        input_tokens=13,
        output_tokens=0,
        reasoning_tokens=0,
    )
    options = _options(tmp_path)
    main_agent = _FakeAgent()
    invocation = _invocation_span()

    with trace.use_span(invocation, end_on_exit=False):
        hooks = _create_generation_hooks(
            main_agent,
            options,
            otel_context.get_current(),
            api_type="chat_completions",
        )
        await hooks.on_llm_start(None, main_agent, None, [])
        await hooks.on_llm_end(None, main_agent, response)
    invocation.end()

    span = _generation_span(span_exporter)
    assert span.attributes["openai.api.type"] == "chat_completions"
    assert span.attributes["gen_ai.response.id"] == "chatcmpl_actual"
    assert span.attributes["gen_ai.usage.input_tokens"] == 13
    assert span.attributes["gen_ai.usage.output_tokens"] == 0
    assert "gen_ai.usage.reasoning.output_tokens" not in span.attributes
    assert "gen_ai.response.model" not in span.attributes
    assert "gen_ai.response.finish_reasons" not in span.attributes
    assert "transport_request_only" not in span.attributes.values()
    assert json.loads(span.attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "parts": [
                {"type": "reasoning", "content": "think-first"},
                {"type": "text", "content": "say-second"},
                {
                    "type": "tool_call",
                    "id": "call-chat-1",
                    "name": "lookup",
                    "arguments": {"pod": "pod-b"},
                },
            ],
        }
    ]


@pytest.mark.asyncio
async def test_generation_omits_usage_without_observed_request(
    span_exporter: Any, tmp_path: Path
) -> None:
    from openai.types.responses.response_output_message import ResponseOutputMessage

    from lightspeed_agentic.providers.openai import _create_generation_hooks

    response = _model_response(
        [
            ResponseOutputMessage(
                id="msg_1",
                content=[],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        requests=0,
        input_tokens=12,
        output_tokens=34,
        reasoning_tokens=56,
    )
    options = _options(tmp_path)
    main_agent = _FakeAgent()
    invocation = _invocation_span()
    with trace.use_span(invocation, end_on_exit=False):
        hooks = _create_generation_hooks(
            main_agent,
            options,
            otel_context.get_current(),
            api_type="responses",
        )
        await hooks.on_llm_start(None, main_agent, None, [])
        await hooks.on_llm_end(None, main_agent, response)
    invocation.end()

    attributes = _generation_span(span_exporter).attributes
    assert "gen_ai.usage.input_tokens" not in attributes
    assert "gen_ai.usage.output_tokens" not in attributes
    assert "gen_ai.usage.reasoning.output_tokens" not in attributes


@pytest.mark.parametrize(
    ("provider_type", "base_url", "api_version", "expected_api_type"),
    [
        pytest.param("openai", None, None, "responses", id="native-openai"),
        pytest.param(
            "openai",
            "https://vllm.example/v1",
            None,
            "chat_completions",
            id="compatible",
        ),
        pytest.param(
            "azure",
            "https://api.openai.com/v1",
            "2024-10-21",
            "chat_completions",
            id="azure-chat-version",
        ),
        pytest.param(
            "azure",
            "https://custom.example/v1",
            "2025-03-01-preview",
            "responses",
            id="azure-responses-version",
        ),
    ],
)
@pytest.mark.asyncio
async def test_query_api_type_tracks_existing_client_routing(
    span_exporter: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider_type: str,
    base_url: str | None,
    api_version: str | None,
    expected_api_type: str,
) -> None:
    provider, options = _install_provider_query(
        monkeypatch,
        tmp_path,
        provider_type=provider_type,
        base_url=base_url,
        azure_api_version=api_version,
    )
    invocation = _invocation_span()
    with trace.use_span(invocation, end_on_exit=False):
        events = [event async for event in provider.query(options)]
        assert (
            trace.get_current_span().get_span_context().span_id
            == invocation.get_span_context().span_id
        )
    invocation.end()

    span = _generation_span(span_exporter)
    assert span.attributes["openai.api.type"] == expected_api_type
    assert span.attributes["gen_ai.provider.name"] == "openai"
    assert span.parent.span_id == invocation.get_span_context().span_id
    assert span.attributes["agenticrun.uid"] == "run-123"
    assert span.attributes["agenticrun.phase"] == "task"
    assert events[-1].type == "result"


@pytest.mark.asyncio
async def test_native_apply_patch_trace_input_does_not_change_events_or_logs(
    span_exporter: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from agents.items import ToolCallItem, ToolCallOutputItem
    from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
    from openai.types.responses.response_custom_tool_call import (
        ResponseCustomToolCall,
    )
    from openai.types.responses.response_function_tool_call import (
        ResponseFunctionToolCall,
    )
    from openai.types.responses.response_reasoning_text_delta_event import (
        ResponseReasoningTextDeltaEvent,
    )
    from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

    from lightspeed_agentic.logging import EventLogger
    from lightspeed_agentic.types import (
        ContentBlockStopEvent,
        ResultEvent,
        TextDeltaEvent,
        ThinkingDeltaEvent,
        ToolCallEvent,
        ToolResultEvent,
        stringify,
    )

    patch_input = "*** Begin Patch\n*** Update File: file.py\n+exact patch input\n*** End Patch"
    custom_call = ResponseCustomToolCall(
        input=patch_input,
        call_id="call-patch",
        name="apply_patch",
        type="custom_tool_call",
    )
    response = _model_response(
        [
            custom_call,
            ResponseFunctionToolCall(
                arguments='{"path": "file.py"}',
                call_id="call-dict",
                name="read_file",
                type="function_call",
            ),
        ],
        response_id="resp-events",
        input_tokens=10,
        output_tokens=4,
    )
    reasoning_delta = ResponseReasoningTextDeltaEvent(
        content_index=0,
        delta="think-delta",
        item_id="rs_1",
        output_index=0,
        sequence_number=1,
        type="response.reasoning_text.delta",
    )
    text_delta = ResponseTextDeltaEvent(
        content_index=0,
        delta="text-delta",
        item_id="msg_1",
        logprobs=[],
        output_index=0,
        sequence_number=2,
        type="response.output_text.delta",
    )
    dictionary_call = {
        "type": "function_call",
        "call_id": "call-dict",
        "name": "read_file",
        "arguments": {"path": "dictionary.py"},
    }

    def tool_events(agent: Any) -> list[Any]:
        return [
            RunItemStreamEvent(name="tool_called", item=ToolCallItem(agent, custom_call)),
            RunItemStreamEvent(
                name="tool_output",
                item=ToolCallOutputItem(
                    agent, SimpleNamespace(call_id="call-patch"), {"safe": True}
                ),
            ),
            RunItemStreamEvent(name="tool_called", item=ToolCallItem(agent, dictionary_call)),
            RunItemStreamEvent(
                name="tool_output",
                item=ToolCallOutputItem(agent, SimpleNamespace(call_id="call-dict"), "read result"),
            ),
        ]

    provider, options = _install_provider_query(
        monkeypatch,
        tmp_path,
        response=response,
        events_before_end=[
            RawResponsesStreamEvent(data=reasoning_delta),
            RawResponsesStreamEvent(data=text_delta),
        ],
        events_after_end=tool_events,
    )
    events = [event async for event in provider.query(options)]

    tool_calls = [event for event in events if isinstance(event, ToolCallEvent)]
    tool_results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert [event for event in events if isinstance(event, ThinkingDeltaEvent)] == [
        ThinkingDeltaEvent(thinking="think-delta")
    ]
    assert [event for event in events if isinstance(event, TextDeltaEvent)] == [
        TextDeltaEvent(text="text-delta")
    ]
    assert [(event.name, event.input, event.call_id) for event in tool_calls] == [
        ("apply_patch", "", "call-patch"),
        ("read_file", "", "call-dict"),
    ]
    assert [event.trace_input for event in tool_calls] == [
        patch_input,
        stringify({"path": "dictionary.py"}),
    ]
    assert [(event.output, event.call_id) for event in tool_results] == [
        (stringify({"safe": True}), "call-patch"),
        ("read result", "call-dict"),
    ]
    assert isinstance(events[-2], ContentBlockStopEvent)
    assert events[-1] == ResultEvent(
        text="terminal answer",
        input_tokens=6,
        output_tokens=7,
        response_model=options.model,
    )

    generation = _generation_span(span_exporter)
    assert json.loads(generation.attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "tool_call",
                    "id": "call-patch",
                    "name": "apply_patch",
                    "arguments": patch_input,
                },
                {
                    "type": "tool_call",
                    "id": "call-dict",
                    "name": "read_file",
                    "arguments": {"path": "file.py"},
                },
            ],
        }
    ]

    with caplog.at_level(logging.INFO, logger="lightspeed_agentic"):
        event_logger = EventLogger("main")
        for event in events:
            event_logger.log(event)
    assert "tool_use: apply_patch()" in caplog.text
    assert "tool_use: read_file()" in caplog.text
    assert patch_input not in caplog.text
    assert 'tool_result: {"safe": true}' in caplog.text
    assert "result: tokens=13" in caplog.text


@pytest.mark.asyncio
async def test_query_excludes_other_agent_generation(
    span_exporter: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from openai.types.responses.response_output_message import ResponseOutputMessage
    from openai.types.responses.response_output_text import ResponseOutputText

    main_response = _model_response(
        [
            ResponseOutputMessage(
                id="main_msg",
                content=[
                    ResponseOutputText(
                        text="main output",
                        type="output_text",
                        annotations=[],
                        logprobs=[],
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        ]
    )
    other_response = _model_response(
        [
            ResponseOutputMessage(
                id="other_msg",
                content=[
                    ResponseOutputText(
                        text="nested output",
                        type="output_text",
                        annotations=[],
                        logprobs=[],
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        ]
    )
    provider, options = _install_provider_query(
        monkeypatch,
        tmp_path,
        response=main_response,
        other_response=other_response,
    )
    [event async for event in provider.query(options)]

    generations = [
        span
        for span in span_exporter.get_finished_spans()
        if span.attributes.get("gen_ai.operation.name") == "chat"
    ]
    assert len(generations) == 1
    attributes = generations[0].attributes
    assert json.loads(attributes["gen_ai.output.messages"]) == [
        {"role": "assistant", "parts": [{"type": "text", "content": "main output"}]}
    ]
    assert "nested output" not in attributes["gen_ai.output.messages"]


@pytest.mark.asyncio
async def test_cancellation_terminally_ignores_late_background_runner_callbacks(
    span_exporter: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import contextlib

    from agents import Runner
    from agents.stream_events import RawResponsesStreamEvent
    from openai.types.responses.response_output_message import ResponseOutputMessage
    from openai.types.responses.response_output_text import ResponseOutputText
    from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

    from lightspeed_agentic.types import TextDeltaEvent

    cancellation = asyncio.CancelledError("cancelled while provider was yielding")
    delta = ResponseTextDeltaEvent(
        content_index=0,
        delta="observed delta only",
        item_id="msg_partial",
        logprobs=[],
        output_index=0,
        sequence_number=1,
        type="response.output_text.delta",
    )
    completed_response = _model_response(
        [
            ResponseOutputMessage(
                id="msg_before_cancel",
                content=[
                    ResponseOutputText(
                        text="completed before cancellation",
                        type="output_text",
                        annotations=[],
                        logprobs=[],
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        response_id="resp_before_cancel",
    )
    late_response = _model_response(
        [
            ResponseOutputMessage(
                id="msg_late",
                content=[
                    ResponseOutputText(
                        text="late background output",
                        type="output_text",
                        annotations=[],
                        logprobs=[],
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        response_id="resp_late",
    )
    provider, options = _install_provider_query(monkeypatch, tmp_path)
    continue_background = asyncio.Event()
    results: list[Any] = []

    class _BackgroundStreamingResult:
        def __init__(self, agent: Any, hooks: Any) -> None:
            self._agent = agent
            self._hooks = hooks
            self.background_task: asyncio.Task[Any] | None = None

        async def stream_events(self):
            queue: asyncio.Queue[Any] = asyncio.Queue()

            async def continue_run() -> None:
                await self._hooks.on_llm_start(None, self._agent, None, [])
                await self._hooks.on_llm_end(None, self._agent, completed_response)
                await self._hooks.on_llm_start(None, self._agent, None, [])
                await queue.put(RawResponsesStreamEvent(data=delta))
                await continue_background.wait()
                await self._hooks.on_llm_end(None, self._agent, late_response)
                await self._hooks.on_llm_start(None, self._agent, None, [])
                await self._hooks.on_llm_end(None, self._agent, late_response)

            self.background_task = asyncio.create_task(continue_run())
            yield await queue.get()

    def run_streamed(agent: Any, _prompt: str, *, hooks: Any, **_kwargs: Any) -> Any:
        result = _BackgroundStreamingResult(agent, hooks)
        results.append(result)
        return result

    monkeypatch.setattr(Runner, "run_streamed", staticmethod(run_streamed))

    invocation = _invocation_span()
    stream = provider.query(options)
    background_task: asyncio.Task[Any] | None = None
    cancellation_delivered = False
    try:
        with trace.use_span(invocation, end_on_exit=False):
            event = await anext(stream)
            assert event == TextDeltaEvent(text="observed delta only")
            background_task = results[0].background_task
            assert background_task is not None
            assert not background_task.done()
            with pytest.raises(asyncio.CancelledError) as exc_info:
                await stream.athrow(cancellation)
            assert exc_info.value is cancellation
            cancellation_delivered = True
    finally:
        if not cancellation_delivered:
            with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                await stream.athrow(cancellation)
        invocation.end()
        continue_background.set()
        if background_task is None and results:
            background_task = results[0].background_task
        if background_task is not None:
            await asyncio.wait_for(background_task, timeout=1)

    generations = [
        span
        for span in span_exporter.get_finished_spans()
        if span.attributes.get("gen_ai.operation.name") == "chat"
    ]
    assert len(generations) == 2
    completed_span, cancelled_span = generations
    assert completed_span.parent.span_id == invocation.get_span_context().span_id
    assert completed_span.status.status_code == StatusCode.UNSET
    assert json.loads(completed_span.attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "completed before cancellation"}],
        }
    ]
    assert cancelled_span.parent.span_id == invocation.get_span_context().span_id
    assert cancelled_span.status.status_code == StatusCode.ERROR
    assert cancelled_span.attributes["error.type"] == "CancelledError"
    assert "gen_ai.output.messages" not in cancelled_span.attributes
