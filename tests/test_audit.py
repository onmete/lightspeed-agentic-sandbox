"""Tests for provider-independent GenAI operation spans."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from opentelemetry import trace
from opentelemetry.trace import SpanKind, StatusCode

from lightspeed_agentic.audit import GenAIRecorder


def _recorder(**kwargs) -> GenAIRecorder:
    defaults = {"phase": "analysis", "provider": "anthropic", "agenticrun_uid": "run-uid"}
    defaults.update(kwargs)
    return GenAIRecorder(**defaults)


_SCHEMAS = {
    kind: Draft202012Validator(
        json.loads((Path(__file__).parent / "fixtures" / f"genai-v1.41-{kind}.json").read_text())
    )
    for kind in ("input", "output")
}


def test_model_span_preserves_ordered_schema_messages_and_parent(span_exporter) -> None:
    tracer = trace.get_tracer("test")
    inputs = [
        {"role": "user", "parts": [{"type": "text", "content": "é?"}]},
        {
            "role": "assistant",
            "parts": [
                {"type": "reasoning", "content": "first"},
                {"type": "tool_call", "id": "call-1", "name": "bash", "arguments": {"x": 1}},
            ],
        },
        {
            "role": "tool",
            "parts": [{"type": "tool_call_response", "id": "call-1", "response": "résultat"}],
        },
    ]
    instructions = [{"type": "text", "content": "Système"}]
    definitions = [{"name": "bash", "description": "écho", "parameters": {"type": "object"}}]
    outputs = [
        {
            "role": "assistant",
            "finish_reason": "stop",
            "parts": [
                {"type": "reasoning", "content": "one"},
                {"type": "text", "content": "二"},
            ],
        },
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "three"}],
            "finish_reason": "stop",
        },
    ]
    with tracer.start_as_current_span("run") as run:
        recorder = _recorder(
            capture_content=True,
            parent_context=trace.set_span_in_context(run),
            output_type="json",
            server_address="api.example.test",
        )
        handle = recorder.start_model(inputs, instructions, "claude", tool_definitions=definitions)
        recorder.end_model(
            handle,
            outputs,
            "claude-resolved",
            {"input_tokens": 23, "output_tokens": 15, "reasoning_tokens": 4},
            None,
        )

    model, run_span = span_exporter.get_finished_spans()
    assert model.name == "chat claude"
    assert model.kind == SpanKind.CLIENT
    assert model.parent.span_id == run_span.context.span_id
    attrs = model.attributes
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.provider.name"] == "anthropic"
    assert attrs["gen_ai.request.model"] == "claude"
    assert attrs["gen_ai.response.model"] == "claude-resolved"
    assert attrs["agenticrun.uid"] == "run-uid"
    assert attrs["agenticrun.phase"] == "analysis"
    assert attrs["gen_ai.output.type"] == "json"
    assert attrs["server.address"] == "api.example.test"
    for key, messages in (
        ("gen_ai.input.messages", inputs),
        ("gen_ai.system_instructions", instructions),
        ("gen_ai.tool.definitions", definitions),
        ("gen_ai.output.messages", outputs),
    ):
        assert attrs[key] == json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        assert json.loads(attrs[key]) == messages
    _SCHEMAS["input"].validate(json.loads(attrs["gen_ai.input.messages"]))
    _SCHEMAS["output"].validate(json.loads(attrs["gen_ai.output.messages"]))
    assert attrs["gen_ai.usage.input_tokens"] == 23
    assert attrs["gen_ai.usage.output_tokens"] == 15  # includes four reasoning tokens
    assert attrs["gen_ai.usage.reasoning.output_tokens"] == 4
    assert "gen_ai.usage.reasoning_tokens" not in attrs
    assert model.status.status_code == StatusCode.UNSET
    assert not model.events


def test_tools_are_siblings_of_models_under_fixed_parent(span_exporter) -> None:
    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("run") as run:
        recorder = _recorder(capture_content=True, parent_context=trace.set_span_in_context(run))
        model = recorder.start_model([], None, "claude")
        with tracer.start_as_current_span("other"):
            tool1 = recorder.start_tool("bash", "call-1", '{"command":"écho"}')
            tool2 = recorder.start_tool("read", "call-2", "opaque output")
            recorder.end_tool(tool2, '[1,"二"]', None)
            recorder.end_tool(tool1, "plain response", None)
        recorder.end_model(model, [], None, {"input_tokens": 0, "output_tokens": 0}, None)

    spans = span_exporter.get_finished_spans()
    run_span = next(s for s in spans if s.name == "run")
    operation_spans = [
        s for s in spans if s.name in {"chat claude", "execute_tool bash", "execute_tool read"}
    ]
    assert len(operation_spans) == 3
    assert all(s.parent.span_id == run_span.context.span_id for s in operation_spans)
    assert all(s.context.trace_id == run_span.context.trace_id for s in operation_spans)
    tools = {s.name: s for s in operation_spans if s.name.startswith("execute_tool")}
    bash = tools["execute_tool bash"]
    read = tools["execute_tool read"]
    assert bash.kind == read.kind == SpanKind.INTERNAL
    for name, span, call_id in (("bash", bash, "call-1"), ("read", read, "call-2")):
        assert span.attributes["gen_ai.operation.name"] == "execute_tool"
        assert span.attributes["gen_ai.tool.name"] == name
        assert span.attributes["gen_ai.tool.call.id"] == call_id
        assert span.attributes["agenticrun.uid"] == "run-uid"
        assert span.attributes["agenticrun.phase"] == "analysis"
        assert span.attributes["gen_ai.tool.type"] == "function"
        assert span.status.status_code == StatusCode.UNSET
        assert not span.events
    assert bash.attributes["gen_ai.tool.call.arguments"] == '{"command":"écho"}'
    assert bash.attributes["gen_ai.tool.call.result"] == '{"content":"plain response"}'
    assert read.attributes["gen_ai.tool.call.arguments"] == '{"content":"opaque output"}'
    assert read.attributes["gen_ai.tool.call.result"] == '{"content":[1,"二"]}'
    model_span = next(s for s in operation_spans if s.name == "chat claude")
    assert model_span.attributes["gen_ai.input.messages"] == "[]"
    assert model_span.attributes["gen_ai.output.messages"] == "[]"
    assert "gen_ai.system_instructions" not in model_span.attributes
    assert "gen_ai.tool.definitions" not in model_span.attributes
    assert model_span.attributes["gen_ai.usage.input_tokens"] == 0
    assert model_span.attributes["gen_ai.usage.output_tokens"] == 0
    assert "gen_ai.usage.reasoning.output_tokens" not in model_span.attributes


@pytest.mark.parametrize(
    ("arguments", "result", "expected_arguments", "expected_result"),
    [
        (' ["π", false] ', [1, {"a": 2}], {"content": ["π", False]}, {"content": [1, {"a": 2}]}),
        ("12", "null", {"content": 12}, {"content": None}),
        (None, '"héllo"', {"content": None}, {"content": "héllo"}),
        ('{"nested":[1]}', {"ok": True}, {"nested": [1]}, {"ok": True}),
        ("", "unparsed 🌍:  \n", {"content": ""}, {"content": "unparsed 🌍:  \n"}),
        ("NaN", '{"value":NaN}', {"content": "NaN"}, {"content": '{"value":NaN}'}),
    ],
)
def test_tool_span_attributes_are_json_objects(
    span_exporter, arguments, result, expected_arguments, expected_result
) -> None:
    recorder = _recorder(capture_content=True)
    tool = recorder.start_tool("read", "call-1", arguments)
    recorder.end_tool(tool, result, None)
    attrs = span_exporter.get_finished_spans()[0].attributes
    for key, expected in (
        ("gen_ai.tool.call.arguments", expected_arguments),
        ("gen_ai.tool.call.result", expected_result),
    ):
        decoded = json.loads(attrs[key])
        assert isinstance(decoded, dict)
        assert decoded == expected


def test_recorder_captures_ambient_parent_once(span_exporter) -> None:
    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("run") as run:
        recorder = _recorder(capture_content=True)
        with tracer.start_as_current_span("other"):
            model = recorder.start_model([], None, "claude")
            tool = recorder.start_tool("bash", "call-1", {})
            recorder.end_tool(tool, {}, None)
            recorder.end_model(model, None, None, {}, None)

    spans = span_exporter.get_finished_spans()
    operations = [s for s in spans if s.name in {"chat claude", "execute_tool bash"}]
    assert len(operations) == 2
    assert all(s.parent.span_id == run.context.span_id for s in operations)
    assert all(s.context.trace_id == run.context.trace_id for s in operations)


def test_unobserved_output_and_usage_are_not_invented(span_exporter) -> None:
    recorder = _recorder(capture_content=True)
    missing = recorder.start_model([], None, "missing")
    recorder.end_model(missing, None, None, {}, None)
    empty = recorder.start_model([], None, "empty")
    recorder.end_model(
        empty, [], None, {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}, None
    )

    missing_span, empty_span = span_exporter.get_finished_spans()
    assert "gen_ai.output.messages" not in missing_span.attributes
    assert not any(key.startswith("gen_ai.usage.") for key in missing_span.attributes)
    assert missing_span.status.status_code == StatusCode.UNSET
    assert empty_span.attributes["gen_ai.output.messages"] == "[]"
    _SCHEMAS["output"].validate(json.loads(empty_span.attributes["gen_ai.output.messages"]))
    assert empty_span.attributes["gen_ai.usage.input_tokens"] == 0
    assert empty_span.attributes["gen_ai.usage.output_tokens"] == 0
    assert empty_span.attributes["gen_ai.usage.reasoning.output_tokens"] == 0
    assert "gen_ai.usage.reasoning_tokens" not in empty_span.attributes
    assert "gen_ai.output.type" not in empty_span.attributes
    assert "server.address" not in empty_span.attributes
    assert empty_span.status.status_code == StatusCode.UNSET


def test_capture_off_keeps_identity_usage_errors_and_no_content(span_exporter) -> None:
    recorder = _recorder(capture_content=False, phase="", agenticrun_uid="")
    model = recorder.start_model(
        [{"role": "user", "parts": [{"type": "text", "content": "private"}]}],
        [{"type": "text", "content": "system secret"}],
        "model",
        tool_definitions=[{"name": "private"}],
    )
    tool = recorder.start_tool("read", "call-9", "secret argument")
    recorder.end_tool(tool, "secret result", ValueError("sensitive tool error"))
    recorder.end_model(
        model,
        [
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": "secret output"}],
                "finish_reason": "error",
            }
        ],
        "response-model",
        {"input_tokens": 0, "output_tokens": 0},
        RuntimeError("sensitive model error"),
    )
    tool_span, model_span = span_exporter.get_finished_spans()
    for span in (tool_span, model_span):
        assert span.status.status_code == StatusCode.ERROR
        assert "agenticrun.uid" not in span.attributes
        assert "agenticrun.phase" not in span.attributes
        assert not span.events
        assert "sensitive" not in str(span.attributes)
    assert tool_span.attributes["gen_ai.operation.name"] == "execute_tool"
    assert tool_span.attributes["gen_ai.tool.name"] == "read"
    assert tool_span.attributes["gen_ai.tool.call.id"] == "call-9"
    assert tool_span.attributes["error.type"] == "ValueError"
    assert "gen_ai.tool.call.arguments" not in tool_span.attributes
    assert "gen_ai.tool.call.result" not in tool_span.attributes
    assert model_span.attributes["gen_ai.operation.name"] == "chat"
    assert model_span.attributes["gen_ai.request.model"] == "model"
    assert model_span.attributes["gen_ai.provider.name"] == "anthropic"
    assert model_span.attributes["gen_ai.response.model"] == "response-model"
    assert model_span.attributes["gen_ai.usage.input_tokens"] == 0
    assert model_span.attributes["gen_ai.usage.output_tokens"] == 0
    assert model_span.attributes["error.type"] == "RuntimeError"
    for key in (
        "gen_ai.input.messages",
        "gen_ai.system_instructions",
        "gen_ai.tool.definitions",
        "gen_ai.output.messages",
        "gen_ai.usage.reasoning.output_tokens",
    ):
        assert key not in model_span.attributes


def test_error_does_not_claim_successful_tool_result_when_capture_on(span_exporter) -> None:
    recorder = _recorder(capture_content=True)
    tool = recorder.start_tool("bash", "call-1", {"command": "echo"})
    recorder.end_tool(tool, {"secret": "partial data"}, OSError("private failure"))
    attrs = span_exporter.get_finished_spans()[0].attributes
    assert attrs["gen_ai.tool.call.arguments"] == '{"command":"echo"}'
    assert span_exporter.get_finished_spans()[0].status.status_code == StatusCode.ERROR
    assert attrs["error.type"] == "OSError"
    assert "gen_ai.tool.call.result" not in attrs


def test_failed_model_captures_partial_messages_and_no_error_text(span_exporter) -> None:
    recorder = _recorder(capture_content=True)
    model = recorder.start_model([], None, "requested")
    output = [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "partial"}],
            "finish_reason": "error",
        }
    ]
    recorder.end_model(
        model,
        output,
        None,
        {"input_tokens": 7, "output_tokens": 2},
        RuntimeError("do not record this error message"),
    )
    span = span_exporter.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == "RuntimeError"
    assert span.attributes["gen_ai.output.messages"] == json.dumps(
        output, ensure_ascii=False, separators=(",", ":")
    )
    _SCHEMAS["output"].validate(json.loads(span.attributes["gen_ai.output.messages"]))
    assert span.attributes["gen_ai.usage.input_tokens"] == 7
    assert span.attributes["gen_ai.usage.output_tokens"] == 2
    assert "do not record this error message" not in str(span.attributes)
    assert not span.events


def test_cleanup_ends_unfinished_operations_once_and_records_tool_duration(
    span_exporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    observations: list[tuple[str, float]] = []

    class ToolObservation:
        def observe(self, duration: float) -> None:
            observations.append(("bash", duration))

    monkeypatch.setattr(
        "lightspeed_agentic.audit.tool_duration.labels",
        lambda *, gen_ai_tool_name: ToolObservation() if gen_ai_tool_name == "bash" else None,
    )
    recorder = _recorder(capture_content=True)
    recorder.start_model([], None, "unfinished")
    recorder.start_tool("bash", "call-1", "input")
    recorder.close()
    recorder.close()
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    for span in spans:
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes["error.type"] == "incomplete"
        assert span.end_time is not None
        assert not span.events
    assert (
        "gen_ai.tool.call.result"
        not in next(s for s in spans if s.name == "execute_tool bash").attributes
    )
    assert len(observations) == 1
    assert observations[0][1] >= 0
