"""Tests for audit OTel instrumentation."""

from __future__ import annotations

import json

import pytest
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from lightspeed_agentic.audit import AuditLogger
from lightspeed_agentic.types import (
    ContentBlockStopEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)


def _make_logger(**kwargs) -> AuditLogger:
    defaults = {"phase": "analysis", "model": "m", "provider": "p", "enabled": True}
    defaults.update(kwargs)
    return AuditLogger(**defaults)


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant: {value}")


def _strict_json_loads(value: str) -> object:
    return json.loads(value, parse_constant=_reject_nonstandard_constant)


class TestToolSpanNaming:
    def test_tool_span_uses_execute_tool_name(self, span_exporter) -> None:
        al = _make_logger()
        al.process_event(ToolCallEvent(name="bash", input="ls"))
        al.process_event(ToolResultEvent(output="file.txt"))
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "execute_tool bash"

    def test_tool_span_has_gen_ai_attributes(self, span_exporter) -> None:
        al = _make_logger()
        al.process_event(ToolCallEvent(name="bash", input="ls -la", call_id="call_1"))
        al.process_event(ToolResultEvent(output="done", call_id="call_1"))
        spans = span_exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)
        assert attrs["gen_ai.operation.name"] == "execute_tool"
        assert attrs["gen_ai.tool.name"] == "bash"
        assert attrs["gen_ai.tool.call.id"] == "call_1"
        assert json.loads(attrs["gen_ai.tool.call.arguments"]) == {"content": "ls -la"}
        assert json.loads(attrs["gen_ai.tool.call.result"]) == {"content": "done"}
        assert "tool.input" not in attrs
        assert "tool.output" not in attrs

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            (
                '{"message": "雪 \\"hello\\"\\nnext"}',
                {"message": '雪 "hello"\nnext'},
            ),
            ("17", {"content": 17}),
            ('["one", 2, null]', {"content": ["one", 2, None]}),
            ("null", {"content": None}),
            ('"scalar"', {"content": "scalar"}),
            ('raw "quote"\n雪', {"content": 'raw "quote"\n雪'}),
            ("", {"content": ""}),
        ],
    )
    def test_tool_payloads_are_exported_as_json_objects(
        self, span_exporter, payload: str, expected: dict[str, object]
    ) -> None:
        al = _make_logger()
        al.process_event(ToolCallEvent(name="bash", input=payload, call_id="c1"))
        al.process_event(ToolResultEvent(output=payload, call_id="c1"))

        attrs = dict(span_exporter.get_finished_spans()[0].attributes)
        expected_json = json.dumps(expected, ensure_ascii=False, separators=(",", ":"))
        assert _strict_json_loads(attrs["gen_ai.tool.call.arguments"]) == expected
        assert _strict_json_loads(attrs["gen_ai.tool.call.result"]) == expected
        assert attrs["gen_ai.tool.call.arguments"] == expected_json
        assert attrs["gen_ai.tool.call.result"] == expected_json

    def test_tool_span_has_agenticrun_correlation(self, span_exporter) -> None:
        al = _make_logger(phase="execution", agenticrun_uid="run-uid")
        al.process_event(ToolCallEvent(name="bash", input="ls"))
        al.process_event(ToolResultEvent(output="done"))

        attrs = dict(span_exporter.get_finished_spans()[0].attributes)
        assert attrs["agenticrun.uid"] == "run-uid"
        assert attrs["agenticrun.phase"] == "execution"

    def test_tool_span_does_not_invent_correlation(self, span_exporter) -> None:
        al = _make_logger(phase="", agenticrun_uid="")
        al.process_event(ToolCallEvent(name="bash", input="ls"))
        al.process_event(ToolResultEvent(output="done"))

        attrs = dict(span_exporter.get_finished_spans()[0].attributes)
        assert "agenticrun.uid" not in attrs
        assert "agenticrun.phase" not in attrs

    def test_tool_span_kind_internal(self, span_exporter) -> None:
        al = _make_logger()
        al.process_event(ToolCallEvent(name="bash", input="ls"))
        al.process_event(ToolResultEvent(output="done"))
        spans = span_exporter.get_finished_spans()
        assert spans[0].kind == trace.SpanKind.INTERNAL

    @pytest.mark.parametrize("trace_input", ['{"patch":"雪\\n"}', ""])
    def test_trace_arguments_do_not_change_developer_logs(
        self, span_exporter, caplog, trace_input: str
    ) -> None:
        from lightspeed_agentic.logging import EventLogger

        event = ToolCallEvent(
            name="apply_patch", input="legacy input", call_id="patch-1", trace_input=trace_input
        )
        al = _make_logger()
        al.process_event(event)
        al.process_event(ToolResultEvent(output="done", call_id="patch-1"))
        with caplog.at_level("INFO", logger="lightspeed_agentic"):
            EventLogger("analysis").log(event)

        attrs = span_exporter.get_finished_spans()[0].attributes
        expected = json.loads(trace_input) if trace_input else {"content": ""}
        assert json.loads(attrs["gen_ai.tool.call.arguments"]) == expected
        assert caplog.messages == ["[provider:analysis] tool_use: apply_patch(legacy input)"]

    @pytest.mark.parametrize("error_type", [None, "tool_error"])
    def test_tool_error_requires_explicit_provider_evidence(
        self, span_exporter, error_type: str | None
    ) -> None:
        al = _make_logger()
        al.process_event(ToolCallEvent(name="lookup", input="{}", call_id="call-1"))
        al.process_event(
            ToolResultEvent(
                output='{"error":"literal result"}', call_id="call-1", error_type=error_type
            )
        )

        span = span_exporter.get_finished_spans()[0]
        assert json.loads(span.attributes["gen_ai.tool.call.result"]) == {"error": "literal result"}
        assert span.status.status_code == (StatusCode.ERROR if error_type else StatusCode.OK)
        assert span.attributes.get("error.type") == error_type


class TestToolPayloadBoundaries:
    @pytest.mark.parametrize(
        "payload",
        [
            "NaN",
            "Infinity",
            "-Infinity",
            '{"value": NaN}',
            '{"value": Infinity}',
            '{"value": -Infinity}',
            "1e999",
            '{"value": 1e999}',
        ],
    )
    def test_nonstandard_numbers_fall_back_to_raw_tool_payload(
        self, span_exporter, payload: str
    ) -> None:
        al = _make_logger()
        al.process_event(ToolCallEvent(name="bash", input=payload, call_id="c1"))
        al.process_event(ToolResultEvent(output=payload, call_id="c1"))

        span = span_exporter.get_finished_spans()[0]
        attrs = dict(span.attributes)
        expected = {"content": payload}
        expected_json = json.dumps(expected, ensure_ascii=False, separators=(",", ":"))
        for name in ("gen_ai.tool.call.arguments", "gen_ai.tool.call.result"):
            assert attrs[name] == expected_json
            assert _strict_json_loads(attrs[name]) == expected
        assert span.status.status_code == StatusCode.OK

    def test_deep_result_falls_back_to_exact_raw_string_and_ends_tool(self, span_exporter) -> None:
        payload = "[" * 10_000 + "0" + "]" * 10_000
        al = _make_logger()
        al.process_event(ToolCallEvent(name="bash", input="{}", call_id="c1"))
        al.process_event(ToolResultEvent(output=payload, call_id="c1"))

        span = span_exporter.get_finished_spans()[0]
        attrs = dict(span.attributes)
        expected = {"content": payload}
        encoded = attrs["gen_ai.tool.call.result"]
        assert encoded == json.dumps(expected, ensure_ascii=False, separators=(",", ":"))
        assert _strict_json_loads(encoded) == expected
        assert span.status.status_code == StatusCode.OK

    def test_encoder_recursion_falls_back_to_raw_result_and_ends_tool(
        self, span_exporter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = '{"value": 1}'
        original_dumps = json.dumps
        injected = False

        def fail_once(value, *args, **kwargs):
            nonlocal injected
            if not injected and value == {"value": 1}:
                injected = True
                raise RecursionError("injected JSON encoder limit")
            return original_dumps(value, *args, **kwargs)

        monkeypatch.setattr(json, "dumps", fail_once)
        al = _make_logger()
        al.process_event(ToolCallEvent(name="bash", input="{}", call_id="c1"))
        al.process_event(ToolResultEvent(output=payload, call_id="c1"))

        spans = span_exporter.get_finished_spans()
        assert injected
        assert len(spans) == 1
        span = spans[0]
        attrs = dict(span.attributes)
        expected = {"content": payload}
        encoded = attrs["gen_ai.tool.call.result"]
        assert encoded == original_dumps(expected, ensure_ascii=False, separators=(",", ":"))
        assert _strict_json_loads(encoded) == expected
        assert span.status.status_code == StatusCode.OK


class TestToolSpanLifecycle:
    def test_tool_result_ends_span(self, span_exporter) -> None:
        al = _make_logger()
        al.process_event(ToolCallEvent(name="bash", input="ls", call_id="c1"))
        assert span_exporter.get_finished_spans() == []

        al.process_event(ToolResultEvent(output="done", call_id="c1"))
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code == StatusCode.OK

    def test_parallel_tool_calls_matched_by_id(self, span_exporter) -> None:
        al = _make_logger()
        al.process_event(ToolCallEvent(name="bash", input="ls", call_id="c1"))
        al.process_event(ToolCallEvent(name="cat", input="file.txt", call_id="c2"))
        al.process_event(ToolResultEvent(output="content", call_id="c2"))
        al.process_event(ToolResultEvent(output="file.txt", call_id="c1"))
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 2
        by_name = {s.name: dict(s.attributes) for s in spans}
        assert json.loads(by_name["execute_tool bash"]["gen_ai.tool.call.result"]) == {
            "content": "file.txt"
        }
        assert json.loads(by_name["execute_tool cat"]["gen_ai.tool.call.result"]) == {
            "content": "content"
        }

    def test_missing_id_result_matches_single_pending_tool(self, span_exporter) -> None:
        al = _make_logger()
        al.process_event(ToolCallEvent(name="bash", input="ls", call_id="c1"))
        al.process_event(ToolResultEvent(output="done"))
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        assert attrs["gen_ai.tool.call.id"] == "c1"
        assert json.loads(attrs["gen_ai.tool.call.result"]) == {"content": "done"}
        assert spans[0].status.status_code == StatusCode.OK

    def test_tool_call_without_id_does_not_publish_internal_id(self, span_exporter) -> None:
        al = _make_logger()
        al.process_event(ToolCallEvent(name="bash", input="ls"))
        al.process_event(ToolResultEvent(output=""))

        span = span_exporter.get_finished_spans()[0]
        attrs = dict(span.attributes)
        assert "gen_ai.tool.call.id" not in attrs
        assert json.loads(attrs["gen_ai.tool.call.arguments"]) == {"content": "ls"}
        assert json.loads(attrs["gen_ai.tool.call.result"]) == {"content": ""}
        assert span.status.status_code == StatusCode.OK

    def test_close_pending_tools_does_not_flush_choice_events(self, span_exporter) -> None:
        al = _make_logger(capture_content=True)
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("agent") as agent_span:
            al.set_parent_context(trace.set_span_in_context(agent_span))
            al.process_event(ToolCallEvent(name="bash", input="ls", call_id="c1"))
            al.process_event(TextDeltaEvent(text="buffered"))
            al.close_pending_tools()

        spans = span_exporter.get_finished_spans()
        agent = next(span for span in spans if span.name == "agent")
        tool = next(span for span in spans if span.name == "execute_tool bash")
        assert not agent.events
        assert tool.status.status_code == StatusCode.ERROR
        assert dict(tool.attributes)["error.type"] == "missing_tool_result"

    def test_complete_ends_orphan_tool_spans_with_error(self, span_exporter) -> None:
        al = _make_logger()
        al.process_event(ToolCallEvent(name="bash", input="ls", call_id="c1"))
        al.process_event(ToolCallEvent(name="cat", input="f", call_id="c2"))
        al.complete(input_tokens=0, output_tokens=0)
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 2
        for span in spans:
            attrs = dict(span.attributes)
            assert span.status.status_code == StatusCode.ERROR
            assert attrs["error.type"] == "missing_tool_result"
            assert "gen_ai.tool.call.result" not in attrs

    def test_ambiguous_missing_id_result_is_not_attached(self, span_exporter) -> None:
        al = _make_logger()
        al.process_event(ToolCallEvent(name="bash", input="ls", call_id="c1"))
        al.process_event(ToolCallEvent(name="cat", input="f", call_id="c2"))
        al.process_event(ToolResultEvent(output="ambiguous"))
        al.complete(input_tokens=0, output_tokens=0)

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 2
        for span in spans:
            attrs = dict(span.attributes)
            assert span.status.status_code == StatusCode.ERROR
            assert attrs["error.type"] == "missing_tool_result"
            assert "gen_ai.tool.call.result" not in attrs

    def test_unknown_result_id_does_not_match_pending_tool(self, span_exporter) -> None:
        al = _make_logger()
        al.process_event(ToolCallEvent(name="bash", input="ls", call_id="c1"))
        al.process_event(ToolResultEvent(output="unmatched", call_id="unknown"))
        al.complete(input_tokens=0, output_tokens=0)

        span = span_exporter.get_finished_spans()[0]
        attrs = dict(span.attributes)
        assert span.status.status_code == StatusCode.ERROR
        assert attrs["error.type"] == "missing_tool_result"
        assert "gen_ai.tool.call.result" not in attrs


class TestGenAiChoiceEvents:
    def test_text_emits_choice_event_with_content(self, span_exporter) -> None:
        al = _make_logger(capture_content=True)
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("chat test") as span:
            al.set_parent_context(trace.set_span_in_context(span))
            al.process_event(TextDeltaEvent(text="hello "))
            al.process_event(TextDeltaEvent(text="world"))
            al.process_event(ContentBlockStopEvent())
        spans = span_exporter.get_finished_spans()
        chat_span = next(s for s in spans if s.name == "chat test")
        events = chat_span.events
        assert len(events) == 1
        assert events[0].name == "gen_ai.choice"
        assert events[0].attributes["gen_ai.completion"] == "hello world"

    def test_thinking_emits_choice_event_with_reasoning(self, span_exporter) -> None:
        al = _make_logger(capture_content=True)
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("chat test") as span:
            al.set_parent_context(trace.set_span_in_context(span))
            al.process_event(ThinkingDeltaEvent(thinking="let me think"))
            al.process_event(ContentBlockStopEvent())
        spans = span_exporter.get_finished_spans()
        chat_span = next(s for s in spans if s.name == "chat test")
        events = chat_span.events
        assert len(events) == 1
        assert events[0].name == "gen_ai.choice"
        assert events[0].attributes["gen_ai.reasoning_content"] == "let me think"

    def test_empty_buffer_no_event(self, span_exporter) -> None:
        al = _make_logger(capture_content=True)
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("chat test"):
            al.process_event(ContentBlockStopEvent())
        spans = span_exporter.get_finished_spans()
        chat_span = next(s for s in spans if s.name == "chat test")
        assert len(chat_span.events) == 0


class TestComplete:
    def test_sets_usage_attributes_on_span(self, span_exporter) -> None:
        al = _make_logger()
        with trace.get_tracer("test").start_as_current_span("agent") as span:
            al.complete(
                input_tokens=100,
                output_tokens=50,
                reasoning_tokens=10,
                span=span,
            )

        attrs = dict(span_exporter.get_finished_spans()[0].attributes)
        assert attrs["gen_ai.usage.input_tokens"] == 100
        assert attrs["gen_ai.usage.output_tokens"] == 50
        assert attrs["gen_ai.usage.reasoning.output_tokens"] == 10
        assert "gen_ai.usage.reasoning_tokens" not in attrs
        assert "gen_ai.response.model" not in attrs

    def test_no_reasoning_attr_when_zero(self, span_exporter) -> None:
        al = _make_logger()
        with trace.get_tracer("test").start_as_current_span("agent") as span:
            al.complete(input_tokens=10, output_tokens=5, span=span)

        attrs = dict(span_exporter.get_finished_spans()[0].attributes)
        assert "gen_ai.usage.reasoning.output_tokens" not in attrs

    def test_complete_preserves_existing_status(self, span_exporter) -> None:
        al = _make_logger()
        with trace.get_tracer("test").start_as_current_span("agent") as span:
            span.set_status(StatusCode.ERROR, "upstream failure")
            al.complete(input_tokens=0, output_tokens=0, span=span)

        finished_span = span_exporter.get_finished_spans()[0]
        assert finished_span.status.status_code == StatusCode.ERROR
        assert finished_span.status.description == "upstream failure"

    def test_no_crash_without_span(self) -> None:
        al = _make_logger()
        al.complete(input_tokens=0, output_tokens=0)


class TestNoJsonEmission:
    def test_no_stdout_json_from_audit_logger(self, capsys: pytest.CaptureFixture[str]) -> None:
        al = _make_logger()
        al.process_event(ToolCallEvent(name="bash", input="ls"))
        al.process_event(ToolResultEvent(output="file.txt"))
        al.complete(input_tokens=100, output_tokens=50)
        out = capsys.readouterr().out
        assert "audit.agent" not in out


class TestContentCapture:
    def test_content_included_when_capture_enabled(self, span_exporter) -> None:
        al = _make_logger(capture_content=True)
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("chat test") as span:
            al.set_parent_context(trace.set_span_in_context(span))
            al.process_event(TextDeltaEvent(text="hello"))
            al.process_event(ContentBlockStopEvent())
        spans = span_exporter.get_finished_spans()
        chat_span = next(s for s in spans if s.name == "chat test")
        assert chat_span.events[0].attributes["gen_ai.completion"] == "hello"

    def test_content_omitted_when_capture_disabled(self, span_exporter) -> None:
        al = _make_logger(capture_content=False)
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("chat test") as span:
            al.set_parent_context(trace.set_span_in_context(span))
            al.process_event(TextDeltaEvent(text="hello"))
            al.process_event(ContentBlockStopEvent())
        spans = span_exporter.get_finished_spans()
        chat_span = next(s for s in spans if s.name == "chat test")
        assert len(chat_span.events) == 1
        assert "gen_ai.completion" not in chat_span.events[0].attributes

    def test_thinking_omitted_when_capture_disabled(self, span_exporter) -> None:
        al = _make_logger(capture_content=False)
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("chat test") as span:
            al.set_parent_context(trace.set_span_in_context(span))
            al.process_event(ThinkingDeltaEvent(thinking="let me think"))
            al.process_event(ContentBlockStopEvent())
        spans = span_exporter.get_finished_spans()
        chat_span = next(s for s in spans if s.name == "chat test")
        assert len(chat_span.events) == 1
        assert "gen_ai.reasoning_content" not in chat_span.events[0].attributes

    def test_tool_io_always_recorded_regardless_of_capture(self, span_exporter) -> None:
        al = _make_logger(capture_content=False)
        al.process_event(ToolCallEvent(name="bash", input="ls -la", call_id="c1"))
        al.process_event(ToolResultEvent(output="done", call_id="c1"))
        spans = span_exporter.get_finished_spans()
        attrs = dict(spans[0].attributes)
        assert json.loads(attrs["gen_ai.tool.call.arguments"]) == {"content": "ls -la"}
        assert json.loads(attrs["gen_ai.tool.call.result"]) == {"content": "done"}
        assert "tool.input" not in attrs
        assert "tool.output" not in attrs


class TestDisabledAudit:
    def test_spans_created_when_disabled(self, span_exporter) -> None:
        al = _make_logger(enabled=False)
        al.process_event(ToolCallEvent(name="bash", input="ls", call_id="c1"))
        assert span_exporter.get_finished_spans() == []

        al.process_event(ToolResultEvent(output="done", call_id="c1"))
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code == StatusCode.OK
