"""Tests for OTEL tracing initialization and traceparent parsing."""

from __future__ import annotations

import json
import logging
import re

import pytest
from opentelemetry import trace
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)

import lightspeed_agentic.tracing as _tracing_mod
from lightspeed_agentic.run_agent import run_agent_query
from lightspeed_agentic.tracing import (
    get_tracer,
    init_tracer,
    otel_runtime_enabled,
    parse_traceparent,
    shutdown_tracer,
)
from lightspeed_agentic.types import ResultEvent, ToolCallEvent, ToolResultEvent

from .conftest import MockProvider

_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")


@pytest.fixture(autouse=True)
def _reset_tracer_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shut down providers around each test so Batch exporters do not leak."""
    # Keep shutdown fast when tests point OTLP at an unreachable localhost.
    monkeypatch.setenv("OTEL_BSP_EXPORT_TIMEOUT", "1000")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "1")
    shutdown_tracer()
    yield
    shutdown_tracer()


class TestParseTraceparent:
    def test_valid_traceparent(self) -> None:
        header = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        trace_id, ctx = parse_traceparent(header)
        assert trace_id == "0af7651916cd43dd8448eb211c80319c"
        assert ctx is not None

    def test_none_header_generates_trace_id(self) -> None:
        trace_id, ctx = parse_traceparent(None)
        assert _TRACE_ID_RE.match(trace_id)
        assert ctx is not None

    def test_empty_header_generates_trace_id(self) -> None:
        trace_id, ctx = parse_traceparent("")
        assert _TRACE_ID_RE.match(trace_id)
        assert ctx is not None

    def test_malformed_header_generates_trace_id(self) -> None:
        trace_id, ctx = parse_traceparent("not-a-traceparent")
        assert _TRACE_ID_RE.match(trace_id)
        assert ctx is not None

    def test_wrong_field_count_generates_trace_id(self) -> None:
        trace_id, ctx = parse_traceparent("00-abc-01")
        assert _TRACE_ID_RE.match(trace_id)
        assert ctx is not None

    def test_all_zero_trace_id_generates_new(self) -> None:
        header = "00-00000000000000000000000000000000-b7ad6b7169203331-01"
        trace_id, ctx = parse_traceparent(header)
        assert trace_id != "00000000000000000000000000000000"
        assert _TRACE_ID_RE.match(trace_id)
        assert ctx is not None

    def test_short_parent_id_generates_new(self) -> None:
        header = "00-0af7651916cd43dd8448eb211c80319c-b7ad-01"
        trace_id, ctx = parse_traceparent(header)
        assert trace_id != "0af7651916cd43dd8448eb211c80319c"
        assert _TRACE_ID_RE.match(trace_id)
        assert ctx is not None

    def test_generated_ids_are_unique(self) -> None:
        id1, _ = parse_traceparent(None)
        id2, _ = parse_traceparent(None)
        assert id1 != id2


class TestOtelRuntimeEnabled:
    def test_false_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LIGHTSPEED_AUDIT_ENABLED", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        assert otel_runtime_enabled() is False

    def test_true_when_audit_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", "true")
        assert otel_runtime_enabled() is True

    def test_true_when_endpoint_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LIGHTSPEED_AUDIT_ENABLED", raising=False)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        assert otel_runtime_enabled() is True


class TestInitTracer:
    def test_init_without_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        init_tracer()
        tracer = get_tracer()
        assert tracer is not None

    def test_init_with_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        init_tracer()
        tracer = get_tracer()
        assert tracer is not None

    def test_init_with_audit_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", "true")
        init_tracer()
        tracer = get_tracer()
        assert tracer is not None

    def test_get_tracer_returns_named_tracer(self) -> None:
        tracer = get_tracer()
        assert isinstance(tracer, trace.Tracer)

    def test_shutdown_tracer_flushes_without_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        init_tracer()
        shutdown_tracer()

    def test_double_init_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        init_tracer()
        with pytest.raises(RuntimeError, match="already initialized"):
            init_tracer()

    def test_init_after_shutdown_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        init_tracer()
        shutdown_tracer()
        init_tracer()
        assert get_tracer() is not None

    def test_shared_resource_excludes_agenticrun_attrs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_UID", "uid-123")
        init_tracer(agenticrun_phase="execution")
        assert _tracing_mod._state.logger_provider is not None
        assert _tracing_mod._state.tracer_provider is not None
        assert (
            _tracing_mod._state.logger_provider.resource
            is _tracing_mod._state.tracer_provider.resource
        )
        attrs = _tracing_mod._state.logger_provider.resource.attributes
        assert "agenticrun.uid" not in attrs
        assert "agenticrun.phase" not in attrs
        assert attrs["service.name"] == "lightspeed-agentic-sandbox"

    def test_log_filter_does_not_invent_missing_phase(self) -> None:
        record = logging.LogRecord("test", logging.INFO, __file__, 1, "message", (), None)
        stamp = _tracing_mod._AgenticRunFilter(agenticrun_uid="run-uid")

        assert stamp.filter(record) is True
        assert record.__dict__["agenticrun.uid"] == "run-uid"
        assert "agenticrun.phase" not in record.__dict__

    def test_logging_handler_attached_when_endpoint_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from opentelemetry.sdk._logs import LoggingHandler

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        init_tracer()
        assert _tracing_mod._state.logging_handler is not None
        assert isinstance(_tracing_mod._state.logging_handler, LoggingHandler)
        assert _tracing_mod._state.logging_handler in logging.getLogger().handlers

    def test_no_logging_handler_without_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        init_tracer()
        assert _tracing_mod._state.logging_handler is None

    def test_span_events_forwarded_to_logs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from opentelemetry.sdk._logs.export import (
            InMemoryLogRecordExporter,
            SimpleLogRecordProcessor,
        )

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", "true")
        monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_UID", "uid-1")
        init_tracer(agenticrun_phase="execution")

        exporter = InMemoryLogRecordExporter()  # type: ignore[no-untyped-call]
        assert _tracing_mod._state.logger_provider is not None
        _tracing_mod._state.logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(exporter)
        )

        assert _tracing_mod._state.tracer_provider is not None
        tracer = _tracing_mod._state.tracer_provider.get_tracer("test")
        with tracer.start_as_current_span("chat") as span:
            span.add_event("gen_ai.choice", {"gen_ai.completion": "hello"})

        records = exporter.get_finished_logs()
        assert len(records) >= 1
        matching = [
            r for r in records if (r.log_record.attributes or {}).get("event") == "gen_ai.choice"
        ]
        assert matching
        rec = matching[0].log_record
        attrs = rec.attributes or {}
        # Collector postgresexporter reads these record attrs (not Resource).
        # Stamped via logging extra → LoggingHandler (not direct OTel emit).
        assert attrs.get("agenticrun.uid") == "uid-1"
        assert attrs.get("agenticrun.phase") == "execution"
        assert attrs.get("event") == "gen_ai.choice"
        assert "hello" in str(rec.body)
        assert rec.trace_id != 0

    def test_empty_choice_body_when_no_event_attrs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Capture-off style events (no content attrs) still forward with body {}."""
        from opentelemetry.sdk._logs.export import (
            InMemoryLogRecordExporter,
            SimpleLogRecordProcessor,
        )

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", "true")
        monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_UID", "uid-1")
        init_tracer(agenticrun_phase="execution")

        exporter = InMemoryLogRecordExporter()  # type: ignore[no-untyped-call]
        assert _tracing_mod._state.logger_provider is not None
        _tracing_mod._state.logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(exporter)
        )

        assert _tracing_mod._state.tracer_provider is not None
        tracer = _tracing_mod._state.tracer_provider.get_tracer("test")
        with tracer.start_as_current_span("chat") as span:
            span.add_event("gen_ai.choice", {})

        matching = [
            r
            for r in exporter.get_finished_logs()
            if (r.log_record.attributes or {}).get("event") == "gen_ai.choice"
        ]
        assert matching
        assert str(matching[0].log_record.body) in ("{}", "")

    def test_exception_span_events_not_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from opentelemetry.sdk._logs.export import (
            InMemoryLogRecordExporter,
            SimpleLogRecordProcessor,
        )

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", "true")
        monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_UID", "uid-1")
        init_tracer(agenticrun_phase="execution")

        exporter = InMemoryLogRecordExporter()  # type: ignore[no-untyped-call]
        assert _tracing_mod._state.logger_provider is not None
        _tracing_mod._state.logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(exporter)
        )

        assert _tracing_mod._state.tracer_provider is not None
        tracer = _tracing_mod._state.tracer_provider.get_tracer("test")
        with tracer.start_as_current_span("chat") as span:
            span.add_event(
                "exception",
                {
                    "exception.type": "ValueError",
                    "exception.message": "boom",
                    "exception.stacktrace": "traceback...",
                },
            )
            span.add_event("gen_ai.choice", {"gen_ai.completion": "ok"})

        events = [
            (r.log_record.attributes or {}).get("event") for r in exporter.get_finished_logs()
        ]
        assert "exception" not in events
        assert "gen_ai.choice" in events

    def test_warns_when_agenticrun_env_unresolved(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", "true")
        monkeypatch.delenv("LIGHTSPEED_AGENTICRUN_UID", raising=False)
        monkeypatch.delenv("LIGHTSPEED_AGENTICRUN_STEP", raising=False)
        with caplog.at_level(logging.WARNING, logger="lightspeed_agentic.tracing"):
            init_tracer(agenticrun_phase="execution")
        assert any(
            "cannot resolve env" in r.message and "LIGHTSPEED_AGENTICRUN_UID" in r.message
            for r in caplog.records
        )
        assert not any("LIGHTSPEED_AGENTICRUN_STEP" in r.message for r in caplog.records)

    def test_span_events_not_forwarded_when_audit_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from opentelemetry.sdk._logs.export import (
            InMemoryLogRecordExporter,
            SimpleLogRecordProcessor,
        )

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", "false")
        monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_UID", "uid-1")
        init_tracer()

        exporter = InMemoryLogRecordExporter()  # type: ignore[no-untyped-call]
        assert _tracing_mod._state.logger_provider is not None
        _tracing_mod._state.logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(exporter)
        )

        assert _tracing_mod._state.tracer_provider is not None
        tracer = _tracing_mod._state.tracer_provider.get_tracer("test")
        with tracer.start_as_current_span("chat") as span:
            span.add_event("gen_ai.choice", {"gen_ai.completion": "hello"})

        matching = [
            r
            for r in exporter.get_finished_logs()
            if (r.log_record.attributes or {}).get("event") == "gen_ai.choice"
        ]
        assert matching == []


@pytest.mark.asyncio
async def test_json_span_attributes_survive_otlp_protobuf_round_trip(span_exporter) -> None:
    text = "left \ud800 café 🌍 middle \udfff right"
    expected_tool_payload = {"text": text}
    tool_payload = json.dumps(expected_tool_payload, ensure_ascii=True, separators=(",", ":"))

    await run_agent_query(
        MockProvider(
            events=[
                ToolCallEvent(name="lookup", input=tool_payload, call_id="call-1"),
                ToolResultEvent(output=tool_payload, call_id="call-1"),
                ResultEvent(text='{"success": true, "summary": "done"}'),
            ]
        ),
        prompt=text,
        system_prompt=text,
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )

    request = _tracing_mod.encode_spans(span_exporter.get_finished_spans())
    round_tripped_request = ExportTraceServiceRequest()
    round_tripped_request.ParseFromString(request.SerializeToString())

    attributes_by_span = {
        span.name: {attribute.key: attribute.value.string_value for attribute in span.attributes}
        for resource_span in round_tripped_request.resource_spans
        for scope_span in resource_span.scope_spans
        for span in scope_span.spans
    }
    invocation_attributes = attributes_by_span["invoke_agent"]
    tool_attributes = attributes_by_span["execute_tool lookup"]
    invocation_json = invocation_attributes["gen_ai.input.messages"]
    tool_arguments_json = tool_attributes["gen_ai.tool.call.arguments"]
    tool_result_json = tool_attributes["gen_ai.tool.call.result"]

    assert json.loads(invocation_json) == [
        {"role": "user", "parts": [{"type": "text", "content": text}]}
    ]
    assert json.loads(tool_arguments_json) == expected_tool_payload
    assert json.loads(tool_result_json) == expected_tool_payload
    for value in (invocation_json, tool_arguments_json, tool_result_json):
        assert r"\ud800" in value
        assert r"\udfff" in value
        assert "café 🌍" in value
