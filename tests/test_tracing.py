"""Tests for OTEL tracing initialization and traceparent parsing."""

from __future__ import annotations

import base64
import json
import logging
import re

import pytest
from opentelemetry import trace
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import lightspeed_agentic.tracing as _tracing_mod
from lightspeed_agentic.tracing import (
    get_tracer,
    init_tracer,
    otel_runtime_enabled,
    parse_traceparent,
    shutdown_tracer,
)

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

    @pytest.mark.parametrize(
        "header",
        [
            "ff-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01-extra",
        ],
    )
    def test_invalid_traceparent_format_generates_new_trace_id(self, header: str) -> None:
        trace_id, ctx = parse_traceparent(header)

        assert trace_id != "0af7651916cd43dd8448eb211c80319c"
        assert _TRACE_ID_RE.match(trace_id)
        assert ctx is not None
        span_context = trace.get_current_span(ctx).get_span_context()
        assert span_context.trace_id == int(trace_id, 16)
        assert not span_context.is_remote

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

    @pytest.mark.parametrize("capture_content", [True, False])
    def test_compliance_views_keep_identity_and_only_filter_content(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        capture_content: bool,
    ) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", "true")
        monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_UID", "uid-1")
        init_tracer(agenticrun_phase="execution", capture_content=capture_content)
        logs = InMemoryLogRecordExporter()  # type: ignore[no-untyped-call]
        assert _tracing_mod._state.logger_provider is not None
        _tracing_mod._state.logger_provider.add_log_record_processor(SimpleLogRecordProcessor(logs))
        traces = InMemorySpanExporter()
        assert _tracing_mod._state.tracer_provider is not None
        _tracing_mod._state.tracer_provider.add_span_processor(SimpleSpanProcessor(traces))
        tracer = _tracing_mod._state.tracer_provider.get_tracer("lightspeed_agentic")
        with tracer.start_as_current_span(
            "chat model",
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": "openai",
                "gen_ai.request.model": "model",
                "gen_ai.input.messages": (
                    '[{"role":"user","parts":[{"type":"text","content":"secret"}]}]'
                ),
                "agenticrun.uid": "uid-1",
                "agenticrun.phase": "execution",
            },
        ) as span:
            span.set_attribute(
                "gen_ai.output.messages",
                '[{"role":"assistant","parts":[{"type":"text","content":"answer"}],"finish_reason":"stop"}]',
            )

        product = traces.get_finished_spans()[0]
        assert "secret" in product.attributes["gen_ai.input.messages"]
        assert "answer" in product.attributes["gen_ai.output.messages"]
        output = json.loads(capsys.readouterr().out)
        stdout_span = output["resource_spans"][0]["scope_spans"][0]["spans"][0]
        stdout_attrs = {attr["key"]: attr["value"] for attr in stdout_span["attributes"]}
        assert (
            stdout_span["trace_id"]
            == base64.b64encode(product.context.trace_id.to_bytes(16, "big")).decode()
        )
        assert (
            stdout_span["span_id"]
            == base64.b64encode(product.context.span_id.to_bytes(8, "big")).decode()
        )
        assert stdout_span["start_time_unix_nano"] == str(product.start_time)
        assert stdout_span["end_time_unix_nano"] == str(product.end_time)
        assert "gen_ai.request.model" in stdout_attrs
        assert ("gen_ai.input.messages" in stdout_attrs) is capture_content
        assert ("gen_ai.output.messages" in stdout_attrs) is capture_content
        matching = [
            rec.log_record
            for rec in logs.get_finished_logs()
            if (rec.log_record.attributes or {}).get("event") == "chat"
        ]
        assert len(matching) == 1
        record = matching[0]
        assert record.trace_id == product.context.trace_id
        assert record.span_id == product.context.span_id
        assert record.attributes["agenticrun.uid"] == "uid-1"
        assert record.attributes["agenticrun.phase"] == "execution"
        body = json.loads(str(record.body))
        assert body["gen_ai.request.model"] == "model"
        assert ("gen_ai.input.messages" in body) is capture_content
        assert ("gen_ai.output.messages" in body) is capture_content

    def test_only_standard_operation_spans_become_compliance_logs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", "true")
        init_tracer()
        logs = InMemoryLogRecordExporter()  # type: ignore[no-untyped-call]
        assert _tracing_mod._state.logger_provider is not None
        _tracing_mod._state.logger_provider.add_log_record_processor(SimpleLogRecordProcessor(logs))
        assert _tracing_mod._state.tracer_provider is not None
        tracer = _tracing_mod._state.tracer_provider.get_tracer("lightspeed_agentic")
        for operation in ("invoke_agent", "chat", "generate_content", "execute_tool", "other"):
            with tracer.start_as_current_span(
                operation, attributes={"gen_ai.operation.name": operation}
            ) as span:
                span.add_event("exception", {"exception.message": "private stack trace"})
        with _tracing_mod._state.tracer_provider.get_tracer(
            "gcp.vertex.agent"
        ).start_as_current_span("native chat", attributes={"gen_ai.operation.name": "chat"}):
            pass
        matching = [
            rec.log_record
            for rec in logs.get_finished_logs()
            if (rec.log_record.attributes or {}).get("event")
            in ("invoke_agent", "chat", "generate_content", "execute_tool", "other")
        ]
        assert {rec.attributes["event"] for rec in matching} == {
            "invoke_agent",
            "chat",
            "generate_content",
            "execute_tool",
        }
        assert all("private stack trace" not in str(rec.body) for rec in matching)

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

    def test_compliance_bridge_disabled_without_audit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", "false")
        init_tracer()
        logs = InMemoryLogRecordExporter()  # type: ignore[no-untyped-call]
        assert _tracing_mod._state.logger_provider is not None
        _tracing_mod._state.logger_provider.add_log_record_processor(SimpleLogRecordProcessor(logs))
        assert _tracing_mod._state.tracer_provider is not None
        tracer = _tracing_mod._state.tracer_provider.get_tracer("lightspeed_agentic")
        with tracer.start_as_current_span(
            "chat m",
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.input.messages": '[{"role":"user"}]',
            },
        ):
            pass
        assert not [
            rec
            for rec in logs.get_finished_logs()
            if (rec.log_record.attributes or {}).get("event") == "chat"
        ]

    def test_adk_logs_scope_excluded_from_otlp_but_developer_logs_export(
        self,
    ) -> None:
        from opentelemetry.sdk._logs import LoggerProvider

        exporter = InMemoryLogRecordExporter()  # type: ignore[no-untyped-call]
        provider = LoggerProvider()
        processor = _tracing_mod._ADKFilteredBatchLogRecordProcessor(exporter)
        provider.add_log_record_processor(processor)
        provider.get_logger("gcp.vertex.agent").emit(body="SDK-generated content")
        provider.get_logger("lightspeed_agentic").emit(body="developer diagnostic")
        provider.force_flush()
        records = exporter.get_finished_logs()
        assert [record.log_record.body for record in records] == ["developer diagnostic"]
        provider.shutdown()
