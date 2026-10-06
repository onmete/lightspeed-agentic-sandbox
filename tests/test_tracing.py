"""Tests for OTEL tracing initialization and traceparent parsing."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from opentelemetry import trace
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
    ExportLogsServiceRequest,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import lightspeed_agentic.tracing as _tracing_mod
from lightspeed_agentic.audit import AuditLogger
from lightspeed_agentic.run_agent import run_agent_query
from lightspeed_agentic.tracing import (
    get_tracer,
    init_tracer,
    otel_runtime_enabled,
    parse_traceparent,
    shutdown_tracer,
)
from lightspeed_agentic.types import (
    ContentBlockStopEvent,
    ProviderEvent,
    ProviderQueryOptions,
    ResultEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)


def _disable_network_exporters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _tracing_mod,
        "_configure_trace_exporter",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        _tracing_mod,
        "_configure_log_exporter",
        lambda *_args, **_kwargs: None,
    )


def _add_span_exporter() -> InMemorySpanExporter:
    assert _tracing_mod._state.tracer_provider is not None
    exporter = InMemorySpanExporter()
    _tracing_mod._state.tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


def _add_log_exporter() -> InMemoryLogRecordExporter:
    assert _tracing_mod._state.logger_provider is not None
    exporter = InMemoryLogRecordExporter()  # type: ignore[no-untyped-call]
    _tracing_mod._state.logger_provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    return exporter


def _production_tracer(monkeypatch: pytest.MonkeyPatch) -> trace.Tracer:
    provider = _tracing_mod._state.tracer_provider
    assert provider is not None
    monkeypatch.setattr(_tracing_mod.trace, "get_tracer_provider", lambda: provider)
    return get_tracer()


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

    def test_none_header_has_no_parent(self) -> None:
        trace_id, ctx = parse_traceparent(None)
        assert trace_id is None
        assert ctx is None

    def test_empty_header_has_no_parent(self) -> None:
        trace_id, ctx = parse_traceparent("")
        assert trace_id is None
        assert ctx is None

    def test_malformed_header_has_no_parent(self) -> None:
        trace_id, ctx = parse_traceparent("not-a-traceparent")
        assert trace_id is None
        assert ctx is None

    def test_invalid_flags_have_no_parent(self) -> None:
        header = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-zz"
        trace_id, ctx = parse_traceparent(header)
        assert trace_id is None
        assert ctx is None

    def test_wrong_field_count_has_no_parent(self) -> None:
        trace_id, ctx = parse_traceparent("00-abc-01")
        assert trace_id is None
        assert ctx is None

    def test_all_zero_trace_id_has_no_parent(self) -> None:
        header = "00-00000000000000000000000000000000-b7ad6b7169203331-01"
        trace_id, ctx = parse_traceparent(header)
        assert trace_id is None
        assert ctx is None

    def test_short_parent_id_has_no_parent(self) -> None:
        header = "00-0af7651916cd43dd8448eb211c80319c-b7ad-01"
        trace_id, ctx = parse_traceparent(header)
        assert trace_id is None
        assert ctx is None


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

    @pytest.mark.parametrize("capture_content", [False, True])
    def test_normalized_choice_logs_keep_legacy_body_and_full_trace(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        capture_content: bool,
    ) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", "true")
        monkeypatch.setenv("LIGHTSPEED_CAPTURE_CONTENT", str(capture_content).lower())
        monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_UID", "configured-uid")
        monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_STEP", "execution")
        _disable_network_exporters(monkeypatch)
        init_tracer()
        span_exporter = _add_span_exporter()
        log_exporter = _add_log_exporter()
        tracer = _production_tracer(monkeypatch)

        audit = AuditLogger(
            phase="event-phase",
            model="requested-model",
            provider="openai",
            enabled=True,
            capture_content=capture_content,
            agenticrun_uid="event-uid",
        )
        source_messages = json.dumps(
            [{"role": "user", "parts": [{"type": "text", "content": "full source input"}]}]
        )
        with tracer.start_as_current_span(
            "invoke_agent lightspeed",
            attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": "lightspeed",
                "gen_ai.provider.name": "openai",
                "gen_ai.input.messages": source_messages,
            },
        ) as agent_span:
            span_context = agent_span.get_span_context()
            audit.set_parent_context(trace.set_span_in_context(agent_span))
            audit.process_event(TextDeltaEvent(text="hello "))
            audit.process_event(TextDeltaEvent(text="world"))
            audit.process_event(ThinkingDeltaEvent(thinking="reasoning"))
            audit.process_event(ContentBlockStopEvent())
            audit.process_event(TextDeltaEvent(text="tool boundary"))
            audit.process_event(ToolCallEvent(name="execute"))
            audit.process_event(TextDeltaEvent(text="pending"))
            audit.process_event(ToolResultEvent(output="raw tool result"))
            audit.process_event(TextDeltaEvent(text=" after tool result"))
            audit.process_event(ResultEvent(text="terminal output"))
            audit.close()
        audit.flush_event_logs()

        source_spans = span_exporter.get_finished_spans()
        assert len(source_spans) == 1
        source_span = source_spans[0]
        assert source_span.events == ()
        assert source_span.attributes["gen_ai.input.messages"] == source_messages

        records = [item.log_record for item in log_exporter.get_finished_logs()]
        assert len(records) == 4
        expected_bodies = (
            [
                {"gen_ai.completion": "hello world"},
                {"gen_ai.reasoning_content": "reasoning"},
                {"gen_ai.completion": "tool boundary"},
                {"gen_ai.completion": "pending after tool result"},
            ]
            if capture_content
            else [{}, {}, {}, {}]
        )
        assert [json.loads(str(record.body)) for record in records] == expected_bodies
        for record in records:
            attributes = record.attributes or {}
            assert attributes["event"] == "gen_ai.choice"
            assert attributes["agenticrun.uid"] == "configured-uid"
            assert attributes["agenticrun.phase"] == "execution"
            assert record.trace_id == span_context.trace_id
            assert record.span_id == span_context.span_id

        stdout_lines = capsys.readouterr().out.splitlines()
        assert len(stdout_lines) == 1
        stdout_spans = [
            span
            for resource in json.loads(stdout_lines[0])["resource_spans"]
            for scope in resource["scope_spans"]
            for span in scope["spans"]
        ]
        assert len(stdout_spans) == 1
        stdout_attributes = {
            item["key"]: item["value"]["string_value"] for item in stdout_spans[0]["attributes"]
        }
        assert stdout_attributes["gen_ai.input.messages"] == source_messages

    @pytest.mark.parametrize(
        ("startup_audit", "current_audit", "expected_choice_log"),
        [(False, True, False), (True, False, True)],
    )
    def test_choice_log_gate_uses_registered_bridge(
        self,
        monkeypatch: pytest.MonkeyPatch,
        startup_audit: bool,
        current_audit: bool,
        expected_choice_log: bool,
    ) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", str(startup_audit).lower())
        monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_UID", "configured-uid")
        monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_STEP", "execution")
        _disable_network_exporters(monkeypatch)
        init_tracer()
        span_exporter = _add_span_exporter()
        log_exporter = _add_log_exporter()
        tracer = _production_tracer(monkeypatch)

        monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", str(current_audit).lower())
        if startup_audit:
            monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

        audit = AuditLogger(
            phase="event-phase",
            model="requested-model",
            provider="openai",
            enabled=True,
            capture_content=True,
            agenticrun_uid="event-uid",
        )
        with tracer.start_as_current_span(
            "invoke_agent lightspeed",
            attributes={"gen_ai.operation.name": "invoke_agent"},
        ) as agent_span:
            span_context = agent_span.get_span_context()
            audit.set_parent_context(trace.set_span_in_context(agent_span))
            audit.process_event(TextDeltaEvent(text="answer"))
            audit.process_event(ResultEvent(text="terminal"))
            audit.close()
        audit.flush_event_logs()

        finished_spans = span_exporter.get_finished_spans()
        assert len(finished_spans) == 1
        records = [
            item.log_record
            for item in log_exporter.get_finished_logs()
            if (item.log_record.attributes or {}).get("event") == "gen_ai.choice"
        ]
        assert bool(records) is expected_choice_log
        if records:
            assert json.loads(str(records[0].body)) == {"gen_ai.completion": "answer"}
            assert records[0].attributes["agenticrun.uid"] == "configured-uid"
            assert records[0].attributes["agenticrun.phase"] == "execution"
            assert records[0].trace_id == span_context.trace_id
            assert records[0].span_id == span_context.span_id

    def test_generic_events_forward_without_exception_duplication(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", "true")
        monkeypatch.setenv("LIGHTSPEED_CAPTURE_CONTENT", "false")
        monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_UID", "uid-1")
        monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_STEP", "execution")
        _disable_network_exporters(monkeypatch)
        init_tracer()
        log_exporter = _add_log_exporter()

        assert _tracing_mod._state.tracer_provider is not None
        tracer = _tracing_mod._state.tracer_provider.get_tracer("operator-events")
        with tracer.start_as_current_span(
            "agenticrun.execute",
            attributes={
                "agenticrun.uid": "span-uid",
                "agenticrun.phase": "verification",
            },
        ) as span:
            span_context = span.get_span_context()
            span.record_exception(ValueError("not duplicated"))
            span.add_event(
                "agenticrun.execution.completed",
                {
                    "result.uid": "result-1",
                    "gen_ai.tool.call.result": '{"secret":"unfiltered"}',
                },
            )
            span.add_event(
                "gen_ai.provider.sdk_event",
                {"gen_ai.provider.response.id": "sdk-event"},
            )

        records = [item.log_record for item in log_exporter.get_finished_logs()]
        assert [record.attributes["event"] for record in records] == [
            "agenticrun.execution.completed",
            "gen_ai.provider.sdk_event",
        ]
        assert json.loads(str(records[0].body)) == {
            "result.uid": "result-1",
            "gen_ai.tool.call.result": '{"secret":"unfiltered"}',
        }
        assert json.loads(str(records[1].body)) == {"gen_ai.provider.response.id": "sdk-event"}
        for record in records:
            attributes = record.attributes or {}
            assert attributes["agenticrun.uid"] == "uid-1"
            assert attributes["agenticrun.phase"] == "execution"
            assert record.trace_id == span_context.trace_id
            assert record.span_id == span_context.span_id

    @pytest.mark.parametrize("failure_kind", ["provider_error", "cancelled"])
    @pytest.mark.asyncio
    async def test_run_agent_flushes_buffered_choice_logs_after_terminal_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        failure_kind: str,
    ) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", "true")
        monkeypatch.setenv("LIGHTSPEED_CAPTURE_CONTENT", "true")
        monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_UID", "run-uid")
        monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_STEP", "execution")
        _disable_network_exporters(monkeypatch)
        init_tracer()
        span_exporter = _add_span_exporter()
        log_exporter = _add_log_exporter()
        _production_tracer(monkeypatch)

        class PartialProvider:
            name = "openai"

            async def query(self, _options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
                yield TextDeltaEvent(text="partial response")
                if failure_kind == "provider_error":
                    raise RuntimeError("provider failed")
                raise asyncio.CancelledError()

        partial_query = run_agent_query(
            PartialProvider(),
            prompt="request",
            system_prompt="instructions",
            output_schema=None,
            context=None,
            skills_dir="/workspace",
            model="test-model",
            max_turns=1,
            timeout_seconds=30,
            audit_enabled=True,
            capture_content=True,
            agenticrun_uid="run-uid",
            step="execution",
        )
        if failure_kind == "provider_error":
            result = await partial_query
            assert result.output["success"] is False
        else:
            with pytest.raises(asyncio.CancelledError):
                await partial_query

        agent_span = next(
            span
            for span in span_exporter.get_finished_spans()
            if span.name == "invoke_agent lightspeed"
        )
        assert "gen_ai.output.messages" not in agent_span.attributes
        if failure_kind == "provider_error":
            assert agent_span.attributes["error.type"] == "RuntimeError"
        else:
            assert agent_span.attributes["error.type"] == "CancelledError"

        records = [
            item.log_record
            for item in log_exporter.get_finished_logs()
            if (item.log_record.attributes or {}).get("event") == "gen_ai.choice"
        ]
        assert len(records) == 1
        record = records[0]
        assert json.loads(str(record.body)) == {"gen_ai.completion": "partial response"}
        assert record.trace_id == agent_span.context.trace_id
        assert record.span_id == agent_span.context.span_id

    def test_audit_disabled_suppresses_stdout_and_span_event_logs(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", "false")
        monkeypatch.delenv("LIGHTSPEED_CAPTURE_CONTENT", raising=False)
        _disable_network_exporters(monkeypatch)
        init_tracer()
        span_exporter = _add_span_exporter()
        log_exporter = _add_log_exporter()

        assert _tracing_mod._state.tracer_provider is not None
        tracer = _tracing_mod._state.tracer_provider.get_tracer("audit-gate")
        attributes = {
            "gen_ai.operation.name": "chat",
            "gen_ai.input.messages": "input",
        }
        with tracer.start_as_current_span("chat model", attributes=attributes):
            pass

        assert capsys.readouterr().out == ""
        assert not log_exporter.get_finished_logs()
        assert dict(span_exporter.get_finished_spans()[0].attributes or {}) == attributes

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

    @pytest.mark.parametrize(
        ("base_suffix", "trace_target", "log_target"),
        [
            ("/", "/v1/traces", "/"),
            (
                "/tenant?token=abc",
                "/tenant/v1/traces?token=abc",
                "/tenant?token=abc",
            ),
        ],
    )
    def test_http_protobuf_keeps_trace_signal_path_and_log_base_endpoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
        base_suffix: str,
        trace_target: str,
        log_target: str,
    ) -> None:
        received_requests: list[tuple[str, bytes]] = []

        class OtlpReceiver(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers["Content-Length"])
                received_requests.append((self.path, self.rfile.read(length)))
                self.send_response(200)
                self.send_header("Content-Type", "application/x-protobuf")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, _format: str, *args: object) -> None:
                pass

        class OtlpHTTPServer(ThreadingHTTPServer):
            daemon_threads = False

        with OtlpHTTPServer(("127.0.0.1", 0), OtlpReceiver) as server:
            server_thread = Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            try:
                monkeypatch.setenv(
                    "OTEL_EXPORTER_OTLP_ENDPOINT",
                    f"http://127.0.0.1:{server.server_port}{base_suffix}",
                )
                monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
                monkeypatch.setenv("LIGHTSPEED_AUDIT_ENABLED", "true")
                monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_UID", "run-uid")
                monkeypatch.setenv("LIGHTSPEED_AGENTICRUN_STEP", "execution")
                monkeypatch.delenv("LIGHTSPEED_CAPTURE_CONTENT", raising=False)
                monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
                monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")

                # Avoid forwarding the global-provider warning as a second OTLP log.
                monkeypatch.setattr(
                    logging.getLogger("opentelemetry.trace"), "level", logging.ERROR
                )

                init_tracer()
                tracer = _production_tracer(monkeypatch)
                audit = AuditLogger(
                    phase="execution",
                    model="test-model",
                    provider="openai",
                    enabled=True,
                    capture_content=True,
                    agenticrun_uid="run-uid",
                )
                with tracer.start_as_current_span(
                    "invoke_agent lightspeed",
                    attributes={
                        "gen_ai.operation.name": "invoke_agent",
                        "gen_ai.agent.name": "lightspeed",
                        "gen_ai.provider.name": "openai",
                    },
                ) as agent_span:
                    agent_context = agent_span.get_span_context()
                    audit.set_parent_context(trace.set_span_in_context(agent_span))
                    tool_span = audit.start_tool(
                        name="execute",
                        call_id="call-1",
                        arguments={"command": "printf hello"},
                    )
                    tool_context = tool_span.get_span_context()
                    audit.end_tool(tool_span, result={"stdout": "hello"})
                    audit.process_event(TextDeltaEvent(text="completed"))
                    audit.process_event(ContentBlockStopEvent())
                    audit.close()
                audit.flush_event_logs()
            finally:
                try:
                    shutdown_tracer()
                finally:
                    try:
                        server.shutdown()
                    finally:
                        server_thread.join()

        assert sorted(path for path, _ in received_requests) == sorted([log_target, trace_target])
        bodies_by_path = {path: body for path, body in received_requests}
        trace_request = ExportTraceServiceRequest.FromString(bodies_by_path[trace_target])
        logs_request = ExportLogsServiceRequest.FromString(bodies_by_path[log_target])

        exported_spans = [
            span
            for resource_spans in trace_request.resource_spans
            for scope_spans in resource_spans.scope_spans
            for span in scope_spans.spans
        ]
        assert len(exported_spans) == 2
        exported_agent = next(
            span for span in exported_spans if span.name == "invoke_agent lightspeed"
        )
        exported_tool = next(span for span in exported_spans if span.name == "execute_tool execute")
        span_attributes = {
            attribute.key: attribute.value.string_value for attribute in exported_tool.attributes
        }
        assert exported_agent.trace_id == agent_context.trace_id.to_bytes(16, "big")
        assert exported_agent.span_id == agent_context.span_id.to_bytes(8, "big")
        assert exported_tool.trace_id == tool_context.trace_id.to_bytes(16, "big")
        assert exported_tool.parent_span_id == exported_agent.span_id
        assert span_attributes["gen_ai.operation.name"] == "execute_tool"
        assert span_attributes["gen_ai.tool.name"] == "execute"
        assert span_attributes["gen_ai.tool.call.id"] == "call-1"
        assert json.loads(span_attributes["gen_ai.tool.call.arguments"]) == {
            "command": "printf hello"
        }
        assert json.loads(span_attributes["gen_ai.tool.call.result"]) == {"stdout": "hello"}

        log_records = [
            record
            for resource_logs in logs_request.resource_logs
            for scope_logs in resource_logs.scope_logs
            for record in scope_logs.log_records
        ]
        assert len(log_records) == 1
        log_record = log_records[0]
        log_attributes = {
            attribute.key: attribute.value.string_value for attribute in log_record.attributes
        }
        assert log_record.trace_id == exported_agent.trace_id
        assert log_record.span_id == exported_agent.span_id
        assert log_attributes["event"] == "gen_ai.choice"
        assert log_attributes["agenticrun.uid"] == "run-uid"
        assert log_attributes["agenticrun.phase"] == "execution"
        assert json.loads(log_record.body.string_value) == {"gen_ai.completion": "completed"}
