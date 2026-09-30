"""OTEL traces and logs with independently gated compliance projections.

Product GenAI spans are exported unchanged to OTLP traces. Audit stdout and
templog are derived views of completed operation spans, not a second product
transcript or additional span events. Developer logs retain LoggingHandler.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from google.protobuf.json_format import MessageToDict  # type: ignore[import-untyped]
from opentelemetry import _logs, trace
from opentelemetry.context import Context, attach, detach
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter as GrpcLogExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GrpcSpanExporter,
)
from opentelemetry.exporter.otlp.proto.http._log_exporter import (
    OTLPLogExporter as HttpLogExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HttpSpanExporter,
)
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler, ReadWriteLogRecord
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, LogRecordExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

_DEFAULT_SERVICE_NAME = "lightspeed-agentic-sandbox"
_TRACER_NAME = "lightspeed_agentic"
_ATTR_AGENTICRUN_UID = "agenticrun.uid"
_ATTR_AGENTICRUN_PHASE = "agenticrun.phase"
_logger = logging.getLogger(__name__)
_TRACE_CONTEXT_PROPAGATOR = TraceContextTextMapPropagator()
_CONTENT_KEYS = frozenset(
    {
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.system_instructions",
        "gen_ai.tool.definitions",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
    }
)
_AUDIT_OPERATIONS = frozenset({"invoke_agent", "chat", "generate_content", "execute_tool"})
_ADK_LOG_SCOPE = "gcp.vertex.agent"
_audit_bridge_logger = logging.getLogger("lightspeed_agentic.audit")


@dataclass
class _OtelState:
    tracer_provider: TracerProvider | None = None
    logger_provider: LoggerProvider | None = None
    logging_handler: LoggingHandler | None = None


_state = _OtelState()


def otel_runtime_enabled() -> bool:
    """Return True when stdout audit or OTLP export should be configured."""
    audit = os.environ.get("LIGHTSPEED_AUDIT_ENABLED", "").strip().lower() == "true"
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    return audit or bool(endpoint)


class OTLPJsonStdoutExporter(SpanExporter):
    """Export a cloned OTLP-JSON compliance view; never mutate product spans."""

    def __init__(self, *, capture_content: bool = True) -> None:
        self._capture_content = capture_content

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        pb = encode_spans(spans)
        if not self._capture_content:
            for resource_spans in pb.resource_spans:
                for scope_spans in resource_spans.scope_spans:
                    for span in scope_spans.spans:
                        for index in reversed(range(len(span.attributes))):
                            if span.attributes[index].key in _CONTENT_KEYS:
                                del span.attributes[index]
        line = json.dumps(MessageToDict(pb, preserving_proto_field_name=True))
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


class _GenAISpansToLogsProcessor(SpanProcessor):
    """Emit one audit-gated templog projection for each GenAI operation span."""

    def __init__(
        self,
        *,
        agenticrun_uid: str = "",
        agenticrun_phase: str = "",
        capture_content: bool = True,
    ) -> None:
        self._capture_content = capture_content
        self._attrs: dict[str, str] = {}
        if agenticrun_uid:
            self._attrs[_ATTR_AGENTICRUN_UID] = agenticrun_uid
        if agenticrun_phase:
            self._attrs[_ATTR_AGENTICRUN_PHASE] = agenticrun_phase

    def on_end(self, span: ReadableSpan) -> None:
        # SDK-native spans are operational Actions, not a second audit copy.
        scope = span.instrumentation_scope
        if scope is None or scope.name != _TRACER_NAME:
            return
        attrs = span.attributes or {}
        operation = attrs.get("gen_ai.operation.name")
        if operation not in _AUDIT_OPERATIONS:
            return
        try:
            content = {
                key: value
                for key, value in attrs.items()
                if key not in _CONTENT_KEYS or self._capture_content
            }
            # Log bodies carry span metadata (including status/timing), while
            # trace/span IDs come from the attached completed-span context.
            content.update(
                {
                    "span.name": span.name,
                    "span.status": span.status.status_code.name,
                }
            )
            start_time = span.start_time
            end_time = span.end_time
            if start_time is not None:
                content["span.start_time"] = start_time
            if end_time is not None:
                content["span.end_time"] = end_time
            token = attach(_span_context_for_logs(span))
            try:
                _audit_bridge_logger.info(
                    json.dumps(content, ensure_ascii=False, default=str),
                    extra={"event": operation, **self._attrs},
                )
            finally:
                detach(token)
        except Exception:
            _logger.exception("failed to forward GenAI span to OTLP logs")


class _ADKFilteredBatchLogRecordProcessor(BatchLogRecordProcessor):
    """Keep SDK-native GenAI Logs API events out of the product OTLP endpoint."""

    def on_emit(self, log_record: ReadWriteLogRecord) -> None:
        scope = log_record.instrumentation_scope
        if scope is not None and scope.name == _ADK_LOG_SCOPE:
            return
        super().on_emit(log_record)


class _AgenticRunFilter(logging.Filter):
    """Stamp agenticrun.uid and agenticrun.phase on every log record.

    Attached to the ``LoggingHandler`` so all stdlib log records forwarded
    to OTLP carry the run identity attributes the collector needs.
    """

    def __init__(self, *, agenticrun_uid: str = "", agenticrun_phase: str = "") -> None:
        super().__init__()
        self._uid = agenticrun_uid
        self._phase = agenticrun_phase

    def filter(self, record: logging.LogRecord) -> bool:
        if self._uid:
            setattr(record, _ATTR_AGENTICRUN_UID, self._uid)
        if self._phase:
            setattr(record, _ATTR_AGENTICRUN_PHASE, self._phase)
        return True


def _span_context_for_logs(span: ReadableSpan) -> Context:
    sc = span.get_span_context()
    if sc is None or not sc.is_valid:
        return Context()
    return trace.set_span_in_context(
        NonRecordingSpan(
            SpanContext(
                trace_id=sc.trace_id,
                span_id=sc.span_id,
                is_remote=sc.is_remote,
                trace_flags=sc.trace_flags,
                trace_state=sc.trace_state,
            )
        )
    )


def init_tracer(
    *,
    agenticrun_uid: str | None = None,
    agenticrun_phase: str | None = None,
    capture_content: bool = True,
) -> None:
    """Configure full-fidelity product traces and independent gated audit views.

    ``capture_content`` applies only to stdout OTLP-JSON and templog
    projections. Product spans and developer LoggingHandler records are not
    modified by the compliance content policy.
    """
    if _state.tracer_provider is not None or _state.logger_provider is not None:
        raise RuntimeError("OTEL providers already initialized; call shutdown_tracer() first")

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    protocol = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc").strip().lower() or "grpc"
    if protocol not in ("grpc", "http/protobuf"):
        _logger.warning("unsupported OTEL_EXPORTER_OTLP_PROTOCOL=%r, defaulting to grpc", protocol)
        protocol = "grpc"

    resource = Resource.create().merge(Resource({SERVICE_NAME: _DEFAULT_SERVICE_NAME}))
    if agenticrun_uid is None:
        agenticrun_uid = os.environ.get("LIGHTSPEED_AGENTICRUN_UID", "").strip()
    if agenticrun_phase is None:
        agenticrun_phase = os.environ.get("LIGHTSPEED_AGENTICRUN_STEP", "").strip()
    audit = os.environ.get("LIGHTSPEED_AUDIT_ENABLED", "").strip().lower() == "true"

    if endpoint and audit:
        missing = [
            name
            for name, value in (
                ("LIGHTSPEED_AGENTICRUN_UID", agenticrun_uid),
                ("LIGHTSPEED_AGENTICRUN_STEP", agenticrun_phase),
            )
            if not value
        ]
        if missing:
            _logger.warning(
                "OTLP audit/templog enabled but cannot resolve env %s; "
                "bridged log records will lack those attributes and the "
                "collector will skip records missing agenticrun.uid",
                ", ".join(missing),
            )

    _state.logger_provider = LoggerProvider(resource=resource)
    _configure_log_exporter(_state.logger_provider, endpoint=endpoint, protocol=protocol)
    _logs.set_logger_provider(_state.logger_provider)

    if endpoint:
        _state.logging_handler = LoggingHandler(logger_provider=_state.logger_provider)
        # Stamp agenticrun.uid / agenticrun.phase on every log record
        # so the collector postgresexporter can index them.
        stamp = _AgenticRunFilter(agenticrun_uid=agenticrun_uid, agenticrun_phase=agenticrun_phase)
        _state.logging_handler.addFilter(stamp)
        root = logging.getLogger()
        root.addHandler(_state.logging_handler)
        # LoggingHandler only sees records that pass the root effective level.
        # App startup uses INFO; pytest often leaves root at WARNING.
        if root.getEffectiveLevel() > logging.INFO:
            root.setLevel(logging.INFO)
        _audit_bridge_logger.setLevel(logging.INFO)

    _state.tracer_provider = TracerProvider(resource=resource)
    if audit:
        _state.tracer_provider.add_span_processor(
            SimpleSpanProcessor(OTLPJsonStdoutExporter(capture_content=capture_content))
        )
    if endpoint and audit:
        _state.tracer_provider.add_span_processor(
            _GenAISpansToLogsProcessor(
                agenticrun_uid=agenticrun_uid,
                agenticrun_phase=agenticrun_phase,
                capture_content=capture_content,
            )
        )
    _configure_trace_exporter(_state.tracer_provider, endpoint=endpoint, protocol=protocol)
    trace.set_tracer_provider(_state.tracer_provider)


def _configure_trace_exporter(provider: TracerProvider, *, endpoint: str, protocol: str) -> None:
    if not endpoint:
        return

    exporter: SpanExporter
    if protocol == "http/protobuf":
        exporter = HttpSpanExporter(endpoint=endpoint)
    else:
        exporter = GrpcSpanExporter(endpoint=endpoint)

    provider.add_span_processor(BatchSpanProcessor(exporter))


def _configure_log_exporter(provider: LoggerProvider, *, endpoint: str, protocol: str) -> None:
    if not endpoint:
        return

    exporter: LogRecordExporter
    if protocol == "http/protobuf":
        exporter = HttpLogExporter(endpoint=endpoint)
    else:
        exporter = GrpcLogExporter(endpoint=endpoint)
    provider.add_log_record_processor(_ADKFilteredBatchLogRecordProcessor(exporter))


def shutdown_tracer() -> None:
    """Shutdown tracer and logger providers, flushing pending exports."""
    if _state.logging_handler is not None:
        logging.getLogger().removeHandler(_state.logging_handler)
        _state.logging_handler = None
    if _state.tracer_provider:
        try:
            _state.tracer_provider.shutdown()
        finally:
            _state.tracer_provider = None
    if _state.logger_provider:
        try:
            _state.logger_provider.shutdown()
        finally:
            _state.logger_provider = None


def get_tracer() -> trace.Tracer:
    """Get a tracer instance for creating spans."""
    return trace.get_tracer(_TRACER_NAME)


def parse_traceparent(header: str | None) -> tuple[str, Context | None]:
    """Parse W3C traceparent with the standard propagator; generate a root if invalid."""
    if header:
        context = _TRACE_CONTEXT_PROPAGATOR.extract({"traceparent": header})
        span_context = trace.get_current_span(context).get_span_context()
        if span_context.is_valid:
            return f"{span_context.trace_id:032x}", context
    return _generate_trace_id()


def _generate_trace_id() -> tuple[str, Context]:
    """Generate a new trace ID and root context."""
    trace_id_hex = secrets.token_hex(16)
    span_id_hex = secrets.token_hex(8)
    span_ctx = SpanContext(
        trace_id=int(trace_id_hex, 16),
        span_id=int(span_id_hex, 16),
        is_remote=False,
        trace_flags=TraceFlags(1),
    )
    ctx = trace.set_span_in_context(NonRecordingSpan(span_ctx))
    return trace_id_hex, ctx
