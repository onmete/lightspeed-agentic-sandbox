"""OTEL tracing and logging — provider initialization and traceparent parsing.

The tracer and logger providers share a Resource with ``service.name`` set to
``lightspeed-agentic-sandbox``. Audit-enabled stdout exports complete source
spans as OTLP JSON. With the OTLP endpoint and audit enabled, normalized
``gen_ai.choice`` audit records and generic span events are dual-shipped through
stdlib ``logging`` and ``LoggingHandler``.

Templog log records stamp configured ``agenticrun.uid`` / ``agenticrun.phase``
and ``event`` through the existing stdlib ``logging`` extras/filter path.
Correlation values default to ``LIGHTSPEED_AGENTICRUN_UID`` and
``LIGHTSPEED_AGENTICRUN_STEP``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

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
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

_DEFAULT_SERVICE_NAME = "lightspeed-agentic-sandbox"
_TRACER_NAME = "lightspeed_agentic"
_SCHEMA_URL = "https://opentelemetry.io/schemas/1.41.0"
_ATTR_AGENTICRUN_UID = "agenticrun.uid"
_ATTR_AGENTICRUN_PHASE = "agenticrun.phase"
_logger = logging.getLogger(__name__)
_audit_bridge_logger = logging.getLogger("lightspeed_agentic.audit")


@dataclass
class _OtelState:
    tracer_provider: TracerProvider | None = None
    logger_provider: LoggerProvider | None = None
    logging_handler: LoggingHandler | None = None
    span_events_to_logs_processor: _SpanEventsToLogsProcessor | None = None


_state = _OtelState()


def otel_runtime_enabled() -> bool:
    """Return True when stdout audit or OTLP export should be configured."""
    audit = os.environ.get("LIGHTSPEED_AUDIT_ENABLED", "").strip().lower() == "true"
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    return audit or bool(endpoint)


class OTLPJsonStdoutExporter(SpanExporter):
    """Exports spans as OTLP JSON wire format to stdout (one line per batch)."""

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        request = encode_spans(spans)
        line = json.dumps(MessageToDict(request, preserving_proto_field_name=True))
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def emit_audit_log_records(
    context: Context,
    records: Iterable[tuple[str, Mapping[str, object] | None]],
) -> None:
    """Forward event bodies through the registered stdlib-to-OTLP log bridge."""
    bridge = _state.span_events_to_logs_processor
    if _state.logging_handler is None or bridge is None:
        return
    try:
        token = attach(context)
        try:
            for event_name, attributes in records:
                if event_name == "exception":
                    continue
                _audit_bridge_logger.info(
                    json.dumps(dict(attributes or {}), default=str),
                    extra={"event": event_name, **bridge._attrs},
                )
        finally:
            detach(token)
    except Exception:
        # Never break span export / request path if log bridging fails.
        _logger.exception("failed to forward span events to OTLP logs")


class _SpanEventsToLogsProcessor(SpanProcessor):
    """Forward non-exception span events through stdlib logging."""

    def __init__(self, *, agenticrun_uid: str = "", agenticrun_phase: str = "") -> None:
        self._attrs: dict[str, str] = {}
        if agenticrun_uid:
            self._attrs[_ATTR_AGENTICRUN_UID] = agenticrun_uid
        if agenticrun_phase:
            self._attrs[_ATTR_AGENTICRUN_PHASE] = agenticrun_phase

    def on_end(self, span: ReadableSpan) -> None:
        events = span.events
        if not events:
            return
        emit_audit_log_records(
            _span_context_for_logs(span),
            ((event.name, event.attributes) for event in events),
        )


class _AgenticRunFilter(logging.Filter):
    """Stamp configured AgenticRun correlation on collector log records."""

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
) -> None:
    """Initialize OTEL TracerProvider and LoggerProvider from env.

    ``agenticrun_phase`` defaults to ``LIGHTSPEED_AGENTICRUN_STEP`` and
    ``agenticrun_uid`` defaults to ``LIGHTSPEED_AGENTICRUN_UID`` when omitted.

    Traces:
    - Full OTLP JSON source spans on stdout when ``LIGHTSPEED_AUDIT_ENABLED=true``.
    - OTLP span export when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set.

    Logs:
    - OTLP log export and stdlib dual-shipping when the endpoint is set.
    - Normalized ``gen_ai.choice`` logs and generic span-event forwarding when
      both the endpoint and audit are enabled; choice logs also require a
      recording enclosing agent span.
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

    tracer_provider = TracerProvider(resource=resource)
    _state.tracer_provider = tracer_provider
    if audit:
        tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPJsonStdoutExporter()))
    if endpoint and audit:
        span_events_to_logs = _SpanEventsToLogsProcessor(
            agenticrun_uid=agenticrun_uid,
            agenticrun_phase=agenticrun_phase,
        )
        _state.span_events_to_logs_processor = span_events_to_logs
        tracer_provider.add_span_processor(span_events_to_logs)
    _configure_trace_exporter(tracer_provider, endpoint=endpoint, protocol=protocol)
    trace.set_tracer_provider(tracer_provider)


def _http_signal_endpoint(endpoint: str, signal: str) -> str:
    base = urlsplit(endpoint)
    return base._replace(path=f"{base.path.rstrip('/')}/v1/{signal}").geturl()


def _configure_trace_exporter(provider: TracerProvider, *, endpoint: str, protocol: str) -> None:
    if not endpoint:
        return

    exporter: SpanExporter
    if protocol == "http/protobuf":
        exporter = HttpSpanExporter(endpoint=_http_signal_endpoint(endpoint, "traces"))
    else:
        exporter = GrpcSpanExporter(endpoint=endpoint)

    provider.add_span_processor(BatchSpanProcessor(exporter))


def _configure_log_exporter(provider: LoggerProvider, *, endpoint: str, protocol: str) -> None:
    if not endpoint:
        return

    if protocol == "http/protobuf":
        provider.add_log_record_processor(
            BatchLogRecordProcessor(HttpLogExporter(endpoint=endpoint))
        )
    else:
        provider.add_log_record_processor(
            BatchLogRecordProcessor(GrpcLogExporter(endpoint=endpoint))
        )


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
            _state.span_events_to_logs_processor = None
    if _state.logger_provider:
        try:
            _state.logger_provider.shutdown()
        finally:
            _state.logger_provider = None


def get_tracer() -> trace.Tracer:
    """Get a tracer instance for creating spans."""
    return trace.get_tracer(_TRACER_NAME, schema_url=_SCHEMA_URL)


def parse_traceparent(header: str | None) -> tuple[str | None, Context | None]:
    """Parse W3C traceparent as (trace_id, parent_context).

    Invalid or missing headers return (None, None), signaling a root span.
    """
    if header:
        parts = header.split("-")
        if len(parts) >= 4:
            trace_id_hex = parts[1]
            parent_id_hex = parts[2]
            flags_hex = parts[3]
            if (
                len(trace_id_hex) == 32
                and trace_id_hex != "0" * 32
                and len(parent_id_hex) == 16
                and parent_id_hex != "0" * 16
            ):
                try:
                    trace_id = int(trace_id_hex, 16)
                    parent_id = int(parent_id_hex, 16)
                    flags = int(flags_hex, 16)
                except ValueError:
                    return None, None
                span_ctx = SpanContext(
                    trace_id=trace_id,
                    span_id=parent_id,
                    is_remote=True,
                    trace_flags=TraceFlags(flags),
                )
                ctx = trace.set_span_in_context(NonRecordingSpan(span_ctx))
                return trace_id_hex, ctx
    return None, None
