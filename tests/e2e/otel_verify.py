"""Verify sandbox batch runs exported OTLP traces and audit logs to the e2e collector."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import Literal

from kubernetes.client import ApiException, CoreV1Api  # type: ignore[import-untyped]

from tests.e2e.suite_setup import DEFAULT_OTEL_DEPLOYMENT

OTEL_COLLECTOR_LABEL = "app=lightspeed-otel-collector"
DEFAULT_POLL_TIMEOUT_SECONDS = 90.0
DEFAULT_POLL_INTERVAL_SECONDS = 3.0
DEFAULT_LOG_TAIL_LINES = 10000
TOOL_RESULT_INSPECTION_TIMEOUT_SECONDS = 180.0
_TRACE_ID_RE = re.compile(r"\bTrace ID\s*:\s*(\S+)")


def fetch_otel_collector_logs(
    core_api: CoreV1Api,
    namespace: str,
    *,
    tail_lines: int | None = DEFAULT_LOG_TAIL_LINES,
) -> str:
    """Return recent stdout from OTEL collector pod(s) (debug exporter output)."""
    pods = core_api.list_namespaced_pod(
        namespace=namespace,
        label_selector=OTEL_COLLECTOR_LABEL,
    )
    if not pods.items:
        msg = f"no OTEL collector pods in {namespace} (selector {OTEL_COLLECTOR_LABEL})"
        raise RuntimeError(msg)

    chunks: list[str] = []
    for pod in pods.items:
        pod_name = pod.metadata.name
        pod_uid = pod.metadata.uid or "unknown"
        try:
            response = core_api.read_namespaced_pod_log(
                name=pod_name,
                namespace=namespace,
                tail_lines=tail_lines,
                timestamps=False,
                _preload_content=False,
            )
        except ApiException as exc:
            raise RuntimeError(f"read logs for pod/{pod_name}: {exc.reason}") from exc
        try:
            chunk = response.data.decode("utf-8")
        finally:
            response.release_conn()
        chunks.append(f"=== collector pod/{pod_name} uid={pod_uid} ===\n{chunk}")
    return "\n".join(chunks)


_DEBUG_RESOURCE_HEADERS = frozenset({"ResourceSpans", "ResourceLog"})
_DEBUG_SCOPE_HEADERS = frozenset({"ScopeSpans", "ScopeLogs"})
_DEBUG_RECORD_HEADERS = frozenset({"Span", "LogRecord"})
_DEBUG_HEADER_LOGGER_PREFIX = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z\tinfo\t"
)
_SANDBOX_SERVICE_NAME = "lightspeed-agentic-sandbox"


def logs_contain_tool_result_inspection_for_run(
    logs: str,
    run_uid: str,
    *,
    expected_outcome: str = "benign",
) -> bool:
    """True when a matching inspection span has the expected outcome."""
    blocks = logs.split("Span #")[1:]
    run_trace_ids = {
        match.group(1)
        for block in blocks
        if f"agenticrun.uid: Str({run_uid})" in block
        if (match := _TRACE_ID_RE.search(block)) is not None
    }
    return any(
        "tool_result.inspection" in block
        and f"inspection.outcome: Str({expected_outcome})" in block
        and (match := _TRACE_ID_RE.search(block)) is not None
        and match.group(1) in run_trace_ids
        for block in blocks
    )


def _debug_header(line: str) -> str | None:
    header_line = _DEBUG_HEADER_LOGGER_PREFIX.sub("", line, count=1).strip()
    header, separator, index = header_line.partition(" #")
    if (
        separator
        and index.isdecimal()
        and header in (_DEBUG_RESOURCE_HEADERS | _DEBUG_SCOPE_HEADERS | _DEBUG_RECORD_HEADERS)
    ):
        return header
    return None


def _debug_records(logs: str, *, signal: Literal["traces", "logs"]) -> list[tuple[str, str, str]]:
    if signal == "traces":
        resource_header, scope_header, record_header = "ResourceSpans", "ScopeSpans", "Span"
    else:
        resource_header, scope_header, record_header = "ResourceLog", "ScopeLogs", "LogRecord"

    records: list[tuple[str, str, str]] = []
    resource: list[str] = []
    scope: list[str] = []
    record: list[str] | None = None

    def save_record() -> None:
        nonlocal record
        if record is not None:
            records.append(("\n".join(resource), "\n".join(scope), "\n".join(record)))
        record = None

    for line in logs.splitlines():
        header = _debug_header(line)
        if header in _DEBUG_RESOURCE_HEADERS:
            save_record()
            resource = [line] if header == resource_header else []
            scope = []
        elif header in _DEBUG_SCOPE_HEADERS:
            save_record()
            scope = [line] if header == scope_header and resource else []
        elif header in _DEBUG_RECORD_HEADERS:
            save_record()
            if header == record_header and scope:
                record = [line]
            else:
                scope = []
        elif record is not None:
            record.append(line)
        elif scope:
            scope.append(line)
        elif resource:
            resource.append(line)

    save_record()
    return records


def _debug_attributes(block: str) -> dict[str, str]:
    attributes: dict[str, str] = {}
    in_attributes = False
    for line in block.splitlines():
        value = line.strip()
        if value in {"Attributes:", "Resource attributes:"}:
            in_attributes = True
            continue
        if not in_attributes or not value:
            continue
        if not value.startswith("->"):
            break
        key, separator, typed_value = value[2:].strip().partition(":")
        typed_value = typed_value.strip()
        if separator and typed_value.startswith("Str(") and typed_value.endswith(")"):
            attributes[key.strip()] = typed_value[4:-1]
    return attributes


def _debug_field(block: str, name: str) -> str:
    for line in block.splitlines():
        key, separator, value = line.strip().partition(":")
        if separator and key.strip() == name:
            return value.strip()
    return ""


def _nonzero_hex_id(value: str, width: int) -> bool:
    return (
        len(value) == width
        and value.lower() != "0" * width
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _genai_spans_for_run(
    logs: str,
    run_uid: str,
    *,
    expected_operation: str,
    expected_provider: str,
) -> dict[tuple[str, str], dict[str, str]]:
    spans: dict[tuple[str, str], dict[str, str]] = {}
    for resource, _scope, span in _debug_records(logs, signal="traces"):
        resource_attributes = _debug_attributes(resource)
        if resource_attributes.get("service.name") != _SANDBOX_SERVICE_NAME:
            continue
        span_attributes = _debug_attributes(span)
        if (
            span_attributes.get("agenticrun.uid") != run_uid
            or span_attributes.get("gen_ai.operation.name") != expected_operation
            or span_attributes.get("gen_ai.provider.name") != expected_provider
        ):
            continue
        trace_id = _debug_field(span, "Trace ID")
        span_id = _debug_field(span, "ID")
        if not _nonzero_hex_id(trace_id, 32) or not _nonzero_hex_id(span_id, 16):
            continue
        spans[(trace_id.lower(), span_id.lower())] = resource_attributes
    return spans


def logs_contain_traces_for_run(
    logs: str,
    run_uid: str,
    *,
    expected_operation: str,
    expected_provider: str,
) -> bool:
    """True when a real inference span for this run and endpoint is exported."""
    return bool(
        _genai_spans_for_run(
            logs,
            run_uid,
            expected_operation=expected_operation,
            expected_provider=expected_provider,
        )
    )


def _json_log_body(record: str) -> dict[str, object] | None:
    for line in record.splitlines():
        key, separator, value = line.strip().partition(":")
        if not separator or key.strip() != "Body":
            continue
        value = value.strip()
        if not value.startswith("Str(") or not value.endswith(")"):
            return None
        try:
            body = json.loads(value[4:-1])
        except json.JSONDecodeError:
            return None
        return body if isinstance(body, dict) else None
    return None


def logs_contain_audit_logs_for_run(
    logs: str,
    run_uid: str,
    *,
    phase: str,
    expected_operation: str,
    expected_provider: str,
) -> bool:
    """True when a legacy choice-event log is correlated to its source trace."""
    spans = _genai_spans_for_run(
        logs,
        run_uid,
        expected_operation=expected_operation,
        expected_provider=expected_provider,
    )
    agent_spans = _genai_spans_for_run(
        logs,
        run_uid,
        expected_operation="invoke_agent",
        expected_provider=expected_provider,
    )
    if not spans or not agent_spans:
        return False
    inference_traces = {trace_id for trace_id, _span_id in spans}

    for resource, _scope, record in _debug_records(logs, signal="logs"):
        resource_attributes = _debug_attributes(resource)
        if resource_attributes.get("service.name") != _SANDBOX_SERVICE_NAME:
            continue
        attributes = _debug_attributes(record)
        if (
            attributes.get("agenticrun.uid") != run_uid
            or attributes.get("agenticrun.phase") != phase
            or attributes.get("event") != "gen_ai.choice"
        ):
            continue
        body = _json_log_body(record)
        if (
            body is None
            or not set(body).issubset({"gen_ai.completion", "gen_ai.reasoning_content"})
            or len(body) > 1
            or any(not isinstance(value, str) for value in body.values())
        ):
            continue
        trace_id = _debug_field(record, "Trace ID").lower()
        span_id = _debug_field(record, "Span ID").lower()
        if not _nonzero_hex_id(trace_id, 32) or not _nonzero_hex_id(span_id, 16):
            continue
        source_resource = agent_spans.get((trace_id, span_id))
        if source_resource == resource_attributes and trace_id in inference_traces:
            return True
    return False


def wait_for_otel_traces(
    core_api: CoreV1Api,
    namespace: str,
    run_uid: str,
    *,
    expected_operation: str,
    expected_provider: str,
    timeout_seconds: float = DEFAULT_POLL_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
) -> str:
    """Poll collector logs until the expected inference span for ``run_uid`` is visible."""
    return _poll_collector_logs(
        core_api,
        namespace,
        run_uid,
        predicate=lambda logs: logs_contain_traces_for_run(
            logs,
            run_uid,
            expected_operation=expected_operation,
            expected_provider=expected_provider,
        ),
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        evidence_kind=f"traces (operation={expected_operation}, provider={expected_provider})",
    )


def wait_for_otel_tool_result_inspection(
    core_api: CoreV1Api,
    namespace: str,
    run_uid: str,
    *,
    expected_outcome: str = "benign",
    timeout_seconds: float = TOOL_RESULT_INSPECTION_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
) -> str:
    """Poll collector logs until a tool-result inspection span is visible."""
    return _poll_collector_logs(
        core_api,
        namespace,
        run_uid,
        predicate=lambda logs: logs_contain_tool_result_inspection_for_run(
            logs,
            run_uid,
            expected_outcome=expected_outcome,
        ),
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        evidence_kind="tool-result inspection span",
        tail_lines=None,
    )


def wait_for_otel_audit_logs(
    core_api: CoreV1Api,
    namespace: str,
    run_uid: str,
    *,
    phase: str,
    expected_operation: str,
    expected_provider: str,
    timeout_seconds: float = DEFAULT_POLL_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
) -> str:
    """Poll collector logs until a legacy ``gen_ai.choice`` audit record is visible."""
    return _poll_collector_logs(
        core_api,
        namespace,
        run_uid,
        predicate=lambda logs: logs_contain_audit_logs_for_run(
            logs,
            run_uid,
            phase=phase,
            expected_operation=expected_operation,
            expected_provider=expected_provider,
        ),
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        evidence_kind=f"audit logs (phase={phase}, event=gen_ai.choice)",
    )


def _poll_collector_logs(
    core_api: CoreV1Api,
    namespace: str,
    run_uid: str,
    *,
    predicate: Callable[[str], bool],
    timeout_seconds: float,
    poll_interval_seconds: float,
    evidence_kind: str,
    tail_lines: int | None = DEFAULT_LOG_TAIL_LINES,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    last_logs = ""
    while time.monotonic() < deadline:
        last_logs = fetch_otel_collector_logs(
            core_api,
            namespace,
            tail_lines=tail_lines,
        )
        if predicate(last_logs):
            return last_logs
        time.sleep(poll_interval_seconds)
    snippet = last_logs[-2000:] if last_logs else "(empty collector logs)"
    msg = (
        f"OTEL collector missing {evidence_kind} for run_uid={run_uid} "
        f"after {timeout_seconds}s; recent collector log tail:\n{snippet}"
    )
    raise AssertionError(msg)


def assert_otel_deployment_present(core_api: CoreV1Api, namespace: str) -> None:
    """Raise if the e2e OTEL collector Deployment is missing."""
    pods = core_api.list_namespaced_pod(
        namespace=namespace,
        label_selector=OTEL_COLLECTOR_LABEL,
    )
    if pods.items:
        return
    msg = (
        f"OTEL collector pods not found in {namespace} "
        f"(expected deployment/{DEFAULT_OTEL_DEPLOYMENT})"
    )
    raise RuntimeError(msg)
