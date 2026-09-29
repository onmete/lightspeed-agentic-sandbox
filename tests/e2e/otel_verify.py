"""Verify sandbox batch runs exported OTLP traces and audit logs to the e2e collector."""

from __future__ import annotations

import time
from collections.abc import Callable

from kubernetes.client import ApiException, CoreV1Api  # type: ignore[import-untyped]

from tests.e2e.suite_setup import DEFAULT_OTEL_DEPLOYMENT

OTEL_COLLECTOR_LABEL = "app=lightspeed-otel-collector"
DEFAULT_POLL_TIMEOUT_SECONDS = 90.0
DEFAULT_POLL_INTERVAL_SECONDS = 3.0
DEFAULT_LOG_TAIL_LINES = 2000


def fetch_otel_collector_logs(
    core_api: CoreV1Api,
    namespace: str,
    *,
    tail_lines: int = DEFAULT_LOG_TAIL_LINES,
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
        try:
            chunk = core_api.read_namespaced_pod_log(
                name=pod_name,
                namespace=namespace,
                tail_lines=tail_lines,
            )
        except ApiException as exc:
            raise RuntimeError(f"read logs for pod/{pod_name}: {exc.reason}") from exc
        chunks.append(chunk)
    return "\n".join(chunks)


def _record_blocks(logs: str, marker: str) -> list[str]:
    """Split debug-exporter output into individual span or log-record blocks."""
    records: list[str] = []
    current: list[str] | None = None
    for line in logs.splitlines():
        if line.lstrip().startswith(marker):
            if current is not None:
                records.append("\n".join(current))
            current = [line]
        elif current is not None:
            current.append(line)
    if current is not None:
        records.append("\n".join(current))
    return records


def logs_contain_traces_for_run(logs: str, run_uid: str) -> bool:
    """True when one span contains both this run ID and a GenAI operation."""
    return any(
        f"agenticrun.uid: Str({run_uid})" in record
        and any(
            f"gen_ai.operation.name: Str({operation})" in record
            for operation in ("invoke_agent", "chat", "execute_tool")
        )
        for record in _record_blocks(logs, "Span #")
    )


def logs_contain_audit_logs_for_run(logs: str, run_uid: str, *, phase: str) -> bool:
    """True when one log record contains this run, phase, and operation event."""
    return any(
        f"agenticrun.uid: Str({run_uid})" in record
        and f"agenticrun.phase: Str({phase})" in record
        and any(
            f"event: Str({operation})" in record
            for operation in ("invoke_agent", "chat", "execute_tool")
        )
        for record in _record_blocks(logs, "LogRecord #")
    )


def wait_for_otel_traces(
    core_api: CoreV1Api,
    namespace: str,
    run_uid: str,
    *,
    timeout_seconds: float = DEFAULT_POLL_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
) -> str:
    """Poll collector logs until trace export for ``run_uid`` is visible."""
    return _poll_collector_logs(
        core_api,
        namespace,
        run_uid,
        predicate=lambda logs: logs_contain_traces_for_run(logs, run_uid),
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        evidence_kind="traces",
    )


def wait_for_otel_audit_logs(
    core_api: CoreV1Api,
    namespace: str,
    run_uid: str,
    *,
    phase: str,
    timeout_seconds: float = DEFAULT_POLL_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
) -> str:
    """Poll collector logs until audit log export for ``run_uid`` is visible."""
    return _poll_collector_logs(
        core_api,
        namespace,
        run_uid,
        predicate=lambda logs: logs_contain_audit_logs_for_run(logs, run_uid, phase=phase),
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        evidence_kind=f"audit logs (phase={phase})",
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
) -> str:
    deadline = time.monotonic() + timeout_seconds
    last_logs = ""
    while time.monotonic() < deadline:
        last_logs = fetch_otel_collector_logs(core_api, namespace)
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
