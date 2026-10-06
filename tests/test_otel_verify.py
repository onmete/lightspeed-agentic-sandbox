"""Unit tests for OTEL evidence predicates."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

import pytest
from kubernetes.client import (
    ApiClient,
    Configuration,
    CoreV1Api,
)  # type: ignore[import-untyped]

from tests.e2e.otel_verify import (
    fetch_otel_collector_logs,
    logs_contain_audit_logs_for_run,
    logs_contain_tool_result_inspection_for_run,
    logs_contain_traces_for_run,
    wait_for_otel_audit_logs,
    wait_for_otel_traces,
)


class TestOtelVerify:
    RUN_UID = "a" * 32
    EXPECTED_OPERATION = "chat"
    EXPECTED_PROVIDER = "openai"

    def _span_block(
        self,
        *,
        span_uid: str | None,
        resource_uid: str | None = None,
        resource_phase: str | None = None,
        operation: str = "chat",
        provider: str = "openai",
        trace_id: str = "1" * 32,
        span_id: str = "2" * 16,
        service_name: str = "lightspeed-agentic-sandbox",
    ) -> str:
        resource_attributes = [f"     -> service.name: Str({service_name})"]
        if resource_uid is not None:
            resource_attributes.append(f"     -> agenticrun.uid: Str({resource_uid})")
        if resource_phase is not None:
            resource_attributes.append(f"     -> agenticrun.phase: Str({resource_phase})")
        span_attributes = [
            f"     -> gen_ai.operation.name: Str({operation})",
            f"     -> gen_ai.provider.name: Str({provider})",
        ]
        if span_uid is not None:
            span_attributes.insert(0, f"     -> agenticrun.uid: Str({span_uid})")
        lines = [
            "2026-10-06T09:37:07.218Z\tinfo\tResourceSpans #0",
            "Resource SchemaURL: https://opentelemetry.io/schemas/1.41.0",
            "Resource attributes:",
            *resource_attributes,
            "ScopeSpans #0",
            "ScopeSpans SchemaURL: https://opentelemetry.io/schemas/1.41.0",
            "InstrumentationScope lightspeed_agentic.audit",
            "Span #0",
            f"    Trace ID       : {trace_id}",
            f"    ID             : {span_id}",
            f"    Name           : {operation} model",
            "    Kind           : Client",
            "    Status code    : Unset",
            "Attributes:",
            *span_attributes,
        ]
        return "\n".join(lines) + "\n"

    def _log_block(
        self,
        *,
        uid: str | None = None,
        phase: str | None = "analysis",
        event: str | None = "gen_ai.choice",
        body: str = '{"gen_ai.completion":"answer"}',
        trace_id: str = "1" * 32,
        span_id: str = "4" * 16,
        resource_uid: str | None = None,
        resource_phase: str | None = None,
        service_name: str = "lightspeed-agentic-sandbox",
    ) -> str:
        resource_attributes = [f"     -> service.name: Str({service_name})"]
        if resource_uid is not None:
            resource_attributes.append(f"     -> agenticrun.uid: Str({resource_uid})")
        if resource_phase is not None:
            resource_attributes.append(f"     -> agenticrun.phase: Str({resource_phase})")
        attributes = []
        if uid is not None:
            attributes.append(f"     -> agenticrun.uid: Str({uid})")
        if phase is not None:
            attributes.append(f"     -> agenticrun.phase: Str({phase})")
        if event is not None:
            attributes.append(f"     -> event: Str({event})")
        lines = [
            "2026-10-06T09:37:07.221Z\tinfo\tResourceLog #0",
            "Resource SchemaURL: https://opentelemetry.io/schemas/1.41.0",
            "Resource attributes:",
            *resource_attributes,
            "ScopeLogs #0",
            "ScopeLogs SchemaURL: https://opentelemetry.io/schemas/1.41.0",
            "InstrumentationScope lightspeed_agentic.audit",
            "LogRecord #0",
            f"    Body: Str({body})",
            "Attributes:",
            *attributes,
            f"    Trace ID       : {trace_id}",
            f"    Span ID        : {span_id}",
        ]
        return "\n".join(lines) + "\n"

    @pytest.mark.parametrize(
        ("span_options", "log_options", "expected_traces", "expected_audit"),
        [
            pytest.param({}, None, True, False, id="trace-only"),
            pytest.param({}, {}, True, True, id="correlated-trace-and-audit"),
            pytest.param(None, None, False, False, id="no-span-markers"),
            pytest.param(
                {"span_uid": None, "resource_uid": RUN_UID},
                {},
                False,
                False,
                id="resource-uid-is-not-span-uid",
            ),
            pytest.param({"trace_id": "0" * 32}, {}, False, False, id="zero-trace-id"),
            pytest.param({"span_id": "0" * 16}, {}, False, False, id="zero-span-id"),
            pytest.param(
                {"operation": "generate_content"},
                {},
                False,
                False,
                id="wrong-operation",
            ),
            pytest.param(
                {"provider": "gcp.vertex_ai"},
                {},
                False,
                False,
                id="wrong-provider",
            ),
            pytest.param({}, {"phase": "execution"}, True, False, id="wrong-phase"),
            pytest.param({}, {"trace_id": "3" * 32}, True, False, id="unmatched-context"),
            pytest.param(None, {}, False, False, id="audit-without-source-span"),
            pytest.param({"span_uid": "b" * 32}, {}, False, False, id="different-run"),
            pytest.param({}, {"event": "invoke_agent"}, True, False, id="event-body-mismatch"),
            pytest.param(
                {},
                {
                    "body": (
                        '{"gen_ai.completion":"answer","gen_ai.provider.name":"gcp.vertex_ai"}'
                    ),
                },
                True,
                False,
                id="non-choice-body",
            ),
            pytest.param({}, {"trace_id": "malformed"}, True, False, id="invalid-log-trace-id"),
            pytest.param({}, {"span_id": "0" * 16}, True, False, id="zero-log-span-id"),
            pytest.param(
                {},
                {"body": '{"gen_ai.completion":'},
                True,
                False,
                id="malformed-body",
            ),
            pytest.param(
                {"resource_uid": RUN_UID},
                {
                    "uid": None,
                    "resource_uid": RUN_UID,
                    "body": '{"gen_ai.completion":"answer"}',
                },
                True,
                False,
                id="uid-required-on-record",
            ),
            pytest.param(
                {"resource_phase": "analysis"},
                {
                    "phase": None,
                    "resource_phase": "analysis",
                    "body": '{"gen_ai.completion":"answer"}',
                },
                True,
                False,
                id="phase-required-on-record",
            ),
            pytest.param({}, {"event": None}, True, False, id="event-required-on-record"),
        ],
    )
    def test_correlated_telemetry_verifier(
        self,
        span_options: dict[str, Any] | None,
        log_options: dict[str, Any] | None,
        expected_traces: bool,
        expected_audit: bool,
    ) -> None:
        logs = f"agenticrun.uid={self.RUN_UID}\n"
        source_options: dict[str, Any] | None = None
        if span_options is not None:
            source_options = {"span_uid": self.RUN_UID, **span_options}
            logs += self._span_block(**source_options)
            if log_options is not None:
                logs += self._span_block(
                    **{**source_options, "operation": "invoke_agent", "span_id": "4" * 16}
                )
        if log_options is not None:
            options: dict[str, Any] = {"uid": self.RUN_UID, "span_id": "4" * 16}
            if source_options is not None:
                options["trace_id"] = source_options.get("trace_id", "1" * 32)
            options.update(log_options)
            logs += self._log_block(**options)
        assert (
            logs_contain_traces_for_run(
                logs,
                self.RUN_UID,
                expected_operation=self.EXPECTED_OPERATION,
                expected_provider=self.EXPECTED_PROVIDER,
            )
            is expected_traces
        )
        assert (
            logs_contain_audit_logs_for_run(
                logs,
                self.RUN_UID,
                phase="analysis",
                expected_operation=self.EXPECTED_OPERATION,
                expected_provider=self.EXPECTED_PROVIDER,
            )
            is expected_audit
        )

    def test_captured_collector_debug_format_with_bedrock(self) -> None:
        trace_id = "3" * 32
        agent_span_id = "4" * 16
        logs = self._span_block(
            span_uid=self.RUN_UID,
            provider="aws.bedrock",
            trace_id=trace_id,
            span_id="2" * 16,
        )
        logs += self._span_block(
            span_uid=self.RUN_UID,
            operation="invoke_agent",
            provider="aws.bedrock",
            trace_id=trace_id,
            span_id=agent_span_id,
        )
        logs += self._log_block(
            uid=self.RUN_UID,
            body='{"gen_ai.completion":"answer"}',
            trace_id=trace_id,
            span_id=agent_span_id,
        )

        assert logs_contain_traces_for_run(
            logs,
            self.RUN_UID,
            expected_operation=self.EXPECTED_OPERATION,
            expected_provider="aws.bedrock",
        )
        assert logs_contain_audit_logs_for_run(
            logs,
            self.RUN_UID,
            phase="analysis",
            expected_operation=self.EXPECTED_OPERATION,
            expected_provider="aws.bedrock",
        )

    @pytest.mark.parametrize(
        ("valid_header", "untrusted_header"),
        [
            pytest.param(
                "2026-10-06T09:37:07.218Z\tinfo\tResourceSpans #0",
                "untrusted text\tResourceSpans #0",
                id="resource-header",
            ),
            pytest.param(
                "ScopeSpans #0",
                "untrusted text\tScopeSpans #0",
                id="scope-header",
            ),
        ],
    )
    def test_untrusted_tab_prefix_is_not_a_trace_header(
        self, valid_header: str, untrusted_header: str
    ) -> None:
        logs = self._span_block(span_uid=self.RUN_UID, provider="aws.bedrock")
        logs = logs.replace(valid_header, untrusted_header, 1)

        assert not logs_contain_traces_for_run(
            logs,
            self.RUN_UID,
            expected_operation=self.EXPECTED_OPERATION,
            expected_provider="aws.bedrock",
        )

    def test_tool_result_inspection_positive(self) -> None:
        logs = (
            "ResourceSpans #0\n"
            "Span #0\n"
            f"     -> agenticrun.uid: Str({self.RUN_UID})\n"
            "[pod/otel-collector/otel-collector] 2026-10-02T19:15:25.104289208Z "
            "Trace ID       : shared-trace\n"
            "Span #1\n"
            "    Name           : tool_result.inspection\n"
            "     -> inspection.outcome: Str(benign)\n"
            "[pod/otel-collector/otel-collector] 2026-10-02T19:15:25.104289208Z "
            "Trace ID       : shared-trace\n"
        )
        assert logs_contain_tool_result_inspection_for_run(logs, self.RUN_UID)

    def test_tool_result_inspection_accepts_expected_malicious_outcome(self) -> None:
        logs = (
            "Span #0\n"
            f"     -> agenticrun.uid: Str({self.RUN_UID})\n"
            "Trace ID       : malicious-trace\n"
            "Span #1\n"
            "    Name           : tool_result.inspection\n"
            "     -> inspection.outcome: Str(malicious)\n"
            "Trace ID       : malicious-trace\n"
        )
        assert logs_contain_tool_result_inspection_for_run(
            logs,
            self.RUN_UID,
            expected_outcome="malicious",
        )

    def test_tool_result_inspection_rejects_non_benign_outcome(self) -> None:
        logs = (
            "ResourceSpans #0\n"
            "Span #0\n"
            f"     -> agenticrun.uid: Str({self.RUN_UID})\n"
            "Trace ID       : shared-trace\n"
            "Span #1\n"
            "    Name           : tool_result.inspection\n"
            "     -> inspection.outcome: Str(classifier_error)\n"
            "Trace ID       : shared-trace\n"
        )
        assert not logs_contain_tool_result_inspection_for_run(logs, self.RUN_UID)

    def test_tool_result_inspection_rejects_inspection_span_from_another_trace(self) -> None:
        logs = (
            "ResourceSpans #0\n"
            "Span #0\n"
            f"     -> agenticrun.uid: Str({self.RUN_UID})\n"
            "Trace ID       : run-trace\n"
            "Span #1\n"
            "    Name           : tool_result.inspection\n"
            "Trace ID       : other-trace\n"
        )
        assert not logs_contain_tool_result_inspection_for_run(logs, self.RUN_UID)

    def test_traces_are_consumed_from_raw_utf8_collector_logs(
        self,
    ) -> None:
        trace_id = "3" * 32
        agent_span_id = "4" * 16
        log_body = (
            "collector terminal: café 東京\n"
            + self._span_block(
                span_uid=self.RUN_UID,
                provider="aws.bedrock",
                trace_id=trace_id,
                span_id="2" * 16,
            )
            + self._span_block(
                span_uid=self.RUN_UID,
                operation="invoke_agent",
                provider="aws.bedrock",
                trace_id=trace_id,
                span_id=agent_span_id,
            )
            + self._log_block(
                uid=self.RUN_UID,
                body='{"gen_ai.completion":"answer"}',
                trace_id=trace_id,
                span_id=agent_span_id,
            )
        ).encode("utf-8")
        pod_list_body = (
            b'{"apiVersion":"v1","kind":"PodList","metadata":{"resourceVersion":"1"},'
            b'"items":[{"metadata":{"name":"otel-collector-pod","uid":"collector-pod-uid"}}]}'
        )

        class CollectorLogHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                if self.path.startswith("/api/v1/namespaces/e2e/pods?"):
                    payload = pod_list_body
                    content_type = "application/json"
                elif self.path.startswith("/api/v1/namespaces/e2e/pods/otel-collector-pod/log"):
                    payload = log_body
                    content_type = "text/plain; charset=utf-8"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        with ThreadingHTTPServer(("127.0.0.1", 0), CollectorLogHandler) as server:
            configuration = Configuration()
            configuration.host = f"http://127.0.0.1:{server.server_port}"
            api_client = ApiClient(configuration=configuration)
            core_api = CoreV1Api(api_client)
            server_thread = Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            try:
                visible_logs = wait_for_otel_traces(
                    core_api,
                    "e2e",
                    self.RUN_UID,
                    expected_operation=self.EXPECTED_OPERATION,
                    expected_provider="aws.bedrock",
                    timeout_seconds=2.0,
                    poll_interval_seconds=0.01,
                )

                assert "collector terminal: café 東京" in visible_logs
                audit_logs = wait_for_otel_audit_logs(
                    core_api,
                    "e2e",
                    self.RUN_UID,
                    phase="analysis",
                    expected_operation=self.EXPECTED_OPERATION,
                    expected_provider="aws.bedrock",
                    timeout_seconds=2.0,
                    poll_interval_seconds=0.01,
                )
                assert "collector terminal: café 東京" in audit_logs
                log_body = b"\xff"
                with pytest.raises(UnicodeDecodeError):
                    fetch_otel_collector_logs(core_api, "e2e")
            finally:
                try:
                    api_client.close()
                finally:
                    server.shutdown()
                    server_thread.join()
