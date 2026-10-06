"""Unit tests for batch E2E helpers (no live cluster)."""

from __future__ import annotations

import json
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from unittest.mock import MagicMock

import pytest
from kubernetes.client import (
    ApiClient,
    ApiException,
    Configuration,
    CoreV1Api,
)  # type: ignore[import-untyped]
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import SpanKind, Status, StatusCode

from lightspeed_agentic.tracing import OTLPJsonStdoutExporter
from tests.e2e.batch_runner import (
    _body_from_result_cr,
    _build_job_spec,
    _delete_config_map_ignore_not_found,
    _enrich_body_from_otlp_stdout,
    _fetch_job_pod_logs,
    _needs_agent_result_enrichment,
    _parse_echo_token_from_otlp_stdout,
    _set_config_map_job_owner,
    build_result_template,
    run_batch_query,
)
from tests.e2e.skills_fixtures import (
    E2E_POD_SKILLS_DIR,
    E2E_POD_SKILLS_SRC_DIR,
    _cm_key_from_rel,
    _rel_from_cm_key,
    skill_materialize_script,
)
from tests.e2e.suite_setup import (
    BatchE2EConfig,
    _session_job_env,
    load_batch_e2e_config,
    resolve_llm_secret,
    resolve_model,
)


def _agent_span(
    output: dict[str, Any] | None,
    *,
    run_uid: str,
    phase: str,
    status_code: StatusCode | None = None,
) -> tuple[str, dict[str, Any], StatusCode | None]:
    attributes: dict[str, Any] = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": "lightspeed",
        "agenticrun.uid": run_uid,
        "agenticrun.phase": phase,
    }
    if output is not None:
        attributes["gen_ai.output.messages"] = json.dumps(
            [
                {
                    "role": "assistant",
                    "parts": [
                        {
                            "type": "text",
                            "content": json.dumps(output, ensure_ascii=False),
                        }
                    ],
                    "finish_reason": "unknown",
                }
            ]
        )
    return "invoke_agent lightspeed", attributes, status_code


def _tool_span(
    result: Any,
    *,
    run_uid: str,
    phase: str,
    tool_name: str = "shell",
    status_code: StatusCode | None = None,
) -> tuple[str, dict[str, Any], StatusCode | None]:
    attributes: dict[str, Any] = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": tool_name,
        "agenticrun.uid": run_uid,
        "agenticrun.phase": phase,
    }
    if result is not None:
        attributes["gen_ai.tool.call.result"] = json.dumps(result)
    return f"execute_tool {tool_name}", attributes, status_code


def _openai_exec_result(output: str, *, exit_code: int) -> str:
    """Format a sanitized `ExecCommandTool._format_response` result."""
    return (
        "Chunk ID: a1b2c3\n"
        "Wall time: 0.0123 seconds\n"
        f"Process exited with code {exit_code}\n"
        "Output:\n"
        f"{output}"
    )


def _export_otlp_stdout(
    capsys: pytest.CaptureFixture[str],
    spans: list[tuple[str, dict[str, Any], StatusCode | None]],
    *,
    capture_content: bool = True,
) -> str:
    provider = TracerProvider(
        resource=Resource.create({"service.name": "lightspeed-agentic-sandbox"})
    )
    provider.add_span_processor(
        SimpleSpanProcessor(OTLPJsonStdoutExporter(capture_content=capture_content))
    )
    tracer = provider.get_tracer(
        "lightspeed_agentic",
        schema_url="https://opentelemetry.io/schemas/1.41.0",
    )
    for name, attributes, status_code in spans:
        with tracer.start_as_current_span(
            name,
            kind=SpanKind.INTERNAL,
            attributes=attributes,
        ) as span:
            if status_code is not None:
                span.set_status(Status(status_code))
    provider.shutdown()
    return capsys.readouterr().out


class TestBuildResultTemplate:
    def test_analysis_template(self) -> None:
        tmpl = build_result_template(
            namespace="openshift-lightspeed",
            result_name="run-analysis-1",
            run_uid="abc123",
            step="analysis",
            agentic_run_name="run",
            session_id="sess1",
        )
        assert tmpl["kind"] == "AnalysisResult"
        assert tmpl["metadata"]["name"] == "run-analysis-1"
        assert tmpl["metadata"]["labels"]["agentic.openshift.io/run"] == "abc123"
        assert tmpl["spec"]["agenticRunName"] == "run"


class TestBodyFromResultCr:
    def test_agent_success(self) -> None:
        cr: dict[str, Any] = {
            "status": {
                "actionRequired": "False",
                "diagnosis": {"summary": "all good", "rootCause": "none"},
                "conditions": [
                    {
                        "type": "Completed",
                        "status": "True",
                        "reason": "Succeeded",
                        "message": "Step completed",
                    }
                ],
            }
        }
        body = _body_from_result_cr(cr)
        assert body["success"] is True
        assert body["summary"] == "all good"

    def test_agent_failure_reason(self) -> None:
        cr = {
            "status": {
                "failureReason": "agent timed out",
                "conditions": [
                    {
                        "type": "Completed",
                        "status": "True",
                        "reason": "Failed",
                        "message": "Step failed",
                    }
                ],
            }
        }
        body = _body_from_result_cr(cr)
        assert body["success"] is False
        assert body["summary"] == "agent timed out"

    def test_flattens_diagnosis_echo_fields(self) -> None:
        cr = {
            "status": {
                "diagnosis": {
                    "summary": "echo ok",
                    "namespaces": "fleet-alpha,fleet-beta",
                    "ticketId": "E2E-STRUCT-001",
                },
                "conditions": [
                    {
                        "type": "Completed",
                        "status": "True",
                        "reason": "Succeeded",
                        "message": "Step completed",
                    }
                ],
            }
        }
        body = _body_from_result_cr(cr)
        assert body["success"] is True
        assert body["namespaces"] == "fleet-alpha,fleet-beta"
        assert body["ticketId"] == "E2E-STRUCT-001"
        assert body["summary"] == "echo ok"


class TestSkillMaterializeScript:
    def test_copies_from_src_to_skills_root(self) -> None:
        script = skill_materialize_script()
        assert "cp -aL" in script
        assert "! -name '..*'" in script
        assert "/mnt/e2e-skills-src" in script
        assert "/app/skills" in script
        assert "/app/skills/.agents" in script

    def test_skips_kubelet_configmap_directories(self, tmp_path: Path) -> None:
        src_root = tmp_path / "src"
        dest_root = tmp_path / "skills"
        skill = src_root / "find-token"
        payload = skill / "..2026_10_01_12_00_00.123456789"
        (payload / "scripts").mkdir(parents=True)
        (payload / "SKILL.md").write_text("skill body\n", encoding="utf-8")
        (payload / "scripts" / "find-token.sh").write_text("echo token\n", encoding="utf-8")
        (skill / "..data").symlink_to(payload.name)
        (skill / "SKILL.md").symlink_to(Path("..data") / "SKILL.md")
        (skill / "scripts").symlink_to(Path("..data") / "scripts")

        script = skill_materialize_script().replace(E2E_POD_SKILLS_SRC_DIR, str(src_root))
        script = script.replace(E2E_POD_SKILLS_DIR, str(dest_root))
        subprocess.run(  # noqa: S603
            ["bash", "-c", script],  # noqa: S607
            check=True,
        )

        copied = dest_root / "find-token"
        assert (copied / "SKILL.md").read_text(encoding="utf-8") == "skill body\n"
        assert (copied / "scripts" / "find-token.sh").read_text(encoding="utf-8") == "echo token\n"
        assert not (copied / "SKILL.md").is_symlink()
        assert not (copied / "scripts").is_symlink()
        assert [path.name for path in copied.iterdir() if path.name.startswith("..")] == []


class TestSkillConfigMapKeys:
    def test_round_trip_simple_path(self) -> None:
        rel = "scripts/echo-token.sh"
        key = _cm_key_from_rel(rel)
        assert key == "scripts__echo-token.sh"
        assert _rel_from_cm_key(key) == rel

    def test_rejects_double_underscore_in_path(self) -> None:
        with pytest.raises(ValueError, match="must not contain '__'"):
            _cm_key_from_rel("docs/a__b.md")


class TestEnrichBodyFromOtlpStdout:
    def test_merges_exact_agent_result_into_generic_cr(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run_uid = "run-uid"
        phase = "analysis"
        output = {
            "success": True,
            "summary": "e2e-flat-ok",
            "ticketId": "E2E-STRUCT-001",
            "items": [{"name": "alpha", "metadata": {"region": "east"}}],
        }
        body = _body_from_result_cr(
            {
                "status": {
                    "conditions": [
                        {
                            "type": "Completed",
                            "status": "True",
                            "reason": "Succeeded",
                            "message": "Step completed",
                        }
                    ]
                }
            }
        )
        stdout = _export_otlp_stdout(
            capsys,
            [_agent_span(output, run_uid=run_uid, phase=phase)],
        )

        assert body == {"success": True, "summary": "Step completed"}
        assert _enrich_body_from_otlp_stdout(
            body,
            stdout,
            run_uid=run_uid,
            phase=phase,
        ) == {**body, **output}

    def test_preserves_domain_false_agent_result(self, capsys: pytest.CaptureFixture[str]) -> None:
        run_uid = "run-uid"
        phase = "analysis"
        output = {
            "success": False,
            "summary": "domain validation failed",
            "invalidFields": ["ticketId"],
        }
        body = _body_from_result_cr(
            {
                "status": {
                    "conditions": [
                        {
                            "type": "Completed",
                            "status": "True",
                            "reason": "Succeeded",
                            "message": "Step completed",
                        }
                    ]
                }
            }
        )
        stdout = _export_otlp_stdout(
            capsys,
            [_agent_span(output, run_uid=run_uid, phase=phase)],
        )

        enriched = _enrich_body_from_otlp_stdout(
            body,
            stdout,
            run_uid=run_uid,
            phase=phase,
        )
        assert enriched["success"] is False
        assert enriched["summary"] == "domain validation failed"
        assert enriched["invalidFields"] == ["ticketId"]

    def test_ignores_wrong_run_phase_agent_and_non_agent_content(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_uid = "run-uid"
        phase = "analysis"
        output = {"success": True, "summary": "unrelated", "ticketId": "not-current"}
        inference_attributes = _agent_span(output, run_uid=run_uid, phase=phase)[1]
        inference_attributes["gen_ai.operation.name"] = "chat"
        wrong_agent = _agent_span(output, run_uid=run_uid, phase=phase)
        wrong_agent[1]["gen_ai.agent.name"] = "other-agent"
        wrong_span_name = _agent_span(output, run_uid=run_uid, phase=phase)
        wrong_span_name = (
            "invoke_agent other-agent",
            wrong_span_name[1],
            wrong_span_name[2],
        )
        stdout = _export_otlp_stdout(
            capsys,
            [
                _agent_span(output, run_uid="other-run", phase=phase),
                _agent_span(output, run_uid=run_uid, phase="execution"),
                wrong_agent,
                wrong_span_name,
                ("chat model", inference_attributes, None),
                _tool_span(
                    {"success": True, "summary": "tool result", "ticketId": "tool-only"},
                    run_uid=run_uid,
                    phase=phase,
                ),
            ],
        )
        logs = f"INFO lightspeed_agentic: response={json.dumps(output)}\n{stdout}"
        body = {"success": True, "summary": "Step completed"}

        assert (
            _enrich_body_from_otlp_stdout(
                body,
                logs,
                run_uid=run_uid,
                phase=phase,
            )
            == body
        )

    def test_filtered_missing_or_failed_terminal_output_does_not_invent_result(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_uid = "run-uid"
        phase = "analysis"
        body = {"success": True, "summary": "Step completed"}
        output = {"success": True, "summary": "must not be recovered"}
        filtered = _export_otlp_stdout(
            capsys,
            [_agent_span(output, run_uid=run_uid, phase=phase)],
            capture_content=False,
        )
        no_terminal = _export_otlp_stdout(
            capsys,
            [_agent_span(None, run_uid=run_uid, phase=phase)],
        )

        failed_terminal = _export_otlp_stdout(
            capsys,
            [
                _agent_span(
                    output,
                    run_uid=run_uid,
                    phase=phase,
                    status_code=StatusCode.ERROR,
                )
            ],
        )
        assert (
            _enrich_body_from_otlp_stdout(
                body,
                filtered,
                run_uid=run_uid,
                phase=phase,
            )
            == body
        )
        assert (
            _enrich_body_from_otlp_stdout(
                body,
                no_terminal,
                run_uid=run_uid,
                phase=phase,
            )
            == body
        )
        assert (
            _enrich_body_from_otlp_stdout(
                body,
                failed_terminal,
                run_uid=run_uid,
                phase=phase,
            )
            == body
        )
        assert (
            _enrich_body_from_otlp_stdout(
                body,
                f"INFO lightspeed_agentic: response={json.dumps(output)}",
                run_uid=run_uid,
                phase=phase,
            )
            == body
        )

    def test_typed_and_failed_cr_status_remain_authoritative(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_uid = "run-uid"
        phase = "analysis"
        output = {
            "success": False,
            "summary": "agent output",
            "diagnosis": {"summary": "agent diagnosis"},
            "ticketId": "must-not-override",
        }
        stdout = _export_otlp_stdout(
            capsys,
            [_agent_span(output, run_uid=run_uid, phase=phase)],
        )
        typed_body = _body_from_result_cr(
            {
                "status": {
                    "actionRequired": "False",
                    "diagnosis": {"summary": "CR diagnosis", "rootCause": "CR root cause"},
                    "conditions": [
                        {
                            "type": "Completed",
                            "status": "True",
                            "reason": "Succeeded",
                            "message": "Step completed",
                        }
                    ],
                }
            }
        )
        failed_body = _body_from_result_cr(
            {
                "status": {
                    "failureReason": "CR failure",
                    "conditions": [
                        {
                            "type": "Completed",
                            "status": "True",
                            "reason": "Failed",
                            "message": "Step failed",
                        }
                    ],
                }
            }
        )
        non_generic_body = {"success": True, "summary": "CR summary"}
        assert _needs_agent_result_enrichment(non_generic_body) is False
        assert (
            _enrich_body_from_otlp_stdout(
                non_generic_body,
                stdout,
                run_uid=run_uid,
                phase=phase,
            )
            == non_generic_body
        )

        assert _needs_agent_result_enrichment(typed_body) is False
        assert (
            _enrich_body_from_otlp_stdout(
                typed_body,
                stdout,
                run_uid=run_uid,
                phase=phase,
            )
            == typed_body
        )
        assert (
            _enrich_body_from_otlp_stdout(
                failed_body,
                stdout,
                run_uid=run_uid,
                phase=phase,
            )
            == failed_body
        )


class TestParseEchoTokenFromToolSpans:
    def test_extracts_latest_token_from_successful_structured_tool_result(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_uid = "run-uid"
        phase = "analysis"
        echoed_token = "a" * 32
        first_token = "b" * 32
        token = "c" * 32
        returncode_token = "f" * 32
        stdout = _export_otlp_stdout(
            capsys,
            [
                _agent_span(
                    {"success": True, "summary": "echoed", "token": echoed_token},
                    run_uid=run_uid,
                    phase=phase,
                ),
                _tool_span(
                    {
                        "exit_code": 0,
                        "stderr": "",
                        "stdout": json.dumps({"token": first_token, "status": "ok"}) + "\n",
                    },
                    run_uid=run_uid,
                    phase=phase,
                ),
                _tool_span(
                    {
                        "exit_code": 0,
                        "stderr": "",
                        "stdout": json.dumps({"token": token, "status": "ok"}) + "\n",
                    },
                    run_uid=run_uid,
                    phase=phase,
                ),
                _tool_span(
                    {
                        "returncode": 0,
                        "stderr": "",
                        "stdout": json.dumps({"token": returncode_token, "status": "ok"}) + "\n",
                    },
                    run_uid=run_uid,
                    phase=phase,
                ),
            ],
        )

        assert (
            _parse_echo_token_from_otlp_stdout(
                stdout,
                run_uid=run_uid,
                phase=phase,
            )
            == returncode_token
        )

    def test_openai_exec_result_requires_success_exit_code(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_uid = "run-uid"
        phase = "analysis"
        token = "f" * 32
        token_json = json.dumps({"token": token, "status": "ok"})
        output = f"Contents of SKILL.md\n{token_json}\n"
        successful = _export_otlp_stdout(
            capsys,
            [
                _tool_span(
                    _openai_exec_result(output, exit_code=0),
                    run_uid=run_uid,
                    phase=phase,
                    tool_name="exec_command",
                )
            ],
        )
        fallback_output = (
            "PTY transport failed before the interactive session opened; "
            "fell back to one-shot exec.\n" + output
        )
        fallback = _export_otlp_stdout(
            capsys,
            [
                _tool_span(
                    _openai_exec_result(fallback_output, exit_code=0),
                    run_uid=run_uid,
                    phase=phase,
                    tool_name="exec_command",
                )
            ],
        )
        failed = _export_otlp_stdout(
            capsys,
            [
                _tool_span(
                    _openai_exec_result(output, exit_code=7),
                    run_uid=run_uid,
                    phase=phase,
                    tool_name="exec_command",
                )
            ],
        )
        embedded = _export_otlp_stdout(
            capsys,
            [
                _tool_span(
                    _openai_exec_result(
                        f"Skill prose mentions {token_json} inline.\n",
                        exit_code=0,
                    ),
                    run_uid=run_uid,
                    phase=phase,
                    tool_name="exec_command",
                )
            ],
        )

        assert (
            _parse_echo_token_from_otlp_stdout(
                successful,
                run_uid=run_uid,
                phase=phase,
            )
            == token
        )
        assert (
            _parse_echo_token_from_otlp_stdout(
                fallback,
                run_uid=run_uid,
                phase=phase,
            )
            == token
        )
        assert (
            _parse_echo_token_from_otlp_stdout(
                failed,
                run_uid=run_uid,
                phase=phase,
            )
            == ""
        )
        assert (
            _parse_echo_token_from_otlp_stdout(
                embedded,
                run_uid=run_uid,
                phase=phase,
            )
            == ""
        )

    def test_deepagents_success_trailer_is_required(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_uid = "run-uid"
        phase = "analysis"
        token = "e" * 32
        script_json = json.dumps({"token": token, "status": "ok"})
        # Sanitized captured DeepAgents result: stdout followed by its command status.
        successful = _export_otlp_stdout(
            capsys,
            [
                _tool_span(
                    script_json + "\n\n[Command succeeded with exit code 0]",
                    run_uid=run_uid,
                    phase=phase,
                )
            ],
        )
        failed = _export_otlp_stdout(
            capsys,
            [
                _tool_span(
                    script_json + "\n\n[Command failed with exit code 1]",
                    run_uid=run_uid,
                    phase=phase,
                )
            ],
        )

        assert (
            _parse_echo_token_from_otlp_stdout(
                successful,
                run_uid=run_uid,
                phase=phase,
            )
            == token
        )
        assert (
            _parse_echo_token_from_otlp_stdout(
                failed,
                run_uid=run_uid,
                phase=phase,
            )
            == ""
        )

    def test_rejects_unrelated_failed_filtered_and_echoed_tokens(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_uid = "run-uid"
        phase = "analysis"
        token = "d" * 32
        tool_result = {
            "returncode": 0,
            "stderr": "",
            "stdout": json.dumps({"token": token, "status": "ok"}),
        }
        nonzero_result = {
            "exit_code": 127,
            "stderr": "",
            "stdout": json.dumps({"token": token, "status": "ok"}),
        }
        nonzero_returncode_result = {
            "returncode": 127,
            "stderr": "",
            "stdout": json.dumps({"token": token, "status": "ok"}),
        }
        error_result = {
            "error": "Execution failed: sanitized command failure",
            "stderr": "",
            "stdout": json.dumps({"token": token, "status": "ok"}),
        }
        stdout = _export_otlp_stdout(
            capsys,
            [
                _agent_span(
                    {"success": True, "summary": "echo", "token": token},
                    run_uid=run_uid,
                    phase=phase,
                ),
                _tool_span(tool_result, run_uid="other-run", phase=phase),
                _tool_span(tool_result, run_uid=run_uid, phase="execution"),
                _tool_span(
                    tool_result,
                    run_uid=run_uid,
                    phase=phase,
                    status_code=StatusCode.ERROR,
                ),
                _tool_span(nonzero_result, run_uid=run_uid, phase=phase),
                _tool_span(
                    nonzero_returncode_result,
                    run_uid=run_uid,
                    phase=phase,
                ),
                _tool_span(error_result, run_uid=run_uid, phase=phase),
                _tool_span(None, run_uid=run_uid, phase=phase),
            ],
        )
        logs = f"INFO lightspeed_agentic: token={token}\n{stdout}"
        filtered = _export_otlp_stdout(
            capsys,
            [_tool_span(tool_result, run_uid=run_uid, phase=phase)],
            capture_content=False,
        )

        assert (
            _parse_echo_token_from_otlp_stdout(
                logs,
                run_uid=run_uid,
                phase=phase,
            )
            == ""
        )
        assert (
            _parse_echo_token_from_otlp_stdout(
                filtered,
                run_uid=run_uid,
                phase=phase,
            )
            == ""
        )


class TestFetchedPodLogs:
    def test_multiline_utf8_logs_feed_result_and_echo_token_consumers(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_uid = "raw-pod-log-run"
        phase = "analysis"
        echo_token = "a" * 32
        output = {
            "success": True,
            "summary": "context-echo-ok — café 東京",
            "namespaces": "ns-e2e-alpha, ns-e2e-bravo",
            "ticketId": "E2E-STRUCT-001",
            "items": [{"name": "café 東京", "count": 1}],
        }
        stdout = _export_otlp_stdout(
            capsys,
            [
                _agent_span(output, run_uid=run_uid, phase=phase),
                _tool_span(
                    _openai_exec_result(
                        "Echo-token skill instructions from SKILL.md\n"
                        + json.dumps({"token": echo_token, "status": "ok"})
                        + "\n",
                        exit_code=0,
                    ),
                    run_uid=run_uid,
                    phase=phase,
                    tool_name="exec_command",
                ),
            ],
        )
        log_lines = [
            json.dumps(json.loads(line), ensure_ascii=False) for line in stdout.splitlines()
        ]
        log_body = ("batch log: café 東京\n" + "\n".join(log_lines) + "\n").encode("utf-8")
        pod_list_body = (
            b'{"apiVersion":"v1","kind":"PodList","metadata":{"resourceVersion":"1"},'
            b'"items":[{"metadata":{"name":"batch-pod","uid":"batch-pod-uid"}}]}'
        )

        class PodLogHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                if self.path.startswith("/api/v1/namespaces/e2e/pods?"):
                    payload = pod_list_body
                    content_type = "application/json"
                elif self.path.startswith("/api/v1/namespaces/e2e/pods/batch-pod/log"):
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

        with ThreadingHTTPServer(("127.0.0.1", 0), PodLogHandler) as server:
            configuration = Configuration()
            configuration.host = f"http://127.0.0.1:{server.server_port}"
            api_client = ApiClient(configuration=configuration)
            core_api = CoreV1Api(api_client)
            server_thread = Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            try:
                pod_logs = _fetch_job_pod_logs(core_api, "e2e", "batch-job")
                generic_body = {"success": True, "summary": "Step completed"}
                enriched = _enrich_body_from_otlp_stdout(
                    generic_body,
                    pod_logs,
                    run_uid=run_uid,
                    phase=phase,
                )

                assert enriched == {**generic_body, **output}
                assert (
                    _parse_echo_token_from_otlp_stdout(
                        pod_logs,
                        run_uid=run_uid,
                        phase=phase,
                    )
                    == echo_token
                )
                log_body = b"\xff"
                with pytest.raises(UnicodeDecodeError):
                    _fetch_job_pod_logs(core_api, "e2e", "batch-job")
            finally:
                try:
                    api_client.close()
                finally:
                    server.shutdown()
                    server_thread.join()


class TestLoadBatchE2EConfig:
    def test_openai_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("E2E_PROVIDER", "openai-agents")
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        monkeypatch.delenv("LIGHTSPEED_MCP_SERVERS", raising=False)
        monkeypatch.delenv("LIGHTSPEED_REASONING_CONFIG", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        expected_secret = resolve_llm_secret("openai-agents")
        config = load_batch_e2e_config()
        assert config.lightspeed_provider == "openai"
        assert config.llm_secret == expected_secret
        assert config.model == "gpt-6-luna"
        assert config.verify_full_fixtures is False
        assert config.job_env == {
            "LIGHTSPEED_TLS_PROFILE": "IntermediateType",
            "LIGHTSPEED_TLS_MIN_VERSION": "VersionTLS12",
            "LIGHTSPEED_TLS_CIPHER_SUITES": '["ECDHE-RSA-AES128-GCM-SHA256"]',
        }

    def test_anthropic_vertex_maps_to_vertex_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("E2E_PROVIDER", "anthropic-vertex-deepagents")
        monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "my-gcp-project")
        monkeypatch.setenv("CLOUD_ML_REGION", "us-central1")

        config = load_batch_e2e_config()
        assert config.lightspeed_provider == "vertex"
        assert config.extra_env == {
            "LIGHTSPEED_MODEL_PROVIDER": "anthropic",
            "LIGHTSPEED_PROVIDER_PROJECT": "my-gcp-project",
            "LIGHTSPEED_PROVIDER_REGION": "us-central1",
        }
        assert config.llm_secret == resolve_llm_secret("anthropic-vertex-deepagents")

    def test_anthropic_bedrock_maps_to_bedrock_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("E2E_PROVIDER", "anthropic-bedrock-deepagents")
        monkeypatch.setenv("AWS_REGION", "us-east-1")

        config = load_batch_e2e_config()
        assert config.lightspeed_provider == "bedrock"
        assert config.extra_env == {"LIGHTSPEED_PROVIDER_REGION": "us-east-1"}
        assert config.llm_secret == resolve_llm_secret("anthropic-bedrock-deepagents")


class TestResolveHelpers:
    def test_resolve_model_env_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_MODEL", "custom-model")
        assert resolve_model("openai-agents") == "custom-model"

    def test_resolve_llm_secret_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_SECRET", "my-secret")
        assert resolve_llm_secret("openai-agents") == "my-secret"


class TestBuildJobSpec:
    def _config(self, *, job_env: dict[str, str] | None = None) -> BatchE2EConfig:
        llm_secret = resolve_llm_secret("openai-agents")
        return BatchE2EConfig(
            namespace="ns",
            sandbox_image="img:tag",
            service_account="sa",
            llm_secret=llm_secret,
            lightspeed_provider="openai",
            model="gpt-5-mini",
            provider_name="openai-agents",
            session_id="sess",
            otel_endpoint="",
            otel_ca_secret="",
            verify_full_fixtures=False,
            job_env=job_env or {},
        )

    def _env(self, job: Any) -> dict[str, str]:
        env = job.spec["template"]["spec"]["containers"][0]["env"]
        return {item["name"]: item["value"] for item in env}

    def test_session_job_env_does_not_enable_tool_output_inspection(self) -> None:
        assert "LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED" not in _session_job_env()

    def test_sets_required_execution_limit_env_defaults(self) -> None:
        job = _build_job_spec(
            self._config(),
            "job-name",
            "input-cm",
            {"app": "test"},
            "run-uid",
            "analysis",
        )

        env = self._env(job)

        assert env["LIGHTSPEED_AGENT_TIMEOUT_SECONDS"] == "600"
        assert env["LIGHTSPEED_AGENT_MAX_TURNS"] == "200"
        assert env["LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED"] == "false"

    def test_applies_job_env_override_without_mutating_config(self) -> None:
        config = self._config()
        job = _build_job_spec(
            config,
            "job-name",
            "input-cm",
            {"app": "test"},
            "run-uid",
            "analysis",
            job_env_overrides={"LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED": "true"},
        )

        assert self._env(job)["LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED"] == "true"
        assert "LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED" not in config.job_env

    def test_timeout_ms_override_rounds_up_to_seconds(self) -> None:
        job = _build_job_spec(
            self._config(),
            "job-name",
            "input-cm",
            {"app": "test"},
            "run-uid",
            "analysis",
            timeout_ms=1500,
        )

        assert self._env(job)["LIGHTSPEED_AGENT_TIMEOUT_SECONDS"] == "2"

    def test_passes_tls_policy_to_job(self) -> None:
        job = _build_job_spec(
            self._config(
                job_env={
                    "LIGHTSPEED_TLS_PROFILE": "IntermediateType",
                    "LIGHTSPEED_TLS_MIN_VERSION": "VersionTLS12",
                    "LIGHTSPEED_TLS_CIPHER_SUITES": '["ECDHE-RSA-AES128-GCM-SHA256"]',
                }
            ),
            "job-name",
            "input-cm",
            {"app": "test"},
            "run-uid",
            "analysis",
        )

        env = self._env(job)

        assert env["LIGHTSPEED_TLS_PROFILE"] == "IntermediateType"
        assert env["LIGHTSPEED_TLS_MIN_VERSION"] == "VersionTLS12"
        assert env["LIGHTSPEED_TLS_CIPHER_SUITES"] == '["ECDHE-RSA-AES128-GCM-SHA256"]'

    def test_job_env_overrides_execution_limit_defaults(self) -> None:
        job = _build_job_spec(
            self._config(
                job_env={
                    "LIGHTSPEED_AGENT_TIMEOUT_SECONDS": "42",
                    "LIGHTSPEED_AGENT_MAX_TURNS": "7",
                    "LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED": "true",
                }
            ),
            "job-name",
            "input-cm",
            {"app": "test"},
            "run-uid",
            "analysis",
        )

        env = self._env(job)

        assert env["LIGHTSPEED_AGENT_TIMEOUT_SECONDS"] == "42"
        assert env["LIGHTSPEED_AGENT_MAX_TURNS"] == "7"
        assert env["LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED"] == "true"


class TestRunBatchQuery:
    def test_job_create_failure_deletes_input_config_map(self) -> None:
        llm_secret = resolve_llm_secret("openai-agents")
        config = BatchE2EConfig(
            namespace="ns",
            sandbox_image="img:tag",
            service_account="sa",
            llm_secret=llm_secret,
            lightspeed_provider="openai",
            model="gpt-5-mini",
            provider_name="openai-agents",
            session_id="sess",
            otel_endpoint="",
            otel_ca_secret="",
            verify_full_fixtures=False,
        )
        core_api = MagicMock()
        batch_api = MagicMock()
        custom_api = MagicMock()
        batch_api.create_namespaced_job.side_effect = ApiException(status=403, reason="Forbidden")

        result = run_batch_query(
            config,
            core_api,
            batch_api,
            custom_api,
            "hello",
        )

        assert result.error is not None
        assert "Forbidden" in result.error
        core_api.delete_namespaced_config_map.assert_called_once()
        delete_args = core_api.delete_namespaced_config_map.call_args[0]
        assert delete_args[0].endswith("-input")
        assert delete_args[1] == "ns"

    def test_missing_result_cr_after_successful_job(self) -> None:
        llm_secret = resolve_llm_secret("openai-agents")
        config = BatchE2EConfig(
            namespace="ns",
            sandbox_image="img:tag",
            service_account="sa",
            llm_secret=llm_secret,
            lightspeed_provider="openai",
            model="gpt-5-mini",
            provider_name="openai-agents",
            session_id="sess",
            otel_endpoint="",
            otel_ca_secret="",
            verify_full_fixtures=False,
        )
        core_api = MagicMock()
        batch_api = MagicMock()
        custom_api = MagicMock()

        job_status = MagicMock()
        job_status.succeeded = 1
        job_status.failed = None
        batch_api.create_namespaced_job.return_value = MagicMock(
            metadata=MagicMock(uid="job-uid-abc"),
        )
        batch_api.read_namespaced_job.return_value = MagicMock(status=job_status)
        core_api.list_namespaced_pod.return_value = MagicMock(items=[])
        custom_api.get_namespaced_custom_object.side_effect = ApiException(status=404)

        result = run_batch_query(
            config,
            core_api,
            batch_api,
            custom_api,
            "hello",
        )

        assert result.job_succeeded is True
        assert result.result_cr is None
        assert result.error is not None
        assert "not found" in result.error
        core_api.replace_namespaced_config_map.assert_called_once()


class TestConfigMapJobOwner:
    def test_set_config_map_job_owner(self) -> None:
        core_api = MagicMock()
        cm = MagicMock()
        cm.metadata.owner_references = None
        core_api.read_namespaced_config_map.return_value = cm

        _set_config_map_job_owner(core_api, "ns", "e2e-input", "e2e-job", "uid-123")

        core_api.replace_namespaced_config_map.assert_called_once_with(
            "e2e-input",
            "ns",
            cm,
        )
        assert cm.metadata.owner_references is not None
        owner = cm.metadata.owner_references[0]
        assert owner.api_version == "batch/v1"
        assert owner.kind == "Job"
        assert owner.name == "e2e-job"
        assert owner.uid == "uid-123"
        assert owner.controller is True

    def test_delete_config_map_ignore_not_found(self) -> None:
        core_api = MagicMock()
        _delete_config_map_ignore_not_found(core_api, "ns", "orphan-cm")
        core_api.delete_namespaced_config_map.assert_called_once_with("orphan-cm", "ns")

    def test_delete_config_map_swallows_404(self) -> None:
        core_api = MagicMock()
        core_api.delete_namespaced_config_map.side_effect = ApiException(status=404)
        _delete_config_map_ignore_not_found(core_api, "ns", "missing-cm")
