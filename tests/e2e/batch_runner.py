"""Batch Job runner for E2E — cluster Job lifecycle (OLS-3926).

Maps each scenario to a batch Job + Result CR, then builds a response envelope
for BDD steps. Generic Result CR status is enriched from the completed agent
source span in the batch pod's OTLP JSON stdout.
"""

from __future__ import annotations

import json
import posixpath
import re
import secrets
import shlex
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from kubernetes.client import (  # type: ignore[import-untyped]
    ApiException,
    BatchV1Api,
    CoreV1Api,
    CustomObjectsApi,
    V1ConfigMap,
    V1Job,
    V1ObjectMeta,
    V1OwnerReference,
)

from tests.e2e.k8s_constants import CRD_GROUP, CRD_VERSION
from tests.e2e.skills_fixtures import (
    E2E_POD_OUTPUT_DIR,
    E2E_POD_SKILLS_DIR,
    E2E_POD_SKILLS_SRC_DIR,
    E2E_POD_SKILLS_WORKDIR,
    SKILLS_SOURCE,
    configmap_items_for_skill,
    ensure_skill_configmaps,
    skill_materialize_script,
)
from tests.e2e.suite_setup import (
    BatchE2EConfig,
    E2E_COMPONENT_LABEL,
    E2E_COMPONENT_VALUE,
    E2E_RUN_LABEL,
)

_KIND_TO_PLURAL: dict[str, str] = {
    "AnalysisResult": "analysisresults",
    "ExecutionResult": "executionresults",
    "VerificationResult": "verificationresults",
    "EscalationResult": "escalationresults",
}

_STEP_TO_KIND: dict[str, str] = {
    "analysis": "AnalysisResult",
    "execution": "ExecutionResult",
    "verification": "VerificationResult",
    "escalation": "EscalationResult",
}

E2E_DEFAULT_AGENT_TIMEOUT_SECONDS = 600
E2E_DEFAULT_AGENT_MAX_TURNS = "200"
_GENERIC_CR_SUMMARIES = frozenset({"step completed", "step failed"})


@dataclass
class RunBatchResult:
    """Outcome of a single batch Job run."""

    job_succeeded: bool = False
    result_cr: dict[str, Any] | None = None
    pod_logs: str = ""
    termination_message: str | None = None
    error: str | None = None
    latency_seconds: float = 0.0
    job_name: str = ""
    result_name: str | None = None
    run_uid: str = ""
    step: str = "analysis"
    body: dict[str, Any] = field(default_factory=dict)
    tool_token: str = ""

    @property
    def agent_succeeded(self) -> bool:
        """True when the agent step succeeded (Completed.reason=Succeeded)."""
        return bool(self.body.get("success"))


def build_result_template(
    *,
    namespace: str,
    result_name: str,
    run_uid: str,
    step: str,
    agentic_run_name: str,
    session_id: str,
) -> dict[str, Any]:
    """Build a Result CR template for ``/input/result-template``."""
    kind = _STEP_TO_KIND.get(step)
    if kind is None:
        msg = f"unknown batch step: {step!r}"
        raise ValueError(msg)
    return {
        "apiVersion": f"{CRD_GROUP}/{CRD_VERSION}",
        "kind": kind,
        "metadata": {
            "name": result_name,
            "namespace": namespace,
            "labels": {
                "agentic.openshift.io/run": run_uid,
                "agentic.openshift.io/step": step,
                E2E_COMPONENT_LABEL: E2E_COMPONENT_VALUE,
                E2E_RUN_LABEL: session_id,
            },
        },
        "spec": {"agenticRunName": agentic_run_name},
    }


def _delete_config_map_ignore_not_found(
    core_api: CoreV1Api,
    namespace: str,
    name: str,
) -> None:
    """Delete an input ConfigMap; ignore 404 when create partially failed."""
    try:
        core_api.delete_namespaced_config_map(name, namespace)
    except ApiException as exc:
        if exc.status != 404:
            raise


def _set_config_map_job_owner(
    core_api: CoreV1Api,
    namespace: str,
    config_map_name: str,
    job_name: str,
    job_uid: str,
) -> None:
    """Tie input ConfigMap lifecycle to the batch Job (GC when Job TTL deletes it)."""
    cm = core_api.read_namespaced_config_map(config_map_name, namespace)
    cm.metadata.owner_references = [  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
        V1OwnerReference(
            api_version="batch/v1",
            kind="Job",
            name=job_name,
            uid=job_uid,
            controller=True,
            block_owner_deletion=False,
        )
    ]
    core_api.replace_namespaced_config_map(config_map_name, namespace, cm)


def run_batch_query(
    config: BatchE2EConfig,
    core_api: CoreV1Api,
    batch_api: BatchV1Api,
    custom_api: CustomObjectsApi,
    query: str,
    *,
    system_prompt: str | None = None,
    output_schema: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    step: str = "analysis",
    wait_timeout_seconds: float = 600.0,
    job_name_prefix: str | None = None,
    timeout_ms: int | None = None,
    mount_skills: bool = False,
    job_env_overrides: Mapping[str, str] | None = None,
) -> RunBatchResult:
    """Create input ConfigMap + batch Job, wait for completion, read Result CR."""
    start = time.monotonic()
    run_uid = secrets.token_hex(16)
    stamp = int(time.time())
    job_name = _sanitize_k8s_name(f"{job_name_prefix or 'e2e'}-{stamp}-{run_uid[:8]}")
    input_cm_name = f"{job_name}-input"
    result_name = _sanitize_k8s_name(f"{job_name}-{step}-1")

    if context is None:
        context = {"targetNamespaces": ["default"]}

    skill_configmaps: dict[str, str] | None = None
    if mount_skills:
        skill_configmaps = ensure_skill_configmaps(
            core_api,
            config.namespace,
            config.session_id,
        )

    result_template = build_result_template(
        namespace=config.namespace,
        result_name=result_name,
        run_uid=run_uid,
        step=step,
        agentic_run_name=job_name,
        session_id=config.session_id,
    )

    labels = {
        E2E_RUN_LABEL: config.session_id,
        E2E_COMPONENT_LABEL: E2E_COMPONENT_VALUE,
        "agentic.openshift.io/run": run_uid,
        "agentic.openshift.io/step": step,
    }

    cm_data: dict[str, str] = {
        "query": query,
        "output-schema": json.dumps({} if output_schema is None else output_schema),
        "context": json.dumps(context),
        "result-template": json.dumps(result_template),
    }
    if system_prompt:
        cm_data["system-prompt"] = system_prompt

    result = RunBatchResult(job_name=job_name, result_name=result_name, run_uid=run_uid, step=step)

    cm_created = False
    try:
        core_api.create_namespaced_config_map(
            config.namespace,
            V1ConfigMap(
                metadata=V1ObjectMeta(name=input_cm_name, labels=labels),
                data=cm_data,
            ),
        )
        cm_created = True
        created_job = batch_api.create_namespaced_job(
            config.namespace,
            _build_job_spec(
                config,
                job_name,
                input_cm_name,
                labels,
                run_uid,
                step,
                timeout_ms=timeout_ms,
                skill_configmaps=skill_configmaps,
                job_env_overrides=job_env_overrides,
            ),
        )
        job_uid = created_job.metadata.uid  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
        if job_uid:
            _set_config_map_job_owner(
                core_api,
                config.namespace,
                input_cm_name,
                job_name,
                job_uid,
            )
    except ApiException as exc:
        if cm_created:
            _delete_config_map_ignore_not_found(core_api, config.namespace, input_cm_name)
        result.error = f"create batch resources: {exc.reason}"
        result.latency_seconds = time.monotonic() - start
        return result

    job_ok, wait_err = _wait_for_job(
        batch_api,
        core_api,
        config.namespace,
        job_name,
        wait_timeout_seconds,
    )
    result.job_succeeded = job_ok
    result.pod_logs = _fetch_job_pod_logs(core_api, config.namespace, job_name)
    result.termination_message = _fetch_termination_message(core_api, config.namespace, job_name)
    if mount_skills:
        result.tool_token = _parse_echo_token_from_otlp_stdout(
            result.pod_logs, run_uid=result.run_uid, phase=result.step
        )

    if not job_ok:
        result.error = wait_err or "batch job did not succeed"
        result.latency_seconds = time.monotonic() - start
        return result

    kind = result_template["kind"]
    try:
        result.result_cr = custom_api.get_namespaced_custom_object(  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
            group=CRD_GROUP,
            version=CRD_VERSION,
            namespace=config.namespace,
            plural=_KIND_TO_PLURAL[kind],
            name=result_name,
        )
    except ApiException as exc:
        if exc.status == 404:
            result.error = f"Result CR {kind}/{result_name} not found after successful Job"
        else:
            result.error = f"read Result CR: {exc.reason}"
        result.latency_seconds = time.monotonic() - start
        return result

    result.body = _body_from_result_cr(result.result_cr)  # pyright: ignore[reportArgumentType]
    result.body = _enrich_body_from_otlp_stdout(
        result.body,
        result.pod_logs,
        run_uid=result.run_uid,
        phase=result.step,
    )
    result.latency_seconds = time.monotonic() - start
    return result


def _needs_agent_result_enrichment(body: dict[str, Any]) -> bool:
    """Return True only when Result CR status has a generic response envelope."""
    if body.get("failureReason") or body.get("success") is False:
        return False
    if any(
        key in body
        for key in ("options", "actionRequired", "diagnosis", "actionsTaken", "checks", "content")
    ):
        return False
    summary = str(body.get("summary", "")).strip().lower()
    return summary in _GENERIC_CR_SUMMARIES or not summary


def _iter_otlp_spans(pod_logs: str) -> Iterator[dict[str, Any]]:
    """Yield spans from the OTLP-JSON requests emitted on separate stdout lines."""
    for line in pod_logs.splitlines():
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(request, dict):
            continue
        resource_spans = request.get("resource_spans")
        if not isinstance(resource_spans, list):
            continue
        for resource_span in resource_spans:
            if not isinstance(resource_span, dict):
                continue
            scope_spans = resource_span.get("scope_spans")
            if not isinstance(scope_spans, list):
                continue
            for scope_span in scope_spans:
                if not isinstance(scope_span, dict):
                    continue
                spans = scope_span.get("spans")
                if isinstance(spans, list):
                    yield from (span for span in spans if isinstance(span, dict))


def _otel_string_attributes(span: dict[str, Any]) -> dict[str, str]:
    """Read only string-typed attributes from an OTLP-JSON Span."""
    attributes: dict[str, str] = {}
    raw_attributes = span.get("attributes")
    if not isinstance(raw_attributes, list):
        return attributes
    for attribute in raw_attributes:
        if not isinstance(attribute, dict):
            continue
        key = attribute.get("key")
        value = attribute.get("value")
        if isinstance(key, str) and isinstance(value, dict):
            string_value = value.get("string_value")
            if isinstance(string_value, str):
                attributes[key] = string_value
    return attributes


def _span_has_error(span: dict[str, Any]) -> bool:
    status = span.get("status")
    return isinstance(status, dict) and status.get("code") in (
        "STATUS_CODE_ERROR",
        "ERROR",
        2,
    )


def _parse_agent_result_from_otlp_stdout(
    pod_logs: str,
    *,
    run_uid: str,
    phase: str,
) -> dict[str, Any] | None:
    """Decode the correlated invoke-agent span's assistant text AgentResult JSON."""
    for span in _iter_otlp_spans(pod_logs):
        attributes = _otel_string_attributes(span)
        if (
            span.get("name") != "invoke_agent lightspeed"
            or attributes.get("gen_ai.operation.name") != "invoke_agent"
            or attributes.get("gen_ai.agent.name") != "lightspeed"
            or attributes.get("agenticrun.uid") != run_uid
            or attributes.get("agenticrun.phase") != phase
            or _span_has_error(span)
        ):
            continue
        messages_json = attributes.get("gen_ai.output.messages")
        if messages_json is None:
            continue
        try:
            messages = json.loads(messages_json)
        except json.JSONDecodeError:
            continue
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            parts = message.get("parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict) or part.get("type") != "text":
                    continue
                content = part.get("content")
                if not isinstance(content, str):
                    continue
                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    continue
                if isinstance(result, dict):
                    return result
    return None


def _enrich_body_from_otlp_stdout(
    body: dict[str, Any],
    pod_logs: str,
    *,
    run_uid: str,
    phase: str,
) -> dict[str, Any]:
    """Merge a correlated completed agent result only into a generic CR body."""
    if not _needs_agent_result_enrichment(body):
        return body
    parsed = _parse_agent_result_from_otlp_stdout(
        pod_logs,
        run_uid=run_uid,
        phase=phase,
    )
    if parsed is None:
        return body
    enriched = dict(body)
    enriched.update(parsed)
    return enriched


# echo-token.sh stdout: {"token": "<32 hex>", "status": "ok"}
_DEEPAGENTS_SUCCESS_TRAILER = "\n\n[Command succeeded with exit code 0]"
_OPENAI_EXEC_SUCCESS_RESULT_RE = re.compile(
    r"\AChunk ID: [0-9a-f]{6}\n"
    r"Wall time: [0-9]+\.[0-9]{4} seconds\n"
    r"Process exited with code 0\n"
    r"(?:Original token count: [0-9]+\n)?"
    r"Output:\n"
    r"(?:PTY transport failed before the interactive session opened; "
    r"fell back to one-shot exec\.\n)?"
    r"(?P<output>.*)\Z",
    re.DOTALL,
)
_ECHO_TOKEN_HEX_RE = re.compile(r"^[0-9a-f]{32}$")

_SHELL_TOOL_ARGUMENT_KEYS = {
    "execute": "command",
    "execute_bash": "command",
    "exec_command": "cmd",
}
_SHELL_TOOL_DEFAULT_CWDS = {
    "execute": E2E_POD_SKILLS_DIR,
    "execute_bash": E2E_POD_SKILLS_DIR,
    "exec_command": posixpath.dirname(E2E_POD_SKILLS_DIR),
}
_ECHO_TOKEN_SKILL_PATHS = frozenset(
    posixpath.join(skills_root, "echo-token")
    for skills_root in (E2E_POD_SKILLS_DIR, E2E_POD_SKILLS_WORKDIR)
)
_ECHO_TOKEN_SCRIPT_PATHS = frozenset(
    posixpath.join(skill_dir, "scripts", "echo-token.sh") for skill_dir in _ECHO_TOKEN_SKILL_PATHS
)
_ECHO_TOKEN_SKILL_INSTRUCTION_PATHS = frozenset(
    posixpath.join(skill_dir, "SKILL.md") for skill_dir in _ECHO_TOKEN_SKILL_PATHS
)


def _tool_command_invokes_echo_token(
    tool_name: str | None,
    arguments_json: str | None,
) -> bool:
    """Match only a native shell command that executes the fixture script."""
    if tool_name is None or not isinstance(arguments_json, str):
        return False
    argument_key = _SHELL_TOOL_ARGUMENT_KEYS.get(tool_name)
    if argument_key is None:
        return False
    try:
        arguments = json.loads(arguments_json)
    except json.JSONDecodeError:
        return False
    if not isinstance(arguments, dict):
        return False
    command = arguments.get(argument_key)
    if not isinstance(command, str) or not command.strip():
        return False

    cwd = _SHELL_TOOL_DEFAULT_CWDS[tool_name]
    if tool_name == "exec_command":
        workdir = arguments.get("workdir")
        if workdir is not None:
            if not isinstance(workdir, str):
                return False
            if workdir:
                cwd = posixpath.normpath(
                    workdir if workdir.startswith("/") else posixpath.join(cwd, workdir)
                )

    # Split only unquoted top-level &&; reject other shell control syntax.
    parts: list[str] = []
    start = 0
    quote: str | None = None
    escaped = False
    position = 0
    while position < len(command):
        char = command[position]
        if escaped:
            escaped = False
        elif quote == "'":
            if char == "'":
                quote = None
        elif quote == '"':
            if char == "\\":
                escaped = True
            elif char == '"':
                quote = None
        elif char == "\\":
            escaped = True
        elif char in {"'", '"'}:
            quote = char
        elif char == "&":
            if command[position : position + 2] != "&&":
                return False
            part = command[start:position].strip()
            if not part:
                return False
            parts.append(part)
            position += 2
            start = position
            continue
        elif char in "|;()\n":
            return False
        position += 1
    if quote is not None or escaped:
        return False
    part = command[start:].strip()
    if not part:
        return False
    parts.append(part)

    try:
        commands = [shlex.split(part, posix=True) for part in parts]
    except ValueError:
        return False
    if any(not command_tokens for command_tokens in commands):
        return False

    command_index = 0
    if commands[0][0] == "cd":
        directory_command = commands[0]
        if len(directory_command) != 2:
            return False
        directory = directory_command[1]
        cwd = posixpath.normpath(
            directory if directory.startswith("/") else posixpath.join(cwd, directory)
        )
        if cwd not in _ECHO_TOKEN_SKILL_PATHS:
            return False
        command_index += 1

    if command_index < len(commands) and commands[command_index][0] == "cat":
        cat_command = commands[command_index]
        if len(cat_command) != 2:
            return False
        instruction = cat_command[1]
        instruction_path = (
            instruction if instruction.startswith("/") else posixpath.join(cwd, instruction)
        )
        if posixpath.normpath(instruction_path) not in _ECHO_TOKEN_SKILL_INSTRUCTION_PATHS:
            return False
        command_index += 1

    if command_index != len(commands) - 1:
        return False
    invocation = commands[command_index]
    if len(invocation) != 2 or invocation[0] not in {"bash", "/bin/bash"}:
        return False
    script = invocation[1]
    script_path = script if script.startswith("/") else posixpath.join(cwd, script)
    return posixpath.normpath(script_path) in _ECHO_TOKEN_SCRIPT_PATHS


def _parse_echo_token_from_otlp_stdout(
    pod_logs: str,
    *,
    run_uid: str,
    phase: str,
) -> str:
    """Return a token only from a correlated successful echo-token tool result."""
    matches = []
    for span in _iter_otlp_spans(pod_logs):
        attributes = _otel_string_attributes(span)
        tool_name = attributes.get("gen_ai.tool.name")
        if (
            attributes.get("gen_ai.operation.name") != "execute_tool"
            or attributes.get("agenticrun.uid") != run_uid
            or attributes.get("agenticrun.phase") != phase
            or not tool_name
            or _span_has_error(span)
        ):
            continue
        if not _tool_command_invokes_echo_token(
            tool_name,
            attributes.get("gen_ai.tool.call.arguments"),
        ):
            continue
        result_json = attributes.get("gen_ai.tool.call.result")
        if result_json is None:
            continue
        try:
            result = json.loads(result_json)
        except json.JSONDecodeError:
            continue
        token = _echo_token_from_tool_result(result)
        if token:
            matches.append(token)
    return matches[-1] if matches else ""


def _echo_token_from_tool_result(value: Any) -> str:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            openai_match = _OPENAI_EXEC_SUCCESS_RESULT_RE.fullmatch(value)
            if openai_match is not None:
                output = openai_match.group("output")
                try:
                    decoded = json.loads(output)
                except json.JSONDecodeError:
                    token = ""
                    for line in output.splitlines():
                        try:
                            line_result = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(line_result, dict):
                            token = _echo_token_from_tool_result(line_result) or token
                    return token
            else:
                text = value.lstrip()
                try:
                    decoded, end = json.JSONDecoder().raw_decode(text)
                except json.JSONDecodeError:
                    return ""
                if text[end:] != _DEEPAGENTS_SUCCESS_TRAILER:
                    return ""
        return _echo_token_from_tool_result(decoded)
    if not isinstance(value, dict):
        return ""
    for key in ("exit_code", "returncode"):
        if key in value and value[key] != 0:
            return ""
    if value.get("error"):
        return ""
    token = value.get("token")
    if (
        value.get("status") == "ok"
        and isinstance(token, str)
        and _ECHO_TOKEN_HEX_RE.fullmatch(token)
    ):
        return token
    if "stdout" in value:
        return _echo_token_from_tool_result(value["stdout"])
    return ""


def _body_from_result_cr(result_cr: dict[str, Any]) -> dict[str, Any]:
    """Map Result CR status to a response body for BDD assertions."""
    status = result_cr.get("status") or {}
    body: dict[str, Any] = {}

    for key in (
        "options",
        "actionRequired",
        "diagnosis",
        "actionsTaken",
        "checks",
        "summary",
        "content",
    ):
        if key in status:
            body[key] = status[key]

    failure_reason = status.get("failureReason")
    if failure_reason:
        body["success"] = False
        body["summary"] = failure_reason
        return body

    completed = _condition(status.get("conditions") or [], "Completed")
    if completed and completed.get("reason") == "Failed":
        body["success"] = False
        body["summary"] = completed.get("message") or "step failed"
        return body

    body["success"] = True
    if "summary" not in body:
        diagnosis = body.get("diagnosis")
        if isinstance(diagnosis, dict) and diagnosis.get("summary"):
            body["summary"] = diagnosis["summary"]
        else:
            body["summary"] = completed.get("message") if completed else "step completed"

    diagnosis = body.get("diagnosis")
    if isinstance(diagnosis, dict):
        for key, value in diagnosis.items():
            body.setdefault(key, value)

    return body


def _condition(conditions: list[dict[str, Any]], cond_type: str) -> dict[str, Any] | None:
    for cond in conditions:
        if cond.get("type") == cond_type:
            return cond
    return None


def _job_container_security_context() -> dict[str, Any]:
    return {
        "allowPrivilegeEscalation": False,
        "runAsNonRoot": True,
        "capabilities": {"drop": ["ALL"]},
        "seccompProfile": {"type": "RuntimeDefault"},
    }


def _build_job_spec(
    config: BatchE2EConfig,
    job_name: str,
    input_cm_name: str,
    labels: dict[str, str],
    run_uid: str,
    step: str,
    timeout_ms: int | None = None,
    skill_configmaps: dict[str, str] | None = None,
    job_env_overrides: Mapping[str, str] | None = None,
) -> V1Job:
    otel_enabled = bool(config.otel_endpoint)
    env = [
        {"name": "LIGHTSPEED_PROVIDER", "value": config.lightspeed_provider},
        {"name": "LIGHTSPEED_MODEL", "value": config.model},
        {"name": "LIGHTSPEED_AGENTICRUN_UID", "value": run_uid},
        {"name": "LIGHTSPEED_AGENTICRUN_STEP", "value": step},
    ]
    for key, value in config.extra_env.items():
        env.append({"name": key, "value": value})
    job_env = {**config.job_env, **(job_env_overrides or {})}
    for key, value in job_env.items():
        env.append({"name": key, "value": value})
    for key in ("LIGHTSPEED_AUDIT_ENABLED", "LIGHTSPEED_CAPTURE_CONTENT"):
        env = [item for item in env if item["name"] != key]
        env.append({"name": key, "value": "true"})

    env_names = {item["name"] for item in env}
    if "LIGHTSPEED_AGENT_TIMEOUT_SECONDS" not in env_names:
        timeout_seconds = E2E_DEFAULT_AGENT_TIMEOUT_SECONDS
        if timeout_ms is not None:
            timeout_seconds = max(1, (timeout_ms + 999) // 1000)
        env.append({"name": "LIGHTSPEED_AGENT_TIMEOUT_SECONDS", "value": str(timeout_seconds)})
    if "LIGHTSPEED_AGENT_MAX_TURNS" not in env_names:
        env.append({"name": "LIGHTSPEED_AGENT_MAX_TURNS", "value": E2E_DEFAULT_AGENT_MAX_TURNS})
    if "LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED" not in env_names:
        env.append({"name": "LIGHTSPEED_TOOL_OUTPUT_INSPECTION_ENABLED", "value": "false"})
    if otel_enabled:
        env.extend(
            [
                {"name": "OTEL_EXPORTER_OTLP_ENDPOINT", "value": config.otel_endpoint},
                {"name": "OTEL_EXPORTER_OTLP_PROTOCOL", "value": "grpc"},
            ]
        )

    volumes: list[dict[str, Any]] = [
        {"name": "input", "configMap": {"name": input_cm_name}},
        {"name": "llm-credentials", "secret": {"secretName": config.llm_secret}},
    ]
    volume_mounts: list[dict[str, Any]] = [
        {"name": "input", "mountPath": "/input", "readOnly": True},
        {
            "name": "llm-credentials",
            "mountPath": "/var/run/secrets/llm-credentials",
            "readOnly": True,
        },
    ]
    if otel_enabled and config.otel_ca_secret:
        volumes.append({"name": "otel-ca", "secret": {"secretName": config.otel_ca_secret}})
        volume_mounts.append(
            {
                "name": "otel-ca",
                "mountPath": "/var/run/secrets/lightspeed/tls/otel-ca",
                "readOnly": True,
            }
        )
    init_containers: list[dict[str, Any]] = []
    init_volume_mounts: list[dict[str, Any]] = []
    if skill_configmaps:
        volumes.append({"name": "skills-root", "emptyDir": {}})
        volume_mounts.append({"name": "skills-root", "mountPath": E2E_POD_SKILLS_DIR})
        for skill_name, cm_name in skill_configmaps.items():
            vol_name = _sanitize_k8s_name(f"skill-{skill_name}")
            volumes.append(
                {
                    "name": vol_name,
                    "configMap": {
                        "name": cm_name,
                        "items": configmap_items_for_skill(SKILLS_SOURCE / skill_name),
                        "defaultMode": 0o555,
                    },
                }
            )
            init_volume_mounts.append(
                {
                    "name": vol_name,
                    "mountPath": f"{E2E_POD_SKILLS_SRC_DIR}/{skill_name}",
                    "readOnly": True,
                }
            )
        init_containers.append(
            {
                "name": "materialize-skills",
                "image": config.sandbox_image,
                "imagePullPolicy": "IfNotPresent",
                "command": ["bash", "-c"],
                "args": [skill_materialize_script()],
                "volumeMounts": [
                    *init_volume_mounts,
                    {"name": "skills-root", "mountPath": E2E_POD_SKILLS_DIR},
                ],
                "securityContext": _job_container_security_context(),
            }
        )
        volumes.append({"name": "e2e-output", "emptyDir": {}})
        volume_mounts.append({"name": "e2e-output", "mountPath": E2E_POD_OUTPUT_DIR})
        env.extend(
            [
                {"name": "LIGHTSPEED_SKILLS_DIR", "value": E2E_POD_SKILLS_DIR},
                {"name": "E2E_OUTPUT_DIR", "value": E2E_POD_OUTPUT_DIR},
            ]
        )

    pod_spec: dict[str, Any] = {
        "serviceAccountName": config.service_account,
        "automountServiceAccountToken": True,
        "restartPolicy": "Never",
        "securityContext": {
            "runAsNonRoot": True,
            "seccompProfile": {"type": "RuntimeDefault"},
        },
        "volumes": volumes,
        "containers": [
            {
                "name": "agent",
                "image": config.sandbox_image,
                "imagePullPolicy": "IfNotPresent",
                "terminationMessagePolicy": "FallbackToLogsOnError",
                "securityContext": _job_container_security_context(),
                "envFrom": [{"secretRef": {"name": config.llm_secret}}],
                "env": env,
                "volumeMounts": volume_mounts,
            }
        ],
    }
    if init_containers:
        pod_spec["initContainers"] = init_containers

    return V1Job(
        metadata=V1ObjectMeta(name=job_name, labels=labels),
        spec={
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 3600,
            "template": {"metadata": {"labels": labels}, "spec": pod_spec},
        },
    )


def _wait_for_job(
    batch_api: BatchV1Api,
    core_api: CoreV1Api,
    namespace: str,
    job_name: str,
    timeout_seconds: float,
) -> tuple[bool, str | None]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        job = batch_api.read_namespaced_job(job_name, namespace)
        status = job.status  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
        if status and status.succeeded:
            return True, None
        if status and status.failed:
            msg = _fetch_termination_message(core_api, namespace, job_name)
            return False, msg or "job failed"
        time.sleep(2.0)
    return False, f"timeout after {timeout_seconds}s waiting for job/{job_name}"


def _fetch_job_pod_logs(core_api: CoreV1Api, namespace: str, job_name: str) -> str:
    pods = core_api.list_namespaced_pod(namespace=namespace, label_selector=f"job-name={job_name}")
    if not pods.items:  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
        return ""
    pod_name = pods.items[0].metadata.name  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess, reportIndexIssue]
    try:
        response = core_api.read_namespaced_pod_log(
            name=pod_name,
            namespace=namespace,
            tail_lines=200,
            _preload_content=False,
        )  # pyright: ignore[reportArgumentType]
    except ApiException:
        return ""
    try:
        return response.data.decode("utf-8")
    finally:
        response.release_conn()


def _fetch_termination_message(core_api: CoreV1Api, namespace: str, job_name: str) -> str | None:
    pods = core_api.list_namespaced_pod(namespace=namespace, label_selector=f"job-name={job_name}")
    if not pods.items:  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
        return None
    statuses = pods.items[0].status.container_statuses or []  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess, reportIndexIssue]
    if not statuses:
        return None
    terminated = statuses[0].state.terminated
    if terminated is None:
        return None
    return terminated.message  # type: ignore[no-any-return]


def _sanitize_k8s_name(value: str) -> str:
    cleaned = value.lower()
    for ch in ". _":
        cleaned = cleaned.replace(ch, "-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned[:63].rstrip("-")
