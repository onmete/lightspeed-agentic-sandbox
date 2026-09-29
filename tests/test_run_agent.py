"""Tests for run_agent_query and context formatting."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator
from langchain_core.messages import ToolMessage
from opentelemetry.trace import SpanKind, StatusCode

from lightspeed_agentic.inspection.errors import ToolResultSafetyInspectionFailed
from lightspeed_agentic.inspection.middleware import ToolResultInspectionMiddleware
from lightspeed_agentic.run_agent import ContextFormatError, format_context_prefix, run_agent_query
from lightspeed_agentic.types import ProviderEvent, ProviderQueryOptions, ResultEvent

from .conftest import MockProvider


@pytest.mark.asyncio
async def test_run_agent_query_success() -> None:
    """Mock provider completes and returns a successful structured result."""
    result = await run_agent_query(
        MockProvider(),
        prompt="Diagnose the issue",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )
    assert result.output["success"] is True
    assert "mock result" in result.output["summary"]


@pytest.mark.asyncio
async def test_run_agent_query_with_system_prompt() -> None:
    """Custom system_prompt is accepted without changing success semantics."""
    result = await run_agent_query(
        MockProvider(),
        prompt="test",
        system_prompt="Custom persona",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )
    assert result.output["success"] is True


@pytest.mark.asyncio
async def test_run_agent_query_with_context() -> None:
    """Workflow context dict is formatted and passed through to the provider."""
    result = await run_agent_query(
        MockProvider(),
        prompt="fix it",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context={
            "targetNamespaces": ["default"],
            "previousAttempts": [{"attempt": 1, "failureReason": "timeout"}],
        },
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )
    assert result.output["success"] is True


@pytest.mark.asyncio
async def test_run_agent_query_with_output_schema() -> None:
    """output_schema is forwarded to the provider query options."""
    schema = {"type": "object", "properties": {"success": {"type": "boolean"}}}
    result = await run_agent_query(
        MockProvider(),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=schema,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )
    assert result.output["success"] is True


@pytest.mark.asyncio
async def test_agent_transcript_has_actual_model_and_tool_children(
    span_exporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two observed calls and a local tool share the fixed agent parent."""
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "vertex")
    final = {"success": False, "summary": "Post-shaped", "diagnosis": {"cause": "RBAC"}}

    class ObservedProvider(MockProvider):
        async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
            assert options.telemetry is not None
            first = options.telemetry.start_model(
                [{"role": "user", "parts": [{"type": "text", "content": options.prompt}]}],
                [{"type": "text", "content": options.system_prompt}],
                options.model,
            )
            options.telemetry.end_model(
                first,
                [
                    {
                        "role": "assistant",
                        "parts": [
                            {
                                "type": "tool_call",
                                "id": "call-1",
                                "name": "read_file",
                                "arguments": {"path": "SKILL.md"},
                            },
                        ],
                        "finish_reason": "tool_call",
                    }
                ],
                None,
                {"input_tokens": 4},
                None,
            )
            tool = options.telemetry.start_tool("read_file", "call-1", {"path": "SKILL.md"})
            options.telemetry.end_tool(tool, {"content": "full skill"}, None)
            second = options.telemetry.start_model(
                [
                    {
                        "role": "tool",
                        "parts": [
                            {
                                "type": "tool_call_response",
                                "id": "call-1",
                                "response": {"content": "full skill"},
                            },
                        ],
                    }
                ],
                None,
                options.model,
            )
            options.telemetry.end_model(
                second,
                [
                    {
                        "role": "assistant",
                        "parts": [
                            {"type": "text", "content": "Intermediate different text"},
                        ],
                        "finish_reason": "stop",
                    }
                ],
                None,
                {"output_tokens": 3},
                None,
            )
            yield ResultEvent(text=json.dumps(final))

    result = await run_agent_query(
        ObservedProvider(),
        prompt="fix it",
        system_prompt="Respect approvals",
        output_schema={"type": "object"},
        context={"targetNamespaces": ["prod"]},
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
        traceparent="00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        agenticrun_uid="run-uid",
        step="execution",
    )
    assert result.output == final
    spans = span_exporter.get_finished_spans()
    agent = next(s for s in spans if s.attributes.get("gen_ai.operation.name") == "invoke_agent")
    children = [s for s in spans if s is not agent]
    assert len(children) == 3
    assert [s.attributes["gen_ai.operation.name"] for s in children].count("chat") == 2
    assert agent.kind == SpanKind.INTERNAL
    assert agent.parent.span_id == int("00f067aa0ba902b7", 16)
    assert agent.context.trace_id == int("4bf92f3577b34da6a3ce929d0e0e4736", 16)
    assert all(s.parent.span_id == agent.context.span_id for s in children)
    assert agent.attributes["gen_ai.provider.name"] == "gcp.vertex_ai"
    assert agent.attributes["gen_ai.agent.name"] == "lightspeed"
    assert agent.attributes["gen_ai.output.type"] == "json"
    assert agent.attributes["agenticrun.uid"] == "run-uid"
    assert agent.attributes["agenticrun.phase"] == "execution"
    assert agent.status.status_code == StatusCode.UNSET
    assert all(s.attributes["agenticrun.uid"] == "run-uid" for s in children)
    assert all(
        s.attributes["gen_ai.provider.name"] == "gcp.vertex_ai"
        for s in children
        if s.attributes["gen_ai.operation.name"] == "chat"
    )
    assert json.loads(agent.attributes["gen_ai.input.messages"]) == [
        {
            "role": "user",
            "parts": [
                {
                    "type": "text",
                    "content": "[context]\nTarget namespaces: prod\n[/context]\n\nfix it",
                },
            ],
        },
    ]
    assert json.loads(agent.attributes["gen_ai.system_instructions"]) == [
        {"type": "text", "content": "Respect approvals"},
    ]
    output = json.loads(agent.attributes["gen_ai.output.messages"])
    Draft202012Validator(
        json.loads(
            await asyncio.to_thread(Path("tests/fixtures/genai-v1.41-output.json").read_text)
        )
    ).validate(output)
    assert json.loads(output[0]["parts"][0]["content"]) == result.output
    assert output[0]["finish_reason"] == "stop"
    assert any("gen_ai.tool.call.result" in s.attributes for s in children)
    assert all(not s.events for s in spans)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "model_provider", "expected"),
    [
        ("vertex", "google", "gcp.vertex_ai"),
        (None, "google", "gcp.gen_ai"),
    ],
)
async def test_gemini_provider_name_reflects_endpoint(
    span_exporter,
    monkeypatch: pytest.MonkeyPatch,
    route: str | None,
    model_provider: str | None,
    expected: str,
) -> None:
    """Gemini's SDK label never replaces the configured endpoint on GenAI spans."""
    if route is None:
        monkeypatch.delenv("LIGHTSPEED_PROVIDER", raising=False)
    else:
        monkeypatch.setenv("LIGHTSPEED_PROVIDER", route)
    if model_provider is None:
        monkeypatch.delenv("LIGHTSPEED_MODEL_PROVIDER", raising=False)
    else:
        monkeypatch.setenv("LIGHTSPEED_MODEL_PROVIDER", model_provider)

    class ObservedGeminiProvider(MockProvider):
        @property
        def name(self) -> str:
            return "gemini"

        async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
            assert options.telemetry is not None
            model_span = options.telemetry.start_model([], None, options.model)
            options.telemetry.end_model(model_span, [], None, {}, None)
            yield ResultEvent(text='{"success": true, "summary": "gemini response"}')

    await run_agent_query(
        ObservedGeminiProvider(),
        prompt="test",
        system_prompt="sys",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="gemini-2.5-flash",
        max_turns=200,
        timeout_seconds=300,
    )
    spans = span_exporter.get_finished_spans()
    agent = next(s for s in spans if s.attributes.get("gen_ai.operation.name") == "invoke_agent")
    model = next(s for s in spans if s.attributes.get("gen_ai.operation.name") == "chat")
    assert agent.attributes["gen_ai.provider.name"] == expected
    assert model.attributes["gen_ai.provider.name"] == expected


@pytest.mark.asyncio
async def test_agent_does_not_invent_correlation(span_exporter) -> None:
    await run_agent_query(
        MockProvider(),
        prompt="test",
        system_prompt="sys",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )
    agent = span_exporter.get_finished_spans()[0]
    assert "agenticrun.uid" not in agent.attributes
    assert "agenticrun.phase" not in agent.attributes


@pytest.mark.asyncio
async def test_run_agent_query_re_raises_tool_result_safety_failure() -> None:
    class SafetyFailureProvider(MockProvider):
        async def query(self, _options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
            raise ToolResultSafetyInspectionFailed()
            yield  # pragma: no cover

    with pytest.raises(ToolResultSafetyInspectionFailed):
        await run_agent_query(
            SafetyFailureProvider(),
            prompt="test",
            system_prompt="You are an AI agent.",
            output_schema=None,
            context=None,
            skills_dir="/workspace",
            model="test-model",
            max_turns=200,
            timeout_seconds=300,
        )


@pytest.mark.asyncio
async def test_run_agent_query_deadline_during_inspection_is_safety_failure() -> None:
    blocked = asyncio.Event()

    async def inspect(_name: str, _result_type: str, _content: object, _call_id: str) -> None:
        await blocked.wait()

    class BlockedInspectorProvider(MockProvider):
        async def query(self, _options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
            middleware = ToolResultInspectionMiddleware(inspect)
            request = SimpleNamespace(
                messages=[ToolMessage(content="result", name="execute", tool_call_id="call-1")]
            )

            async def model_handler(_request: object) -> None:
                return None

            await middleware.awrap_model_call(request, model_handler)
            yield ResultEvent(text='{"success":true}')

    with pytest.raises(ToolResultSafetyInspectionFailed):
        await run_agent_query(
            BlockedInspectorProvider(),
            prompt="test",
            system_prompt="You are an AI agent.",
            output_schema=None,
            context=None,
            skills_dir="/workspace",
            model="test-model",
            max_turns=200,
            timeout_seconds=0.05,
        )


@pytest.mark.asyncio
async def test_run_agent_query_timeout(span_exporter) -> None:
    """Timeout closes unfinished child spans and omits terminal output."""

    class SlowProvider(MockProvider):
        async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
            assert options.telemetry is not None
            options.telemetry.start_model([], None, options.model)
            await asyncio.sleep(2)
            yield ResultEvent(text="late")

    result = await run_agent_query(
        SlowProvider(),
        prompt="test",
        system_prompt="sys",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=0.05,
    )
    assert result.timed_out
    assert result.output["success"] is False
    agent = next(
        s
        for s in span_exporter.get_finished_spans()
        if s.attributes["gen_ai.operation.name"] == "invoke_agent"
    )
    child = next(
        s
        for s in span_exporter.get_finished_spans()
        if s.attributes["gen_ai.operation.name"] == "chat"
    )
    assert agent.status.status_code == StatusCode.ERROR
    assert agent.attributes["error.type"] == "TimeoutError"
    assert "gen_ai.output.messages" not in agent.attributes
    assert child.attributes["error.type"] == "incomplete"


@pytest.mark.asyncio
async def test_provider_exception_omits_terminal_output(span_exporter) -> None:
    class FailingProvider(MockProvider):
        async def query(self, _options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
            raise RuntimeError("private provider exception")
            yield ResultEvent(text="unreachable")

    result = await run_agent_query(
        FailingProvider(),
        prompt="test",
        system_prompt="sys",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )
    assert result.output["success"] is False
    agent = next(
        s
        for s in span_exporter.get_finished_spans()
        if s.attributes["gen_ai.operation.name"] == "invoke_agent"
    )
    assert agent.status.status_code == StatusCode.ERROR
    assert agent.attributes["error.type"] == "RuntimeError"
    assert "gen_ai.output.messages" not in agent.attributes
    assert not agent.events


@pytest.mark.asyncio
async def test_run_agent_query_empty_response() -> None:
    """Empty ResultEvent text is treated as agent failure."""
    result = await run_agent_query(
        MockProvider(events=[ResultEvent(text="")]),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )
    assert result.output["success"] is False
    assert result.output["summary"] == "Agent returned empty response"


@pytest.mark.asyncio
async def test_run_agent_query_text_response() -> None:
    """Plain-text ResultEvent becomes summary when no JSON schema is required."""
    result = await run_agent_query(
        MockProvider(events=[ResultEvent(text="plain text answer")]),
        prompt="test",
        system_prompt="You are an AI agent.",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )
    assert result.output["success"] is True
    assert result.output["summary"] == "plain text answer"



def test_format_context_envelope_markers_only() -> None:
    """Rule 12: block starts and ends with fixed marker lines."""
    text = format_context_prefix({})
    assert text == "[context]\n[/context]"


def test_format_context_unknown_keys_ignored() -> None:
    """Keys outside the supported context schema are omitted from the prefix."""
    text = format_context_prefix({"workflowPhase": "diagnose"})
    assert text == "[context]\n[/context]"


def test_format_context_target_namespaces() -> None:
    """Rule 13: comma-separated namespace list."""
    text = format_context_prefix({"targetNamespaces": ["default", "kube-system"]})
    assert "Target namespaces: default, kube-system" in text
    assert text.startswith("[context]")
    assert text.endswith("[/context]")


def test_format_context_target_namespaces_empty_list_omitted() -> None:
    """Empty targetNamespaces list produces no namespace line."""
    text = format_context_prefix({"targetNamespaces": []})
    assert "Target namespaces:" not in text


def test_format_context_attempt_includes_of_max_literal() -> None:
    """Rule 14: attempt line uses literal 'of max' placeholder."""
    text = format_context_prefix({"attempt": 2})
    assert "Attempt: 2 of max" in text


def test_format_context_attempt_zero_included() -> None:
    """Attempt zero is formatted like any other attempt number."""
    text = format_context_prefix({"attempt": 0})
    assert "Attempt: 0 of max" in text


def test_format_context_previous_attempts_with_failure_reason() -> None:
    """Previous attempts list failure reasons when present."""
    text = format_context_prefix(
        {
            "previousAttempts": [
                {"attempt": 1, "failureReason": "timeout"},
                {"attempt": 2},
            ],
        }
    )
    assert "  Attempt 1: timeout" in text
    assert "  Attempt 2" in text
    assert "  Attempt 2:" not in text


def test_format_context_previous_attempts_empty_list_omitted() -> None:
    """Empty previousAttempts list produces no attempts section."""
    text = format_context_prefix({"previousAttempts": []})
    assert "Previous attempts:" not in text


def test_format_context_approved_option_with_actions() -> None:
    """Approved option remediation actions are listed under Actions to execute."""
    text = format_context_prefix(
        {
            "approvedOption": {
                "title": "Restart pod",
                "diagnosis": {"rootCause": "CrashLoopBackOff"},
                "remediationPlan": {
                    "description": "Delete pod to trigger restart",
                    "reversible": True,
                    "actions": [
                        {
                            "type": "mutation",
                            "description": "Delete the crashing pod",
                        },
                    ],
                },
            },
        }
    )
    assert "Title: Restart pod" in text
    assert "  - [mutation] Delete the crashing pod" in text


def test_format_context_approved_option_with_command() -> None:
    """Action commands are included in the formatted remediation plan."""
    text = format_context_prefix(
        {
            "approvedOption": {
                "title": "Patch configmap",
                "diagnosis": {"rootCause": "wrong value"},
                "remediationPlan": {
                    "description": "Patch configmap data",
                    "actions": [
                        {
                            "type": "mutation",
                            "command": 'kubectl patch configmap foo -p \'{"data":{"k":"v"}}\'',
                            "description": "Apply patch",
                        },
                    ],
                },
            },
        }
    )
    assert "kubectl patch configmap foo" in text


def test_format_context_approved_option_without_actions() -> None:
    """Remediation plan without actions omits the Actions to execute section."""
    text = format_context_prefix(
        {
            "approvedOption": {
                "title": "Manual step",
                "diagnosis": {"rootCause": "needs human"},
                "remediationPlan": {"description": "Contact admin"},
            },
        }
    )
    assert "Title: Manual step" in text
    assert "Actions to execute:" not in text


def test_format_context_combined_fields() -> None:
    """All supported context fields appear together inside the envelope."""
    text = format_context_prefix(
        {
            "targetNamespaces": ["openshift-logging"],
            "attempt": 3,
            "previousAttempts": [{"attempt": 2, "failureReason": "denied"}],
            "approvedOption": {
                "title": "Fix RBAC",
                "diagnosis": {"rootCause": "missing role"},
                "remediationPlan": {
                    "description": "Apply RoleBinding",
                    "reversible": False,
                },
            },
        }
    )
    lines = text.splitlines()
    assert lines[0] == "[context]"
    assert lines[-1] == "[/context]"
    assert "Target namespaces: openshift-logging" in text
    assert "Attempt: 3 of max" in text
    assert "  Attempt 2: denied" in text
    assert "Title: Fix RBAC" in text


def test_format_context_approved_option_missing_diagnosis() -> None:
    """Missing approvedOption.diagnosis raises ContextFormatError."""
    with pytest.raises(ContextFormatError, match=r"approvedOption\.diagnosis"):
        format_context_prefix(
            {
                "approvedOption": {
                    "title": "Fix",
                    "remediationPlan": {"description": "plan"},
                },
            }
        )


def test_format_context_approved_option_missing_root_cause() -> None:
    """Missing approvedOption.diagnosis.rootCause raises ContextFormatError."""
    with pytest.raises(ContextFormatError, match=r"approvedOption\.diagnosis\.rootCause"):
        format_context_prefix(
            {
                "approvedOption": {
                    "title": "Fix",
                    "diagnosis": {},
                    "remediationPlan": {"description": "plan"},
                },
            }
        )


def test_format_context_previous_attempts_missing_attempt() -> None:
    """Previous attempt entry without attempt number raises ContextFormatError."""
    with pytest.raises(ContextFormatError, match="previousAttempts\\[0\\] missing attempt"):
        format_context_prefix({"previousAttempts": [{"failureReason": "timeout"}]})


@pytest.mark.asyncio
async def test_run_agent_query_invalid_context_returns_agent_failure() -> None:
    """Invalid context formatting returns agent failure instead of raising."""
    provider = MockProvider(events=[ResultEvent(text='{"success":true,"summary":"ok"}')])

    result = await run_agent_query(
        provider,
        prompt="run",
        system_prompt="sys",
        output_schema=None,
        context={"approvedOption": {"title": "only title"}},
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )

    assert result.output["success"] is False
    assert "Invalid context:" in result.output["summary"]
    assert "approvedOption.diagnosis" in result.output["summary"]


@pytest.mark.asyncio
async def test_run_agent_query_returns_token_counts() -> None:
    """Token counts from ResultEvent are included in the returned dict (OLS-3994)."""
    provider = MockProvider(
        events=[
            ResultEvent(
                text='{"success": true, "summary": "ok"}',
                input_tokens=500,
                output_tokens=200,
            )
        ]
    )
    result = await run_agent_query(
        provider,
        prompt="test",
        system_prompt="sys",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )
    assert result.input_tokens == 500
    assert result.output_tokens == 200


@pytest.mark.asyncio
async def test_run_agent_query_token_counts_zero_on_timeout() -> None:
    """Token counts default to 0 when the agent times out (OLS-3994)."""

    class SlowProvider(MockProvider):
        async def query(self, _options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
            await asyncio.sleep(10)
            yield ResultEvent(text="late")

    result = await run_agent_query(
        SlowProvider(),
        prompt="test",
        system_prompt="sys",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=1,
    )
    assert result.input_tokens == 0
    assert result.output_tokens == 0


@pytest.mark.asyncio
async def test_run_agent_query_token_counts_on_text_response() -> None:
    """Token counts present on plain text (non-JSON) responses (OLS-3994)."""
    provider = MockProvider(
        events=[ResultEvent(text="plain text", input_tokens=10, output_tokens=5)]
    )
    result = await run_agent_query(
        provider,
        prompt="test",
        system_prompt="sys",
        output_schema=None,
        context=None,
        skills_dir="/workspace",
        model="test-model",
        max_turns=200,
        timeout_seconds=300,
    )
    assert result.input_tokens == 10
    assert result.output_tokens == 5
