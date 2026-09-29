"""Tests for OpenAI provider configuration and manifest building."""

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_adds_additional_properties_false() -> None:
    from lightspeed_agentic.providers.openai import _make_strict  # type: ignore[import-untyped]

    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    result = _make_strict(schema)
    assert result["additionalProperties"] is False


def test_sets_required_to_all_keys() -> None:
    from lightspeed_agentic.providers.openai import _make_strict

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name"],
    }
    result = _make_strict(schema)
    assert sorted(result["required"]) == ["age", "name"]


def test_adds_required_when_missing() -> None:
    from lightspeed_agentic.providers.openai import _make_strict

    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    result = _make_strict(schema)
    assert result["required"] == ["x"]


def test_recurses_into_nested_objects() -> None:
    from lightspeed_agentic.providers.openai import _make_strict

    schema = {
        "type": "object",
        "properties": {"inner": {"type": "object", "properties": {"val": {"type": "string"}}}},
    }
    result = _make_strict(schema)
    inner = result["properties"]["inner"]
    assert inner["additionalProperties"] is False
    assert inner["required"] == ["val"]


def test_recurses_into_array_items() -> None:
    from lightspeed_agentic.providers.openai import _make_strict

    schema = {
        "type": "object",
        "properties": {
            "items_list": {
                "type": "array",
                "items": {"type": "object", "properties": {"id": {"type": "integer"}}},
            }
        },
    }
    result = _make_strict(schema)
    items_obj = result["properties"]["items_list"]["items"]
    assert items_obj["additionalProperties"] is False
    assert items_obj["required"] == ["id"]


def test_recurses_into_anyof() -> None:
    from lightspeed_agentic.providers.openai import _make_strict

    schema = {
        "anyOf": [
            {"type": "object", "properties": {"a": {"type": "string"}}},
            {"type": "string"},
        ]
    }
    result = _make_strict(schema)
    assert result["anyOf"][0]["additionalProperties"] is False
    assert result["anyOf"][0]["required"] == ["a"]
    assert result["anyOf"][1] == {"type": "string"}


def test_converts_oneof_to_anyof() -> None:
    from lightspeed_agentic.providers.openai import _make_strict

    schema = {"oneOf": [{"type": "object", "properties": {"b": {"type": "integer"}}}]}
    result = _make_strict(schema)
    assert "oneOf" not in result
    assert result["anyOf"][0]["additionalProperties"] is False


def test_oneof_preserves_existing_anyof() -> None:
    from lightspeed_agentic.providers.openai import _make_strict

    schema = {
        "anyOf": [{"type": "object", "properties": {"x": {"type": "string"}}}],
        "oneOf": [{"type": "object", "properties": {"y": {"type": "integer"}}}],
    }
    result = _make_strict(schema)
    assert "oneOf" not in result
    assert len(result["anyOf"]) == 2
    assert result["anyOf"][0]["additionalProperties"] is False
    assert result["anyOf"][1]["additionalProperties"] is False


def test_recurses_into_allof() -> None:
    from lightspeed_agentic.providers.openai import _make_strict

    result = _make_strict({"allOf": [{"type": "object", "properties": {"c": {"type": "boolean"}}}]})
    assert result["allOf"][0]["additionalProperties"] is False


def test_recurses_into_not() -> None:
    from lightspeed_agentic.providers.openai import _make_strict

    result = _make_strict({"not": {"type": "object", "properties": {"d": {"type": "string"}}}})
    assert result["not"]["additionalProperties"] is False


def test_recurses_into_defs() -> None:
    from lightspeed_agentic.providers.openai import _make_strict

    result = _make_strict(
        {"$defs": {"thing": {"type": "object", "properties": {"e": {"type": "string"}}}}}
    )
    assert result["$defs"]["thing"]["additionalProperties"] is False


def test_does_not_modify_original() -> None:
    from lightspeed_agentic.providers.openai import _make_strict

    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    _make_strict(schema)
    assert "additionalProperties" not in schema


def test_non_object_passthrough() -> None:
    from lightspeed_agentic.providers.openai import _make_strict

    assert _make_strict({"type": "string"}) == {"type": "string"}


def test_non_dict_passthrough() -> None:
    from lightspeed_agentic.providers.openai import _make_strict

    schema: Any = "not a dict"
    assert _make_strict(schema) == "not a dict"


def test_native_openai_schema_adds_strict_requirements() -> None:
    from lightspeed_agentic.providers.openai import _RawJsonSchema

    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    wrapper = _RawJsonSchema(schema, is_native=True)
    assert wrapper.json_schema()["additionalProperties"] is False
    assert wrapper.is_strict_json_schema() is True
    assert "additionalProperties" not in schema


def test_custom_endpoint_keeps_schema_non_strict() -> None:
    from lightspeed_agentic.providers.openai import _RawJsonSchema

    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    wrapper = _RawJsonSchema(schema, is_native=False)
    assert wrapper.is_strict_json_schema() is False
    assert "additionalProperties" not in wrapper.json_schema()


def test_build_manifest_parent_of_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Manifest root should be cwd's parent so exec_command reaches the full workspace."""
    monkeypatch.delenv("E2E_OUTPUT_DIR", raising=False)
    from lightspeed_agentic.providers.openai import _build_manifest

    manifest = _build_manifest(str(Path("/app/skills").parent))
    assert manifest.root == "/app"


def test_build_manifest_without_e2e_output_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("E2E_OUTPUT_DIR", raising=False)
    from lightspeed_agentic.providers.openai import _build_manifest

    manifest = _build_manifest("/app/skills")
    assert manifest.root == "/app/skills"
    assert manifest.extra_path_grants == ()


def test_build_manifest_grants_e2e_output_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("E2E_OUTPUT_DIR", str(tmp_path))
    from lightspeed_agentic.providers.openai import _build_manifest

    manifest = _build_manifest("/app/skills")
    assert len(manifest.extra_path_grants) == 1
    grant = manifest.extra_path_grants[0]
    assert grant.path == str(tmp_path.resolve())
    assert grant.read_only is False


def test_build_manifest_skips_e2e_output_dir_outside_temp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("E2E_OUTPUT_DIR", "/etc")
    from lightspeed_agentic.providers.openai import _build_manifest

    manifest = _build_manifest("/app/skills")
    assert manifest.extra_path_grants == ()


@pytest.mark.asyncio
async def test_openai_model_uses_shared_tls_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared_context = object()
    from lightspeed_agentic.providers.openai import OpenAIProvider

    monkeypatch.setattr(OpenAIProvider, "_client", None)
    monkeypatch.delenv("E2E_OUTPUT_DIR", raising=False)
    import lightspeed_agentic.tls as tls  # type: ignore[import-untyped]

    monkeypatch.setattr(tls, "get_ssl_context", lambda: shared_context)

    with patch("openai.DefaultAsyncHttpxClient") as http_client:
        http_client.return_value = MagicMock()
        await _run_openai_provider(str(tmp_path))

    http_client.assert_called_once_with(verify=shared_context)


async def _empty_stream() -> AsyncIterator[None]:
    return
    yield


def _run_openai_provider(cwd: str) -> Any:
    """Run OpenAIProvider.query() with mocked SDK internals.

    Returns (events, mock_sandbox_agent_cls, mock_runner) so callers can inspect
    the emitted events, SandboxAgent kwargs, and run configuration.
    """
    from lightspeed_agentic.providers.openai import OpenAIProvider
    from lightspeed_agentic.types import ProviderQueryOptions  # type: ignore[import-untyped]

    mock_result = MagicMock()
    mock_result.stream_events = _empty_stream
    mock_result.final_output = ""
    mock_result.context_wrapper.usage.input_tokens = 0
    mock_result.context_wrapper.usage.output_tokens = 0

    async def _collect() -> tuple[list[Any], MagicMock, MagicMock]:
        with (
            patch("agents.Runner.run_streamed", return_value=mock_result) as mock_runner,
            patch("agents.sandbox.SandboxAgent", return_value=MagicMock()) as mock_cls,
            patch("agents.models.openai_responses.OpenAIResponsesModel"),
            patch("openai.AsyncOpenAI"),
        ):
            provider = OpenAIProvider()
            options = ProviderQueryOptions(
                prompt="test",
                system_prompt="you are a test agent",
                model="gpt-4.1-mini",
                max_turns=1,
                allowed_tools=[],
                cwd=cwd,
            )
            events = [e async for e in provider.query(options)]
            return events, mock_cls, mock_runner

    return _collect()


@pytest.mark.asyncio
async def test_tool_output_trimmer_uses_shared_limits(tmp_path: Path) -> None:
    _, _, mock_runner = await _run_openai_provider(str(tmp_path))

    from agents.extensions import ToolOutputTrimmer

    from lightspeed_agentic.types import (
        MAX_TOOL_RETURN_CHARS,
        TOOL_RETURN_PREVIEW_CHARS,
    )

    run_config = mock_runner.call_args.kwargs["run_config"]
    trimmer = run_config.call_model_input_filter

    assert isinstance(trimmer, ToolOutputTrimmer)
    assert trimmer.max_output_chars == MAX_TOOL_RETURN_CHARS
    assert trimmer.preview_chars == TOOL_RETURN_PREVIEW_CHARS


@pytest.mark.asyncio
async def test_mcp_servers_are_passed_to_sandbox_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """MCP servers must use SandboxAgent's mcp_servers argument, not capabilities."""
    monkeypatch.delenv("E2E_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    from lightspeed_agentic.mcp import (  # type: ignore[import-untyped]
        AdmittedMCPProviderServer,
    )

    mcp_server = object()
    manager = MagicMock(active_servers=[mcp_server])
    manager.__aenter__ = AsyncMock(return_value=manager)
    manager.__aexit__ = AsyncMock(return_value=None)
    admitted_server = AdmittedMCPProviderServer(
        name="test",
        url="http://mcp.test/mcp",
        allowed_tool_names=("get_pod",),
    )

    from lightspeed_agentic.providers.openai import OpenAIProvider
    from lightspeed_agentic.types import ProviderQueryOptions

    mock_result = MagicMock()
    mock_result.stream_events = _empty_stream
    mock_result.final_output = ""
    mock_result.context_wrapper.usage.input_tokens = 0
    mock_result.context_wrapper.usage.output_tokens = 0

    async def collect() -> MagicMock:
        with (
            patch("agents.sandbox.SandboxAgent", return_value=MagicMock()) as mock_cls,
            patch("agents.Runner.run_streamed", return_value=mock_result),
            patch("agents.models.openai_responses.OpenAIResponsesModel"),
            patch("openai.AsyncOpenAI"),
            patch("agents.mcp.MCPServerManager", return_value=manager),
            patch(
                "lightspeed_agentic.mcp.to_openai_mcp_servers",
                return_value=[admitted_server],
            ),
        ):
            options = ProviderQueryOptions(
                prompt="test",
                system_prompt="you are a test agent",
                model="gpt-4.1-mini",
                max_turns=1,
                allowed_tools=[],
                cwd=str(tmp_path),
                mcp_servers=[admitted_server],
            )
            provider = OpenAIProvider()
            [event async for event in provider.query(options)]
            return mock_cls

    mock_cls = await collect()
    assert mock_cls.call_args.kwargs["mcp_servers"] == [mcp_server]
    assert mcp_server not in mock_cls.call_args.kwargs["capabilities"]


@pytest.mark.asyncio
async def test_admitted_mcp_conversion_failure_fails_query(tmp_path: Path) -> None:
    from lightspeed_agentic.mcp import AdmittedMCPProviderServer
    from lightspeed_agentic.providers.openai import OpenAIProvider
    from lightspeed_agentic.types import ProviderQueryOptions

    options = ProviderQueryOptions(
        prompt="test",
        system_prompt="system",
        model="gpt-4.1-mini",
        max_turns=1,
        allowed_tools=[],
        cwd=str(tmp_path),
        mcp_servers=[
            AdmittedMCPProviderServer(
                name="openshift",
                url="https://mcp.example/mcp",
                allowed_tool_names=("get_pod",),
            )
        ],
    )

    provider = OpenAIProvider()
    with (
        patch("agents.models.openai_responses.OpenAIResponsesModel"),
        patch("lightspeed_agentic.mcp.to_openai_mcp_servers", return_value=[]),
        pytest.raises(RuntimeError, match="conversion produced no servers"),
    ):
        [event async for event in provider.query(options)]


@pytest.mark.asyncio
async def test_missing_admitted_mcp_active_server_fails_query(tmp_path: Path) -> None:
    from lightspeed_agentic.mcp import AdmittedMCPProviderServer
    from lightspeed_agentic.providers.openai import OpenAIProvider
    from lightspeed_agentic.types import ProviderQueryOptions

    options = ProviderQueryOptions(
        prompt="test",
        system_prompt="system",
        model="gpt-4.1-mini",
        max_turns=1,
        allowed_tools=[],
        cwd=str(tmp_path),
        mcp_servers=[
            AdmittedMCPProviderServer(
                name="openshift",
                url="https://mcp.example/mcp",
                allowed_tool_names=("get_pod",),
            )
        ],
    )
    converted_server = object()
    manager = MagicMock(active_servers=[])
    manager.__aenter__ = AsyncMock(return_value=manager)
    manager.__aexit__ = AsyncMock(return_value=None)

    provider = OpenAIProvider()
    with (
        patch("agents.models.openai_responses.OpenAIResponsesModel"),
        patch("lightspeed_agentic.mcp.to_openai_mcp_servers", return_value=[converted_server]),
        patch("agents.mcp.MCPServerManager", return_value=manager),
        pytest.raises(RuntimeError, match="initialized 0 of 1 admitted servers"),
    ):
        [event async for event in provider.query(options)]

    manager.__aexit__.assert_awaited_once_with(None, None, None)


@pytest.mark.asyncio
async def test_admitted_mcp_manager_initialization_failure_propagates(tmp_path: Path) -> None:
    from lightspeed_agentic.mcp import AdmittedMCPProviderServer
    from lightspeed_agentic.providers.openai import OpenAIProvider
    from lightspeed_agentic.types import ProviderQueryOptions

    options = ProviderQueryOptions(
        prompt="test",
        system_prompt="system",
        model="gpt-4.1-mini",
        max_turns=1,
        allowed_tools=[],
        cwd=str(tmp_path),
        mcp_servers=[
            AdmittedMCPProviderServer(
                name="openshift",
                url="https://mcp.example/mcp",
                allowed_tool_names=("get_pod",),
            )
        ],
    )
    manager = MagicMock()
    manager.__aenter__ = AsyncMock(side_effect=ConnectionError("unavailable"))
    manager.__aexit__ = AsyncMock(return_value=None)

    provider = OpenAIProvider()
    with (
        patch("agents.models.openai_responses.OpenAIResponsesModel"),
        patch("lightspeed_agentic.mcp.to_openai_mcp_servers", return_value=[object()]),
        patch("agents.mcp.MCPServerManager", return_value=manager),
        pytest.raises(ConnectionError, match="unavailable"),
    ):
        [event async for event in provider.query(options)]

    manager.__aexit__.assert_not_awaited()


@pytest.mark.asyncio
async def test_skills_registered_when_skill_md_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Skills capability must be registered when a subdirectory contains SKILL.md."""
    monkeypatch.delenv("E2E_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    (tmp_path / "my-skill").mkdir()
    (tmp_path / "my-skill" / "SKILL.md").write_text("# skill")

    _, mock_cls, _ = await _run_openai_provider(str(tmp_path))

    capabilities = mock_cls.call_args.kwargs["capabilities"]

    from agents.sandbox.capabilities import Skills

    skills_caps = [c for c in capabilities if isinstance(c, Skills)]
    assert len(skills_caps) == 1
    assert skills_caps[0].skills_path == "skills/.agents"


@pytest.mark.asyncio
async def test_skills_capability_omitted_when_no_skill_md(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Skills capability must not be registered when no SKILL.md exists under cwd."""
    monkeypatch.delenv("E2E_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    _, mock_cls, _ = await _run_openai_provider(str(tmp_path))

    capabilities = mock_cls.call_args.kwargs["capabilities"]

    from agents.sandbox.capabilities import Skills

    skills_caps = [c for c in capabilities if isinstance(c, Skills)]
    assert len(skills_caps) == 0


class TestExecCommandShellCoercion:
    """OLS-3257: model sends shell:bool instead of shell:string."""

    @pytest.fixture(autouse=True)
    def _init_openai(self) -> None:
        from lightspeed_agentic.providers.openai import _ensure_openai_init

        _ensure_openai_init()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("shell_input", "expected"),
        [
            (True, None),
            (False, None),
            ("/bin/bash", "/bin/bash"),
            (None, None),
        ],
        ids=["bool-true", "bool-false", "string-path", "null"],
    )
    async def test_shell_coercion(self, shell_input: Any, expected: Any) -> None:
        from agents.sandbox.capabilities.tools.shell_tool import ExecCommandTool

        raw_input = json.dumps({"cmd": "echo hello", "shell": shell_input})
        captured: list[Any] = []
        original_run = ExecCommandTool.run

        async def mock_run(self: object, args: Any) -> str:  # noqa: ARG001
            captured.append(args.shell)
            return "ok"

        type.__setattr__(ExecCommandTool, "run", mock_run)
        try:
            tool = ExecCommandTool.__new__(ExecCommandTool)
            object.__setattr__(tool, "args_model", ExecCommandTool.args_model)
            await tool._invoke(None, raw_input)
            assert captured == [expected]
        finally:
            type.__setattr__(ExecCommandTool, "run", original_run)


class _RecordingTelemetry:
    def __init__(self) -> None:
        self.models: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []

    def start_model(
        self,
        input_messages: list[dict[str, Any]],
        system_instructions: Any,
        request_model: str,
        *,
        _tool_definitions: Any = None,
    ) -> object:
        entry = {
            "input": input_messages,
            "system": system_instructions,
            "request_model": request_model,
        }
        self.models.append(entry)
        return entry

    def end_model(
        self,
        handle: object,
        output_messages: list[dict[str, Any]] | None,
        response_model: str | None,
        usage: dict[str, int],
        error: BaseException | None,
    ) -> None:
        assert isinstance(handle, dict)
        handle.update(
            output=output_messages, response_model=response_model, usage=usage, error=error
        )

    def start_tool(self, name: str, call_id: str, arguments: Any) -> object:
        entry = {"name": name, "call_id": call_id, "arguments": arguments}
        self.tools.append(entry)
        return entry

    def end_tool(self, handle: object, result: Any, error: BaseException | None) -> None:
        assert isinstance(handle, dict)
        handle.update(result=result, error=error)


def _telemetry_options(tmp_path: Path, observer: _RecordingTelemetry) -> Any:
    from lightspeed_agentic.types import ProviderQueryOptions

    return ProviderQueryOptions(
        prompt="hi",
        system_prompt="system",
        model="requested",
        max_turns=2,
        allowed_tools=[],
        cwd=str(tmp_path),
        telemetry=observer,
    )


@pytest.mark.asyncio
async def test_model_hooks_capture_filtered_ordered_messages_and_per_request_usage(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from jsonschema import Draft202012Validator

    from lightspeed_agentic.providers.openai import _telemetry_hooks

    observer = _RecordingTelemetry()
    hooks = _telemetry_hooks(_telemetry_options(tmp_path, observer))
    agent = SimpleNamespace(model=SimpleNamespace(model="requested"))
    filtered = [
        {"role": "user", "content": [{"type": "input_text", "text": "café"}]},
        {
            "type": "function_call",
            "name": "search",
            "call_id": "call-1",
            "arguments": '{"query":"été"}',
        },
        {"type": "function_call_output", "call_id": "call-1", "output": '{"count":1}'},
    ]
    await hooks.on_llm_start(None, agent, "system", filtered)
    response = SimpleNamespace(
        response_id="resp-1",
        output=[
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "think"}]},
            {"type": "message", "content": [{"type": "output_text", "text": "Bonjour"}]},
            {
                "type": "function_call",
                "call_id": "next-2",
                "name": "read_file",
                "arguments": json.dumps({"path": str(tmp_path / "é")}, ensure_ascii=False),
            },
        ],
        usage=SimpleNamespace(
            input_tokens=18,
            output_tokens=11,
            output_tokens_details=SimpleNamespace(reasoning_tokens=4),
        ),
    )
    await hooks.on_llm_end(None, agent, response)
    hooks.observe_response(SimpleNamespace(id="resp-1", model="actual-model"))
    model = observer.models[0]
    fixtures = Path(__file__).parent / "fixtures"
    for category, value in (("input", model["input"]), ("output", model["output"])):
        schema = json.loads((fixtures / f"genai-v1.41-{category}.json").read_text())
        Draft202012Validator(schema).validate(value)
    assert model["system"] == [{"type": "text", "content": "system"}]
    assert [message["role"] for message in model["input"]] == ["user", "assistant", "tool"]
    assert model["input"][2]["parts"] == [
        {"type": "tool_call_response", "id": "call-1", "response": '{"count":1}'}
    ]
    assert [part["type"] for part in model["output"][0]["parts"]] == [
        "reasoning",
        "text",
        "tool_call",
    ]
    assert model["output"][0]["parts"][-1]["arguments"] == {"path": str(tmp_path / "é")}
    assert model["output"][0]["finish_reason"] == "tool_call"
    assert model["response_model"] == "actual-model"
    assert model["usage"] == {"input_tokens": 18, "output_tokens": 11, "reasoning_tokens": 4}
    assert model["error"] is None


@pytest.mark.asyncio
async def test_tool_hooks_only_record_execution_and_pair_ids_and_errors(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from lightspeed_agentic.providers.openai import _telemetry_hooks

    observer = _RecordingTelemetry()
    hooks = _telemetry_hooks(_telemetry_options(tmp_path, observer))
    agent = SimpleNamespace(model=SimpleNamespace(model="requested"))
    await hooks.on_llm_start(None, agent, None, [{"role": "user", "content": "go"}])
    await hooks.on_llm_end(
        None,
        agent,
        SimpleNamespace(
            response_id=None,
            output=[
                {
                    "type": "function_call",
                    "call_id": "not-run",
                    "name": "search",
                    "arguments": "{}",
                },
                {
                    "type": "function_call",
                    "call_id": "executed",
                    "name": "search",
                    "arguments": "{}",
                },
            ],
            usage=None,
        ),
    )
    assert observer.tools == []  # model-proposed calls are not tool executions
    context = SimpleNamespace(tool_call_id="executed", tool_arguments='{"q":"é"}')
    tool = SimpleNamespace(name="search")
    await hooks.on_tool_start(context, agent, tool)
    await hooks.on_tool_end(context, agent, tool, "x" * 5000)
    failed = SimpleNamespace(tool_call_id="failed", tool_arguments="{}")
    await hooks.on_tool_start(failed, agent, tool)
    failure = RuntimeError("SDK tool failed")
    hooks.close(failure)
    assert [(item["name"], item["call_id"]) for item in observer.tools] == [
        ("search", "executed"),
        ("search", "failed"),
    ]
    assert observer.tools[0]["arguments"] == {"q": "é"}
    assert observer.tools[0]["result"] == "x" * 5000
    assert observer.tools[0]["error"] is None
    assert observer.tools[1]["result"] is None
    assert observer.tools[1]["error"] is failure
    assert observer.models[0]["output"][0]["parts"][0]["id"] == "not-run"


@pytest.mark.asyncio
async def test_missing_call_ids_pair_model_messages_and_out_of_order_executions(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from lightspeed_agentic.providers.openai import _telemetry_hooks

    observer = _RecordingTelemetry()
    hooks = _telemetry_hooks(_telemetry_options(tmp_path, observer))
    agent = SimpleNamespace(model=SimpleNamespace(model="requested"))
    tool = SimpleNamespace(name="search")
    calls = [
        {"type": "function_call", "name": "search", "arguments": '{"q":1}'},
        {"type": "function_call", "name": "search", "arguments": '{"q":2}'},
    ]
    await hooks.on_llm_start(None, agent, None, [{"role": "user", "content": "search"}])
    await hooks.on_llm_end(
        None,
        agent,
        SimpleNamespace(response_id="turn-1", output=calls, usage=None),
    )
    hooks.observe_response(SimpleNamespace(id="turn-1", model=None))
    request_ids = [part["id"] for part in observer.models[0]["output"][0]["parts"]]
    assert len(set(request_ids)) == 2
    assert all(request_ids)

    second = SimpleNamespace(tool_call_id=None, tool_arguments='{"q":2}')
    first = SimpleNamespace(tool_call_id=None, tool_arguments='{"q":1}')
    await hooks.on_tool_start(second, agent, tool)
    await hooks.on_tool_end(second, agent, tool, "second result")
    await hooks.on_tool_start(first, agent, tool)
    await hooks.on_tool_end(first, agent, tool, "first result")
    assert [row["call_id"] for row in observer.tools] == request_ids[::-1]
    # SDK output events carry each completed execution into the next model turn.
    for output in ("second result", "first result"):
        hooks.call_ids.observed_output(
            SimpleNamespace(call_id=None, output=output, raw_item={"output": output})
        )

    input_items = [
        {"type": "function_call", "name": "search", "arguments": call["arguments"]}
        for call in calls
    ]
    input_items.extend(
        [
            {"type": "function_call_output", "output": "second result"},
            {"type": "function_call_output", "output": "first result"},
        ]
    )
    await hooks.on_llm_start(None, agent, None, input_items)
    messages = observer.models[1]["input"]
    assert [message["role"] for message in messages] == [
        "assistant",
        "assistant",
        "tool",
        "tool",
    ]
    assert [message["parts"][0]["id"] for message in messages] == request_ids + request_ids[::-1]
    assert [message["parts"][0]["response"] for message in messages[2:]] == [
        "second result",
        "first result",
    ]
    await hooks.on_llm_end(
        None,
        agent,
        SimpleNamespace(
            response_id="turn-2",
            output=[{"type": "message", "content": "done"}],
            usage=None,
        ),
    )
    hooks.observe_response(SimpleNamespace(id="turn-2", model=None))


@pytest.mark.asyncio
async def test_explicit_generated_id_does_not_consume_duplicate_idless_result(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from lightspeed_agentic.providers.openai import _telemetry_hooks

    observer = _RecordingTelemetry()
    hooks = _telemetry_hooks(_telemetry_options(tmp_path, observer))
    agent = SimpleNamespace(model=SimpleNamespace(model="requested"))
    tool = SimpleNamespace(name="search")
    calls = [
        {"type": "function_call", "name": "search", "arguments": "{}"},
        {"type": "function_call", "name": "search", "arguments": "{}"},
    ]
    await hooks.on_llm_start(None, agent, None, "search")
    await hooks.on_llm_end(
        None, agent, SimpleNamespace(response_id="turn-1", output=calls, usage=None)
    )
    hooks.observe_response(SimpleNamespace(id="turn-1", model=None))
    ids = [part["id"] for part in observer.models[0]["output"][0]["parts"]]
    assert ids[0] != ids[1]

    first = SimpleNamespace(tool_call_id=None, tool_arguments="{}")
    second = SimpleNamespace(tool_call_id=None, tool_arguments="{}")
    await hooks.on_tool_start(first, agent, tool)
    await hooks.on_tool_start(second, agent, tool)
    await hooks.on_tool_end(second, agent, tool, "second result")
    await hooks.on_tool_end(first, agent, tool, "first result")
    hooks.call_ids.observed_output(
        SimpleNamespace(call_id=ids[0], output="first result", raw_item={"output": "first result"})
    )
    hooks.call_ids.observed_output(
        SimpleNamespace(call_id=None, output="second result", raw_item={"output": "second result"})
    )
    assert [row["call_id"] for row in observer.tools] == ids

    await hooks.on_llm_start(
        None,
        agent,
        None,
        [
            *calls,
            {"type": "function_call_output", "call_id": ids[0], "output": "first result"},
            {"type": "function_call_output", "output": "second result"},
        ],
    )
    messages = observer.models[1]["input"]
    assert [message["parts"][0]["id"] for message in messages] == ids + ids
    assert [message["parts"][0]["response"] for message in messages[2:]] == [
        "first result",
        "second result",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_index", [0, 1])
@pytest.mark.parametrize("generated_first", [False, True])
async def test_duplicate_calls_keep_explicit_and_generated_ids_across_execution_and_results(
    tmp_path: Path, explicit_index: int, generated_first: bool
) -> None:
    from types import SimpleNamespace

    from lightspeed_agentic.providers.openai import _telemetry_hooks

    observer = _RecordingTelemetry()
    hooks = _telemetry_hooks(_telemetry_options(tmp_path, observer))
    agent = SimpleNamespace(model=SimpleNamespace(model="requested"))
    tool = SimpleNamespace(name="search")
    calls = [
        {
            "type": "function_call",
            "name": "search",
            "arguments": '{"q":1}',
            **({"call_id": "sdk-call"} if index == explicit_index else {}),
        }
        for index in range(2)
    ]
    await hooks.on_llm_start(None, agent, None, "search")
    await hooks.on_llm_end(
        None, agent, SimpleNamespace(response_id="turn-1", output=calls, usage=None)
    )
    hooks.observe_response(SimpleNamespace(id="turn-1", model=None))
    ids = [part["id"] for part in observer.models[0]["output"][0]["parts"]]
    generated_index = 1 - explicit_index
    assert ids[explicit_index] == "sdk-call"
    assert ids[generated_index]
    assert ids[generated_index] != "sdk-call"

    explicit = SimpleNamespace(tool_call_id="sdk-call", tool_arguments='{"q":1}')
    generated = SimpleNamespace(tool_call_id=None, tool_arguments='{"q":1}')
    # Finish the executions in reverse order, then emit the ID-less output first.
    start_order = (generated, explicit) if generated_first else (explicit, generated)
    for context in start_order:
        await hooks.on_tool_start(context, agent, tool)
    assert [row["call_id"] for row in observer.tools] == (
        [ids[generated_index], "sdk-call"]
        if generated_first
        else ["sdk-call", ids[generated_index]]
    )
    end_order = (
        ((explicit, "explicit result"), (generated, "generated result"))
        if generated_first
        else ((generated, "generated result"), (explicit, "explicit result"))
    )
    for context, result in end_order:
        await hooks.on_tool_end(context, agent, tool, result)
    hooks.call_ids.observed_output(
        SimpleNamespace(
            call_id=None, output="generated result", raw_item={"output": "generated result"}
        )
    )
    hooks.call_ids.observed_output(
        SimpleNamespace(
            call_id="sdk-call", output="explicit result", raw_item={"output": "explicit result"}
        )
    )

    await hooks.on_llm_start(
        None,
        agent,
        None,
        [
            *calls,
            {"type": "function_call_output", "output": "generated result"},
            {
                "type": "function_call_output",
                "call_id": "sdk-call",
                "output": "explicit result",
            },
        ],
    )
    messages = observer.models[1]["input"]
    assert [message["parts"][0]["id"] for message in messages] == [
        *ids,
        ids[generated_index],
        "sdk-call",
    ]
    assert [message["parts"][0]["response"] for message in messages[2:]] == [
        "generated result",
        "explicit result",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_index", [0, 1])
@pytest.mark.parametrize("explicit_result_first", [False, True])
@pytest.mark.parametrize("explicit_output_first", [False, True])
async def test_equal_results_preserve_explicit_and_generated_execution_ids(
    tmp_path: Path, explicit_index: int, explicit_result_first: bool, explicit_output_first: bool
) -> None:
    from types import SimpleNamespace

    from lightspeed_agentic.providers.openai import _telemetry_hooks

    observer = _RecordingTelemetry()
    hooks = _telemetry_hooks(_telemetry_options(tmp_path, observer))
    agent = SimpleNamespace(model=SimpleNamespace(model="requested"))
    tool = SimpleNamespace(name="search")
    calls = [
        {
            "type": "function_call",
            "name": "search",
            "arguments": '{"q":1}',
            **({"call_id": "sdk-call"} if index == explicit_index else {}),
        }
        for index in range(2)
    ]
    await hooks.on_llm_start(None, agent, None, "search")
    await hooks.on_llm_end(
        None, agent, SimpleNamespace(response_id="turn-1", output=calls, usage=None)
    )
    hooks.observe_response(SimpleNamespace(id="turn-1", model=None))
    ids = [part["id"] for part in observer.models[0]["output"][0]["parts"]]
    generated_id = ids[1 - explicit_index]
    assert ids[explicit_index] == "sdk-call"
    assert generated_id
    assert generated_id != "sdk-call"

    explicit = SimpleNamespace(tool_call_id="sdk-call", tool_arguments='{"q":1}')
    generated = SimpleNamespace(tool_call_id=None, tool_arguments='{"q":1}')
    await hooks.on_tool_start(explicit, agent, tool)
    await hooks.on_tool_start(generated, agent, tool)
    assert [row["call_id"] for row in observer.tools] == ["sdk-call", generated_id]
    result = "same result"
    for context in (explicit, generated) if explicit_result_first else (generated, explicit):
        await hooks.on_tool_end(context, agent, tool, result)
    assert [row["result"] for row in observer.tools] == [result, result]

    explicit_output = SimpleNamespace(
        call_id="sdk-call", output=result, raw_item={"output": result}
    )
    generated_output = SimpleNamespace(call_id=None, output=result, raw_item={"output": result})
    first, second = (
        (explicit_output, generated_output)
        if explicit_output_first
        else (generated_output, explicit_output)
    )
    hooks.call_ids.observed_output(first)
    assert hooks.call_ids.results == [
        (generated_id if explicit_output_first else "sdk-call", result)
    ]
    hooks.call_ids.observed_output(second)
    assert not hooks.call_ids.results

    outputs = [
        {"type": "function_call_output", "call_id": "sdk-call", "output": result},
        {"type": "function_call_output", "output": result},
    ]
    if not explicit_output_first:
        outputs.reverse()
    await hooks.on_llm_start(None, agent, None, [*calls, *outputs])
    messages = observer.models[1]["input"]
    assert [part["id"] for message in messages[:2] for part in message["parts"]] == ids
    responses = [message["parts"][0] for message in messages[2:]]
    expected = [
        {"type": "tool_call_response", "id": "sdk-call", "response": result},
        {"type": "tool_call_response", "id": generated_id, "response": result},
    ]
    if not explicit_output_first:
        expected.reverse()
    assert responses == expected
    assert sorted(part["id"] for part in responses) == sorted(
        row["call_id"] for row in observer.tools
    )


@pytest.mark.asyncio
async def test_skill_staging_and_instruction_read_keep_original_tool_identity(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from lightspeed_agentic.providers.openai import _telemetry_hooks

    observer = _RecordingTelemetry()
    hooks = _telemetry_hooks(_telemetry_options(tmp_path, observer))
    stage = SimpleNamespace(tool_call_id="stage-1", tool_arguments='{"skill_name":"trees"}')
    read = SimpleNamespace(
        tool_call_id="read-2",
        tool_arguments='{"path":"skills/.agents/trees/SKILL.md"}',
    )
    await hooks.on_tool_start(stage, None, SimpleNamespace(name="load_skill"))
    await hooks.on_tool_end(stage, None, SimpleNamespace(name="load_skill"), {"status": "loaded"})
    await hooks.on_tool_start(read, None, SimpleNamespace(name="read_file"))
    content = "# Trees 🌳\n" + "instructions " * 900
    await hooks.on_tool_end(read, None, SimpleNamespace(name="read_file"), content)
    assert observer.tools == [
        {
            "name": "load_skill",
            "call_id": "stage-1",
            "arguments": {"skill_name": "trees"},
            "result": {"status": "loaded"},
            "error": None,
        },
        {
            "name": "read_file",
            "call_id": "read-2",
            "arguments": {"path": "skills/.agents/trees/SKILL.md"},
            "result": content,
            "error": None,
        },
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_native", [True, False], ids=["native", "chat-completions"])
async def test_runner_attaches_hooks_only_with_observer_and_records_each_model_turn(
    tmp_path: Path,
    is_native: bool,
) -> None:
    from types import SimpleNamespace

    from agents import RawResponsesStreamEvent

    from lightspeed_agentic.providers.openai import OpenAIProvider

    observer = _RecordingTelemetry()
    model_usage = SimpleNamespace(
        input_tokens=3,
        output_tokens=7,
        output_tokens_details=SimpleNamespace(reasoning_tokens=2),
    )
    result = MagicMock()
    result.final_output = "ok"
    result.context_wrapper.model = None
    result.context_wrapper.usage = model_usage
    agent = SimpleNamespace(model=SimpleNamespace(model="requested"))

    async def stream() -> AsyncIterator[Any]:
        hooks = mocked_runner.call_args.kwargs["hooks"]
        for number in range(2):
            await hooks.on_llm_start(
                None,
                agent,
                "system",
                [
                    {"role": "user", "content": f"turn {number}"},
                ],
            )
            await hooks.on_llm_end(
                None,
                agent,
                SimpleNamespace(
                    response_id=f"response-{number}",
                    output=[
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": f"answer {number}"},
                            ],
                        }
                    ],
                    usage=model_usage,
                ),
            )
            yield RawResponsesStreamEvent(
                data=SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(
                        id=f"response-{number}",
                        model=f"actual-{number}" if is_native else None,
                    ),
                )
            )

    result.stream_events = stream
    with (
        patch("agents.Runner.run_streamed", return_value=result) as mocked_runner,
        patch("agents.sandbox.SandboxAgent", return_value=agent),
        patch("agents.models.openai_responses.OpenAIResponsesModel"),
        patch("agents.models.openai_chatcompletions.OpenAIChatCompletionsModel"),
        patch("lightspeed_agentic.providers.openai._is_native_openai", return_value=is_native),
        patch("openai.AsyncOpenAI"),
    ):
        provider = OpenAIProvider()
        events = [event async for event in provider.query(_telemetry_options(tmp_path, observer))]
    assert [row["response_model"] for row in observer.models] == (
        ["actual-0", "actual-1"] if is_native else [None, None]
    )
    assert [row["output"][0]["parts"][0]["content"] for row in observer.models] == [
        "answer 0",
        "answer 1",
    ]
    assert events[-1].response_model == ("actual-1" if is_native else "requested")
    assert events[-1].output_tokens == 7
    assert events[-1].reasoning_tokens == 2


@pytest.mark.asyncio
async def test_model_hooks_fall_back_to_requested_model_and_zero_unavailable_usage(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from lightspeed_agentic.providers.openai import _telemetry_hooks

    observer = _RecordingTelemetry()
    hooks = _telemetry_hooks(_telemetry_options(tmp_path, observer))
    agent = SimpleNamespace(model=SimpleNamespace(model=None))
    await hooks.on_llm_start(None, agent, None, [{"role": "user", "content": "hello"}])
    await hooks.on_llm_end(
        None,
        agent,
        SimpleNamespace(
            response_id=None,
            output=[{"type": "message", "content": "done"}],
            usage=None,
        ),
    )
    hooks.close()
    assert observer.models[0]["request_model"] == "requested"
    assert observer.models[0]["response_model"] is None
    assert observer.models[0]["usage"] == {}
    assert observer.models[0]["output"][0]["finish_reason"] == "unknown"


@pytest.mark.asyncio
async def test_chat_completion_without_response_id_does_not_claim_synthetic_model(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from lightspeed_agentic.providers.openai import _telemetry_hooks

    observer = _RecordingTelemetry()
    hooks = _telemetry_hooks(_telemetry_options(tmp_path, observer), is_native=False)
    agent = SimpleNamespace(model=SimpleNamespace(model="requested"))
    await hooks.on_llm_start(None, agent, None, [{"role": "user", "content": "hello"}])
    await hooks.on_llm_end(
        None,
        agent,
        SimpleNamespace(
            response_id=None,
            output=[{"type": "message", "content": "answer"}],
            usage=SimpleNamespace(requests=0, input_tokens=0, output_tokens=0),
        ),
    )
    hooks.observe_response(SimpleNamespace(id="SDK response id", model="actual-model"))
    assert observer.models[0]["response_model"] is None
    assert observer.models[0]["usage"] == {}


@pytest.mark.asyncio
async def test_failed_model_request_does_not_claim_output_or_usage(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from lightspeed_agentic.providers.openai import _telemetry_hooks

    observer = _RecordingTelemetry()
    hooks = _telemetry_hooks(_telemetry_options(tmp_path, observer))
    await hooks.on_llm_start(
        None,
        SimpleNamespace(model=SimpleNamespace(model="requested")),
        None,
        [{"role": "user", "content": "hi"}],
    )
    failure = RuntimeError("API failed")
    hooks.close(failure)
    assert observer.models[0]["output"] is None
    assert observer.models[0]["usage"] == {}
    assert observer.models[0]["error"] is failure


@pytest.mark.asyncio
async def test_native_skill_read_preserves_raw_shell_output_and_call_id(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from lightspeed_agentic.providers.openai import _telemetry_hooks

    observer = _RecordingTelemetry()
    hooks = _telemetry_hooks(_telemetry_options(tmp_path, observer))
    output = "Chunk ID: 1\nProcess exited with code 0\nOutput:\nforecast 🌦️"
    command = "cat skills/.agents/weather/SKILL.md"
    context = SimpleNamespace(
        tool_call_id="shell-read",
        tool_arguments=json.dumps({"cmd": command}),
    )
    tool = SimpleNamespace(name="exec_command")
    await hooks.on_tool_start(context, None, tool)
    await hooks.on_tool_end(context, None, tool, output)
    assert observer.tools == [
        {
            "name": "exec_command",
            "call_id": "shell-read",
            "arguments": {"cmd": command},
            "result": output,
            "error": None,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "expected"),
    [("content_filter", "content_filter"), ("max_output_tokens", "length")],
)
async def test_backend_finish_status_preserved(
    tmp_path: Path,
    reason: str,
    expected: str,
) -> None:
    from types import SimpleNamespace

    from lightspeed_agentic.providers.openai import _telemetry_hooks

    observer = _RecordingTelemetry()
    hooks = _telemetry_hooks(_telemetry_options(tmp_path, observer))
    agent = SimpleNamespace(model=SimpleNamespace(model="requested"))
    await hooks.on_llm_start(None, agent, None, [{"role": "user", "content": "hi"}])
    await hooks.on_llm_end(
        None,
        agent,
        SimpleNamespace(
            response_id="resp-finish",
            output=[{"type": "message", "content": "partial"}],
            usage=None,
        ),
    )
    hooks.observe_response(
        SimpleNamespace(
            id="resp-finish",
            status="incomplete",
            incomplete_details=SimpleNamespace(reason=reason),
            model=None,
        )
    )
    assert observer.models[0]["output"][0]["finish_reason"] == expected
    assert observer.models[0]["response_model"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", [None, "unrecognized_reason"])
async def test_incomplete_without_observed_reason_uses_unknown_finish(
    tmp_path: Path,
    reason: str | None,
) -> None:
    from types import SimpleNamespace

    from lightspeed_agentic.providers.openai import _telemetry_hooks

    observer = _RecordingTelemetry()
    hooks = _telemetry_hooks(_telemetry_options(tmp_path, observer))
    agent = SimpleNamespace(model=SimpleNamespace(model="requested"))
    await hooks.on_llm_start(None, agent, None, [{"role": "user", "content": "hi"}])
    await hooks.on_llm_end(
        None,
        agent,
        SimpleNamespace(
            response_id="resp-unknown",
            output=[{"type": "message", "content": "partial", "status": "incomplete"}],
            usage=None,
        ),
    )
    hooks.observe_response(
        SimpleNamespace(
            id="resp-unknown",
            status="incomplete",
            incomplete_details=SimpleNamespace(reason=reason) if reason else None,
            model=None,
        )
    )
    assert observer.models[0]["output"][0]["finish_reason"] == "unknown"


@pytest.mark.asyncio
async def test_explicit_tool_error_never_claims_result_but_bare_text_is_preserved(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from lightspeed_agentic.providers.openai import _telemetry_hooks

    observer = _RecordingTelemetry()
    hooks = _telemetry_hooks(_telemetry_options(tmp_path, observer))
    tool = SimpleNamespace(name="read_file")
    explicit = SimpleNamespace(tool_call_id="failed", tool_arguments='{"path":"a"}')
    await hooks.on_tool_start(explicit, None, tool)
    await hooks.on_tool_end(explicit, None, tool, {"status": "error", "message": "denied"})
    textual = SimpleNamespace(tool_call_id="text", tool_arguments='{"path":"b"}')
    await hooks.on_tool_start(textual, None, tool)
    await hooks.on_tool_end(textual, None, tool, "Error: read failed")
    assert observer.tools[0]["call_id"] == "failed"
    assert observer.tools[0]["result"] is None
    assert isinstance(observer.tools[0]["error"], RuntimeError)
    assert observer.tools[1]["call_id"] == "text"
    assert observer.tools[1]["result"] == "Error: read failed"
    assert observer.tools[1]["error"] is None


@pytest.mark.asyncio
async def test_sdk_handled_tool_exception_preserves_response_and_marks_matching_call_failed(
    tmp_path: Path,
) -> None:
    from agents import function_tool
    from agents.tool import default_tool_error_function, invoke_function_tool
    from agents.tool_context import ToolContext

    from lightspeed_agentic.providers.openai import _telemetry_hooks

    @function_tool
    async def lookup(value: str) -> str:
        """Look up a value."""
        if value == "fail":
            raise ValueError("unavailable")
        return default_tool_error_function(None, ValueError("unavailable"))

    observer = _RecordingTelemetry()
    hooks = _telemetry_hooks(_telemetry_options(tmp_path, observer))
    failed = ToolContext(
        context=None,
        tool_name="lookup",
        tool_call_id="call-failed",
        tool_arguments='{"value":"fail"}',
    )
    successful = ToolContext(
        context=None,
        tool_name="lookup",
        tool_call_id="call-success",
        tool_arguments='{"value":"success"}',
    )
    await hooks.on_tool_start(failed, None, lookup)
    await hooks.on_tool_start(successful, None, lookup)
    failed_output, successful_output = await asyncio.gather(
        invoke_function_tool(function_tool=lookup, context=failed, arguments=failed.tool_arguments),
        invoke_function_tool(
            function_tool=lookup, context=successful, arguments=successful.tool_arguments
        ),
    )
    await hooks.on_tool_end(successful, None, lookup, successful_output)
    await hooks.on_tool_end(failed, None, lookup, failed_output)

    expected = default_tool_error_function(None, ValueError("unavailable"))
    assert failed_output == successful_output == expected
    assert [(entry["call_id"], entry["arguments"]) for entry in observer.tools] == [
        ("call-failed", {"value": "fail"}),
        ("call-success", {"value": "success"}),
    ]
    assert observer.tools[0]["result"] is None
    assert isinstance(observer.tools[0]["error"], ValueError)
    assert observer.tools[1]["result"] == expected
    assert observer.tools[1]["error"] is None


@pytest.mark.asyncio
async def test_failed_response_preserves_received_content_and_length_reason(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from lightspeed_agentic.providers.openai import _telemetry_hooks

    observer = _RecordingTelemetry()
    hooks = _telemetry_hooks(_telemetry_options(tmp_path, observer))
    await hooks.on_llm_start(
        None,
        SimpleNamespace(model=SimpleNamespace(model="requested")),
        None,
        [{"role": "user", "content": "hi"}],
    )
    hooks.observe_failed_response(
        SimpleNamespace(
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            output=[{"type": "message", "content": "received before failure"}],
            usage=SimpleNamespace(input_tokens=4, output_tokens=3),
            model="actual-model",
        )
    )
    failure = RuntimeError("incomplete")
    hooks.close(failure)
    assert observer.models[0]["output"][0]["finish_reason"] == "length"
    assert observer.models[0]["output"][0]["parts"] == [
        {"type": "text", "content": "received before failure"}
    ]
    assert observer.models[0]["response_model"] == "actual-model"
    assert observer.models[0]["usage"] == {"input_tokens": 4, "output_tokens": 3}
    assert observer.models[0]["error"] is failure
