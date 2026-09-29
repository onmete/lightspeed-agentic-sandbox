"""Offline ADK-boundary behavior of the Gemini v1.41 observer."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from google.adk.models import Gemini
from google.adk.telemetry.context import ContentCapturingMode
from google.genai import types
from jsonschema import Draft202012Validator

from lightspeed_agentic.providers.gemini import (
    GeminiProvider,
    _finish_reason,
    _messages,
)
from lightspeed_agentic.types import MAX_TOOL_RETURN_CHARS, ProviderQueryOptions


def _response(content=None, **kwargs):
    fields = dict(
        content=content, partial=False, usage_metadata=None, model_version=None, finish_reason=None
    )
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def _event(content, **kwargs):
    calls = (
        [part.function_call for part in content.parts or [] if part.function_call]
        if content
        else []
    )
    return SimpleNamespace(
        content=content,
        partial=False,
        usage_metadata=None,
        get_function_calls=lambda: calls,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("instruction", "expected_system"),
    [
        ("system actual", [{"type": "text", "content": "system actual"}]),
        (
            types.Content(
                role="system",
                parts=[types.Part(text="system actual"), types.Part(text="more context")],
            ),
            [
                {"type": "text", "content": "system actual"},
                {"type": "text", "content": "more context"},
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_model_tool_model_transcript_and_native_content_disabled(
    tmp_path, instruction, expected_system
):
    recorder = MagicMock()
    recorder.start_model.side_effect = ["model-1", "model-2"]
    recorder.start_tool.return_value = "tool-1"
    captured = {}
    raw = {"name": "big-skill", "instructions": "🌳" * (MAX_TOOL_RETURN_CHARS + 1)}
    call = types.FunctionCall(name="load_skill", args={"skill_name": "big-skill"})

    async def sdk_model_call(model, request, _stream=False):
        model._maybe_append_user_content(request)
        if "request" not in captured:
            captured["request"] = request
            yield _response(
                types.Content(role="model", parts=[types.Part(text="partial only")]), partial=True
            )
            yield _response(
                types.Content(
                    role="model",
                    parts=[
                        types.Part(text="think", thought=True),
                        types.Part(text="do it"),
                        types.Part(function_call=call),
                    ],
                ),
                usage_metadata=SimpleNamespace(
                    prompt_token_count=0, candidates_token_count=7, thoughts_token_count=2
                ),
                model_version="gemini-observed",
            )
        else:
            captured["next_request"] = request
            yield _response(
                types.Content(
                    role="model",
                    parts=[
                        types.Part(text="after", thought=True),
                        types.Part(text="done"),
                    ],
                )
            )

    async def fake_base(self, llm_request, stream=False):
        async for response in sdk_model_call(self, llm_request, stream):
            yield response

    class FakeRunner:
        def __init__(self, *, agent, **_kwargs):
            self.agent = agent

        async def run_async(self, *, new_message, run_config, **_kwargs):
            captured["run_config"] = run_config
            model = self.agent.model
            first = SimpleNamespace(
                model="gemini-requested",
                contents=[new_message],
                config=SimpleNamespace(
                    system_instruction=instruction,
                    tools=[
                        SimpleNamespace(
                            function_declarations=[
                                SimpleNamespace(
                                    model_dump=lambda **_kw: {
                                        "name": "load_skill",
                                        "description": "Read skill",
                                    }
                                )
                            ]
                        )
                    ],
                ),
            )
            async for response in model.generate_content_async(first):
                yield _event(response.content)
            assert call.id
            assert call.id.startswith("call-")
            ctx = SimpleNamespace(function_call_id=call.id)
            tool = SimpleNamespace(name="load_skill")
            self.agent.before_tool_callback(tool, dict(call.args), ctx)
            replacement = None
            for callback in self.agent.after_tool_callback:
                replacement = callback(tool, dict(call.args), ctx, raw)
                if replacement is not None:
                    break
            assert replacement["status"] == "truncated"
            captured["preview"] = replacement
            tool_content = types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name="load_skill", id=call.id, response=replacement
                        )
                    )
                ],
            )
            second = SimpleNamespace(
                model="gemini-requested",
                contents=[
                    new_message,
                    types.Content(role="model", parts=[types.Part(function_call=call)]),
                    tool_content,
                ],
                config=SimpleNamespace(system_instruction=instruction, tools=[]),
            )
            async for response in model.generate_content_async(second):
                yield _event(response.content)
            yield _event(tool_content)

    options = ProviderQueryOptions(
        prompt="ask",
        system_prompt="system original",
        model="gemini-requested",
        max_turns=3,
        allowed_tools=["Bash"],
        cwd=str(tmp_path),
        telemetry=recorder,
    )
    with (
        patch.object(Gemini, "generate_content_async", fake_base),
        patch("google.adk.runners.Runner", FakeRunner),
        patch("lightspeed_agentic.providers.gemini._load_skills_toolset", return_value=None),
    ):
        events = [event async for event in GeminiProvider().query(options)]

    assert len([e for e in events if e.type == "tool_call"]) == 1
    assert recorder.start_model.call_count == recorder.end_model.call_count == 2
    first_input = recorder.start_model.call_args_list[0].args[0]
    assert first_input == [{"role": "user", "parts": [{"type": "text", "content": "ask"}]}]
    schema = json.loads(
        (Path(__file__).parent / "fixtures" / "genai-v1.41-system-instructions.json").read_text()
    )
    for model_call in recorder.start_model.call_args_list:
        system_instructions = model_call.args[1]
        assert system_instructions == expected_system
        Draft202012Validator(schema).validate(system_instructions)
    assert recorder.start_model.call_args_list[0].kwargs["tool_definitions"] == [
        {"type": "function", "name": "load_skill", "description": "Read skill"}
    ]
    output = recorder.end_model.call_args_list[0].args
    assert output[0] == "model-1"
    assert output[1] == [
        {
            "role": "assistant",
            "parts": [
                {"type": "reasoning", "content": "think"},
                {"type": "text", "content": "do it"},
                {
                    "type": "tool_call",
                    "id": call.id,
                    "name": "load_skill",
                    "arguments": {"skill_name": "big-skill"},
                },
            ],
            "finish_reason": "tool_call",
        }
    ]
    assert output[2:] == (
        "gemini-observed",
        {"input_tokens": 0, "output_tokens": 7, "reasoning_tokens": 2},
        None,
    )
    assert recorder.start_tool.call_args.args == (
        "load_skill",
        call.id,
        {"skill_name": "big-skill"},
    )
    recorder.end_tool.assert_called_once_with("tool-1", raw, None)
    next_input = recorder.start_model.call_args_list[1].args[0]
    assert next_input[1]["parts"][0]["id"] == call.id
    assert next_input[2] == {
        "role": "tool",
        "parts": [{"type": "tool_call_response", "id": call.id, "response": captured["preview"]}],
    }
    assert recorder.end_model.call_args_list[1].args == (
        "model-2",
        [
            {
                "role": "assistant",
                "parts": [
                    {"type": "reasoning", "content": "after"},
                    {"type": "text", "content": "done"},
                ],
                "finish_reason": "unknown",
            }
        ],
        None,
        {},
        None,
    )
    schema = json.loads(
        (Path(__file__).parent / "fixtures" / "genai-v1.41-output.json").read_text()
    )
    Draft202012Validator(schema).validate(recorder.end_model.call_args_list[1].args[1])
    telemetry_config = captured["run_config"].telemetry
    assert telemetry_config.capture_message_content is ContentCapturingMode.NO_CONTENT
    assert telemetry_config.genai_semconv_stability_opt_in == "stable"


@pytest.mark.asyncio
async def test_tool_failure_never_records_success_result(tmp_path):
    recorder = MagicMock()
    recorder.start_tool.return_value = "failure-handle"
    captured = {}

    class FakeRunner:
        def __init__(self, *, agent, **_kwargs):
            self.agent = agent

        async def run_async(self, **_kwargs):
            context = SimpleNamespace(function_call_id="sdk-supplied-id")
            tool = SimpleNamespace(name="load_skill")
            self.agent.before_tool_callback(tool, {"skill_name": "missing"}, context)
            failure = ValueError("unavailable")
            self.agent.on_tool_error_callback(tool, {"skill_name": "missing"}, context, failure)
            captured["failure"] = failure
            for callback in self.agent.after_tool_callback:
                callback(tool, {"skill_name": "missing"}, context, {"error": "unavailable"})
            if False:
                yield None

    options = ProviderQueryOptions(
        prompt="ask",
        system_prompt="system",
        model="gemini-requested",
        max_turns=3,
        allowed_tools=[],
        cwd=str(tmp_path),
        telemetry=recorder,
    )
    with (
        patch("google.adk.runners.Runner", FakeRunner),
        patch("lightspeed_agentic.providers.gemini._load_skills_toolset", return_value=None),
    ):
        [event async for event in GeminiProvider().query(options)]
    recorder.start_tool.assert_called_once_with(
        "load_skill", "sdk-supplied-id", {"skill_name": "missing"}
    )
    recorder.end_tool.assert_called_once_with("failure-handle", None, captured["failure"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "failed"),
    [
        ({"isError": True, "content": [{"type": "text", "text": "private MCP failure"}]}, True),
        ({"isError": False, "content": [{"type": "text", "text": "Error: harmless text"}]}, False),
    ],
)
async def test_mcp_error_flag_preserves_model_visible_response(tmp_path, response, failed):
    recorder = MagicMock()
    recorder.start_tool.return_value = "mcp-handle"
    captured = {}

    class FakeRunner:
        def __init__(self, *, agent, **_kwargs):
            self.agent = agent

        async def run_async(self, *, new_message, **_kwargs):
            context = SimpleNamespace(function_call_id="mcp-call-id")
            tool = SimpleNamespace(name="mcp_tool")
            self.agent.before_tool_callback(tool, {}, context)
            for callback in self.agent.after_tool_callback:
                assert callback(tool, {}, context, response) is None
            captured["unchanged_response"] = response
            tool_content = types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name="mcp_tool", id=context.function_call_id, response=response
                        )
                    )
                ],
            )
            self.agent.model._maybe_append_user_content(
                SimpleNamespace(
                    model="gemini-requested",
                    contents=[new_message, tool_content],
                    config=SimpleNamespace(system_instruction="system", tools=[]),
                )
            )
            yield _event(tool_content)

    options = ProviderQueryOptions(
        prompt="ask",
        system_prompt="system",
        model="gemini-requested",
        max_turns=3,
        allowed_tools=[],
        cwd=str(tmp_path),
        telemetry=recorder,
    )
    with (
        patch("google.adk.runners.Runner", FakeRunner),
        patch("lightspeed_agentic.providers.gemini._load_skills_toolset", return_value=None),
    ):
        events = [event async for event in GeminiProvider().query(options)]

    assert captured["unchanged_response"] is response
    assert recorder.start_model.call_args.args[0][1] == {
        "role": "tool",
        "parts": [
            {"type": "tool_call_response", "id": "mcp-call-id", "response": response}
        ],
    }
    tool_results = [event for event in events if event.type == "tool_result"]
    assert len(tool_results) == 1
    assert json.loads(tool_results[0].output) == response
    handle, recorded_result, error = recorder.end_tool.call_args.args
    assert handle == "mcp-handle"
    if failed:
        assert recorded_result is None
        assert isinstance(error, RuntimeError)
        assert str(error) == "MCP tool call failed"
    else:
        assert recorded_result is response
        assert error is None


@pytest.mark.parametrize(
    ("sdk_reason", "has_calls", "expected"),
    [
        (None, False, "unknown"),
        ("STOP", False, "stop"),
        ("MAX_TOKENS", False, "length"),
        ("SAFETY", False, "content_filter"),
        ("TOOL_CALL", False, "tool_call"),
        ("error", False, "error"),
        ("OTHER", False, "error"),
        (None, True, "tool_call"),
    ],
)
def test_observed_finish_reasons(sdk_reason, has_calls, expected):
    assert _finish_reason(SimpleNamespace(finish_reason=sdk_reason), has_calls) == expected


def test_sdk_error_code_is_not_normal_completion():
    assert (
        _finish_reason(SimpleNamespace(finish_reason=None, error_code="SAFETY"), False) == "error"
    )


def test_supplied_sdk_call_id_and_response_field_are_preserved():
    contents = [
        types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id="sdk-original", name="read", args={"path": "a"}
                    )
                )
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id="sdk-original", name="read", response={"result": "original"}
                    )
                )
            ],
        ),
    ]
    messages = _messages(contents)
    assert messages[0]["parts"][0]["id"] == "sdk-original"
    assert messages[1] == {
        "role": "tool",
        "parts": [
            {"type": "tool_call_response", "id": "sdk-original", "response": {"result": "original"}}
        ],
    }


@pytest.mark.parametrize("value", [["π", False], 12, None, "unparsed 🌍", ' ["π", false] '])
def test_sdk_tool_response_value_is_preserved_in_message(value):
    content = SimpleNamespace(
        role="user",
        parts=[
            SimpleNamespace(
                text=None,
                function_call=None,
                function_response=SimpleNamespace(id="call-1", response=value),
            )
        ],
    )
    messages = _messages([content])
    assert messages == [
        {
            "role": "tool",
            "parts": [{"type": "tool_call_response", "id": "call-1", "response": value}],
        }
    ]
    schema = json.loads((Path(__file__).parent / "fixtures" / "genai-v1.41-input.json").read_text())
    Draft202012Validator(schema).validate(messages)
