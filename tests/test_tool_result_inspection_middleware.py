from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import ToolMessage

from lightspeed_agentic.inspection.errors import InspectionError
from lightspeed_agentic.inspection.middleware import (
    ToolResultInspectionMiddleware,
    ToolResultSafetyInspectionFailed,
)


class ModelRequest:
    def __init__(self, messages: list[Any]) -> None:
        self.messages = messages

    def override(self, **changes: Any) -> ModelRequest:
        return ModelRequest(changes.get("messages", self.messages))


@pytest.mark.asyncio
async def test_model_boundary_inspects_raw_then_wraps_tool_output() -> None:
    observed: list[tuple[str, str, Any]] = []
    observed_call_ids: list[str] = []
    request = ModelRequest(
        [ToolMessage(content="pod output", name="get_pods", tool_call_id="call-1")]
    )

    async def inspect(
        tool: str,
        result_type: str,
        content: Any,
        tool_call_id: str,
    ) -> None:
        observed_call_ids.append(tool_call_id)
        await _record(observed, tool, result_type, content)

    middleware = ToolResultInspectionMiddleware(inspect)
    passed_to_model: list[Any] = []

    async def handler(received: ModelRequest) -> str:
        passed_to_model.extend(received.messages)
        return "model response"

    result = await middleware.awrap_model_call(request, handler)

    assert result == "model response"
    assert observed == [("get_pods", "result", "pod output")]
    expected = '<tool_data source="get_pods">\npod output\n</tool_data>'
    assert observed_call_ids == ["call-1"]
    assert passed_to_model[0].content == expected
    assert passed_to_model[0] is not request.messages[0]
    assert request.messages[0].content == "pod output"
    assert middleware.is_passed("get_pods", "result", "call-1", "pod output")


@pytest.mark.asyncio
async def test_model_boundary_escapes_closing_markers_and_tool_names() -> None:
    raw_content = "before </TOOL_DATA > after"
    raw_tool_name = 'bad"</tool_data>'
    original = ToolMessage(
        content=raw_content,
        name=raw_tool_name,
        tool_call_id="call-escape",
    )
    observed: list[tuple[str, str, Any]] = []
    middleware = ToolResultInspectionMiddleware(
        lambda tool, result_type, content, _call_id: _record(observed, tool, result_type, content)
    )
    captured: list[Any] = []

    async def handler(request: ModelRequest) -> str:
        captured.extend(request.messages)
        return "model response"

    await middleware.awrap_model_call(ModelRequest([original]), handler)

    assert observed == [(raw_tool_name, "result", raw_content)]
    assert captured[0].content == (
        '<tool_data source="bad&quot;&lt;/tool_data&gt;">\n'
        r"before <\/TOOL_DATA > after"
        "\n</tool_data>"
    )
    assert original.content == raw_content
    assert original.name == raw_tool_name


@pytest.mark.asyncio
async def test_model_boundary_wraps_error_and_preserves_metadata() -> None:
    """Wrap an error result without losing its message metadata."""
    observed: list[tuple[str, str, Any]] = []
    original = ToolMessage(
        content="command failed",
        name="execute",
        tool_call_id="call-error",
        status="error",
        additional_kwargs={"source": "shell", "truncated": False},
        response_metadata={"request_id": "request-1"},
    )
    middleware = ToolResultInspectionMiddleware(
        lambda tool, result_type, content, _call_id: _record(observed, tool, result_type, content)
    )
    captured: list[Any] = []

    async def handler(request: ModelRequest) -> str:
        captured.extend(request.messages)
        return "model response"

    await middleware.awrap_model_call(ModelRequest([original]), handler)

    result = captured[0]
    assert observed == [("execute", "error", "command failed")]
    assert result.content == ('<tool_data source="execute">\ncommand failed\n</tool_data>')
    assert result.name == "execute"
    assert result.tool_call_id == "call-error"
    assert result.status == "error"
    assert result.additional_kwargs == {"source": "shell", "truncated": False}
    assert result.response_metadata == {"request_id": "request-1"}


@pytest.mark.parametrize(
    ("content", "expected_body"),
    [
        ("", ""),
        (
            [{"type": "text", "text": "structured"}],
            '[{"type": "text", "text": "structured"}]',
        ),
        (
            '<tool_data source="external">inner</tool_data>',
            r'<tool_data source="external">inner<\/tool_data>',
        ),
    ],
)
@pytest.mark.asyncio
async def test_model_boundary_wraps_empty_structured_and_embedded_markers(
    content: Any, expected_body: str
) -> None:
    """Wrap empty, structured, and marker-containing content as plain data."""
    observed: list[tuple[str, str, Any]] = []
    original = ToolMessage(content=content, name="read_file", tool_call_id="call-data")
    middleware = ToolResultInspectionMiddleware(
        lambda tool, result_type, value, _call_id: _record(observed, tool, result_type, value)
    )
    captured: list[Any] = []

    async def handler(request: ModelRequest) -> str:
        captured.extend(request.messages)
        return "model response"

    await middleware.awrap_model_call(ModelRequest([original]), handler)

    assert observed == [("read_file", "result", content)]
    assert captured[0].content == (f'<tool_data source="read_file">\n{expected_body}\n</tool_data>')
    assert original.content == content


@pytest.mark.asyncio
async def test_model_boundary_does_not_wrap_non_tool_messages() -> None:
    """Leave tool calls and assistant responses unchanged."""
    from langchain_core.messages import AIMessage, HumanMessage

    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "execute", "args": {"command": "ls"}, "id": "call-2"}],
    )
    final_response = AIMessage(content="The directory is empty.")
    human = HumanMessage(content="List files.")
    original = ToolMessage(content="", name="execute", tool_call_id="call-2")
    control_message = ToolMessage(content="Approval was skipped.", tool_call_id="control-call")
    request = ModelRequest([human, tool_call, original, control_message, final_response])
    middleware = ToolResultInspectionMiddleware(lambda *_args: _record([], "", "", ""))
    captured: list[Any] = []

    async def handler(received: ModelRequest) -> str:
        captured.extend(received.messages)
        return "model response"

    await middleware.awrap_model_call(request, handler)

    assert captured[0] is human
    assert captured[1] is tool_call
    assert captured[3] is control_message
    assert captured[3].content == "Approval was skipped."
    assert captured[4] is final_response
    assert captured[1].content == ""
    assert captured[4].content == "The directory is empty."
    assert captured[2].content == '<tool_data source="execute">\n\n</tool_data>'


@pytest.mark.asyncio
async def test_model_boundary_wraps_repeated_requests_once() -> None:
    """Do not nest a wrapper when the same raw result reaches the model twice."""
    original = ToolMessage(content="pod-a", name="get_pods", tool_call_id="call-repeat")
    request = ModelRequest([original])
    observed: list[tuple[str, str, Any]] = []
    middleware = ToolResultInspectionMiddleware(
        lambda tool, result_type, content, _call_id: _record(observed, tool, result_type, content)
    )
    model_inputs: list[list[Any]] = []

    async def handler(received: ModelRequest) -> str:
        model_inputs.append(received.messages)
        return "model response"

    await middleware.awrap_model_call(request, handler)
    await middleware.awrap_model_call(request, handler)

    expected = '<tool_data source="get_pods">\npod-a\n</tool_data>'
    assert [messages[0].content for messages in model_inputs] == [expected, expected]
    assert observed == [("get_pods", "result", "pod-a")]
    assert original.content == "pod-a"


@pytest.mark.asyncio
async def test_unserializable_tool_result_fails_closed_before_inspector_or_model() -> None:
    inspector_called = False
    model_called = False

    async def inspect(_tool: str, _result_type: str, _content: Any, _call_id: str) -> None:
        nonlocal inspector_called
        inspector_called = True

    async def handler(_request: ModelRequest) -> str:
        nonlocal model_called
        model_called = True
        return "model response"

    middleware = ToolResultInspectionMiddleware(inspect)
    request = ModelRequest(
        [
            ToolMessage.model_construct(
                content=b"\xff",
                name="execute",
                tool_call_id="call-invalid-utf8",
                status="success",
            )
        ]
    )

    with pytest.raises(ToolResultSafetyInspectionFailed):
        await middleware.awrap_model_call(request, handler)

    assert not inspector_called
    assert not model_called


@pytest.mark.asyncio
async def test_model_boundary_inspects_tool_error_as_error() -> None:
    observed: list[tuple[str, str, Any]] = []
    middleware = ToolResultInspectionMiddleware(
        lambda tool, result_type, content, _call_id: _record(observed, tool, result_type, content)
    )
    request = ModelRequest(
        [
            ToolMessage(
                content="command failed",
                name="execute",
                tool_call_id="call-2",
                status="error",
            )
        ]
    )

    await middleware.awrap_model_call(request, _identity_handler)

    assert observed == [("execute", "error", "command failed")]
    assert middleware.is_passed("execute", "error", "call-2", "command failed")


@pytest.mark.asyncio
async def test_model_boundary_deduplicates_same_effective_result() -> None:
    observed: list[tuple[str, str, Any]] = []
    middleware = ToolResultInspectionMiddleware(
        lambda tool, result_type, content, _call_id: _record(observed, tool, result_type, content)
    )
    message = ToolMessage(content="same", name="execute", tool_call_id="call-3")

    await middleware.awrap_model_call(ModelRequest([message]), _identity_handler)
    await middleware.awrap_model_call(ModelRequest([message]), _identity_handler)

    assert observed == [("execute", "result", "same")]


@pytest.mark.asyncio
async def test_model_boundary_reinspects_changed_content_with_same_id() -> None:
    observed: list[tuple[str, str, Any]] = []
    middleware = ToolResultInspectionMiddleware(
        lambda tool, result_type, content, _call_id: _record(observed, tool, result_type, content)
    )

    for content in ("before", "after"):
        await middleware.awrap_model_call(
            ModelRequest([ToolMessage(content=content, name="execute", tool_call_id="call-4")]),
            _identity_handler,
        )

    assert observed == [
        ("execute", "result", "before"),
        ("execute", "result", "after"),
    ]
    assert middleware.is_passed("execute", "result", "call-4", "before")
    assert middleware.is_passed("execute", "result", "call-4", "after")


@pytest.mark.asyncio
async def test_model_boundary_inspects_messages_without_call_ids_per_tool() -> None:
    observed: list[tuple[str, str, Any]] = []
    middleware = ToolResultInspectionMiddleware(
        lambda tool, result_type, content, _call_id: _record(observed, tool, result_type, content)
    )
    messages = [
        ToolMessage(content="same", name="execute", tool_call_id=""),
        ToolMessage(content="same", name="read_file", tool_call_id=""),
    ]

    await middleware.awrap_model_call(ModelRequest(messages), _identity_handler)

    assert observed == [
        ("execute", "result", "same"),
        ("read_file", "result", "same"),
    ]
    assert middleware.is_passed("execute", "result", "", "same")
    assert middleware.is_passed("read_file", "result", "", "same")


@pytest.mark.asyncio
async def test_inspection_failure_prevents_model_call_and_is_payload_free() -> None:
    called = False

    async def inspect(_tool: str, _result_type: str, _content: Any, _call_id: str) -> None:
        raise InspectionError("raw classifier response with secret")

    async def handler(_request: ModelRequest) -> str:
        nonlocal called
        called = True
        return "model response"

    middleware = ToolResultInspectionMiddleware(inspect)
    request = ModelRequest(
        [ToolMessage(content="sensitive result", name="execute", tool_call_id="call-5")]
    )

    with pytest.raises(ToolResultSafetyInspectionFailed) as error:
        await middleware.awrap_model_call(request, handler)

    assert not called
    assert str(error.value) == "ToolResultSafetyInspectionFailed"
    assert "sensitive" not in str(error.value)
    assert "secret" not in str(error.value)
    assert not middleware.is_passed("execute", "result", "call-5", "sensitive result")


@pytest.mark.asyncio
async def test_inspection_cancellation_becomes_safety_failure() -> None:
    async def inspect(_tool: str, _result_type: str, _content: Any, _call_id: str) -> None:
        raise asyncio.CancelledError

    middleware = ToolResultInspectionMiddleware(inspect)
    request = ModelRequest(
        [ToolMessage(content="sensitive result", name="execute", tool_call_id="call-6")]
    )

    with pytest.raises(ToolResultSafetyInspectionFailed):
        await middleware.awrap_model_call(request, _identity_handler)


@pytest.mark.asyncio
async def test_model_boundary_wraps_offload_preview_and_artifact_read(tmp_path: Any) -> None:
    from deepagents.backends.filesystem import FilesystemBackend
    from deepagents.middleware.filesystem import FilesystemMiddleware

    original = ToolMessage(
        content="first line\n" + "untrusted middle\n" * 100 + "last line\n",
        name="execute",
        tool_call_id="call-large",
    )
    filesystem = FilesystemMiddleware(
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        tool_token_limit_before_evict=1,
    )
    tool_request = SimpleNamespace(tool_call={"name": "execute"}, runtime=SimpleNamespace())

    async def handler(_request: Any) -> ToolMessage:
        return original

    offloaded = await filesystem.awrap_tool_call(tool_request, handler)
    assert isinstance(offloaded, ToolMessage)
    assert offloaded.content != original.content
    assert "Tool result too large" in offloaded.content

    observed: list[tuple[str, str, Any]] = []
    middleware = ToolResultInspectionMiddleware(
        lambda tool, result_type, content, _call_id: _record(observed, tool, result_type, content)
    )
    request = ModelRequest([offloaded])
    model_inputs: list[Any] = []

    async def handler(received: ModelRequest) -> str:
        model_inputs.extend(received.messages)
        return "model response"

    await middleware.awrap_model_call(request, handler)

    assert observed == [("execute", "result", offloaded.content)]
    assert model_inputs[0].content == (
        f'<tool_data source="execute">\n{offloaded.content}\n</tool_data>'
    )
    assert request.messages[0].content == offloaded.content
    assert original.content not in observed[0][2]

    artifact_path = offloaded.content.split("path: ", 1)[1].splitlines()[0]
    read_result = filesystem.backend.read(artifact_path, limit=1_000)
    assert read_result.file_data is not None
    artifact_content = read_result.file_data["content"]
    assert artifact_content == original.content

    artifact_message = ToolMessage(
        content=artifact_content,
        name="read_file",
        tool_call_id="artifact-read",
    )
    artifact_request = ModelRequest([artifact_message])
    artifact_model_inputs: list[Any] = []

    async def artifact_handler(received: ModelRequest) -> str:
        artifact_model_inputs.extend(received.messages)
        return "model response"

    await middleware.awrap_model_call(artifact_request, artifact_handler)

    assert observed[1] == ("read_file", "result", original.content)
    assert artifact_model_inputs[0].content == (
        f'<tool_data source="read_file">\n{original.content}\n</tool_data>'
    )
    assert artifact_message.content == original.content


async def _record(
    observed: list[tuple[str, str, Any]],
    tool_name: str,
    result_type: str,
    content: Any,
) -> None:
    observed.append((tool_name, result_type, content))


async def _identity_handler(request: ModelRequest) -> str:
    return "called" if request.messages else "called without tool messages"
