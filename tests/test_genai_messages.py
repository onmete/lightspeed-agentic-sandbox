"""Provider-neutral GenAI payloads checked against the pinned OTel v1.41 schemas."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from lightspeed_agentic.genai_messages import encode_messages, encode_tool_object
from lightspeed_agentic.types import ProviderQueryOptions

FIXTURES = Path(__file__).parent / "fixtures"


def schema_for(kind):
    return json.loads((FIXTURES / f"genai-v1.41-{kind}.json").read_text())


@pytest.mark.parametrize("kind", ["input", "output"])
def test_ordered_messages_validate_against_pinned_schema(kind):
    user = {"role": "user", "parts": [{"type": "text", "content": "Olá 🌍"}]}
    assistant = {
        "role": "assistant",
        "parts": [
            {"type": "reasoning", "content": "Let me think"},
            {"type": "text", "content": ""},
            {
                "type": "tool_call",
                "id": "call-1",
                "name": "search",
                "arguments": {"term": "café"},
            },
        ],
    }
    result = {
        "role": "tool",
        "parts": [
            {
                "type": "tool_call_response",
                "id": "call-1",
                "response": "found 🌳",
            }
        ],
    }
    messages = [user, assistant, result]
    if kind == "output":
        messages = [{**message, "finish_reason": "stop"} for message in messages]

    encoded = encode_messages(messages)
    decoded = json.loads(encoded)
    Draft202012Validator(schema_for(kind)).validate(decoded)
    assert decoded == messages
    assert [part["type"] for part in decoded[1]["parts"]] == ["reasoning", "text", "tool_call"]
    assert '"content":"Olá 🌍"' in encoded
    assert '"content":""' in encoded
    assert "\\u00e1" not in encoded
    assert ": " not in encoded
    assert "\n" not in encoded


def test_tool_objects_preserve_structure_and_untruncated_opaque_text():
    structured = {"nested": ["héllo", {"ok": True}], "count": 7}
    assert encode_tool_object(structured) is structured
    assert encode_tool_object([1, {"a": 2}]) == {"content": [1, {"a": 2}]}
    assert encode_tool_object('{"nested":[1,2]}') == {"nested": [1, 2]}
    assert encode_tool_object(' ["π", false] ') == {"content": ["π", False]}
    assert encode_tool_object("null") == {"content": None}
    assert encode_tool_object(None) == {"content": None}
    assert encode_tool_object("12") == {"content": 12}
    assert encode_tool_object(12) == {"content": 12}
    assert encode_tool_object('"literal"') == {"content": "literal"}
    opaque = "unparsed 🌍: " + "x" * 5000
    assert encode_tool_object(opaque) == {"content": opaque}
    assert encode_tool_object("") == {"content": ""}
    assert encode_tool_object("not {json}") == {"content": "not {json}"}


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", '{"value":NaN}'])
def test_nonfinite_tool_text_remains_literal_content(value):
    assert encode_tool_object(value) == {"content": value}


@pytest.mark.parametrize("kind", ["input", "output"])
@pytest.mark.parametrize("value", [["π", False], 12, None, "literal", ""])
def test_tool_message_parts_preserve_non_object_values(kind, value):
    messages = [
        {
            "role": "assistant",
            "parts": [{"type": "tool_call", "id": "call-1", "name": "read", "arguments": value}],
        },
        {
            "role": "tool",
            "parts": [{"type": "tool_call_response", "id": "call-1", "response": value}],
        },
    ]
    if kind == "output":
        messages = [{**message, "finish_reason": "tool_call"} for message in messages]
    decoded = json.loads(encode_messages(messages))
    Draft202012Validator(schema_for(kind)).validate(decoded)
    assert decoded[0]["parts"][0]["arguments"] == value
    assert decoded[1]["parts"][0]["response"] == value
    assert decoded == messages


@pytest.mark.parametrize("kind", ["input", "output"])
def test_tool_response_requires_response_not_result(kind):
    schema = schema_for(kind)
    # The full message schema permits GenericPart extensions, so check the
    # named tool-response definition when enforcing its required properties.
    tool_response = Draft202012Validator(schema["$defs"]["ToolCallResponsePart"])
    part = {"type": "tool_call_response", "id": "call-1", "response": {"ok": True}}
    tool_response.validate(part)
    assert not tool_response.is_valid({"type": "tool_call_response", "id": "call-1"})
    assert not tool_response.is_valid(
        {"type": "tool_call_response", "id": "call-1", "result": {"ok": True}}
    )
    assert tool_response.is_valid({**part, "result": "provider extra"})


@pytest.mark.parametrize("kind", ["input", "output"])
def test_invalid_message_and_part_fields_are_rejected(kind):
    schema = schema_for(kind)
    validator = Draft202012Validator(schema)
    for bad_message in ({"parts": []}, {"role": "assistant"}):
        assert not validator.is_valid([bad_message])
    text = Draft202012Validator(schema["$defs"]["TextPart"])
    assert not text.is_valid({"type": "text"})
    assert not text.is_valid({"type": "text", "content": 42})
    assert not text.is_valid({"content": "hello"})
    tool_call = Draft202012Validator(schema["$defs"]["ToolCallRequestPart"])
    assert not tool_call.is_valid({"type": "tool_call", "arguments": {}})


def test_output_requires_finish_reason_and_valid_type():
    validator = Draft202012Validator(schema_for("output"))
    assert not validator.is_valid([{"role": "assistant", "parts": []}])
    assert not validator.is_valid([{"role": "assistant", "parts": [], "finish_reason": None}])
    validator.validate([{"role": "assistant", "parts": [], "finish_reason": "tool_call"}])


def test_provider_telemetry_remains_optional():
    args = ("prompt", "system", "model", 5, ["Bash"], "/workspace")
    assert ProviderQueryOptions(*args).telemetry is None
    telemetry = object()
    assert ProviderQueryOptions(*args, telemetry=telemetry).telemetry is telemetry
