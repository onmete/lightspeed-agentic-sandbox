"""Then step definitions — assertions on service responses."""

from __future__ import annotations

import jsonschema
from pytest_bdd import parsers, then

from ..runner import assert_tool_token
from ..schemas import ANALYSIS_SCHEMA, EXECUTION_SCHEMA, VERIFICATION_SCHEMA


@then("the response status is 200")
def then_status_200(query_result):
    assert query_result.error is None, f"Request failed: {query_result.error}"
    assert query_result.status_code == 200, f"Expected 200, got {query_result.status_code}"


@then("the response body validates against the analysis schema")
def then_validates_analysis(query_result):
    jsonschema.validate(query_result.raw, ANALYSIS_SCHEMA)


@then("the response body validates against the execution schema")
def then_validates_execution(query_result):
    jsonschema.validate(query_result.raw, EXECUTION_SCHEMA)


@then("the response body validates against the verification schema")
def then_validates_verification(query_result):
    jsonschema.validate(query_result.raw, VERIFICATION_SCHEMA)


@then(parsers.parse('the response contains "{field}" as a non-empty array'))
def then_non_empty_array(query_result, field):
    assert field in query_result.raw, f"Missing field '{field}' in response"
    value = query_result.raw[field]
    assert isinstance(value, list), f"Expected '{field}' to be a list, got {type(value)}"
    assert len(value) > 0, f"Expected '{field}' to be non-empty"


@then(parsers.parse('the response contains "{field}" as an array'))
def then_is_array(query_result, field):
    assert field in query_result.raw, f"Missing field '{field}' in response"
    assert isinstance(query_result.raw[field], list), (
        f"Expected '{field}' to be a list, got {type(query_result.raw[field])}"
    )


@then(parsers.parse('the response contains "{field}" as a boolean'))
def then_is_boolean(query_result, field):
    assert field in query_result.raw, f"Missing field '{field}' in response"
    assert isinstance(query_result.raw[field], bool), (
        f"Expected '{field}' to be a boolean, got {type(query_result.raw[field])}"
    )


@then(parsers.parse('each option has required fields "{f1}" "{f2}" "{f3}"'))
def then_options_have_fields(query_result, f1, f2, f3):
    for i, option in enumerate(query_result.raw.get("options", [])):
        for field_name in (f1, f2, f3):
            assert field_name in option, f"Option[{i}] missing required field '{field_name}'"


@then(parsers.parse('the response contains "{parent}" with boolean "{field}"'))
def then_nested_boolean(query_result, parent, field):
    assert parent in query_result.raw, f"Missing '{parent}' in response"
    obj = query_result.raw[parent]
    assert isinstance(obj, dict), f"Expected '{parent}' to be a dict"
    assert field in obj, f"Missing '{field}' in '{parent}'"
    assert isinstance(obj[field], bool), (
        f"Expected '{parent}.{field}' to be boolean, got {type(obj[field])}"
    )


@then(parsers.parse('the response contains wrapper field "{field}"'))
def then_wrapper_field(query_result, field):
    assert field in query_result.raw, f"Missing wrapper field '{field}' in response"


@then("all analysis schema fields are present in the response")
def then_all_schema_fields(query_result):
    for required_field in ANALYSIS_SCHEMA.get("required", []):
        assert required_field in query_result.raw, (
            f"Schema-required field '{required_field}' missing from response"
        )


# --- Chat SSE assertions ---


@then(parsers.parse('the stream starts with a "{event_type}" event'))
def then_stream_starts_with(chat_result, event_type):
    assert chat_result.events, "No SSE events received"
    assert chat_result.events[0].event == event_type, (
        f"Expected first event to be '{event_type}', got '{chat_result.events[0].event}'"
    )


@then(parsers.parse('the stream contains at least one "{event_type}" event with content'))
def then_stream_has_event(chat_result, event_type):
    matching = [e for e in chat_result.events if e.event == event_type]
    assert matching, f"No '{event_type}' events in stream"
    has_content = any(
        (isinstance(e.data, dict) and e.data.get("content"))
        or (isinstance(e.data, str) and e.data)
        for e in matching
    )
    assert has_content, f"No '{event_type}' event has non-empty content"


@then(parsers.parse('the stream ends with a "{event_type}" event'))
def then_stream_ends_with(chat_result, event_type):
    assert chat_result.events, "No SSE events received"
    assert chat_result.events[-1].event == event_type, (
        f"Expected last event to be '{event_type}', got '{chat_result.events[-1].event}'"
    )


@then(parsers.parse('the "{event_type}" event contains "{field}"'))
def then_event_contains_field(chat_result, event_type, field):
    matching = [e for e in chat_result.events if e.event == event_type]
    assert matching, f"No '{event_type}' events in stream"
    last = matching[-1]
    assert isinstance(last.data, dict), f"'{event_type}' event data is not a dict"
    assert field in last.data, f"'{event_type}' event missing field '{field}'"


@then(parsers.parse('the follow-up stream ends with a "{event_type}" event'))
def then_followup_ends_with(followup_result, event_type):
    assert followup_result.events, "No SSE events in follow-up"
    assert followup_result.events[-1].event == event_type, (
        f"Expected last follow-up event to be '{event_type}', "
        f"got '{followup_result.events[-1].event}'"
    )


@then(parsers.parse('the follow-up "{event_type}" event contains the same conversation ID'))
def then_followup_same_conv_id(first_chat_result, followup_result, event_type):
    first_done = [e for e in first_chat_result.events if e.event == event_type]
    followup_done = [e for e in followup_result.events if e.event == event_type]
    assert first_done, "Missing done event in first chat"
    assert followup_done, "Missing done event in follow-up"
    first_data = first_done[-1].data
    followup_data = followup_done[-1].data
    first_id = first_data.get("conversationId") if isinstance(first_data, dict) else None
    followup_id = followup_data.get("conversationId") if isinstance(followup_data, dict) else None
    assert first_id == followup_id, (
        f"Conversation ID mismatch: first={first_id}, followup={followup_id}"
    )


@then(parsers.parse('each "{event_type}" event has "{f1}" and "{f2}" fields'))
def then_events_have_fields(chat_result, event_type, f1, f2):
    matching = [e for e in chat_result.events if e.event == event_type]
    for i, event in enumerate(matching):
        assert isinstance(event.data, dict), f"{event_type}[{i}] data is not a dict"
        assert f1 in event.data, f"{event_type}[{i}] missing field '{f1}'"
        assert f2 in event.data, f"{event_type}[{i}] missing field '{f2}'"


@then(parsers.parse('each "{event_type}" event has an "{field}" field'))
def then_events_have_field(chat_result, event_type, field):
    matching = [e for e in chat_result.events if e.event == event_type]
    for i, event in enumerate(matching):
        assert isinstance(event.data, dict), f"{event_type}[{i}] data is not a dict"
        assert field in event.data, f"{event_type}[{i}] missing field '{field}'"


# --- Skill invocation assertions ---


@then("the agent response validates against the analysis schema")
def then_agent_validates_analysis(query_result):
    assert query_result.error is None, f"Request failed: {query_result.error}"
    jsonschema.validate(query_result.raw, ANALYSIS_SCHEMA)


@then("the hidden token file exists in the workspace output")
def then_token_file_exists(eval_workspace):
    matches = list(eval_workspace.rglob(".hidden_token"))
    assert matches, f"No .hidden_token file found in {eval_workspace}"


@then("the response contains both tokens from the hidden token file")
def then_response_has_tokens(query_result, eval_workspace, provider):
    assert_tool_token(eval_workspace, ".hidden_token", query_result, provider, "find-token.sh")
