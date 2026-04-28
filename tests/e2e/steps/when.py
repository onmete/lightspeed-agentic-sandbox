"""When step definitions — actions that invoke the service."""

from __future__ import annotations

from pytest_bdd import when

from ..schemas import ANALYSIS_SCHEMA


@when("I send an analysis query with the output schema", target_fixture="query_result")
async def when_send_analysis(analyze_runner, output_schema):
    return await analyze_runner(
        query="Analyze a test pod that is in CrashLoopBackOff state in the default namespace.",
        system_prompt="You are an SRE assistant analyzing OpenShift issues.",
        output_schema=output_schema,
    )


@when(
    "I send an execution query with the output schema and context",
    target_fixture="query_result",
)
async def when_send_execution(execute_runner, output_schema, query_context):
    return await execute_runner(
        query="Execute the approved remediation for the CrashLoopBackOff pod.",
        system_prompt="You are an SRE assistant executing approved remediations.",
        output_schema=output_schema,
        context=query_context,
    )


@when(
    "I send a verification query with the output schema and context",
    target_fixture="query_result",
)
async def when_send_verification(verify_runner, output_schema, query_context):
    return await verify_runner(
        query="Verify that the remediation for the CrashLoopBackOff pod was successful.",
        system_prompt="You are an SRE assistant verifying remediation outcomes.",
        output_schema=output_schema,
        context=query_context,
    )


@when("I send a chat message", target_fixture="chat_result")
async def when_send_chat(chat_runner):
    return await chat_runner(
        message="What namespaces exist in this cluster?",
    )


@when(
    "I send a chat message and receive a conversation ID",
    target_fixture="first_chat_result",
)
async def when_send_chat_get_id(chat_runner):
    return await chat_runner(
        message="Hello, what can you help me with?",
    )


@when("I send a follow-up message with that conversation ID", target_fixture="followup_result")
async def when_send_followup(chat_runner, first_chat_result):
    done_events = [e for e in first_chat_result.events if e.event == "done"]
    assert done_events, "First chat did not produce a done event"
    last_data = done_events[-1].data
    conv_id = last_data.get("conversationId") if isinstance(last_data, dict) else None
    assert conv_id, "First chat done event missing conversationId"
    return await chat_runner(
        message="Tell me more about that.",
        conversation_id=conv_id,
    )


@when("I send a chat message that triggers tool use", target_fixture="chat_result")
async def when_send_chat_with_tools(chat_runner):
    return await chat_runner(
        message="List the files in the current directory using the bash tool.",
    )


@when("the stream contains \"tool_call\" events", target_fixture="has_tool_calls")
def when_check_tool_calls(chat_result):
    tool_calls = [e for e in chat_result.events if e.event == "tool_call"]
    if not tool_calls:
        import pytest

        pytest.skip("Provider did not use tools for this message")
    return True


@when("I ask the agent to find the hidden token", target_fixture="query_result")
async def when_find_token(analyze_runner):
    return await analyze_runner(
        query="Find the hidden token using the 'find-token' skill.",
        system_prompt="You are an assistant. Use your available skills to accomplish tasks.",
        output_schema=ANALYSIS_SCHEMA,
    )
