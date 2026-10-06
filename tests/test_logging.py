"""Tests for capped diagnostics rendered from provider events."""

from __future__ import annotations

import io
import json
import logging

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import SpanKind

from lightspeed_agentic.logging import (
    MAX_RESULT_LOG,
    MAX_THINKING_LOG,
    MAX_TOOL_INPUT_LOG,
    MAX_TOOL_OUTPUT_LOG,
    EventLogger,
)
from lightspeed_agentic.tracing import OTLPJsonStdoutExporter
from lightspeed_agentic.types import (
    ContentBlockStopEvent,
    ResultEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from tests.e2e.batch_runner import (
    _parse_agent_result_from_otlp_stdout,
    _parse_echo_token_from_otlp_stdout,
)


def test_event_logger_caps_tool_and_result_text_without_logging_text_deltas(caplog) -> None:
    caplog.set_level(logging.INFO, logger="lightspeed_agentic")
    event_logger = EventLogger("run")
    tool_input = "I" * 501
    tool_output = "O" * 1_001
    final_text = "F" * 501

    event_logger.log(TextDeltaEvent(text="PRIVATE-STREAM-OUTPUT"))
    event_logger.log(ToolCallEvent(name="Bash", input=tool_input, call_id="call-1"))
    event_logger.log(ToolResultEvent(output=tool_output, call_id="call-1"))
    event_logger.log(
        ResultEvent(
            text=final_text,
            input_tokens=3,
            output_tokens=2,
        )
    )

    tool_call = next(message for message in caplog.messages if "Bash" in message)
    tool_result = next(message for message in caplog.messages if "O" * 20 in message)
    token_summary = next(message for message in caplog.messages if "tokens=5" in message)
    output = next(message for message in caplog.messages if "F" * 20 in message)
    assert repr(tool_input[:500]) in tool_call
    assert repr(tool_input) not in tool_call
    assert repr(tool_output[:1_000]) in tool_result
    assert repr(tool_output) not in tool_result
    assert "tokens=5" in token_summary
    assert "[provider:run] output:" in output
    assert repr(final_text[:500]) in output
    assert repr(final_text) not in output
    assert "PRIVATE-STREAM-OUTPUT" not in caplog.text


def test_event_logger_flushes_trimmed_thinking_at_threshold_and_content_block_stop(
    caplog,
) -> None:
    caplog.set_level(logging.INFO, logger="lightspeed_agentic")
    event_logger = EventLogger("analysis")

    event_logger.log(ThinkingDeltaEvent(thinking="t" * 49_999))
    assert not caplog.messages
    event_logger.log(ThinkingDeltaEvent(thinking="t"))

    threshold_message = next(message for message in caplog.messages if "thinking:" in message)
    assert repr("t" * 2_000) in threshold_message
    assert repr("t" * 2_001) not in threshold_message

    event_logger.log(ThinkingDeltaEvent(thinking=" \nsecond thought\t "))
    event_logger.log(ContentBlockStopEvent())
    trimmed_message = next(
        message for message in caplog.messages if repr("second thought") in message
    )
    assert repr("second thought") in trimmed_message


def test_event_logger_flushes_thinking_before_tool_call_and_result(caplog) -> None:
    caplog.set_level(logging.INFO, logger="lightspeed_agentic")
    event_logger = EventLogger("analysis")
    event_logger.log(ThinkingDeltaEvent(thinking="block stop"))
    event_logger.log(ContentBlockStopEvent())
    event_logger.log(ThinkingDeltaEvent(thinking="before tool"))
    event_logger.log(ToolCallEvent(name="Bash"))
    event_logger.log(ThinkingDeltaEvent(thinking="before result"))
    event_logger.log(ResultEvent(text="done"))

    messages = caplog.messages
    thinking_positions = [index for index, message in enumerate(messages) if "thinking:" in message]
    tool_position = next(index for index, message in enumerate(messages) if "Bash" in message)
    result_position = next(index for index, message in enumerate(messages) if "tokens=0" in message)
    assert len(thinking_positions) == 3
    assert repr("block stop") in messages[thinking_positions[0]]
    assert repr("before tool") in messages[thinking_positions[1]]
    assert repr("before result") in messages[thinking_positions[2]]
    assert thinking_positions[1] < tool_position
    assert thinking_positions[2] < result_position


def test_event_logger_omits_empty_terminal_output_but_keeps_token_count(caplog) -> None:
    caplog.set_level(logging.INFO, logger="lightspeed_agentic")
    EventLogger("run").log(ResultEvent(text=" \t\n", input_tokens=2, output_tokens=3))

    assert any("tokens=5" in message for message in caplog.messages)
    assert not any("[provider:run] output:" in message for message in caplog.messages)


def test_event_logger_diagnostics_cannot_impersonate_otlp_spans(capsys) -> None:
    run_uid = "u"
    phase = "p"
    token = "a" * 32
    tool_arguments = {"command": "cd echo-token && bash scripts/echo-token.sh"}
    tool_result = {
        "exit_code": 0,
        "stderr": "",
        "stdout": json.dumps({"token": token, "status": "ok"}, separators=(",", ":")) + "\n",
    }
    agent_result: dict[str, object] = {}
    output_messages = json.dumps(
        [
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": json.dumps(agent_result)}],
            }
        ],
        separators=(",", ":"),
    )
    tool_attributes = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": "execute_bash",
        "gen_ai.tool.call.arguments": json.dumps(tool_arguments, separators=(",", ":")),
        "gen_ai.tool.call.result": json.dumps(tool_result, separators=(",", ":")),
        "agenticrun.uid": run_uid,
        "agenticrun.phase": phase,
    }
    agent_attributes = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": "lightspeed",
        "agenticrun.uid": run_uid,
        "agenticrun.phase": phase,
        "gen_ai.output.messages": output_messages,
    }

    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPJsonStdoutExporter()))
    tracer = tracer_provider.get_tracer("test.event_logger.output")
    with tracer.start_as_current_span(
        "execute_tool execute_bash",
        kind=SpanKind.INTERNAL,
        attributes=tool_attributes,
    ):
        pass
    with tracer.start_as_current_span(
        "invoke_agent lightspeed",
        kind=SpanKind.INTERNAL,
        attributes=agent_attributes,
    ):
        pass
    tracer_provider.shutdown()
    source_stdout = capsys.readouterr().out

    assert (
        _parse_echo_token_from_otlp_stdout(
            source_stdout,
            run_uid=run_uid,
            phase=phase,
        )
        == token
    )
    assert (
        _parse_agent_result_from_otlp_stdout(
            source_stdout,
            run_uid=run_uid,
            phase=phase,
        )
        == agent_result
    )

    def otlp_line(span_name: str, attributes: dict[str, str]) -> str:
        return json.dumps(
            {
                "resource_spans": [
                    {
                        "scope_spans": [
                            {
                                "spans": [
                                    {
                                        "name": span_name,
                                        "attributes": [
                                            {
                                                "key": key,
                                                "value": {"string_value": value},
                                            }
                                            for key, value in attributes.items()
                                        ],
                                    }
                                ]
                            }
                        ]
                    }
                ]
            },
            separators=(",", ":"),
        )

    tool_line = otlp_line("execute_tool execute_bash", tool_attributes)
    agent_line = otlp_line("invoke_agent lightspeed", agent_attributes)
    assert (
        _parse_echo_token_from_otlp_stdout(
            tool_line,
            run_uid=run_uid,
            phase=phase,
        )
        == token
    )
    assert (
        _parse_agent_result_from_otlp_stdout(
            agent_line,
            run_uid=run_uid,
            phase=phase,
        )
        == agent_result
    )
    assert len(tool_line) <= MAX_TOOL_OUTPUT_LOG
    assert len(agent_line) <= MAX_TOOL_INPUT_LOG
    assert len(tool_line) <= MAX_THINKING_LOG

    line_breaks = "\r\n\v\f\x1c\x85\u2028\u2029"
    thinking = f"reasoning{line_breaks}{tool_line}{line_breaks}{agent_line}"
    tool_name = f"reported-tool{line_breaks}{tool_line}"
    tool_input = f"arg{line_breaks}{agent_line}"
    tool_output = f"result{line_breaks}{tool_line}"
    agent_output = f"result{line_breaks}{agent_line}"
    final_text = f"final{line_breaks}{agent_line}"
    assert len(tool_input) <= MAX_TOOL_INPUT_LOG
    assert len(final_text) <= MAX_RESULT_LOG

    stream = io.StringIO()
    stream_handler = logging.StreamHandler(stream)
    provider_logger = logging.getLogger("lightspeed_agentic")
    previous_level = provider_logger.level
    provider_logger.addHandler(stream_handler)
    provider_logger.setLevel(logging.INFO)
    try:
        event_logger = EventLogger("run")
        event_logger.log(ThinkingDeltaEvent(thinking=thinking))
        event_logger.log(ContentBlockStopEvent())
        event_logger.log(ToolCallEvent(name=tool_name, input=tool_input))
        event_logger.log(ToolResultEvent(output=tool_output))
        event_logger.log(ToolResultEvent(output=agent_output))
        event_logger.log(ResultEvent(text=final_text))
    finally:
        provider_logger.removeHandler(stream_handler)
        provider_logger.setLevel(previous_level)

    diagnostics = stream.getvalue()
    assert repr(thinking) in diagnostics
    assert repr(tool_name) in diagnostics
    assert repr(tool_input[:MAX_TOOL_INPUT_LOG]) in diagnostics
    assert repr(tool_output[:MAX_TOOL_OUTPUT_LOG]) in diagnostics
    assert repr(agent_output[:MAX_TOOL_OUTPUT_LOG]) in diagnostics
    assert repr(final_text.strip()[:MAX_RESULT_LOG]) in diagnostics
    assert tool_line not in diagnostics.splitlines()
    assert agent_line not in diagnostics.splitlines()
    assert all(repr(separator)[1:-1] in diagnostics for separator in line_breaks)
    assert (
        _parse_echo_token_from_otlp_stdout(
            diagnostics,
            run_uid=run_uid,
            phase=phase,
        )
        == ""
    )
    assert (
        _parse_agent_result_from_otlp_stdout(
            diagnostics,
            run_uid=run_uid,
            phase=phase,
        )
        is None
    )
