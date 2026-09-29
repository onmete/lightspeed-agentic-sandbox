"""DeepAgents provider — wraps langchain-ai/deepagents for Anthropic model support.

Uses create_deep_agent() with LocalShellBackend for shell + filesystem access,
native skills loading, and v3 event streaming for event mapping.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import uuid
from collections import Counter, defaultdict, deque
from collections.abc import AsyncIterator
from typing import Any, Literal, cast
from uuid import UUID

from lightspeed_agentic.skills import has_skills
from lightspeed_agentic.types import (
    MAX_TOOL_RETURN_CHARS,
    AgentProvider,
    ContentBlockStopEvent,
    ProviderEvent,
    ProviderQueryOptions,
    ResultEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    stringify,
)

# Provider SDK imports (deepagents, langchain-*, MCP) stay inside functions:
# - _resolve_model loads only the active backend branch (Vertex / Bedrock / direct).
# - query() / shape / MCP load their SDKs on first use, not at module import.
# That keeps optional-extra isolation and avoids importing unused backends; it does
# not skip work on the hot path once a run is underway.

logger = logging.getLogger(__name__)


_JSON_SCHEMA_TYPE_MAP: dict[str, type[Any]] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def _anthropic_backend() -> Literal["vertex", "bedrock", "direct"]:
    """Resolve Anthropic backend from env; reject conflicting Vertex/Bedrock flags."""
    use_vertex = os.environ.get("CLAUDE_CODE_USE_VERTEX") == "1"
    use_bedrock = os.environ.get("CLAUDE_CODE_USE_BEDROCK") == "1"
    if use_vertex and use_bedrock:
        raise ValueError("CLAUDE_CODE_USE_VERTEX and CLAUDE_CODE_USE_BEDROCK cannot both be set")
    if use_vertex:
        return "vertex"
    if use_bedrock:
        return "bedrock"
    return "direct"


def _resolve_model(model: str, reasoning_config: dict[str, Any] | None = None) -> Any:
    """Build a LangChain chat model instance based on env vars set by config.py."""
    from functools import cached_property

    thinking = reasoning_config.get("thinking") if reasoning_config else None
    backend = _anthropic_backend()

    if backend == "vertex":
        from langchain_google_vertexai.model_garden import ChatAnthropicVertex

        kwargs: dict[str, Any] = {
            "model_name": model,
            "project": os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", ""),
            "location": os.environ.get("CLOUD_ML_REGION", "us-east5"),
        }
        if thinking:
            kwargs["thinking"] = thinking
        from lightspeed_agentic.tls import create_async_http_client, create_http_client

        kwargs["http_client"] = create_http_client()
        kwargs["async_http_client"] = create_async_http_client()
        return ChatAnthropicVertex(**kwargs)

    if backend == "bedrock":
        # langchain_aws uses these Anthropic Bedrock clients internally, but does not
        # expose a stable injection point for a custom HTTPX client. Keep this import
        # aligned with the installed anthropic SDK version.
        from anthropic.lib.bedrock._client import AnthropicBedrock, AsyncAnthropicBedrock
        from langchain_aws import ChatAnthropicBedrock

        from lightspeed_agentic.tls import create_async_http_client, create_http_client

        class TLSChatAnthropicBedrock(ChatAnthropicBedrock):
            @cached_property
            def _client(self) -> Any:
                return AnthropicBedrock(**self._client_params, http_client=create_http_client())

            @cached_property
            def _async_client(self) -> Any:
                return AsyncAnthropicBedrock(
                    **self._client_params,
                    http_client=create_async_http_client(),
                )

        kwargs = {
            "model": model,
            "region_name": os.environ.get("AWS_REGION", "us-east-1"),
        }
        if thinking:
            kwargs["thinking"] = thinking
        if not isinstance(ChatAnthropicBedrock, type):
            return ChatAnthropicBedrock(**kwargs)
        return TLSChatAnthropicBedrock(**kwargs)

    from anthropic import Anthropic, AsyncAnthropic
    from langchain_anthropic import ChatAnthropic

    from lightspeed_agentic.tls import create_async_http_client, create_http_client

    class TLSChatAnthropic(ChatAnthropic):
        @cached_property
        def _client(self) -> Any:
            return Anthropic(**self._client_params, http_client=create_http_client())

        @cached_property
        def _async_client(self) -> Any:
            return AsyncAnthropic(**self._client_params, http_client=create_async_http_client())

    kwargs = {"model": model}
    if thinking:
        kwargs["thinking"] = thinking

    # Support bearer token auth for vLLM and other Anthropic-compatible endpoints
    default_headers = {}
    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if auth_token:
        default_headers["Authorization"] = f"Bearer {auth_token}"
    if default_headers:
        kwargs["default_headers"] = default_headers

    if not isinstance(ChatAnthropic, type):
        return ChatAnthropic(**kwargs)
    return TLSChatAnthropic(**kwargs)


async def _close_model_clients(model: Any) -> None:
    """Close already-created sync and async clients without triggering lazy creation."""
    clients: list[Any] = []
    model_state = getattr(model, "__dict__", {})
    for name in ("_async_client", "async_client", "_client", "client"):
        if name in model_state and model_state[name] not in clients:
            clients.append(model_state[name])

    for client in clients:
        close = getattr(client, "aclose", None) or getattr(client, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result


def _json_schema_to_pydantic(schema: dict[str, Any], name: str = "OutputModel") -> Any:
    """Convert a JSON schema dict to a dynamic Pydantic model."""
    import pydantic

    if "properties" not in schema:
        raise ValueError(f"Schema {name!r} missing 'properties'")

    props = schema["properties"]
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}

    for field_name, field_schema in props.items():
        field_type = _resolve_field_type(field_schema, field_name)
        if field_name in required:
            fields[field_name] = (field_type, ...)
        else:
            fields[field_name] = (field_type | None, None)

    return pydantic.create_model(name, **fields)


def _resolve_field_type(schema: dict[str, Any], name: str) -> Any:
    json_type = schema.get("type", "string")

    if json_type == "object":
        return _json_schema_to_pydantic(schema, name.title().replace("_", ""))

    if json_type == "array":
        if "items" not in schema:
            raise ValueError(f"Array field {name!r} missing 'items'")
        item_type = _resolve_field_type(schema["items"], f"{name}_item")
        return list[item_type]  # type: ignore[valid-type]

    if "enum" in schema:
        return Literal[tuple(schema["enum"])]

    return _JSON_SCHEMA_TYPE_MAP.get(json_type, str)


def _model_usage(msg: Any) -> dict[str, int]:
    usage = getattr(msg, "usage_metadata", None) or {}
    details = usage.get("output_token_details") or {}
    result: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens"):
        value = usage.get(key)
        if type(value) is int:
            result[key] = value
    reasoning = details.get("reasoning")
    if type(reasoning) is int:
        result["reasoning_tokens"] = reasoning
    return result


def _response_model(msg: Any) -> str | None:
    metadata = getattr(msg, "response_metadata", None) or {}
    if not isinstance(metadata, dict):
        return None
    return metadata.get("model_name") or metadata.get("model")


def _message_parts(message: Any, ids: Any = None, *, output: bool = False) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    content = getattr(message, "content", "")
    blocks = getattr(message, "content_blocks", None)
    if blocks:
        for block in blocks:
            kind = block.get("type") if isinstance(block, dict) else getattr(block, "type", "")
            get = (
                block.get
                if isinstance(block, dict)
                else lambda key, default=None, _block=block: getattr(_block, key, default)
            )
            if kind == "non_standard":
                wrapped = get("value", {})
                if not isinstance(wrapped, dict):
                    continue
                get = wrapped.get
                kind = get("type", "")
            if kind in ("reasoning", "thinking"):
                reasoning = get("reasoning", get("thinking", ""))
                if isinstance(reasoning, str) and reasoning:
                    parts.append({"type": "reasoning", "content": reasoning})
            elif kind == "text":
                parts.append({"type": "text", "content": get("text", "")})
            elif kind in ("tool_call", "tool_use"):
                parts.append(
                    {
                        "type": "tool_call",
                        "id": get("id"),
                        "name": get("name", ""),
                        "arguments": get("args", get("input", {})),
                    }
                )
    elif content:
        if isinstance(content, str):
            parts.append({"type": "text", "content": content})
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    parts.append({"type": "text", "content": block})
                elif isinstance(block, dict):
                    if block.get("type") in ("thinking", "reasoning"):
                        reasoning = block.get("thinking", block.get("reasoning", ""))
                        if isinstance(reasoning, str) and reasoning:
                            parts.append({"type": "reasoning", "content": reasoning})
                    elif block.get("type") == "text":
                        parts.append({"type": "text", "content": block.get("text", "")})
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:

        def call_key(name: str, arguments: Any) -> tuple[str, str]:
            return (
                name,
                json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str),
            )

        call_indexes_by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
        explicit_call_indexes: dict[str, deque[int]] = defaultdict(deque)
        for index, call in enumerate(tool_calls):
            key = call_key(call.get("name", ""), call.get("args", {}))
            call_indexes_by_key[key].append(index)
            if call_id := call.get("id"):
                explicit_call_indexes[call_id].append(index)

        matched_call_positions: dict[int, int] = {}
        unmatched_block_positions: dict[tuple[str, str], list[int]] = defaultdict(list)
        for position, part in enumerate(parts):
            if part["type"] != "tool_call":
                continue
            key = call_key(part["name"], part["arguments"])
            indexes = explicit_call_indexes[part["id"]] if part.get("id") else None
            if indexes:
                index = indexes.popleft()
                matched_call_positions[index] = position
                call = tool_calls[index]
                parts[position]["id"] = call.get("id")
                parts[position]["name"] = call.get("name", "")
                parts[position]["arguments"] = call.get("args", {})
            else:
                unmatched_block_positions[key].append(position)

        duplicate_block_positions: set[int] = set()
        for key, positions in unmatched_block_positions.items():
            call_indexes = call_indexes_by_key.get(key)
            if not call_indexes:
                continue
            unmatched_calls = [
                index for index in call_indexes if index not in matched_call_positions
            ]
            paired_count = 0
            if unmatched_calls and (
                len(unmatched_calls) == 1 or len(positions) >= len(unmatched_calls)
            ):
                paired_count = min(len(positions), len(unmatched_calls))
                for position, index in zip(
                    positions[:paired_count], unmatched_calls[:paired_count], strict=True
                ):
                    matched_call_positions[index] = position
                    call = tool_calls[index]
                    parts[position]["id"] = call.get("id")
                    parts[position]["name"] = call.get("name", "")
                    parts[position]["arguments"] = call.get("args", {})
            duplicate_block_positions.update(positions[paired_count:])

        following_positions: dict[int, int | None] = {}
        following_position = None
        for index in range(len(tool_calls) - 1, -1, -1):
            following_positions[index] = following_position
            if index in matched_call_positions:
                following_position = matched_call_positions[index]

        insert_before: dict[int, list[dict[str, Any]]] = defaultdict(list)
        insert_after: dict[int, list[dict[str, Any]]] = defaultdict(list)
        trailing_calls: list[dict[str, Any]] = []
        previous_position = None
        for index, call in enumerate(tool_calls):
            if index in matched_call_positions:
                previous_position = matched_call_positions[index]
                continue
            name, arguments = call.get("name", ""), call.get("args", {})
            part = {
                "type": "tool_call",
                "id": call.get("id"),
                "name": name,
                "arguments": arguments,
            }
            next_position = following_positions[index]
            if next_position is not None:
                insert_before[next_position].append(part)
            elif previous_position is not None:
                insert_after[previous_position].append(part)
            else:
                trailing_calls.append(part)

        if insert_before or insert_after or trailing_calls or duplicate_block_positions:
            merged_parts: list[dict[str, Any]] = []
            for position, part in enumerate(parts):
                merged_parts.extend(insert_before.get(position, ()))
                if position not in duplicate_block_positions:
                    merged_parts.append(part)
                merged_parts.extend(insert_after.get(position, ()))
            merged_parts.extend(trailing_calls)
            parts = merged_parts
    for part in parts:
        if part["type"] == "tool_call" and not part["id"]:
            part["id"] = (
                (
                    ids.output_id(part["name"], part["arguments"])
                    if output
                    else ids.input_id(part["name"], part["arguments"])
                )
                if ids
                else ""
            )
    if getattr(message, "type", "") == "tool":
        call_id = getattr(message, "tool_call_id", "") or (ids.result_id(message) if ids else "")
        return [{"type": "tool_call_response", "id": call_id, "response": content}]
    return parts


def _genai_tool_definitions(tools: Any) -> list[dict[str, Any]] | None:
    """Normalize LangChain tool definitions to the GenAI semantic-convention shape."""
    if not isinstance(tools, list):
        return None
    definitions = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        definition = function if isinstance(function, dict) else tool
        tool_type = tool.get("type") or "function"
        name = definition.get("name")
        if not isinstance(tool_type, str) or not isinstance(name, str):
            continue
        normalized: dict[str, Any] = {"type": tool_type, "name": name}
        description = definition.get("description")
        if isinstance(description, str):
            normalized["description"] = description
        parameters = definition.get("parameters")
        if parameters is None:
            parameters = definition.get("input_schema")
        if isinstance(parameters, dict):
            normalized["parameters"] = parameters
        definitions.append(normalized)
    return definitions or None


def _telemetry_handler(observer: Any, requested_model: str) -> Any:
    """LangChain callback boundaries see the final post-middleware model request."""
    from langchain_core.callbacks import AsyncCallbackHandler
    from langchain_core.messages import BaseMessage

    class Handler(AsyncCallbackHandler):
        def __init__(self) -> None:
            self.models: dict[Any, list[tuple[object, str]]] = {}
            self.tools: dict[Any, tuple[object, str]] = {}
            self.pending_tool_results: dict[str, tuple[object, Any]] = {}
            self.model_input_tool_results: dict[str, Any] = {}
            self.proposed_ids: dict[tuple[str, str], deque[str]] = defaultdict(deque)
            self.pending_ids: dict[tuple[str, str], deque[str]] = defaultdict(deque)
            self.preassigned_ids: dict[tuple[str, str], deque[str]] = defaultdict(deque)
            self.history_ids: dict[tuple[str, str], list[str]] = defaultdict(list)
            self.completed_ids: dict[str, list[str]] = defaultdict(list)
            self.input_call_positions: Counter[tuple[str, str]] = Counter()
            self.input_call_totals: Counter[tuple[str, str]] = Counter()
            self.input_history_count: dict[tuple[str, str], int] = {}
            self.input_result_positions: Counter[str] = Counter()
            self.input_result_totals: Counter[str] = Counter()
            self.stream_result_positions: Counter[str] = Counter()
            self.usage_seen = False
            self.model_completed = False
            self.input_tokens = 0
            self.output_tokens = 0
            self.reasoning_tokens = 0
            self.response_model = ""

        def key(self, name: str, arguments: Any) -> tuple[str, str]:
            return name, json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)

        def call_id(self, name: str, arguments: Any) -> str:
            key = self.key(name, arguments)
            if self.preassigned_ids[key]:
                return self.preassigned_ids[key].popleft()
            call_id = uuid.uuid4().hex
            self.proposed_ids[key].append(call_id)
            return call_id

        def output_id(self, name: str, arguments: Any) -> str:
            key = self.key(name, arguments)
            call_id = (
                self.proposed_ids[key].popleft() if self.proposed_ids[key] else uuid.uuid4().hex
            )
            self.pending_ids[key].append(call_id)
            self.history_ids[key].append(call_id)
            return call_id

        def bind_output_id(self, name: str, arguments: Any) -> str:
            key = self.key(name, arguments)
            has_stream_id = bool(self.proposed_ids[key])
            call_id = self.output_id(name, arguments)
            if not has_stream_id:
                self.preassigned_ids[key].append(call_id)
            return call_id

        def input_id(self, name: str, arguments: Any) -> str:
            key = self.key(name, arguments)
            index = self.input_call_positions[key]
            self.input_call_positions[key] += 1
            previous = min(self.input_call_totals[key], self.input_history_count.get(key, 0))
            if index < previous:
                return self.history_ids[key][-previous + index]
            call_id = uuid.uuid4().hex
            self.history_ids[key].append(call_id)
            return call_id

        def begin_input_batch(self, batch: list[Any]) -> None:
            self.preassigned_ids.clear()
            self.input_call_positions.clear()
            self.input_result_positions.clear()
            self.input_call_totals = Counter(
                self.key(call.get("name", ""), call.get("args", {}))
                for msg in batch
                for call in getattr(msg, "tool_calls", None) or []
                if not call.get("id")
            )
            self.input_history_count = {
                key: len(self.history_ids[key]) for key in self.input_call_totals
            }
            self.input_result_totals = Counter(
                str(msg.content)
                for msg in batch
                if msg.type == "tool" and not getattr(msg, "tool_call_id", None)
            )

        def result_id(self, message: Any, *, stream: bool = False) -> str:
            content = str(getattr(message, "content", ""))
            positions = self.stream_result_positions if stream else self.input_result_positions
            index = positions[content]
            positions[content] += 1
            count = (
                len(self.completed_ids[content]) if stream else self.input_result_totals[content]
            )
            offset = count - index
            return (
                self.completed_ids[content][-offset]
                if 0 < offset <= len(self.completed_ids[content])
                else ""
            )

        def model_tool_result(self, call_id: str) -> tuple[bool, Any]:
            if call_id not in self.model_input_tool_results:
                return False, None
            return True, self.model_input_tool_results[call_id]

        def fail_pending_tool_results(self, error: BaseException) -> None:
            pending = self.pending_tool_results
            self.pending_tool_results = {}
            for handle, _result in pending.values():
                observer.end_tool(handle, None, error)

        async def on_chat_model_start(
            self,
            serialized: dict[str, Any],  # noqa: ARG002
            messages: list[list[BaseMessage]],
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,  # noqa: ARG002
            tags: list[str] | None = None,  # noqa: ARG002
            metadata: dict[str, Any] | None = None,  # noqa: ARG002
            **kwargs: Any,
        ) -> None:
            params = kwargs.get("invocation_params") or {}
            model = params.get("model") or params.get("model_name") or requested_model
            tool_definitions = _genai_tool_definitions(params.get("tools"))
            self.model_input_tool_results.clear()
            started = []
            for batch in messages:
                self.begin_input_batch(batch)
                system = [
                    part
                    for msg in batch
                    if getattr(msg, "type", "") == "system"
                    for part in _message_parts(msg, self)
                ]
                inputs = []
                for msg in batch:
                    if msg.type == "system":
                        continue
                    parts = _message_parts(msg, self)
                    if msg.type == "tool":
                        response = next(
                            (part for part in parts if part.get("type") == "tool_call_response"),
                            None,
                        )
                        call_id = response.get("id") if response else None
                        if isinstance(call_id, str) and call_id:
                            self.model_input_tool_results[call_id] = msg.content
                            pending_result = self.pending_tool_results.pop(call_id, None)
                            if pending_result is not None:
                                observer.end_tool(pending_result[0], pending_result[1], None)
                    inputs.append(
                        {
                            "role": "assistant"
                            if msg.type == "ai"
                            else "tool"
                            if msg.type == "tool"
                            else "user",
                            "parts": parts,
                        }
                    )
                started.append(
                    (
                        observer.start_model(
                            inputs,
                            system or None,
                            model,
                            tool_definitions=tool_definitions,
                        ),
                        model,
                    )
                )
            self.models[run_id] = started

        async def on_llm_end(self, response: Any, *, run_id: Any, **_kwargs: Any) -> None:
            started = self.models.pop(run_id, [])
            llm_output = getattr(response, "llm_output", None) or {}
            for index, (handle, _request_model) in enumerate(started):
                choices = response.generations[index] if index < len(response.generations) else []
                outputs = []
                usage: dict[str, int] = {}
                response_model = None
                for choice in choices:
                    msg = getattr(choice, "message", None)
                    if msg is None:
                        continue
                    for call in getattr(msg, "tool_calls", None) or []:
                        if not call.get("id"):
                            call["id"] = self.bind_output_id(
                                call.get("name", ""), call.get("args", {})
                            )
                    metadata = getattr(msg, "response_metadata", None) or {}
                    outputs.append(
                        {
                            "role": "assistant",
                            "parts": _message_parts(msg, self, output=True),
                            "finish_reason": (
                                (getattr(choice, "generation_info", None) or {}).get(
                                    "finish_reason"
                                )
                                or metadata.get("finish_reason")
                                or metadata.get("stop_reason")
                                or "unknown"
                            ),
                        }
                    )
                    response_model = _response_model(msg) or response_model
                    for key, value in _model_usage(msg).items():
                        usage[key] = usage.get(key, 0) + value
                if len(started) == 1:
                    if not usage:
                        token_usage = llm_output.get("token_usage") or {}
                        for key, alternate in (
                            ("input_tokens", "prompt_tokens"),
                            ("output_tokens", "completion_tokens"),
                        ):
                            token_value = token_usage.get(key, token_usage.get(alternate))
                            if type(token_value) is int:
                                usage[key] = token_value
                        details = token_usage.get("output_token_details") or {}
                        reasoning = details.get("reasoning")
                        if type(reasoning) is int:
                            usage["reasoning_tokens"] = reasoning
                    response_model = (
                        response_model or llm_output.get("model_name") or llm_output.get("model")
                    )
                self.usage_seen = self.usage_seen or bool(usage)
                self.model_completed = True
                for key, value in usage.items():
                    setattr(self, key, getattr(self, key) + value)
                if isinstance(response_model, str) and response_model:
                    self.response_model = response_model
                observer.end_model(
                    handle, outputs if outputs else None, response_model, usage, None
                )

        async def on_llm_error(self, error: BaseException, *, run_id: Any, **_kwargs: Any) -> None:
            for handle, _model in self.models.pop(run_id, []):
                observer.end_model(handle, None, None, {}, error)

        async def on_tool_start(
            self,
            serialized: dict[str, Any],
            input_str: str,
            *,
            run_id: Any,
            inputs: dict[str, Any] | None = None,
            tool_call_id: str | None = None,
            **_kwargs: Any,
        ) -> None:
            name = serialized.get("name", "")
            arguments = inputs if inputs is not None else input_str
            key = self.key(name, arguments)
            pending = self.pending_ids[key]
            if tool_call_id:
                if tool_call_id in pending:
                    pending.remove(tool_call_id)
                call_id = tool_call_id
            else:
                call_id = pending.popleft() if pending else uuid.uuid4().hex
            self.tools[run_id] = observer.start_tool(name, call_id, arguments), call_id

        async def on_tool_end(self, output: Any, *, run_id: Any, **_kwargs: Any) -> None:
            entry = self.tools.pop(run_id, None)
            if entry is None:
                return
            handle, call_id = entry
            content = getattr(output, "content", output)
            if not getattr(output, "tool_call_id", None):
                self.completed_ids[str(content)].append(call_id)
            if getattr(output, "status", "success") == "error":
                observer.end_tool(handle, None, RuntimeError(str(content)))
            else:
                self.pending_tool_results[call_id] = (handle, content)

        async def on_tool_error(self, error: BaseException, *, run_id: Any, **_kwargs: Any) -> None:
            entry = self.tools.pop(run_id, None)
            if entry is not None:
                observer.end_tool(entry[0], None, error)

    return Handler()


def _structured_output_method() -> str:
    """Anthropic rejects large json_schema grammars; function_calling avoids compilation."""
    if _anthropic_backend() in {"direct", "bedrock"}:
        return "function_calling"
    return "json_schema"


async def _shape_structured_output(
    model: str,
    schema: Any,
    system_prompt: str,
    prompt: str,
    agent_text: str,
    telemetry: Any = None,
) -> tuple[Any, int, int, int, str]:
    """Shape pass: tool-free structured binding on a model without thinking."""
    from langchain_core.messages import HumanMessage, SystemMessage

    format_model = _resolve_model(model, reasoning_config=None)
    structured = format_model.with_structured_output(
        schema,
        method=_structured_output_method(),
        include_raw=True,
    )
    shape_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Original user request:\n{prompt}\n\n"
                f"Agent run output:\n{agent_text}\n\n"
                "Produce the structured response matching the required schema."
            )
        ),
    ]
    try:
        config = {"callbacks": [_telemetry_handler(telemetry, model)]} if telemetry else None
        result = (
            await structured.ainvoke(shape_messages, config=config)
            if config
            else await structured.ainvoke(shape_messages)
        )
    finally:
        await _close_model_clients(format_model)
    if isinstance(result, dict) and "parsed" in result:
        parsed = result["parsed"]
        raw = result.get("raw")
        usage = _model_usage(raw)
        return (
            parsed,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            usage.get("reasoning_tokens", 0),
            _response_model(raw) or "",
        )
    return result, 0, 0, 0, ""


def _tool_call_events(msg: Any, ids: Any = None) -> list[ProviderEvent]:
    """Map complete AI-message tool calls to provider events."""
    return [
        ToolCallEvent(
            name=call.get("name", ""),
            input=json.dumps(call.get("args", {}), ensure_ascii=False, separators=(",", ":")),
            call_id=call.get("id")
            or (ids.call_id(call.get("name", ""), call.get("args", {})) if ids else ""),
        )
        for call in msg.tool_calls or []
    ]



def _process_ai_message(
    msg: Any,
    ids: Any = None,
    *,
    include_tool_calls: bool = True,
) -> tuple[list[ProviderEvent], str, int, int]:
    """Map one AIMessage chunk to provider events and token deltas."""
    events: list[ProviderEvent] = _tool_call_events(msg, ids) if include_tool_calls else []
    text_delta = ""
    input_tokens = 0
    output_tokens = 0

    for block in getattr(msg, "content_blocks", []):
        btype = block["type"] if isinstance(block, dict) else getattr(block, "type", "")
        if btype == "non_standard":
            block = (
                block.get("value", {}) if isinstance(block, dict) else getattr(block, "value", {})
            )
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
        if btype in ("reasoning", "thinking"):
            reasoning = (
                block.get("reasoning", block.get("thinking", ""))
                if isinstance(block, dict)
                else getattr(block, "reasoning", "")
            )
            if isinstance(reasoning, str) and reasoning:
                events.append(ThinkingDeltaEvent(thinking=reasoning))
                events.append(ContentBlockStopEvent())
        elif btype == "text":
            text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
            if text:
                events.append(TextDeltaEvent(text=text))
                text_delta += text

    if not getattr(msg, "content_blocks", None):
        content = msg.content if isinstance(msg.content, str) else stringify(msg.content)
        if content and not msg.tool_calls:
            events.append(TextDeltaEvent(text=content))
            text_delta += content

    usage = _model_usage(msg)
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    return events, text_delta, input_tokens, output_tokens


class DeepAgentsProvider(AgentProvider):
    @property
    def name(self) -> str:
        return "deepagents"

    async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
        from deepagents import create_deep_agent
        from deepagents.backends import LocalShellBackend
        from lightspeed_agentic.inspection.errors import ToolResultSafetyInspectionFailed

        classifier_model: Any | None = None
        inspection_middleware: Any | None = None

        logger.debug(
            "Starting deepagents query model=%s cwd=%s max_turns=%s",
            options.model,
            options.cwd,
            options.max_turns,
        )

        chat_model = _resolve_model(options.model, options.reasoning_config)
        backend = LocalShellBackend(
            root_dir=options.cwd,
            inherit_env=True,
            max_output_bytes=MAX_TOOL_RETURN_CHARS,
        )

        agent_kwargs: dict[str, Any] = {
            "model": chat_model,
            "backend": backend,
            "system_prompt": options.system_prompt,
        }

        if options.tool_output_inspection_enabled:

            try:
                from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT

                from lightspeed_agentic.inspection.chunking import Utf8ByteCodec
                from lightspeed_agentic.inspection.client import LangChainClassifierClient
                from lightspeed_agentic.inspection.inspector import (
                    inspect_tool_result as run_inspection,
                )
                from lightspeed_agentic.inspection.middleware import ToolResultInspectionMiddleware

                classifier_model = _resolve_model(options.model, reasoning_config=None)
                classifier_client = LangChainClassifierClient(classifier_model)
                model_profile = getattr(classifier_model, "profile", None) or {}
                context_window_tokens = (
                    model_profile.get("max_input_tokens")
                    or model_profile.get("max_context_size")
                    or 100_000
                )

                async def inspect_tool_result_callback(
                    tool_name: str,
                    result_type: str,
                    value: Any,
                    tool_call_id: str,
                ) -> Any:
                    return await run_inspection(
                        classifier_client,
                        tool_name=tool_name,
                        result_type=result_type,
                        value=value,
                        codec=Utf8ByteCodec(),
                        tool_call_id=tool_call_id or None,
                        context_window_tokens=context_window_tokens,
                        instruction_tokens=512,
                        output_tokens=128,
                        deadline=options.deadline,
                        provider="anthropic",
                        model=options.model,
                    )

                inspection_middleware = ToolResultInspectionMiddleware(inspect_tool_result_callback)
                agent_kwargs["middleware"] = [inspection_middleware]
                agent_kwargs["subagents"] = [
                    {
                        **GENERAL_PURPOSE_SUBAGENT,
                        "middleware": [inspection_middleware],
                    }
                ]
            except Exception as exc:
                if classifier_model is not None:
                    await _close_model_clients(classifier_model)
                raise ToolResultSafetyInspectionFailed() from exc

        if has_skills(options.cwd):
            agent_kwargs["skills"] = ["/"]

        schema_model: Any | None = None
        if options.output_schema:
            schema_model = (
                _json_schema_to_pydantic(options.output_schema)
                if isinstance(options.output_schema, dict)
                else options.output_schema
            )

        mcp_tools: list[Any] = []
        if options.mcp_servers:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            from lightspeed_agentic.tls import create_async_http_client

            client = MultiServerMCPClient(
                {
                    server.name: {  # type: ignore[misc]
                        "transport": "http",
                        "url": server.url,
                        "headers": {h.name: h.value for h in server.headers},
                        "timeout": server.timeout,
                        "httpx_client_factory": create_async_http_client,
                    }
                    for server in options.mcp_servers
                }
            )
            for server in options.mcp_servers:
                allowed_tool_names = set(server.allowed_tool_names)
                server_tools = await client.get_tools(server_name=server.name)
                mcp_tools.extend(tool for tool in server_tools if tool.name in allowed_tool_names)

        if mcp_tools:
            agent_kwargs["tools"] = mcp_tools

        # allowed_tools is not forwarded: deepagents' LocalShellBackend exposes a broader
        # built-in tool set than DEFAULT_ALLOWED_TOOLS. Filtering is a follow-up.
        agent = create_deep_agent(**agent_kwargs)

        thread_id = f"ls-{uuid.uuid4().hex[:12]}"
        stream_config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": options.max_turns,
        }
        telemetry_handler = (
            _telemetry_handler(options.telemetry, options.model) if options.telemetry else None
        )
        if telemetry_handler is not None:
            stream_config["callbacks"] = [telemetry_handler]
        result_text = ""
        pending_tool_results: list[tuple[str, str, str, Any]] = []
        total_input_tokens = 0
        total_output_tokens = 0
        streamed_reasoning_tokens = 0
        streamed_response_model = ""
        pending_tool_call_chunk: Any | None = None
        input_state = {"messages": [{"role": "user", "content": options.prompt}]}

        def flush_pending_tool_calls() -> list[ProviderEvent]:
            nonlocal pending_tool_call_chunk
            if pending_tool_call_chunk is None:
                return []
            events = _tool_call_events(pending_tool_call_chunk, telemetry_handler)
            pending_tool_call_chunk = None
            return events

        try:
            async for msg, _stream_metadata in cast(Any, agent).astream(
                input_state,
                config=stream_config,
                stream_mode="messages",
            ):
                if msg.type in ("ai", "AIMessageChunk"):
                    is_chunk = msg.type == "AIMessageChunk"
                    tool_call_chunks = getattr(msg, "tool_call_chunks", None) if is_chunk else None
                    if inspection_middleware is not None:
                        for tool_name, result_type, call_id, content in pending_tool_results:
                            model_content = content
                            if telemetry_handler is not None:
                                has_model_content, model_content = telemetry_handler.model_tool_result(
                                    call_id
                                )
                                if not has_model_content:
                                    raise ToolResultSafetyInspectionFailed()
                            if not inspection_middleware.is_passed(
                                tool_name,
                                result_type,
                                call_id,
                                model_content,
                            ):
                                raise ToolResultSafetyInspectionFailed()
                            yield ToolResultEvent(
                                output=stringify(model_content),
                                call_id=call_id,
                            )
                        pending_tool_results.clear()

                    if tool_call_chunks:
                        current_tool_call_chunk = type(msg)(
                            content="",
                            tool_call_chunks=tool_call_chunks,
                        )
                        pending_tool_call_chunk = (
                            current_tool_call_chunk
                            if pending_tool_call_chunk is None
                            else pending_tool_call_chunk + current_tool_call_chunk
                        )
                    elif not is_chunk and pending_tool_call_chunk is not None:
                        if getattr(msg, "tool_calls", None):
                            pending_tool_call_chunk = None
                        else:
                            for event in flush_pending_tool_calls():
                                yield event

                    events, text_delta, in_tok, out_tok = _process_ai_message(
                        msg,
                        telemetry_handler,
                        include_tool_calls=not (is_chunk and bool(tool_call_chunks)),
                    )
                    for event in events:
                        yield event
                    result_text += text_delta
                    total_input_tokens += in_tok
                    total_output_tokens += out_tok
                    streamed_reasoning_tokens += _model_usage(msg).get("reasoning_tokens", 0)
                    streamed_response_model = _response_model(msg) or streamed_response_model
                    if is_chunk and getattr(msg, "chunk_position", None) == "last":
                        for event in flush_pending_tool_calls():
                            yield event

                elif msg.type in ("tool", "ToolMessageChunk"):
                    for event in flush_pending_tool_calls():
                        yield event
                    tool_name = getattr(msg, "name", "") or ""
                    result_type = (
                        "error" if getattr(msg, "status", "success") == "error" else "result"
                    )
                    call_id = getattr(msg, "tool_call_id", "") or (
                        telemetry_handler.result_id(msg, stream=True) if telemetry_handler else ""
                    )
                    if inspection_middleware is None:
                        yield ToolResultEvent(
                            output=stringify(msg.content),
                            call_id=call_id,
                        )
                    else:
                        pending_tool_results.append((tool_name, result_type, call_id, msg.content))
            for event in flush_pending_tool_calls():
                yield event
        except ToolResultSafetyInspectionFailed as exc:
            if telemetry_handler is not None:
                telemetry_handler.fail_pending_tool_results(exc)
            raise
        finally:
            await _close_model_clients(chat_model)
            if classifier_model is not None:
                await _close_model_clients(classifier_model)

        response_model = (
            (telemetry_handler.response_model or streamed_response_model)
            if telemetry_handler and telemetry_handler.model_completed
            else streamed_response_model
        )
        reasoning_tokens = (
            telemetry_handler.reasoning_tokens
            if telemetry_handler and telemetry_handler.usage_seen
            else streamed_reasoning_tokens
        )
        if telemetry_handler and telemetry_handler.usage_seen:
            total_input_tokens = telemetry_handler.input_tokens
            total_output_tokens = telemetry_handler.output_tokens
        if schema_model is not None:
            shape_args = (
                options.model,
                schema_model,
                options.system_prompt,
                options.prompt,
                result_text,
            )
            structured, in_tok, out_tok, shape_reasoning, shape_model = (
                await _shape_structured_output(*shape_args, telemetry=options.telemetry)
                if options.telemetry
                else await _shape_structured_output(*shape_args)
            )
            if isinstance(shape_model, str) and shape_model:
                response_model = shape_model
            result_text = stringify(structured)
            total_input_tokens += in_tok
            total_output_tokens += out_tok
            reasoning_tokens += shape_reasoning

        yield ResultEvent(
            text=result_text,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            reasoning_tokens=reasoning_tokens,
            response_model=response_model,
        )
