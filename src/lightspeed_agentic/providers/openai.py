"""OpenAI provider — wraps openai-agents SDK.

Uses SandboxAgent with native Shell, Filesystem, and Skills capabilities.
The SDK handles tool registration, skill discovery, and command execution.
"""
# mypy: disable-error-code=unused-ignore

from __future__ import annotations

import inspect
import json
import logging
import os
import tempfile
from collections.abc import AsyncIterator
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:

    class AgentOutputSchemaBase:
        """Type-checker stub for the optional openai-agents base class."""

        pass

else:
    try:
        from agents.agent_output import AgentOutputSchemaBase
    except ImportError:

        class AgentOutputSchemaBase:  # pragma: no cover - optional SDK fallback
            """Fallback base so the module imports without the openai extra."""

            pass


from lightspeed_agentic.skills import has_skills
from lightspeed_agentic.types import (
    MAX_TOOL_RETURN_CHARS,
    TOOL_RETURN_PREVIEW_CHARS,
    AgentProvider,
    ContentBlockStopEvent,
    ProviderEvent,
    ProviderQueryOptions,
    ProviderTelemetry,
    ResultEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    stringify,
)

logger = logging.getLogger(__name__)


def _make_strict(schema: dict[str, Any]) -> dict[str, Any]:
    """Add OpenAI strict-schema requirements recursively without mutating input."""
    if not isinstance(schema, dict):
        return schema
    schema = dict(schema)
    if schema.get("type") == "object" and "properties" in schema:
        schema["additionalProperties"] = False
        schema["required"] = list(schema["properties"].keys())
        schema["properties"] = {k: _make_strict(v) for k, v in schema["properties"].items()}
    if "items" in schema and isinstance(schema["items"], dict):
        schema["items"] = _make_strict(schema["items"])
    if "oneOf" in schema and isinstance(schema["oneOf"], list):
        logger.info("Converting oneOf to anyOf for OpenAI compatibility")
        schema.setdefault("anyOf", []).extend(schema.pop("oneOf"))
    for keyword in ("anyOf", "allOf"):
        if keyword in schema and isinstance(schema[keyword], list):
            schema[keyword] = [_make_strict(item) for item in schema[keyword]]
    if "not" in schema and isinstance(schema["not"], dict):
        schema["not"] = _make_strict(schema["not"])
    for defs_key in ("$defs", "definitions"):
        if defs_key in schema and isinstance(schema[defs_key], dict):
            schema[defs_key] = {
                name: _make_strict(value) for name, value in schema[defs_key].items()
            }
    return schema


_OPENAI_HOSTS = ("api.openai.com",)


def _is_native_openai() -> bool:
    """True when talking to api.openai.com (explicitly or by default)."""
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        return True
    try:
        from urllib.parse import urlparse

        return urlparse(base_url).hostname in _OPENAI_HOSTS
    except Exception:
        return False


_openai_initialized = False


class _RawJsonSchema(AgentOutputSchemaBase):
    """Wraps JSON schema for OpenAI model output type.

    Args:
        schema: The JSON schema dict for structured output.
        is_native: True if using native OpenAI Responses API (strict mode),
                   False if using Chat Completions (non-strict mode).
    """

    def __init__(self, schema: dict[str, Any], is_native: bool) -> None:
        self._schema = _make_strict(schema) if is_native else schema
        self._is_native = is_native

    def is_plain_text(self) -> bool:
        return False

    def name(self) -> str:
        return "raw_json_schema"

    def json_schema(self) -> dict[str, Any]:
        return self._schema

    def is_strict_json_schema(self) -> bool:
        return self._is_native

    def validate_json(self, json_str: str) -> Any:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in structured output: {e}") from e


def _patch_exec_command_args() -> None:
    """Sanitize ExecCommandArgs input before Pydantic validation.

    The openai-agents SDK registers exec_command with strict_json_schema=False,
    so the model can send "shell": true (boolean) instead of a string path.
    Pydantic rejects the bool, crashing the execution step. OLS-3257.
    Remove when openai-agents fixes exec_command schema validation.
    """
    from agents.sandbox.capabilities.tools.shell_tool import ExecCommandTool

    _original_invoke = ExecCommandTool._invoke

    async def _sanitized_invoke(self: ExecCommandTool, ctx: object, raw_input: str) -> str:
        try:
            parsed = json.loads(raw_input)
        except (json.JSONDecodeError, ValueError):
            # If parsing fails, pass through to original handler
            return await _original_invoke(self, ctx, raw_input)
        if isinstance(parsed, dict) and isinstance(parsed.get("shell"), bool):
            logger.debug("Coercing exec_command shell=%s to None (OLS-3257)", parsed["shell"])
            parsed["shell"] = None
            raw_input = json.dumps(parsed)
        return await _original_invoke(self, ctx, raw_input)

    ExecCommandTool._invoke = _sanitized_invoke  # type: ignore[assignment]


def _ensure_openai_init() -> None:
    global _openai_initialized
    if _openai_initialized:
        return
    from agents import enable_verbose_stdout_logging
    from agents.tracing import set_tracing_disabled

    set_tracing_disabled(True)
    enable_verbose_stdout_logging()  # type: ignore[no-untyped-call]
    _patch_exec_command_args()
    _openai_initialized = True


def _validated_e2e_output_dir() -> str | None:
    """Return E2E_OUTPUT_DIR when it resolves under the system temp directory."""
    raw = os.environ.get("E2E_OUTPUT_DIR", "").strip()
    if not raw:
        return None
    try:
        resolved = Path(raw).resolve()
    except OSError:
        logger.warning("E2E_OUTPUT_DIR is not a valid path: %s", raw)
        return None
    temp_root = Path(tempfile.gettempdir()).resolve()
    if resolved != temp_root and not str(resolved).startswith(str(temp_root) + os.sep):
        logger.warning(
            "E2E_OUTPUT_DIR outside temp root %s: %s",
            temp_root,
            resolved,
        )
        return None
    return str(resolved)


def _build_manifest(cwd: str) -> Any:
    """Build sandbox manifest, optionally granting write access to E2E_OUTPUT_DIR."""
    from agents.sandbox.manifest import Manifest, SandboxPathGrant  # type: ignore[attr-defined]

    kwargs: dict[str, Any] = {"root": cwd}
    output_dir = _validated_e2e_output_dir()
    if output_dir:
        kwargs["extra_path_grants"] = (
            SandboxPathGrant(
                path=output_dir,
                read_only=False,
                description="e2e skill token output",
            ),
        )
    return Manifest(**kwargs)


async def _build_mcp_function_tools(servers: list[Any]) -> list[Any]:
    """Expose MCP tools as function tools for Chat Completions models."""
    from agents.mcp.util import MCPUtil

    function_tools: list[Any] = []
    for server in servers:
        for tool in await server.list_tools():
            function_tools.append(
                MCPUtil.to_function_tool(
                    tool,
                    server,
                    convert_schemas_to_strict=False,
                )
            )
    return function_tools


def _item_value(item: Any, key: str, default: Any = None) -> Any:
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)


def _tool_arguments(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


class _ToolCallIds:
    """Keep missing SDK call IDs stable across model messages and tool hooks."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any, str]] = []
        self.explicit_ids: set[str] = set()
        self.output_ids: dict[int, tuple[Any, str]] = {}
        self.executed: set[int] = set()
        self.results: list[tuple[str, Any]] = []
        self.output_results: list[tuple[str, Any]] = []

    def completed_result(self, call_id: str, result: Any) -> None:
        self.results.append((call_id, result))

    def observed_output(self, item: Any) -> None:
        explicit_id = _item_value(item, "call_id")
        if explicit_id:
            for index, (call_id, _) in enumerate(self.results):
                if call_id == explicit_id:
                    self.results.pop(index)
                    break
            return
        result = _item_value(item, "output")
        for identical in (True, False):
            matches = [
                (index, call_id)
                for index, (call_id, value) in enumerate(self.results)
                if call_id not in self.explicit_ids
                and (value is result if identical else value == result)
            ]
            if matches:
                # The SDK emits duplicate-value outputs in call order, even when
                # the corresponding executions finished in a different order.
                index, call_id = min(
                    matches,
                    key=lambda match: next(
                        (
                            position
                            for position, call in enumerate(self.calls)
                            if call[2] == match[1]
                        ),
                        len(self.calls),
                    ),
                )
                self.results.pop(index)
                self.output_results.append((call_id, _item_value(item.raw_item, "output")))
                return

    def result_id(self, output: Any, pending: list[str]) -> str | None:
        for entries in (self.output_results, self.results):
            for identical in (True, False):
                for index, (call_id, result) in enumerate(entries):
                    if (
                        call_id in pending
                        and call_id not in self.explicit_ids
                        and (result is output if identical else result == output)
                    ):
                        entries.pop(index)
                        return call_id
        return None

    def output_id(self, item: Any) -> str:
        key = id(item)
        if key not in self.output_ids:
            explicit_id = _item_value(item, "call_id") or _item_value(item, "id")
            call_id = explicit_id or uuid4().hex
            if explicit_id:
                self.explicit_ids.add(explicit_id)
            self.output_ids[key] = (item, call_id)
            self.calls.append(
                (
                    _item_value(item, "name", ""),
                    _tool_arguments(_item_value(item, "arguments", _item_value(item, "input", ""))),
                    call_id,
                )
            )
        return self.output_ids[key][1]

    def input_id(self, item: Any, arguments: Any, start: int) -> tuple[str, int]:
        call_id = _item_value(item, "call_id") or _item_value(item, "id")
        if call_id:
            for index in range(start, len(self.calls)):
                if self.calls[index][2] == call_id:
                    return call_id, index + 1
            return call_id, start
        name = _item_value(item, "name", "")
        for index in range(start, len(self.calls)):
            if (
                self.calls[index][:2] == (name, arguments)
                and self.calls[index][2] not in self.explicit_ids
            ):
                return self.calls[index][2], index + 1
        call_id = uuid4().hex
        self.calls.append((name, arguments, call_id))
        return call_id, len(self.calls)

    def execution_id(self, name: str, arguments: Any, explicit_id: str | None = None) -> str:
        if explicit_id:
            for index, (_, _, call_id) in enumerate(self.calls):
                if call_id == explicit_id:
                    self.executed.add(index)
                    break
            return explicit_id
        # Execution order need not match request order; prefer the exact call.
        for exact in (True, False):
            for index, (call_name, call_arguments, call_id) in enumerate(self.calls):
                if (
                    index not in self.executed
                    and call_id not in self.explicit_ids
                    and call_name == name
                    and (not exact or call_arguments == arguments)
                ):
                    self.executed.add(index)
                    return call_id
        return uuid4().hex


def _generic_part(item: Any, kind: str) -> dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    dump = getattr(item, "model_dump", None)
    return dump(mode="json") if callable(dump) else {"type": kind, "content": str(item)}


def _message_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "content": content}]
    parts = []
    for part in content or []:
        kind = _item_value(part, "type")
        if kind in ("input_text", "output_text", "text", "refusal"):
            parts.append(
                {
                    "type": "text",
                    "content": _item_value(part, "text", None) or _item_value(part, "refusal", ""),
                }
            )
        elif kind in ("input_image", "image_url"):
            image = _item_value(part, "image_url")
            uri = _item_value(image, "url", image)
            if isinstance(uri, str):
                if uri.startswith("data:") and ";base64," in uri:
                    mime, content = uri[5:].split(";base64,", 1)
                    parts.append(
                        {
                            "type": "blob",
                            "modality": "image",
                            "mime_type": mime,
                            "content": content,
                        }
                    )
                else:
                    parts.append({"type": "uri", "modality": "image", "uri": uri})
        elif kind in ("reasoning_text", "summary_text"):
            parts.append({"type": "reasoning", "content": _item_value(part, "text", "")})
        elif isinstance(kind, str):
            parts.append(_generic_part(part, kind))
    return parts


def _model_input_messages(items: Any, call_ids: _ToolCallIds) -> list[dict[str, Any]]:
    if isinstance(items, str):
        return [{"role": "user", "parts": [{"type": "text", "content": items}]}]
    messages = []
    request_index = 0
    request_ids: list[str] = []
    for item in items:
        kind = _item_value(item, "type", "message")
        if kind in ("function_call", "custom_tool_call"):
            arguments = _tool_arguments(
                _item_value(item, "arguments", _item_value(item, "input", ""))
            )
            call_id, request_index = call_ids.input_id(item, arguments, request_index)
            request_ids.append(call_id)
            messages.append(
                {
                    "role": "assistant",
                    "parts": [
                        {
                            "type": "tool_call",
                            "id": call_id,
                            "name": _item_value(item, "name", ""),
                            "arguments": arguments,
                        }
                    ],
                }
            )
        elif kind in ("function_call_output", "custom_tool_call_output"):
            explicit_id = _item_value(item, "call_id") or _item_value(item, "id")
            if explicit_id:
                paired_id = explicit_id
            else:
                paired_id = call_ids.result_id(_item_value(item, "output"), request_ids)
                if paired_id is None:
                    paired_id = request_ids[0] if request_ids else uuid4().hex
            if paired_id in request_ids:
                request_ids.remove(paired_id)
            messages.append(
                {
                    "role": "tool",
                    "parts": [
                        {
                            "type": "tool_call_response",
                            "id": paired_id,
                            "response": _item_value(item, "output"),
                        }
                    ],
                }
            )
        elif kind == "reasoning":
            content = _item_value(item, "content") or _item_value(item, "summary") or []
            messages.append({"role": "assistant", "parts": _message_parts(content)})
        elif kind == "message":
            messages.append(
                {
                    "role": _item_value(item, "role", "user"),
                    "parts": _message_parts(_item_value(item, "content")),
                }
            )
        elif isinstance(kind, str):
            messages.append({"role": "assistant", "parts": [_generic_part(item, kind)]})
    return messages


def _model_output_messages(
    items: list[Any],
    finish_reason: str | None,
    call_ids: _ToolCallIds,
) -> list[dict[str, Any]]:
    parts = []
    has_tool_call = False
    for item in items:
        kind = _item_value(item, "type")
        if kind == "message":
            parts.extend(_message_parts(_item_value(item, "content")))
        elif kind == "reasoning":
            parts.extend(
                _message_parts(_item_value(item, "content") or _item_value(item, "summary"))
            )
        elif kind in ("function_call", "custom_tool_call"):
            has_tool_call = True
            parts.append(
                {
                    "type": "tool_call",
                    "id": call_ids.output_id(item),
                    "name": _item_value(item, "name", ""),
                    "arguments": _tool_arguments(
                        _item_value(item, "arguments", _item_value(item, "input", ""))
                    ),
                }
            )
        elif isinstance(kind, str):
            parts.append(_generic_part(item, kind))
    if finish_reason is None:
        finish_reason = "tool_call" if has_tool_call else "unknown"
    return [{"role": "assistant", "parts": parts, "finish_reason": finish_reason}]


def _response_finish_reason(response: Any) -> str | None:
    reason = _item_value(response, "finish_reason")
    if reason == "tool_calls":
        return "tool_call"
    if isinstance(reason, str) and reason in (
        "stop",
        "length",
        "content_filter",
        "tool_call",
        "error",
    ):
        return reason
    status = _item_value(response, "status")
    if status == "failed":
        return "error"
    if status == "incomplete":
        detail = _item_value(_item_value(response, "incomplete_details"), "reason")
        if detail == "content_filter":
            return "content_filter"
        if detail == "max_output_tokens":
            return "length"
        return None
    return None


def _model_usage(usage: Any) -> dict[str, int]:
    if _item_value(usage, "requests") == 0:
        return {}
    counts: dict[str, int] = {}
    for field in ("input_tokens", "output_tokens"):
        value = _item_value(usage, field)
        if isinstance(value, int):
            counts[field] = value
    details = _item_value(usage, "output_tokens_details")
    reasoning = _item_value(details, "reasoning_tokens")
    if isinstance(reasoning, int):
        counts["reasoning_tokens"] = reasoning
    return counts


def _telemetry_hooks(options: ProviderQueryOptions, *, is_native: bool = True) -> Any:
    """Translate actual Agents SDK model and local-tool lifecycle callbacks."""
    from agents.lifecycle import RunHooksBase

    telemetry = options.telemetry
    if telemetry is None:
        raise ValueError("Telemetry hooks require a telemetry observer")
    observer: ProviderTelemetry = telemetry

    class TelemetryHooks(RunHooksBase):
        def __init__(self) -> None:
            self.models: list[object] = []
            self.completed_unidentified: list[tuple[str | None, str | None]] = []
            self.completed: dict[str, tuple[str | None, str | None]] = {}
            self.pending: list[tuple[object, Any]] = []
            self.tools: dict[int, tuple[object, str]] = {}
            self.tool_errors: dict[int, BaseException] = {}
            self.tool_formatters: dict[int, tuple[Any, bool, Any, int]] = {}
            self.call_ids = _ToolCallIds()
            self.last_response_model: str | None = None
            self.failed_response: Any | None = None

        async def on_llm_start(
            self,
            _context: Any,
            agent: Any,
            system_prompt: str | None,
            input_items: Any,
        ) -> None:
            instructions = [{"type": "text", "content": system_prompt}] if system_prompt else None
            model = getattr(agent.model, "model", None) or options.model
            self.models.append(
                observer.start_model(
                    _model_input_messages(input_items, self.call_ids),
                    instructions,
                    model,
                )
            )

        async def on_llm_end(self, _context: Any, _agent: Any, response: Any) -> None:
            for item in response.output:
                if _item_value(item, "type") in ("function_call", "custom_tool_call"):
                    self.call_ids.output_id(item)
            self.pending.append((self.models.pop(0), response))
            self.flush()

        def observe_response(self, response: Any) -> None:
            response_id = _item_value(response, "id")
            # The Chat Completions adapter synthesizes response.model from the requested
            # model; only native Responses metadata identifies the actual backend model.
            model = _item_value(response, "model") if is_native else None
            if not isinstance(model, str) or not model:
                model = None
            metadata = (model, _response_finish_reason(response))
            if self.pending and not _item_value(self.pending[0][1], "response_id"):
                self.completed_unidentified.append(metadata)
            elif response_id:
                self.completed[response_id] = metadata
            else:
                self.completed_unidentified.append(metadata)
            self.last_response_model = model
            self.flush()

        def observe_failed_response(self, response: Any) -> None:
            self.failed_response = response

        def flush(self, fallback: bool = False) -> None:
            while self.pending:
                handle, response = self.pending[0]
                response_id = _item_value(response, "response_id")
                if not fallback:
                    if response_id and response_id not in self.completed:
                        break
                    if not response_id and not self.completed_unidentified:
                        break
                self.pending.pop(0)
                model, finish_reason = (
                    self.completed.pop(response_id, (None, None))
                    if response_id
                    else self.completed_unidentified.pop(0)
                    if self.completed_unidentified
                    else (None, None)
                )
                observer.end_model(
                    handle,
                    _model_output_messages(response.output, finish_reason, self.call_ids),
                    model,
                    _model_usage(response.usage),
                    None,
                )

        async def on_tool_start(self, context: Any, _agent: Any, tool: Any) -> None:
            from agents.tool import (
                FunctionTool,
                resolve_function_tool_failure_error_function,
                set_function_tool_failure_error_function,
            )

            args = _tool_arguments(getattr(context, "tool_arguments", ""))
            call_id = self.call_ids.execution_id(
                tool.name, args, getattr(context, "tool_call_id", None)
            )
            self.tools[id(context)] = (observer.start_tool(tool.name, call_id, args), call_id)
            if not isinstance(tool, FunctionTool):
                return
            key = id(tool)
            if key in self.tool_formatters:
                original, was_default, configured, active = self.tool_formatters[key]
                self.tool_formatters[key] = (original, was_default, configured, active + 1)
                return
            original = resolve_function_tool_failure_error_function(tool)
            if original is None:
                return
            self.tool_formatters[key] = (
                tool, tool._use_default_failure_error_function, tool._failure_error_function, 1
            )

            async def record_failure(ctx: Any, error: Exception) -> str:
                response = original(ctx, error)
                if inspect.isawaitable(response):
                    response = await response
                # SDK passes the same ToolContext to its formatter and lifecycle hooks.
                # Only a formatted, model-visible failure counts as handled.
                if response is not None and id(ctx) in self.tools:
                    self.tool_errors[id(ctx)] = error
                return response

            set_function_tool_failure_error_function(tool, record_failure)

        async def on_tool_end(
            self,
            context: Any,
            _agent: Any,
            _tool: Any,
            result: object,
        ) -> None:
            handle, call_id = self.tools.pop(id(context))
            failure = self.tool_errors.pop(id(context), None)
            self.call_ids.completed_result(call_id, result)
            status = _item_value(result, "status")
            if failure is not None:
                observer.end_tool(handle, None, failure)
            elif isinstance(result, BaseException):
                observer.end_tool(handle, None, result)
            elif status in ("error", "failed") or _item_value(result, "is_error") is True:
                observer.end_tool(handle, None, RuntimeError(str(result)))
            else:
                observer.end_tool(handle, result, None)
            entry = self.tool_formatters.get(id(_tool))
            if entry is not None:
                tool, was_default, configured, active = entry
                if active == 1:
                    tool._use_default_failure_error_function = was_default
                    tool._failure_error_function = configured
                    del self.tool_formatters[id(_tool)]
                else:
                    self.tool_formatters[id(_tool)] = (tool, was_default, configured, active - 1)

        def close(self, error: BaseException | None = None) -> None:
            self.flush(fallback=True)
            for handle in self.models:
                response = self.failed_response
                output = _item_value(response, "output")
                observer.end_model(
                    handle,
                    (
                        _model_output_messages(
                            output, _response_finish_reason(response), self.call_ids
                        )
                        if output
                        else None
                    ),
                    _item_value(response, "model") if is_native else None,
                    _model_usage(_item_value(response, "usage")),
                    error or RuntimeError("Model request did not complete"),
                )
            self.models.clear()
            for handle, _call_id in self.tools.values():
                observer.end_tool(
                    handle, None, error or RuntimeError("Tool execution did not complete")
                )
            self.tools.clear()
            self.tool_errors.clear()
            for tool, was_default, configured, _active in self.tool_formatters.values():
                tool._use_default_failure_error_function = was_default
                tool._failure_error_function = configured
            self.tool_formatters.clear()

    return TelemetryHooks()


class OpenAIProvider(AgentProvider):
    _client: Any = None

    @property
    def name(self) -> str:
        return "openai"

    def _build_model_settings(self, reasoning_config: dict[str, Any]) -> Any:
        """Build ModelSettings from reasoning config.

        Helper to avoid code duplication between single and two-phase paths.
        """
        from agents.model_settings import ModelSettings
        from openai.types.shared import Reasoning

        rc = dict(reasoning_config)
        model_settings_kwargs: dict[str, Any] = {}
        if "verbosity" in rc:
            model_settings_kwargs["verbosity"] = rc.pop("verbosity")
        if rc:
            model_settings_kwargs["reasoning"] = Reasoning(**rc)
        if model_settings_kwargs:
            return ModelSettings(**model_settings_kwargs)
        return None

    async def query(self, options: ProviderQueryOptions) -> AsyncIterator[ProviderEvent]:
        """Execute agent query using OpenAI Responses or Chat Completions model.

        For native OpenAI: Uses OpenAIResponsesModel with full capabilities.
        For vLLM/custom endpoints: Uses OpenAIChatCompletionsModel with manually
        implemented filesystem and MCP function tools (ChatCompletions compatible).
        """
        _ensure_openai_init()

        if self._client is None:
            from openai import AsyncOpenAI, DefaultAsyncHttpxClient

            from lightspeed_agentic.tls import get_ssl_context

            self._client = AsyncOpenAI(
                base_url=os.environ.get("OPENAI_BASE_URL"),
                api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                http_client=DefaultAsyncHttpxClient(verify=get_ssl_context()),
            )

        from agents import (
            RawResponsesStreamEvent,
            RunItemStreamEvent,
            Runner,
        )
        from agents.extensions import ToolOutputTrimmer
        from agents.items import ToolCallItem, ToolCallOutputItem
        from agents.run_config import RunConfig, SandboxRunConfig
        from agents.sandbox import SandboxAgent
        from agents.sandbox.capabilities import Filesystem, Shell, Skills
        from agents.sandbox.capabilities.skills import LocalDirLazySkillSource
        from agents.sandbox.entries import LocalDir
        from agents.sandbox.sandboxes.unix_local import (
            UnixLocalSandboxClient,
        )
        from openai.types.responses import (
            ResponseReasoningSummaryTextDeltaEvent,
            ResponseReasoningTextDeltaEvent,
            ResponseTextDeltaEvent,
        )

        # Setup model and capabilities based on endpoint
        is_native = _is_native_openai()
        capabilities: list[Any] = [Shell()]
        function_tools_list: list[Any] | None = None

        if is_native:
            from agents.models.openai_responses import OpenAIResponsesModel

            model: Any = OpenAIResponsesModel(model=options.model, openai_client=self._client)
            # Native OpenAI: use full Filesystem() capability
            capabilities.append(Filesystem())
        else:
            from agents.models.openai_chatcompletions import (
                OpenAIChatCompletionsModel,
            )

            from lightspeed_agentic.function_tools import (
                apply_patch,
                list_directory,
                read_file,
                write_file,
            )

            model = OpenAIChatCompletionsModel(
                model=options.model,
                openai_client=self._client,
                buffer_streamed_tool_calls=True,
            )
            # vLLM/custom: use manually implemented filesystem function tools
            # (avoids incompatible CustomTool apply_patch in Filesystem)
            function_tools_list = [read_file, write_file, list_directory, apply_patch]

        # Add Skills if present
        if has_skills(options.cwd):
            capabilities.append(
                Skills(
                    lazy_from=LocalDirLazySkillSource(
                        source=LocalDir(src=Path(options.cwd)),
                    ),
                    skills_path="skills/.agents",
                ),
            )
        # Manifest root is cwd's parent (/app) so shell commands can reach workspace
        manifest = _build_manifest(str(Path(options.cwd).parent))

        # Setup MCP servers. Once admitted, the full set must remain available.
        mcp_manager = None
        mcp_servers_for_agent: list[Any] = []
        if options.mcp_servers:
            from agents.mcp import MCPServerManager

            from lightspeed_agentic.mcp import to_openai_mcp_servers

            mcp_servers_list = to_openai_mcp_servers(options.mcp_servers)
            if not mcp_servers_list:
                raise RuntimeError("MCP server conversion produced no servers")

            mcp_manager = MCPServerManager(mcp_servers_list)
            entered = False
            try:
                await mcp_manager.__aenter__()
                entered = True
                active_servers = getattr(mcp_manager, "active_servers", None) or []
                active_count = len(active_servers)
                expected_count = len(mcp_servers_list)
                if active_count != expected_count:
                    raise RuntimeError(
                        f"MCP manager initialized {active_count} of "
                        f"{expected_count} admitted servers"
                    )
                mcp_servers_for_agent = list(active_servers)
                logger.debug("Initialized %d MCP servers", active_count)
            except Exception:
                if entered:
                    await mcp_manager.__aexit__(None, None, None)
                mcp_manager = None
                raise

        try:
            if not is_native and mcp_servers_for_agent and function_tools_list is not None:
                function_tools_list.extend(await _build_mcp_function_tools(mcp_servers_for_agent))
            # Tool failure formatters are instrumented per run; shared decorated
            # function tools must not carry another concurrent run's formatter.
            if function_tools_list and options.telemetry is not None:
                function_tools_list = [copy(tool) for tool in function_tools_list]

            agent_kwargs: dict[str, Any] = {
                "name": "lightspeed",
                "instructions": options.system_prompt,
                "model": model,
                "capabilities": capabilities,
                "default_manifest": manifest,
                "mcp_servers": mcp_servers_for_agent if is_native else [],
            }

            # Add function tools for vLLM/custom endpoints, including MCP tools.
            if function_tools_list:
                agent_kwargs["tools"] = function_tools_list

            if options.reasoning_config:
                agent_kwargs["model_settings"] = self._build_model_settings(
                    options.reasoning_config
                )

            # Set output_type for structured output
            if options.output_schema:
                agent_kwargs["output_type"] = _RawJsonSchema(
                    options.output_schema, is_native=is_native
                )

            agent = SandboxAgent(**agent_kwargs)
            hooks = None
            if options.telemetry is not None:
                hooks = _telemetry_hooks(options, is_native=is_native)

            run_config = RunConfig(
                sandbox=SandboxRunConfig(
                    client=UnixLocalSandboxClient(),
                ),
                call_model_input_filter=ToolOutputTrimmer(
                    max_output_chars=MAX_TOOL_RETURN_CHARS,
                    preview_chars=TOOL_RETURN_PREVIEW_CHARS,
                ),
            )

            runner_kwargs: dict[str, Any] = {
                "max_turns": options.max_turns,
                "run_config": run_config,
            }
            if hooks is not None:
                runner_kwargs["hooks"] = hooks
            result = Runner.run_streamed(agent, options.prompt, **runner_kwargs)
            last_actual_model: str | None = None

            # Stream events from the runner
            async for event in result.stream_events():
                # Handle text and reasoning deltas from both Responses and ChatCompletions.
                # Both models emit ResponseTextDeltaEvent and ResponseReasoningTextDeltaEvent
                # (ChatCompletions converts its reasoning_content to Responses event types).
                if isinstance(event, RawResponsesStreamEvent):
                    event_type = getattr(event.data, "type", None)
                    response = getattr(event.data, "response", None)
                    if event_type == "response.completed":
                        if is_native:
                            observed_model = _item_value(response, "model")
                            if isinstance(observed_model, str) and observed_model:
                                last_actual_model = observed_model
                        if hooks is not None:
                            hooks.observe_response(response)
                    elif (
                        hooks is not None
                        and event_type in ("response.incomplete", "response.failed")
                        and response is not None
                    ):
                        hooks.observe_failed_response(response)
                    if isinstance(event.data, ResponseTextDeltaEvent) and event.data.delta:
                        yield TextDeltaEvent(text=event.data.delta)
                    elif (
                        isinstance(
                            event.data,
                            (
                                ResponseReasoningTextDeltaEvent,
                                ResponseReasoningSummaryTextDeltaEvent,
                            ),
                        )
                        and event.data.delta
                    ):
                        yield ThinkingDeltaEvent(thinking=event.data.delta)

                # Handle tool call and result events (both Responses and ChatCompletions)
                elif isinstance(event, RunItemStreamEvent):
                    if isinstance(event.item, ToolCallItem):
                        raw = event.item.raw_item
                        name = (
                            getattr(raw, "name", None)
                            or (raw.get("name") if isinstance(raw, dict) else "")
                            or ""
                        )
                        args = getattr(raw, "arguments", None) or ""
                        yield ToolCallEvent(
                            name=name,
                            input=args,
                            call_id=getattr(event.item, "call_id", "") or "",
                        )
                    elif isinstance(event.item, ToolCallOutputItem):
                        if hooks is not None:
                            hooks.call_ids.observed_output(event.item)
                        full_output = stringify(event.item.output)
                        yield ToolResultEvent(
                            output=full_output,
                            call_id=getattr(event.item, "call_id", "") or "",
                        )

            if hooks is not None:
                hooks.close()
            yield ContentBlockStopEvent()

            usage = result.context_wrapper.usage
            resp_model = last_actual_model or options.model
            counts = _model_usage(usage)
            yield ResultEvent(
                text=stringify(result.final_output),
                input_tokens=counts.get("input_tokens", 0),
                output_tokens=counts.get("output_tokens", 0),
                reasoning_tokens=counts.get("reasoning_tokens", 0),
                response_model=resp_model,
            )
        except BaseException as exc:
            if "hooks" in locals() and hooks is not None:
                hooks.close(exc)
            raise
        finally:
            if mcp_manager:
                await mcp_manager.__aexit__(None, None, None)
