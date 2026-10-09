"""OpenAI provider — wraps openai-agents SDK.

Uses SandboxAgent with native Shell, Filesystem, and Skills capabilities.
The SDK handles tool registration, skill discovery, and command execution.
"""
# mypy: disable-error-code=unused-ignore

from __future__ import annotations

import inspect
import json
import logging
import math
import os
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from opentelemetry.context import Context

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


from lightspeed_agentic.config import (
    azure_api_version_supports_responses,
    azure_api_version_supports_structured_outputs,
)
from lightspeed_agentic.skills import has_skills
from lightspeed_agentic.types import (
    MAX_TOOL_RETURN_CHARS,
    TOOL_RETURN_PREVIEW_CHARS,
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


def _model_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _finite_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite JSON number: {value}")
    return number


def _decode_tool_arguments(arguments: Any) -> Any:
    if not isinstance(arguments, str):
        return arguments
    try:
        return json.loads(
            arguments,
            parse_float=_finite_json_float,
            parse_constant=_finite_json_float,
        )
    except (ValueError, RecursionError):
        return arguments


def _generation_output_messages(response: Any) -> list[dict[str, Any]] | None:
    output = _model_field(response, "output")
    if output is None:
        return None

    parts: list[dict[str, Any]] = []
    for item in output:
        item_type = _model_field(item, "type")
        if item_type == "message":
            for content in _model_field(item, "content") or []:
                content_type = _model_field(content, "type")
                if content_type == "output_text":
                    text = _model_field(content, "text")
                    if isinstance(text, str):
                        parts.append({"type": "text", "content": text})
                elif content_type == "refusal":
                    refusal = _model_field(content, "refusal")
                    if isinstance(refusal, str):
                        parts.append({"type": "refusal", "content": refusal})
        elif item_type == "reasoning":
            for field, part_type in (
                ("content", "reasoning_text"),
                ("summary", "summary_text"),
            ):
                for reasoning in _model_field(item, field) or []:
                    text = _model_field(reasoning, "text")
                    if _model_field(reasoning, "type") == part_type and isinstance(text, str):
                        parts.append({"type": "reasoning", "content": text})
        elif item_type in {"function_call", "custom_tool_call"}:
            name = _model_field(item, "name")
            if name is None:
                continue
            part: dict[str, Any] = {"type": "tool_call", "name": name}
            call_id = _model_field(item, "call_id")
            if call_id:
                part["id"] = call_id
            if item_type == "function_call":
                arguments = _model_field(item, "arguments")
                if arguments is not None:
                    part["arguments"] = _decode_tool_arguments(arguments)
            else:
                tool_input = _model_field(item, "input")
                if tool_input is not None:
                    part["arguments"] = tool_input
            parts.append(part)

    return [{"role": "assistant", "parts": parts}]


def _generation_response_id(response: Any, api_type: str) -> str | None:
    if api_type == "responses":
        response_id = _model_field(response, "response_id")
        return response_id if isinstance(response_id, str) and response_id else None

    response_ids: set[str] = set()
    for item in _model_field(response, "output") or []:
        provider_data = _model_field(item, "provider_data")
        response_id = _model_field(provider_data, "response_id")
        if isinstance(response_id, str) and response_id:
            response_ids.add(response_id)
    return next(iter(response_ids)) if len(response_ids) == 1 else None


def _record_generation_response(span: Any, response: Any, api_type: str) -> None:
    from lightspeed_agentic.tracing import set_json_span_attribute

    if not span.is_recording():
        return

    messages = _generation_output_messages(response)
    if messages is not None:
        set_json_span_attribute(span, "gen_ai.output.messages", messages)

    response_id = _generation_response_id(response, api_type)
    if response_id is not None:
        span.set_attribute("gen_ai.response.id", response_id)

    usage = _model_field(response, "usage")
    if usage is not None and _model_field(usage, "requests", 0) > 0:
        for attribute, field in (
            ("gen_ai.usage.input_tokens", "input_tokens"),
            ("gen_ai.usage.output_tokens", "output_tokens"),
        ):
            value = _model_field(usage, field)
            if value is not None:
                span.set_attribute(attribute, value)

        output_details = _model_field(usage, "output_tokens_details")
        reasoning_tokens = _model_field(output_details, "reasoning_tokens")
        if reasoning_tokens:
            span.set_attribute("gen_ai.usage.reasoning.output_tokens", reasoning_tokens)


def _create_generation_hooks(
    main_agent: Any,
    options: ProviderQueryOptions,
    parent_context: Context,
    *,
    api_type: str,
) -> Any:
    """Capture completed main-agent OpenAI model responses as GenAI spans."""
    from agents import RunHooks
    from opentelemetry.trace import StatusCode

    from lightspeed_agentic.tracing import start_generation_span

    class GenerationHooks(RunHooks[Any]):
        def __init__(self) -> None:
            self._active_span: Any = None
            self._terminal = False

        def _close_active(self, error_type: str) -> None:
            span = self._active_span
            self._active_span = None
            if span is None:
                return
            if span.is_recording():
                span.set_attribute("error.type", error_type)
                span.set_status(StatusCode.ERROR)
            span.end()

        def close_open(self, error_type: str) -> None:
            self._terminal = True
            self._close_active(error_type)

        async def on_llm_start(
            self,
            context: Any,
            agent: Any,
            system_prompt: str | None,
            input_items: list[Any],
        ) -> None:
            del context, system_prompt, input_items
            if self._terminal or agent is not main_agent:
                return
            self._close_active("generation_interrupted")
            span = start_generation_span(
                "chat",
                options.model,
                "openai",
                parent_context=parent_context,
            )
            if span.is_recording():
                span.set_attribute("openai.api.type", api_type)
            self._active_span = span

        async def on_llm_end(
            self,
            context: Any,
            agent: Any,
            response: Any,
        ) -> None:
            del context
            if self._terminal or agent is not main_agent:
                return
            span = self._active_span
            self._active_span = None
            if span is None:
                return
            try:
                _record_generation_response(span, response, api_type)
            finally:
                span.end()

    return GenerationHooks()


class OpenAIProvider(AgentProvider):
    _client: Any = None
    _azure_credentials: dict[str, str] | None = None
    _azure_credential: Any = None

    async def aclose(self) -> None:
        """Close provider-owned async client and Azure credential resources."""
        client = self._client
        credential = self._azure_credential
        self._client = None
        self._azure_credential = None

        try:
            if client is not None:
                close = getattr(client, "close", None) or getattr(client, "aclose", None)
                if close is not None:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
        finally:
            if credential is not None:
                result = credential.close()
                if inspect.isawaitable(result):
                    await result

    @property
    def name(self) -> str:
        return "openai"

    def _build_azure_client(self, model: str) -> tuple[Any, Any]:
        """Build AsyncAzureOpenAI client and compatible model wrapper for Azure.

        Returns (client, model_wrapper). Entra ID mode uses azure_ad_token_provider;
        API-key mode uses api_key. API versions before 2025-03-01-preview use
        Chat Completions; newer versions use Responses API.
        """
        from openai import AsyncAzureOpenAI, DefaultAsyncHttpxClient

        from lightspeed_agentic.tls import get_ssl_context

        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
        if endpoint and urlparse(endpoint).scheme != "https":
            raise ValueError("AZURE_OPENAI_ENDPOINT must use https")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "").strip()
        if not api_version:
            raise ValueError("AZURE_OPENAI_API_VERSION is required")

        client_kwargs: dict[str, Any] = {
            "azure_endpoint": endpoint or None,
            "api_version": api_version,
            # Leave the base URL deployment-free so Chat Completions routes by request model.
            "http_client": DefaultAsyncHttpxClient(
                verify=get_ssl_context(),
                follow_redirects=False,
            ),
        }

        if self._azure_credentials:
            # Entra ID mode — import azure.identity.aio inside method (optional-extra)
            import aiohttp
            from azure.core.pipeline.transport import AioHttpTransport
            from azure.identity.aio import ClientSecretCredential, get_bearer_token_provider
            from openai.lib.azure import API_KEY_SENTINEL

            identity_session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=get_ssl_context()),
                cookie_jar=aiohttp.DummyCookieJar(),
                auto_decompress=False,
                trust_env=True,
            )
            identity_transport = AioHttpTransport(
                session=identity_session,
                session_owner=True,
            )
            credential = ClientSecretCredential(
                self._azure_credentials["tenant_id"],
                self._azure_credentials["client_id"],
                self._azure_credentials["client_secret"],
                transport=identity_transport,
            )
            self._azure_credential = credential
            token_provider = get_bearer_token_provider(
                credential,
                "https://cognitiveservices.azure.com/.default",
            )
            client_kwargs["api_key"] = API_KEY_SENTINEL
            client_kwargs["azure_ad_token_provider"] = token_provider
            logger.info("Azure OpenAI client: Entra ID (service principal)")
        else:
            # API-key mode
            client_kwargs["api_key"] = os.environ.get("AZURE_OPENAI_API_KEY", "")
            logger.info("Azure OpenAI client: API key")

        client = AsyncAzureOpenAI(**client_kwargs)
        return client, self._build_azure_model(client, model, api_version)

    def _build_azure_model(self, client: Any, model: str, api_version: str) -> Any:
        """Build the Azure model wrapper for the configured API version."""
        if azure_api_version_supports_responses(api_version):
            from agents.models.openai_responses import OpenAIResponsesModel

            return OpenAIResponsesModel(
                model=model,
                openai_client=client,
            )

        from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

        return OpenAIChatCompletionsModel(
            model=model,
            openai_client=client,
            buffer_streamed_tool_calls=True,
        )

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
        from opentelemetry import context as otel_context

        parent_context = otel_context.get_current()

        _ensure_openai_init()

        is_azure = os.environ.get("LIGHTSPEED_PROVIDER", "").strip().lower() == "azure"
        if (
            is_azure
            and options.output_schema
            and not azure_api_version_supports_structured_outputs(
                os.environ.get("AZURE_OPENAI_API_VERSION", "").strip()
            )
        ):
            raise ValueError(
                "Azure OpenAI structured output requires API version 2024-08-01 or later"
            )
        azure_model: Any | None = None

        if self._client is None:
            if is_azure:
                self._client, azure_model = self._build_azure_client(options.model)
            else:
                from openai import AsyncOpenAI, DefaultAsyncHttpxClient

                from lightspeed_agentic.tls import get_ssl_context

                self._client = AsyncOpenAI(
                    base_url=os.environ.get("OPENAI_BASE_URL"),
                    api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                    http_client=DefaultAsyncHttpxClient(
                        verify=get_ssl_context(),
                        follow_redirects=False,
                    ),
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

        # Setup model and capabilities based on endpoint.
        azure_uses_responses_api = is_azure and azure_api_version_supports_responses(
            os.environ.get("AZURE_OPENAI_API_VERSION", "").strip()
        )
        uses_responses_api = azure_uses_responses_api if is_azure else _is_native_openai()
        capabilities: list[Any] = [Shell()]
        function_tools_list: list[Any] | None = None

        if is_azure:
            # Azure client is cached, but the model wrapper is query/model specific.
            model: Any = azure_model or self._build_azure_model(
                self._client,
                options.model,
                os.environ.get("AZURE_OPENAI_API_VERSION", "").strip(),
            )
            if azure_uses_responses_api:
                capabilities.append(Filesystem())
            else:
                from lightspeed_agentic.function_tools import (
                    apply_patch,
                    list_directory,
                    read_file,
                    write_file,
                )

                function_tools_list = [read_file, write_file, list_directory, apply_patch]
        elif uses_responses_api:
            from agents.models.openai_responses import OpenAIResponsesModel

            model = OpenAIResponsesModel(model=options.model, openai_client=self._client)
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

        generation_hooks: Any = None

        try:
            if not uses_responses_api and mcp_servers_for_agent and function_tools_list is not None:
                function_tools_list.extend(await _build_mcp_function_tools(mcp_servers_for_agent))

            agent_kwargs: dict[str, Any] = {
                "name": "lightspeed",
                "instructions": options.system_prompt,
                "model": model,
                "capabilities": capabilities,
                "default_manifest": manifest,
                "mcp_servers": mcp_servers_for_agent if uses_responses_api else [],
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
                    options.output_schema, is_native=uses_responses_api
                )

            agent = SandboxAgent(**agent_kwargs)
            generation_hooks = _create_generation_hooks(
                agent,
                options,
                parent_context,
                api_type="responses" if uses_responses_api else "chat_completions",
            )

            run_config = RunConfig(
                sandbox=SandboxRunConfig(
                    client=UnixLocalSandboxClient(),
                ),
                call_model_input_filter=ToolOutputTrimmer(
                    max_output_chars=MAX_TOOL_RETURN_CHARS,
                    preview_chars=TOOL_RETURN_PREVIEW_CHARS,
                ),
            )

            result = Runner.run_streamed(
                agent,
                options.prompt,
                max_turns=options.max_turns,
                hooks=generation_hooks,
                run_config=run_config,
            )

            # Stream events from the runner
            async for event in result.stream_events():
                # Handle text and reasoning deltas from both Responses and ChatCompletions.
                # Both models emit ResponseTextDeltaEvent and ResponseReasoningTextDeltaEvent
                # (ChatCompletions converts its reasoning_content to Responses event types).
                if isinstance(event, RawResponsesStreamEvent):
                    if isinstance(event.data, ResponseTextDeltaEvent) and event.data.delta:
                        yield TextDeltaEvent(text=event.data.delta)
                    elif (
                        isinstance(
                            event.data,
                            ResponseReasoningTextDeltaEvent
                            | ResponseReasoningSummaryTextDeltaEvent,
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
                        raw_input = (
                            raw.get("input")
                            if isinstance(raw, dict)
                            else getattr(raw, "input", None)
                        )
                        raw_arguments = (
                            raw.get("arguments")
                            if isinstance(raw, dict)
                            else getattr(raw, "arguments", None)
                        )
                        trace_input = None
                        if raw_input is not None:
                            trace_input = stringify(raw_input)
                        elif raw_arguments is not None and (
                            isinstance(raw, dict) or isinstance(raw_arguments, dict)
                        ):
                            trace_input = stringify(raw_arguments)
                        yield ToolCallEvent(
                            name=name,
                            input=args,
                            call_id=getattr(event.item, "call_id", "") or "",
                            trace_input=trace_input,
                        )
                    elif isinstance(event.item, ToolCallOutputItem):
                        full_output = stringify(event.item.output)
                        yield ToolResultEvent(
                            output=full_output,
                            call_id=getattr(event.item, "call_id", "") or "",
                        )

            yield ContentBlockStopEvent()

            usage = result.context_wrapper.usage
            resp_model = getattr(result.context_wrapper, "model", "") or options.model
            details = getattr(usage, "output_tokens_details", None)
            reasoning = getattr(details, "reasoning_tokens", 0) if details else 0

            final_text = stringify(result.final_output)

            yield ResultEvent(
                text=final_text,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                reasoning_tokens=reasoning,
                response_model=resp_model,
            )
        except BaseException as error:
            if generation_hooks is not None:
                generation_hooks.close_open(type(error).__name__)
            raise
        finally:
            if generation_hooks is not None:
                generation_hooks.close_open("generation_interrupted")
            if mcp_manager:
                await mcp_manager.__aexit__(None, None, None)
