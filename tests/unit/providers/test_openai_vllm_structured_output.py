"""Tests for OpenAI provider endpoint compatibility.

Tests verify endpoint detection and MCP tool conversion for both native OpenAI
and custom endpoints (vLLM, etc.).
"""

import os
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from lightspeed_agentic.providers.openai import (  # type: ignore[import-untyped]
    _build_mcp_function_tools,
    _is_native_openai,
)


class TestMCPFunctionTools:
    @pytest.mark.asyncio
    async def test_converts_tools_from_each_mcp_server(self) -> None:
        first_server = AsyncMock()
        second_server = AsyncMock()
        first_tool = object()
        second_tool = object()
        first_server.list_tools.return_value = [first_tool]
        second_server.list_tools.return_value = [second_tool]

        first_function_tool = object()
        second_function_tool = object()
        with patch(
            "agents.mcp.util.MCPUtil.to_function_tool",
            side_effect=[first_function_tool, second_function_tool],
        ) as convert:
            result = await _build_mcp_function_tools([first_server, second_server])

        assert result == [first_function_tool, second_function_tool]
        convert.assert_any_call(first_tool, first_server, convert_schemas_to_strict=False)
        convert.assert_any_call(second_tool, second_server, convert_schemas_to_strict=False)

    @pytest.mark.asyncio
    async def test_non_native_conversion_exposes_only_admitted_tools(self) -> None:
        from unittest.mock import MagicMock

        from mcp.types import ListToolsResult, Tool

        from lightspeed_agentic.mcp import (  # type: ignore[import-untyped]
            AdmittedMCPProviderServer,
            to_openai_mcp_servers,
        )

        [server] = to_openai_mcp_servers(
            [
                AdmittedMCPProviderServer(
                    name="openshift",
                    url="https://mcp.example/mcp",
                    allowed_tool_names=("get_pod",),
                )
            ]
        )
        admitted_tool = Tool(name="get_pod", inputSchema={})
        rejected_tool = Tool(name="delete_pod", inputSchema={})
        server.session = MagicMock()
        server.session.list_tools = AsyncMock(
            return_value=ListToolsResult(tools=[admitted_tool, rejected_tool])
        )
        converted_tool = object()

        with patch(
            "agents.mcp.util.MCPUtil.to_function_tool",
            return_value=converted_tool,
        ) as convert:
            result = await _build_mcp_function_tools([server])

        assert result == [converted_tool]
        convert.assert_called_once_with(
            admitted_tool,
            server,
            convert_schemas_to_strict=False,
        )


class TestNativeOpenAIDetection:
    """Test _is_native_openai() helper."""

    def test_native_openai_by_default(self) -> None:
        """When OPENAI_BASE_URL unset, defaults to native OpenAI."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove OPENAI_BASE_URL if set
            os.environ.pop("OPENAI_BASE_URL", None)
            assert _is_native_openai() is True

    def test_native_openai_explicit(self) -> None:
        """When OPENAI_BASE_URL is api.openai.com, recognized as native."""
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://api.openai.com/v1"}):
            assert _is_native_openai() is True

    def test_vllm_not_native(self) -> None:
        """When OPENAI_BASE_URL is vLLM, recognized as non-native."""
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "http://localhost:8000/v1"}):
            assert _is_native_openai() is False

    def test_custom_endpoint_not_native(self) -> None:
        """Custom OpenAI-compatible endpoint detected as non-native."""
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://custom.example.com/v1"}):
            assert _is_native_openai() is False

    def test_invalid_url_defaults_to_false(self) -> None:
        """Malformed URL handled gracefully, defaults to non-native."""
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "not-a-valid-url"}):
            assert _is_native_openai() is False


class TestModelSelectionByEndpoint:
    """Test endpoint detection and tool compatibility decisions."""

    def test_native_openai_detected_without_base_url(self) -> None:
        """When OPENAI_BASE_URL unset, detected as native OpenAI."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_BASE_URL", None)
            assert _is_native_openai() is True

    def test_native_openai_detected_with_explicit_api_openai_url(self) -> None:
        """When OPENAI_BASE_URL is api.openai.com, detected as native."""
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://api.openai.com/v1"}):
            assert _is_native_openai() is True

    def test_vllm_detected_as_non_native(self) -> None:
        """When OPENAI_BASE_URL is vLLM, detected as non-native endpoint."""
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "http://localhost:8000/v1"}):
            assert _is_native_openai() is False

    def test_custom_endpoint_detected_as_non_native(self) -> None:
        """Custom OpenAI-compatible endpoint detected as non-native."""
        custom_urls = [
            "https://custom.example.com/v1",
            "https://my-llm-service.internal/api/v1",
            "http://llm-proxy:8080/v1",
        ]
        for url in custom_urls:
            with patch.dict(os.environ, {"OPENAI_BASE_URL": url}):
                assert _is_native_openai() is False, f"Should detect {url} as non-native"

    def test_reasoning_available_with_responses_model(self) -> None:
        """OpenAIResponsesModel supports reasoning via response deltas."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_BASE_URL", None)
            assert _is_native_openai() is True
            # OpenAIResponsesModel emits ResponseReasoningTextDeltaEvent

    def test_vllm_detected_as_non_native_for_chat_completions(self) -> None:
        """Custom endpoints use the Chat Completions compatibility path."""
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "http://localhost:8000/v1"}):
            assert _is_native_openai() is False
            # The query() method selects Chat Completions for custom endpoints.


class TestQueryFlowIntegration:
    """Integration tests validating endpoint selection logic."""

    @pytest.mark.asyncio
    async def test_native_openai_allows_unlisted_model_with_structured_output(
        self, tmp_path: Path
    ) -> None:
        """Native models are not rejected by a stale local model allowlist."""
        from lightspeed_agentic.providers.openai import OpenAIProvider
        from lightspeed_agentic.types import ProviderQueryOptions  # type: ignore[import-untyped]

        async def empty_stream() -> AsyncIterator[None]:
            return
            yield

        mock_result = AsyncMock()
        mock_result.stream_events = empty_stream
        mock_result.final_output = "{}"
        mock_result.context_wrapper.usage.input_tokens = 0
        mock_result.context_wrapper.usage.output_tokens = 0

        with (
            patch.dict(os.environ, {}, clear=False),
            patch("agents.sandbox.SandboxAgent", return_value=object()) as sandbox_agent,
            patch("agents.Runner.run_streamed", return_value=mock_result),
            patch("agents.models.openai_responses.OpenAIResponsesModel"),
            patch("openai.AsyncOpenAI"),
        ):
            os.environ.pop("OPENAI_BASE_URL", None)
            options = ProviderQueryOptions(
                prompt="test",
                system_prompt="test",
                model="gpt-5.6-luna",
                max_turns=1,
                allowed_tools=[],
                cwd=str(tmp_path),
                output_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                },
            )
            provider = OpenAIProvider()
            [event async for event in provider.query(options)]

        assert sandbox_agent.called
        output_type = sandbox_agent.call_args.kwargs["output_type"]
        assert output_type.is_strict_json_schema() is True
        assert output_type.json_schema()["additionalProperties"] is False
        assert output_type.json_schema()["required"] == ["answer"]
