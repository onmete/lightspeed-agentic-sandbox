"""Tests for Azure OpenAI adapter construction in the OpenAI provider.

Verifies that the adapter constructs AsyncAzureOpenAI + OpenAIResponsesModel
for Azure (Responses API supported since api-version 2025-03-01-preview),
and correctly wires Entra ID token provider vs API key.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from lightspeed_agentic.providers.openai import OpenAIProvider


@pytest.fixture(autouse=True)
def _clean_azure_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove Azure-specific env vars to isolate tests."""
    for var in [
        "LIGHTSPEED_PROVIDER",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_BASE_URL",
    ]:
        monkeypatch.delenv(var, raising=False)


def _setup_azure_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTSPEED_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://myresource.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1")


class TestAzureClientConstruction:
    """Verify _build_azure_client returns AsyncAzureOpenAI with correct params."""

    def test_api_key_mode_builds_azure_client(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """API-key mode: AsyncAzureOpenAI with api_key, wrapped in OpenAIResponsesModel."""
        _setup_azure_env(monkeypatch)
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")

        provider = OpenAIProvider()
        client, model_wrapper = provider._build_azure_client("gpt-4.1")

        # Verify the client type
        from openai import AsyncAzureOpenAI

        assert isinstance(client, AsyncAzureOpenAI)

        # Verify model wrapper type — Azure uses Responses API
        from agents.models.openai_responses import OpenAIResponsesModel

        assert isinstance(model_wrapper, OpenAIResponsesModel)

    def test_azure_client_preserves_tls_context_and_disables_redirects(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Azure HTTP client must preserve TLS config and avoid auth header redirects."""
        _setup_azure_env(monkeypatch)
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        shared_context = object()

        with (
            patch("lightspeed_agentic.tls.get_ssl_context", return_value=shared_context),
            patch("openai.DefaultAsyncHttpxClient") as http_client,
            patch("openai.AsyncAzureOpenAI.__init__", return_value=None),
        ):
            http_client.return_value = MagicMock()
            OpenAIProvider()._build_azure_client("gpt-4.1")

        http_client.assert_called_once_with(
            verify=shared_context,
            follow_redirects=False,
        )

    def test_rejects_non_https_endpoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Configured Azure endpoints must use HTTPS."""
        _setup_azure_env(monkeypatch)
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "http://myresource.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")

        with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT must use https"):
            OpenAIProvider()._build_azure_client("gpt-4.1")

    def test_empty_endpoint_is_preserved_as_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Empty endpoint keeps existing None handling for SDK/env fallback."""
        _setup_azure_env(monkeypatch)
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")

        with patch("openai.AsyncAzureOpenAI.__init__", return_value=None) as azure_init:
            OpenAIProvider()._build_azure_client("gpt-4.1")

        assert azure_init.call_args.kwargs["azure_endpoint"] is None

    def test_legacy_api_version_uses_chat_completions_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Azure API versions before 2025-03-01-preview use Chat Completions."""
        _setup_azure_env(monkeypatch)
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")

        provider = OpenAIProvider()
        client, model_wrapper = provider._build_azure_client("gpt-4.1")

        from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
        from openai import AsyncAzureOpenAI

        assert isinstance(client, AsyncAzureOpenAI)
        assert isinstance(model_wrapper, OpenAIChatCompletionsModel)

    def test_missing_api_version_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Azure client construction requires an explicit API version."""
        _setup_azure_env(monkeypatch)
        monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")

        with pytest.raises(ValueError, match="AZURE_OPENAI_API_VERSION is required"):
            OpenAIProvider()._build_azure_client("gpt-4.1")

    def test_entra_id_mode_builds_azure_client_with_token_provider(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Entra ID mode: AsyncAzureOpenAI with azure_ad_token_provider."""
        _setup_azure_env(monkeypatch)

        mock_credential = MagicMock()
        mock_token_provider = AsyncMock(return_value="fake-token")
        mock_session = MagicMock()
        mock_connector = MagicMock()
        mock_transport = MagicMock()
        mock_cookie_jar = MagicMock()
        shared_context = object()

        with (
            patch("lightspeed_agentic.tls.get_ssl_context", return_value=shared_context),
            patch("aiohttp.TCPConnector", return_value=mock_connector) as connector_cls,
            patch("aiohttp.DummyCookieJar", return_value=mock_cookie_jar),
            patch("aiohttp.ClientSession", return_value=mock_session) as session_cls,
            patch("azure.core.pipeline.transport.AioHttpTransport", return_value=mock_transport),
            patch(
                "azure.identity.aio.ClientSecretCredential",
                return_value=mock_credential,
            ) as csc_cls,
            patch(
                "azure.identity.aio.get_bearer_token_provider",
                return_value=mock_token_provider,
            ) as gbtp,
        ):
            provider = OpenAIProvider()
            provider._azure_credentials = {
                "client_id": "cid",
                "tenant_id": "tid",
                "client_secret": "csec",
            }
            client, model_wrapper = provider._build_azure_client("gpt-4.1")

        connector_cls.assert_called_once_with(ssl=shared_context)
        session_cls.assert_called_once_with(
            connector=mock_connector,
            cookie_jar=mock_cookie_jar,
            auto_decompress=False,
            trust_env=True,
        )

        # Verify ClientSecretCredential was constructed
        csc_cls.assert_called_once_with("tid", "cid", "csec", transport=mock_transport)

        # Verify get_bearer_token_provider was called with the credential
        gbtp.assert_called_once_with(
            mock_credential,
            "https://cognitiveservices.azure.com/.default",
        )

        from openai import AsyncAzureOpenAI

        assert isinstance(client, AsyncAzureOpenAI)

        from agents.models.openai_responses import OpenAIResponsesModel

        assert isinstance(model_wrapper, OpenAIResponsesModel)

    def test_entra_id_suppresses_env_api_key_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Entra ID mode must not read AZURE_OPENAI_API_KEY from the SDK fallback."""
        from openai.lib.azure import API_KEY_SENTINEL

        _setup_azure_env(monkeypatch)
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key-must-not-leak")

        mock_credential = MagicMock()
        mock_token_provider = AsyncMock(return_value="fake-token")

        with (
            patch("aiohttp.TCPConnector"),
            patch("aiohttp.DummyCookieJar"),
            patch("aiohttp.ClientSession"),
            patch("azure.core.pipeline.transport.AioHttpTransport"),
            patch(
                "azure.identity.aio.ClientSecretCredential",
                return_value=mock_credential,
            ),
            patch(
                "azure.identity.aio.get_bearer_token_provider",
                return_value=mock_token_provider,
            ),
            patch(
                "openai.AsyncAzureOpenAI.__init__",
                return_value=None,
            ) as azure_init,
        ):
            provider = OpenAIProvider()
            provider._azure_credentials = {
                "client_id": "cid",
                "tenant_id": "tid",
                "client_secret": "csec",
            }
            provider._build_azure_client("gpt-4.1")

        call_kwargs = azure_init.call_args
        assert call_kwargs.kwargs["api_key"] == API_KEY_SENTINEL

    @pytest.mark.asyncio
    async def test_second_azure_query_rebuilds_model_from_cached_client(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Provider reuse must not depend on first-query local Azure model state."""
        _setup_azure_env(monkeypatch)

        from lightspeed_agentic.types import ProviderQueryOptions

        async def empty_stream() -> Any:
            return
            yield

        mock_result = MagicMock()
        mock_result.stream_events = empty_stream
        mock_result.final_output = "ok"
        mock_result.context_wrapper.usage.input_tokens = 0
        mock_result.context_wrapper.usage.output_tokens = 0
        client = MagicMock()
        first_model = MagicMock(name="first_model")
        second_model = MagicMock(name="second_model")
        provider = OpenAIProvider()
        options = ProviderQueryOptions(
            prompt="test",
            system_prompt="system",
            model="gpt-4.1",
            max_turns=1,
            allowed_tools=[],
            cwd=str(tmp_path),
        )

        with (
            patch("lightspeed_agentic.providers.openai._ensure_openai_init"),
            patch.object(provider, "_build_azure_client", return_value=(client, first_model)),
            patch("agents.Runner.run_streamed", return_value=mock_result),
            patch("agents.sandbox.SandboxAgent", return_value=MagicMock()) as sandbox_agent,
            patch(
                "agents.models.openai_responses.OpenAIResponsesModel", return_value=second_model
            ) as responses_model,
        ):
            [event async for event in provider.query(options)]
            [event async for event in provider.query(options)]

        responses_model.assert_called_once_with(model="gpt-4.1", openai_client=client)
        assert sandbox_agent.call_args_list[0].kwargs["model"] is first_model
        assert sandbox_agent.call_args_list[1].kwargs["model"] is second_model

    @pytest.mark.asyncio
    async def test_chat_completions_routes_each_query_to_its_deployment(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A cached Azure client must route a model override to its own deployment."""
        from lightspeed_agentic.types import ProviderQueryOptions

        _setup_azure_env(monkeypatch)
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        requested_paths: list[str] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requested_paths.append(request.url.path)
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        provider = OpenAIProvider()

        def fake_run(agent: Any, *_args: Any, **_kwargs: Any) -> Any:
            async def events() -> Any:
                await agent.model._client.chat.completions.create(
                    model=agent.model.model,
                    messages=[{"role": "user", "content": "hi"}],
                )
                if False:
                    yield

            result = MagicMock()
            result.stream_events = events
            result.final_output = "ok"
            result.context_wrapper.usage.input_tokens = 0
            result.context_wrapper.usage.output_tokens = 0
            return result

        try:
            with (
                patch("lightspeed_agentic.providers.openai._ensure_openai_init"),
                patch("openai.DefaultAsyncHttpxClient", return_value=http_client),
                patch(
                    "agents.sandbox.SandboxAgent", side_effect=lambda **kw: SimpleNamespace(**kw)
                ),
                patch("agents.Runner.run_streamed", side_effect=fake_run),
            ):
                for deployment in ("first-deployment", "second-deployment"):
                    options = ProviderQueryOptions(
                        prompt="hi",
                        system_prompt="system",
                        model=deployment,
                        max_turns=1,
                        allowed_tools=[],
                        cwd=str(tmp_path),
                    )
                    [event async for event in provider.query(options)]
        finally:
            await provider.aclose()

        assert requested_paths == [
            "/openai/deployments/first-deployment/chat/completions",
            "/openai/deployments/second-deployment/chat/completions",
        ]

    @pytest.mark.asyncio
    async def test_legacy_azure_rejects_unsupported_structured_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Do not send json_schema response_format to pre-2024-08 Azure APIs."""
        from lightspeed_agentic.types import ProviderQueryOptions

        _setup_azure_env(monkeypatch)
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        provider = OpenAIProvider()
        provider._client = MagicMock()
        options = ProviderQueryOptions(
            prompt="hi",
            system_prompt="system",
            model="deployment",
            max_turns=1,
            allowed_tools=[],
            cwd=str(tmp_path),
            output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
        )
        result = MagicMock()
        result.final_output = "ok"
        result.context_wrapper.usage.input_tokens = 0
        result.context_wrapper.usage.output_tokens = 0

        async def empty_stream() -> Any:
            return
            yield

        result.stream_events = empty_stream
        with (
            patch("lightspeed_agentic.providers.openai._ensure_openai_init"),
            patch("agents.sandbox.SandboxAgent"),
            patch("agents.Runner.run_streamed", return_value=result) as runner,
        ):
            with pytest.raises(
                ValueError, match="Azure OpenAI structured output requires API version"
            ):
                [event async for event in provider.query(options)]
            runner.assert_not_called()

            options.output_schema = None
            [event async for event in provider.query(options)]
            runner.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("api_version", "expect_responses"),
        [("2024-08-01-preview", False), ("2025-03-01-preview", True)],
    )
    async def test_azure_query_mcp_and_schema_follow_api_version(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        api_version: str,
        expect_responses: bool,
    ) -> None:
        """Legacy Azure converts MCP tools and leaves output schema non-strict."""
        from agents.mcp.util import MCPUtil

        from lightspeed_agentic.mcp import AdmittedMCPProviderServer
        from lightspeed_agentic.types import ProviderQueryOptions

        _setup_azure_env(monkeypatch)
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", api_version)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        admitted = AdmittedMCPProviderServer(
            name="test", url="https://mcp.example/mcp", allowed_tool_names=("lookup",)
        )
        tool = object()
        server = MagicMock()
        server.list_tools = AsyncMock(return_value=[object()])
        manager = MagicMock(active_servers=[server])
        manager.__aenter__ = AsyncMock(return_value=manager)
        manager.__aexit__ = AsyncMock(return_value=None)

        async def empty_stream() -> Any:
            return
            yield

        result = MagicMock()
        result.stream_events = empty_stream
        result.final_output = "ok"
        result.context_wrapper.usage.input_tokens = 0
        result.context_wrapper.usage.output_tokens = 0
        options = ProviderQueryOptions(
            prompt="hi",
            system_prompt="system",
            model="deployment",
            max_turns=1,
            allowed_tools=[],
            cwd=str(tmp_path),
            mcp_servers=[admitted],
            output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
        )
        provider = OpenAIProvider()
        provider._client = MagicMock()
        with (
            patch("lightspeed_agentic.providers.openai._ensure_openai_init"),
            patch("lightspeed_agentic.mcp.to_openai_mcp_servers", return_value=[object()]),
            patch("agents.mcp.MCPServerManager", return_value=manager),
            patch.object(MCPUtil, "to_function_tool", return_value=tool) as convert_tool,
            patch("agents.sandbox.SandboxAgent") as agent,
            patch("agents.Runner.run_streamed", return_value=result),
        ):
            [event async for event in provider.query(options)]

        kwargs = agent.call_args.kwargs
        assert kwargs["output_type"].is_strict_json_schema() is expect_responses
        if expect_responses:
            assert kwargs["mcp_servers"] == [server]
            assert "tools" not in kwargs
            convert_tool.assert_not_called()
        else:
            assert kwargs["mcp_servers"] == []
            assert tool in kwargs["tools"]
            convert_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_aclose_closes_cached_azure_credential(self) -> None:
        """Provider teardown closes the async Entra credential transport."""
        provider = OpenAIProvider()
        mock_credential = MagicMock()
        mock_credential.close = AsyncMock()
        provider._azure_credential = mock_credential

        await provider.aclose()

        mock_credential.close.assert_awaited_once_with()
