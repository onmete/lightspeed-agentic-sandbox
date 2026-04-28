"""E2E runner — HTTP client for all agent endpoints."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx


@dataclass
class QueryResult:
    provider: str = ""
    success: bool = False
    summary: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    status_code: int = 0
    latency_seconds: float = 0.0
    error: str | None = None


@dataclass
class SSEEvent:
    event: str
    data: dict[str, Any] | str


@dataclass
class ChatResult:
    provider: str = ""
    events: list[SSEEvent] = field(default_factory=list)
    status_code: int = 0
    latency_seconds: float = 0.0
    error: str | None = None


async def run_query(
    server_url: str,
    endpoint: str,
    query: str,
    system_prompt: str = "You are a helpful assistant.",
    output_schema: dict | None = None,
    context: dict | None = None,
) -> QueryResult:
    """POST to a query endpoint and return the parsed result."""
    result = QueryResult()
    start = time.monotonic()

    body: dict[str, Any] = {
        "query": query,
        "systemPrompt": system_prompt,
    }
    if output_schema:
        body["outputSchema"] = output_schema
    if context:
        body["context"] = context

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(f"{server_url}/v1/agent/{endpoint}", json=body)
            result.status_code = resp.status_code
            resp.raise_for_status()
            data = resp.json()

        result.success = data.get("success", False)
        result.summary = data.get("summary", "")
        result.raw = data
    except Exception as e:
        result.error = str(e)

    result.latency_seconds = time.monotonic() - start
    return result


async def run_analyze(
    server_url: str,
    query: str,
    system_prompt: str = "You are a helpful assistant.",
    output_schema: dict | None = None,
) -> QueryResult:
    return await run_query(server_url, "analyze", query, system_prompt, output_schema)


async def run_execute(
    server_url: str,
    query: str,
    system_prompt: str = "You are a helpful assistant.",
    output_schema: dict | None = None,
    context: dict | None = None,
) -> QueryResult:
    return await run_query(server_url, "execute", query, system_prompt, output_schema, context)


async def run_verify(
    server_url: str,
    query: str,
    system_prompt: str = "You are a helpful assistant.",
    output_schema: dict | None = None,
    context: dict | None = None,
) -> QueryResult:
    return await run_query(server_url, "verify", query, system_prompt, output_schema, context)


def _parse_sse_line(line: str) -> tuple[str | None, str | None]:
    """Parse a single SSE line into (field, value) or (None, None)."""
    if not line or line.startswith(":"):
        return None, None
    if ":" in line:
        field_name, _, value = line.partition(":")
        return field_name.strip(), value.strip()
    return line.strip(), ""


async def run_chat(
    server_url: str,
    message: str,
    conversation_id: str | None = None,
) -> ChatResult:
    """Send a chat message and collect SSE events."""
    result = ChatResult()
    start = time.monotonic()

    body: dict[str, Any] = {"message": message}
    if conversation_id:
        body["conversationId"] = conversation_id

    try:
        async with (
            httpx.AsyncClient(timeout=300.0) as client,
            client.stream("POST", f"{server_url}/v1/agent/chat", json=body) as resp,
        ):
                result.status_code = resp.status_code
                current_event = ""
                current_data = ""

                async for line in resp.aiter_lines():
                    if line.startswith("event:"):
                        current_event = line[6:].strip()
                    elif line.startswith("data:"):
                        current_data = line[5:].strip()
                    elif line == "" and current_event:
                        try:
                            data = json.loads(current_data)
                        except (json.JSONDecodeError, ValueError):
                            data = current_data
                        result.events.append(SSEEvent(event=current_event, data=data))
                        current_event = ""
                        current_data = ""
    except Exception as e:
        result.error = str(e)

    result.latency_seconds = time.monotonic() - start
    return result


def assert_tool_token(
    workspace: Path,
    token_file_name: str,
    result: QueryResult,
    provider_name: str,
    script_name: str,
) -> None:
    """Verify that tool-generated tokens appear in the response."""
    matches = list(workspace.rglob(token_file_name))
    if not matches:
        raise AssertionError(
            f"{provider_name} did not run {script_name} "
            f"(no {token_file_name} found in {workspace})"
        )

    response_text = json.dumps(result.raw)
    tokens = matches[0].read_text().strip().splitlines()
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        assert token in response_text, (
            f"{provider_name} did not report verification token {token} from {script_name}. "
            f"response={result.summary[:200]}"
        )
