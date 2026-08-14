"""Lightweight mock MCP server for E2E testing.

Exposes two tools via Streamable HTTP transport:
- echo(message): returns the message as-is
- list_namespaces(): returns a static list of cluster namespaces

Run standalone:
    python tests/e2e/mock_mcp_server.py [--port 19090]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

import uvicorn
from mcp.server import Server
from mcp.server.streamable_http import StreamableHTTPServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Route

logger = logging.getLogger("mock_mcp_server")

# "e2e-sentinel-ns-7f3a9" is an unguessable marker: it can only appear in a run
# summary if the agent actually invoked list_namespaces (a model cannot hallucinate
# it the way it can guess "default"/"kube-system"). Keep in sync with steps/then.py.
MOCK_NAMESPACES = [
    "default",
    "kube-system",
    "openshift-lightspeed",
    "openshift-monitoring",
    "e2e-sentinel-ns-7f3a9",
]

mcp_server = Server("mock-ocp-mcp")


@mcp_server.list_tools()
async def list_tools():
    from mcp.types import Tool

    return [
        Tool(
            name="echo",
            description="Echoes back the provided message.",
            inputSchema={
                "type": "object",
                "properties": {"message": {"type": "string", "description": "Message to echo"}},
                "required": ["message"],
            },
        ),
        Tool(
            name="list_namespaces",
            description="Lists cluster namespaces.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict):
    from mcp.types import TextContent

    if name == "echo":
        msg = arguments.get("message", "")
        return [TextContent(type="text", text=msg)]
    if name == "list_namespaces":
        ns_list = ", ".join(MOCK_NAMESPACES)
        return [TextContent(type="text", text=f"Namespaces: {ns_list}")]
    raise ValueError(f"Unknown tool: {name}")


class _SessionState:
    def __init__(self):
        self.transport: StreamableHTTPServerTransport | None = None
        self.task: asyncio.Task | None = None
        self.ready = asyncio.Event()


_sessions: dict[str, _SessionState] = {}


async def _start_session(session_id: str) -> _SessionState:
    """Create transport, wire it to the MCP server, return session state once ready."""
    from mcp.server import InitializationOptions
    from mcp.types import ServerCapabilities, ToolsCapability

    state = _SessionState()
    state.transport = StreamableHTTPServerTransport(mcp_session_id=session_id)

    async def _run():
        async with state.transport.connect() as (read_stream, write_stream):
            state.ready.set()
            await mcp_server.run(
                read_stream,
                write_stream,
                initialization_options=InitializationOptions(
                    server_name="mock-ocp-mcp",
                    server_version="0.1.0",
                    capabilities=ServerCapabilities(tools=ToolsCapability()),
                ),
                stateless=True,
            )

    state.task = asyncio.create_task(_run())
    _sessions[session_id] = state
    try:
        await asyncio.wait_for(state.ready.wait(), timeout=10)
    except TimeoutError:
        state.task.cancel()
        del _sessions[session_id]
        raise RuntimeError(f"MCP session {session_id} failed to start within 10s") from None
    return state


class _MCPEndpoint:
    """Raw ASGI endpoint — hands control directly to the MCP transport."""

    async def __call__(self, scope, receive, send) -> None:
        request = Request(scope, receive, send)
        session_id = request.headers.get("mcp-session-id")

        if session_id and session_id in _sessions:
            state = _sessions[session_id]
        else:
            session_id = str(uuid.uuid4())
            state = await _start_session(session_id)

        await state.transport.handle_request(scope, receive, send)


def create_app() -> Starlette:
    @asynccontextmanager
    async def lifespan(_app):
        yield
        for state in _sessions.values():
            if state.task:
                state.task.cancel()
        _sessions.clear()

    app = Starlette(
        routes=[Route("/mcp", _MCPEndpoint(), methods=["GET", "POST", "DELETE"])],
        lifespan=lifespan,
    )
    return app


def main():
    parser = argparse.ArgumentParser(description="Mock MCP server for E2E tests")
    parser.add_argument("--port", type=int, default=19090)
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
