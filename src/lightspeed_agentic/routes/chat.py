"""SSE chat endpoint — maps to lightspeed-agent/src/chat.ts.

Streaming chat with fence-aware text buffering for inline UI components.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from lightspeed_agentic.types import AgentProvider, ProviderQueryOptions

logger = logging.getLogger("lightspeed_agentic")


# ---------------------------------------------------------------------------
# Conversation store — maps to chat.ts conversation management
# ---------------------------------------------------------------------------

_MAX_CONVERSATIONS = 100
_CONVERSATION_TTL_S = 3600
_MAX_HISTORY_MESSAGES = 20


class _ConversationEntry(BaseModel):
    role: str
    content: str


class _Conversation(BaseModel):
    id: str
    messages: list[_ConversationEntry] = []
    created_at: float = 0
    last_accessed_at: float = 0


_conversations: dict[str, _Conversation] = {}


def _cleanup() -> None:
    import time

    now = time.time()
    expired = [k for k, v in _conversations.items() if now - v.last_accessed_at > _CONVERSATION_TTL_S]
    for k in expired:
        del _conversations[k]
    if len(_conversations) > _MAX_CONVERSATIONS:
        by_access = sorted(_conversations.items(), key=lambda kv: kv[1].last_accessed_at)
        for k, _ in by_access[: len(_conversations) - _MAX_CONVERSATIONS]:
            del _conversations[k]


def _get_or_create(conversation_id: str | None) -> _Conversation:
    import time

    _cleanup()
    if conversation_id and conversation_id in _conversations:
        conv = _conversations[conversation_id]
        conv.last_accessed_at = time.time()
        return conv
    conv = _Conversation(id=str(uuid.uuid4()), created_at=time.time(), last_accessed_at=time.time())
    _conversations[conv.id] = conv
    return conv


# ---------------------------------------------------------------------------
# Chat request model
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel, extra="allow"):
    message: str
    conversationId: str | None = None
    context: dict[str, Any]


# ---------------------------------------------------------------------------
# SSE formatting
# ---------------------------------------------------------------------------


def _sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Fence-aware text buffer — maps to chat.ts processTextBuffer/flushTextBuffer
# ---------------------------------------------------------------------------

_FENCE_OPEN = re.compile(r"```ui:(\w+)\n")
_FENCE_PARTIAL = re.compile(r"`{1,3}(?:u(?:i(?::(?:\w+)?)?)?)?$")


class _FenceBuffer:
    def __init__(self) -> None:
        self.buf = ""
        self.in_fence = False
        self.fence_type = ""
        self.output: list[str] = []

    def add(self, text: str) -> None:
        self.buf += text
        self._process()

    def _process(self) -> None:
        while self.buf:
            if self.in_fence:
                close = self.buf.find("```")
                if close < 0:
                    break
                json_str = self.buf[:close].strip()
                self.buf = self.buf[close + 3:]
                self.in_fence = False
                try:
                    props = json.loads(json_str)
                    self.output.append(_sse_event("ui_component", {"type": self.fence_type, "props": props}))
                except (json.JSONDecodeError, TypeError):
                    pass
                self.fence_type = ""
            else:
                m = _FENCE_OPEN.search(self.buf)
                if m and m.start() is not None:
                    before = self.buf[:m.start()]
                    if before:
                        self.output.append(_sse_event("text", {"content": before}))
                    self.fence_type = m.group(1)
                    self.buf = self.buf[m.end():]
                    self.in_fence = True
                else:
                    partial = _FENCE_PARTIAL.search(self.buf)
                    safe = len(self.buf) - len(partial.group(0)) if partial else len(self.buf)
                    if safe > 0:
                        self.output.append(_sse_event("text", {"content": self.buf[:safe]}))
                        self.buf = self.buf[safe:]
                    break

    def flush(self) -> None:
        if self.in_fence:
            self.output.append(_sse_event("text", {"content": "```ui:" + self.fence_type + "\n" + self.buf}))
        elif self.buf:
            self.output.append(_sse_event("text", {"content": self.buf}))
        self.buf = ""
        self.in_fence = False
        self.fence_type = ""

    def drain(self) -> list[str]:
        events = self.output
        self.output = []
        return events


# ---------------------------------------------------------------------------
# Chat system prompt builder — maps to chat.ts buildChatSystemPrompt
# ---------------------------------------------------------------------------


def _build_chat_system_prompt(ctx: dict[str, Any]) -> str:
    parts: list[str] = [
        "You are an expert SRE agent for OpenShift clusters, having a conversation "
        "with a human operator about an alert remediation. Be conversational, helpful, "
        "and concise. You have read-only access to the cluster via CLI tools.\n\n"
        "You have the following CLI tools available:\n"
        "- `oc` — OpenShift CLI for inspecting the cluster (kubectl compatible)\n"
        "- `curl` — HTTP client for API calls\n"
        "- The `prometheus` skill for Prometheus metric queries\n\n"
        "You also have access to skills for Red Hat support, platform documentation, "
        "and GitHub operations.",
    ]

    if rem := ctx.get("remediation"):
        parts.append(f"\n## Current Remediation: {rem.get('name', '')}")
        parts.append(f"Namespace: {rem.get('namespace', '')}")

    if alert := ctx.get("alert"):
        parts.append(f"\n### Alert\n- Name: {alert.get('name', '')}")
        parts.append(f"- Status: {alert.get('status', '')}")
        parts.append(f"- Severity: {alert.get('severity', '')}")

    if diag := ctx.get("diagnosis"):
        parts.append(f"\n### Diagnosis\n- Summary: {diag.get('summary', '')}")
        parts.append(f"- Root Cause: {diag.get('rootCause', '')}")

    if prop := ctx.get("proposal"):
        parts.append(f"\n### Current Proposal\n- Description: {prop.get('description', '')}")
        parts.append(f"- Risk: {prop.get('risk', '')}")

    parts.append("\n## Rules\n- You are in READ-ONLY mode. Do NOT modify any cluster resources.")

    return "\n".join(parts)


def _build_chat_prompt(message: str, history: list[_ConversationEntry]) -> str:
    if not history:
        return message
    recent = history[-_MAX_HISTORY_MESSAGES:]
    parts = ["## Previous Conversation\n"]
    for entry in recent:
        role = "Human" if entry.role == "user" else "Agent"
        parts.append(f"### {role}\n{entry.content}\n")
    parts.append(f"## Current Message\n{message}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def register_chat_routes(
    router: APIRouter,
    *,
    provider: AgentProvider,
    skills_dir: str,
    model: str,
    max_turns: int,
    timeout_ms: int,
    max_budget_usd: float,
) -> None:

    @router.post("/chat")
    async def chat(req: ChatRequest, request: Request) -> StreamingResponse:

        async def stream() -> AsyncGenerator[str, None]:
            conversation = _get_or_create(req.conversationId)
            system_prompt = _build_chat_system_prompt(req.context)
            user_prompt = _build_chat_prompt(req.message, conversation.messages)

            conversation.messages.append(_ConversationEntry(role="user", content=req.message))

            yield _sse_event("status", {"status": "thinking"})

            agent_text = ""
            total_cost = 0.0
            fence = _FenceBuffer()

            try:
                result = provider.query(ProviderQueryOptions(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    model=model,
                    max_turns=max_turns,
                    max_budget_usd=max_budget_usd,
                    allowed_tools=["Bash", "Read", "Glob", "Grep", "Skill"],
                    cwd=skills_dir,
                    stream=True,
                ))

                async def run() -> None:
                    nonlocal agent_text, total_cost
                    async for event in result:
                        if await request.is_disconnected():
                            break
                        match event.type:
                            case "text_delta":
                                fence.add(event.text)
                            case "thinking_delta":
                                fence.output.append(_sse_event("thinking", {"content": event.thinking}))
                            case "content_block_stop":
                                fence.flush()
                            case "tool_call":
                                fence.output.append(_sse_event("tool_call", {"name": event.name, "input": event.input}))
                            case "tool_result":
                                fence.output.append(_sse_event("tool_result", {"output": event.output}))
                            case "result":
                                total_cost = event.cost_usd
                                agent_text = event.text

                await asyncio.wait_for(run(), timeout=timeout_ms / 1000)

            except asyncio.TimeoutError:
                fence.output.append(_sse_event("error", {"message": f"Chat timed out after {timeout_ms}ms"}))
            except Exception as e:
                logger.exception("[agent] Chat error")
                fence.output.append(_sse_event("error", {"message": str(e)}))

            fence.flush()
            for chunk in fence.drain():
                yield chunk

            if agent_text:
                clean_text = re.sub(r"```ui:\w+\n[\s\S]*?```", "", agent_text).strip()
                conversation.messages.append(_ConversationEntry(role="assistant", content=clean_text))

            yield _sse_event("done", {"conversationId": conversation.id})
            logger.info("[agent] Chat complete: conversation=%s, cost=$%.4f", conversation.id, total_cost)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        })
