"""SSE chat endpoint — maps to lightspeed-agent/src/chat.ts."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import AsyncGenerator, Iterator
from dataclasses import dataclass
from typing import Any, Literal

from ag_ui.core import (
    CustomEvent,
    EventType,
    ReasoningEndEvent,
    ReasoningMessageContentEvent,
    ReasoningMessageEndEvent,
    ReasoningMessageStartEvent,
    ReasoningStartEvent,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from ag_ui.encoder import EventEncoder
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from lightspeed_agentic.tools import DEFAULT_ALLOWED_TOOLS
from lightspeed_agentic.types import AgentProvider, ProviderEvent, ProviderQueryOptions

logger = logging.getLogger("lightspeed_agentic")

_MAX_CONVERSATIONS = 100
_CONVERSATION_TTL_S = 3600
_MAX_HISTORY_MESSAGES = 20

_FenceMarker = tuple[Literal["text"], str] | tuple[Literal["ui"], str, dict[str, Any]]


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
    now = time.time()
    expired = [
        k for k, v in _conversations.items() if now - v.last_accessed_at > _CONVERSATION_TTL_S
    ]
    for k in expired:
        del _conversations[k]
    if len(_conversations) > _MAX_CONVERSATIONS:
        by_access = sorted(_conversations.items(), key=lambda kv: kv[1].last_accessed_at)
        for k, _ in by_access[: len(_conversations) - _MAX_CONVERSATIONS]:
            del _conversations[k]


def _get_or_create(conversation_id: str | None) -> _Conversation:
    _cleanup()
    if conversation_id and conversation_id in _conversations:
        conv = _conversations[conversation_id]
        conv.last_accessed_at = time.time()
        return conv
    conv = _Conversation(id=str(uuid.uuid4()), created_at=time.time(), last_accessed_at=time.time())
    _conversations[conv.id] = conv
    return conv


class ChatRequest(BaseModel, extra="allow"):
    message: str
    conversationId: str | None = None
    context: dict[str, Any]


_FENCE_OPEN = re.compile(r"```ui:(\w+)\n")
_FENCE_PARTIAL = re.compile(r"`{1,3}(?:u(?:i(?::(?:\w+)?)?)?)?$")


class _FenceBuffer:
    """Buffers streamed model text; extracts ui:type fenced blocks into structured markers."""

    def __init__(self) -> None:
        self.buf = ""
        self.in_fence = False
        self.fence_type = ""
        self._pending: list[_FenceMarker] = []

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
                self.buf = self.buf[close + 3 :]
                self.in_fence = False
                try:
                    props = json.loads(json_str)
                    self._pending.append(("ui", self.fence_type, props))
                except (json.JSONDecodeError, TypeError):
                    pass
                self.fence_type = ""
            else:
                m = _FENCE_OPEN.search(self.buf)
                if m:
                    before = self.buf[: m.start()]
                    if before:
                        self._pending.append(("text", before))
                    self.fence_type = m.group(1)
                    self.buf = self.buf[m.end() :]
                    self.in_fence = True
                else:
                    partial = _FENCE_PARTIAL.search(self.buf)
                    safe = len(self.buf) - len(partial.group(0)) if partial else len(self.buf)
                    if safe > 0:
                        self._pending.append(("text", self.buf[:safe]))
                        self.buf = self.buf[safe:]
                    break

    def flush(self) -> None:
        if self.in_fence:
            self._pending.append(("text", "```ui:" + self.fence_type + "\n" + self.buf))
        elif self.buf:
            self._pending.append(("text", self.buf))
        self.buf = ""
        self.in_fence = False
        self.fence_type = ""

    def drain(self) -> list[_FenceMarker]:
        events = self._pending
        self._pending = []
        return events


@dataclass
class _AgUiStreamState:
    encoder: EventEncoder
    thread_id: str
    run_id: str
    current_text_message_id: str | None = None
    last_tool_call_id: str | None = None
    reasoning_message_id: str | None = None
    reasoning_message_open: bool = False
    reasoning_phase_open: bool = False

    def encode_run_started(self) -> str:
        return self.encoder.encode(
            RunStartedEvent(
                type=EventType.RUN_STARTED,
                thread_id=self.thread_id,
                run_id=self.run_id,
            )
        )

    def encode_run_finished(self) -> str:
        return self.encoder.encode(
            RunFinishedEvent(
                type=EventType.RUN_FINISHED,
                thread_id=self.thread_id,
                run_id=self.run_id,
                result=None,
            )
        )

    def encode_run_error(self, message: str) -> str:
        return self.encoder.encode(RunErrorEvent(type=EventType.RUN_ERROR, message=message))

    def end_text_if_open(self) -> Iterator[str]:
        if self.current_text_message_id is None:
            return
        mid = self.current_text_message_id
        self.current_text_message_id = None
        yield self.encoder.encode(
            TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id=mid)
        )

    def close_reasoning(self) -> Iterator[str]:
        if self.reasoning_message_open and self.reasoning_message_id is not None:
            rid = self.reasoning_message_id
            yield self.encoder.encode(
                ReasoningMessageEndEvent(
                    type=EventType.REASONING_MESSAGE_END,
                    message_id=rid,
                )
            )
            self.reasoning_message_open = False
        if self.reasoning_phase_open and self.reasoning_message_id is not None:
            rid = self.reasoning_message_id
            yield self.encoder.encode(
                ReasoningEndEvent(type=EventType.REASONING_END, message_id=rid)
            )
            self.reasoning_phase_open = False
        self.reasoning_message_id = None

    def emit_thinking_delta(self, delta: str) -> Iterator[str]:
        if not self.reasoning_phase_open:
            rid = str(uuid.uuid4())
            self.reasoning_message_id = rid
            yield self.encoder.encode(
                ReasoningStartEvent(type=EventType.REASONING_START, message_id=rid)
            )
            yield self.encoder.encode(
                ReasoningMessageStartEvent(
                    type=EventType.REASONING_MESSAGE_START,
                    message_id=rid,
                    role="reasoning",
                )
            )
            self.reasoning_phase_open = True
            self.reasoning_message_open = True
        if self.reasoning_message_id is None:
            raise RuntimeError("reasoning_message_id not set for thinking delta")
        rid = self.reasoning_message_id
        yield self.encoder.encode(
            ReasoningMessageContentEvent(
                type=EventType.REASONING_MESSAGE_CONTENT,
                message_id=rid,
                delta=delta,
            )
        )

    def encode_fence_markers(self, markers: list[_FenceMarker]) -> Iterator[str]:
        for marker in markers:
            if marker[0] == "text":
                chunk = marker[1]
                if not chunk:
                    continue
                yield from self._emit_text_delta(chunk)
            else:
                ui_type = marker[1]
                props = marker[2]
                yield from self.end_text_if_open()
                yield self.encoder.encode(
                    CustomEvent(
                        type=EventType.CUSTOM,
                        name="ui_component",
                        value={"type": ui_type, "props": props},
                    )
                )

    def _emit_text_delta(self, chunk: str) -> Iterator[str]:
        if not chunk:
            return
        if self.current_text_message_id is None:
            mid = str(uuid.uuid4())
            self.current_text_message_id = mid
            yield self.encoder.encode(
                TextMessageStartEvent(
                    type=EventType.TEXT_MESSAGE_START,
                    message_id=mid,
                    role="assistant",
                )
            )
        else:
            mid = self.current_text_message_id
        yield self.encoder.encode(
            TextMessageContentEvent(
                type=EventType.TEXT_MESSAGE_CONTENT,
                message_id=mid,
                delta=chunk,
            )
        )

    def emit_tool_call(self, name: str, input_payload: str) -> Iterator[str]:
        tcid = str(uuid.uuid4())
        self.last_tool_call_id = tcid
        yield self.encoder.encode(
            ToolCallStartEvent(
                type=EventType.TOOL_CALL_START,
                tool_call_id=tcid,
                tool_call_name=name,
            )
        )
        yield self.encoder.encode(
            ToolCallArgsEvent(
                type=EventType.TOOL_CALL_ARGS,
                tool_call_id=tcid,
                delta=input_payload,
            )
        )
        yield self.encoder.encode(ToolCallEndEvent(type=EventType.TOOL_CALL_END, tool_call_id=tcid))

    def emit_tool_result(self, output: str) -> Iterator[str]:
        tcid = self.last_tool_call_id or str(uuid.uuid4())
        msg_id = str(uuid.uuid4())
        yield self.encoder.encode(
            ToolCallResultEvent(
                type=EventType.TOOL_CALL_RESULT,
                message_id=msg_id,
                tool_call_id=tcid,
                content=output,
                role="tool",
            )
        )


def _closes_reasoning_on_non_thinking_event(event: ProviderEvent) -> bool:
    return event.type in ("text_delta", "tool_call", "tool_result", "result")


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
            if len(conversation.messages) > _MAX_HISTORY_MESSAGES * 2:
                conversation.messages = conversation.messages[-_MAX_HISTORY_MESSAGES * 2 :]

            run_id = str(uuid.uuid4())
            thread_id = conversation.id
            state = _AgUiStreamState(encoder=EventEncoder(), thread_id=thread_id, run_id=run_id)

            yield state.encode_run_started()

            agent_text = ""
            total_cost = 0.0
            fence = _FenceBuffer()
            stream_failed = False

            try:
                agen = provider.query(
                    ProviderQueryOptions(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        model=model,
                        max_turns=max_turns,
                        max_budget_usd=max_budget_usd,
                        allowed_tools=DEFAULT_ALLOWED_TOOLS,
                        cwd=skills_dir,
                        stream=True,
                    )
                )
                async for event in agen:
                    if await request.is_disconnected():
                        break

                    if _closes_reasoning_on_non_thinking_event(event):
                        for chunk in state.close_reasoning():
                            yield chunk

                    if event.type in ("tool_call", "tool_result", "result"):
                        for chunk in state.end_text_if_open():
                            yield chunk

                    if event.type == "thinking_delta":
                        for chunk in state.end_text_if_open():
                            yield chunk

                    match event.type:
                        case "text_delta":
                            fence.add(event.text)
                        case "thinking_delta":
                            for chunk in state.emit_thinking_delta(event.thinking):
                                yield chunk
                        case "content_block_stop":
                            fence.flush()
                            for chunk in state.encode_fence_markers(fence.drain()):
                                yield chunk
                            for chunk in state.end_text_if_open():
                                yield chunk
                            for chunk in state.close_reasoning():
                                yield chunk
                        case "tool_call":
                            for chunk in state.emit_tool_call(event.name, event.input):
                                yield chunk
                        case "tool_result":
                            for chunk in state.emit_tool_result(event.output):
                                yield chunk
                        case "result":
                            total_cost = event.cost_usd
                            agent_text = event.text

                    for chunk in state.encode_fence_markers(fence.drain()):
                        yield chunk

            except TimeoutError:
                stream_failed = True
                yield state.encode_run_error(f"Chat timed out after {timeout_ms}ms")
            except Exception as e:
                stream_failed = True
                logger.exception("[agent] Chat error")
                yield state.encode_run_error(str(e))

            if stream_failed:
                return

            fence.flush()
            for chunk in state.encode_fence_markers(fence.drain()):
                yield chunk
            for chunk in state.end_text_if_open():
                yield chunk
            for chunk in state.close_reasoning():
                yield chunk

            if agent_text:
                clean_text = re.sub(r"```ui:\w+\n[\s\S]*?```", "", agent_text).strip()
                conversation.messages.append(
                    _ConversationEntry(role="assistant", content=clean_text)
                )

            yield state.encode_run_finished()
            logger.info(
                "[agent] Chat complete: conversation=%s, cost=$%.4f", conversation.id, total_cost
            )

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
