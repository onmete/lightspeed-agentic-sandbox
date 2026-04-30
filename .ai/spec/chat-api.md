# Chat API

Behavioral rules for the `/chat` SSE endpoint.

## Purpose

The chat endpoint provides a conversational interface for
human operators to discuss an alert remediation with the
agent. It is separate from the query endpoints: query is
operator-driven and structured; chat is human-driven and
free-form.

## Read-only invariant

The chat system prompt explicitly instructs the agent to
**not modify any cluster resources**. This is a security
invariant — the chat is for investigation and discussion,
never for execution. The same tools are available, but the
system prompt restricts their use to read-only operations.

## SSE protocol (AG-UI)

The endpoint returns a `text/event-stream` response. Each SSE
message is a `data:` line carrying one JSON object. The wire
format follows the [Agent User Interaction (AG-UI)
protocol](https://docs.ag-ui.com/): the JSON includes a
`type` field that discriminates events (see `EventType` in
the AG-UI Python SDK). Encoding uses `ag_ui.encoder.EventEncoder`.

| AG-UI event type | When |
|---|---|
| `RUN_STARTED` | First event. Carries `threadId` (conversation) and `runId` (this request). |
| `TEXT_MESSAGE_START` / `TEXT_MESSAGE_CONTENT` / `TEXT_MESSAGE_END` | Assistant text streaming. Start includes `messageId` and `role: "assistant"`. Content carries `delta` (non-empty). End closes the message. |
| `TOOL_CALL_START` / `TOOL_CALL_ARGS` / `TOOL_CALL_END` | Tool invocation. `toolCallId` and `toolCallName` on start; args stream as `delta` on the args event. |
| `TOOL_CALL_RESULT` | Tool output. `toolCallId` matches the preceding tool call; includes `messageId` and `role: "tool"`. |
| `REASONING_START` / `REASONING_MESSAGE_START` / `REASONING_MESSAGE_CONTENT` / `REASONING_MESSAGE_END` / `REASONING_END` | Model reasoning trace (optional — provider-dependent). |
| `CUSTOM` | Extensible events. UI fence payloads use `name: "ui_component"` and `value: { type, props }`. |
| `RUN_FINISHED` | Terminal event on success. Carries `threadId` and `runId`. |
| `RUN_ERROR` | Terminal event on error. Carries `message` (and optional `code`). No `RUN_FINISHED` after an error. |

The request body still uses `conversationId` (UUID) to
continue a thread. That value is the AG-UI `threadId` for
`RUN_STARTED` and `RUN_FINISHED`.

## UI fence protocol

The model can emit structured UI components inline in its
text output using a fenced block syntax:

    ```ui:<type>
    {"key": "value"}
    ```

When the fence parser encounters this pattern:

1. Text before the fence is emitted as `TEXT_MESSAGE_*` events.
2. The JSON body is parsed. If valid, a `CUSTOM` event is
   emitted with `name: "ui_component"` and
   `value: { type, props }` (type is the fence name, props is
   the parsed object).
3. If the JSON is malformed, the content is **silently
   dropped** — no error event, no fallback text.
4. If the fence is never closed (model stops mid-fence), the
   partial content is flushed as plain text (via
   `TEXT_MESSAGE_*`) so the client can render something.

The fence parser also guards against streaming partial fence
markers (e.g., a lone backtick at the end of a chunk) — these
are held back until enough characters arrive to determine
whether a fence is opening.

## Conversation lifecycle

Conversations are stored **in memory** (not persisted). They
are identified by UUID and subject to:

| Limit | Value |
|---|---|
| Max concurrent conversations | 100 |
| TTL since last access | 3600 seconds (1 hour) |
| History window in prompt | last 20 messages |
| Stored message cap | 40 entries (oldest trimmed) |

Cleanup runs on every new request: expired conversations are
removed first, then LRU eviction if over capacity.

If the client sends no conversation ID or an unknown one, a
new conversation is created. Sending a known ID continues
that conversation and refreshes its TTL.

## History hygiene

The conversation history stores **clean text** — UI fences
are stripped from the assistant's response before it is
appended to history. The frontend receives `CUSTOM` events
with `name: "ui_component"` during streaming, but the history
retains only readable prose. This avoids re-parsing fences
on subsequent turns and keeps the prompt token-efficient.

## Budget and turns

Chat uses a separate budget from query routes: default $1.00
per message and 30 max turns. These are lower because chat
is interactive — individual messages should be cheaper and
faster than autonomous query runs.
