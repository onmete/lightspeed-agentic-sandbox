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

## SSE protocol

The endpoint returns a `text/event-stream` response. Each
SSE message has an `event` name and a JSON `data` payload.

| Event | Payload | When |
|---|---|---|
| status | `{status}` | Stream start ("thinking") |
| text | `{content}` | Incremental text from the model |
| thinking | `{content}` | Model reasoning trace |
| tool_call | `{name, input}` | Model invoked a tool |
| tool_result | `{output}` | Tool returned output |
| ui_component | `{type, props}` | Parsed UI fence (see below) |
| error | `{message}` | Timeout or unhandled error |
| done | `{conversationId}` | Always the final event |

The `done` event always includes the conversation ID. The
frontend needs this to continue the conversation in
subsequent requests.

## UI fence protocol

The model can emit structured UI components inline in its
text output using a fenced block syntax:

    ```ui:<type>
    {"key": "value"}
    ```

When the fence parser encounters this pattern:

1. Text before the fence is emitted as a normal `text` event.
2. The JSON body is parsed. If valid, a `ui_component` event
   is emitted with `{type, props}`.
3. If the JSON is malformed, the content is **silently
   dropped** — no error event, no fallback text.
4. If the fence is never closed (model stops mid-fence), the
   partial content is flushed as plain `text` so the client
   can render something.

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
appended to history. The frontend receives structured
`ui_component` events during streaming, but the history
retains only readable prose. This avoids re-parsing fences
on subsequent turns and keeps the prompt token-efficient.

## Budget and turns

Chat uses a separate budget from query routes: default $1.00
per message and 30 max turns. These are lower because chat
is interactive — individual messages should be cheaper and
faster than autonomous query runs.
