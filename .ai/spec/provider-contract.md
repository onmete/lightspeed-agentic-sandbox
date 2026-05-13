# Provider Contract

Behavioral rules that every provider adapter must honor.

## Origin

This codebase is a Python port of the TypeScript
`lightspeed-agent`. The two stacks expose the same HTTP
interface and the same event semantics. When in doubt, match
the TypeScript behavior.

## Events

A provider emits a stream of events with six possible types:

| Type | Meaning |
|---|---|
| text_delta | Incremental text from the model |
| thinking_delta | Incremental reasoning trace (optional — not all providers support it) |
| content_block_stop | Signals the end of a logical content block |
| tool_call | The model invoked a tool (name + input) |
| tool_result | Output returned from the tool |
| result | Final aggregated text, cost, and token counts |

**Ordering rule for query routes:** the handler consumes
events until it sees the first `result` and then stops.
Everything after that first `result` is ignored. Providers
must guarantee that `result` carries the complete final
answer.

**Ordering rule for chat:** events are streamed to the
client as they arrive. `content_block_stop` triggers a fence
flush (see chat-api.md). `result` captures the final text
for conversation history.

## Thin-adapter principle

Providers are mapping layers — they configure the SDK and
translate SDK-specific stream events into the common event
types above. They must not:

- Re-implement tool execution that the SDK already provides.
- Parse or load skill files manually — the SDK's native
  skill integration handles discovery and invocation.
- Add business logic (timeouts, budgets, prompt assembly) —
  that belongs in the route layer.

Shared path logic (default tool lists) lives in a central
utility module. Skills directory resolution is used by
Gemini, OpenAI, and Deep Agents to find skill files within
the mounted directory. Claude does not use this resolution —
it passes the raw skills directory as its working directory
and relies on the SDK's native `skills="all"` discovery.

## Structured output

When the caller requests structured output, each provider
uses its SDK's native mechanism. The approaches differ:

- **Gemini** disables the explicit exit-loop tool when an
  output schema is present and relies on the SDK's native
  response-schema path instead. This is a deliberate
  workaround — the ADK's built-in structured-output tool
  was broken at the time of implementation.
- **OpenAI** wraps the schema with strict-JSON disabled to
  maintain compatibility with vLLM-compatible endpoints.
- **Deep Agents** dynamically generates a Pydantic model from
  the JSON schema. Nested objects are handled recursively;
  arrays are typed loosely.

These per-provider differences are expected. The contract is:
if the caller sends an output schema, the provider must
attempt to make the model conform. Failure to conform is a
model-level issue, not a provider bug.

## Cost reporting

Only the Claude adapter returns meaningful cost data.
Gemini, OpenAI, and Deep Agents report zero cost in their
result events. This is a known gap — cost tracking depends
on SDK support and is not a provider-level requirement yet.

## What not to add to a provider

- Custom tool executors — use the SDK's native tools.
- Retry or fallback logic — that belongs upstream.
- Request validation — the route layer validates before
  calling the provider.

## Verification

BDD feature files that exercise this spec:

| Feature file | Scenarios |
|---|---|
| `tests/e2e/features/structured_output.feature` | Run with flat schema and required fields; Run with nested schema; Adversarial schema does not return HTTP 500 |

Coverage mapping:
- **Structured output** (each provider uses its native mechanism) →
  "Run with flat schema", "Run with nested schema"
- **Graceful degradation on schema failure** →
  "Adversarial schema does not return HTTP 500"
- **Events** → not yet covered by E2E (unit-tested only)
- **Cost reporting** → not yet covered
