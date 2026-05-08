# Query API

Behavioral rules for the `POST /run` endpoint.

## Single endpoint

The operator controls what the agent does through the request
body — system prompt, output schema, context, and timeout —
not through endpoint choice.

## Timeout

The caller passes `timeout_ms` per request. If omitted, the
server default applies (300s). This lets the operator set
different timeouts for analysis vs execution vs verification
without needing separate endpoints.

## Budget

Per-query budget is hardcoded at $5.00 USD. It is not
configurable via environment variable or request parameter.
This is a deliberate ceiling — the operator should not be
able to accidentally authorize expensive runs.

## Response parsing

The handler parses the model's final text output:

1. If valid JSON **and** a dictionary: extract `success` and
   `summary` from it, spread remaining keys as extra fields
   into the response. Missing `success` defaults to `true`.
2. If not JSON or not a dictionary: treat the entire text as
   a successful result with the text as the summary.

This means a model that returns plain English is always
treated as success. The operator's output schema steers the
model toward structured output, but the handler degrades
gracefully.

## Context prefix

When the request includes context, it is formatted as a
`[context]...[/context]` block and **prepended to the user
prompt** (not to the system prompt). The block includes:

- Target namespaces
- Attempt number and previous attempt outcomes
- Approved remediation details (when present)

## Approved remediation constraint

When `approvedOption` is present in the context, the agent
receives an explicit instruction block listing the approved
actions and a directive to execute **only** those actions.
This is a **prompt-level security boundary**: the agent is
told not to deviate, but there is no server-side enforcement
mechanism beyond the instruction itself.

## Streaming

Query routes do not use streaming — the provider runs to
completion and the handler waits for the final result event.
