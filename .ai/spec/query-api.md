# Query API

Behavioral rules for the `/analyze`, `/execute`, and
`/verify` endpoints.

## One handler, three labels

All three endpoints run the same handler. The only
differences are the phase label (used for logging and
timeout selection) and the URL path. The operator controls
what the agent does through the request body — system prompt,
output schema, and context — not through endpoint choice.

A future consolidation into a single endpoint with a phase
field in the request body is anticipated.

## Timeout structure

| Phase | Timeout source |
|---|---|
| analysis | analysis timeout (default 5 min) |
| execution | execution timeout (default 10 min) |
| verification | **analysis** timeout, not execution |

Verification reuses the analysis timeout because it is a
check, not a long-running action. This is intentional.

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

The approved option block includes the diagnosis, plan, risk
assessment, and a specific action list. The closing line
reinforces the constraint: perform nothing beyond what is
listed.

## Streaming

Query routes do not use streaming — the provider runs to
completion and the handler waits for the final result event.
Streaming is reserved for the chat endpoint.

## Verification

`tests/e2e/features/schema_compliance.feature` — schema
conformance for all three query endpoints, response wrapping
behavior.
