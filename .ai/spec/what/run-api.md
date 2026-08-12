# Behavioral spec: Run API

Audience: AI agents (Claude). Precision over narrative.

Cross-references: provider behavior and events → `provider-contract.md`. Env defaults and ports → `configuration.md`.

> **[OLS-3066] Batch execution model planned.** The HTTP API (rules 1–24 below) is the current implementation. OLS-3066 replaces it with a batch entrypoint: the sandbox reads input from a ConfigMap volume mount, runs the agent, creates a Result CR via `oc`, and exits. See the "Batch Entrypoint" section at the end of this file for the target behavior. The HTTP rules below remain authoritative until OLS-3066 is implemented.

## Behavioral Rules

1. **Operator integration boundary.** The Kubernetes operator (workflow engine) invokes the sandbox over HTTP using `POST /v1/agent/run` with a JSON body matching `RunRequest`. The sandbox returns `RunResponse` JSON. The operator carries step **input** via `query`, structured-output hints via `outputSchema`, and runtime envelope via `context`. [PLANNED: OLS-3491] Step **system instructions** are carried via `systemPrompt` (non-empty after operator materialization). Until OLS-3491, the operator may still send `systemPrompt` as empty and embed role text in `query`; the sandbox applies a default persona when `systemPrompt` is empty or omitted (see rule 5). The sandbox does not interpret workflow phase names.

2. **Route mounting.** Agent routes are mounted under the path prefix `/v1/agent` on the FastAPI application. Probe routes (`/health`, `/ready`) are **not** under that prefix.

3. **Canonical run endpoint.** `POST /v1/agent/run` accepts `RunRequest` and returns `RunResponse`.

4. **RunRequest — `query` (required).** Step input / user task text (not system role instructions after OLS-3491). When `context` is present, the handler prepends a formatted context block to this text before sending the combined string to the provider (see rules 12–16).

5. **RunRequest — `systemPrompt`.** Optional. When omitted, null, or empty, the handler substitutes a fixed default assistant persona string. [PLANNED: OLS-3491] When the operator sends non-empty `systemPrompt`, the handler MUST use it as-is (full replacement of the default persona). The sandbox MUST NOT append the default persona to a non-empty caller `systemPrompt`.

6. **RunRequest — `outputSchema`.** Optional JSON-object schema. When present, forwarded to the provider as structured-output hints (see `provider-contract.md`). The HTTP response still follows `RunResponse` shaping rules (rules 18–22).

7. **RunRequest — `context`.** Optional object. When present, must be formatted by the rules in 12–16; unknown keys are ignored if not read by the formatter.

8. **RunRequest — `timeout_ms`.** Optional. When set, caps wall-clock time for consuming the provider event stream until the first `result` event. When omitted, a router-level default timeout applies (see `configuration.md`).

9. **Per-run spend ceiling.** The route passes a fixed USD budget cap into provider options. This cap is **not** configurable via `RunRequest`.

10. **GET /health.** Returns a JSON object `{ "status": "ok" }` when the process is up (not mounted under `/v1/agent`). _(Authoritative definitions are in `what/health-probes.md`. These rules provide a summary; for full liveness/readiness probe semantics, see that file.)_

11. **GET /ready.** Readiness probe (not under `/v1/agent`). Returns HTTP 200 with `{ "status": "ok" }` when all checks pass; HTTP 503 with `{ "status": "error", "checks": { ... } }` when any check fails. Checks and semantics: `health-probes.md`. _(Authoritative definitions are in `what/health-probes.md`. These rules provide a summary; for full liveness/readiness probe semantics, see that file.)_

12. **Context prefix — envelope.** When `context` is non-empty, the formatter produces a block that starts with a fixed marker line, ends with a closing marker line, and is prepended to `query` with separating newlines.

13. **Context — `targetNamespaces`.** When present and non-empty (list), include a line listing target namespaces as a comma-separated join.

14. **Context — `attempt`.** When present (any), include a line labeling the attempt with placeholder text for the maximum (literal substring `of max` in the line; the formatter does not inject the max value).

15. **Context — `previousAttempts`.** When present and non-empty (iterable of objects), include a header line then one bullet line per entry with attempt index and optional `failureReason`.

16. **Context — `approvedOption`.** When present and non-empty (object), append a bounded block: title, `diagnosis.rootCause`, and from `approvedOption.remediationPlan` the `description`, `risk`, `reversible` flag, and optional `actions` list (each with `command`, `type`, and `description`); surround with explicit “approved remediation” and “do not exceed listed actions” banners. Each action's `command` field contains the exact bash command (kubectl/oc) to execute.

17. **Stream consumption.** The handler iterates the provider async iterator until a `result` event; earlier events are logged but do not terminate the request. See `provider-contract.md` for event types.

18. **RunResponse — core fields.** Every response includes `success` (boolean) and `summary` (string). Additional keys are allowed on the response object.

19. **Structured agent output.** When the final `result` text is JSON parsing as an object, the handler builds `RunResponse` with `success` from that object’s `success` key defaulting to true when absent, `summary` from `summary` defaulting to the raw result text when absent, and merges remaining keys as extra top-level fields.

20. **Text fallback.** When the final `result` text is not a JSON object (parse failure or non-object JSON), the handler returns `success=true` and `summary` equal to the full result text with no extra keys from parsing.

21. **Timeout.** When waiting for the provider exceeds the effective timeout, the handler returns `success=false` and a summary string that states timeout and includes the timeout duration in milliseconds.

22. **Agent errors.** On any other exception during the provider call, the handler returns `success=false` and a summary prefixed with a fixed agent-error label and the exception message.

23. **Empty result.** When the stream ends without non-empty final `result` text, the handler returns `success=false` with a fixed empty-response summary.

24. **Allowed tools.** The route passes the default allowed-tools list into provider options; callers cannot override via `RunRequest` (see `provider-contract.md`).

## Configuration Surface

| Mechanism | Purpose |
|-----------|---------|
| `RunRequest.timeout_ms` | Per-request wall-clock limit for waiting on the first `result` event (milliseconds). |
| Router `default_timeout_ms` | Used when `timeout_ms` is omitted (see `configuration.md`). |
| `LIGHTSPEED_SKILLS_DIR` | Working directory / skill root forwarded as provider `cwd` (see `configuration.md`). |

## Constraints

- The handler does not expose `max_turns`, model id, provider id, or tool allowlists on `RunRequest`; those are fixed or environment-driven per `configuration.md` and router construction.
- Streaming to the HTTP client is out of scope for `POST /run`; provider streaming may be used internally only if the adapter enables it (see `how/provider-architecture.md`).

## Planned Changes

- Operator payload may later include `llm` and `allowedTools` per target architecture docs; sandbox route does not read them today. [PLANNED: OLS-3033]
- ~~TLS, network policy, and ingress hardening for the sandbox service. [PLANNED: OLS-3038–OLS-3043]~~ No longer applicable — OLS-3066 removes the HTTP server.
- [PLANNED: OLS-3066] **Batch execution model** replaces the HTTP API. See "Batch Entrypoint" section below.

## Verification

Harness scope (live vs unit, run modes, flake policy):
[e2e-testing.md](e2e-testing.md).

Two layers:

1. **Unit tests** (`tests/test_routes.py`) — mocked provider, deterministic handler
   behavior (timeouts, empty result, response shaping). Preferred for rules 21 and 23.
2. **Container BDD** (`tests/e2e/features/`, `scripts/e2e-containers.sh`) — live
   `/v1/agent/run` against one sandbox container per process with real credentials.

Rules **10–11** (`/health`, `/ready`) are verified under `health-probes.md`, not here.

| Artifact | Rules exercised | Notes |
|----------|-----------------|-------|
| [structured_output.feature](../../../tests/e2e/features/structured_output.feature) | 3, 6, 18–20 | Live structured output and text fallback; adversarial schema stays HTTP 200 with envelope (rule 22 not triggered) |
| [skills.feature](../../../tests/e2e/features/skills.feature) | 3, 18–20 | `/run` success paths with skills mounted (see `provider-contract.md`) |
| [test_routes.py](../../../tests/test_routes.py) | 3, 5, 6, 8, 18–21, 23 | Mocked provider: `systemPrompt`, `outputSchema`, `timeout_ms`, timeout failure, empty result, text fallback |
| [sandbox_e2e.feature](../../../tests/e2e/features/sandbox_e2e.feature) (Context prefix) | 4, 7, 12–16 | Live **targetNamespaces**, **previousAttempts**, and **approvedOption** echo via structured output; exact prefix strings in [test_routes.py](../../../tests/test_routes.py) |
| [sandbox_e2e.feature](../../../tests/e2e/features/sandbox_e2e.feature) (Run error handling) | 21 | Live **timeout** only (`timeout_ms=1` → HTTP 200, `success=false`, timed-out summary). Rules 22–23 and no-500 adversarial path: `test_routes.py`, `structured_output.feature` |

---

## [PLANNED: OLS-3066] Batch Entrypoint

Replaces the HTTP API (rules 1–24) with a batch execution model. The sandbox runs as a one-shot process: read input, run agent, write output, exit.

### Behavioral Rules (Batch)

B1. **No HTTP server.** The sandbox MUST NOT start a FastAPI/HTTP server. There are no routes, no probes (`/health`, `/ready`), no inbound connections. The process reads files, runs the agent, writes results, and exits.

B2. **Input files.** The operator mounts a ConfigMap at `/input/` (read-only) with keys:
- `/input/query` — step input text (same content as the former `RunRequest.query`; after OLS-3491 this MUST NOT embed role/system instructions)
- `/input/system-prompt` — [PLANNED: OLS-3491] optional step system instructions (same content as former `RunRequest.systemPrompt`). When absent or empty, the sandbox uses the fixed default persona (same as HTTP rule 5). **Absence of this file is valid input and MUST NOT be treated as a sandbox input-read failure** (contrast rule B6, which applies to required inputs that cannot be read).
- `/input/output-schema` — JSON schema for structured output (same as former `RunRequest.outputSchema`)
- `/input/context` — JSON object with `targetNamespaces`, `previousAttempts`, `approvedOption`, `executionResult` (same structure as former `RunRequest.context`)
- `/input/result-template` — pre-filled Result CR JSON with `apiVersion`, `kind`, `metadata` (name, namespace, labels, ownerReferences), and `spec` (agenticRunName, retryIndex). The sandbox fills in `status` only.

B3. **Agent execution.** The sandbox reads the input files, initializes the LLM provider (same provider adapters as today — unchanged), and runs the agent with the system prompt, query, output schema, and context. Tool execution (kubectl, oc) is unchanged. Skills and MCP servers are configured via environment variables and volume mounts (unchanged).

B4. **Output — success path.** On successful agent completion, the sandbox MUST: (a) merge the agent's structured JSON output into the Result CR's `status` fields (options, diagnosis, actionRequired, actionsTaken, checks, summary — varies by step type), (b) set `status.conditions` to include `Started=True` and `Completed=True`, (c) run `oc create -f <result.json>` to create the CR (metadata + spec from template), (d) run `oc patch <resultCR> --type=merge --subresource=status -p '<status-json>'` to set the status, (e) exit 0.

B5. **Output — agent failure path.** When the agent returns `success=false` or throws an exception during execution, the sandbox MUST still create the Result CR: set `status.failureReason` with the error message, set `status.conditions` to include `Completed=True`, and create+patch via `oc` as in rule B4. Exit 0 (the sandbox succeeded; the agent failed).

B6. **Output — sandbox failure path.** When the sandbox cannot read input files, `oc create` fails, or any other infrastructure error occurs, the sandbox MUST write a human-readable error message to `/dev/termination-log` (max 4096 bytes) and exit non-zero. The operator reads the termination message from `pod.status.containerStatuses[0].state.terminated.message`.

B7. **Context formatting.** The batch entrypoint MUST apply the same context prefix formatting as the current HTTP handler (rules 12–16 above): prepend targetNamespaces, attempt info, previousAttempts, and approvedOption blocks to the query text before sending to the provider.

B8. **Provider selection and configuration.** Provider selection (`LIGHTSPEED_PROVIDER`, `LIGHTSPEED_MODEL`, etc.), credential loading, skills directory, MCP servers, and reasoning config are unchanged — all configured via environment variables and volume mounts on the pod spec, not via the input ConfigMap.

B9. **RBAC requirements.** The sandbox ServiceAccount MUST have `create` and `patch` (with `status` subresource) permissions on `AnalysisResult`, `ExecutionResult`, `VerificationResult`, and `EscalationResult` resources in the AgenticRun namespace. `oc` authenticates using the auto-mounted SA token.

### What Changes vs HTTP

| Component | HTTP (current) | Batch (OLS-3066) |
|---|---|---|
| FastAPI server, routes, probes | Required | **Removed** |
| LLM provider adapters | Unchanged | Unchanged |
| Structured output / schema | Unchanged | Unchanged |
| Tool execution (kubectl, oc) | Unchanged | Unchanged |
| Skills, MCP servers | Unchanged | Unchanged |
| Input source | HTTP request body | `/input/` ConfigMap mount |
| Output destination | HTTP response body | Result CR via `oc create` + `oc patch --subresource=status` |
| Error reporting | HTTP response with `success=false` | Result CR with `failureReason` (agent errors) or `/dev/termination-log` (sandbox errors) |
| Dependencies added | — | Zero — `oc` is already in the image |
