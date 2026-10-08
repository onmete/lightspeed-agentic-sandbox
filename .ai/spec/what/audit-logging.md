# Audit Logging

Implementation spec for compliance logging and the sandbox PR1 trace profile. Parent spec: `ols/.ai/spec/what/audit-logging.md` remains authoritative for cross-repository audit/logging and correlation requirements; the sandbox producer exception is scoped in `data-collection.md`, without changing the parent contract.

Telemetry follows the named [OTel GenAI profile pinned at commit `4f85037ef86e92c510d2ef881a58f1076f6fc0e4`](https://github.com/open-telemetry/semantic-conventions-genai/tree/4f85037ef86e92c510d2ef881a58f1076f6fc0e4/docs/gen-ai). Upstream status at that revision is Development; this does not claim every optional convention or add a dependency.

## Behavioral Rules

### Span Naming and Kinds

1. The sandbox MUST create an `invoke_agent` span with `gen_ai.operation.name="invoke_agent"` and `SpanKind.INTERNAL`, as a child of the received operator context. It carries the configured request model and is not a provider inference span: it MUST NOT claim `gen_ai.provider.name` or `gen_ai.response.model`.

2. The sandbox MUST create an `execute_tool {gen_ai.tool.name}` span for each local tool execution. Tool spans are `INTERNAL` children of the invocation span and use only actual SDK tool-call IDs.

3. The shared invocation/tool spans are produced by `run_agent_query()` and `AuditLogger`. New sandbox-owned canonical per-generation spans are planned for later PRs, not PR1; existing SDK-native ADK `call_llm`/`generate_content` spans remain unchanged framework detail and outside the canonical profile until PR3's exported-view normalization (see `data-collection.md`).

### GenAI Attributes — Invocation Span

4. The `invoke_agent` span MUST carry the PR1 attributes below and retain native span context, start/end time, and OTel status.

| Attribute | Requirement | Description |
|---|---|---|
| `gen_ai.operation.name` | Required | `"invoke_agent"` |
| `gen_ai.request.model` | Required | Configured request model; not an actual response model |
| `gen_ai.input.messages` | Required | JSON message array containing the effective post-context-prefix user prompt |
| `gen_ai.system_instructions` | Required | JSON instruction array containing the configured supplied system string |
| `gen_ai.output.type` | Conditional | `"json"` iff `output_schema is not None` |
| `gen_ai.output.messages` | Conditional | Exact observed `ResultEvent.text` before parsing/shaping; empty is present, no `ResultEvent` is omitted |
| `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens` | Existing aggregate | Preserve the existing terminal `ResultEvent` usage behavior |
| `gen_ai.usage.reasoning.output_tokens` | Nonzero only | Existing aggregate reasoning count; do not write zero when absent/zero |
| `error.type` | On operational root error | `timeout`, exception class name (including `CancelledError`), or `empty_response`; cancellation propagates unchanged. Unsuccessful domain outcomes remain output content, not span errors. |
| `agenticrun.uid` / `agenticrun.phase` | When available | Existing run correlation values; do not invent missing values |

Message attributes are compact JSON strings matching the pinned schemas. The invocation captures the effective prompt and configured instructions supplied to the run, not all SDK-added instructions or repeated request history. The root output is a result projection, not another generation. Root usage remains aggregate and MUST NOT be summed with future per-generation usage.

### GenAI Attributes — Tool Span

6. Each local tool span MUST retain native span context, start/end time, and OTel status.

| Attribute | Requirement | Description |
|---|---|---|
| `gen_ai.operation.name` | Required | `"execute_tool"` |
| `gen_ai.tool.name` | Required | Tool name |
| `gen_ai.tool.call.id` | Optional | Actual SDK ID only; omit when unavailable |
| `gen_ai.tool.call.arguments` | When observed | JSON string encoding a JSON object |
| `gen_ai.tool.call.result` | When observed | JSON string encoding a JSON object |
| `agenticrun.uid` / `agenticrun.phase` | When available | Same existing correlation values as the invocation span |

Use strict JSON parsing for tool strings: accept standards-compliant finite JSON only; treat `NaN`, `Infinity`, `-Infinity`, exponent overflow, and parser/encoder-limit failures as non-JSON. Pass decoded dictionaries unchanged and wrap other decoded values as `{"content": value}`. On parse, encode, or limit rejection, preserve the complete original raw string in `{"content": raw_string}`; this is sandbox normalization, not a provider-native field. Never truncate it or raise a telemetry-only provider error. Preserve observed empty values and omit missing arguments/results. A missing-ID result may match only one pending call; ambiguous results stay unmatched. Unresolved calls end ERROR as `error.type="missing_tool_result"`; on cancellation, close pending tool spans without result or tool-duration histogram observation.

### Legacy Choice Events

7. Existing `gen_ai.choice` span events remain attached to the invocation span as a legacy audit projection. Preserve their existing text/reasoning mapping, emission/content gates, and buffer flush order. The buffered text is flushed before reasoning, so these events are not a canonical cross-type chronology. Cancellation does not force another choice/developer-log buffer flush or emit additional choice-derived templog records.

8. Do not add separate `audit.agent.started`/`audit.agent.completed` events or custom transcript events. Canonical invocation status, output, and usage follow the span-attribute profile above; provider identity and actual response model are not inferred for the root.

### Content Capture Policy

9. Whenever an invocation/tool span is recording, canonical PR1 attributes ignore `LIGHTSPEED_AUDIT_ENABLED` and `LIGHTSPEED_CAPTURE_CONTENT`. Existing `gen_ai.choice` event emission remains audit-gated; its text/reasoning payload retains the previous content-capture policy (unset defaults to content when audit is enabled; false or audit-disabled omits those payloads from compliance copies). Developer logs and the span-event → templog bridge retain their existing logging/endpoint/audit gates and flush order. None of these projections is a canonical transcript.

### Trace Context Reception

10. When audit or OTLP export is enabled (`otel_runtime_enabled()`), `batch.main()` calls `init_tracer()` before `run_agent_query()`. When the operator sets W3C `TRACEPARENT` on the pod, `batch.main()` passes it to `run_agent_query()` so `invoke_agent` is a child of the operator phase span.

11. If `TRACEPARENT` is unset or invalid, the sandbox MUST generate a new trace ID for the run (graceful degradation).

### Trace and Log Projections

12. The invocation and tool spans are emitted once through the shared TracerProvider; exporters/processors project the same spans. The legacy choice events remain a separate existing audit projection. PR1 adds no OTel Logs API path for canonical span attributes.
    - **OTLP span exporter** sends native spans and their attributes/events through the shared `OTEL_EXPORTER_OTLP_ENDPOINT` when set.
    - **Stdout exporter** serializes its trace projection as OTLP JSON when audit is enabled.
    - **Span-event → log processor** forwards existing audit events as OTLP templog records only when the endpoint is set and audit is enabled.

13. Python `logging` MUST emit developer-debugging messages and MUST NOT be used at AuditLogger call sites to re-record span/event data. When `OTEL_EXPORTER_OTLP_ENDPOINT` is set, stdlib logging is dual-shipped to stderr and OTLP (`LoggingHandler` on the root logger). The span-event → log bridge also emits through that same stdlib path so templog gets dual-ship without a separate OTel Logs API emit. This collapses into:
    - OTel spans and legacy events for audit (stdout + OTLP traces), with templog OTLP logs (and stderr) via the bridge → LoggingHandler.
    - Standard logging for developer debugging (stderr + OTLP when the endpoint is set).

### Structured Log Format

14. The stdout exporter MUST emit OTLP JSON — the OTel standard wire format. It is a projection of the same TracerProvider spans, including PR1 attributes when enabled, not a custom transcript format.

15. The stdout exporter MUST NOT truncate span attributes or event attributes. Full fidelity is preserved; downstream size limits remain best-effort collection constraints.

### Legacy Provider Event Projection

16. **DeepAgents / Anthropic** (`providers/deepagents.py`): Preserve existing `gen_ai.choice` text events from `AIMessage` and reasoning events from reasoning `content_blocks`; create tool spans from `AIMessage.tool_calls` and `ToolMessage` content. Existing aggregate usage remains on the invocation span through `ResultEvent`.

17. **OpenAI** (`providers/openai.py`): Preserve buffered `gen_ai.choice` text events from stream deltas and reasoning events when present; create tool spans from existing tool call/output items. Existing aggregate usage remains on the invocation span through `ResultEvent`.

18. **Gemini** (`providers/gemini.py`): Preserve buffered `gen_ai.choice` text events from text parts and reasoning events from thought parts when present; create tool spans from function-call/response parts. Existing aggregate usage remains on the invocation span through `ResultEvent`.

These legacy projections and their SDK/event behavior remain unchanged in PR1. Later PRs plan new sandbox-owned canonical per-generation capture; existing SDK-native ADK spans remain unchanged framework detail until PR3's exported-view normalization.

### Tool-Result Inspection [PLANNED: OLS-3928]

18a. Sandbox telemetry MUST conform to `openshift/ols/.ai/spec/what/tool-result-inspection.md`.

18b. Each DeepAgents inspection MUST create the contract's `tool_result.inspection` span and attach it to the parent agent trace.

18c. The sandbox can add controlled tool, provider, model, and tool-call correlation identifiers to the contract-defined attributes. Each inspection span MUST include `gen_ai.tool.call.id` when the inspected `ToolMessage` provides a non-empty call ID. This value MUST match the correlated `execute_tool` span.

18d. Existing generic inference instrumentation can observe classifier calls. The sandbox MUST add no feature-specific Prometheus metric.

18e. Developer logs and `tool_result.inspection` telemetry MUST NOT contain tool arguments, tool results, or tool-generated errors.

18f. After inspection passes, `AuditLogger` MUST retain the complete normalized result for the approved `gen_ai.tool.call.result` span attribute.

18g. The canonical tool-result span attribute is recorded whenever its span is recording, independent of audit/content flags. Existing developer-log redaction and legacy choice-event/log gates remain unchanged.

18h. If inspection fails, `AuditLogger` MUST receive no approved result payload; the rejected result MUST NOT appear in a canonical tool-result attribute.

18i. The DeepAgents adapter MUST hold normalized `ToolResultEvent` records until the model-boundary middleware accepts the associated results. If inspection rejects any result, the adapter MUST release no pending result events from that boundary. Rejected output and pending sibling output MUST NOT enter canonical trace attributes or legacy audit payloads.

### Metrics

19. The sandbox MUST record the following `gen_ai.*` Prometheus histograms during agent execution (`metrics.py`). Histograms are **in-process only** (`prometheus_client`); the batch entrypoint MUST NOT expose a `/metrics` HTTP scrape endpoint and MUST NOT export histograms to OTLP or Pushgateway at shutdown. Short-lived one-shot pods are a poor fit for pull-based Prometheus scraping; the current aggregate `gen_ai.usage.*` values on the invocation span remain the OTLP trace usage signal. Future per-generation values are separate and MUST NOT be summed with the root aggregate. Unit tests (`tests/test_metrics.py`) verify histogram recording.

| Metric | Type | Unit | Labels |
|---|---|---|---|
| `gen_ai_client_token_usage` | Histogram | `{token}` | `gen_ai_token_type`, `gen_ai_request_model`, `gen_ai_provider_name`, `gen_ai_operation_name` |
| `gen_ai_client_operation_duration_seconds` | Histogram | `s` | `gen_ai_request_model`, `gen_ai_provider_name`, `gen_ai_operation_name` |
| `gen_ai_execute_tool_duration_seconds` | Histogram | `s` | `gen_ai_tool_name` |

20. Token usage histogram bucket boundaries MUST be `[1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864]` (per semconv recommendation). Root reasoning usage is recorded as `gen_ai.usage.reasoning.output_tokens` only when the existing aggregate value is nonzero; it is not a `gen_ai.token.type` value.

### Configuration

21. The sandbox receives audit and shared tracing configuration through `LIGHTSPEED_AUDIT_ENABLED`, `LIGHTSPEED_CAPTURE_CONTENT`, and `OTEL_EXPORTER_OTLP_ENDPOINT`; run correlation uses `LIGHTSPEED_AGENTICRUN_UID` and `LIGHTSPEED_AGENTICRUN_STEP`. Audit is enabled only when `LIGHTSPEED_AUDIT_ENABLED` is `"true"` after strip and lowercasing. Audit-disabled suppresses stdout and span-event log copies, but MUST NOT suppress tracing when the shared OTLP endpoint is configured. PR1 span attributes bypass audit/content payload gates only on spans that the existing runtime records.

22. When `OTEL_EXPORTER_OTLP_ENDPOINT` is configured, the sandbox MUST configure OTLP exporters for traces and logs targeting that same endpoint. Trace export is active whenever the endpoint is set. The span-event → log processor is attached only when the endpoint is set and audit is enabled; the stdout span exporter emits when audit is enabled. When the endpoint is absent, no OTLP exporters or span-event log forwarding are configured.

### OTLP Log Emission (Templog) [OLS-3515]

23. When `OTEL_EXPORTER_OTLP_ENDPOINT` is set and audit is enabled, the sandbox MUST emit the compliance view of audit span events as OTLP log records to that endpoint, in addition to stdout and OTLP trace export.

24. Each forwarded span-event OTLP log record MUST carry log record attributes `agenticrun.uid` and `agenticrun.phase` (from `LIGHTSPEED_AGENTICRUN_UID` and `LIGHTSPEED_AGENTICRUN_STEP` when set), plus `event` (the span event name). These MUST be stamped via stdlib `logging` `extra` so `LoggingHandler` preserves them. The span event attributes are the JSON log-record body; when content capture is disabled, `gen_ai.choice` copies MAY have an empty body. TraceID MUST come from the ended span's context. TracerProvider and LoggerProvider share one Resource with pinned `service.name`; run UID and phase remain record/span attributes. When audit and the OTLP endpoint are enabled but either correlation value cannot be resolved, the sandbox MUST log a startup warning without failing startup. The span-event → log processor MUST NOT forward automatic OTel `exception` events; other intentional span events remain eligible.

25. When `OTEL_EXPORTER_OTLP_ENDPOINT` is set, stdlib Python logging MUST be exported as OTLP logs via `LoggingHandler` (dual-ship with stderr). Templog audit records use that same path: the span-event processor logs through stdlib, not a separate OTel Logs API emit.

26. When `OTEL_EXPORTER_OTLP_ENDPOINT` is absent, no OTLP log records are emitted. Graceful degradation.

### Agentic Trace Profile

27. The audit/tracing layer MUST emit the PR1 invocation/tool span attributes defined in `data-collection.md` through the existing shared trace runtime. Existing `gen_ai.choice` events and their log projections remain legacy outputs with their old gates and ordering. New sandbox-owned canonical per-generation capture is planned for later PRs, not PR1; existing SDK-native ADK spans remain unchanged framework detail until PR3's exported-view normalization. The sandbox producer exception does not change the parent collection contract outside this scope.

## Verification

- Exported regressions: [test_run_agent.py](../../../tests/test_run_agent.py), [test_audit.py](../../../tests/test_audit.py), and [test_tracing.py](../../../tests/test_tracing.py) cover exported invocation/tool attributes and unchanged audit/log projections.
- Offline smoke proof: the controller exercised actual `run_agent_query()` plus `init_tracer()` with in-memory trace/log exporters, OTLP encode/decode with shuffled spans, and validation against the five pinned JSON schemas. Full input/system/terminal/tool attributes survived; all four audit/content-flag combinations retained exactly the prior developer/templog semantics and choice events.
- Cancellation boundary/OTLP smoke: root ERROR/`CancelledError` and pending-tool ERROR/`missing_tool_result` survived the wire with no tool result, duration-histogram observation, or cancellation-triggered choice/log flush.
- Scope limit: no provider API or deployed collector/FileExporter/Dataverse delivery was exercised. Existing size limits remain best effort; no producer truncation was added.
- Existing behavior checks: [test_logging.py](../../../tests/test_logging.py) — payload-free developer records for inspected DeepAgents results; [test_metrics.py](../../../tests/test_metrics.py) — in-process histogram recording.
- Existing live batch coverage: [sandbox_e2e.feature](../../../tests/e2e/features/sandbox_e2e.feature) checks trace and bridged audit-log export; it is not deployed FileExporter/Dataverse proof for this profile.

### MCP Semantic Conventions [UNTRACKED]

28. MCP tool connectivity is implemented. Additional MCP span attributes (`mcp.method.name`, `mcp.session.id`, `mcp.protocol.version`, `network.transport`) are not implemented and have no Jira story. Do not treat this table as a current MUST until a ticket exists. Prefer `gen_ai.tool.*` on tool spans today.

## Cross-References

- `run-api.md` — batch tracing lifecycle and invocation span
- `provider-contract.md` — unchanged provider events and planned canonical generation capture
- Parent workspace `ols/.ai/spec/what/templog.md` — temporary audit log storage; sandbox emission tracked by OLS-3515
- `ols/.ai/spec/what/audit-logging.md` — parent cross-repository audit/logging and correlation requirements; sandbox producer exception is scoped in `data-collection.md`
- `data-collection.md` — PR1 canonical span profile and its parent-contract scope
- `ols/.ai/spec/what/agentic-data-collection.md` — parent collection contract, unchanged outside the sandbox producer exception
- [Pinned OTel GenAI profile](https://github.com/open-telemetry/semantic-conventions-genai/tree/4f85037ef86e92c510d2ef881a58f1076f6fc0e4/docs/gen-ai) — Development snapshot; named alignment profile
- [OTel MCP Semantic Conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/mcp.md)
- Parent workspace `ols/.ai/spec/what/tool-result-inspection.md` — cross-repository tool-result inspection contract
