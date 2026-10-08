# Agentic Data Collection

Sandbox producer contract for the PR1 GenAI span profile, aligned to the [pinned OTel GenAI conventions](https://github.com/open-telemetry/semantic-conventions-genai/tree/4f85037ef86e92c510d2ef881a58f1076f6fc0e4/docs/gen-ai). The conventions were Development at that revision; this named profile does not claim every optional convention or add a semantic-conventions dependency.

For the sandbox producer only, this profile supersedes conflicting candidate/event producer requirements in `ols/.ai/spec/what/agentic-data-collection.md`. The parent contract and downstream collection requirements remain unchanged. PR1 is implemented; new sandbox-owned canonical per-generation capture is planned for later PRs.

## Producer Boundary

1. When the existing trace runtime records spans, PR1 content is stored in span attributes. Do not add a custom transcript attribute/event or skill events, and do not infer skill use from text or generic tool output.
2. Full PR1 attributes are recorded on recording spans independently of audit/content flags. Existing runtime and exporter activation remain as defined in `run-api.md` and `audit-logging.md`; this profile does not activate a previously disabled trace path.
3. Preserve existing safety inspection, execution limits, and prior result redaction. Capture only values exposed by the existing normalized execution path; do not recover pre-redaction output or bypass inspection.
4. Export only through the existing OTLP trace runtime. FileExporter, rotation/retention, upload, and Dataverse behavior are downstream and outside the sandbox runtime. Delivery is best-effort; FileExporter may reject an over-limit batch. The producer adds no truncation to fit that limit and does not write product-data files or read collection state.

## PR1 Span Profile

Structured message and tool values use compact JSON strings with `ensure_ascii=False` and separators `(",", ":")`, matching the schemas at the pinned upstream revision. Escape unpaired surrogate code points for UTF-8-safe OTLP export while preserving ordinary Unicode and JSON-decoded content.

### Invocation Span

- The invocation span is `invoke_agent`, `SpanKind.INTERNAL`, with `gen_ai.operation.name="invoke_agent"`. It is a child of the received operator context and retains available `agenticrun.uid` and `agenticrun.phase` values; missing correlation is not invented.
- `gen_ai.request.model` is the configured request model. The invocation span does not claim `gen_ai.provider.name` or `gen_ai.response.model`.
- `gen_ai.input.messages` is `[{"role":"user","parts":[{"type":"text","content":prompt}]}]`, where `prompt` is the effective post-context-prefix input, including an observed empty string.
- `gen_ai.system_instructions` is `[{"type":"text","content":system_prompt}]`, using the configured supplied string, including an empty string.
- Set `gen_ai.output.type="json"` iff `output_schema is not None`.
- When a `ResultEvent` is observed, set `gen_ai.output.messages` to `[{"role":"assistant","parts":[{"type":"text","content":event.text}]}]` before parsing or result shaping. Preserve an observed empty string; omit the attribute when no `ResultEvent` arrives. Do not concatenate choice events.
- Keep existing aggregate `ResultEvent` usage on this span. Record `gen_ai.usage.reasoning.output_tokens` only for a nonzero aggregate reasoning count, as today. Do not sum root aggregates with future per-generation usage.
- Set ERROR status and `error.type` to `timeout`, the exception class name, or `empty_response` for operational invocation failures. Cancellation is propagated unchanged after setting root ERROR and `error.type="CancelledError"`. An unsuccessful analysis/execution outcome in a normal response does not set ERROR or `error.type`; preserve it in the exact terminal output and Result payload, independently of span status.

### Tool Spans

- Each local tool execution uses an `execute_tool {name}` `SpanKind.INTERNAL` child span with `gen_ai.operation.name="execute_tool"` and `gen_ai.tool.name`.
- Set `gen_ai.tool.call.id` only from an actual SDK call ID; never publish a fabricated ID. Set `gen_ai.tool.call.arguments` and `gen_ai.tool.call.result` only when their respective values were observed.
- Normalize string payloads with strict JSON parsing: accept standards-compliant finite JSON only; treat `NaN`, `Infinity`, `-Infinity`, exponent overflow, and parser/encoder-limit failures as non-JSON. Pass decoded dictionaries through unchanged and wrap other successfully decoded values as `{"content": value}`.
- On any parse, encode, or limit rejection, preserve the complete original raw string in `{"content": raw_string}`. This is sandbox normalization, not a provider-native field; do not truncate it or raise a telemetry-only error into the provider path. Preserve observed empty values; absent arguments/results remain absent.
- A result without an ID may match only when exactly one call is pending. Do not attach an ambiguous result to an arbitrary call. Unresolved calls end with ERROR and `error.type="missing_tool_result"`; do not invent a result. On cancellation, pending tool spans close ERROR with that type and no result or tool-duration histogram observation.

## Reconstruction and Ordering

1. Reconstruct from native trace/span IDs, parent relationships, and span times. Read the invocation's system instructions and initial user message once; expose its terminal output separately, not as a model generation.
2. Pair tool calls/results by trace and actual SDK call ID where available; only the producer's documented unique-pending fallback applies when a result ID is missing. Retain native tool spans and concurrent intervals; span/export/file order is not a causal total order.
3. Existing `gen_ai.choice` events and developer/templog log projections remain legacy outputs, not a canonical ordered transcript. Their buffer flush order does not establish cross-type text/reasoning chronology. Do not infer root delta chronology or repeated SDK input history.
4. Missing spans or files are missing evidence, not reconstructed successes.

## Planned Canonical Provider Capture

Later PRs are planned to add new sandbox-owned canonical CLIENT `chat`/`generate_content` spans for main-agent SDK generations, including the separate DeepAgents structured-output shaping generation; PR1 adds none. Existing SDK-native ADK `call_llm`/`generate_content` spans remain unchanged framework detail, outside the canonical profile until PR3's exported-view normalization. Planned canonical capture excludes nested-agent, classifier, and summarization generations; no new optional provider-event fields are part of PR1.

## Verification

- Exported regressions: [test_run_agent.py](../../../tests/test_run_agent.py), [test_audit.py](../../../tests/test_audit.py), and [test_tracing.py](../../../tests/test_tracing.py) cover invocation/tool attributes, exported span behavior, and legacy log/event gates.
- Offline smoke proof: the controller exercised actual `run_agent_query()` plus `init_tracer()` with in-memory trace/log exporters, OTLP encode/decode with shuffled spans, and validation against the five pinned JSON schemas. Full input/system/terminal/tool attributes survived; all four audit/content-flag combinations retained exactly the prior semantic developer/templog logs and choice events.
- Scope limit: no provider API or deployed collector/FileExporter/Dataverse delivery was exercised. Oversized batches remain best-effort and are not silently truncated by the producer.

## Cross-References

- Parent contract (unchanged outside the sandbox producer exception): `ols/.ai/spec/what/agentic-data-collection.md`
- `audit-logging.md` — shared trace runtime, tool spans, and legacy logging projections
- `provider-contract.md` — unchanged provider events and planned canonical provider-generation capture
- `run-api.md` — effective input construction and trace lifecycle
