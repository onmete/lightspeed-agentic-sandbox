# Agentic Data Collection

Sandbox producer contract for the PR1+PR2 named GenAI trace profile, aligned to the [pinned OTel GenAI conventions](https://github.com/open-telemetry/semantic-conventions-genai/tree/4f85037ef86e92c510d2ef881a58f1076f6fc0e4/docs/gen-ai). The conventions were Development at that revision; this named profile does not claim every optional convention or add a semantic-conventions dependency.

For the sandbox producer only, this profile supersedes conflicting candidate/event producer requirements in `ols/.ai/spec/what/agentic-data-collection.md`. The parent contract and downstream collection requirements remain unchanged. PR1 invocation/tool spans and PR2 DeepAgents/OpenAI main-agent generation spans are implemented. Gemini generation capture and ADK exported-view normalization remain explicitly planned for PR3.

## Producer Boundary

1. When the existing trace runtime records spans, canonical profile content is stored in span attributes. Do not add a custom transcript attribute/event or skill events, and do not infer skill use from text or generic tool output.
2. Full profile attributes are recorded on recording spans independently of audit/content flags. Existing runtime and exporter activation remain as defined in `run-api.md` and `audit-logging.md`; this profile does not activate a previously disabled trace path.
3. Preserve existing safety inspection, execution limits, and prior result redaction. Capture only values exposed by the existing normalized execution path; do not recover pre-redaction output or bypass inspection.
4. Export only through the existing OTLP trace runtime. FileExporter, rotation/retention, upload, and Dataverse behavior are downstream and outside the sandbox runtime. Delivery is best-effort; FileExporter may reject an over-limit batch. The producer adds no truncation to fit that limit and does not write product-data files or read collection state.

## Invocation and Tool Spans (PR1)

Structured message and tool values use compact JSON strings with `ensure_ascii=False` and separators `(",", ":")`, matching the schemas at the pinned upstream revision. Escape unpaired surrogate code points for UTF-8-safe OTLP export while preserving ordinary Unicode and JSON-decoded content.

### Invocation Span

- The invocation span is `invoke_agent`, `SpanKind.INTERNAL`, with `gen_ai.operation.name="invoke_agent"`. It is a child of the received operator context and retains available `agenticrun.uid` and `agenticrun.phase` values; missing correlation is not invented.
- `gen_ai.request.model` is the configured request model. The invocation span does not claim `gen_ai.provider.name` or `gen_ai.response.model`.
- `gen_ai.input.messages` is `[{"role":"user","parts":[{"type":"text","content":prompt}]}]`, where `prompt` is the effective post-context-prefix input, including an observed empty string.
- `gen_ai.system_instructions` is `[{"type":"text","content":system_prompt}]`, using the configured supplied string, including an empty string.
- Set `gen_ai.output.type="json"` iff `output_schema is not None`.
- When a `ResultEvent` is observed, set `gen_ai.output.messages` to `[{"role":"assistant","parts":[{"type":"text","content":event.text}]}]` before parsing or result shaping. Preserve an observed empty string; omit the attribute when no `ResultEvent` arrives. Do not concatenate choice events.
- Keep existing aggregate `ResultEvent` usage on this span. Record `gen_ai.usage.reasoning.output_tokens` only for a nonzero aggregate reasoning count, as today. This root usage remains a separate aggregate signal and MUST NOT be summed with child-generation usage.
- Set ERROR status and `error.type` to `timeout`, the exception class name, or `empty_response` for operational invocation failures. Cancellation is propagated unchanged after setting root ERROR and `error.type="CancelledError"`. An unsuccessful analysis/execution outcome in a normal response does not set ERROR or `error.type`; preserve it in the exact terminal output and Result payload, independently of span status.

### Tool Spans

- Each local tool execution uses an `execute_tool {name}` `SpanKind.INTERNAL` child span with `gen_ai.operation.name="execute_tool"` and `gen_ai.tool.name`.
- Set `gen_ai.tool.call.id` only from an actual SDK call ID; never publish a fabricated ID. Set `gen_ai.tool.call.arguments` and `gen_ai.tool.call.result` only when their respective values were observed.
- Normalize string payloads with strict JSON parsing: accept standards-compliant finite JSON only; treat `NaN`, `Infinity`, `-Infinity`, exponent overflow, and parser/encoder-limit failures as non-JSON. Pass decoded dictionaries through unchanged and wrap other successfully decoded values as `{"content": value}`.
- On any parse, encode, or limit rejection, preserve the complete original raw string in `{"content": raw_string}`. This is sandbox normalization, not a provider-native field; do not truncate it or raise a telemetry-only error into the provider path. Preserve observed empty values; absent arguments/results remain absent.
- A result without an ID may match only when exactly one call is pending. Do not attach an ambiguous result to an arbitrary call. Unresolved calls end with ERROR and `error.type="missing_tool_result"`; do not invent a result. On cancellation, pending tool spans close ERROR with that type and no result or tool-duration histogram observation.

## Provider Generation Spans (PR2)

Only DeepAgents and OpenAI currently produce canonical provider-generation spans. Each accepted main-agent model request creates a standard `SpanKind.CLIENT` span named `chat {gen_ai.request.model}` with `gen_ai.operation.name="chat"`, the configured `gen_ai.request.model`, and the provider identity resolved by the adapter (`anthropic`, `aws.bedrock`, `gcp.vertex_ai`, or `openai`). `start_generation_span` uses the invocation context captured before the provider query, does not make the generation span current, and copies available `agenticrun.uid` and `agenticrun.phase` values. This preserves the existing parent and active logging context.

The DeepAgents callback selects main-agent requests and excludes named nested agents, `nostream` classifier calls, and summarization calls. It also captures the separate structured-output shaping generation from its raw `AIMessage`, not the parsed JSON result. OpenAI hooks select only the main `SandboxAgent` and record its completed `ModelResponse`. OpenAI sets `openai.api.type` to `responses` or `chat_completions` using the adapter's existing `uses_responses_api` routing decision. Azure can select either API type based on API-version support; do not infer the value from the provider label alone.

Generation spans contain output, not `gen_ai.input.messages` or `gen_ai.system_instructions`; SDK input messages and repeated request histories are ignored.

### Ordered Output, Metadata, and Usage

`gen_ai.output.messages` contains only SDK-observed output, serialized as compact JSON. Preserve native message, item, content, and part order:

- DeepAgents maps each observed `LLMResult.generations` choice to an assistant message using its `message.content_blocks`; plain string content is a text part when no blocks exist. Complete `tool_call` blocks retain the observed ID, name, and arguments (omit an unavailable ID). A partial `tool_call_chunk` remains an upstream `GenericPart` with its native `type` discriminator and each observed field (`id`, `name`, `args`, `index`); it is not promoted to a complete `tool_call` or reassembled.
- OpenAI maps `ModelResponse.output` and nested content arrays in source order into an assistant message. Text and refusal content remain literal; reasoning content and summary text become reasoning parts. Function calls use their actual call ID/name and JSON-decode arguments only when valid, otherwise preserving the original string. Custom calls use their actual call ID/name and literal `.input`, not a nonexistent `.arguments`.
- Set `gen_ai.response.model`, `gen_ai.response.id`, and `gen_ai.response.finish_reasons` only when the SDK result exposes those actual values. DeepAgents may use message/generation metadata or observed `LLMResult.llm_output`; do not substitute the configured model or a LangChain run ID. OpenAI uses the actual Responses `response_id`, or a single consistent completion ID from Chat Completions output `provider_data`; do not use a synthetic Responses item ID or transport request ID. Omit OpenAI response model and finish reasons when the `ModelResponse` does not provide them.
- DeepAgents records present `usage_metadata` input, output, and reasoning counts, preserving explicit zero and omitting missing fields. OpenAI records `ModelResponse.usage` only when `requests > 0`; preserve observed zero input/output counts and omit missing counts. Emit OpenAI reasoning output tokens only for a supplied nonzero reasoning detail. Never manufacture zero usage from missing evidence.

On a DeepAgents generation error, keep partial output/metadata/usage only if the callback receives an `LLMResult`; otherwise output is absent. Do not add a delta-recovery buffer. OpenAI records output only from the completed response callback; if an exception or cancellation occurs before that response is exposed, output remains absent rather than being reconstructed from stream deltas. OpenAI terminal cleanup ignores late callbacks after cleanup but does not cancel SDK background execution. An observed exception marks the generation ERROR; early closure without one uses `error.type="generation_interrupted"`.

## Reconstruction and Ordering

1. Select the `invoke_agent` trace subtree using native trace/span IDs and parent relationships; retain available `agenticrun.uid`/phase and native span IDs.
2. Read invocation system instructions and the initial user message once. Expose the exact terminal root output separately, not as another model generation.
3. For PR1+PR2, read DeepAgents/OpenAI `chat` generation outputs in native span start-time order. Preserve each response's message/part order, including the raw DeepAgents shaping generation. Do not append the root output to generation output or reconstruct repeated input histories.
4. Join complete local tool-call parts to `execute_tool` spans by trace ID and actual call ID. A partial `tool_call_chunk` is not a complete call. Retain concurrent tool intervals; do not treat export-batch or file order as a causal total order.
5. Use per-generation usage for model analysis and existing invocation usage separately; never sum the root aggregate and child usage. Ignore legacy `gen_ai.choice` events for canonical reconstruction.
6. Missing spans or files are missing evidence, not reconstructed successes.

## PR3 Planned Provider Work

Canonical Gemini `generate_content` capture and the stateless ADK native-span exported-view normalization remain PR3 work. No Gemini canonical generation capture is implemented in the PR1+PR2 profile; existing ADK `call_llm`/`generate_content` spans remain unnormalized framework detail until PR3.

## Verification

- Exported regressions: [test_run_agent.py](../../../tests/test_run_agent.py), [test_audit.py](../../../tests/test_audit.py), and [test_tracing.py](../../../tests/test_tracing.py) cover invocation/tool spans and legacy log/event gates. Provider-generation scenarios are isolated in [test_deepagents_generation_spans.py](../../../tests/test_deepagents_generation_spans.py) and [test_openai_generation_spans.py](../../../tests/test_openai_generation_spans.py), covering output mapping, metadata/usage, selection, and lifecycle cleanup.
- Offline PR1+PR2 smoke passed with `uv run --offline --frozen --all-extras python /tmp/ols-genai-pr2-smoke.py`: actual `run_agent_query()` through a DeepAgents fake-model graph (including nested-agent exclusion and raw shaping), the OpenAI Agents `Runner` with a scripted `Model`, and `init_tracer()` with in-memory trace/log exporters. All four audit/content-flag combinations retained the prior developer/templog and choice-event behavior. The smoke exercised protobuf encode/decode and shuffled-span reconstruction, inspection rejection, failure, and cancellation. The wire decoded 51 spans (33,029 bytes); five pinned schemas validated 74 values.
- Scope limit: this is producer/wire proof only. It did not contact live model APIs or exercise a deployed collector, FileExporter pipeline, or Dataverse delivery. The smoke did not exercise the DeepAgents `LLMResult.llm_output` metadata fallback or partial `tool_call_chunk` GenericPart mapping; those mappings preserve only SDK-observed values/fields. Collection remains best-effort and producer-side truncation is not added.

## Cross-References

- Parent contract (unchanged outside the sandbox producer exception): `ols/.ai/spec/what/agentic-data-collection.md`
- `audit-logging.md` — shared invocation/tool/generation spans and legacy logging projections
- `provider-contract.md` — unchanged provider-event behavior and PR1+PR2 generation capture
- `run-api.md` — effective input construction and trace lifecycle
