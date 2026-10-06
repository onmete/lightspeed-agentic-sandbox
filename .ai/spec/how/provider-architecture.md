# Architecture: data flow, SDK integration

Audience: AI agents. File paths and symbols allowed here.
Package tree: `AGENTS.md`. Behavioral rules: `what/run-api.md`, `what/provider-contract.md`, `what/configuration.md`, `what/audit-logging.md`.

## Data Flow

1. Startup: `batch.main()` reads `/input/`, then calls `resolve_sdk()`, `parse_reasoning_config()`, and `parse_mcp_servers()` (fail-fast on bad env), `run_readiness_checks()`, `init_tracer()` when `otel_runtime_enabled()`, and `create_provider()`. [PLANNED: OLS-3743] Startup also requires and validates `LIGHTSPEED_AGENT_TIMEOUT_SECONDS` and `LIGHTSPEED_AGENT_MAX_TURNS` before provider invocation.
2. `run_agent_query()` applies context prefix, passes pre-parsed `mcp_servers` and operator-resolved maximum turns into `ProviderQueryOptions`, and calls `provider.query(...)`. [PLANNED: OLS-3743] The outer agent invocation is bounded by the operator-resolved timeout; timeout returns a structured classification used by Result status assembly.
2a. When enabled, the DeepAgents adapter inspects effective model-bound tool results and errors after SDK output limits and artifact offload. The isolated classifier uses the resolved DeepAgents model; a benign preview does not classify an entire stored artifact.
2b. DeepAgents applies SAFE-02 at the model boundary even when inspection is disabled. Output limits and artifact offload run first; enabled inspection runs before wrapping. The middleware escapes external closing markers, HTML-escapes tool names, and wraps model-facing results before delivery to the main-agent and general-purpose-subagent models, whose prompts include the trust instruction. Before automatic summarization, separate middleware wraps copies of historical tool results and supplies the trust instruction, leaving original messages unchanged for state and history offload.
3. Provider SDK lifecycle callbacks report actual model-request/tool-execution boundaries to `AuditLogger` and in-process metrics, independently of normalized events. A successful native tool span records the raw callback result at execution completion before inspection and remains UNSET even if later rejected. Inspection gates model delivery and release of pending normalized application result events; the handler consumes normalized events through `EventLogger` until the first `result`. The consumer sanitizes a logging-only copy of DeepAgents tool-call/result events while leaving normalized events and callback-derived source spans unchanged. Normalized events do not create OTel operation spans or metrics.
4. `publish_agent_result()` builds status from agent output, creates Result CR via Kubernetes API (`create_namespaced_custom_object`), replaces status (`replace_namespaced_custom_object_status`).
5. `shutdown_tracer()`; exit 0 on sandbox success (including agent failure), non-zero on infrastructure failure with termination log.

## Key Abstractions

- **Config mapping:** `resolve_sdk()` owns env → SDK name; factory does not read provider env vars.
- **Factory:** `create_provider(name)` lazy-imports the selected adapter.
- **Events:** Normalized `ProviderEvent` union decouples agent layer from vendor streaming models.
- **Options:** `ProviderQueryOptions` is the single bundle passed into every adapter (includes `mcp_servers`, `reasoning_config`, and optional `audit_logger`).
- **Model resolution:** `resolve_router_model()` / `resolve_startup_model()` in `config.py`.
- **Result publishing:** `publish_results/publish.py` + `status.py` — Kubernetes client, no `oc` subprocess.
- **Result inspector:** A focused module implements `openshift/ols/.ai/spec/what/tool-result-inspection.md` and exposes `ToolResultSafetyInspectionFailed` to the batch path.
- **Tool-output boundary:** A focused formatter escapes external closing markers and tool names, then supplies the fixed markers. DeepAgents middleware applies it to model-facing copies at the model boundary, not in the provider-event consumer; summarization middleware wraps copies of historical tool results for the summary model. See `../decisions/0001-tool-output-boundary.md`.

## Provider-egress CA bundle [PLANNED: OLS-3042]

The batch startup/configuration path discovers `.crt` and `.pem` files below
`/var/run/secrets/lightspeed/tls/`, combines them with the platform/system
trust store, and exposes one runtime bundle to Python and provider TLS code.
Provider adapters MUST consume the shared bundle and MUST NOT contain
operator Secret names, source-specific CA paths, or per-source trust logic.
The generic TLS values passed by the agentic operator are parsed once at the
runtime boundary; adapters receive configured TLS behavior through shared
configuration rather than independent CA arguments.

## Integration Points

- **Batch entrypoint:** `python -m lightspeed_agentic.batch` (`batch.py`).
- **Kubernetes API:** `kubernetes` Python client for Result CR create + status update (ServiceAccount token).
- **deepagents (+ langchain-anthropic, langchain-google-vertexai, langchain-aws, langchain-mcp-adapters):** `create_deep_agent`, `LocalShellBackend`, MCP via `MultiServerMCPClient`.
- **google-adk / google.genai:** Optional `google-adk>=2.5.0`; SDK imports remain lazy. `Agent`, `Runner`, `ExecuteBashTool`, `SkillToolset`; MCP via `McpToolset` + `StreamableHTTPConnectionParams`. Apply ADK's process-wide native-telemetry alias suppression once in the sandbox's one-shot batch process; this is not a per-invocation toggle or a long-lived, multi-run process contract.
- **openai-agents (+ openai):** `SandboxAgent`, `Runner`, `UnixLocalSandboxClient`. MCP via `MCPServerStreamableHttp`. Client selection by provider (`what/provider-contract.md` rule 29): native OpenAI → `AsyncOpenAI` + `OpenAIResponsesModel`; Azure → the SDK's built-in `AsyncAzureOpenAI` + `OpenAIChatCompletionsModel`.
- **azure.identity (Azure Entra ID):** `ClientSecretCredential` + `get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")` supplies the `azure_ad_token_provider` passed to `AsyncAzureOpenAI`; the library owns token caching/refresh (`what/provider-contract.md` rule 38). Imported inside the adapter method (optional-extra convention). **New dependency** [OLS-3050]: `azure-identity` (pulls `azure-core`) is added to the `openai` optional extra — `AsyncAzureOpenAI` and its `azure_ad_token_provider` param ship in `openai` (via `openai-agents`), but the credential classes do not. Adding it requires regenerating the Konflux hashed requirements/lockfiles.
- **OpenTelemetry:** `tracing.py` configures the TracerProvider; `audit.py` records standard GenAI spans; `metrics.py` records in-process Prometheus histograms. Each invocation has one `invoke_agent lightspeed` span with native per-request model and tool spans beneath it. Standalone safety-classifier inference spans retain actual operation metadata and metrics but omit input/output/system content under the tool-result-inspection boundary. See `../what/audit-logging.md`.

## Implementation Notes

- **DeepAgents model routing:** `_resolve_model()` checks `CLAUDE_CODE_USE_VERTEX` and `CLAUDE_CODE_USE_BEDROCK`.
- **OpenAI/Azure adapter (`providers/openai.py`):** branches on `LIGHTSPEED_PROVIDER`. For `azure`, builds `AsyncAzureOpenAI` from `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_API_VERSION` / deployment; Entra ID mode (per `_resolve_azure()` in `config.py`, `what/configuration.md` rule 9a) reads `client_id`/`tenant_id`/`client_secret` from `/var/run/secrets/llm-credentials/` and passes `azure_ad_token_provider`; API-key mode passes `api_key`. Fail-fast on definitive token-acquisition failure — do not construct or use a broken client.
- **OpenAI telemetry:** Use the model-request proxy, native RunHooks, and the narrow FunctionTool failure wrapper. Shell, Filesystem, and MCP objects remain SDK-owned; telemetry does not recover failures that the SDK has already converted into normal results.
- **Telemetry encoders:** Keep SDK-specific messages, response metadata, output types, and JSON normalization in each provider module. `_telemetry_base.py` shares only field/mapping access and JSON argument decoding. Gemini's sandbox encoder covers text, reasoning, function calls, and function responses; multimodal and code-execution parts are not currently supported by this path.
- **Bedrock credentials (`config.py::_resolve_bedrock`):** [OLS-4092] the Anthropic-on-Bedrock model path (`ChatBedrockConverse` via `deepagents`) is unchanged; only credential resolution grows. Reads `aws_access_key_id` / `aws_secret_access_key` / optional `role_arn` from `/var/run/secrets/llm-credentials/` (`what/configuration.md` rule 9b). With `role_arn`, `botocore` performs STS assume-role and owns short-lived-credential refresh (delegated-token principle, `what/provider-contract.md` rule 38); without it, static keys are used. `boto3`/`botocore` are already present via `langchain-aws` — no new dependency.
- **DeepAgents streaming:** `astream(stream_mode="messages")`.
- **DeepAgents result inspection:** Middleware inspects each effective model-bound preview, result, error, file read, or search result after SDK output limits and artifact offload, before model delivery and normalized result-event release. The isolated classifier uses the resolved DeepAgents model; a benign preview does not attest to the full stored artifact. Classifier resources are constructed only when inspection is enabled.
- **DeepAgents result boundary:** `inspection/middleware.py` applies SAFE-02 at `awrap_model_call()` after effective-result transformations, independent of `options.tool_output_inspection_enabled`. It covers the main-agent and general-purpose-subagent model paths. Separate summarization middleware wraps copies of historical tool results for the summary model while leaving original messages unchanged for state and history offload.
- **Model/event separation:** Preserve original effective content for inspection correlation and normalized events. Track sandbox-owned wrapping through internal identity/state, never external marker text; preserve message metadata and ordering, and do not modify stored artifacts. Rejected inspection suppresses model delivery and all pending `ToolResultEvent`s from that model boundary, not completed native source spans.
- **DeepAgents boundary tests:** Offline middleware and adapter tests cover the requirements in `what/provider-contract.md`. They exercise actual model-request interception and normalized events; a formatter-only assertion does not prove that the model receives wrapped content.
- **Classifier integration:** Construct the contract-defined isolated invocation from the resolved model and bind its schema through the existing LangChain structured-output interface.
- **Inspection failure:** Do not emit a rejected `ToolResultEvent`. Propagate `ToolResultSafetyInspectionFailed` to `batch.py`, which exits nonzero without publishing a Result CR.
- **Gemini bash:** Monkey-patches `run_async` for confirmation and `bash -c` wrapping.
- **MCP Secret headers:** `Secret` sources resolve one mounted Secret value per header. No Secret key selection is performed, and resolved values are used as complete header values.
- **Containerfile:** Multi-stage hermetic build; `oc`/`kubectl` in image for **agent tools** (not Result CR publishing); user `agent`; `catatonit`; batch CMD.
- **Unit tests:** `test_run_agent.py`, `test_batch.py`, `test_ready.py`, `test_publish_results_*.py`, `test_batch_e2e_helpers.py` (harness helpers, no cluster).
- **[PLANNED: OLS-3743] Execution limits:** Parse timeout/max-turn environment values once in `batch.py`; pass the parsed values to `run_agent_query()`. Provider adapters continue to receive maximum turns only through `ProviderQueryOptions`. Preserve timeout as structured internal state through `publish_results/status.py` so Result condition selection never depends on matching summary text.
- **Live batch BDD:** `tests/e2e/` feature files via `scripts/e2e-containers.sh` — see [e2e-testing.md](../what/e2e-testing.md).
- **Live cluster tests:** batch BDD (`tests/e2e/`); shares `run_batch_query` with the e2e harness. See [e2e-testing.md](../what/e2e-testing.md).
