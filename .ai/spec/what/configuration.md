# Behavioral spec: configuration, environment, deployment

Audience: AI agents (Claude). Precision over narrative.

Cross-references: how options are consumed in code → `how/provider-architecture.md`. HTTP fields → `run-api.md`. Provider options → `provider-contract.md`.

## Behavioral Rules

### Operator contract env var resolver

1. **Operator contract detection.** When `LIGHTSPEED_LLM_TYPE` is present in the process environment, the sandbox MUST derive provider selection and SDK-specific env vars from the operator contract vars (rules 2–7). When absent, the sandbox MUST fall back to legacy direct-env-var behavior (rules 8 onward) for backward compatibility with non-operator deployments.

2. **Provider derivation from operator contract.** The sandbox MUST map `LIGHTSPEED_LLM_TYPE` + `LIGHTSPEED_VERTEX_MODEL_PROVIDER` (when applicable) to the internal provider adapter:

    | `LIGHTSPEED_LLM_TYPE` | `LIGHTSPEED_VERTEX_MODEL_PROVIDER` | Provider adapter |
    |---|---|---|
    | `Anthropic` | — | `claude` |
    | `GoogleCloudVertex` | `Anthropic` | `claude` |
    | `GoogleCloudVertex` | `Google` | `gemini` |
    | `GoogleCloudVertex` | `OpenAI` | `openai` |
    | `OpenAI` | — | `openai` |
    | `AzureOpenAI` | — | `openai` |
    | `AWSBedrock` | — | `claude` |

3. **Model derivation from operator contract.** `LIGHTSPEED_MODEL` MUST be mapped to the appropriate SDK-specific model env var based on the derived provider adapter (`ANTHROPIC_MODEL` for claude, `GEMINI_MODEL` for gemini, `OPENAI_MODEL` for openai). This sets the env var in-process before provider construction.

4. **Vertex derivation (Claude).** When `LIGHTSPEED_LLM_TYPE=GoogleCloudVertex` and derived adapter is `claude`, the resolver MUST set `CLAUDE_CODE_USE_VERTEX=1`, `ANTHROPIC_VERTEX_PROJECT_ID` from `LIGHTSPEED_GCP_PROJECT`, and `CLOUD_ML_REGION` from `LIGHTSPEED_GCP_REGION`.

5. **Vertex derivation (Gemini).** When `LIGHTSPEED_LLM_TYPE=GoogleCloudVertex` and derived adapter is `gemini`, the resolver MUST set `GOOGLE_GENAI_USE_VERTEXAI=true` and configure project/region for the GenAI SDK.

6. **Azure derivation.** When `LIGHTSPEED_LLM_TYPE=AzureOpenAI`, the resolver MUST set `OPENAI_BASE_URL` from `LIGHTSPEED_AZURE_ENDPOINT` (if not already set), and `AZURE_OPENAI_API_VERSION` from `LIGHTSPEED_AZURE_API_VERSION` when present.

7. **URL override.** When `LIGHTSPEED_LLM_URL` is set, the resolver MUST map it to the appropriate SDK-specific URL var for the derived adapter (`ANTHROPIC_BASE_URL`, `OPENAI_BASE_URL`, etc.).

### Legacy direct-env-var behavior (backward compatibility)

8. **Provider selection.** Process environment variable `LIGHTSPEED_AGENT_PROVIDER` selects the backend. Supported logical values: `claude`, `gemini`, `openai`. Unknown values are rejected at startup when constructing the provider.

9. **Default provider.** When `LIGHTSPEED_AGENT_PROVIDER` is unset (and `LIGHTSPEED_LLM_TYPE` is also unset), the provider defaults to Claude.

10. **Model resolution — Claude.** Read `ANTHROPIC_MODEL`; if unset, use the package default model constant for Claude.

11. **Model resolution — Gemini.** Read `GEMINI_MODEL`; if unset, fall back to the same package default model constant used for Claude-branded defaults.

12. **Model resolution — OpenAI.** Read `OPENAI_MODEL`; if unset, fall back to that package default model constant.

13. **Router override.** Callers of the library `build_router` may pass an explicit `model` string; when provided, it overrides environment-based resolution for that router instance.

14. **Skills directory.** `LIGHTSPEED_SKILLS_DIR` sets the filesystem root for skills and provider `cwd`. Default when unset is the container default path under `/app`.

15. **Provider credentials.** API authentication uses the conventional env vars expected by each vendor SDK (Anthropic, Google/Gemini, OpenAI). The sandbox does not define alternate names beyond what adapters read for routing (e.g., Gemini API key fallbacks). Credentials are injected by the operator via `envFrom.secretRef` — the sandbox reads them as normal env vars.

16. **Vertex / Google GenAI (legacy).** `GOOGLE_GENAI_USE_VERTEXAI` toggles Vertex behavior for the Gemini adapter (tool composition rules per `provider-contract.md`).

17. **OpenAI base URL (legacy).** `OPENAI_BASE_URL` overrides the OpenAI client base URL when set.

18. **Claude via Vertex (legacy).** `CLAUDE_CODE_USE_VERTEX` gates Vertex-hosted Claude (consumed by the Claude agent SDK / Claude Code runtime). Project and region strings are read from `ANTHROPIC_VERTEX_PROJECT_ID` and `CLOUD_ML_REGION`.

### Runtime configuration

19. **Router defaults — `max_turns`.** The router supplies a built-in default maximum turn count to provider options when routes are registered (not exposed on `RunRequest`).

20. **Router defaults — `default_timeout_ms`.** The router supplies a built-in default milliseconds timeout for the run handler when `RunRequest.timeout_ms` is null.

21. **Process entry.** The container process invokes Uvicorn serving the FastAPI app on TCP port `8080` on all interfaces.

22. **Container filesystem layout.** A read-only skills mount path, a writable per-pod workspace path under system temp, and a writable home directory path for the non-root runtime user are provisioned with ownership for that UID.

23. **Python load path.** Runtime sets process environment so application source under `/app` and installed site-packages are on `PYTHONPATH` as defined in the image.

24. **Hermetic / Konflux build inputs.** Release images are built with network isolation after prefetch: per-architecture Python requirements files with hashes, RPM lockfile input, generic binary lockfile for oc/kubectl/ripgrep/dumb-init, and npm lockfile for the Claude Code CLI. Regeneration of those artifacts is via the project automation commands (see implementation notes in `how/provider-architecture.md`).

25. **Non-hermetic fallback.** When prefetch directories are absent, the container build recipe may fetch selected binaries from external URLs for developer builds.

26. **System packages — minimum expectations.** Runtime image includes Bash, Git, OpenShift CLI (`oc`), Kubernetes CLI (`kubectl`), ripgrep, Node.js (Claude Code CLI), and supporting OS utilities for debugging and archives per the container recipe.

## Configuration Surface

### Operator contract vars (set by operator, rule 1–7)

| Variable | Role |
|----------|------|
| `LIGHTSPEED_LLM_TYPE` | LLM hosting backend type. Triggers resolver (rule 1). |
| `LIGHTSPEED_MODEL` | Model identifier from `Agent.spec.model`. |
| `LIGHTSPEED_VERTEX_MODEL_PROVIDER` | Which SDK stack on Vertex (`Anthropic`, `Google`, `OpenAI`). |
| `LIGHTSPEED_GCP_PROJECT` | GCP project ID for Vertex. |
| `LIGHTSPEED_GCP_REGION` | GCP region for Vertex. |
| `LIGHTSPEED_AZURE_ENDPOINT` | Azure OpenAI resource endpoint. |
| `LIGHTSPEED_AZURE_API_VERSION` | Azure API version. |
| `LIGHTSPEED_AWS_REGION` | AWS region for Bedrock. |
| `LIGHTSPEED_LLM_URL` | Optional provider URL override. |

### Legacy / SDK-specific vars (direct use or derived by resolver)

| Variable | Role |
|----------|------|
| `LIGHTSPEED_AGENT_PROVIDER` | Selects agent backend (legacy, rule 8). |
| `ANTHROPIC_MODEL`, `GEMINI_MODEL`, `OPENAI_MODEL` | Per-provider model ID. |
| `LIGHTSPEED_SKILLS_DIR` | Skill root and provider working directory default. |
| `ANTHROPIC_API_KEY` | Claude SDK credential (when not using Vertex mode). |
| `GOOGLE_API_KEY`, `GEMINI_API_KEY` | Google GenAI credential for Gemini. |
| `OPENAI_API_KEY` | OpenAI SDK credential. |
| `GOOGLE_GENAI_USE_VERTEXAI` | Vertex mode for Gemini adapter. |
| `ANTHROPIC_VERTEX_PROJECT_ID`, `CLOUD_ML_REGION` | Vertex project/region for Claude via Vertex. |
| `CLAUDE_CODE_USE_VERTEX` | Enables Vertex-hosted Claude when set to sentinel value `1`. |
| `OPENAI_BASE_URL` | OpenAI-compatible endpoint override. |
| `build_router(..., skills_dir=..., model=..., max_turns=..., default_timeout_ms=...)` | Library-level defaults when embedding the router. |

## Constraints

- `RunRequest` does not carry provider name, model, max turns, or budget; changing those requires env vars, router constructor args, or future API extensions.
- Optional Python extras gate which provider SDKs are installed in a given environment; the image recipe installs all extras.

## Planned Changes

- [PLANNED: OLS-3153] **Env var resolver implementation**: Add startup resolver module that reads `LIGHTSPEED_*` operator contract vars and derives SDK-specific env vars before provider construction. Spec rules 1–7 describe the target behavior.
- TLS termination, mTLS, and network policies for operator-to-sandbox traffic. [PLANNED: OLS-3038–OLS-3043]
- Readiness endpoint (`GET /ready`). [PLANNED: OLS-3060]
- Konflux pipeline and lockfile policy updates as Red Hat platform requirements evolve. [PLANNED: OLS-2894]
