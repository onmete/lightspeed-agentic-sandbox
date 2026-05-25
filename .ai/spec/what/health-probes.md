# Behavioral spec: health probes

Origin: [OLS-3058](https://redhat.atlassian.net/browse/OLS-3058) — sandbox failure modes audit; [OLS-3060](https://redhat.atlassian.net/browse/OLS-3060) — `/readiness` implementation.

Cross-references: `configuration.md` (env vars, port), `run-api.md` (route mounting).

## Principles

Sandbox pods are ephemeral one-shot workers. Probes confirm the pod can accept work; all real failures surface on `POST /v1/agent/run` where the operator handles them. Probes MUST NOT make authenticated API calls or spend tokens.

## Endpoints

### `GET /health` (liveness)

Existing endpoint, unchanged. Returns `{"status": "ok"}` if uvicorn is alive. No subsystem checks.

### `GET /readiness` (readiness)

Returns HTTP 200 when all checks pass, HTTP 503 when any check fails. Not under `/v1/agent`.

**Healthy:**
```json
{
  "status": "ok",
  "checks": {
    "skills_dir": "ok",
    "provider_credentials": "ok",
    "provider_sdk": "ok",
    "mcp_servers": "skipped: not configured"
  }
}
```

**Unhealthy:**
```json
{
  "status": "error",
  "checks": {
    "skills_dir": "error: not a directory (/app/skills)",
    "provider_credentials": "ok",
    "provider_sdk": "ok",
    "mcp_servers": "skipped: not configured"
  }
}
```

Each check value is either `ok`, `skipped: …` (optional subsystem not configured), or `error: …` with a short reason.

## Readiness Checks

**skills_dir** — Skills mount exists, is a directory, and contains at least one non-hidden entry.

**provider_credentials** — Required credential env var(s) for `LIGHTSPEED_AGENT_PROVIDER` are set and non-empty. Does NOT validate key value.

| Provider | Required env var(s) |
|----------|-------------------|
| `claude` (default API) | `ANTHROPIC_API_KEY` |
| `claude` (Vertex, `CLAUDE_CODE_USE_VERTEX=1`) | `ANTHROPIC_VERTEX_PROJECT_ID` |
| `gemini` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| `openai` | `OPENAI_API_KEY` |

**provider_sdk** — Provider adapter module and vendor SDK import without error for the configured provider.

**mcp_servers** — When `LIGHTSPEED_MCP_SERVERS` lists comma-separated HTTP(S) URLs, each endpoint gets an unauthenticated GET with a 3-second timeout. Any HTTP response (including 4xx) counts as reachable. When unset or empty, reports `skipped: not configured`.

## Caching

Results are cached in-process for `LIGHTSPEED_READINESS_CACHE_TTL_SECONDS` (default 30). Set to `0` to disable caching (tests). Cache avoids repeated MCP probes when Kubernetes calls readiness frequently.

## Recommended Probe Config

```yaml
livenessProbe:
  httpGet: { path: /health, port: 8080 }
  periodSeconds: 10
  timeoutSeconds: 3
  failureThreshold: 3

readinessProbe:
  httpGet: { path: /readiness, port: 8080 }
  initialDelaySeconds: 2
  periodSeconds: 5
  timeoutSeconds: 5
  failureThreshold: 2
```

## Out of Scope for Probes

Credential validity against vendor APIs, skills content quality, model availability, tool execution — all caught by `/run` and handled by the operator.
