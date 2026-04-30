# Lightspeed Agentic Sandbox

Multi-provider agentic sandbox for OpenShift Lightspeed. This repo exposes a
FastAPI app plus provider adapters for Claude, Gemini, OpenAI, and Deep Agents.
When editing it, optimize for thin provider wrappers, consistent event mapping,
and tests that stay offline unless you are intentionally running containerized
evals.

## Behavioral Specs

Before changing code, read the relevant spec in `.ai/spec/`:

| Working on | Read |
|---|---|
| Provider adapters | [provider-contract.md](.ai/spec/provider-contract.md) |
| Query routes (analyze/execute/verify) | [query-api.md](.ai/spec/query-api.md) |
| Chat endpoint or SSE streaming | [chat-api.md](.ai/spec/chat-api.md) |
| Deployment, env vars, or defaults | [configuration.md](.ai/spec/configuration.md) |

Specs capture invariants, design decisions, and known quirks that the code
cannot express about itself. The code and this file cover the "how" — specs
cover the "why" and the "must."

## Quick Commands

```bash
make install                           # create/update .venv with dev dependencies via uv
make install-all                       # install all providers + dev + eval extras
make lock                              # refresh uv.lock after dependency changes
make test                              # unit tests only; mocked providers, no API calls
make lint                              # ruff check src/ tests/ evals/
make format                            # ruff format + autofix
make eval                              # build image and run live evals in containers
make eval EVAL_ARGS="-k claude"        # run a subset of evals
make eval-report                       # write evals/report.json
```

## Architecture

```text
src/lightspeed_agentic/
├── app.py                # FastAPI entry point, mounts router at /v1/agent
├── factory.py            # create_provider(...) using LIGHTSPEED_AGENT_PROVIDER
├── logging.py            # Event logging helpers for query flows
├── tools.py              # Shared tool/skill utilities and defaults
├── types.py              # Provider events, query options, AgentProvider ABC
├── providers/
│   ├── claude.py         # claude-agent-sdk adapter
│   ├── gemini.py         # google-adk adapter
│   ├── openai.py         # openai-agents adapter
│   └── deepagents.py     # deepagents/langgraph adapter
└── routes/
    ├── __init__.py       # build_router(...)
    ├── query.py          # /analyze, /execute, /verify
    ├── chat.py           # /chat SSE endpoint
    └── models.py         # Pydantic request/response models
```

| Feature | Claude (`claude-agent-sdk`) | Gemini (`google-adk`) | OpenAI (`openai-agents`) | Deep Agents (`deepagents`) |
| --- | --- | --- | --- | --- |
| Tools | Built-in SDK tools | Native `ExecuteBashTool` plus built-in web tools | Native `SandboxAgent` shell/filesystem/skills | Built-in local shell and file tools |
| Skills | Native `skills="all"` | Native `SkillToolset` | Native `Skills` capability | Native skills middleware |
| Structured output | `output_format` JSON schema | Native response schema path | `output_type` wrapper | `response_format` / generated Pydantic model |
| Streaming | Partial message stream events | `StreamingMode.SSE` | `Runner.run_streamed()` | LangGraph async stream |

Keep provider adapters thin. The SDK should own tool execution and skill
discovery; shared path logic belongs in `tools.py`, not in duplicated provider
helpers.

## Code Conventions

- Keep provider SDK imports inside methods or narrow helpers in provider modules.
  These SDKs are optional extras, so top-level imports must not break the base
  package import path.
- `types.py` event objects are frozen dataclasses. New event types should follow
  the same pattern and stay simple to serialize/log.
- Route payloads use Pydantic models in `routes/models.py` and `routes/chat.py`.
  Prefer extending those models over passing around untyped dicts.
- Streaming paths are async all the way through: providers yield async event
  streams, query handlers consume async iterators, and chat emits SSE via an
  async generator.
- Preserve the "thin adapter" shape when touching provider files: map SDK
  events into `ProviderEvent`, do not re-implement SDK behavior locally unless a
  testable workaround is required.

## Testing Conventions

- `make test` is the default verification path for code changes. Unit tests use
  mocked providers and must not require live credentials.
- Put reusable fake providers and event fixtures in `tests/conftest.py`.
  Prefer exercising real route/provider glue over deep mocking of SDK internals.
- Route tests should build a FastAPI app with `build_router(...)` and use
  `httpx.AsyncClient` plus `ASGITransport`.
- `make eval` and `make eval-report` are integration-only checks. They build the
  container image, start one container per provider, and run evals against live
  `/v1/agent/analyze` endpoints.
- See [`evals/README.md`](evals/README.md) for eval setup, credential handling,
  provider coverage, and report details.
- Evals are container-only. If you change eval workspace fixtures, skills, or
  mounted tool behavior, verify the corresponding assumptions in `evals/run.sh`.

## Konflux Hermetic Builds

The container image is built in [Konflux](https://konflux-ci.dev/) with hermetic
builds enabled — all dependencies are prefetched and verified before the build
starts, with no network access during the build itself.

### Dependency files

| File | Purpose | How to regenerate |
|---|---|---|
| `requirements.x86_64.txt` | Python deps with hashes (x86_64) | `make requirements` |
| `requirements.aarch64.txt` | Python deps with hashes (aarch64) | `make requirements` |
| `requirements-build.txt` | Build-time deps (empty — we use wheels) | N/A |
| `rpms.in.yaml` | System RPM package list | Edit manually |
| `rpms.lock.yaml` | Resolved RPM lockfile | `make rpm-lockfile` |
| `ubi.repo` | UBI 9 repo definitions for RPM resolution | Rarely changes |
| `artifacts.lock.yaml` | External binaries (oc, ripgrep, dumb-init) | Edit manually, update checksums |
| `package.json` / `package-lock.json` | npm deps (claude-code CLI) | `npm install --package-lock-only` |

### Bumping dependencies

```bash
make bump-deps          # upgrade uv.lock + regenerate requirements.{arch}.txt
make rpm-lockfile       # regenerate rpms.lock.yaml (needs podman)
npm update              # update package-lock.json for claude-code
```

After bumping, commit all changed lockfiles and requirements files together.
The Konflux pipeline will prefetch the new versions on the next PR.

### Adding a new system package

1. Add the package name to `rpms.in.yaml`
2. Run `make rpm-lockfile` to regenerate `rpms.lock.yaml`
3. Add the `dnf install` line to the appropriate section in `Containerfile`

### Adding a new external binary

1. Add an entry to `artifacts.lock.yaml` with the download URL, checksum, and
   filename (per-arch if needed)
2. Add the install logic to the generic-fetcher section in `Containerfile`

## What To Avoid

- Do not add top-level imports of provider SDK packages in `src/lightspeed_agentic/providers/`.
- Do not make unit tests hit real model APIs. Live coverage belongs in `evals/`.
- Do not edit `evals/workspace/skills` or `evals/workspace/tools` without
  checking how `evals/run.sh` copies and mounts them.
- Keep runtime dependencies aligned with the shipped container entrypoint. If
  the image invokes a module directly, declare it in `pyproject.toml`.
- Do not turn this file back into a long-form architecture tutorial. It should
  stay focused on how an agent works in this repo.

## Environment Variables

| Variable | Purpose |
| --- | --- |
| `LIGHTSPEED_AGENT_PROVIDER` | Default provider selected by `create_provider()` (`claude`, `gemini`, `openai`, `deepagents`) |
| `LIGHTSPEED_SKILLS_DIR` | Skills root mounted by the FastAPI app, default `/app/skills` |
| `ANTHROPIC_MODEL` | Default Claude model for query routes |
| `GEMINI_MODEL` | Default Gemini model for query routes |
| `OPENAI_MODEL` | Default OpenAI model for query routes |
| `DEEPAGENTS_MODEL` | Default Deep Agents model for query routes |
| `OPENAI_BASE_URL` | Optional OpenAI-compatible endpoint override |
| `CLAUDE_CODE_USE_VERTEX` | When set to `1`, Deep Agents Claude models use Vertex-backed Anthropic |
| `ANTHROPIC_VERTEX_PROJECT_ID` | Vertex project for Claude via Deep Agents |
| `CLOUD_ML_REGION` | Vertex region for Claude via Deep Agents (default `us-east5`) |
| `EVAL_SERVER_URLS` | Provider-to-URL map exported by `evals/run.sh` for eval pytest fixtures |
| `EVAL_WORKSPACES` | Provider-to-output-workspace map exported by `evals/run.sh` for eval pytest fixtures |

Provider credentials such as `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`,
`GEMINI_API_KEY`, and `OPENAI_API_KEY` are expected by the underlying SDKs or
passed through by the eval container launcher.
