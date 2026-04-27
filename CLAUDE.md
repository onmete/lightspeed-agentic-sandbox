# Lightspeed Agentic Sandbox

Multi-provider agentic sandbox for OpenShift Lightspeed. This repo exposes a
FastAPI app plus provider adapters for Claude, Gemini, OpenAI, and Deep Agents.
When editing it, optimize for thin provider wrappers, consistent event mapping,
and tests that stay offline unless you are intentionally running containerized
evals.

## Quick Commands

```bash
make venv                              # create .venv
make install-all                       # install all providers + dev + eval extras
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
- Evals are container-only. If you change eval workspace fixtures, skills, or
  mounted tool behavior, verify the corresponding assumptions in `evals/run.sh`.

## What To Avoid

- Do not add top-level imports of provider SDK packages in `src/lightspeed_agentic/providers/`.
- Do not make unit tests hit real model APIs. Live coverage belongs in `evals/`.
- Do not edit `evals/workspace/skills` or `evals/workspace/tools` without
  checking how `evals/run.sh` copies and mounts them.
- Do not add `uvicorn` to non-dev dependencies in `pyproject.toml`. The package
  exposes an ASGI app; process management is a deployment concern.
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
