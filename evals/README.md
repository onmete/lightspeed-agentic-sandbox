# Evals

End-to-end evaluations that run real API calls against all providers. Each test is independent and runs in parallel.

## Contents

- [Quick Start](#quick-start)
- [Providers & Models](#providers--models)
- [Test Categories](#test-categories)
- [Credentials](#credentials)
- [Running Evals](#running-evals)
- [Reports](#reports)
- [Container Evals](#container-evals)
- [Adding Tests](#adding-tests)

## Quick Start

```bash
make install-all    # install all provider SDKs + eval deps
make eval           # run all evals (skips providers without credentials)
```

## Providers & Models

| Provider | Default Model | Override Env Var |
|---|---|---|
| `claude` | `claude-sonnet-4-6` | `ANTHROPIC_MODEL` |
| `gemini` | `gemini-3.1-pro-preview` | `GEMINI_MODEL` |
| `openai` | `gpt-5.4` | `OPENAI_MODEL` |
| `deepagents` | `claude-opus-4-6` | `DEEPAGENTS_MODEL` |
| `deepagents-gemini` | `gemini-3.1-pro-preview` | `DEEPAGENTS_GEMINI_MODEL` |
| `deepagents-openai` | `gpt-5.4` | `DEEPAGENTS_OPENAI_MODEL` |

The `deepagents-*` variants run the same deepagents provider (langchain) with different LLM backends.

## Test Categories

10 tests per provider, 60 total:

| Category | Tests | What it validates |
|---|---|---|
| **Basic Query** | `test_basic_response`, `test_cost_tracking` | Prompt/response sanity and token usage reporting |
| **Structured Output** | `test_analysis_schema`, `test_calculation_schema`, `test_schema_with_enum` | JSON schema enforcement — nested objects, required fields, enum constraints |
| **Skill Invocation** | `test_calculator_skill`, `test_lookup_skill` | Model discovers and uses skills from `workspace/skills/` |
| **Tool Usage** | `test_greet_tool`, `test_compute_tool_with_structured_output`, `test_lookup_data_tool` | Model invokes bash scripts from `workspace/tools/` and uses their output |

## Credentials

Providers without valid credentials are automatically skipped. Credential detection order per provider:

| Provider | Primary | Fallbacks |
|---|---|---|
| `claude` | `ANTHROPIC_API_KEY` | Vertex AI (`CLAUDE_CODE_USE_VERTEX=1` + gcloud ADC), Bedrock (`CLAUDE_CODE_USE_BEDROCK=1` + AWS creds) |
| `gemini` | `GOOGLE_API_KEY` | `GEMINI_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS` file, gcloud ADC |
| `openai` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` (keyless endpoints) |
| `deepagents` | Depends on model | Checks credentials matching the configured `DEEPAGENTS_MODEL` |
| `deepagents-gemini` | Same as `gemini` | — |
| `deepagents-openai` | Same as `openai` | — |

## Running Evals

All evals run in parallel by default (`pytest-xdist -n auto`). Each test gets an isolated workspace copy.

```bash
# All providers
make eval

# Single provider
pytest evals/ -k claude

# Single test category
pytest evals/ -k structured_output

# Single test + single provider
pytest evals/test_tool_usage.py::test_greet_tool -k gemini

# Override model for a run
ANTHROPIC_MODEL=claude-opus-4-6 pytest evals/ -k claude

# Sequential with stdout (debugging)
pytest evals/ -n0 -s

# Verbose with per-test timing
pytest evals/ -v --durations=0
```

## Reports

Generate a JSON report at `evals/report.json`:

```bash
make eval-report
# or
pytest evals/ --eval-report=evals/report.json
```

## Container Evals

Run evals against the production container image:

```bash
make eval-container
```

This builds the image, mounts `evals/`, and runs `pytest` inside the container. Credentials are forwarded via environment variables.

## Adding Tests

Tests live in 4 files matching the categories above:

```
evals/
├── test_basic_query.py
├── test_skill_invocation.py
├── test_structured_output.py
├── test_tool_usage.py
└── workspace/
    ├── skills/          # dummy skills (SKILL.md files)
    └── tools/           # bash scripts the model invokes
```

Each test receives a `provider_name`, `default_model`, `eval_workspace`, and `eval_runner` fixture. The `eval_runner` handles provider setup, event streaming, and result collection. Tests are parametrized across all providers automatically.
