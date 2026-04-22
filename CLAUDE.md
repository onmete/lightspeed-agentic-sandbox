# Lightspeed Agentic Sandbox

Multi-provider agentic sandbox library for OpenShift Lightspeed.

Python port of `lightspeed-agent/src/providers/` (TypeScript). Same interface,
same event types, same tool/skill behavior — different language.

## Architecture

```
src/lightspeed_agentic/
├── types.py              # ProviderEvent (6 types), ProviderQueryOptions, AgentProvider ABC
├── factory.py            # create_provider("claude"|"gemini"|"openai")
├── tools.py              # Shared executors (bash/read/write/glob/grep) for non-Claude providers
├── logging.py            # Normalized event logging
└── providers/
    ├── claude.py          # claude-agent-sdk — SDK handles tools+skills natively
    ├── gemini.py          # google-adk — SkillToolset + plain function tools
    └── openai.py          # openai-agents — ShellTool + ShellToolLocalSkill
```

## Provider SDK Mapping

| Feature | Claude (`claude-agent-sdk`) | Gemini (`google-adk`) | OpenAI (`openai-agents`) |
|---------|---------------------------|----------------------|------------------------|
| Tools | Built-in (Bash, Read, etc.) | Plain callables (auto-wrapped) | ShellTool + ShellExecutor |
| Skills | Native `skills=` param | Native `SkillToolset` | Native `ShellToolLocalSkill` |
| Structured output | `output_format` | Schema in prompt (output_schema disables tools) | `output_type` Pydantic model |
| Streaming | `include_partial_messages` | `StreamingMode.SSE` | `Runner.run_streamed()` |

## Key Design Decisions

- **No manual SKILL.md parsing** — each SDK handles skill discovery natively
- **No ToolDescriptor abstraction** — each provider uses its SDK's native tool format
- **tools.py only has raw executors** — Gemini wraps them as callables, OpenAI wraps via ShellExecutor
- **Claude provider is the thinnest** — SDK does everything

## Environment Variables

- `LIGHTSPEED_AGENT_PROVIDER` — `claude` (default), `gemini`, or `openai`
- `ANTHROPIC_MODEL` — model for Claude (default: `claude-opus-4-6`)
- `GEMINI_MODEL` — model for Gemini (default: `gemini-2.5-flash`)
- `OPENAI_MODEL` — model for OpenAI (default: `gpt-4.1`)

## Usage

```python
from lightspeed_agentic import create_provider, ProviderQueryOptions

provider = create_provider("claude")  # or "gemini" or "openai"

async for event in provider.query(ProviderQueryOptions(
    prompt="Diagnose the failing pod",
    system_prompt="You are an SRE agent.",
    model="claude-opus-4-6",
    max_turns=50,
    max_budget_usd=5.0,
    allowed_tools=["Bash", "Read", "Glob", "Grep", "Skill"],
    cwd="/app/skills",
)):
    match event.type:
        case "text_delta": print(event.text, end="")
        case "result": print(f"\nCost: ${event.cost_usd:.4f}")
```

## Integration with lightspeed-service

```toml
# In lightspeed-service/pyproject.toml
dependencies = [
    "lightspeed-agentic[all] @ git+https://github.com/harche/lightspeed-agentic-sandbox",
]
```

```python
# In lightspeed-service router
from lightspeed_agentic import create_provider
provider = create_provider()
```
