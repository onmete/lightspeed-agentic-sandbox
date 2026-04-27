# Configuration

How the service resolves its runtime settings.

## Provider selection

The `LIGHTSPEED_AGENT_PROVIDER` environment variable
selects which SDK adapter to use. Defaults to `claude` if
unset.

Accepted values: `claude`, `gemini`, `openai`, `deepagents`,
`deepagents-gemini`, `deepagents-openai`.

The `deepagents-*` variants are aliases — they all
instantiate the same Deep Agents adapter. The distinct names
exist for logging and operational clarity, not for different
behavior.

## Model resolution

The model used for a request follows a resolution chain:

1. Explicit `model` parameter passed to the router builder
2. Provider-specific environment variable
   (`ANTHROPIC_MODEL`, `GEMINI_MODEL`, `OPENAI_MODEL`,
   `DEEPAGENTS_MODEL`)
3. Fallback: the global default model (`claude-opus-4-6`)

**Known quirk:** the fallback model is a Claude model
regardless of which provider is active. If you run the
Gemini provider without setting `GEMINI_MODEL`, it will
attempt to use a Claude model name. Operators must set the
provider-specific env var for non-Claude providers.

**Chat vs query inconsistency:** query routes resolve the
model per-provider using the env var chain above. Chat
receives a pre-resolved model from the router builder,
which defaults to the global fallback if no explicit model
is passed. In practice this means chat always uses the
global default unless the deployer provides an explicit
model. This is a known gap, not a bug per se — it works
correctly in the default Claude configuration.

## Skills directory resolution

The skills directory (`LIGHTSPEED_SKILLS_DIR`, default
`/app/skills`) is where the agent finds its domain-knowledge
skill files. The resolution logic tries subdirectories in
order:

1. `{dir}/skills/` — standard deployment layout
2. `{dir}/.claude/skills/` — Claude SDK convention
3. `{dir}` itself — flat layout fallback

This supports multiple packaging conventions without
requiring configuration changes.

**Note:** Claude does not use this resolution chain. It
receives the raw skills directory as its working directory
and relies on its SDK's native skill discovery. The
subdirectory fallback only applies to Gemini, OpenAI, and
Deep Agents.

## Timeouts

| Setting | Default | Used by |
|---|---|---|
| Analysis timeout | 5 minutes (300,000 ms) | analyze, verify |
| Execution timeout | 10 minutes (600,000 ms) | execute |
| Chat timeout | 2 minutes (120,000 ms) | chat (declared but not actively enforced in the stream) |

Verification intentionally shares the analysis timeout — see
query-api.md for rationale.

The chat timeout is passed to the route but the streaming
handler does not currently wrap the stream in a timeout
guard. The timeout path exists for error handling but is
effectively dead unless the provider itself raises a timeout.
This is a known gap.

## Container defaults

The production container runs as user 1001 (non-root) and
bundles `oc`, `kubectl`, `git`, and `rg` alongside the
Python runtime. Skills are mounted at `/app/skills`.

## Default allowed tools

The agent is given a fixed set of tool capabilities:
Bash, Read, Glob, Grep, and Skill. This list is shared
across all providers and both query and chat routes.
Individual providers wire these into their SDK's native tool
mechanism.
