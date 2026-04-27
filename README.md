# lightspeed-agentic-sandbox

Multi-provider agentic sandbox library for OpenShift Lightspeed.

See [CLAUDE.md](CLAUDE.md) for architecture and usage, and [evals/](evals/README.md) for the evaluation framework.

Local development uses `uv`. Run `make install` for dev dependencies,
`make install-all` for all providers plus evals, and `make lock` to refresh
`uv.lock` after dependency changes.
