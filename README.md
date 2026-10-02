# lightspeed-agentic-sandbox

Multi-provider agentic sandbox library for OpenShift Lightspeed.

See [ARCHITECTURE.md](ARCHITECTURE.md) for architecture and
[AGENTS.md](AGENTS.md) for development guidance.

Local development uses `uv`. Run `make install` for dev dependencies,
`make install-all` for all providers plus e2e extras, and `make lock` to refresh
`uv.lock` after dependency changes.

## Bumping Dependencies

The release image uses RHOAI 3.5 and is built hermetically in Konflux.
Local development uses `uv.lock`; Konflux resolves independently from
`pyproject.toml` against the RHOAI and PyPI indexes, so versions can differ.

Requirements regeneration needs Podman or Docker, access to
`quay.io/syedriko/uv:prefer-index`, Python 3.12+ with `packaging`, and
`pybuild-deps` on `PATH`. RPM regeneration also needs a Red Hat subscription
with `ACTIVATION_KEY` and `ORG_ID` set.

```bash
make bump-deps          # upgrade uv.lock + regenerate .konflux/requirements.* and Tekton package lists
make rpm-lockfile       # regenerate .konflux/rpms.lock.yaml
make verify-hermetic-requirements  # check package coverage; version skew is a warning
```

Review and commit all changed lockfiles, requirements, and Tekton files together.
See [AGENTS.md](AGENTS.md#konflux-hermetic-builds) for the build inputs,
prerequisites, and instructions for adding dependencies.
