---
name: lookup
description: Look up information from the local knowledge base. Use when asked to look up, find, or search for data. Returns structured data from local data files.
allowed-tools: Bash(bash:*) Bash(cat:*) Bash(grep:*) Bash(jq:*)
---

# Lookup Skill

Query local data files in the workspace for structured information.

## Usage

```bash
bash tools/lookup-data.sh QUERY_KEY
```

## Available Keys

- `status` -- returns system status (health, uptime, load)
- `version` -- returns version information (version, build, date)
- `config` -- returns configuration summary (env, region, replicas)

## Rules

1. Always use the lookup-data.sh script for lookups
2. Parse the JSON output with jq if needed
3. If the key is unknown, the script returns available keys
