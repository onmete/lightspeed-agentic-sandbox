---
name: formatter
description: Format data into different output formats. Use when asked to format, convert, or transform data between JSON, YAML, CSV, or plain text.
allowed-tools: Bash(python3:*) Bash(echo:*)
---

# Formatter Skill

Transform data between formats using Python.

## Usage

```bash
# Pretty-print JSON
echo '{"key": "value"}' | python3 -c "import json,sys; data=json.load(sys.stdin); print(json.dumps(data, indent=2))"

# JSON to key=value
echo '{"a": 1, "b": 2}' | python3 -c "import json,sys; [print(f'{k}={v}') for k,v in json.load(sys.stdin).items()]"
```

## Rules

1. Use python3 for all formatting operations
2. Output the formatted result to stdout
