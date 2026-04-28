---
name: find-token
description: Find the hidden verification token. Run the find-token script to retrieve a unique token.
allowed-tools: Bash(bash:*)
---

# Find Token Skill

Retrieve a hidden verification token by running the find-token script.

## Usage

```bash
bash tools/find-token.sh
```

## Output

The script returns JSON with a unique token:
```json
{"token": "TOKEN_..."}
```
