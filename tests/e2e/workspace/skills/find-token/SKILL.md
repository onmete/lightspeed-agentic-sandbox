---
name: find-token
description: Find the hidden verification token. Run the find-token script to retrieve unique DIAG and VERIFY tokens.
allowed-tools: Bash(bash:*)
---

# Find Token Skill

Retrieve hidden verification tokens by running the find-token script.

## Usage

Run from the skills root (`LIGHTSPEED_SKILLS_DIR`):

```bash
bash find-token/scripts/find-token.sh
```

## Output

The script prints a full structured analysis JSON object (actionRequired, options with
remediationPlan.actions, components with DIAG_/VERIFY_ tokens). The component token
values are the verification result; do not read a separate token file. Use that JSON
as the basis for your structured response.
