---
name: calculator
description: Perform arithmetic calculations and return verified results. The calc.sh script returns a verification token that must be included in your response.
allowed-tools: Bash(bash:*)
---

# Calculator Skill

Evaluate arithmetic expressions using the calculator script. The script returns a JSON object with the numeric result and a verification token.

## Usage

```bash
bash tools/calc.sh 'EXPRESSION'
```

## Output Format

The script returns JSON with the numeric result and a unique verification token.

## Examples

```bash
bash tools/calc.sh '2 + 3'
# Returns: {"result": 5, "token": "CALC_..."}
```
