---
name: calculator
description: Perform arithmetic calculations. Use when asked to compute, calculate, or do math. Supports basic operations (add, subtract, multiply, divide) and expressions.
allowed-tools: Bash(python3:*) Bash(echo:*)
---

# Calculator Skill

Evaluate arithmetic expressions using Python.

## Usage

```bash
python3 -c "print(eval('EXPRESSION'))"
```

## Examples

```bash
# Addition
python3 -c "print(eval('2 + 3'))"

# Complex expression
python3 -c "print(eval('(10 * 5) + 3'))"
```

## Rules

1. Only use basic arithmetic operators: +, -, *, /, //, %, **
2. Always use python3 for calculation
3. Print only the numeric result
