#!/bin/bash
# Usage: bash tools/compute.sh EXPRESSION
# Evaluates a simple arithmetic expression and returns JSON.
EXPR="${*}"
RESULT=$(python3 -c "print(eval('${EXPR}'))" 2>&1)
if [ $? -eq 0 ]; then
    echo "{\"expression\": \"${EXPR}\", \"result\": ${RESULT}, \"status\": \"ok\"}"
else
    echo "{\"expression\": \"${EXPR}\", \"error\": \"${RESULT}\", \"status\": \"error\"}"
fi
