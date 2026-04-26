#!/bin/bash
# Usage: bash tools/compute.sh EXPRESSION
EXPR="${*}"
TOKEN=$(head -c 8 /dev/urandom | base64 | tr -d '=/+')
RESULT=$(python3 -c "print(eval('${EXPR}'))" 2>&1)
if [ $? -eq 0 ]; then
    echo "COMPUTE_${TOKEN}" > .compute_token
    echo "{\"expression\": \"${EXPR}\", \"result\": ${RESULT}, \"status\": \"ok\", \"token\": \"COMPUTE_${TOKEN}\"}"
else
    echo "{\"expression\": \"${EXPR}\", \"error\": \"${RESULT}\", \"status\": \"error\"}"
fi
