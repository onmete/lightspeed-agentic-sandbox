#!/bin/bash
# Usage: bash tools/calc.sh 'EXPRESSION'
TOKEN=$(head -c 8 /dev/urandom | base64 | tr -d '=/+')
RESULT=$(python3 -c "print(eval('${1}'))")
echo "CALC_${TOKEN}" > .calc_token
echo "{\"result\": ${RESULT}, \"token\": \"CALC_${TOKEN}\"}"
