#!/bin/bash
# Usage: bash tools/greet.sh NAME
TOKEN=$(head -c 8 /dev/urandom | base64 | tr -d '=/+')
echo "GREET_${TOKEN}" > .greet_token
echo "Hello, ${1:-World}! Verification: GREET_${TOKEN}"
