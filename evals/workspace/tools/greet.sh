#!/bin/bash
# Usage: bash tools/greet.sh NAME
echo "Hello, ${1:-World}! The current time is $(date +%H:%M:%S)."
