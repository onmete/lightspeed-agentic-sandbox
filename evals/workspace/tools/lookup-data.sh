#!/bin/bash
# Usage: bash tools/lookup-data.sh KEY
case "${1:-}" in
    status)  echo '{"system": "healthy", "uptime": "72h", "load": 0.42}' ;;
    version) TOKEN=$(head -c 8 /dev/urandom | base64 | tr -d '=/+'); echo "LOOKUP_${TOKEN}" > .lookup_token; echo "{\"version\": \"2.1.0\", \"build\": \"abc123\", \"date\": \"2026-04-20\", \"token\": \"LOOKUP_${TOKEN}\"}" ;;
    config)  echo '{"env": "production", "region": "us-east-1", "replicas": 3}' ;;
    *)       echo '{"error": "unknown key", "available": ["status", "version", "config"]}' ;;
esac
