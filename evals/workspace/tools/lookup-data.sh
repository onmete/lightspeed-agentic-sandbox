#!/bin/bash
# Usage: bash tools/lookup-data.sh KEY
case "${1:-}" in
    status)  echo '{"system": "healthy", "uptime": "72h", "load": 0.42}' ;;
    version) echo '{"version": "2.1.0", "build": "abc123", "date": "2026-04-20"}' ;;
    config)  echo '{"env": "production", "region": "us-east-1", "replicas": 3}' ;;
    *)       echo '{"error": "unknown key", "available": ["status", "version", "config"]}' ;;
esac
