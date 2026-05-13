"""Small synthetic JSON Schema dicts for structured-output contract tests."""

from __future__ import annotations

from typing import Any

FLAT_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "summary": {"type": "string"},
        "ticketId": {"type": "string"},
    },
    "required": ["success", "summary", "ticketId"],
}

NESTED_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "summary": {"type": "string"},
        "items": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["name", "count"],
            },
        },
    },
    "required": ["success", "summary", "items"],
}

ECHO_TOKEN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "summary": {"type": "string"},
        "token": {"type": "string", "minLength": 1},
        "status": {"type": "string", "enum": ["ok"]},
    },
    "required": ["success", "summary", "token", "status"],
}

# Strict schema used with a prompt that encourages invalid / non-JSON output.
STRICT_CONFLICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "success": {"type": "boolean"},
        "summary": {"type": "string"},
        "onlyFieldAlpha": {"type": "string", "const": "alpha"},
        "onlyFieldBeta": {"type": "integer", "const": 42},
    },
    "required": ["success", "summary", "onlyFieldAlpha", "onlyFieldBeta"],
}
