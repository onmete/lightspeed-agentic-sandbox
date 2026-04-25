"""JSON Schema definitions for structured output eval tests."""

from __future__ import annotations

from typing import Any

ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "summary": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["category", "description"],
            },
        },
    },
    "required": ["success", "summary", "confidence", "findings"],
}

CALCULATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "expression": {"type": "string"},
        "result": {"type": "number"},
        "method": {"type": "string"},
    },
    "required": ["expression", "result"],
}

TOOL_USAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tools_used": {
            "type": "array",
            "items": {"type": "string"},
        },
        "output": {"type": "string"},
        "success": {"type": "boolean"},
    },
    "required": ["tools_used", "output", "success"],
}
