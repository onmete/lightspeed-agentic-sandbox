"""Serialize OpenTelemetry GenAI v1.41 message values for scalar span attributes."""

import json
from typing import Any


def encode_messages(messages: list[dict[str, Any]]) -> str:
    """Keep the provider's message and part order intact in compact Unicode JSON."""
    return json.dumps(messages, ensure_ascii=False, separators=(",", ":"))


def _reject_nonfinite_constant(_constant: str) -> None:
    raise ValueError("Non-finite values are not JSON")


def encode_tool_object(value: Any) -> dict[str, Any]:
    """Normalize a tool span attribute to a JSON object without changing message parts."""
    if isinstance(value, str):
        try:
            value = json.loads(value, parse_constant=_reject_nonfinite_constant)
        except ValueError:
            return {"content": value}
    return value if isinstance(value, dict) else {"content": value}
