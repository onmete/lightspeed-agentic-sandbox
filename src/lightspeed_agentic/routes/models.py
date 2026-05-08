"""Pydantic request/response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class RunRequest(BaseModel):
    query: str
    systemPrompt: str | None = None
    outputSchema: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    timeout_ms: int | None = None


class RunResponse(BaseModel, extra="allow"):
    success: bool
    summary: str
