"""Pydantic request/response models — maps to agent.ts types."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class PreviousAttempt(BaseModel):
    attempt: int
    failureReason: str | None = None


class ApprovedAction(BaseModel):
    type: str
    description: str


class ApprovedDiagnosis(BaseModel):
    summary: str
    confidence: str
    rootCause: str


class ApprovedProposal(BaseModel):
    description: str
    actions: list[ApprovedAction] = []
    risk: str
    reversible: bool


class ApprovedOption(BaseModel):
    title: str
    summary: str | None = None
    diagnosis: ApprovedDiagnosis
    proposal: ApprovedProposal
    verification: dict[str, Any] | None = None
    rbac: dict[str, Any] | None = None


class QueryContext(BaseModel):
    targetNamespaces: list[str] | None = None
    attempt: int | None = None
    previousAttempts: list[PreviousAttempt] | None = None
    approvedOption: ApprovedOption | None = None


class QueryRequest(BaseModel):
    query: str
    systemPrompt: str | None = None
    outputSchema: dict[str, Any] | None = None
    context: QueryContext | None = None


class QueryResponse(BaseModel, extra="allow"):
    success: bool
    summary: str


class RunRequest(BaseModel):
    """Step-agnostic request for POST /run.

    The caller (operator) controls behaviour entirely through the payload —
    the agent has no awareness of which workflow step it is serving.
    """

    query: str
    systemPrompt: str | None = None
    outputSchema: dict[str, Any] | None = None
    context: QueryContext | None = None
    timeout_ms: int | None = None


class RunResponse(BaseModel, extra="allow"):
    """Response from POST /run — same shape as QueryResponse."""

    success: bool
    summary: str
