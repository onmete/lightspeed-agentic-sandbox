"""Generic query endpoints — maps to lightspeed-agent/src/agent.ts.

POST /analyze, /execute, /verify all use the same handler with different
phase labels. The operator sends {query, systemPrompt, outputSchema, context}
and the agent runs the LLM and returns {success, summary, ...structured fields}.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

from fastapi import APIRouter

from lightspeed_agentic.logging import log_provider_event
from lightspeed_agentic.routes.models import QueryRequest, QueryResponse
from lightspeed_agentic.types import AgentProvider, ProviderQueryOptions

logger = logging.getLogger("lightspeed_agentic")


def _format_context_prefix(context: dict) -> str:
    lines: list[str] = ["[context]"]

    if ns := context.get("targetNamespaces"):
        lines.append(f"Target namespaces: {', '.join(ns)}")
    if (attempt := context.get("attempt")) is not None:
        lines.append(f"Attempt: {attempt} of max")
    if prev := context.get("previousAttempts"):
        lines.append("Previous attempts:")
        for p in prev:
            reason = f": {p['failureReason']}" if p.get("failureReason") else ""
            lines.append(f"  Attempt {p['attempt']}{reason}")
    if opt := context.get("approvedOption"):
        lines.append("")
        lines.append("=== APPROVED REMEDIATION (execute ONLY these actions) ===")
        lines.append(f"Title: {opt['title']}")
        lines.append(f"Diagnosis: {opt['diagnosis']['rootCause']}")
        lines.append(f"Plan: {opt['proposal']['description']}")
        lines.append(f"Risk: {opt['proposal']['risk']}, Reversible: {opt['proposal']['reversible']}")
        if actions := opt["proposal"].get("actions"):
            lines.append("Actions to execute:")
            for action in actions:
                lines.append(f"  - [{action['type']}] {action['description']}")
        lines.append("=== DO NOT perform any actions beyond what is listed above ===")
        lines.append("")

    lines.append("[/context]")
    return "\n".join(lines)


async def _handle_query(
    req: QueryRequest,
    phase: str,
    provider: AgentProvider,
    skills_dir: str,
    model: str,
    max_turns: int,
    timeout_ms: int,
) -> QueryResponse:
    system_prompt = req.systemPrompt or f"You are an AI agent operating in {phase} mode."

    prompt = req.query
    if req.context:
        prefix = _format_context_prefix(req.context.model_dump(exclude_none=True))
        prompt = f"{prefix}\n\n{req.query}"

    logger.info("[agent] Starting %s query (model=%s, provider=%s)", phase, model, provider.name)

    try:
        result = provider.query(ProviderQueryOptions(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            max_turns=max_turns,
            max_budget_usd=5.0,
            allowed_tools=["Bash", "Read", "Glob", "Grep", "Skill"],
            cwd=skills_dir,
            output_schema=req.outputSchema,
        ))

        text = ""
        cost = 0.0
        tokens = 0

        async def run() -> None:
            nonlocal text, cost, tokens
            async for event in result:
                log_provider_event(phase, event)
                if event.type == "result":
                    text = event.text
                    cost = event.cost_usd
                    tokens = event.input_tokens + event.output_tokens
                    break

        await asyncio.wait_for(run(), timeout=timeout_ms / 1000)

    except asyncio.TimeoutError:
        return QueryResponse(success=False, summary=f"{phase} timed out after {timeout_ms}ms")
    except Exception as e:
        logger.exception("[agent] %s error", phase)
        return QueryResponse(success=False, summary=f"Agent error during {phase}: {e}")

    if not text:
        return QueryResponse(success=False, summary=f"Agent returned empty response during {phase}")

    try:
        parsed = json.loads(text)
        logger.info("[agent] %s complete: success=%s, cost=$%.4f", phase, parsed.get("success", True), cost)
        return QueryResponse(
            success=parsed.get("success", True),
            summary=parsed.get("summary", text),
            **{k: v for k, v in parsed.items() if k not in ("success", "summary")},
        )
    except (json.JSONDecodeError, TypeError):
        logger.info("[agent] %s complete (text response), cost=$%.4f", phase, cost)
        return QueryResponse(success=True, summary=text)


_PHASE_ENDPOINTS = [
    ("/analyze", "analysis"),
    ("/execute", "execution"),
    ("/verify", "verification"),
]

_PHASE_TIMEOUTS: dict[str, str] = {
    "analysis": "analysis",
    "execution": "execution",
    "verification": "analysis",
}


def register_query_routes(
    router: APIRouter,
    *,
    provider: AgentProvider,
    skills_dir: str,
    model: str | None,
    max_turns: int,
    analysis_timeout_ms: int,
    execution_timeout_ms: int,
) -> None:
    resolved_model = model or os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-6")
    timeouts = {"analysis": analysis_timeout_ms, "execution": execution_timeout_ms}

    for path, phase in _PHASE_ENDPOINTS:

        async def endpoint(
            req: QueryRequest,
            _phase: str = phase,
            _timeout_key: str = _PHASE_TIMEOUTS[phase],
        ) -> QueryResponse:
            return await _handle_query(
                req,
                _phase,
                provider,
                skills_dir,
                resolved_model,
                max_turns,
                timeouts.get(_timeout_key, analysis_timeout_ms),
            )

        router.add_api_route(path, endpoint, methods=["POST"], response_model=QueryResponse)
