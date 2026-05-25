from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from lightspeed_agentic.factory import create_provider
from lightspeed_agentic.readiness import run_readiness_checks
from lightspeed_agentic.routes import build_router

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(title="lightspeed-agentic-sandbox")

provider = create_provider()
router = build_router(
    provider,
    skills_dir=os.environ.get("LIGHTSPEED_SKILLS_DIR", "/app/skills"),
)
app.include_router(router, prefix="/v1/agent")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readiness", response_model=None)
def readiness() -> JSONResponse | dict[str, object]:
    """Readiness probe with per-subsystem status (OLS-3060)."""
    skills_dir = os.environ.get("LIGHTSPEED_SKILLS_DIR", "/app/skills")
    report = run_readiness_checks(skills_dir)
    body: dict[str, object] = {"status": report.status, "checks": report.checks}
    if report.status != "ok":
        return JSONResponse(status_code=503, content=body)
    return body
