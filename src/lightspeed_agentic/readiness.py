"""Readiness checks for Kubernetes probes (OLS-3060)."""

from __future__ import annotations

import importlib
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

_PROBE_TIMEOUT_S = 3

_PROVIDER_MODULES: dict[str, str] = {
    "claude": "lightspeed_agentic.providers.claude",
    "gemini": "lightspeed_agentic.providers.gemini",
    "openai": "lightspeed_agentic.providers.openai",
}

_SDK_MODULES: dict[str, str] = {
    "claude": "claude_agent_sdk",
    "gemini": "google.adk",
    "openai": "agents",
}

_cache: tuple[float, dict[str, Any]] | None = None


@dataclass(frozen=True)
class ReadinessReport:
    status: str
    checks: dict[str, str]


def _cache_ttl_seconds() -> int:
    raw = os.environ.get("LIGHTSPEED_READINESS_CACHE_TTL_SECONDS", "30")
    try:
        return max(0, int(raw))
    except ValueError:
        return 30


def _provider_name() -> str:
    return os.environ.get("LIGHTSPEED_AGENT_PROVIDER", "claude")


def check_skills_dir(skills_dir: str) -> str:
    path = os.path.abspath(skills_dir)
    if not os.path.isdir(path):
        return f"error: not a directory ({path})"
    entries = [name for name in os.listdir(path) if not name.startswith(".")]
    if not entries:
        return f"error: no files in {path}"
    return "ok"


def check_provider_credentials(provider: str) -> str:
    if provider == "claude":
        if os.environ.get("CLAUDE_CODE_USE_VERTEX", "") == "1":
            if os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", "").strip():
                return "ok"
            return "error: ANTHROPIC_VERTEX_PROJECT_ID is not set"
        if os.environ.get("ANTHROPIC_API_KEY", "").strip():
            return "ok"
        return "error: ANTHROPIC_API_KEY is not set"
    if provider == "gemini":
        if (
            os.environ.get("GOOGLE_API_KEY", "").strip()
            or os.environ.get("GEMINI_API_KEY", "").strip()
        ):
            return "ok"
        return "error: GOOGLE_API_KEY or GEMINI_API_KEY is not set"
    if provider == "openai":
        if os.environ.get("OPENAI_API_KEY", "").strip():
            return "ok"
        return "error: OPENAI_API_KEY is not set"
    return f"error: unknown provider {provider}"


def check_provider_sdk(provider: str) -> str:
    adapter = _PROVIDER_MODULES.get(provider)
    if adapter is None:
        return f"error: unknown provider {provider}"
    try:
        importlib.import_module(adapter)
    except Exception as exc:
        return f"error: adapter import failed ({exc})"

    sdk = _SDK_MODULES.get(provider)
    if sdk is None:
        return "ok"
    try:
        importlib.import_module(sdk)
    except ImportError as exc:
        return f"error: SDK not importable ({exc})"
    except Exception as exc:
        return f"error: SDK import failed ({exc})"
    return "ok"


def _probe_url(url: str) -> str:
    scheme = urlparse(url).scheme
    if scheme not in ("http", "https"):
        return f"error: unsupported URL scheme {scheme!r}"
    try:
        request = Request(url, method="GET")  # noqa: S310
        with urlopen(request, timeout=_PROBE_TIMEOUT_S) as response:  # noqa: S310
            if response.status >= 500:
                return f"error: HTTP {response.status}"
    except HTTPError as exc:
        if exc.code < 500:
            return "ok"
        return f"error: HTTP {exc.code}"
    except (TimeoutError, URLError, OSError) as exc:
        return f"error: {exc}"
    return "ok"


def check_mcp_servers() -> str:
    raw = os.environ.get("LIGHTSPEED_MCP_SERVERS", "").strip()
    if not raw:
        return "skipped: not configured"
    urls = [part.strip() for part in raw.split(",") if part.strip()]
    if not urls:
        return "skipped: not configured"
    failures: list[str] = []
    for url in urls:
        result = _probe_url(url)
        if result != "ok":
            failures.append(f"{url} ({result})")
    if failures:
        return "error: " + "; ".join(failures)
    return "ok"


def run_readiness_checks(skills_dir: str) -> ReadinessReport:
    """Run subsystem checks; results may be cached per LIGHTSPEED_READINESS_CACHE_TTL_SECONDS."""
    global _cache

    ttl = _cache_ttl_seconds()
    now = time.monotonic()
    if _cache is not None and ttl > 0 and (now - _cache[0]) < ttl:
        cached = _cache[1]
        return ReadinessReport(status=cached["status"], checks=dict(cached["checks"]))

    provider = _provider_name()
    checks = {
        "skills_dir": check_skills_dir(skills_dir),
        "provider_credentials": check_provider_credentials(provider),
        "provider_sdk": check_provider_sdk(provider),
        "mcp_servers": check_mcp_servers(),
    }
    failed = any(value.startswith("error:") for value in checks.values())
    payload: dict[str, Any] = {
        "status": "error" if failed else "ok",
        "checks": checks,
    }
    if ttl > 0:
        _cache = (now, payload)
    return ReadinessReport(status=payload["status"], checks=checks)


def clear_readiness_cache() -> None:
    """Clear cached readiness result (for tests)."""
    global _cache
    _cache = None
