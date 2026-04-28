"""Credential auto-detection for LLM providers.

Resolution order per provider mirrors the deploy scripts in
lightspeed-operator/hack/ — env vars first, CLI tools as fallback.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderCredentialStatus:
    provider: str
    available: bool
    source: str
    reason: str
    env_vars: dict[str, str] = field(default_factory=dict)


def _run_quiet(cmd: list[str], timeout: int = 10) -> tuple[bool, str]:
    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, ""


def _check_claude() -> ProviderCredentialStatus:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ProviderCredentialStatus("claude", True, "env", "ANTHROPIC_API_KEY set")

    if os.environ.get("CLAUDE_CODE_USE_VERTEX") == "1":
        gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if gac and os.path.isfile(gac):
            return ProviderCredentialStatus(
                "claude", True, "env", "Vertex AI credentials file"
            )
        ok, _ = _run_quiet(["gcloud", "auth", "application-default", "print-access-token"])
        if ok:
            return ProviderCredentialStatus(
                "claude", True, "gcloud", "gcloud application-default credentials"
            )
        return ProviderCredentialStatus(
            "claude", False, "none",
            "CLAUDE_CODE_USE_VERTEX=1 but no credentials file or gcloud ADC",
        )

    if os.environ.get("CLAUDE_CODE_USE_BEDROCK") == "1":
        if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
            return ProviderCredentialStatus(
                "claude", True, "env", "AWS Bedrock credentials via env vars"
            )
        ok, _ = _run_quiet(["aws", "configure", "get", "aws_access_key_id"])
        if ok:
            return ProviderCredentialStatus(
                "claude", True, "aws_cli", "AWS credentials via aws configure"
            )
        return ProviderCredentialStatus(
            "claude", False, "none",
            "CLAUDE_CODE_USE_BEDROCK=1 but no AWS credentials found",
        )

    return ProviderCredentialStatus(
        "claude", False, "none",
        "ANTHROPIC_API_KEY not set (or set CLAUDE_CODE_USE_VERTEX=1 / CLAUDE_CODE_USE_BEDROCK=1)",
    )


def _check_gemini() -> ProviderCredentialStatus:
    if os.environ.get("GOOGLE_API_KEY"):
        return ProviderCredentialStatus("gemini", True, "env", "GOOGLE_API_KEY set")

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        return ProviderCredentialStatus(
            "gemini", True, "env", "GEMINI_API_KEY set",
            env_vars={"GOOGLE_API_KEY": gemini_key},
        )

    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if gac and os.path.isfile(gac):
        return ProviderCredentialStatus(
            "gemini", True, "env", "GOOGLE_APPLICATION_CREDENTIALS file"
        )

    ok, _ = _run_quiet(["gcloud", "auth", "application-default", "print-access-token"])
    if ok:
        return ProviderCredentialStatus(
            "gemini", True, "gcloud", "gcloud application-default credentials"
        )

    return ProviderCredentialStatus(
        "gemini", False, "none",
        "No Gemini credentials: set GOOGLE_API_KEY, GEMINI_API_KEY, "
        "GOOGLE_APPLICATION_CREDENTIALS, or configure gcloud ADC",
    )


def _check_openai() -> ProviderCredentialStatus:
    if os.environ.get("OPENAI_API_KEY"):
        return ProviderCredentialStatus("openai", True, "env", "OPENAI_API_KEY set")

    if os.environ.get("OPENAI_BASE_URL"):
        return ProviderCredentialStatus(
            "openai", True, "env", "OPENAI_BASE_URL set (custom endpoint, no key required)"
        )

    return ProviderCredentialStatus(
        "openai", False, "none",
        "OPENAI_API_KEY not set (or set OPENAI_BASE_URL for keyless endpoints)",
    )


def _check_deepagents() -> ProviderCredentialStatus:
    model = os.environ.get("DEEPAGENTS_MODEL", "claude-opus-4-6")

    if model.startswith(("gpt", "o1", "o3")):
        if os.environ.get("OPENAI_API_KEY"):
            return ProviderCredentialStatus(
                "deepagents", True, "env", f"OPENAI_API_KEY set (model: {model})"
            )
        return ProviderCredentialStatus(
            "deepagents", False, "none", f"OPENAI_API_KEY not set for model {model}"
        )

    if model.startswith("gemini") and (
        os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    ):
        return ProviderCredentialStatus(
            "deepagents", True, "env", f"Google API key set (model: {model})",
            env_vars={"GOOGLE_API_KEY": os.environ.get("GEMINI_API_KEY", "")}
            if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY")
            else {},
        )

    if os.environ.get("ANTHROPIC_API_KEY"):
        return ProviderCredentialStatus("deepagents", True, "env", "ANTHROPIC_API_KEY set")

    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if gac and os.path.isfile(gac):
        return ProviderCredentialStatus(
            "deepagents", True, "env", "GOOGLE_APPLICATION_CREDENTIALS file (Vertex AI)"
        )

    ok, _ = _run_quiet(["gcloud", "auth", "application-default", "print-access-token"])
    if ok:
        return ProviderCredentialStatus(
            "deepagents", True, "gcloud", "gcloud application-default credentials"
        )

    return ProviderCredentialStatus(
        "deepagents", False, "none",
        "No DeepAgents credentials: set ANTHROPIC_API_KEY, GOOGLE_API_KEY, "
        "OPENAI_API_KEY, or configure gcloud ADC",
    )


def _rewrap(status: ProviderCredentialStatus, name: str) -> ProviderCredentialStatus:
    return ProviderCredentialStatus(
        name, status.available, status.source, status.reason, status.env_vars
    )


def _check_deepagents_gemini() -> ProviderCredentialStatus:
    return _rewrap(_check_gemini(), "deepagents-gemini")


def _check_deepagents_openai() -> ProviderCredentialStatus:
    return _rewrap(_check_openai(), "deepagents-openai")


_CHECKERS = {
    "claude": _check_claude,
    "gemini": _check_gemini,
    "openai": _check_openai,
    "deepagents": _check_deepagents,
    "deepagents-gemini": _check_deepagents_gemini,
    "deepagents-openai": _check_deepagents_openai,
}

PROVIDER_NAMES = list(_CHECKERS.keys())


def detect_credentials(provider: str) -> ProviderCredentialStatus:
    checker = _CHECKERS.get(provider)
    if checker is None:
        return ProviderCredentialStatus(provider, False, "none", f"Unknown provider: {provider}")
    return checker()


def detect_all() -> dict[str, ProviderCredentialStatus]:
    return {name: detect_credentials(name) for name in PROVIDER_NAMES}
