"""E2E credential validation (standalone; do not import from evals/).

Resolution order per provider mirrors the deploy scripts in
lightspeed-operator/hack/ — env vars first, CLI tools as fallback.

Unlike evals credential checks, missing credentials raise instead of soft-skipping.

E2E matrix ids use {model-or-vendor}-{transport?}-{runtime} (not AgentProvider.name /
CR LIGHTSPEED_PROVIDER).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field

PROVIDER_ANTHROPIC_VERTEX_DEEPAGENTS = "anthropic-vertex-deepagents"
PROVIDER_GEMINI_VERTEX_ADK = "gemini-vertex-adk"
PROVIDER_OPENAI_AGENTS = "openai-agents"


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


def _check_anthropic_vertex_deepagents() -> ProviderCredentialStatus:
    name = PROVIDER_ANTHROPIC_VERTEX_DEEPAGENTS
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ProviderCredentialStatus(
            name,
            True,
            "env",
            "ANTHROPIC_API_KEY set",
        )

    if os.environ.get("CLAUDE_CODE_USE_VERTEX") == "1":
        gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if gac and os.path.isfile(gac):
            return ProviderCredentialStatus(
                name,
                True,
                "env",
                "Vertex AI credentials file",
            )
        ok, _ = _run_quiet(["gcloud", "auth", "application-default", "print-access-token"])
        if ok:
            return ProviderCredentialStatus(
                name,
                True,
                "gcloud",
                "gcloud application-default credentials",
            )
        return ProviderCredentialStatus(
            name,
            False,
            "none",
            "CLAUDE_CODE_USE_VERTEX=1 but no credentials file or gcloud ADC",
        )

    if os.environ.get("CLAUDE_CODE_USE_BEDROCK") == "1":
        if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
            return ProviderCredentialStatus(
                name,
                True,
                "env",
                "AWS Bedrock credentials via env vars",
            )
        ok, _ = _run_quiet(["aws", "configure", "get", "aws_access_key_id"])
        if ok:
            return ProviderCredentialStatus(
                name,
                True,
                "aws_cli",
                "AWS credentials via aws configure",
            )
        return ProviderCredentialStatus(
            name,
            False,
            "none",
            "CLAUDE_CODE_USE_BEDROCK=1 but no AWS credentials found",
        )

    return ProviderCredentialStatus(
        name,
        False,
        "none",
        "ANTHROPIC_API_KEY not set (or set CLAUDE_CODE_USE_VERTEX=1 / CLAUDE_CODE_USE_BEDROCK=1)",
    )


def _vertex_env_vars() -> dict[str, str]:
    """Build env vars needed to switch google-genai / ADK to Vertex AI."""
    env: dict[str, str] = {"GOOGLE_GENAI_USE_VERTEXAI": "TRUE"}
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        project = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", "")
        if not project:
            ok, project = _run_quiet(["gcloud", "config", "get-value", "project"])
            if not ok:
                project = ""
        if project:
            env["GOOGLE_CLOUD_PROJECT"] = project
    if not os.environ.get("GOOGLE_CLOUD_LOCATION"):
        env["GOOGLE_CLOUD_LOCATION"] = os.environ.get("CLOUD_ML_REGION", "us-east5")
    return env


def _check_gemini_vertex_adk() -> ProviderCredentialStatus:
    name = PROVIDER_GEMINI_VERTEX_ADK
    if os.environ.get("GOOGLE_API_KEY"):
        return ProviderCredentialStatus(
            name,
            True,
            "env",
            "GOOGLE_API_KEY set",
        )

    # Bridge GEMINI_API_KEY → GOOGLE_API_KEY (ADK reads GOOGLE_API_KEY)
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        return ProviderCredentialStatus(
            name,
            True,
            "env",
            "GEMINI_API_KEY set",
            env_vars={"GOOGLE_API_KEY": gemini_key},
        )

    # ADC file → Vertex AI mode (ADC is for GCP APIs, not the Gemini Developer API)
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if gac and os.path.isfile(gac):
        return ProviderCredentialStatus(
            name,
            True,
            "env",
            "GOOGLE_APPLICATION_CREDENTIALS file (Vertex AI)",
            env_vars=_vertex_env_vars(),
        )

    ok, _ = _run_quiet(["gcloud", "auth", "application-default", "print-access-token"])
    if ok:
        return ProviderCredentialStatus(
            name,
            True,
            "gcloud",
            "gcloud ADC (Vertex AI)",
            env_vars=_vertex_env_vars(),
        )

    return ProviderCredentialStatus(
        name,
        False,
        "none",
        "No Gemini credentials: set GOOGLE_API_KEY, GEMINI_API_KEY, "
        "GOOGLE_APPLICATION_CREDENTIALS, or configure gcloud ADC",
    )


def _check_openai_agents() -> ProviderCredentialStatus:
    name = PROVIDER_OPENAI_AGENTS
    if os.environ.get("OPENAI_API_KEY"):
        return ProviderCredentialStatus(
            name,
            True,
            "env",
            "OPENAI_API_KEY set",
        )

    if os.environ.get("OPENAI_BASE_URL"):
        return ProviderCredentialStatus(
            name,
            True,
            "env",
            "OPENAI_BASE_URL set (custom endpoint, no key required)",
        )

    return ProviderCredentialStatus(
        name,
        False,
        "none",
        "OPENAI_API_KEY not set (or set OPENAI_BASE_URL for keyless endpoints)",
    )


_CHECKERS = {
    PROVIDER_ANTHROPIC_VERTEX_DEEPAGENTS: _check_anthropic_vertex_deepagents,
    PROVIDER_GEMINI_VERTEX_ADK: _check_gemini_vertex_adk,
    PROVIDER_OPENAI_AGENTS: _check_openai_agents,
}

PROVIDER_NAMES = list(_CHECKERS.keys())


def detect_credentials(provider: str) -> ProviderCredentialStatus:
    checker = _CHECKERS.get(provider)
    if checker is None:
        return ProviderCredentialStatus(provider, False, "none", f"Unknown provider: {provider}")
    return checker()


def detect_all() -> dict[str, ProviderCredentialStatus]:
    return {name: detect_credentials(name) for name in PROVIDER_NAMES}


def require_credentials(provider: str) -> None:
    """Fail fast with a clear message when the host cannot run E2E for this provider."""
    status = detect_credentials(provider)
    if not status.available:
        msg = f"E2E credentials missing for provider {provider!r}: {status.reason}"
        raise RuntimeError(msg)
    for key, value in status.env_vars.items():
        if value and not os.environ.get(key):
            os.environ[key] = value


def main() -> None:
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "check":
        require_credentials(sys.argv[2])
        return
    raise SystemExit("usage: python credentials.py check <provider>")


if __name__ == "__main__":
    main()
