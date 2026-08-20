"""Pure-Python constants and phase logic — no kubernetes dependency."""

from __future__ import annotations

CRD_GROUP = "agentic.openshift.io"
CRD_VERSION = "v1alpha1"
TERMINAL_PHASES = frozenset({"Completed", "Failed", "Denied", "Escalated"})


def derive_phase(conditions: list[dict[str, str]]) -> str:
    """Derive the AgenticRun phase from its status conditions.

    Mirror the Go DerivePhase() logic by checking condition types in priority
    order: Escalated, Verified, Executed, Approved, Analyzed.
    """
    cond_map: dict[str, dict[str, str]] = {c["type"]: c for c in conditions}

    if "Escalated" in cond_map:
        status = cond_map["Escalated"]["status"]
        if status == "True":
            return "Escalated"
        # Unknown = escalation in progress after a verification failure
        # (matches Go DerivePhase); non-terminal, so polling must keep going.
        if status == "Unknown":
            return "Escalating"
        return "Failed"

    if "Verified" in cond_map:
        status = cond_map["Verified"]["status"]
        if status == "True":
            return "Completed"
        # Unknown = verification in progress (matches Go DerivePhase)
        if status == "Unknown":
            return "Verifying"
        return "Failed"

    if "Executed" in cond_map:
        status = cond_map["Executed"]["status"]
        if status == "Unknown":
            return "Executing"
        if status == "True":
            return "Verifying"
        return "Failed"

    if "Approved" in cond_map:
        if cond_map["Approved"]["status"] == "False":
            return "Denied"
        return "Executing"

    if "Analyzed" in cond_map:
        status = cond_map["Analyzed"]["status"]
        if status == "True":
            return "Proposed"
        if status == "Unknown":
            return "Analyzing"
        return "Failed"

    return "Pending"
