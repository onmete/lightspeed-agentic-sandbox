"""Unit tests for derive_phase — pure logic, no kubernetes dependency."""

from __future__ import annotations

from tests.e2e.k8s_constants import TERMINAL_PHASES, derive_phase


class TestDerivePhase:
    """Tests for derive_phase mapping conditions to phases."""

    def test_pending_no_conditions(self) -> None:
        assert derive_phase([]) == "Pending"

    def test_escalated(self) -> None:
        conditions = [{"type": "Escalated", "status": "True"}]
        assert derive_phase(conditions) == "Escalated"

    def test_escalating(self) -> None:
        # Verification failed and escalation is in flight: the operator sets
        # Verified=False plus Escalated=Unknown. Escalated takes priority over
        # Verified, so this derives Escalating (non-terminal), not Failed.
        conditions = [
            {"type": "Verified", "status": "False"},
            {"type": "Escalated", "status": "Unknown"},
        ]
        assert derive_phase(conditions) == "Escalating"

    def test_verified_true(self) -> None:
        conditions = [
            {"type": "Analyzed", "status": "True"},
            {"type": "Approved", "status": "True"},
            {"type": "Executed", "status": "True"},
            {"type": "Verified", "status": "True"},
        ]
        assert derive_phase(conditions) == "Completed"

    def test_verified_false(self) -> None:
        conditions = [
            {"type": "Verified", "status": "False"},
        ]
        assert derive_phase(conditions) == "Failed"

    def test_verified_unknown(self) -> None:
        conditions = [
            {"type": "Verified", "status": "Unknown"},
        ]
        assert derive_phase(conditions) == "Verifying"

    def test_executed_unknown(self) -> None:
        conditions = [
            {"type": "Analyzed", "status": "True"},
            {"type": "Approved", "status": "True"},
            {"type": "Executed", "status": "Unknown"},
        ]
        assert derive_phase(conditions) == "Executing"

    def test_executed_true(self) -> None:
        conditions = [
            {"type": "Executed", "status": "True"},
        ]
        assert derive_phase(conditions) == "Verifying"

    def test_executed_false(self) -> None:
        conditions = [{"type": "Executed", "status": "False"}]
        assert derive_phase(conditions) == "Failed"

    def test_approved_false(self) -> None:
        conditions = [
            {"type": "Analyzed", "status": "True"},
            {"type": "Approved", "status": "False"},
        ]
        assert derive_phase(conditions) == "Denied"

    def test_approved_true(self) -> None:
        conditions = [
            {"type": "Analyzed", "status": "True"},
            {"type": "Approved", "status": "True"},
        ]
        assert derive_phase(conditions) == "Executing"

    def test_analyzed_true(self) -> None:
        conditions = [{"type": "Analyzed", "status": "True"}]
        assert derive_phase(conditions) == "Proposed"

    def test_analyzed_unknown(self) -> None:
        conditions = [{"type": "Analyzed", "status": "Unknown"}]
        assert derive_phase(conditions) == "Analyzing"

    def test_analyzed_false(self) -> None:
        conditions = [{"type": "Analyzed", "status": "False"}]
        assert derive_phase(conditions) == "Failed"


class TestConstants:
    """Verify exported constants."""

    def test_terminal_phases(self) -> None:
        assert frozenset({"Completed", "Failed", "Denied", "Escalated"}) == TERMINAL_PHASES

    def test_escalating_is_not_terminal(self) -> None:
        # Escalation in progress must not stop phase polling early.
        assert "Escalating" not in TERMINAL_PHASES
