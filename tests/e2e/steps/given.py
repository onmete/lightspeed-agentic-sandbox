"""Given step definitions — preconditions for E2E scenarios."""

from __future__ import annotations

from pytest_bdd import given

from ..schemas import ANALYSIS_SCHEMA, EXECUTION_SCHEMA, VERIFICATION_SCHEMA


@given("a running agent service", target_fixture="service_ready")
def given_running_service(server_url):
    return server_url


@given("the analysis output schema", target_fixture="output_schema")
def given_analysis_schema():
    return ANALYSIS_SCHEMA


@given("the execution output schema", target_fixture="output_schema")
def given_execution_schema():
    return EXECUTION_SCHEMA


@given("the verification output schema", target_fixture="output_schema")
def given_verification_schema():
    return VERIFICATION_SCHEMA


@given("an execution context with an approved option", target_fixture="query_context")
def given_execution_context():
    return {
        "approvedOption": {
            "title": "Test remediation",
            "diagnosis": {
                "summary": "Test issue detected",
                "confidence": "high",
                "rootCause": "E2E test scenario",
            },
            "proposal": {
                "description": "Apply test fix",
                "actions": [
                    {"type": "patch", "description": "Apply a no-op patch for testing"},
                ],
                "risk": "low",
                "reversible": True,
            },
        },
        "attempt": 1,
    }


@given("a verification context from a previous execution", target_fixture="query_context")
def given_verification_context():
    return {
        "attempt": 1,
        "previousAttempts": [
            {
                "attempt": 1,
                "status": "executed",
                "summary": "Test remediation applied successfully",
            },
        ],
    }


@given("a workspace with the find-token skill", target_fixture="workspace_ready")
def given_workspace_with_skill(eval_workspace):
    return eval_workspace
