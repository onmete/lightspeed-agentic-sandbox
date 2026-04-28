"""JSON Schema definitions for E2E contract tests.

Schemas mirror the operator's expected output formats for each phase:
- Analysis: AnalysisOutputSchema from lightspeed-operator
- Execution: ExecutionOutputSchema
- Verification: VerificationOutputSchema
"""

from __future__ import annotations

from typing import Any

ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "options": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "diagnosis": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"},
                            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                            "rootCause": {"type": "string"},
                            "token": {"type": "string"},
                        },
                        "required": ["summary", "confidence", "rootCause", "token"],
                    },
                    "proposal": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "actions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {"type": "string"},
                                        "description": {"type": "string"},
                                    },
                                    "required": ["type", "description"],
                                },
                            },
                            "risk": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "critical"],
                            },
                            "reversible": {"type": "boolean"},
                            "estimatedImpact": {"type": "string"},
                        },
                        "required": ["description", "actions", "risk", "reversible"],
                    },
                    "verification": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "token": {"type": "string"},
                            "steps": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "command": {"type": "string"},
                                        "expected": {"type": "string"},
                                        "type": {"type": "string"},
                                    },
                                },
                            },
                            "rollbackPlan": {
                                "type": "object",
                                "properties": {
                                    "description": {"type": "string"},
                                    "command": {"type": "string"},
                                },
                            },
                        },
                    },
                    "rbac": {
                        "type": "object",
                        "properties": {
                            "namespaceScoped": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "namespace": {"type": "string"},
                                        "apiGroups": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "resources": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "resourceNames": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "verbs": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "justification": {"type": "string"},
                                    },
                                    "required": [
                                        "apiGroups", "resources", "verbs", "justification",
                                    ],
                                },
                            },
                            "clusterScoped": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "apiGroups": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "resources": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "resourceNames": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "verbs": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "justification": {"type": "string"},
                                    },
                                    "required": [
                                        "apiGroups", "resources", "verbs", "justification",
                                    ],
                                },
                            },
                        },
                    },
                    "components": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "source": {
                                    "type": "object",
                                    "properties": {
                                        "generator": {"type": "string"},
                                        "timestamp": {"type": "string"},
                                        "entropy_bits": {"type": "integer"},
                                    },
                                    "required": ["generator", "timestamp"],
                                },
                                "tokens": {
                                    "type": "object",
                                    "properties": {
                                        "primary": {
                                            "type": "object",
                                            "properties": {
                                                "value": {"type": "string"},
                                                "algorithm": {"type": "string"},
                                                "valid": {"type": "boolean"},
                                            },
                                            "required": ["value", "valid"],
                                        },
                                        "secondary": {
                                            "type": "object",
                                            "properties": {
                                                "value": {"type": "string"},
                                                "algorithm": {"type": "string"},
                                                "valid": {"type": "boolean"},
                                            },
                                            "required": ["value", "valid"],
                                        },
                                    },
                                    "required": ["primary", "secondary"],
                                },
                                "audit": {
                                    "type": "object",
                                    "properties": {
                                        "outcome": {
                                            "type": "string",
                                            "enum": ["pass", "fail", "partial"],
                                        },
                                        "checks_performed": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "findings": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "check": {"type": "string"},
                                                    "result": {"type": "string"},
                                                    "severity": {
                                                        "type": "string",
                                                        "enum": ["info", "warning", "critical"],
                                                    },
                                                    "detail": {"type": "string"},
                                                },
                                                "required": ["check", "result", "severity"],
                                            },
                                        },
                                    },
                                    "required": ["outcome", "checks_performed", "findings"],
                                },
                            },
                            "required": ["type", "source", "tokens", "audit"],
                        },
                    },
                },
                "required": [
                    "title",
                    "diagnosis",
                    "proposal",
                    "rbac",
                    "verification",
                    "components",
                ],
            },
        },
    },
    "required": ["options"],
}

EXECUTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "actionsTaken": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "result": {"type": "string"},
                    "status": {"type": "string", "enum": ["completed", "failed", "skipped"]},
                },
                "required": ["action", "result", "status"],
            },
        },
        "verification": {
            "type": "object",
            "properties": {
                "conditionImproved": {"type": "boolean"},
                "summary": {"type": "string"},
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "check": {"type": "string"},
                            "result": {"type": "string"},
                        },
                        "required": ["check", "result"],
                    },
                },
            },
            "required": ["conditionImproved", "summary"],
        },
    },
    "required": ["success", "actionsTaken", "verification"],
}

VERIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "conditionImproved": {"type": "boolean"},
        "summary": {"type": "string"},
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "passed": {"type": "boolean"},
                    "detail": {"type": "string"},
                },
                "required": ["name", "passed"],
            },
        },
        "recommendation": {"type": "string"},
    },
    "required": ["conditionImproved", "summary"],
}
