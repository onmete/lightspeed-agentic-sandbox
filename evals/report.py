"""pytest plugin for JSON eval reports.

Usage: pytest evals/ --eval-report=evals/report.json
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pytest

from .credentials import detect_all
from .runner import EvalResult

_STASH_KEY = pytest.StashKey[EvalResult]()


def store_eval_result(item: pytest.Item, result: EvalResult) -> None:
    item.stash[_STASH_KEY] = result


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--eval-report",
        default=None,
        help="Path to write JSON eval report",
    )


@dataclass
class TestResult:
    test_id: str
    provider: str
    model: str
    status: str
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls_count: int = 0
    schema_valid: bool | None = None
    error: str | None = None


@dataclass
class Report:
    timestamp: str = ""
    duration_seconds: float = 0.0
    summary: dict[str, int] = field(default_factory=lambda: {
        "total": 0, "passed": 0, "failed": 0, "skipped": 0, "error": 0,
    })
    providers: dict[str, dict[str, str]] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)


class EvalReportPlugin:
    def __init__(self, report_path: str) -> None:
        self.report_path = report_path
        self.report = Report()
        self.start_time = 0.0

    def pytest_sessionstart(self, session: pytest.Session) -> None:
        self.start_time = time.time()
        self.report.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        creds = detect_all()
        self.report.providers = {
            name: {
                "status": "available" if status.available else "skipped",
                "source": status.source,
                **({"reason": status.reason} if not status.available else {}),
            }
            for name, status in creds.items()
        }

    def pytest_runtest_makereport(self, item: pytest.Item, call: pytest.CallInfo) -> None:
        if call.when != "call":
            return

        eval_result = item.stash.get(_STASH_KEY, None)

        provider_name = ""
        if hasattr(item, "callspec") and "provider_name" in item.callspec.params:
            provider_name = item.callspec.params["provider_name"]

        model = ""
        if hasattr(item, "callspec") and "default_model" in item.callspec.params:
            model = item.callspec.params["default_model"]

        status = "passed" if call.excinfo is None else "failed"

        tr = TestResult(
            test_id=item.nodeid,
            provider=provider_name,
            model=model,
            status=status,
        )

        if eval_result:
            tr.latency_seconds = round(eval_result.latency_seconds, 3)
            tr.cost_usd = round(eval_result.cost_usd, 6)
            tr.input_tokens = eval_result.input_tokens
            tr.output_tokens = eval_result.output_tokens
            tr.tool_calls_count = len(eval_result.tool_calls)
            tr.error = eval_result.error

        self.report.results.append(asdict(tr))

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when == "call":
            self.report.summary["total"] += 1
            if report.passed:
                self.report.summary["passed"] += 1
            elif report.failed:
                self.report.summary["failed"] += 1
        elif report.when == "setup" and report.skipped:
            self.report.summary["total"] += 1
            self.report.summary["skipped"] += 1

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        self.report.duration_seconds = round(time.time() - self.start_time, 2)
        Path(self.report_path).write_text(json.dumps(asdict(self.report), indent=2) + "\n")


def pytest_configure(config: pytest.Config) -> None:
    report_path = config.getoption("--eval-report", default=None)
    if report_path:
        config.pluginmanager.register(EvalReportPlugin(report_path), "eval_report")
