#!/usr/bin/env python3
"""Build ACP PR review prompt; kept out of workflow YAML (GitHub strips HTML comments)."""

from __future__ import annotations

import json
import os
import sys

CLOSURE_OPEN = "\x3c!-- AC_REVIEW_CLOSURE"
CLOSURE_CLOSE = "--\x3e"


def main() -> None:
    runner_temp = os.environ["RUNNER_TEMP"]
    with open(f"{runner_temp}/pr.json", encoding="utf-8") as f:
        pr = json.load(f)

    testing = os.environ.get("ACP_REVIEW_TESTING", "true").lower() == "true"

    closure_spec = f"""
## Required summary closure (machine-readable — do not skip)

After your human-readable review, append **exactly** this HTML comment block as the last thing in your reply.
Copy the format verbatim (raw HTML comment only — no markdown code fences around it).
GitHub Actions parses it to pass/fail the check.

Testing mode:
{CLOSURE_OPEN}
VERDICT=PASS
JIRA_KEYS=OLS-1234
SMOKE_TEST=OK
SUMMARY=Jira and repository access verified.
{CLOSURE_CLOSE}

Full review mode:
{CLOSURE_OPEN}
VERDICT=FAIL
JIRA_KEYS=OLS-1234,OCPSTRAT-99
CRITERIA_TOTAL=5
CRITERIA_MET=3
CRITERIA_NOT_MET=2
CRITERIA_UNCLEAR=0
SUMMARY=Two acceptance criteria lack tests and error handling.
{CLOSURE_CLOSE}

Rules:
- `VERDICT` must be one of: `PASS`, `FAIL`, `WARN`
- `PASS` — all criteria met (full mode) or smoke test OK (testing mode)
- `FAIL` — any criterion **Not met** or merge-blocking gap
- `WARN` — no Jira key, Jira/AC unavailable, or review inconclusive
- `JIRA_KEYS` — comma-separated keys, or `NONE`
- Count fields must be non-negative integers; in testing mode you may set criteria counts to `0`
- `SUMMARY` — one line, no newlines inside the value
"""

    testing_block = """## Testing mode (current)
Smoke test only:
1. Extract Jira keys from the PR title and description (e.g. OLS-1234, OCPSTRAT-1234).
2. Fetch each issue and its acceptance criteria via the Jira integration.
3. Confirm you can read the repository at the PR head branch.
4. Do **not** judge criteria met/not met yet.

Write a short human summary, then the closure block with `SMOKE_TEST=OK` or `SMOKE_TEST=FAILED`.
"""

    full_review_block = """## Full review mode
1. Extract Jira keys from the PR title and description.
2. Fetch each issue and acceptance criteria (or AC / Definition of Done).
3. Review the full PR diff against the base branch.
4. For each criterion: **Met** / **Not met** / **Unclear** with evidence.
5. Human summary for the PR comment, then the closure block with accurate counts.

Do not push commits.
"""

    mode = testing_block if testing else full_review_block
    body = pr.get("body") or "(none)"

    prompt = f"""# ACP PR review — Jira acceptance criteria

Repository: {os.environ["REPOSITORY"]}
Pull request: #{os.environ["PR_NUMBER"]}
URL: {pr["url"]}
Base: {pr["baseRefName"]} ← Head: {os.environ["HEAD_BRANCH"]}

Title: {pr["title"]}

Description:
{body}

{mode}
{closure_spec}

Format the human-readable portion as GitHub markdown. The closure block must be last.
"""

    out_path = os.environ["GITHUB_OUTPUT"]
    with open(out_path, "a", encoding="utf-8") as out:
        out.write("text<<EOF\n")
        out.write(prompt.rstrip())
        out.write("\nEOF\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"::error::{exc}", file=sys.stderr)
        sys.exit(1)
