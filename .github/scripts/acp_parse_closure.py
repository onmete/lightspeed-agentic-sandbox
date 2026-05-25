#!/usr/bin/env python3
"""Parse AC_REVIEW_CLOSURE from ACP session output."""

from __future__ import annotations

import os
import re
import sys

CLOSURE_RE = re.compile(
    "\x3c\x21--\\s*AC_REVIEW_CLOSURE\\s*(.*?)\\s*--\x3e",
    re.DOTALL | re.IGNORECASE,
)


def main() -> None:
    runner_temp = os.environ["RUNNER_TEMP"]
    path = f"{runner_temp}/session-result.txt"
    text = open(path, encoding="utf-8").read()
    public_path = f"{runner_temp}/review-public.md"

    match = CLOSURE_RE.search(text)
    out_path = os.environ["GITHUB_OUTPUT"]

    if not match:
        with open(out_path, "a", encoding="utf-8") as out:
            out.write("found=false\n")
        with open(public_path, "w", encoding="utf-8") as f:
            f.write(text.strip())
        return

    fields: dict[str, str] = {}
    for line in match.group(1).strip().splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip().upper()] = value.strip()

    lines = [
        "found=true",
        f"verdict={fields.get('VERDICT', '')}",
        f"jira_keys={fields.get('JIRA_KEYS', '')}",
        f"summary={fields.get('SUMMARY', '')}",
        f"smoke_test={fields.get('SMOKE_TEST', '')}",
        f"criteria_total={fields.get('CRITERIA_TOTAL', '')}",
        f"criteria_met={fields.get('CRITERIA_MET', '')}",
        f"criteria_not_met={fields.get('CRITERIA_NOT_MET', '')}",
        f"criteria_unclear={fields.get('CRITERIA_UNCLEAR', '')}",
    ]
    with open(out_path, "a", encoding="utf-8") as out:
        out.write("\n".join(lines) + "\n")

    public = CLOSURE_RE.sub("", text, count=1).strip()
    with open(public_path, "w", encoding="utf-8") as f:
        f.write(public)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"::error::{exc}", file=sys.stderr)
        sys.exit(1)
