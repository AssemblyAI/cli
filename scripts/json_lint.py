#!/usr/bin/env python
"""Validate that every JSON file in the repo is well-formed.

A malformed JSON file (a committed Datadog dashboard, a vercel.json in an init
template, a recorded API fixture) fails silently downstream — a bad dashboard just
won't import, a broken fixture surfaces as a confusing test error. This gate parses
each tracked-or-staged ``*.json`` and fails loudly on the first one that won't parse.

Scope is git-tracked plus untracked-not-ignored files, so a new JSON file is checked
before it is committed (matching how the gitleaks step scans the working tree). It is
a validity check, not a formatter: the recorded ``tests/fixtures/api/*.json`` are
regenerated snapshots and must not be reflowed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def json_files() -> list[str]:
    """Tracked + untracked-not-ignored ``*.json`` paths, deduped and sorted."""
    paths: set[str] = set()
    for extra in ([], ["--others", "--exclude-standard"]):
        out = subprocess.run(
            ["git", "ls-files", "-z", *extra, "*.json"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        paths.update(p for p in out.split("\0") if p)
    return sorted(paths)


def main() -> int:
    failures: list[str] = []
    files = json_files()
    for path in files:
        try:
            with Path(path).open(encoding="utf-8") as handle:
                json.load(handle)
        except (OSError, json.JSONDecodeError) as err:
            failures.append(f"{path}: {err}")

    if failures:
        sys.stdout.write("Malformed JSON:\n")
        for failure in failures:
            sys.stdout.write(f"  {failure}\n")
        return 1

    sys.stdout.write(f"  {len(files)} JSON files OK\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
