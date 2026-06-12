#!/usr/bin/env python
"""Run CodeQL's security + quality suites locally — the same alerts GitHub shows.

The CodeQL workflow (.github/workflows/codeql.yml) uploads two alert sets per
language: the default security suite (the repo's *code scanning* tab) and the
code-quality suite (the *quality* tab). Those only surface after a push, so an
agent or dev session can land a PR that's green on check.sh yet grows alerts on
GitHub. This gate runs the exact same suites against the working tree and fails
on any finding.

Scope mirrors codeql.yml minus swift: python, actions, and javascript-typescript
extract with ``--build-mode=none``; the swift helper needs a real macOS build and
stays CI-only. Requires the CodeQL *bundle* (CLI + bundled query packs) on PATH —
check.sh self-skips when it's absent, and the web session-start hook provisions it.

stdlib-only on purpose (nothing here imports the package under test).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Cluster sub-database name -> query-pack prefix. Keys are the directory names
# `codeql database create --db-cluster` produces for codeql.yml's non-swift
# languages (the `javascript-typescript` language extracts into `javascript`).
_LANGUAGES = {
    "python": "python",
    "actions": "actions",
    "javascript": "javascript",
}
_SUITES = ("code-scanning", "code-quality")


def _run(args: list[str]) -> None:
    """Run a codeql command, surfacing its output only when it fails."""
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout + proc.stderr)
        raise SystemExit(f"codeql command failed: {' '.join(args)}")


def _findings(sarif_path: Path) -> list[str]:
    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    lines: list[str] = []
    for run in sarif["runs"]:
        for result in run.get("results", []):
            location = result["locations"][0]["physicalLocation"]
            uri = location["artifactLocation"]["uri"]
            line = location.get("region", {}).get("startLine", "?")
            lines.append(f"  {result['ruleId']}\t{uri}:{line}\t{result['message']['text']}")
    return lines


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="aai-codeql-") as tmp:
        cluster = Path(tmp) / "dbs"
        _run(
            [
                "codeql",
                "database",
                "create",
                str(cluster),
                "--db-cluster",
                "--language=python,actions,javascript-typescript",
                "--build-mode=none",
                f"--source-root={repo_root}",
                "--threads=0",
            ]
        )
        for db_name, pack in sorted(_LANGUAGES.items()):
            suites = [
                f"codeql/{pack}-queries:codeql-suites/{pack}-{suite}.qls" for suite in _SUITES
            ]
            sarif_path = Path(tmp) / f"{db_name}.sarif"
            _run(
                [
                    "codeql",
                    "database",
                    "analyze",
                    str(cluster / db_name),
                    *suites,
                    "--format=sarif-latest",
                    f"--output={sarif_path}",
                    "--threads=0",
                ]
            )
            found = _findings(sarif_path)
            failures.extend(found)
            status = f"{len(found)} finding(s)" if found else "clean"
            sys.stdout.write(f"  {db_name}: {status}\n")

    if failures:
        sys.stdout.write("CodeQL findings (fix them; GitHub will alert on these):\n")
        sys.stdout.write("\n".join(failures) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
