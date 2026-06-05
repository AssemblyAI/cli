from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
import sys

_MUTANTS_DIR = "mutants"
_PASSING_STATUSES = {"killed", "caught by type check"}
_SMOKE_MUTANTS = (
    "aai_cli.auth.ams.x__json_or_raise__mutmut_*",
    "aai_cli.config_builder.x__merge__mutmut_*",
    "aai_cli.context.x__persist_browser_login__mutmut_*",
    "aai_cli.errors.x_is_auth_failure__mutmut_*",
    "aai_cli.output.x_mask_secret__mutmut_*",
)


def _run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=capture,
        check=False,
        text=True,
    )


def _selected(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in _SMOKE_MUTANTS)


def _bad_results(output: str) -> tuple[list[str], int]:
    bad: list[str] = []
    seen = 0
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or ": " not in stripped:
            continue
        name, status = stripped.rsplit(": ", 1)
        if not _selected(name):
            continue
        seen += 1
        if status not in _PASSING_STATUSES:
            bad.append(stripped)
    return bad, seen


def main() -> int:
    shutil.rmtree(_MUTANTS_DIR, ignore_errors=True)
    max_children = os.environ.get("MUTMUT_MAX_CHILDREN", "1")
    run = _run(["mutmut", "run", "--max-children", max_children, *_SMOKE_MUTANTS], capture=True)
    if run.returncode != 0:
        if run.stdout:
            sys.stdout.write(run.stdout)
        if run.stderr:
            sys.stderr.write(run.stderr)
        return run.returncode

    results = _run(["mutmut", "results", "--all", "true"], capture=True)
    if results.returncode != 0:
        if results.stdout:
            sys.stdout.write(results.stdout)
        if results.stderr:
            sys.stderr.write(results.stderr)
        return results.returncode

    bad, seen = _bad_results(results.stdout)
    if seen == 0:
        sys.stderr.write("Mutation testing did not report any configured smoke mutants.\n")
        return 1
    if bad:
        sys.stderr.write("\nMutation testing found surviving or untested mutants:\n")
        for line in bad:
            sys.stderr.write(f"  {line}\n")
        return 1
    sys.stdout.write(f"mutation smoke killed {seen} mutants\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
