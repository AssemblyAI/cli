# aai_cli/init/procfile.py
from __future__ import annotations

import re
import shlex
from collections.abc import Mapping
from pathlib import Path

from aai_cli.errors import CLIError

# Matches ${VAR}, ${VAR:-default}, or $VAR — the shell-style refs that appear in a
# Procfile's web: line (we expand them ourselves rather than invoking a shell).
_VAR = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}|\$(\w+)")


def _expand(token: str, env: Mapping[str, str]) -> str:
    """Expand $VAR / ${VAR} / ${VAR:-default} in one token against `env`."""

    def repl(match: re.Match[str]) -> str:
        if match.group(1) is not None:  # ${VAR} or ${VAR:-default}
            name, default = match.group(1), match.group(2)
            return env.get(name) or (default if default is not None else "")
        return env.get(match.group(3), "")  # $VAR

    return _VAR.sub(repl, token)


def require_procfile(target: Path) -> Path:
    """The project's Procfile path, or the standard not-a-project usage error.

    This is how `assembly dev`/`assembly share`/`assembly deploy` all detect they aren't sitting
    inside a scaffolded project, so they fail with the same message.
    """
    procfile = target / "Procfile"
    if not procfile.exists():
        raise CLIError(
            "No Procfile here (expected ./Procfile). cd into a project created by "
            "`assembly init`, or run `assembly init` to scaffold one.",
            error_type="usage_error",
            exit_code=1,
        )
    return procfile


def web_argv(target: Path, *, env: Mapping[str, str]) -> list[str]:
    """The template Procfile's `web:` process, as an expanded argv.

    Raises a usage `CLIError` when there's no Procfile or no `web:` line — that's how
    `assembly dev` detects it isn't sitting inside a scaffolded project.
    """
    procfile = require_procfile(target)
    for line in procfile.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("web:"):
            web = stripped[len("web:") :].strip()
            if web:
                return [_expand(token, env) for token in shlex.split(web)]
    raise CLIError(
        "Procfile has no `web:` process to run.",
        error_type="usage_error",
        exit_code=1,
    )
