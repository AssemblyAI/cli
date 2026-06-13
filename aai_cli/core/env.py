"""The single chokepoint for raw environment access.

`os.environ` / `os.getenv` are banned project-wide via ruff's `banned-api`
(TID251); this module is the one place allowlisted to touch them, so "who reads
the environment" is answered structurally by the import graph rather than by a
hand-maintained per-file allowlist. Callers keep ownership of their variable
*names* (e.g. `config.ENV_API_KEY`, `telemetry.ENV_DISABLED`) and pass them in;
this module only performs the raw read/write.

Process spawning is the sibling boundary, owned by the modules that shell out to
their specific external tool (see the TID251 allowlist in pyproject.toml).
"""

from __future__ import annotations

import os


def get(name: str, default: str | None = None) -> str | None:
    """Return the value of environment variable ``name`` (or ``default``)."""
    return os.environ.get(name, default)


def child_env(**overrides: str) -> dict[str, str]:
    """A copy of the current environment with ``overrides`` applied.

    For handing a tweaked environment to a child process (e.g. injecting ``PORT``)
    without mutating this process's own ``os.environ``.
    """
    return {**os.environ, **overrides}


def force_color() -> None:
    """Force color on for this process and its children.

    Sets ``FORCE_COLOR`` and clears ``NO_COLOR`` so consoles built later — and
    child processes — agree with the explicit ``--color always``.
    """
    os.environ["FORCE_COLOR"] = "1"
    os.environ.pop("NO_COLOR", None)


def disable_color() -> None:
    """Force color off for this process and its children (the ``--color never`` half)."""
    os.environ["NO_COLOR"] = "1"
    os.environ.pop("FORCE_COLOR", None)
