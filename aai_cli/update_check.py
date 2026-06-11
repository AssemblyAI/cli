"""The "update available" notifier.

Best-effort and never-blocking, in the style of npm's update-notifier / Vercel:
the notice always renders from a ``config.toml`` cache (zero latency), and the
cache is refreshed by a detached ``assembly _update-check`` process — the same
detached-spawn shape as ``telemetry.dispatch`` (see ``aai_cli/telemetry.py``).
Every failure is swallowed: the notice must never delay or break a command.
"""

from __future__ import annotations

import sys

from packaging.version import InvalidVersion, Version


def is_newer(latest: str, current: str) -> bool:
    """True only when ``latest`` is a strictly greater, parseable version."""
    try:
        return Version(latest) > Version(current)
    except InvalidVersion:
        return False


def detect_upgrade_command() -> str:
    """The exact upgrade command for the install method the running interpreter
    lives in, or "" when it can't be determined (callers show a docs hint)."""
    exe = (sys.executable or "").lower()
    if "/cellar/" in exe or "/homebrew/" in exe or exe.startswith("/usr/local/"):
        return "brew upgrade assembly"
    if "pipx" in exe:
        return "pipx upgrade assembly"
    if "/uv/tools/" in exe:
        return "uv tool upgrade assembly"
    return ""
