"""The "update available" notifier.

Best-effort and never-blocking, in the style of npm's update-notifier / Vercel:
the notice always renders from a ``config.toml`` cache (zero latency), and the
cache is refreshed by a detached ``assembly _update-check`` process — the same
detached-spawn shape as ``telemetry.dispatch`` (see ``aai_cli/telemetry.py``).
Every failure is swallowed: the notice must never delay or break a command.
"""

from __future__ import annotations

import sys
import time

from packaging.version import InvalidVersion, Version

from aai_cli import __version__, config
from aai_cli.errors import CLIError

_RELEASES_URL = "https://api.github.com/repos/AssemblyAI/cli/releases/latest"
_FETCH_TIMEOUT_SECONDS = 5.0
_USER_AGENT = f"assembly-cli/{__version__}"


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


def fetch_and_cache() -> None:
    """Fetch the latest release tag from GitHub and cache it. Best-effort.

    Runs only in the detached ``assembly _update-check`` child, so it imports
    ``httpx2`` lazily (keeping it off every command's import path) and swallows
    all network/parse/IO errors — failures simply mean "no notice next run".
    """
    import httpx2 as httpx

    now = time.time()
    latest: str | None = None
    try:
        resp = httpx.get(
            _RELEASES_URL,
            headers={"User-Agent": _USER_AGENT, "Accept": "application/vnd.github+json"},
            timeout=_FETCH_TIMEOUT_SECONDS,
            follow_redirects=True,
        )
        resp.raise_for_status()
        tag = resp.json().get("tag_name")
        if isinstance(tag, str) and tag:
            latest = tag.lstrip("v")
    except (httpx.HTTPError, ValueError, KeyError, OSError):
        latest = None
    try:
        config.set_update_cache(last_check=now, latest_version=latest)
    except (OSError, CLIError):
        return
