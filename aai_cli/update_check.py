"""The "update available" notifier.

Best-effort and never-blocking, in the style of npm's update-notifier / Vercel:
the notice always renders from a ``config.toml`` cache (zero latency), and the
cache is refreshed by a detached ``assembly _update-check`` process — the same
detached-spawn shape as ``telemetry.dispatch`` (see ``aai_cli/telemetry.py``).
Every failure is swallowed: the notice must never delay or break a command.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

from packaging.version import InvalidVersion, Version
from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from aai_cli import __version__, config, output
from aai_cli.errors import CLIError

ENV_DISABLED = "AAI_NO_UPDATE_CHECK"
_RELEASES_URL = "https://api.github.com/repos/AssemblyAI/cli/releases/latest"
_DOCS_URL = "https://github.com/AssemblyAI/cli#installation"
_CHECK_INTERVAL_SECONDS = 24 * 60 * 60
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
    # pipx/uv track installs by *distribution* name (aai-cli), not the console
    # command (assembly) — "pipx upgrade assembly" fails with "not installed".
    if "pipx" in exe:
        return "pipx upgrade aai-cli"
    if "/uv/tools/" in exe:
        return "uv tool upgrade aai-cli"
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


def spawn_refresh() -> None:
    """Spawn the detached ``assembly _update-check`` child to refresh the cache.

    Own session + discarded stdio so the user's command never waits; the child's
    env disables the notifier so a refresh can never spawn another (mirrors
    ``telemetry.dispatch``). S603 is ignored project-wide for the CLI's own shell-outs.
    """
    subprocess.Popen(
        [sys.executable, "-m", "aai_cli", "_update-check"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env={**os.environ, ENV_DISABLED: "1"},
    )


def _should_notify(*, json_mode: bool) -> bool:
    """Notify only on human, interactive, opted-in, non-CI runs."""
    if json_mode:
        return False
    if os.environ.get(ENV_DISABLED) or os.environ.get("CI"):
        return False
    return bool(output.error_console.is_terminal)


def _render(current: str, latest: str) -> None:
    upgrade = detect_upgrade_command()
    if upgrade:
        action: Text = Text.assemble("Run ", (upgrade, "aai.success"), " to update")
    else:
        action = Text(f"See {_DOCS_URL} to upgrade")
    body = Group(
        Text.assemble("Update available  ", (current, "aai.muted"), " → ", (latest, "aai.success")),
        action,
    )
    # Cosmetic panel styling (padding/expand) — not worth pinning behaviorally.
    panel = Panel(body, border_style="aai.muted", padding=(1, 3), expand=False)  # pragma: no mutate
    output.error_console.print(panel)


def maybe_notify(*, json_mode: bool) -> None:
    """Render the cached notice (if newer) and refresh the cache if stale.

    The single entry point ``run_command`` calls on a command's success path.
    Best-effort: a config/render failure is swallowed, never surfaced.
    """
    try:
        _maybe_notify(json_mode=json_mode)
    except (OSError, CLIError):
        return


def _maybe_notify(*, json_mode: bool) -> None:
    if not _should_notify(json_mode=json_mode):
        return
    last_check, latest = config.get_update_cache()
    if latest and is_newer(latest, __version__):
        _render(__version__, latest)
    if last_check is None or (time.time() - last_check) > _CHECK_INTERVAL_SECONDS:
        spawn_refresh()
