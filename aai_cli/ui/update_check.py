"""The "update available" notifier.

Best-effort and never-blocking, in the style of npm's update-notifier / Vercel:
the notice always renders from a ``config.toml`` cache (zero latency), and the
cache is refreshed by a detached ``assembly _update-check`` process — the shared
detached-spawn recipe in ``aai_cli/procs.py``, same as ``telemetry.dispatch``.
Every failure is swallowed: the notice must never delay or break a command.
"""

from __future__ import annotations

import shlex
import sys
import time

import typer
from packaging.version import InvalidVersion, Version
from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from aai_cli import __version__
from aai_cli.core import config, env, procs, stdio
from aai_cli.core.errors import CLIError
from aai_cli.ui import output

ENV_DISABLED = "AAI_NO_UPDATE_CHECK"
_RELEASES_URL = "https://api.github.com/repos/AssemblyAI/cli/releases/latest"
DOCS_URL = "https://github.com/AssemblyAI/cli#installation"
_INSTALL_SCRIPT_URL = "https://raw.githubusercontent.com/AssemblyAI/cli/main/install.sh"
# Generic fallback when the install channel is unknown: the canonical one-liner
# installer, which re-installs over any existing copy (it runs through a shell
# because of the pipe — see ``_upgrade_argv``).
_INSTALL_SCRIPT_COMMAND = f"curl -LsSf {_INSTALL_SCRIPT_URL} | sh"
_CHECK_INTERVAL_SECONDS = 24 * 60 * 60
_FETCH_TIMEOUT_SECONDS = 5.0
_USER_AGENT = f"assembly-cli/{__version__}"
_HOMEBREW_PATH_MARKERS = ("/cellar/", "/homebrew/")
_UPGRADE_COMMAND_MARKERS = (
    ("pipx", "pipx upgrade aai-cli"),
    ("/uv/tools/", "uv tool upgrade aai-cli"),
)


def is_newer(latest: str, current: str) -> bool:
    """True only when ``latest`` is a strictly greater, parseable version."""
    try:
        return Version(latest) > Version(current)
    except InvalidVersion:
        return False


def _is_homebrew_executable(executable: str) -> bool:
    # /usr/local/ is Homebrew only on Intel macOS; on Linux it's the conventional
    # prefix for source/manually-built interpreters, so don't claim brew there.
    if sys.platform == "darwin" and executable.startswith("/usr/local/"):
        return True
    return any(marker in executable for marker in _HOMEBREW_PATH_MARKERS)


def detect_upgrade_command() -> str:
    """The exact upgrade command for the install method the running interpreter
    lives in, or "" when it can't be determined (callers show a docs hint)."""
    executable = (sys.executable or "").lower()
    if _is_homebrew_executable(executable):
        return "brew upgrade assembly"
    # pipx/uv track installs by *distribution* name (aai-cli), not the console
    # command (assembly) — "pipx upgrade assembly" fails with "not installed".
    return next(
        (command for marker, command in _UPGRADE_COMMAND_MARKERS if marker in executable),
        "",
    )


def resolve_upgrade_command() -> str:
    """The command that upgrades the running install, always non-empty.

    The detected channel command (brew/pipx/uv) when known, otherwise the canonical
    install-script one-liner — which works regardless of how the CLI was installed.
    """
    return detect_upgrade_command() or _INSTALL_SCRIPT_COMMAND


def _upgrade_argv(command: str) -> list[str]:
    """The argv for running ``command``. The install-script fallback is a shell
    pipeline (``curl … | sh``) so it runs through ``sh -c``; the package-manager
    commands are plain argv split on whitespace."""
    if command == _INSTALL_SCRIPT_COMMAND:
        return ["sh", "-c", command]
    return shlex.split(command)


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
            latest = tag.removeprefix("v")
    except (httpx.HTTPError, ValueError, KeyError, OSError):
        latest = None
    try:
        config.set_update_cache(last_check=now, latest_version=latest)
    except (OSError, CLIError):
        return


def spawn_refresh() -> None:
    """Spawn the detached ``assembly _update-check`` child to refresh the cache.

    The shared recipe (own session, discarded stdio, self-disabling env) keeps the
    user's command from ever waiting and a refresh from spawning another.
    """
    procs.spawn_detached(["_update-check"], disable_env_var=ENV_DISABLED)


def _should_notify(*, json_mode: bool) -> bool:
    """Notify only on human, interactive, opted-in, non-CI runs."""
    if json_mode:
        return False
    if env.get(ENV_DISABLED) or env.get("CI"):
        return False
    return bool(output.error_console.is_terminal)


def _render(current: str, latest: str) -> None:
    upgrade = detect_upgrade_command()
    if upgrade:
        action: Text = Text.assemble("Run ", (upgrade, "aai.success"), " to update")
    else:
        action = Text(f"See {DOCS_URL} to upgrade")
    body = Group(
        Text.assemble("Update available  ", (current, "aai.muted"), " → ", (latest, "aai.success")),
        action,
    )
    # Cosmetic panel styling (padding/expand) — not worth pinning behaviorally.
    panel = Panel(body, border_style="aai.muted", padding=(1, 3), expand=False)  # pragma: no mutate
    output.error_console.print(panel)


def _confirm_upgrade() -> bool:
    """Ask whether to upgrade now (interactive sessions only). Default is No, so a
    bare Enter declines; an aborted prompt (Ctrl-C / EOF) is treated as No too."""
    try:
        return typer.confirm("Update now?", default=False, err=True)
    except (typer.Abort, EOFError):
        return False


def _report_upgrade(latest: str, command: str, returncode: int) -> None:
    if returncode == 0:
        msg = f"Updated to {latest}. Restart assembly to use it."
        output.error_console.print(output.success(msg))
    else:
        output.error_console.print(output.fail(f"Update failed — run '{command}' manually."))


def _maybe_prompt_upgrade(latest: str) -> None:
    """After the notice, offer to run the upgrade in place. Only when stdin is a real
    terminal, so a human can answer; a piped/redirected stdin is left untouched."""
    if not stdio.stdin_is_tty():
        return
    command = resolve_upgrade_command()
    if not _confirm_upgrade():
        return
    returncode = procs.run_foreground(_upgrade_argv(command))
    _report_upgrade(latest, command, returncode)


def _cache_is_stale(last_check: float | None, *, now: float) -> bool:
    if last_check is None:
        return True
    return (now - last_check) > _CHECK_INTERVAL_SECONDS


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
    now = time.time()
    if latest is not None and is_newer(latest, __version__):
        _render(__version__, latest)
        _maybe_prompt_upgrade(latest)
    if _cache_is_stale(last_check, now=now):
        spawn_refresh()
