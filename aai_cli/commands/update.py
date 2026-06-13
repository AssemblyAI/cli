"""`assembly update` — upgrade the CLI in place.

The startup "Update available" notice (``update_check.py``) tells the user a
newer release exists; this command is the action it points at. It detects how
the CLI was installed (Homebrew, pipx, uv tool) and shells out to that
channel's own upgrade command, the same install-channel dispatch ``codex
update`` does — there is no bespoke self-replacing binary logic to go wrong.
"""

from __future__ import annotations

import shlex
import subprocess

import typer

from aai_cli import (
    __version__,
    command_registry,
    config,
    help_panels,
    options,
    output,
    update_check,
)
from aai_cli.context import AppState, run_command
from aai_cli.errors import APIError, CLIError
from aai_cli.help_text import examples_epilog

SPEC = command_registry.CommandModuleSpec(
    panel=help_panels.SETUP,
    order=25,  # pragma: no mutate -- sparse rank; a +-1 shift is order-equivalent
    commands=("update",),
)

app = typer.Typer()


def _latest_version() -> str:
    """The newest released version, fetched now (not the day-old startup cache)."""
    update_check.fetch_and_cache()
    _, latest = config.get_update_cache()
    if latest is None:
        raise APIError(
            "Couldn't determine the latest version from GitHub releases.",
            suggestion="Check your network connection and retry.",
        )
    return latest


def _run_upgrade(command: str) -> None:
    """Run the install channel's own upgrade command, inheriting stdio so the user
    watches the real brew/pipx/uv output stream by."""
    returncode = subprocess.run(shlex.split(command), check=False).returncode
    if returncode != 0:
        raise CLIError(
            f"'{command}' exited with status {returncode}.",
            error_type="update_failed",
            suggestion="Re-run it directly to see the full output.",
        )


@app.command(
    rich_help_panel=help_panels.SETUP,
    epilog=examples_epilog(
        [
            ("Upgrade to the latest release", "assembly update"),
            ("Only check whether one exists", "assembly update --check"),
        ]
    ),
)
def update(
    ctx: typer.Context,
    check: bool = typer.Option(
        False, "--check", help="Report whether a newer release exists without installing it."
    ),
    json_out: bool = options.json_option(),
) -> None:
    """Update the CLI to the latest release via your install method (brew/pipx/uv)."""

    def body(_state: AppState, json_mode: bool) -> None:
        latest = _latest_version()
        newer = update_check.is_newer(latest, __version__)
        if check or not newer:
            status = (
                f"Update available: {__version__} → {latest}. Run `assembly update` to install."
                if newer
                else f"Already up to date ({__version__})."
            )
            output.emit(
                {"current": __version__, "latest": latest, "update_available": newer},
                lambda _d: status,
                json_mode=json_mode,
            )
            return
        command = update_check.detect_upgrade_command()
        if not command:
            raise CLIError(
                "Couldn't detect how this CLI was installed, so it can't self-update.",
                error_type="unknown_install",
                exit_code=2,
                suggestion=f"Upgrade with your install method — see {update_check.DOCS_URL}.",
            )
        _run_upgrade(command)
        output.emit(
            {"updated": True, "from": __version__, "to": latest, "command": command},
            lambda _d: output.success(f"Updated {__version__} → {latest} (via '{command}')."),
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)
