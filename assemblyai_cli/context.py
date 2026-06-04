from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import typer

from assemblyai_cli import config, output
from assemblyai_cli.errors import CLIError


@dataclass
class AppState:
    profile: str | None = None


def resolve_profile(state: AppState) -> str:
    """The profile to act on: explicit --profile, else the active profile."""
    return state.profile or config.get_active_profile()


def run_command(
    ctx: typer.Context, fn: Callable[[AppState, bool], None], *, json: bool = False
) -> None:
    """Execute a command body, mapping CLIError to clean output + exit code."""
    state: AppState = ctx.obj
    json_mode = output.resolve_json(explicit=json)
    try:
        fn(state, json_mode)
    except CLIError as err:
        output.emit_error(err, json_mode=json_mode)
        raise typer.Exit(code=err.exit_code) from None
