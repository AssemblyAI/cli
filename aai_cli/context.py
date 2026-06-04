from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import typer

from aai_cli import config, environments, output
from aai_cli.environments import Environment
from aai_cli.errors import CLIError


@dataclass
class AppState:
    profile: str | None = None
    env: str | None = None


def resolve_profile(state: AppState) -> str:
    """The profile to act on: explicit --profile, else the active profile."""
    return state.profile or config.get_active_profile()


def resolve_environment(state: AppState) -> Environment:
    """The backend environment: --env > AAI_ENV > the profile's stored env > default."""
    profile_env = config.get_profile_env(resolve_profile(state))
    return environments.resolve(state.env, profile_env)


def env_override_warning(state: AppState) -> str | None:
    """A warning when an explicit --env contradicts the profile's stored env.

    The stored key was minted against the profile's environment, so forcing a
    different --env points the client at hosts that key won't authenticate to.
    """
    if state.env is None:
        return None
    profile = resolve_profile(state)
    profile_env = config.get_profile_env(profile)
    if profile_env is None or profile_env == state.env:
        return None
    return (
        f"Using --env {state.env}, but profile '{profile}' was set up for "
        f"{profile_env}; its stored key may be rejected by {state.env}."
    )


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
