from __future__ import annotations

import typer
from rich.markup import escape

from aai_cli import client, config, environments, output
from aai_cli.auth import run_login_flow
from aai_cli.context import AppState, resolve_profile, run_command
from aai_cli.errors import APIError, NotAuthenticated

app = typer.Typer()


@app.command()
def login(
    ctx: typer.Context,
    api_key: str = typer.Option(None, "--api-key", help="Provide key non-interactively."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Authenticate via your browser (Stytch); stores a CLI API key."""

    def body(state: AppState, json_mode: bool) -> None:
        profile = resolve_profile(state)
        if api_key:
            # Non-interactive escape hatch for CI/automation.
            if not client.validate_key(api_key):
                raise APIError("That API key was rejected (HTTP 401). Check it and retry.")
            key = api_key
        else:
            key = run_login_flow()
        env = environments.active().name
        config.set_api_key(profile, key)
        config.set_profile_env(profile, env)
        output.emit(
            {"authenticated": True, "profile": profile, "env": env},
            lambda _d: f"[aai.success]Authenticated[/aai.success] on profile "
            f"'{escape(profile)}' (env: {escape(env)}).",
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)


@app.command()
def logout(
    ctx: typer.Context,
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Clear stored credentials for the active profile."""

    def body(state: AppState, json_mode: bool) -> None:
        profile = resolve_profile(state)
        config.clear_api_key(profile)
        output.emit(
            {"logged_out": True, "profile": profile},
            lambda _d: f"Logged out of profile '{escape(profile)}'.",
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)


@app.command()
def whoami(
    ctx: typer.Context,
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Show the active profile and whether its key is usable."""

    def body(state: AppState, json_mode: bool) -> None:
        profile = resolve_profile(state)
        key = config.get_api_key(profile)
        if not key:
            raise NotAuthenticated()
        masked = f"{key[:3]}…{key[-4:]}" if len(key) > 7 else "***"
        env = environments.active().name
        reachable = client.validate_key(key)
        output.emit(
            {"profile": profile, "env": env, "api_key": masked, "reachable": reachable},
            lambda _d: f"profile={escape(profile)} env={escape(env)} "
            f"key={escape(masked)} reachable={reachable}",
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)
