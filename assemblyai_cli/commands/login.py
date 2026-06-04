from __future__ import annotations

import webbrowser

import typer
from rich.markup import escape

from assemblyai_cli import client, config, output
from assemblyai_cli.context import AppState, resolve_profile, run_command
from assemblyai_cli.errors import APIError, NotAuthenticated

app = typer.Typer()

DASHBOARD_KEYS_URL = "https://www.assemblyai.com/dashboard/api-keys"


@app.command()
def login(
    ctx: typer.Context,
    api_key: str = typer.Option(None, "--api-key", help="Provide key non-interactively."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Authenticate by storing an API key (browser-assisted on a terminal)."""

    def body(state: AppState, json_mode: bool) -> None:
        profile = resolve_profile(state)
        key = api_key
        if not key:
            output.console.print(
                f"Opening the AssemblyAI dashboard to get your API key:\n  {DASHBOARD_KEYS_URL}"
            )
            try:
                webbrowser.open(DASHBOARD_KEYS_URL)
            except Exception:  # noqa: BLE001 - opening a browser is best-effort
                output.console.print(
                    "[aai.muted]Could not open a browser; open the URL above manually.[/aai.muted]"
                )
            key = typer.prompt("Paste your API key", hide_input=True)
        if not client.validate_key(key):
            raise APIError("That API key was rejected (HTTP 401). Check it and retry.")
        config.set_api_key(profile, key)
        output.emit(
            {"authenticated": True, "profile": profile},
            lambda _d: f"[aai.success]Authenticated[/aai.success] on profile '{escape(profile)}'.",
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
        reachable = client.validate_key(key)
        output.emit(
            {"profile": profile, "api_key": masked, "reachable": reachable},
            lambda _d: f"profile={escape(profile)} key={escape(masked)} reachable={reachable}",
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)
