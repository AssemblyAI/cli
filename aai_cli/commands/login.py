from __future__ import annotations

import typer
from rich.markup import escape
from rich.table import Table

from aai_cli import client, config, environments, help_panels, output
from aai_cli.auth import run_login_flow
from aai_cli.context import AppState, resolve_profile, run_command
from aai_cli.errors import APIError, NotAuthenticated
from aai_cli.help_text import examples_epilog

app = typer.Typer()


@app.command(
    rich_help_panel=help_panels.ACCOUNT,
    epilog=examples_epilog(
        [
            ("Log in with your browser", "aai login"),
            ("Log in non-interactively (CI)", "aai login --api-key sk_..."),
        ]
    ),
)
def login(
    ctx: typer.Context,
    api_key: str | None = typer.Option(None, "--api-key", help="Provide key non-interactively."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Authenticate via your browser; stores a CLI API key."""

    def body(state: AppState, json_mode: bool) -> None:
        profile = resolve_profile(state)
        env = environments.active().name
        if api_key:
            # Non-interactive escape hatch for CI/automation: no AMS session is
            # obtained, so account self-service commands won't work for this profile.
            if not client.validate_key(api_key):
                raise APIError(
                    "That API key was rejected (HTTP 401).",
                    suggestion="Check the key and retry.",
                )
            config.set_api_key(profile, api_key)
            config.set_profile_env(profile, env)
            # Clear any session from a prior browser login: this profile is now
            # api-key-only, so account self-service must report it needs a browser
            # login rather than silently reusing the old (possibly different) identity.
            config.clear_session(profile)
        else:
            result = run_login_flow()
            config.set_api_key(profile, result.api_key)
            config.set_profile_env(profile, env)
            config.set_session(
                profile,
                session_jwt=result.session_jwt,
                session_token=result.session_token,
                account_id=result.account_id,
            )
        output.emit(
            {"authenticated": True, "profile": profile, "env": env},
            lambda _d: (
                output.success(f"Signed in as {escape(profile)} ({escape(env)}).")
                + "\n"
                + output.hint("Run `aai transcribe <file>` to make your first transcript.")
            ),
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)


@app.command(
    rich_help_panel=help_panels.ACCOUNT,
    epilog=examples_epilog(
        [
            ("Clear stored credentials for the active profile", "aai logout"),
        ]
    ),
)
def logout(
    ctx: typer.Context,
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Clear stored credentials for the active profile."""

    def body(state: AppState, json_mode: bool) -> None:
        profile = resolve_profile(state)
        config.clear_api_key(profile)
        config.clear_session(profile)
        output.emit(
            {"logged_out": True, "profile": profile},
            lambda _d: (
                output.success(f"Signed out of {escape(profile)}.")
                + "\n"
                + output.hint("Run `aai login` to sign back in.")
            ),
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)


@app.command(
    rich_help_panel=help_panels.ACCOUNT,
    epilog=examples_epilog(
        [
            ("Show the active profile and whether its key works", "aai whoami"),
        ]
    ),
)
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
        masked = output.mask_secret(key)
        env = environments.active().name
        reachable = client.validate_key(key)
        session_label = "stored" if config.get_session(profile) else "none"
        account_id = config.get_account_id(profile)

        def render(_d: dict[str, object]) -> Table:
            table = output.detail_table()
            table.add_row("Profile", escape(profile))
            table.add_row("Env", escape(env))
            table.add_row("API key", escape(masked))
            table.add_row(
                "Status",
                output.success("reachable") if reachable else output.fail("key rejected"),
            )
            table.add_row("Account", escape(str(account_id)) if account_id else "—")
            table.add_row("Session", escape(session_label))
            return table

        data: dict[str, object] = {
            "profile": profile,
            "env": env,
            "api_key": masked,
            "reachable": reachable,
            "account_id": account_id,
            "session": session_label,
        }
        output.emit(data, render, json_mode=json_mode)

    run_command(ctx, body, json=json_out)
