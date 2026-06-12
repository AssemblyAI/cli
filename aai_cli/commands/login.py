from __future__ import annotations

import typer
from rich.markup import escape
from rich.table import Table

from aai_cli import client, config, environments, help_panels, options, output
from aai_cli.context import AppState, persist_browser_login, run_command
from aai_cli.errors import APIError, CLIError, UsageError
from aai_cli.help_text import examples_epilog

app = typer.Typer()


@app.command(
    rich_help_panel=help_panels.ACCOUNT,
    epilog=examples_epilog(
        [
            ("Log in with your browser", "assembly login"),
            ("Log in non-interactively (CI)", "assembly login --api-key sk_..."),
        ]
    ),
)
def login(
    ctx: typer.Context,
    api_key: str | None = typer.Option(None, "--api-key", help="Provide key non-interactively."),
    json_out: bool = options.json_option(),
) -> None:
    """Authenticate via your browser; stores a CLI API key."""

    def body(state: AppState, json_mode: bool) -> None:
        profile = state.resolve_profile()
        env = environments.active().name
        if api_key is not None and not api_key.strip():
            # An explicitly-passed empty/whitespace key (e.g. --api-key "$UNSET_VAR")
            # must fail loudly, not silently fall into the browser flow as if the
            # flag had never been passed.
            raise UsageError(
                "--api-key was given an empty value.",
                suggestion=(
                    "Pass a real key: assembly login --api-key <KEY> "
                    "(check that the shell variable you expanded is set)."
                ),
            )
        # Both login paths persist to the OS keyring, so probe it before any
        # browser/network work: completing the whole OAuth dance only to fail on
        # the final keyring write is the worst place to discover a headless box.
        if not config.keyring_usable():
            raise CLIError(
                "Your OS keyring isn't usable, so login can't store credentials.",
                error_type="keyring_unusable",
                exit_code=2,
                suggestion=(
                    "Set ASSEMBLYAI_API_KEY instead (login can't store a key without "
                    "a keyring), or unlock/install an OS keyring and retry."
                ),
            )
        if api_key is None:
            persist_browser_login(profile, env, json_mode=json_mode)
        else:
            # Non-interactive escape hatch for CI/automation: no AMS session is
            # obtained, so account self-service commands won't work for this profile.
            if not client.validate_key(api_key):
                raise APIError(
                    "That API key was rejected (HTTP 401/403).",
                    suggestion="Check the key and retry.",
                )
            config.set_api_key(profile, api_key)
            config.set_profile_env(profile, env)
            # Clear any session from a prior browser login: this profile is now
            # api-key-only, so account self-service must report it needs a browser
            # login rather than silently reusing the old (possibly different) identity.
            config.clear_session(profile)
        # An --api-key login stores no browser session, so the AMS self-service
        # commands won't work for this profile — say so up front instead of letting
        # the user hit "needs a browser login" later. Named `key_only` (not api_key_*):
        # CodeQL's name heuristic would classify the boolean itself as a secret and
        # flag the emit below (py/clear-text-logging-sensitive-data).
        key_only = api_key is not None

        def render(_d: object) -> str:
            lines = [
                output.success(f"Signed in as {escape(profile)} ({escape(env)})."),
                output.hint(
                    "Run `assembly onboard` to finish setup, or `assembly transcribe <file>`."
                ),
            ]
            if key_only:
                lines.append(
                    output.hint(
                        "Account commands (keys/balance/usage/limits/audit) need "
                        "`assembly login` without --api-key."
                    )
                )
            return "\n".join(lines)

        output.emit(
            {"authenticated": True, "profile": profile, "env": env, "api_key_only": key_only},
            render,
            json_mode=json_mode,
        )

    # auto_login=False: this command owns sign-in; an auth failure here (e.g. the
    # browser flow timing out) must surface, not trigger a second login attempt.
    run_command(ctx, body, json=json_out, auto_login=False)


@app.command(
    rich_help_panel=help_panels.ACCOUNT,
    epilog=examples_epilog(
        [
            ("Clear stored credentials for the active profile", "assembly logout"),
        ]
    ),
)
def logout(
    ctx: typer.Context,
    json_out: bool = options.json_option(),
) -> None:
    """Clear stored credentials for the active profile."""

    def body(state: AppState, json_mode: bool) -> None:
        profile = state.resolve_profile()
        # Look before clearing so the report is truthful: "Signed out" on a fresh
        # machine (or a typo'd --profile) would claim something happened when
        # nothing was stored. Still exit 0 either way — logout is idempotent.
        had_key = config.get_api_key(profile) is not None
        had_session = config.get_session(profile) is not None
        cleared = had_key or had_session
        config.clear_api_key(profile)
        config.clear_session(profile)

        def render(_d: dict[str, object]) -> object:
            if cleared:
                return (
                    output.success(f"Signed out of {escape(profile)}.")
                    + "\n"
                    + output.hint("Run `assembly login` to sign back in.")
                )
            return output.muted(f"No stored credentials for '{profile}' — nothing to clear.")

        data: dict[str, object] = {"logged_out": True, "profile": profile, "cleared": cleared}
        output.emit(data, render, json_mode=json_mode)

    # auto_login=False: signing out while signed out must not start a sign-in.
    run_command(ctx, body, json=json_out, auto_login=False)


@app.command(
    rich_help_panel=help_panels.ACCOUNT,
    epilog=examples_epilog(
        [
            ("Show the active profile and whether its key works", "assembly whoami"),
        ]
    ),
)
def whoami(
    ctx: typer.Context,
    json_out: bool = options.json_option(),
) -> None:
    """Show the active profile and whether its key is usable."""

    def body(state: AppState, json_mode: bool) -> None:
        profile = state.resolve_profile()
        # The full env -> keyring chain (raises NotAuthenticated when empty), so a CI
        # box authenticated via ASSEMBLYAI_API_KEY can use whoami as a preflight check.
        key = state.resolve_api_key()
        masked = output.redact_secret(key)
        env = environments.active().name
        # A network failure must not suppress the local table: profile, env, masked
        # key, and session are all known offline. reachable=None marks "couldn't
        # check" — distinct from False, which means the server rejected the key.
        network_error: APIError | None = None
        reachable: bool | None
        try:
            reachable = client.validate_key(key)
        except APIError as exc:
            reachable = None
            network_error = exc
        session_label = "stored" if config.get_session(profile) else "none"
        account_id = config.get_account_id(profile)

        def status_cell() -> str:
            if reachable:
                return output.success("reachable")
            if reachable is None:
                return output.warn("unreachable (network error)")
            return output.fail("key rejected")

        def render(_d: dict[str, object]) -> Table:
            table = output.detail_table()
            table.add_row("Profile", escape(profile))
            table.add_row("Env", escape(env))
            table.add_row("API key", escape(masked))
            table.add_row("Status", status_cell())
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
        if network_error is not None:
            # Couldn't validate ≠ rejected: surface the network failure with its own
            # suggestion and exit code 1 (api_error), keeping the auth exit 4 reserved
            # for a key the server actually refused.
            raise APIError(
                network_error.message,
                suggestion=(
                    "Check your network connection and retry — the key was not rejected; "
                    "it just couldn't be validated."
                ),
            )
        if not reachable:
            # A rejected key must fail the command (exit 4, the auth code used by
            # NotAuthenticated) so CI can use whoami as a preflight check; the
            # rendered status above still lands on stdout in both modes.
            raise typer.Exit(code=4)

    run_command(ctx, body, json=json_out)
