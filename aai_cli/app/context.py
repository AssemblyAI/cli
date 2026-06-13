from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import NoReturn

import keyring.errors
import typer

from aai_cli.core import config, debuglog, env, environments, telemetry
from aai_cli.core.environments import Environment
from aai_cli.core.errors import APIError, CLIError, NotAuthenticated
from aai_cli.ui import output, update_check


@dataclass
class AppState:
    """Request-scoped CLI state (the global --profile / --env) and the single place
    that turns it into a concrete profile, environment, session, or API key.

    Centralizing resolution here keeps the precedence rules in one spot instead of
    being re-derived per command.
    """

    profile: str | None = None
    env: str | None = None
    quiet: bool = False
    # Set by the root callback when config.toml is unreadable during environment
    # resolution: most commands re-raise it (run_command), but `config path` — the
    # command you reach for to *find* the broken file — tolerates it.
    deferred_config_error: CLIError | None = None

    def resolve_profile(self) -> str:
        """The profile to act on: explicit --profile, else the active profile.

        An explicit ``--profile`` is validated here, at resolution time, so a typo'd
        name is a fast usage error in the root callback — before any network
        round-trip — instead of only failing at keyring-write time after the work.
        """
        if self.profile is not None:
            config.validate_profile(self.profile)
            return self.profile
        return config.get_active_profile()

    def resolve_environment(self) -> Environment:
        """The backend environment: --env > AAI_ENV > the profile's stored env > default."""
        profile_env = config.get_profile_env(self.resolve_profile())
        return environments.resolve(self.env, profile_env)

    def resolve_api_key(self) -> str:
        """The API key for SDK/gateway calls: ASSEMBLYAI_API_KEY, else the profile's
        keyring entry. Raises NotAuthenticated when neither is set."""
        return config.resolve_api_key(profile=self.profile)

    def resolve_session(self) -> tuple[int, str]:
        """Account id + Stytch session JWT for AMS self-service commands.

        These endpoints authenticate with the browser-login session, not the API
        key. Raises NotAuthenticated when the active profile has no stored session
        (e.g. it was set up with `assembly login --api-key`, or the session expired).
        """
        profile = self.resolve_profile()
        session = config.get_session(profile)
        account_id = config.get_account_id(profile)
        if session is None or account_id is None:
            # The inherited default suggestion offers ASSEMBLYAI_API_KEY, which can
            # never satisfy these endpoints — they authenticate with the browser
            # session, not the API key — so spell out the only fix that works.
            raise NotAuthenticated(
                "Account commands read account-level data and authenticate with your "
                "browser-login session, which this profile doesn't have.",
                suggestion=(
                    "Run 'assembly login' to sign in via your browser. Unlike transcription "
                    "commands (which work with an API key), account data like balance, usage, "
                    "and keys needs that session."
                ),
            )
        # Registered like the API key in config.resolve_api_key: -v/-vv diagnostics
        # must never print the session JWT in clear.
        debuglog.register_secret(session["jwt"])
        return account_id, session["jwt"]

    def env_override_warning(self) -> str | None:
        """A warning when the selected environment contradicts the profile's stored env.

        The stored key was minted against the profile's environment, so pointing the
        client at a different one sends it to hosts that key won't authenticate to. This
        catches both an explicit ``--env`` and the easier-to-miss case of an inherited
        ``AAI_ENV`` silently swapping the environment, and names which source selected it.
        """
        if self.env is not None:
            source, selected = "--env", self.env
        elif (from_env := env.get("AAI_ENV")) is not None:
            source, selected = "AAI_ENV", from_env
        else:
            return None
        profile = self.resolve_profile()
        profile_env = config.get_profile_env(profile)
        if profile_env is None or profile_env == selected:
            return None
        fix = "Unset AAI_ENV" if source == "AAI_ENV" else "Drop --env"
        return (
            f"Using {source} {selected}, but profile '{profile}' was set up for "
            f"{profile_env}; its stored key may be rejected by {selected}. "
            f"{fix}, or run 'assembly login' for {selected}."
        )


def persist_browser_login(profile: str, env: str, *, json_mode: bool = False) -> None:
    """Run the browser login flow and persist its credentials for `profile`/`env`.

    ``json_mode`` keeps the flow's stderr progress notes machine-readable under
    ``--json`` (each becomes a ``{"hint": …}`` object instead of prose).
    """
    # Imported here, not at module top: context is on every command's import path,
    # and the auth package (httpx, Stytch discovery, loopback server) is only needed
    # when a browser login actually starts.
    from aai_cli.auth import run_login_flow

    result = run_login_flow(json_mode=json_mode)
    config.persist_login(
        profile,
        api_key=result.api_key,
        env=env,
        session_jwt=result.session_jwt,
        session_token=result.session_token,
        account_id=result.account_id,
    )


def _persist_browser_login(state: AppState, *, json_mode: bool) -> None:
    persist_browser_login(state.resolve_profile(), environments.active().name, json_mode=json_mode)


def _fail(err: CLIError, *, json_mode: bool) -> NoReturn:
    """Emit a CLIError and exit with its code — the one error-exit shape every
    failure path in this module shares."""
    output.emit_error(err, json_mode=json_mode)
    raise typer.Exit(code=err.exit_code) from None


def _interactive_session() -> bool:
    """True only when a human can complete a browser login: stdin and stderr are both
    real TTYs and no agent/CI context is detected (`output.is_agentic`)."""
    return sys.stdin.isatty() and sys.stderr.isatty() and not output.is_agentic()


def _should_auto_login(err: NotAuthenticated) -> bool:
    # CI/pipelines/agents have no human to finish a browser sign-in; starting one
    # would bind a loopback port and block for up to two minutes. Surface the
    # original NotAuthenticated (with its 'assembly login' / ASSEMBLYAI_API_KEY
    # suggestion) instead.
    if not _interactive_session():
        return False
    # An invalid ASSEMBLYAI_API_KEY would still take precedence after browser login,
    # so retrying cannot fix that case. `rejected_key` is the structured marker set
    # by auth_failure(); auth-owning commands (login/logout) opt out at their
    # run_command call site with auto_login=False instead of being name-matched here.
    return not (env.get(config.ENV_API_KEY) and err.rejected_key)


def _auto_login_and_exit(state: AppState, *, json_mode: bool) -> NoReturn:
    """Run the browser login for an unauthenticated command, then exit.

    Always raises typer.Exit: with the login error's code on failure, or the
    "signed in — run the command again" code (4) on success.
    """
    try:
        # Suppressed in json_mode too: --json stderr must stay machine-readable,
        # never mix human prose into it.
        if not state.quiet and not json_mode:
            output.error_console.print(
                "[aai.muted]Not signed in; starting browser login.[/aai.muted]"
            )
        _persist_browser_login(state, json_mode=json_mode)
    except CLIError as login_err:
        _fail(login_err, json_mode=json_mode)
    except (OSError, RuntimeError, TypeError, keyring.errors.KeyringError) as exc:
        # TypeError covers a value the TOML writer can't serialize: the login itself
        # succeeded, so the user must see "could not save", not "unexpected error".
        _fail(
            APIError(
                f"Signed in, but could not save the credentials locally: {exc}",
                suggestion="Run 'assembly login' again, or check your keyring/config permissions.",
            ),
            json_mode=json_mode,
        )
    _fail(
        CLIError(
            "Signed in. Run the command again to continue.",
            error_type="login_required",
            exit_code=4,
            suggestion="Run the same command again.",
        ),
        json_mode=json_mode,
    )


def run_command(
    ctx: typer.Context,
    fn: Callable[[AppState, bool], None],
    *,
    json: bool = False,
    auto_login: bool = True,
    tolerate_unreadable_config: bool = False,
) -> None:
    """Execute a command body, mapping CLIError to clean output + exit code.

    `tolerate_unreadable_config` lets a command (only `config path`) run even when the
    root callback deferred a corrupt-config error, so it can still report the file's
    location; every other command re-raises that error here.
    """
    state: AppState = ctx.obj
    json_mode = output.resolve_json(explicit=json)
    deferred = state.deferred_config_error
    if deferred is not None and not tolerate_unreadable_config:
        # The root callback couldn't read config.toml. Surface that for ordinary
        # commands (which depend on it) the same way the callback used to — emit and
        # exit, without telemetry/update-check, both of which would just re-parse it.
        _fail(deferred, json_mode=json_mode)
    try:
        if deferred is not None:
            # `config path` opted in (tolerate_unreadable_config): it reports a
            # contents-independent location, so run just the body and skip the
            # telemetry/update-check wrappers that re-parse the broken config.
            fn(state, json_mode)
        else:
            # Inside the try so telemetry sees the raw CLIError (and its error_type)
            # before it's folded into a typer.Exit below.
            with telemetry.track(ctx.command_path):
                fn(state, json_mode)
            update_check.maybe_notify(json_mode=json_mode)
    except NotAuthenticated as err:
        if not auto_login or not _should_auto_login(err):
            _fail(err, json_mode=json_mode)
        _auto_login_and_exit(state, json_mode=json_mode)
    except CLIError as err:
        _fail(err, json_mode=json_mode)
    except (typer.Exit, typer.Abort, BrokenPipeError):
        # Deliberate control flow (and the closed-pipe contract handled in main.run);
        # these must reach Click/the entry point untouched.
        raise
    except Exception as exc:
        # Last resort: a bug must surface as one clean line (and the JSON error shape
        # under --json), never as a raw traceback.
        internal = CLIError(
            f"Unexpected error: {exc}",
            error_type="internal_error",
            suggestion=(
                "This looks like a bug in the CLI — please report it at "
                "https://github.com/AssemblyAI/cli/issues."
            ),
        )
        output.emit_error(internal, json_mode=json_mode)
        # `from exc`, unlike _fail's `from None`: the original exception is a bug
        # worth keeping on the chain for anyone re-raising with tracebacks enabled.
        raise typer.Exit(code=internal.exit_code) from exc
