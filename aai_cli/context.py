from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import NoReturn

import keyring.errors
import typer

from aai_cli import config, environments, output, telemetry, update_check
from aai_cli.auth import run_login_flow
from aai_cli.environments import Environment
from aai_cli.errors import APIError, CLIError, NotAuthenticated


@dataclass
class AppState:
    """Request-scoped CLI state (the global --profile / --env) and the single place
    that turns it into a concrete profile, environment, or session.

    Centralizing resolution here keeps the precedence rules in one spot instead of
    being re-derived per command. The module-level ``resolve_*`` functions below are
    thin adapters retained for existing call sites and tests.
    """

    profile: str | None = None
    env: str | None = None
    quiet: bool = False

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

    def resolve_session(self) -> tuple[int, str]:
        """Account id + Stytch session JWT for AMS self-service commands.

        These endpoints authenticate with the browser-login session, not the API
        key. Raises NotAuthenticated when the active profile has no stored session
        (e.g. it was set up with `assembly login --api-key`, or the session expired).
        """
        from aai_cli.errors import NotAuthenticated

        profile = self.resolve_profile()
        session = config.get_session(profile)
        account_id = config.get_account_id(profile)
        if session is None or account_id is None:
            # The inherited default suggestion offers ASSEMBLYAI_API_KEY, which can
            # never satisfy these endpoints — they authenticate with the browser
            # session, not the API key — so spell out the only fix that works.
            raise NotAuthenticated(
                "These commands need a browser login. Run 'assembly login' (without --api-key).",
                suggestion=(
                    "Run 'assembly login' to sign in via your browser — an API key alone "
                    "can't access account commands."
                ),
            )
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
        elif (from_env := os.environ.get("AAI_ENV")) is not None:
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


def resolve_profile(state: AppState) -> str:
    return state.resolve_profile()


def resolve_environment(state: AppState) -> Environment:
    return state.resolve_environment()


def resolve_session(state: AppState) -> tuple[int, str]:
    return state.resolve_session()


def env_override_warning(state: AppState) -> str | None:
    return state.env_override_warning()


def persist_browser_login(profile: str, env: str, *, json_mode: bool = False) -> None:
    """Run the browser login flow and persist its credentials for `profile`/`env`.

    ``json_mode`` keeps the flow's stderr progress notes machine-readable under
    ``--json`` (each becomes a ``{"hint": …}`` object instead of prose).
    """
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


def _login_persistence_error(exc: object) -> APIError:
    return APIError(
        f"Signed in, but could not save the credentials locally: {exc}",
        suggestion="Run 'assembly login' again, or check your keyring/config permissions.",
    )


def _rerun_after_login_error() -> CLIError:
    return CLIError(
        "Signed in. Run the command again to continue.",
        error_type="login_required",
        exit_code=4,
        suggestion="Run the same command again.",
    )


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
    return not (os.environ.get(config.ENV_API_KEY) and err.rejected_key)


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
        output.emit_error(login_err, json_mode=json_mode)
        raise typer.Exit(code=login_err.exit_code) from None
    except (OSError, RuntimeError, keyring.errors.KeyringError) as exc:
        persistence_err = _login_persistence_error(exc)
        output.emit_error(persistence_err, json_mode=json_mode)
        raise typer.Exit(code=persistence_err.exit_code) from None
    rerun_err = _rerun_after_login_error()
    output.emit_error(rerun_err, json_mode=json_mode)
    raise typer.Exit(code=rerun_err.exit_code) from None


def run_command(
    ctx: typer.Context,
    fn: Callable[[AppState, bool], None],
    *,
    json: bool = False,
    auto_login: bool = True,
) -> None:
    """Execute a command body, mapping CLIError to clean output + exit code."""
    state: AppState = ctx.obj
    json_mode = output.resolve_json(explicit=json)
    try:
        # Inside the try so telemetry sees the raw CLIError (and its error_type)
        # before it's folded into a typer.Exit below.
        with telemetry.track(ctx.command_path):
            fn(state, json_mode)
        update_check.maybe_notify(json_mode=json_mode)
    except NotAuthenticated as err:
        if not auto_login or not _should_auto_login(err):
            output.emit_error(err, json_mode=json_mode)
            raise typer.Exit(code=err.exit_code) from None
        _auto_login_and_exit(state, json_mode=json_mode)
    except CLIError as err:
        output.emit_error(err, json_mode=json_mode)
        raise typer.Exit(code=err.exit_code) from None
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
        raise typer.Exit(code=internal.exit_code) from exc
