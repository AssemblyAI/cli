import json
import sys

import pytest
import typer
from typer.testing import CliRunner

from aai_cli.app.context import AppState, _interactive_session, run_command
from aai_cli.auth.flow import LoginResult
from aai_cli.core import config, environments
from aai_cli.core.errors import APIError, NotAuthenticated, auth_failure

runner = CliRunner()


def _make_app(body, **run_options):
    app = typer.Typer()

    @app.callback()
    def cb(ctx: typer.Context):
        ctx.obj = AppState()

    @app.command()
    def go(ctx: typer.Context):
        run_command(ctx, body, **run_options)

    return app


def _force_interactive(monkeypatch):
    """Pretend a human is at the terminal (CliRunner/pytest streams are never TTYs)."""
    monkeypatch.setattr("aai_cli.app.context._interactive_session", lambda: True)


class _TtyProbe:
    def __init__(self, tty):
        self._tty = tty

    def isatty(self):
        return self._tty


@pytest.mark.parametrize(
    ("stdin_tty", "stderr_tty", "agentic", "expected"),
    [
        (True, True, False, True),  # a real terminal session
        (False, True, False, False),  # stdin piped/redirected (CI input)
        (True, False, False, False),  # stderr redirected (logged pipeline)
        (True, True, True, False),  # agent/CI env detected despite TTYs
    ],
)
def test_interactive_session_requires_both_ttys_and_no_agent(
    monkeypatch, stdin_tty, stderr_tty, agentic, expected
):
    monkeypatch.setattr(sys, "stdin", _TtyProbe(stdin_tty))
    monkeypatch.setattr(sys, "stderr", _TtyProbe(stderr_tty))
    monkeypatch.setattr("aai_cli.ui.output.is_agentic", lambda: agentic)
    assert _interactive_session() is expected


def test_run_command_skips_auto_login_when_session_not_interactive(monkeypatch):
    # CliRunner/pytest streams are not TTYs, so this is a genuine non-interactive
    # session: no browser login may start (it would bind a port and block 120s),
    # and the ORIGINAL NotAuthenticated must surface with its actionable suggestion.
    monkeypatch.setattr(
        "aai_cli.auth.run_login_flow",
        lambda **_: (_ for _ in ()).throw(AssertionError("non-interactive must not auto-login")),
    )

    def body(state, json_mode):
        raise NotAuthenticated()

    result = runner.invoke(_make_app(body), ["go"])
    assert result.exit_code == 4
    assert "starting browser login" not in result.output
    assert "You're not signed in." in result.output
    assert "assembly login" in result.output
    assert "ASSEMBLYAI_API_KEY" in result.output


def test_run_command_not_interactive_json_keeps_clean_error_shape(monkeypatch):
    monkeypatch.setattr(
        "aai_cli.auth.run_login_flow",
        lambda **_: (_ for _ in ()).throw(AssertionError("non-interactive must not auto-login")),
    )

    def body(state, json_mode):
        raise NotAuthenticated()

    result = runner.invoke(_make_app(body, json=True), ["go"])
    assert result.exit_code == 4
    payload = json.loads(result.output)  # the only output line is machine-readable
    assert payload["error"]["type"] == "not_authenticated"
    assert "assembly login" in payload["error"]["suggestion"]
    assert "ASSEMBLYAI_API_KEY" in payload["error"]["suggestion"]


def test_run_command_auto_login_notice_suppressed_in_json_mode(monkeypatch):
    # Even when auto-login runs, --json stderr must stay machine-readable: the
    # human "starting browser login" prose is suppressed and only the JSON error
    # shape is emitted.
    _force_interactive(monkeypatch)
    monkeypatch.setattr(
        "aai_cli.auth.run_login_flow",
        lambda **_: LoginResult(
            api_key="sk_auto", session_jwt="j", session_token="t", account_id=1
        ),
    )

    def body(state, json_mode):
        raise NotAuthenticated()

    result = runner.invoke(_make_app(body, json=True), ["go"])
    assert result.exit_code == 4
    assert "starting browser login" not in result.output
    payload = json.loads(result.output)
    assert payload["error"]["type"] == "login_required"


def test_run_command_maps_cli_error_to_exit_code():
    def body(state, json_mode):
        raise NotAuthenticated()

    result = runner.invoke(_make_app(body, auto_login=False), ["go"])
    assert result.exit_code == 4


def test_run_command_auto_logs_in_and_asks_for_rerun(monkeypatch):
    _force_interactive(monkeypatch)
    monkeypatch.setattr(
        "aai_cli.auth.run_login_flow",
        lambda **_: LoginResult(
            api_key="sk_auto",
            session_jwt="jwt_auto",
            session_token="tok_auto",
            account_id=42,
        ),
    )
    calls = {"count": 0}

    def body(state, json_mode):
        calls["count"] += 1
        raise NotAuthenticated()

    result = runner.invoke(_make_app(body), ["go"])
    assert result.exit_code == 4
    assert calls["count"] == 1
    assert config.get_session("default") == {"jwt": "jwt_auto", "token": "tok_auto"}
    assert config.get_account_id("default") == 42
    assert config.resolve_api_key() == "sk_auto"
    assert "Run the same command again" in result.output
    # The auto-login notice prints because AppState defaults to non-quiet (quiet=False).
    assert "starting browser login" in result.output


def test_run_command_auto_login_persistence_failure_is_clean(monkeypatch):
    _force_interactive(monkeypatch)
    monkeypatch.setattr(
        "aai_cli.auth.run_login_flow",
        lambda **_: LoginResult(
            api_key="sk_auto",
            session_jwt="jwt_auto",
            session_token="tok_auto",
            account_id=42,
        ),
    )
    monkeypatch.setattr(
        "aai_cli.app.context.config.set_api_key",
        lambda *_args: (_ for _ in ()).throw(OSError("keyring is unavailable")),
    )

    def body(state, json_mode):
        raise NotAuthenticated()

    result = runner.invoke(_make_app(body), ["go"])
    assert result.exit_code == 1
    assert "could not save the credentials" in result.output
    assert "Traceback" not in result.output


def test_run_command_auto_login_persistence_type_error_is_clean(monkeypatch):
    # The TOML writer raises TypeError for a value it can't serialize. The sign-in
    # itself succeeded, so the user must see the "could not save the credentials"
    # message — not the generic "Unexpected error" internal-bug line.
    _force_interactive(monkeypatch)
    monkeypatch.setattr(
        "aai_cli.auth.run_login_flow",
        lambda **_: LoginResult(
            api_key="sk_auto",
            session_jwt="jwt_auto",
            session_token="tok_auto",
            account_id=42,
        ),
    )
    monkeypatch.setattr(
        "aai_cli.app.context.config.set_api_key",
        lambda *_args: (_ for _ in ()).throw(TypeError("not TOML-serializable")),
    )

    def body(state, json_mode):
        raise NotAuthenticated()

    result = runner.invoke(_make_app(body), ["go"])
    assert result.exit_code == 1
    assert "could not save the credentials" in result.output
    assert "Unexpected error" not in result.output


def test_run_command_auto_login_failure_is_clean(monkeypatch):
    _force_interactive(monkeypatch)

    def fail_login(**_kwargs):
        raise APIError("Login failed: the server returned an unexpected response.")

    monkeypatch.setattr("aai_cli.auth.run_login_flow", fail_login)

    def body(state, json_mode):
        raise NotAuthenticated()

    result = runner.invoke(_make_app(body), ["go"])
    assert result.exit_code == 1
    assert "Login failed" in result.output


def test_run_command_auto_login_timeout_maps_to_auth_error(monkeypatch):
    # The loopback timeout is an auth failure (not_authenticated, exit 4), not a
    # generic api_error.
    _force_interactive(monkeypatch)

    def fail_login(**_kwargs):
        raise NotAuthenticated("Login timed out waiting for the browser.")

    monkeypatch.setattr("aai_cli.auth.run_login_flow", fail_login)

    def body(state, json_mode):
        raise NotAuthenticated()

    result = runner.invoke(_make_app(body, json=True), ["go"])
    assert result.exit_code == 4
    payload = json.loads(result.output)
    assert payload["error"]["type"] == "not_authenticated"
    assert "Login timed out" in payload["error"]["message"]


def test_run_command_skips_auto_login_for_rejected_env_key(monkeypatch):
    _force_interactive(monkeypatch)
    monkeypatch.setenv(config.ENV_API_KEY, "sk_bad")
    monkeypatch.setattr(
        "aai_cli.auth.run_login_flow",
        lambda **_: (_ for _ in ()).throw(AssertionError("env key retry cannot be fixed")),
    )

    def body(state, json_mode):
        raise auth_failure()

    result = runner.invoke(_make_app(body), ["go"])
    assert result.exit_code == 4


def test_run_command_runs_body_on_success():
    seen = {}

    def body(state, json_mode):
        seen["ran"] = True

    result = runner.invoke(_make_app(body), ["go"])
    assert result.exit_code == 0
    assert seen.get("ran") is True


def test_env_override_warning_when_flag_contradicts_profile():
    config.set_profile_env("default", "sandbox000")
    assert AppState(env="production").env_override_warning() is not None


def test_env_override_warning_none_when_flag_matches_profile():
    config.set_profile_env("default", "sandbox000")
    assert AppState(env="sandbox000").env_override_warning() is None


def test_env_override_warning_none_without_explicit_flag():
    config.set_profile_env("default", "sandbox000")
    assert AppState(env=None).env_override_warning() is None


def test_env_override_warning_none_when_profile_has_no_env():
    assert AppState(env="production").env_override_warning() is None


def test_env_override_warning_when_aai_env_contradicts_profile(monkeypatch):
    # A silent AAI_ENV swap points a profile's key at the wrong hosts just like --env
    # does, so it must warn too — and name AAI_ENV as the source.
    config.set_profile_env("default", "sandbox000")
    monkeypatch.setenv("AAI_ENV", "production")
    warning = AppState(env=None).env_override_warning()
    assert warning is not None
    assert "AAI_ENV" in warning
    # The warning now ends with an actionable remediation naming how to fix it.
    assert "Unset AAI_ENV" in warning


def test_env_override_warning_flag_beats_aai_env(monkeypatch):
    # An explicit --env wins precedence and is the named source, not AAI_ENV.
    config.set_profile_env("default", "sandbox000")
    monkeypatch.setenv("AAI_ENV", "sandbox000")
    warning = AppState(env="production").env_override_warning()
    assert warning is not None
    assert "--env" in warning
    assert "Drop --env" in warning


def test_env_override_warning_none_when_aai_env_matches_profile(monkeypatch):
    config.set_profile_env("default", "sandbox000")
    monkeypatch.setenv("AAI_ENV", "sandbox000")
    assert AppState(env=None).env_override_warning() is None


def test_resolve_session_returns_account_and_jwt():
    from aai_cli.app.context import AppState
    from aai_cli.core import config

    config.set_session("default", session_jwt="jwt_1", session_token="tok_1", account_id=42)
    account_id, jwt = AppState().resolve_session()
    assert account_id == 42
    assert jwt == "jwt_1"


def test_resolve_session_raises_when_no_session():
    import pytest

    from aai_cli.app.context import AppState
    from aai_cli.core.errors import NotAuthenticated

    with pytest.raises(NotAuthenticated):
        AppState().resolve_session()


def test_resolve_session_raises_when_only_account_id_missing(monkeypatch):
    # A stored session JWT but no account id is still incomplete and must raise,
    # pinning the `session is None or account_id is None` guard (an `and` there would
    # fall through and return a None account id instead of failing cleanly).
    import pytest

    from aai_cli.app.context import AppState
    from aai_cli.core.errors import NotAuthenticated

    monkeypatch.setattr(config, "get_session", lambda _profile: {"jwt": "j", "token": "t"})
    monkeypatch.setattr(config, "get_account_id", lambda _profile: None)
    with pytest.raises(NotAuthenticated):
        AppState().resolve_session()


def test_resolve_session_raises_when_only_jwt_missing(monkeypatch):
    # The mirror case: an account id but no stored session must also raise.
    import pytest

    from aai_cli.app.context import AppState
    from aai_cli.core.errors import NotAuthenticated

    monkeypatch.setattr(config, "get_session", lambda _profile: None)
    monkeypatch.setattr(config, "get_account_id", lambda _profile: 42)
    with pytest.raises(NotAuthenticated):
        AppState().resolve_session()


def test_run_command_auto_logs_in_when_env_key_set_but_error_is_not_a_rejection(monkeypatch):
    # ENV key present but the failure is a generic NotAuthenticated (not a key
    # rejection): a browser login can still fix it, so we DO auto-login. This pins
    # the `and` in _should_auto_login — an `or` would wrongly skip the retry here.
    _force_interactive(monkeypatch)
    monkeypatch.setenv(config.ENV_API_KEY, "sk_env")
    ran = {"login": 0}

    def fake_login(*, json_mode=False):
        ran["login"] += 1
        return LoginResult(api_key="sk_auto", session_jwt="j", session_token="t", account_id=7)

    monkeypatch.setattr("aai_cli.auth.run_login_flow", fake_login)

    def body(state, json_mode):
        raise NotAuthenticated()  # rejected_key is False: not a key rejection

    result = runner.invoke(_make_app(body), ["go"])
    assert ran["login"] == 1  # auto-login was attempted despite the env key
    assert result.exit_code == 4
    assert "Run the same command again" in result.output


def test_appstate_methods_are_the_single_source_of_truth():
    # The precedence (default profile + default env) must hold.
    config.set_profile_env("default", "sandbox000")
    state = AppState(env="production")

    assert state.resolve_profile() == "default"
    assert state.resolve_environment().name == "production"  # --env wins over profile
    assert state.env_override_warning() is not None  # production contradicts sandbox000


def test_appstate_environment_falls_back_to_default():
    assert AppState().resolve_environment().name == environments.DEFAULT_ENV


def test_run_command_maps_unexpected_exception_to_clean_internal_error():
    # The last-resort handler: a bug surfaces as one clean line, never a traceback.
    def body(state, json_mode):
        raise ValueError("kaboom")

    result = runner.invoke(_make_app(body), ["go"])
    assert result.exit_code == 1
    assert "Unexpected error: kaboom" in result.output
    assert "Traceback" not in result.output
    assert "report it" in result.output


def test_run_command_unexpected_exception_keeps_json_error_shape():
    def body(state, json_mode):
        raise ValueError("kaboom")

    result = runner.invoke(_make_app(body, json=True), ["go"])
    assert result.exit_code == 1
    assert json.loads(result.output)["error"]["type"] == "internal_error"


def test_run_command_lets_typer_exit_pass_through():
    def body(state, json_mode):
        raise typer.Exit(code=7)

    result = runner.invoke(_make_app(body), ["go"])
    assert result.exit_code == 7
    assert "Unexpected error" not in result.output


def test_run_command_lets_broken_pipe_propagate():
    # The closed-pipe contract is owned by main.run(); run_command must not eat it.
    def body(state, json_mode):
        raise BrokenPipeError()

    result = runner.invoke(_make_app(body), ["go"], standalone_mode=False)
    assert isinstance(result.exception, BrokenPipeError)


def test_resolve_session_suggestion_never_offers_api_key_env_var():
    # The inherited NotAuthenticated suggestion offers ASSEMBLYAI_API_KEY, which can
    # never satisfy AMS session commands (they authenticate with the browser-session
    # JWT, not the API key). The session-specific suggestion must say what actually
    # works and drop the dead-end env-var advice.
    with pytest.raises(NotAuthenticated) as exc:
        AppState().resolve_session()
    assert exc.value.suggestion is not None
    assert "browser" in exc.value.suggestion
    assert "API key alone" in exc.value.suggestion
    assert "ASSEMBLYAI_API_KEY" not in exc.value.suggestion


def test_resolve_profile_rejects_invalid_explicit_profile_fast():
    # Validated at resolution time (the root callback), so a typo'd --profile is a
    # fast exit-2 before any network round-trip, not a keyring-write-time failure.
    from aai_cli.core.errors import CLIError

    with pytest.raises(CLIError) as exc:
        AppState(profile="bad name!").resolve_profile()
    assert exc.value.exit_code == 2
    assert exc.value.message.startswith("Invalid profile name")


def test_resolve_profile_returns_valid_explicit_profile():
    assert AppState(profile="staging").resolve_profile() == "staging"


def test_persist_browser_login_passes_json_mode_to_flow(monkeypatch):
    from aai_cli.app.context import persist_browser_login

    seen = {}

    def fake(*, json_mode):
        seen["json_mode"] = json_mode
        return LoginResult(api_key="sk", session_jwt="j", session_token="t", account_id=1)

    monkeypatch.setattr("aai_cli.auth.run_login_flow", fake)
    persist_browser_login("default", "production")
    assert seen["json_mode"] is False  # human prose stays the default
    persist_browser_login("default", "production", json_mode=True)
    assert seen["json_mode"] is True  # --json reaches the flow's stderr notes
