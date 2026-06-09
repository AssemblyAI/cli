import typer
from typer.testing import CliRunner

from aai_cli import config, environments
from aai_cli.auth.flow import LoginResult
from aai_cli.context import AppState, env_override_warning, run_command
from aai_cli.errors import APIError, NotAuthenticated, auth_failure

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


def test_run_command_maps_cli_error_to_exit_code():
    def body(state, json_mode):
        raise NotAuthenticated()

    result = runner.invoke(_make_app(body, auto_login=False), ["go"])
    assert result.exit_code == 4


def test_run_command_auto_logs_in_and_asks_for_rerun(monkeypatch):
    monkeypatch.setattr(
        "aai_cli.context.run_login_flow",
        lambda: LoginResult(
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
    monkeypatch.setattr(
        "aai_cli.context.run_login_flow",
        lambda: LoginResult(
            api_key="sk_auto",
            session_jwt="jwt_auto",
            session_token="tok_auto",
            account_id=42,
        ),
    )
    monkeypatch.setattr(
        "aai_cli.context.config.set_api_key",
        lambda *_args: (_ for _ in ()).throw(OSError("keyring is unavailable")),
    )

    def body(state, json_mode):
        raise NotAuthenticated()

    result = runner.invoke(_make_app(body), ["go"])
    assert result.exit_code == 1
    assert "could not save the credentials" in result.output
    assert "Traceback" not in result.output


def test_run_command_auto_login_failure_is_clean(monkeypatch):
    def fail_login():
        raise APIError("Login timed out waiting for the browser.")

    monkeypatch.setattr("aai_cli.context.run_login_flow", fail_login)

    def body(state, json_mode):
        raise NotAuthenticated()

    result = runner.invoke(_make_app(body), ["go"])
    assert result.exit_code == 1
    assert "Login timed out" in result.output


def test_run_command_skips_auto_login_for_rejected_env_key(monkeypatch):
    monkeypatch.setenv(config.ENV_API_KEY, "sk_bad")
    monkeypatch.setattr(
        "aai_cli.context.run_login_flow",
        lambda: (_ for _ in ()).throw(AssertionError("env key retry cannot be fixed")),
    )

    def body(state, json_mode):
        raise auth_failure()

    result = runner.invoke(_make_app(body), ["go"])
    assert result.exit_code == 4


def test_run_command_never_auto_logs_in_login_command(monkeypatch):
    monkeypatch.setattr(
        "aai_cli.context.run_login_flow",
        lambda: (_ for _ in ()).throw(AssertionError("login command must not auto-login")),
    )

    app = typer.Typer()

    @app.callback()
    def cb(ctx: typer.Context):
        ctx.obj = AppState()

    @app.command(name="login")
    def login_cmd(ctx: typer.Context):
        def body(state, json_mode):
            raise NotAuthenticated()

        run_command(ctx, body)

    result = runner.invoke(app, ["login"])
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
    assert env_override_warning(AppState(env="production")) is not None


def test_env_override_warning_none_when_flag_matches_profile():
    config.set_profile_env("default", "sandbox000")
    assert env_override_warning(AppState(env="sandbox000")) is None


def test_env_override_warning_none_without_explicit_flag():
    config.set_profile_env("default", "sandbox000")
    assert env_override_warning(AppState(env=None)) is None


def test_env_override_warning_none_when_profile_has_no_env():
    assert env_override_warning(AppState(env="production")) is None


def test_env_override_warning_when_aai_env_contradicts_profile(monkeypatch):
    # A silent AAI_ENV swap points a profile's key at the wrong hosts just like --env
    # does, so it must warn too — and name AAI_ENV as the source.
    config.set_profile_env("default", "sandbox000")
    monkeypatch.setenv("AAI_ENV", "production")
    warning = env_override_warning(AppState(env=None))
    assert warning is not None
    assert "AAI_ENV" in warning


def test_env_override_warning_flag_beats_aai_env(monkeypatch):
    # An explicit --env wins precedence and is the named source, not AAI_ENV.
    config.set_profile_env("default", "sandbox000")
    monkeypatch.setenv("AAI_ENV", "sandbox000")
    warning = env_override_warning(AppState(env="production"))
    assert warning is not None
    assert "--env" in warning


def test_env_override_warning_none_when_aai_env_matches_profile(monkeypatch):
    config.set_profile_env("default", "sandbox000")
    monkeypatch.setenv("AAI_ENV", "sandbox000")
    assert env_override_warning(AppState(env=None)) is None


def test_resolve_session_returns_account_and_jwt():
    from aai_cli import config
    from aai_cli.context import AppState, resolve_session

    config.set_session("default", session_jwt="jwt_1", session_token="tok_1", account_id=42)
    account_id, jwt = resolve_session(AppState())
    assert account_id == 42
    assert jwt == "jwt_1"


def test_resolve_session_raises_when_no_session():
    import pytest

    from aai_cli.context import AppState, resolve_session
    from aai_cli.errors import NotAuthenticated

    with pytest.raises(NotAuthenticated):
        resolve_session(AppState())


def test_resolve_session_raises_when_only_account_id_missing(monkeypatch):
    # A stored session JWT but no account id is still incomplete and must raise,
    # pinning the `session is None or account_id is None` guard (an `and` there would
    # fall through and return a None account id instead of failing cleanly).
    import pytest

    from aai_cli.context import AppState, resolve_session
    from aai_cli.errors import NotAuthenticated

    monkeypatch.setattr(config, "get_session", lambda _profile: {"jwt": "j", "token": "t"})
    monkeypatch.setattr(config, "get_account_id", lambda _profile: None)
    with pytest.raises(NotAuthenticated):
        resolve_session(AppState())


def test_resolve_session_raises_when_only_jwt_missing(monkeypatch):
    # The mirror case: an account id but no stored session must also raise.
    import pytest

    from aai_cli.context import AppState, resolve_session
    from aai_cli.errors import NotAuthenticated

    monkeypatch.setattr(config, "get_session", lambda _profile: None)
    monkeypatch.setattr(config, "get_account_id", lambda _profile: 42)
    with pytest.raises(NotAuthenticated):
        resolve_session(AppState())


def test_run_command_auto_logs_in_when_env_key_set_but_error_is_not_a_rejection(monkeypatch):
    # ENV key present but the failure is a generic NotAuthenticated (not a key
    # rejection): a browser login can still fix it, so we DO auto-login. This pins
    # the `and` in _should_auto_login — an `or` would wrongly skip the retry here.
    monkeypatch.setenv(config.ENV_API_KEY, "sk_env")
    ran = {"login": 0}

    def fake_login():
        ran["login"] += 1
        return LoginResult(api_key="sk_auto", session_jwt="j", session_token="t", account_id=7)

    monkeypatch.setattr("aai_cli.context.run_login_flow", fake_login)

    def body(state, json_mode):
        raise NotAuthenticated()  # message != REJECTED_KEY_MESSAGE

    result = runner.invoke(_make_app(body), ["go"])
    assert ran["login"] == 1  # auto-login was attempted despite the env key
    assert result.exit_code == 4
    assert "Run the same command again" in result.output


def test_appstate_methods_are_the_single_source_of_truth():
    # The module-level resolve_* helpers are thin adapters over the AppState methods;
    # both must agree, and the precedence (default profile + default env) must hold.
    config.set_profile_env("default", "sandbox000")
    state = AppState(env="production")

    assert state.resolve_profile() == "default"
    assert state.resolve_environment().name == "production"  # --env wins over profile
    assert state.env_override_warning() == env_override_warning(state)
    assert state.env_override_warning() is not None  # production contradicts sandbox000


def test_appstate_environment_falls_back_to_default():
    assert AppState().resolve_environment().name == environments.DEFAULT_ENV
