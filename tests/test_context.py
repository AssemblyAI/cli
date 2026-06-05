import typer
from typer.testing import CliRunner

from aai_cli import config
from aai_cli.context import AppState, env_override_warning, run_command
from aai_cli.errors import NotAuthenticated

runner = CliRunner()


def _make_app(body):
    app = typer.Typer()

    @app.callback()
    def cb(ctx: typer.Context):
        ctx.obj = AppState()

    @app.command()
    def go(ctx: typer.Context):
        run_command(ctx, body)

    return app


def test_run_command_maps_cli_error_to_exit_code():
    def body(state, json_mode):
        raise NotAuthenticated()

    result = runner.invoke(_make_app(body), ["go"])
    assert result.exit_code == 2


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
