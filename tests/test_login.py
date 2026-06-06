from unittest.mock import patch

from typer.testing import CliRunner

from aai_cli import config
from aai_cli.auth.flow import LoginResult
from aai_cli.main import app

runner = CliRunner()


def _fake_login_result(key="sk_from_oauth"):
    return LoginResult(api_key=key, session_jwt="jwt_x", session_token="tok_x", account_id=7)


def test_login_with_api_key_flag_stores_key():
    with patch("aai_cli.commands.login.client.validate_key", return_value=True):
        result = runner.invoke(app, ["login", "--api-key", "sk_flag"])
    assert result.exit_code == 0
    assert config.get_api_key("default") == "sk_flag"


def test_login_rejects_invalid_key():
    with patch("aai_cli.commands.login.client.validate_key", return_value=False):
        result = runner.invoke(app, ["login", "--api-key", "sk_bad"])
    assert result.exit_code != 0
    assert config.get_api_key("default") is None


def test_login_stores_under_named_profile():
    with patch("aai_cli.commands.login.client.validate_key", return_value=True):
        result = runner.invoke(app, ["--profile", "staging", "login", "--api-key", "sk_s"])
    assert result.exit_code == 0
    assert config.get_api_key("staging") == "sk_s"


def test_whoami_reports_authenticated():
    import json

    config.set_api_key("default", "sk_1234567890")
    with patch("aai_cli.commands.login.client.validate_key", return_value=True):
        result = runner.invoke(app, ["whoami", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["profile"] == "default"
    assert data["reachable"] is True
    assert data["api_key"].startswith("sk_") and "…" in data["api_key"]


def test_whoami_unauthenticated_runs_login(monkeypatch):
    monkeypatch.setattr("aai_cli.context.run_login_flow", _fake_login_result)
    with patch("aai_cli.commands.login.client.validate_key", return_value=True) as validate:
        result = runner.invoke(app, ["whoami", "--json"])
    assert result.exit_code == 2
    assert config.get_api_key("default") == "sk_from_oauth"
    validate.assert_not_called()
    assert "Run the same command again" in result.output


def test_logout_clears_key():
    config.set_api_key("default", "sk_1234567890")
    result = runner.invoke(app, ["logout"])
    assert result.exit_code == 0
    assert config.get_api_key("default") is None


def test_login_oauth_flow_stores_returned_key(monkeypatch):
    monkeypatch.setattr("aai_cli.commands.login.run_login_flow", _fake_login_result)
    result = runner.invoke(app, ["login"])
    assert result.exit_code == 0
    assert config.get_api_key("default") == "sk_from_oauth"


def test_login_oauth_persists_session(monkeypatch):
    monkeypatch.setattr("aai_cli.commands.login.run_login_flow", _fake_login_result)
    result = runner.invoke(app, ["login"])
    assert result.exit_code == 0
    assert config.get_session("default") == {"jwt": "jwt_x", "token": "tok_x"}
    assert config.get_account_id("default") == 7


def test_login_api_key_flag_does_not_persist_session():
    with patch("aai_cli.commands.login.client.validate_key", return_value=True):
        result = runner.invoke(app, ["login", "--api-key", "sk_flag"])
    assert result.exit_code == 0
    assert config.get_session("default") is None


def test_login_api_key_flag_clears_prior_browser_session():
    # A profile previously authenticated via browser becomes api-key-only on
    # re-login; the stale session must not linger and silently authenticate
    # account self-service commands as the previous identity.
    config.set_session("default", session_jwt="old_j", session_token="old_t", account_id=5)
    with patch("aai_cli.commands.login.client.validate_key", return_value=True):
        result = runner.invoke(app, ["login", "--api-key", "sk_flag"])
    assert result.exit_code == 0
    assert config.get_session("default") is None
    assert config.get_account_id("default") is None


def test_logout_clears_session():
    config.set_api_key("default", "sk_1234567890")
    config.set_session("default", session_jwt="j", session_token="t", account_id=7)
    result = runner.invoke(app, ["logout"])
    assert result.exit_code == 0
    assert config.get_session("default") is None


def test_login_oauth_flow_failure_exits_nonzero(monkeypatch):
    from aai_cli.errors import APIError

    def boom():
        raise APIError("Login timed out waiting for the browser.")

    monkeypatch.setattr("aai_cli.commands.login.run_login_flow", boom)
    result = runner.invoke(app, ["login"])
    assert result.exit_code != 0
    assert config.get_api_key("default") is None


def test_login_api_key_flag_still_bypasses_oauth(monkeypatch):
    monkeypatch.setattr(
        "aai_cli.commands.login.run_login_flow",
        lambda: (_ for _ in ()).throw(AssertionError("OAuth must not run with --api-key")),
    )
    with patch("aai_cli.commands.login.client.validate_key", return_value=True):
        result = runner.invoke(app, ["login", "--api-key", "sk_flag2"])
    assert result.exit_code == 0
    assert config.get_api_key("default") == "sk_flag2"


def test_login_binds_env_to_profile(monkeypatch):
    monkeypatch.setattr("aai_cli.commands.login.run_login_flow", _fake_login_result)
    result = runner.invoke(app, ["--env", "sandbox000", "login"])
    assert result.exit_code == 0
    assert config.get_api_key("default") == "sk_from_oauth"
    assert config.get_profile_env("default") == "sandbox000"


def test_sandbox_flag_is_shortcut_for_env(monkeypatch):
    monkeypatch.setattr("aai_cli.commands.login.run_login_flow", lambda: _fake_login_result("sk_x"))
    result = runner.invoke(app, ["--sandbox", "login"])
    assert result.exit_code == 0
    assert config.get_profile_env("default") == "sandbox000"


def test_whoami_reports_env():
    import json

    config.set_api_key("default", "sk_1234567890")
    with patch("aai_cli.commands.login.client.validate_key", return_value=True):
        result = runner.invoke(app, ["--env", "production", "whoami", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["env"] == "production"


def test_unknown_env_exits_2():
    result = runner.invoke(app, ["--env", "bogus", "whoami"])
    assert result.exit_code == 2


def test_env_override_prints_warning_to_stderr():
    # The root callback warns when an explicit --env contradicts the profile's stored
    # env (the stored key was minted for a different environment).
    config.set_api_key("default", "sk_1234567890")
    config.set_profile_env("default", "production")
    with patch("aai_cli.commands.login.client.validate_key", return_value=True):
        result = runner.invoke(app, ["--env", "sandbox000", "whoami", "--json"])
    assert result.exit_code == 0
    assert "may be rejected by sandbox000" in result.output


def test_rejected_api_key_has_suggestion(monkeypatch):
    from aai_cli import client

    monkeypatch.setattr(client, "validate_key", lambda key: False)
    result = runner.invoke(app, ["login", "--api-key", "sk_bad", "--json"])
    assert result.exit_code == 1
    assert "Check the key and retry" in result.output


def test_whoami_reports_session_and_account():
    import json

    config.set_api_key("default", "sk_1234567890")
    config.set_session("default", session_jwt="j", session_token="t", account_id=77)
    with patch("aai_cli.commands.login.client.validate_key", return_value=True):
        result = runner.invoke(app, ["whoami", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["account_id"] == 77
    assert data["session"] == "stored"


def test_whoami_session_none_without_browser_login():
    import json

    config.set_api_key("default", "sk_1234567890")
    with patch("aai_cli.commands.login.client.validate_key", return_value=True):
        result = runner.invoke(app, ["whoami", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["session"] == "none"
    assert data["account_id"] is None


def test_whoami_renders_human_table_reachable():
    # Human-readable (non-JSON) render path: the grid lists profile, env, masked key,
    # a reachable status, and the account/session rows.
    config.set_api_key("default", "sk_1234567890")
    config.set_session("default", session_jwt="j", session_token="t", account_id=77)
    with (
        patch("aai_cli.output._is_agentic", return_value=False),
        patch("aai_cli.commands.login.client.validate_key", return_value=True),
    ):
        result = runner.invoke(app, ["whoami"])
    assert result.exit_code == 0
    assert "Profile" in result.output
    assert "default" in result.output
    assert "reachable" in result.output
    assert "77" in result.output
    assert "stored" in result.output
    assert "sk_1234567890" not in result.output  # masked


def test_whoami_renders_human_table_rejected_key():
    # The non-JSON render path also covers the "key rejected" branch and the
    # account/session "none" fallbacks (the em-dash placeholder).
    config.set_api_key("default", "sk_1234567890")
    with (
        patch("aai_cli.output._is_agentic", return_value=False),
        patch("aai_cli.commands.login.client.validate_key", return_value=False),
    ):
        result = runner.invoke(app, ["whoami"])
    assert result.exit_code == 0
    assert "key rejected" in result.output
    assert "none" in result.output
