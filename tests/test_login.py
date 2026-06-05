from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from aai_cli import config
from aai_cli.main import app

runner = CliRunner()


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


def test_whoami_unauthenticated_exits_2():
    result = runner.invoke(app, ["whoami"])
    assert result.exit_code == 2


def test_logout_clears_key():
    config.set_api_key("default", "sk_1234567890")
    result = runner.invoke(app, ["logout"])
    assert result.exit_code == 0
    assert config.get_api_key("default") is None


def test_login_oauth_flow_stores_returned_key(monkeypatch):
    monkeypatch.setattr("aai_cli.commands.login.run_login_flow", lambda: "sk_from_oauth")
    result = runner.invoke(app, ["login"])
    assert result.exit_code == 0
    assert config.get_api_key("default") == "sk_from_oauth"


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
    monkeypatch.setattr("aai_cli.commands.login.run_login_flow", lambda: "sk_from_oauth")
    result = runner.invoke(app, ["--env", "sandbox000", "login"])
    assert result.exit_code == 0
    assert config.get_api_key("default") == "sk_from_oauth"
    assert config.get_profile_env("default") == "sandbox000"


def test_sandbox_flag_is_shortcut_for_env(monkeypatch):
    monkeypatch.setattr("aai_cli.commands.login.run_login_flow", lambda: "sk_x")
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


def test_rejected_api_key_has_suggestion(monkeypatch):
    from typer.testing import CliRunner

    from aai_cli import client
    from aai_cli.main import app

    monkeypatch.setattr(client, "validate_key", lambda key: False)
    result = CliRunner().invoke(app, ["login", "--api-key", "sk_bad", "--json"])
    assert result.exit_code == 1
    assert "Check the key and retry" in result.output


@pytest.mark.parametrize("cmd", ["login", "logout", "whoami"])
def test_auth_commands_help_has_examples(cmd):
    result = CliRunner().invoke(app, [cmd, "--help"])
    assert result.exit_code == 0
    assert "Examples" in result.output
