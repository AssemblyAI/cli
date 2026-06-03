from unittest.mock import patch

from typer.testing import CliRunner

from assemblyai_cli import config
from assemblyai_cli.main import app

runner = CliRunner()


def test_login_with_api_key_flag_stores_key():
    with patch("assemblyai_cli.commands.login.client.validate_key", return_value=True):
        result = runner.invoke(app, ["login", "--api-key", "sk_flag"])
    assert result.exit_code == 0
    assert config.get_api_key("default") == "sk_flag"


def test_login_rejects_invalid_key():
    with patch("assemblyai_cli.commands.login.client.validate_key", return_value=False):
        result = runner.invoke(app, ["login", "--api-key", "sk_bad"])
    assert result.exit_code != 0
    assert config.get_api_key("default") is None


def test_login_stores_under_named_profile():
    with patch("assemblyai_cli.commands.login.client.validate_key", return_value=True):
        result = runner.invoke(app, ["--profile", "staging", "login", "--api-key", "sk_s"])
    assert result.exit_code == 0
    assert config.get_api_key("staging") == "sk_s"


def test_whoami_reports_authenticated():
    import json

    config.set_api_key("default", "sk_1234567890")
    with patch("assemblyai_cli.commands.login.client.validate_key", return_value=True):
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
