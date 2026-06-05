import json
from unittest.mock import patch

from typer.testing import CliRunner

from aai_cli import config
from aai_cli.main import app

runner = CliRunner()


def _auth():
    config.set_session("default", session_jwt="jwt", session_token="tok", account_id=42)


def test_keys_list_flattens_tokens():
    _auth()
    projects = [
        {
            "project": {"id": 1, "name": "Default"},
            "tokens": [{"id": 10, "name": "ci", "api_key": "sk_abcdef1234", "is_disabled": False}],
        }
    ]
    with patch("aai_cli.commands.keys.ams.list_projects", return_value=projects):
        result = runner.invoke(app, ["keys", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data[0]["id"] == 10
    assert "sk_abcdef1234" not in result.output  # api key is masked


def test_keys_list_requires_session():
    result = runner.invoke(app, ["keys", "list"])
    assert result.exit_code == 2


def test_keys_create_prints_new_key():
    _auth()
    projects = [{"project": {"id": 1, "name": "Default"}, "tokens": []}]
    created = {
        "id": 11,
        "project_id": 1,
        "name": "ci",
        "api_key": "sk_newkey9999",
        "is_disabled": False,
    }
    with (
        patch("aai_cli.commands.keys.ams.list_projects", return_value=projects),
        patch("aai_cli.commands.keys.ams.create_token", return_value=created) as create,
    ):
        result = runner.invoke(app, ["keys", "create", "--name", "ci"])
    assert result.exit_code == 0
    assert "sk_newkey9999" in result.output
    create.assert_called_once_with(42, 1, "ci", "jwt")


def test_keys_rename_calls_ams():
    _auth()
    with patch("aai_cli.commands.keys.ams.rename_token") as rename:
        result = runner.invoke(app, ["keys", "rename", "10", "prod"])
    assert result.exit_code == 0
    rename.assert_called_once_with(42, 10, "prod", "jwt")
