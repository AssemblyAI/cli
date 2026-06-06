import json
from unittest.mock import patch

from typer.testing import CliRunner

from aai_cli import config
from aai_cli.auth.flow import LoginResult
from aai_cli.commands import keys
from aai_cli.main import app

runner = CliRunner()


def _auth():
    config.set_session("default", session_jwt="jwt", session_token="tok", account_id=42)


def _login_result():
    return LoginResult(
        api_key="sk_from_oauth", session_jwt="jwt", session_token="tok", account_id=42
    )


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


def test_keys_shape_helpers_filter_invalid_values():
    assert keys._project_id({"id": True}) is None
    assert keys._project_id({"id": 7}) == 7
    assert keys._project_id({"id": "8"}) == 8
    assert keys._project_id({"id": "bad"}) is None
    assert keys._project_id({"id": object()}) is None


def test_keys_create_rejects_missing_default_project_object():
    _auth()
    with (
        patch("aai_cli.commands.keys.ams.list_projects", return_value=[{"project": "bad"}]),
        patch("aai_cli.commands.keys.ams.create_token") as create,
    ):
        result = runner.invoke(app, ["keys", "create", "--name", "ci"])
    assert result.exit_code == 1
    create.assert_not_called()


def test_keys_create_rejects_default_project_without_int_id():
    _auth()
    with (
        patch("aai_cli.commands.keys.ams.list_projects", return_value=[{"project": {"id": "bad"}}]),
        patch("aai_cli.commands.keys.ams.create_token") as create,
    ):
        result = runner.invoke(app, ["keys", "create", "--name", "ci"])
    assert result.exit_code == 1
    create.assert_not_called()


def test_keys_list_without_session_runs_login(monkeypatch):
    monkeypatch.setattr("aai_cli.context.run_login_flow", _login_result)
    with patch("aai_cli.commands.keys.ams.list_projects", return_value=[]) as list_projects:
        result = runner.invoke(app, ["keys", "list", "--json"])
    assert result.exit_code == 2
    assert config.get_session("default") == {"jwt": "jwt", "token": "tok"}
    list_projects.assert_not_called()
    assert "Run the same command again" in result.output


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


def test_keys_list_falls_back_to_token_name_when_name_is_null():
    _auth()
    projects = [
        {
            "project": {"id": 1, "name": "Default"},
            "tokens": [
                {
                    "id": 10,
                    "name": None,
                    "token_name": "AssemblyAI CLI",
                    "api_key": "sk_abcdef1234",
                    "is_disabled": False,
                }
            ],
        }
    ]
    with patch("aai_cli.commands.keys.ams.list_projects", return_value=projects):
        result = runner.invoke(app, ["keys", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data[0]["name"] == "AssemblyAI CLI"


def test_keys_rename_calls_ams():
    _auth()
    with patch("aai_cli.commands.keys.ams.rename_token") as rename:
        result = runner.invoke(app, ["keys", "rename", "10", "prod"])
    assert result.exit_code == 0
    rename.assert_called_once_with(42, 10, "prod", "jwt")


def test_keys_list_renders_human_table():
    # Human-readable (non-JSON) render path: the table includes names, project, and
    # the disabled flag rendered as yes/no.
    _auth()
    projects = [
        {
            "project": {"id": 1, "name": "Default"},
            "tokens": [
                {"id": 10, "name": "ci", "api_key": "sk_abcdef1234", "is_disabled": False},
                {"id": 11, "name": "old", "api_key": "sk_zzzzzz9999", "is_disabled": True},
            ],
        }
    ]
    with (
        patch("aai_cli.output._is_agentic", return_value=False),
        patch("aai_cli.commands.keys.ams.list_projects", return_value=projects),
    ):
        result = runner.invoke(app, ["keys", "list"])
    assert result.exit_code == 0
    assert "ci" in result.output
    assert "Default" in result.output
    assert "yes" in result.output  # the disabled key
    assert "no" in result.output  # the enabled key
    assert "sk_abcdef1234" not in result.output  # masked


def test_keys_create_rejects_empty_project_list():
    # No projects at all -> clean APIError, create_token never called.
    _auth()
    with (
        patch("aai_cli.commands.keys.ams.list_projects", return_value=[]),
        patch("aai_cli.commands.keys.ams.create_token") as create,
    ):
        result = runner.invoke(app, ["keys", "create", "--name", "ci"])
    assert result.exit_code == 1
    create.assert_not_called()


def test_keys_create_with_explicit_project_skips_lookup():
    # Passing --project bypasses the default-project lookup entirely.
    _auth()
    created = {"id": 5, "name": "ci", "api_key": "sk_explicit999", "is_disabled": False}
    with (
        patch("aai_cli.commands.keys.ams.list_projects") as list_projects,
        patch("aai_cli.commands.keys.ams.create_token", return_value=created) as create,
    ):
        result = runner.invoke(app, ["keys", "create", "--name", "ci", "--project", "9"])
    assert result.exit_code == 0
    list_projects.assert_not_called()
    create.assert_called_once_with(42, 9, "ci", "jwt")
