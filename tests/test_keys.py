import json

from typer.testing import CliRunner

from aai_cli import config
from aai_cli.auth.flow import LoginResult
from aai_cli.commands import keys
from aai_cli.main import app

runner = CliRunner()


def _auth():
    config.set_session("default", session_jwt="jwt", session_token="tok", account_id=42)


def _login_result(*, json_mode=False):
    return LoginResult(
        api_key="sk_from_oauth", session_jwt="jwt", session_token="tok", account_id=42
    )


def test_keys_list_flattens_tokens(mocker):
    _auth()
    projects = [
        {
            "project": {"id": 1, "name": "Default"},
            "tokens": [{"id": 10, "name": "ci", "api_key": "sk_abcdef1234", "is_disabled": False}],
        }
    ]
    mocker.patch("aai_cli.commands.keys.ams.list_projects", autospec=True, return_value=projects)
    result = runner.invoke(app, ["keys", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data[0]["id"] == 10
    assert "sk_abcdef1234" not in result.output  # api key is masked


def test_keys_list_renders_table_human(mocker):
    _auth()
    projects = [
        {
            "project": {"id": 1, "name": "Default"},
            "tokens": [{"id": 10, "name": "ci", "api_key": "sk_abcdef1234", "is_disabled": False}],
        }
    ]
    mocker.patch("aai_cli.commands.keys.ams.list_projects", autospec=True, return_value=projects)
    result = runner.invoke(app, ["keys", "list"])
    assert result.exit_code == 0
    assert "ci" in result.output and "Default" in result.output
    assert "sk_abcdef1234" not in result.output  # masked in the human table too


def test_keys_shape_helpers_filter_invalid_values():
    assert keys._project_id({"id": True}) is None
    assert keys._project_id({"id": 7}) == 7
    assert keys._project_id({"id": "8"}) == 8
    assert keys._project_id({"id": "bad"}) is None
    assert keys._project_id({"id": object()}) is None


def test_keys_create_rejects_missing_default_project_object(mocker):
    _auth()
    mocker.patch(
        "aai_cli.commands.keys.ams.list_projects", autospec=True, return_value=[{"project": "bad"}]
    )
    create = mocker.patch("aai_cli.commands.keys.ams.create_token", autospec=True)
    result = runner.invoke(app, ["keys", "create", "--name", "ci"])
    assert result.exit_code == 1
    create.assert_not_called()


def test_keys_create_rejects_default_project_without_int_id(mocker):
    _auth()
    mocker.patch(
        "aai_cli.commands.keys.ams.list_projects",
        autospec=True,
        return_value=[{"project": {"id": "bad"}}],
    )
    create = mocker.patch("aai_cli.commands.keys.ams.create_token", autospec=True)
    result = runner.invoke(app, ["keys", "create", "--name", "ci"])
    assert result.exit_code == 1
    create.assert_not_called()


def test_keys_list_without_session_runs_login(monkeypatch, mocker):
    monkeypatch.setattr("aai_cli.context._interactive_session", lambda: True)
    monkeypatch.setattr("aai_cli.auth.run_login_flow", _login_result)
    list_projects = mocker.patch(
        "aai_cli.commands.keys.ams.list_projects", autospec=True, return_value=[]
    )
    result = runner.invoke(app, ["keys", "list", "--json"])
    assert result.exit_code == 4
    assert config.get_session("default") == {"jwt": "jwt", "token": "tok"}
    list_projects.assert_not_called()
    assert "Run the same command again" in result.output


def test_keys_create_prints_new_key(mocker):
    _auth()
    projects = [{"project": {"id": 1, "name": "Default"}, "tokens": []}]
    created = {
        "id": 11,
        "project_id": 1,
        "name": "ci",
        "api_key": "sk_newkey9999",
        "is_disabled": False,
    }
    mocker.patch("aai_cli.commands.keys.ams.list_projects", autospec=True, return_value=projects)
    create = mocker.patch(
        "aai_cli.commands.keys.ams.create_token", autospec=True, return_value=created
    )
    result = runner.invoke(app, ["keys", "create", "--name", "ci"])
    assert result.exit_code == 0
    assert "sk_newkey9999" in result.output
    create.assert_called_once_with(42, 1, "ci", "jwt")


def test_keys_list_falls_back_to_token_name_when_name_is_null(mocker):
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
    mocker.patch("aai_cli.commands.keys.ams.list_projects", autospec=True, return_value=projects)
    result = runner.invoke(app, ["keys", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data[0]["name"] == "AssemblyAI CLI"


def test_keys_rename_calls_ams(mocker):
    _auth()
    rename = mocker.patch("aai_cli.commands.keys.ams.rename_token", autospec=True)
    result = runner.invoke(app, ["keys", "rename", "10", "prod"])
    assert result.exit_code == 0
    rename.assert_called_once_with(42, 10, "prod", "jwt")


def test_keys_list_renders_human_table(mocker):
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
    mocker.patch("aai_cli.commands.keys.ams.list_projects", autospec=True, return_value=projects)
    result = runner.invoke(app, ["keys", "list"])
    assert result.exit_code == 0
    assert "ci" in result.output
    assert "Default" in result.output
    assert "yes" in result.output  # the disabled key
    assert "no" in result.output  # the enabled key
    assert "sk_abcdef1234" not in result.output  # masked


def test_keys_create_rejects_empty_project_list(mocker):
    # No projects at all -> clean APIError, create_token never called.
    _auth()
    mocker.patch("aai_cli.commands.keys.ams.list_projects", autospec=True, return_value=[])
    create = mocker.patch("aai_cli.commands.keys.ams.create_token", autospec=True)
    result = runner.invoke(app, ["keys", "create", "--name", "ci"])
    assert result.exit_code == 1
    create.assert_not_called()
    # The error now carries an actionable remediation (it previously had none).
    assert "dashboard" in result.output


def test_keys_list_empty_shows_human_empty_state(mocker):
    _auth()
    # A project with no tokens yields no rows -> a friendly empty state, not a bare header.
    projects = [{"project": {"id": 1, "name": "Default"}, "tokens": []}]
    mocker.patch("aai_cli.commands.keys.ams.list_projects", autospec=True, return_value=projects)
    result = runner.invoke(app, ["keys", "list"])
    assert result.exit_code == 0
    assert "No API keys found." in result.output


def test_keys_create_with_explicit_project_skips_lookup(mocker):
    # Passing --project bypasses the default-project lookup entirely.
    _auth()
    created = {"id": 5, "name": "ci", "api_key": "sk_explicit999", "is_disabled": False}
    list_projects = mocker.patch("aai_cli.commands.keys.ams.list_projects", autospec=True)
    create = mocker.patch(
        "aai_cli.commands.keys.ams.create_token", autospec=True, return_value=created
    )
    result = runner.invoke(app, ["keys", "create", "--name", "ci", "--project", "9"])
    assert result.exit_code == 0
    list_projects.assert_not_called()
    create.assert_called_once_with(42, 9, "ci", "jwt")


def test_keys_create_rejects_empty_name(monkeypatch, mocker):
    # Local validation fires before session resolution or any AMS call — even when
    # not logged in (no login flow may start for a request that can never be valid).
    monkeypatch.setattr("aai_cli.context._interactive_session", lambda: True)
    monkeypatch.setattr(
        "aai_cli.auth.run_login_flow",
        lambda **_: (_ for _ in ()).throw(AssertionError("login must not start")),
    )
    create = mocker.patch("aai_cli.commands.keys.ams.create_token", autospec=True)
    result = runner.invoke(app, ["keys", "create", "--name", ""])
    assert result.exit_code == 2
    assert "must not be empty" in result.output
    create.assert_not_called()


def test_keys_create_rejects_whitespace_only_name(mocker):
    _auth()
    create = mocker.patch("aai_cli.commands.keys.ams.create_token", autospec=True)
    result = runner.invoke(app, ["keys", "create", "--name", "   ", "--json"])
    assert result.exit_code == 2
    assert json.loads(result.output)["error"]["type"] == "usage_error"
    create.assert_not_called()


def test_bare_keys_command_shows_subcommand_help():
    # `assembly keys` with no subcommand renders the sub-app help (no_args_is_help)
    # instead of a bare "Missing command." error.
    result = runner.invoke(app, ["keys"])
    assert "list" in result.output
    assert "create" in result.output
    assert "rename" in result.output
    assert "Missing command" not in result.output
