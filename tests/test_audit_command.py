import json

from typer.testing import CliRunner

from aai_cli import config
from aai_cli.auth.flow import LoginResult
from aai_cli.commands import audit
from aai_cli.main import app

runner = CliRunner()


def _auth():
    config.set_session("default", session_jwt="jwt", session_token="tok", account_id=42)


def _login_result():
    return LoginResult(
        api_key="sk_from_oauth", session_jwt="jwt", session_token="tok", account_id=42
    )


def _human(monkeypatch):
    monkeypatch.setattr("aai_cli.output.resolve_json", lambda *, explicit: explicit)


def test_audit_renders_rows(mocker):
    _auth()
    payload = {
        "page_details": {"has_more": False},
        "data": [
            {
                "id": 1,
                "log_time": "2026-06-01T12:00:00Z",
                "actor_type": "member",
                "actor_id": 7,
                "action_taken": "token.create",
                "resource_type": "token",
                "resource_id": "10",
            }
        ],
    }
    mocker.patch("aai_cli.commands.audit.ams.list_audit_logs", autospec=True, return_value=payload)
    result = runner.invoke(app, ["audit", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data[0]["action_taken"] == "token.create"


def test_audit_rows_filter_invalid_items():
    assert audit._audit_rows({"data": "bad"}) == []
    assert audit._audit_rows({"data": [{"action_taken": "token.create"}, "bad"]}) == [
        {"action_taken": "token.create"}
    ]


def test_audit_passes_filters(mocker):
    _auth()
    list_logs = mocker.patch(
        "aai_cli.commands.audit.ams.list_audit_logs", autospec=True, return_value={"data": []}
    )
    result = runner.invoke(app, ["audit", "--limit", "5", "--action", "token.create"])
    assert result.exit_code == 0
    list_logs.assert_called_once_with(
        "jwt", limit=5, action_taken="token.create", resource_type=None
    )


def test_audit_human_mode_renders_table(monkeypatch, mocker):
    _auth()
    _human(monkeypatch)
    payload = {
        "data": [
            {
                "id": 1,
                "log_time": "2026-06-01T12:00:00.123456Z",
                "actor_type": "member",
                "actor_id": 7,
                "action_taken": "token__created",
                "resource_type": "token",
                "resource_id": "10",
            },
            {
                "id": 2,
                "log_time": "2026-06-01T12:01:00Z",
                "actor_type": "member",
                "actor_id": 7,
                "action_taken": "login__succeeded",
                "resource_type": None,
            },
        ]
    }
    mocker.patch("aai_cli.commands.audit.ams.list_audit_logs", autospec=True, return_value=payload)
    result = runner.invoke(app, ["audit"])
    assert result.exit_code == 0
    assert "API key created" in result.output
    assert "2026-06-01 12:00:00" in result.output
    assert "token #10" in result.output
    assert "member #7" in result.output
    assert "Login succeeded" not in result.output
    assert "Hidden: 1 login event" in result.output


def test_audit_helpers_format_edge_cases():
    # A raw action that is already a known dotted key is mapped directly (no normalize).
    assert audit._format_action("token.create") == "API key created"
    assert audit._format_action("account__created") == "Account created"
    assert audit._format_action("account__tos.accepted") == "Terms accepted"
    assert audit._format_action("custom_event.name") == "Custom event name"
    assert audit._format_action(None) == "Unknown"
    assert audit._is_login({"action_taken": "login.succeeded"}) is True
    assert audit._actor_label({"actor_type": "system"}) == "system"
    assert audit._actor_label({"actor_type": "system", "actor_id": 1}) == "system #1"
    assert audit._resource_label({}) == ""
    assert audit._resource_label({"resource_type": "api_token"}) == "api token"
    assert audit._audit_rows({"data": "bad"}) == []


def test_audit_human_empty_result(monkeypatch, mocker):
    _auth()
    _human(monkeypatch)
    mocker.patch(
        "aai_cli.commands.audit.ams.list_audit_logs", autospec=True, return_value={"data": []}
    )
    result = runner.invoke(app, ["audit"])
    assert result.exit_code == 0
    assert "No audit events found" in result.output


def test_audit_can_include_login_events(monkeypatch, mocker):
    _auth()
    _human(monkeypatch)
    payload = {
        "data": [
            {
                "id": 1,
                "log_time": "2026-06-01T12:01:00Z",
                "actor_type": "member",
                "actor_id": 7,
                "action_taken": "login__succeeded",
                "resource_type": "account",
                "resource_id": "42",
            }
        ]
    }
    mocker.patch("aai_cli.commands.audit.ams.list_audit_logs", autospec=True, return_value=payload)
    result = runner.invoke(app, ["audit", "--include-logins"])
    assert result.exit_code == 0
    assert "Login succeeded" in result.output
    assert "Hidden:" not in result.output


def test_audit_summarizes_all_login_rows(monkeypatch, mocker):
    _auth()
    _human(monkeypatch)
    payload = {
        "data": [
            {
                "id": 1,
                "log_time": "2026-06-01T12:01:00Z",
                "actor_type": "member",
                "actor_id": 7,
                "action_taken": "login__succeeded",
                "resource_type": "account",
                "resource_id": "42",
            }
        ]
    }
    mocker.patch("aai_cli.commands.audit.ams.list_audit_logs", autospec=True, return_value=payload)
    result = runner.invoke(app, ["audit"])
    assert result.exit_code == 0
    assert "No notable audit events" in result.output
    assert "Hidden: 1 login event" in result.output


def test_audit_without_session_runs_login(monkeypatch, mocker):
    monkeypatch.setattr("aai_cli.context.run_login_flow", _login_result)
    logs = mocker.patch(
        "aai_cli.commands.audit.ams.list_audit_logs", autospec=True, return_value={"data": []}
    )
    result = runner.invoke(app, ["audit", "--json"])
    assert result.exit_code == 4
    assert config.get_session("default") == {"jwt": "jwt", "token": "tok"}
    logs.assert_not_called()
    assert "Run the same command again" in result.output
