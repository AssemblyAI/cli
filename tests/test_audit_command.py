import json
from unittest.mock import patch

from typer.testing import CliRunner

from aai_cli import config
from aai_cli.main import app

runner = CliRunner()


def _auth():
    config.set_session("default", session_jwt="jwt", session_token="tok", account_id=42)


def _human(monkeypatch):
    monkeypatch.setattr("aai_cli.output.resolve_json", lambda *, explicit: explicit)


def test_audit_renders_rows():
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
    with patch("aai_cli.commands.audit.ams.list_audit_logs", return_value=payload):
        result = runner.invoke(app, ["audit", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data[0]["action_taken"] == "token.create"


def test_audit_passes_filters():
    _auth()
    with patch(
        "aai_cli.commands.audit.ams.list_audit_logs", return_value={"data": []}
    ) as list_logs:
        result = runner.invoke(app, ["audit", "--limit", "5", "--action", "token.create"])
    assert result.exit_code == 0
    list_logs.assert_called_once_with(
        "jwt", limit=5, action_taken="token.create", resource_type=None
    )


def test_audit_human_mode_renders_table(monkeypatch):
    _auth()
    _human(monkeypatch)
    payload = {
        "data": [
            {
                "id": 1,
                "log_time": "2026-06-01T12:00:00Z",
                "actor_type": "member",
                "actor_id": 7,
                "action_taken": "token.create",
                "resource_type": "token",
                "resource_id": "10",
            },
            {
                "id": 2,
                "log_time": "2026-06-01T12:01:00Z",
                "actor_type": "member",
                "action_taken": "login",
                "resource_type": None,
            },
        ]
    }
    with patch("aai_cli.commands.audit.ams.list_audit_logs", return_value=payload):
        result = runner.invoke(app, ["audit"])
    assert result.exit_code == 0
    assert "login" in result.output and "token.create" in result.output


def test_audit_requires_session():
    result = runner.invoke(app, ["audit"])
    assert result.exit_code == 2
