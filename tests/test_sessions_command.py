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


def test_sessions_list_renders_rows():
    _auth()
    payload = {
        "page_details": {"has_more": False},
        "data": [
            {
                "session_id": "s_1",
                "status": "completed",
                "created_at": "2026-06-01",
                "audio_duration_sec": 12.0,
                "speech_model": "universal",
            }
        ],
    }
    with patch("aai_cli.commands.sessions.ams.list_streaming", return_value=payload):
        result = runner.invoke(app, ["sessions", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data[0]["session_id"] == "s_1"


def test_sessions_list_renders_table_human(monkeypatch):
    _auth()
    _human(monkeypatch)
    payload = {
        "data": [
            {
                "session_id": "s_1",
                "status": "completed",
                "created_at": "2026-06-01",
                "audio_duration_sec": 12.0,
                "speech_model": "universal",
            }
        ]
    }
    with patch("aai_cli.commands.sessions.ams.list_streaming", return_value=payload):
        result = runner.invoke(app, ["sessions", "list"])
    assert result.exit_code == 0
    assert "s_1" in result.output and "universal" in result.output


def test_sessions_list_passes_status_filter():
    _auth()
    with patch(
        "aai_cli.commands.sessions.ams.list_streaming", return_value={"data": []}
    ) as list_streaming:
        result = runner.invoke(app, ["sessions", "list", "--status", "error", "--limit", "5"])
    assert result.exit_code == 0
    list_streaming.assert_called_once_with("jwt", limit=5, status="error")


def test_sessions_get_renders_detail(monkeypatch):
    _auth()
    _human(monkeypatch)
    detail = {
        "session_id": "s_1",
        "status": "completed",
        "speech_model": "universal",
        "audio_duration_sec": 30.0,
        "error": None,
    }
    with patch("aai_cli.commands.sessions.ams.get_streaming", return_value=detail):
        result = runner.invoke(app, ["sessions", "get", "s_1"])
    assert result.exit_code == 0
    assert "s_1" in result.output and "universal" in result.output


def test_sessions_requires_session():
    result = runner.invoke(app, ["sessions", "list"])
    assert result.exit_code == 2
