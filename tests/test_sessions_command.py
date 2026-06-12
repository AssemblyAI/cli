import json

import pytest
from typer.testing import CliRunner

from aai_cli import config
from aai_cli.auth.flow import LoginResult
from aai_cli.commands import sessions
from aai_cli.main import app

runner = CliRunner()


def _auth():
    config.set_session("default", session_jwt="jwt", session_token="tok", account_id=42)


def _login_result(*, json_mode=False):
    return LoginResult(
        api_key="sk_from_oauth", session_jwt="jwt", session_token="tok", account_id=42
    )


def _human(monkeypatch):
    monkeypatch.setattr("aai_cli.output.resolve_json", lambda *, explicit: explicit)


def test_sessions_help_lists_list_before_get():
    # Pins the list-then-get subcommand order `transcripts --help` mirrors.
    result = runner.invoke(app, ["sessions", "--help"])
    assert result.exit_code == 0
    lines = result.output.splitlines()
    list_idx = next(i for i, line in enumerate(lines) if "List recent streaming sessions" in line)
    get_idx = next(i for i, line in enumerate(lines) if "Show details for one" in line)
    assert list_idx < get_idx


def test_sessions_list_renders_rows(mocker):
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
    mocker.patch(
        "aai_cli.commands.sessions.ams.list_streaming", autospec=True, return_value=payload
    )
    result = runner.invoke(app, ["sessions", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data[0]["session_id"] == "s_1"


def test_session_rows_filter_invalid_items():
    assert sessions._session_rows("bad") == []
    assert sessions._session_rows([{"session_id": "s_1"}, "bad"]) == [{"session_id": "s_1"}]


def test_sessions_list_renders_table_human(monkeypatch, mocker):
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
    mocker.patch(
        "aai_cli.commands.sessions.ams.list_streaming", autospec=True, return_value=payload
    )
    result = runner.invoke(app, ["sessions", "list"])
    assert result.exit_code == 0
    assert "s_1" in result.output and "universal" in result.output
    # The created column is normalized to a UTC YYYY-MM-DD HH:MM:SS string (was raw),
    # and the duration must still render (pins `value or ""`).
    assert "2026-06-01 00:00:00" in result.output
    assert "12.0" in result.output


def test_sessions_list_renders_zero_duration_as_zero(monkeypatch, mocker):
    _auth()
    _human(monkeypatch)
    # 0 is a legitimate duration (a session that connected but streamed no audio):
    # it must render as "0", not be coerced to a blank cell like a missing value.
    # Neither row carries created_at, so the duration is the only digit in the table.
    payload = {
        "data": [
            {
                "session_id": "s_one",
                "status": "completed",
                "audio_duration_sec": 0,
                "speech_model": "universal",
            },
            {
                "session_id": "s_two",
                "status": "completed",
                "speech_model": "universal",
            },
        ]
    }
    mocker.patch(
        "aai_cli.commands.sessions.ams.list_streaming", autospec=True, return_value=payload
    )
    result = runner.invoke(app, ["sessions", "list"])
    assert result.exit_code == 0
    assert "0" in result.output


def test_sessions_list_empty_shows_human_empty_state(monkeypatch, mocker):
    _auth()
    _human(monkeypatch)
    mocker.patch(
        "aai_cli.commands.sessions.ams.list_streaming",
        autospec=True,
        return_value={"data": []},
    )
    result = runner.invoke(app, ["sessions", "list"])
    assert result.exit_code == 0
    assert "No streaming sessions yet." in result.output


def test_sessions_list_passes_status_filter(mocker):
    _auth()
    list_streaming = mocker.patch(
        "aai_cli.commands.sessions.ams.list_streaming", autospec=True, return_value={"data": []}
    )
    result = runner.invoke(app, ["sessions", "list", "--status", "error", "--limit", "5"])
    assert result.exit_code == 0
    list_streaming.assert_called_once_with("jwt", limit=5, status="error")


def test_sessions_list_limit_must_be_at_least_one(mocker):
    # min=1 on --limit: 0 and negatives are rejected client-side, before any
    # request (parity with `transcripts list`).
    _auth()
    list_streaming = mocker.patch("aai_cli.commands.sessions.ams.list_streaming", autospec=True)
    for bad in ("0", "-3"):
        result = runner.invoke(app, ["sessions", "list", "--limit", bad])
        assert result.exit_code == 2
        assert "limit" in result.output.lower()
    list_streaming.assert_not_called()


def test_sessions_list_rejects_unknown_status(mocker):
    # --status is a closed choice set; a typo fails instantly instead of silently
    # filtering nothing server-side.
    _auth()
    list_streaming = mocker.patch("aai_cli.commands.sessions.ams.list_streaming", autospec=True)
    result = runner.invoke(app, ["sessions", "list", "--status", "comlpeted"])
    assert result.exit_code == 2
    assert "status" in result.output.lower()
    list_streaming.assert_not_called()


@pytest.mark.parametrize("status", ["created", "completed", "error"])
def test_sessions_list_passes_each_status_value(mocker, status):
    _auth()
    list_streaming = mocker.patch(
        "aai_cli.commands.sessions.ams.list_streaming", autospec=True, return_value={"data": []}
    )
    result = runner.invoke(app, ["sessions", "list", "--status", status])
    assert result.exit_code == 0
    list_streaming.assert_called_once_with("jwt", limit=10, status=status)


def test_sessions_list_without_status_passes_none(mocker):
    _auth()
    list_streaming = mocker.patch(
        "aai_cli.commands.sessions.ams.list_streaming", autospec=True, return_value={"data": []}
    )
    result = runner.invoke(app, ["sessions", "list"])
    assert result.exit_code == 0
    list_streaming.assert_called_once_with("jwt", limit=10, status=None)


def test_sessions_get_renders_detail(monkeypatch, mocker):
    _auth()
    _human(monkeypatch)
    detail = {
        "session_id": "s_1",
        "status": "completed",
        "speech_model": "universal",
        "audio_duration_sec": 30.0,
        "error": None,
    }
    mocker.patch("aai_cli.commands.sessions.ams.get_streaming", autospec=True, return_value=detail)
    result = runner.invoke(app, ["sessions", "get", "s_1"])
    assert result.exit_code == 0
    assert "s_1" in result.output and "universal" in result.output
    # Field labels are humanized (underscores -> spaces) for the detail view.
    assert "speech model" in result.output and "speech_model" not in result.output


def test_sessions_without_session_runs_login(monkeypatch, mocker):
    monkeypatch.setattr("aai_cli.context._interactive_session", lambda: True)
    monkeypatch.setattr("aai_cli.auth.run_login_flow", _login_result)
    list_ = mocker.patch(
        "aai_cli.commands.sessions.ams.list_streaming", autospec=True, return_value={"data": []}
    )
    result = runner.invoke(app, ["sessions", "list", "--json"])
    assert result.exit_code == 4
    assert config.get_session("default") == {"jwt": "jwt", "token": "tok"}
    list_.assert_not_called()
    assert "Run the same command again" in result.output
