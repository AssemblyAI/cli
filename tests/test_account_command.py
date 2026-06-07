import json
from unittest.mock import patch

from typer.testing import CliRunner

from aai_cli import config
from aai_cli.auth.flow import LoginResult
from aai_cli.commands import account
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


def test_balance_formats_dollars(monkeypatch):
    _auth()
    _human(monkeypatch)
    with patch(
        "aai_cli.commands.account.ams.get_balance",
        return_value={"account_id": 42, "balance_in_cents": 2575},
    ):
        result = runner.invoke(app, ["balance"])
    assert result.exit_code == 0
    assert "$25.75" in result.output


def test_balance_without_session_runs_login(monkeypatch):
    monkeypatch.setattr("aai_cli.context.run_login_flow", _login_result)
    with patch(
        "aai_cli.commands.account.ams.get_balance",
        return_value={"account_id": 42, "balance_in_cents": 2575},
    ) as get_balance:
        result = runner.invoke(app, ["balance", "--json"])
    assert result.exit_code == 4
    assert config.get_session("default") == {"jwt": "jwt", "token": "tok"}
    get_balance.assert_not_called()
    assert "Run the same command again" in result.output


def test_usage_defaults_date_range_and_renders(monkeypatch):
    _auth()
    captured = {}

    def fake_usage(jwt, start, end, window):
        captured["start"], captured["end"] = start, end
        return {
            "usage_items": [
                {
                    "start_timestamp": "2026-05-01",
                    "end_timestamp": "2026-05-02",
                    "total": 12.5,
                    "line_items": [],
                }
            ]
        }

    with patch("aai_cli.commands.account.ams.get_usage", side_effect=fake_usage):
        result = runner.invoke(app, ["usage", "--json"])
    assert result.exit_code == 0
    # both bounds are tz-aware UTC ISO-8601 timestamps, defaulted when not passed
    # (AMS rejects naive datetimes with a 400).
    for bound in (captured["start"], captured["end"]):
        assert bound.endswith("+00:00") and "T" in bound, bound
    # The default range spans exactly the last 30 days (pins `today - timedelta(days=30)`).
    from datetime import datetime as _dt

    start_day = _dt.fromisoformat(captured["start"]).date()
    end_day = _dt.fromisoformat(captured["end"]).date()
    assert (end_day - start_day).days == 30
    data = json.loads(result.output)
    assert data["usage_items"][0]["total"] == 12.5


def test_usage_renders_table_human(monkeypatch):
    _auth()
    _human(monkeypatch)
    payload = {
        "usage_items": [
            {
                "start_timestamp": "2026-05-01",
                "end_timestamp": "2026-05-02",
                "total": 12.5,
                "line_items": [],
            }
        ]
    }
    with patch("aai_cli.commands.account.ams.get_usage", return_value=payload):
        result = runner.invoke(app, ["usage"])
    assert result.exit_code == 0
    assert "2026-05-01" in result.output and "12.5" in result.output


def test_usage_helpers_handle_unparseable_values():
    assert account._parse_usage_timestamp(None) is None
    assert account._parse_usage_timestamp("") is None
    assert account._parse_usage_timestamp("not-a-date") is None
    assert account._format_usage_day(None) == ""


def test_usage_helpers_format_windows_and_line_items():
    assert account._usage_items({"usage_items": "bad"}) == []
    assert account._usage_items({"usage_items": [{"total": 1}, "bad"]}) == [{"total": 1}]
    assert account._window_label({"start_timestamp": "bad"}) == "bad"
    assert (
        account._window_label(
            {
                "start_timestamp": "2026-01-01T00:00:00Z",
                "end_timestamp": "2026-01-03T00:00:00Z",
            }
        )
        == "2026-01-01 to 2026-01-03"
    )
    # Exactly one parseable bound falls back to the single start-day label (pins the
    # `start is None or end is None` guard; an `and` would dereference the None end).
    assert account._window_label({"start_timestamp": "2026-01-01T00:00:00Z"}) == "2026-01-01"
    # A one-day window (end == start + 1 day) collapses to a single day, not a range
    # (pins the `start.date() + timedelta(days=1)`).
    assert (
        account._window_label(
            {
                "start_timestamp": "2026-01-01T00:00:00Z",
                "end_timestamp": "2026-01-02T00:00:00Z",
            }
        )
        == "2026-01-01"
    )
    assert account._line_item_label({"name": "minutes", "total": "12.500"}) == "minutes: 12.5"
    assert account._line_item_label({"product": "streaming"}) == "streaming"
    assert account._line_item_label({"quantity": 3}) == "3"
    assert account._line_item_label({}) == ""
    assert account._line_items_summary({"line_items": "bad"}) == ""


def test_usage_human_renders_breakdown(monkeypatch):
    _auth()
    _human(monkeypatch)
    payload = {
        "usage_items": [
            {
                "start_timestamp": "2026-01-01T00:00:00Z",
                "end_timestamp": "2026-01-02T00:00:00Z",
                "total": 10,
                "line_items": [{"name": "minutes", "total": 10}],
            }
        ]
    }
    with patch("aai_cli.commands.account.ams.get_usage", return_value=payload):
        result = runner.invoke(app, ["usage"])
    assert result.exit_code == 0
    assert "breakdown" in result.output
    assert "minutes: 10" in result.output


def test_usage_human_summarizes_empty_range(monkeypatch):
    _auth()
    _human(monkeypatch)
    with patch("aai_cli.commands.account.ams.get_usage", return_value={"usage_items": []}):
        result = runner.invoke(app, ["usage"])
    assert result.exit_code == 0
    assert "No usage windows returned" in result.output


def test_usage_human_hides_zero_windows_by_default(monkeypatch):
    _auth()
    _human(monkeypatch)
    payload = {
        "usage_items": [
            {
                "start_timestamp": "2026-01-01T00:00:00Z",
                "end_timestamp": "2026-01-02T00:00:00Z",
                "total": 0.0,
                "line_items": [],
            },
            {
                "start_timestamp": "2026-01-02T00:00:00Z",
                "end_timestamp": "2026-01-03T00:00:00Z",
                "total": 12.5,
                "line_items": [],
            },
        ]
    }
    with patch("aai_cli.commands.account.ams.get_usage", return_value=payload):
        result = runner.invoke(app, ["usage"])
    assert result.exit_code == 0
    assert "Usage total: 12.5" in result.output
    assert "2026-01-01" not in result.output
    assert "2026-01-02" in result.output
    assert "Hidden: 1 zero-usage window" in result.output


def test_usage_human_can_include_zero_windows(monkeypatch):
    _auth()
    _human(monkeypatch)
    payload = {
        "usage_items": [
            {
                "start_timestamp": "2026-01-01T00:00:00Z",
                "end_timestamp": "2026-01-02T00:00:00Z",
                "total": 0.0,
                "line_items": [],
            }
        ]
    }
    with patch("aai_cli.commands.account.ams.get_usage", return_value=payload):
        result = runner.invoke(app, ["usage", "--all"])
    assert result.exit_code == 0
    assert "2026-01-01" in result.output
    assert "No usage in this range" not in result.output


def test_usage_human_summarizes_all_zero_range(monkeypatch):
    _auth()
    _human(monkeypatch)
    payload = {
        "usage_items": [
            {
                "start_timestamp": "2026-01-01T00:00:00Z",
                "end_timestamp": "2026-01-02T00:00:00Z",
                "total": 0.0,
                "line_items": [],
            }
        ]
    }
    with patch("aai_cli.commands.account.ams.get_usage", return_value=payload):
        result = runner.invoke(app, ["usage"])
    assert result.exit_code == 0
    assert "Usage total: 0" in result.output
    assert "No usage in this range" in result.output
    assert "2026-01-01" not in result.output


def test_usage_passes_explicit_dates():
    _auth()
    with patch(
        "aai_cli.commands.account.ams.get_usage", return_value={"usage_items": []}
    ) as get_usage:
        result = runner.invoke(app, ["usage", "--start", "2026-01-01", "--end", "2026-02-01"])
    assert result.exit_code == 0
    # Dates are normalized to tz-aware UTC timestamps before hitting AMS.
    get_usage.assert_called_once_with(
        "jwt", "2026-01-01T00:00:00+00:00", "2026-02-01T00:00:00+00:00", None
    )


def test_usage_rejects_invalid_date():
    _auth()
    with patch("aai_cli.commands.account.ams.get_usage") as get_usage:
        result = runner.invoke(app, ["usage", "--start", "not-a-date"])
    assert result.exit_code == 2
    assert "Invalid date" in result.output
    get_usage.assert_not_called()


def test_limits_renders_services(monkeypatch):
    _auth()
    _human(monkeypatch)
    with patch(
        "aai_cli.commands.account.ams.get_rate_limits",
        return_value={"rate_limits": ["bad", {"service": "transcript", "magnitude": 200}]},
    ):
        result = runner.invoke(app, ["limits"])
    assert result.exit_code == 0
    assert "transcript" in result.output and "200" in result.output
