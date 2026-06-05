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


def test_balance_requires_session():
    result = runner.invoke(app, ["balance"])
    assert result.exit_code == 2


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
    # both bounds are ISO dates (YYYY-MM-DD), defaulted when not passed
    assert len(captured["start"]) == 10 and len(captured["end"]) == 10
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


def test_usage_passes_explicit_dates():
    _auth()
    with patch(
        "aai_cli.commands.account.ams.get_usage", return_value={"usage_items": []}
    ) as get_usage:
        result = runner.invoke(app, ["usage", "--start", "2026-01-01", "--end", "2026-02-01"])
    assert result.exit_code == 0
    get_usage.assert_called_once_with("jwt", "2026-01-01", "2026-02-01", None)


def test_limits_renders_services(monkeypatch):
    _auth()
    _human(monkeypatch)
    with patch(
        "aai_cli.commands.account.ams.get_rate_limits",
        return_value={"rate_limits": [{"service": "transcript", "magnitude": 200}]},
    ):
        result = runner.invoke(app, ["limits"])
    assert result.exit_code == 0
    assert "transcript" in result.output and "200" in result.output
