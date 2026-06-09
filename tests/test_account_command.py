import json

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


def test_balance_formats_dollars(monkeypatch, mocker):
    _auth()
    _human(monkeypatch)
    mocker.patch(
        "aai_cli.commands.account.ams.get_balance",
        autospec=True,
        return_value={"account_id": 42, "balance_in_cents": 2575},
    )
    result = runner.invoke(app, ["balance"])
    assert result.exit_code == 0
    assert "$25.75" in result.output


def test_balance_without_session_runs_login(monkeypatch, mocker):
    monkeypatch.setattr("aai_cli.context.run_login_flow", _login_result)
    get_balance = mocker.patch(
        "aai_cli.commands.account.ams.get_balance",
        autospec=True,
        return_value={"account_id": 42, "balance_in_cents": 2575},
    )
    result = runner.invoke(app, ["balance", "--json"])
    assert result.exit_code == 4
    assert config.get_session("default") == {"jwt": "jwt", "token": "tok"}
    get_balance.assert_not_called()
    assert "Run the same command again" in result.output


def test_usage_defaults_date_range_and_renders(mocker):
    _auth()
    captured = {}

    def fake_usage(jwt, start, end, window):
        captured["start"], captured["end"] = start, end
        return {
            "usage_items": [
                {
                    "start_timestamp": "2026-05-01",
                    "end_timestamp": "2026-05-02",
                    "total": 0.0,
                    "line_items": [{"name": "Streaming", "price": 1250.0, "quantity": 12.5}],
                }
            ]
        }

    mocker.patch("aai_cli.commands.account.ams.get_usage", autospec=True, side_effect=fake_usage)
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
    # --json is a raw passthrough of the AMS response (the dead top-level `total`
    # included), so downstream tooling sees exactly what the endpoint returned.
    data = json.loads(result.output)
    assert data["usage_items"][0]["line_items"][0]["price"] == 1250.0


def test_usage_renders_table_human(monkeypatch, mocker):
    _auth()
    _human(monkeypatch)
    payload = {
        "usage_items": [
            {
                "start_timestamp": "2026-05-01",
                "end_timestamp": "2026-05-02",
                "total": 0.0,
                "line_items": [{"name": "Streaming", "price": 1250.0, "quantity": 12.5}],
            }
        ]
    }
    mocker.patch("aai_cli.commands.account.ams.get_usage", autospec=True, return_value=payload)
    result = runner.invoke(app, ["usage"])
    assert result.exit_code == 0
    # price (cents) is summed per window and shown as dollars, mirroring `aai balance`.
    assert "2026-05-01" in result.output and "$12.50" in result.output


def test_usage_helpers_format_windows_and_line_items():
    assert account._usage_items({"usage_items": "bad"}) == []
    assert account._usage_items({"usage_items": [{"total": 1}, "bad"]}) == [{"total": 1}]
    # Window total is the sum of line-item `price` (cents); the dead top-level
    # `total` field the AMS endpoint returns is ignored.
    assert (
        account._window_total_cents(
            {"total": 0.0, "line_items": [{"price": 1250.0}, {"price": 0.5}]}
        )
        == 1250.5
    )
    assert account._window_total_cents({"total": 99.0, "line_items": []}) == 0.0
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
    # Every recognized label key resolves (pins each entry in the lookup tuple).
    for key in ("name", "product", "service", "feature", "model", "type", "description"):
        assert account._line_item_name({key: "X"}) == "X"
    assert account._line_item_name({"name": "minutes", "total": "12.500"}) == "minutes"
    assert account._line_item_name({"product": "streaming"}) == "streaming"
    assert account._line_item_name({"quantity": 3}) == ""
    assert account._line_item_name({}) == ""
    # Breakdown aggregates by product and shows dollars (from `price` cents), biggest
    # first, so the line items sum to the window total and reconcile with it.
    assert (
        account._line_items_summary(
            {
                "line_items": [
                    {"name": "minutes", "price": 1000.0},
                    {"name": "streaming", "price": 2500.0},
                    {"name": "minutes", "price": 250.0},
                ]
            }
        )
        == "streaming: $25.00, minutes: $12.50"
    )
    # Equal-dollar products break the tie by name (pins the nc[0] secondary sort key).
    assert (
        account._line_items_summary(
            {"line_items": [{"name": "zeta", "price": 500.0}, {"name": "alpha", "price": 500.0}]}
        )
        == "alpha: $5.00, zeta: $5.00"
    )
    # A line item with no recognizable product label is grouped under "other".
    assert account._line_items_summary({"line_items": [{"price": 500.0}]}) == "other: $5.00"
    # Zero-dollar products are dropped (they only add noise and still reconcile to 0).
    assert account._line_items_summary({"line_items": [{"name": "free", "price": 0.0}]}) == ""
    assert account._line_items_summary({"line_items": "bad"}) == ""


def test_usage_human_renders_breakdown(monkeypatch, mocker):
    _auth()
    _human(monkeypatch)
    payload = {
        "usage_items": [
            {
                "start_timestamp": "2026-01-01T00:00:00Z",
                "end_timestamp": "2026-01-02T00:00:00Z",
                "total": 0.0,
                "line_items": [{"name": "minutes", "price": 1000.0, "quantity": 10}],
            }
        ]
    }
    mocker.patch("aai_cli.commands.account.ams.get_usage", autospec=True, return_value=payload)
    result = runner.invoke(app, ["usage"])
    assert result.exit_code == 0
    assert "breakdown" in result.output
    # The breakdown shows each product's spend in dollars (1000 cents = $10.00), the
    # same unit as the `total` column, so the two reconcile.
    assert "minutes: $10.00" in result.output


def test_usage_human_summarizes_empty_range(monkeypatch, mocker):
    _auth()
    _human(monkeypatch)
    mocker.patch(
        "aai_cli.commands.account.ams.get_usage", autospec=True, return_value={"usage_items": []}
    )
    result = runner.invoke(app, ["usage"])
    assert result.exit_code == 0
    assert "No usage windows returned" in result.output


def test_usage_human_hides_zero_windows_by_default(monkeypatch, mocker):
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
                "total": 0.0,
                "line_items": [{"name": "Streaming", "price": 1250.0, "quantity": 12.5}],
            },
        ]
    }
    mocker.patch("aai_cli.commands.account.ams.get_usage", autospec=True, return_value=payload)
    result = runner.invoke(app, ["usage"])
    assert result.exit_code == 0
    assert "Usage total: $12.50" in result.output
    assert "2026-01-01" not in result.output
    assert "2026-01-02" in result.output
    assert "Hidden: 1 zero-usage window" in result.output


def test_usage_human_can_include_zero_windows(monkeypatch, mocker):
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
    mocker.patch("aai_cli.commands.account.ams.get_usage", autospec=True, return_value=payload)
    result = runner.invoke(app, ["usage", "--all"])
    assert result.exit_code == 0
    assert "2026-01-01" in result.output
    assert "No usage in this range" not in result.output


def test_usage_human_summarizes_all_zero_range(monkeypatch, mocker):
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
    mocker.patch("aai_cli.commands.account.ams.get_usage", autospec=True, return_value=payload)
    result = runner.invoke(app, ["usage"])
    assert result.exit_code == 0
    assert "Usage total: $0.00" in result.output
    assert "No usage in this range" in result.output
    assert "2026-01-01" not in result.output


def test_usage_passes_explicit_dates(mocker):
    _auth()
    get_usage = mocker.patch(
        "aai_cli.commands.account.ams.get_usage", autospec=True, return_value={"usage_items": []}
    )
    result = runner.invoke(app, ["usage", "--start", "2026-01-01", "--end", "2026-02-01"])
    assert result.exit_code == 0
    # Dates are normalized to tz-aware UTC timestamps before hitting AMS.
    get_usage.assert_called_once_with(
        "jwt", "2026-01-01T00:00:00+00:00", "2026-02-01T00:00:00+00:00", None
    )


def test_usage_rejects_invalid_date(mocker):
    _auth()
    get_usage = mocker.patch("aai_cli.commands.account.ams.get_usage", autospec=True)
    result = runner.invoke(app, ["usage", "--start", "not-a-date"])
    assert result.exit_code == 2
    assert "Invalid date" in result.output
    get_usage.assert_not_called()


def test_limits_renders_services(monkeypatch, mocker):
    _auth()
    _human(monkeypatch)
    mocker.patch(
        "aai_cli.commands.account.ams.get_rate_limits",
        autospec=True,
        return_value={"rate_limits": ["bad", {"service": "transcript", "magnitude": 200}]},
    )
    result = runner.invoke(app, ["limits"])
    assert result.exit_code == 0
    assert "transcript" in result.output and "200" in result.output


def test_limits_human_summarizes_empty(monkeypatch, mocker):
    _auth()
    _human(monkeypatch)
    # The AMS endpoint returns an empty array when no custom rate limits are
    # configured; show a clear message instead of a bare header-only table.
    mocker.patch(
        "aai_cli.commands.account.ams.get_rate_limits",
        autospec=True,
        return_value={"rate_limits": []},
    )
    result = runner.invoke(app, ["limits"])
    assert result.exit_code == 0
    assert "No custom rate limits" in result.output


def test_limits_json_passthrough_when_empty(mocker):
    _auth()
    mocker.patch(
        "aai_cli.commands.account.ams.get_rate_limits",
        autospec=True,
        return_value={"rate_limits": []},
    )
    result = runner.invoke(app, ["limits", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == {"rate_limits": []}
