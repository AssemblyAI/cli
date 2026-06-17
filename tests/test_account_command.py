import json

import pytest
from typer.testing import CliRunner

from aai_cli.auth.flow import LoginResult
from aai_cli.commands import account
from aai_cli.core import config
from aai_cli.main import app

runner = CliRunner()


def _auth():
    config.set_session("default", session_jwt="jwt", session_token="tok", account_id=42)


def _login_result(*, json_mode=False):
    return LoginResult(
        api_key="sk_from_oauth", session_jwt="jwt", session_token="tok", account_id=42
    )


def test_balance_formats_dollars(mocker):
    _auth()
    mocker.patch(
        "aai_cli.commands.account.ams.get_balance",
        autospec=True,
        return_value={"account_id": 42, "balance_in_cents": 2575},
    )
    result = runner.invoke(app, ["balance"])
    assert result.exit_code == 0
    assert "$25.75" in result.output


def test_balance_without_session_runs_login(monkeypatch, mocker):
    monkeypatch.setattr("aai_cli.app.context._interactive_session", lambda: True)
    monkeypatch.setattr("aai_cli.auth.run_login_flow", _login_result)
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
        captured["start"], captured["end"], captured["window"] = start, end, window
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
    # --window defaults to None when omitted, so AMS aggregates over the whole range.
    assert captured["window"] is None
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


def test_usage_renders_table_human(mocker):
    _auth()
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
    # price (cents) is summed per window and shown as dollars, mirroring `assembly balance`.
    assert "2026-05-01" in result.output and "$12.50" in result.output


def _window(payload):
    return account._Window.model_validate(payload)


def _line_item(payload):
    return account._LineItem.model_validate(payload)


def test_usage_models_tolerate_junk_shapes():
    # A non-list `usage_items` and non-object windows degrade to nothing, not a crash.
    assert account._Usage.model_validate({"usage_items": "bad"}).usage_items == []
    kept = account._Usage.model_validate({"usage_items": [{"total": 1}, "bad"]}).usage_items
    assert kept == [account._Window()]  # the object survives; the junk item is dropped
    assert _window({"line_items": "bad"}).breakdown == ""
    # A line item without a price counts as zero spend (pins the 0.0 field default).
    assert _window({"line_items": [{"name": "x"}]}).total_cents == 0.0
    assert _line_item({"price": "junk"}).price == 0.0


def test_usage_models_format_windows_and_line_items():
    # Window total is the sum of line-item `price` (cents); the dead top-level
    # `total` field the AMS endpoint returns is ignored.
    assert (
        _window({"total": 0.0, "line_items": [{"price": 1250.0}, {"price": 0.5}]}).total_cents
        == 1250.5
    )
    assert _window({"total": 99.0, "line_items": []}).total_cents == 0.0
    assert _window({"start_timestamp": "bad"}).label == "bad"
    assert (
        _window(
            {
                "start_timestamp": "2026-01-01T00:00:00Z",
                "end_timestamp": "2026-01-03T00:00:00Z",
            }
        ).label
        == "2026-01-01 to 2026-01-03"
    )
    # Exactly one parseable bound falls back to the single start-day label (pins the
    # `start is None or end is None` guard; an `and` would dereference the None end).
    assert _window({"start_timestamp": "2026-01-01T00:00:00Z"}).label == "2026-01-01"
    # A one-day window (end == start + 1 day) collapses to a single day, not a range
    # (pins the `start.date() + timedelta(days=1)`).
    assert (
        _window(
            {
                "start_timestamp": "2026-01-01T00:00:00Z",
                "end_timestamp": "2026-01-02T00:00:00Z",
            }
        ).label
        == "2026-01-01"
    )
    # Every recognized label key resolves (pins each entry in the lookup tuple).
    for key in ("name", "product", "service", "feature", "model", "type", "description"):
        assert _line_item({key: "X"}).label == "X"
    assert _line_item({"name": "minutes", "total": "12.500"}).label == "minutes"
    assert _line_item({"product": "streaming"}).label == "streaming"
    assert _line_item({"quantity": 3}).label == ""
    assert _line_item({}).label == ""
    # Breakdown aggregates by product and shows dollars (from `price` cents), biggest
    # first, so the line items sum to the window total and reconcile with it.
    assert (
        _window(
            {
                "line_items": [
                    {"name": "minutes", "price": 1000.0},
                    {"name": "streaming", "price": 2500.0},
                    {"name": "minutes", "price": 250.0},
                ]
            }
        ).breakdown
        == "streaming: $25.00, minutes: $12.50"
    )
    # Equal-dollar products break the tie by name (pins the nc[0] secondary sort key).
    assert (
        _window(
            {"line_items": [{"name": "zeta", "price": 500.0}, {"name": "alpha", "price": 500.0}]}
        ).breakdown
        == "alpha: $5.00, zeta: $5.00"
    )
    # A line item with no recognizable product label is grouped under "other".
    assert _window({"line_items": [{"price": 500.0}]}).breakdown == "other: $5.00"
    # Zero-dollar products are dropped (they only add noise and still reconcile to 0).
    assert _window({"line_items": [{"name": "free", "price": 0.0}]}).breakdown == ""


def test_usage_human_renders_breakdown(mocker):
    _auth()
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


def test_usage_human_summarizes_empty_range(mocker):
    _auth()
    mocker.patch(
        "aai_cli.commands.account.ams.get_usage", autospec=True, return_value={"usage_items": []}
    )
    result = runner.invoke(app, ["usage"])
    assert result.exit_code == 0
    assert "No usage windows returned" in result.output


def test_usage_human_hides_zero_windows_by_default(mocker):
    _auth()
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
    # The hidden-note names the flag that reveals them, matching --include-logins on audit.
    assert "Use --include-zero to show them." in result.output


def test_usage_human_can_include_zero_windows(mocker):
    _auth()
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
    # The primary flag name is --include-zero (matching --include-logins on `assembly audit`).
    result = runner.invoke(app, ["usage", "--include-zero"])
    assert result.exit_code == 0
    assert "2026-01-01" in result.output
    assert "No usage in this range" not in result.output


def test_usage_all_is_a_back_compat_alias_for_include_zero(mocker):
    _auth()
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


def test_usage_human_summarizes_all_zero_range(mocker):
    _auth()
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


def test_usage_invalid_date_fails_before_session_resolution(monkeypatch, mocker):
    # Not logged in + a bad --start/--end: date validation must run before
    # resolve_session, so the user gets a fast exit-2 usage error, not a login flow.
    def _no_login(**_kwargs):
        raise AssertionError("login flow must not start for an invalid date")

    monkeypatch.setattr("aai_cli.app.context._interactive_session", lambda: True)
    monkeypatch.setattr("aai_cli.auth.run_login_flow", _no_login)
    get_usage = mocker.patch("aai_cli.commands.account.ams.get_usage", autospec=True)
    result = runner.invoke(app, ["usage", "--end", "not-a-date"])
    assert result.exit_code == 2
    assert "Invalid date 'not-a-date'" in result.output
    get_usage.assert_not_called()


def test_limits_renders_services(mocker):
    _auth()
    mocker.patch(
        "aai_cli.commands.account.ams.get_rate_limits",
        autospec=True,
        return_value={"rate_limits": ["bad", {"service": "transcript", "magnitude": 200}]},
    )
    result = runner.invoke(app, ["limits"])
    assert result.exit_code == 0
    assert "transcript" in result.output and "200" in result.output


def test_limits_human_summarizes_empty(mocker):
    _auth()
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


def test_format_usage_number_fractional_trims_trailing_zeros():
    # Non-integers keep up to six decimals with trailing zeros (and a bare dot) trimmed.
    assert account._format_usage_number(1234.5) == "1,234.5"
    assert account._format_usage_number(0.000001) == "0.000001"
    assert account._format_usage_number(2.5000004) == "2.5"


def test_usage_rejects_end_before_start(monkeypatch, mocker):
    # A reversed range is a fast exit-2 usage error before session resolution or
    # any AMS call — even when not logged in.
    monkeypatch.setattr("aai_cli.app.context._interactive_session", lambda: True)
    monkeypatch.setattr(
        "aai_cli.auth.run_login_flow",
        lambda **_: (_ for _ in ()).throw(AssertionError("login must not start")),
    )
    get_usage = mocker.patch("aai_cli.commands.account.ams.get_usage", autospec=True)
    result = runner.invoke(app, ["usage", "--start", "2026-06-01", "--end", "2026-01-01"])
    assert result.exit_code == 2
    assert "is before" in result.output
    get_usage.assert_not_called()


def test_usage_allows_equal_start_and_end(mocker):
    # A single-day range (end == start) is valid: pins the strict `<` in the check.
    _auth()
    get_usage = mocker.patch(
        "aai_cli.commands.account.ams.get_usage", autospec=True, return_value={"usage_items": []}
    )
    result = runner.invoke(app, ["usage", "--start", "2026-01-01", "--end", "2026-01-01", "--json"])
    assert result.exit_code == 0
    get_usage.assert_called_once()


def test_usage_rejects_unknown_window(monkeypatch, mocker):
    # --window is a closed choice set (choices.UsageWindow), so Typer rejects an
    # unknown value at parse time — before session resolution or any AMS call.
    monkeypatch.setattr("aai_cli.app.context._interactive_session", lambda: True)
    monkeypatch.setattr(
        "aai_cli.auth.run_login_flow",
        lambda **_: (_ for _ in ()).throw(AssertionError("login must not start")),
    )
    get_usage = mocker.patch("aai_cli.commands.account.ams.get_usage", autospec=True)
    result = runner.invoke(app, ["usage", "--window", "fortnight"])
    assert result.exit_code == 2
    assert "'fortnight' is not one of" in result.output
    assert "'day'" in result.output  # the valid choices are listed for the user
    get_usage.assert_not_called()


@pytest.mark.parametrize("window", ["day", "week", "month"])
def test_usage_accepts_each_known_window(mocker, window):
    _auth()
    get_usage = mocker.patch(
        "aai_cli.commands.account.ams.get_usage", autospec=True, return_value={"usage_items": []}
    )
    result = runner.invoke(app, ["usage", "--window", window, "--json"])
    assert result.exit_code == 0
    assert get_usage.call_args[0][3] == window  # passed through to AMS unchanged


def test_balance_projects_field(mocker):
    _auth()
    mocker.patch(
        "aai_cli.commands.account.ams.get_balance",
        autospec=True,
        return_value={"account_id": 42, "balance_in_cents": 2575},
    )
    result = runner.invoke(app, ["balance", "-o", "balance_in_cents"])
    assert result.exit_code == 0
    # The bare scalar, not the "$25.75" human line nor a JSON object.
    assert result.output == "2575\n"


def test_limits_projects_nested_field(mocker):
    _auth()
    mocker.patch(
        "aai_cli.commands.account.ams.get_rate_limits",
        autospec=True,
        return_value={"account_id": 42, "rate_limits": [{"service": "transcript"}]},
    )
    result = runner.invoke(app, ["limits", "-o", "account_id"])
    assert result.exit_code == 0
    assert result.output == "42\n"


def test_usage_projects_field(mocker):
    _auth()
    payload = {"usage_items": [{"line_items": [{"price": 10.0}]}], "currency": "usd"}
    mocker.patch("aai_cli.commands.account.ams.get_usage", autospec=True, return_value=payload)
    result = runner.invoke(app, ["usage", "-o", "currency"])
    assert result.exit_code == 0
    assert result.output == "usd\n"
