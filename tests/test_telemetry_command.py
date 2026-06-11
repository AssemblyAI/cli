"""Tests for `aai telemetry status/enable/disable` and the run_command integration."""

import json

from typer.testing import CliRunner

from aai_cli import config, telemetry
from aai_cli.main import app

runner = CliRunner()


def _human(monkeypatch):
    monkeypatch.setattr("aai_cli.output.resolve_json", lambda *, explicit: explicit)


def _capture_events(monkeypatch, *, token="pub_test"):
    monkeypatch.setenv(telemetry.ENV_CLIENT_TOKEN, token)
    captured = []
    monkeypatch.setattr(telemetry, "dispatch", captured.append)
    return captured


def test_status_json_when_inert():
    result = runner.invoke(app, ["telemetry", "status", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "enabled": False,
        "consent": "granted",
        "token_configured": False,
    }


def test_status_json_when_enabled(monkeypatch):
    _capture_events(monkeypatch)
    result = runner.invoke(app, ["telemetry", "status", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "enabled": True,
        "consent": "granted",
        "token_configured": True,
    }


def test_status_json_when_opted_out(monkeypatch):
    _capture_events(monkeypatch)
    config.set_telemetry_enabled(enabled=False)
    result = runner.invoke(app, ["telemetry", "status", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "enabled": False,
        "consent": "denied",
        "token_configured": True,
    }


def test_status_human_disabled(monkeypatch):
    _human(monkeypatch)
    result = runner.invoke(app, ["telemetry", "status"])
    assert result.exit_code == 0
    assert "Telemetry is disabled." in result.output
    assert "Consent: granted. Intake token configured: no." in result.output
    assert "aai telemetry disable" in result.output


def test_status_human_enabled(monkeypatch):
    _human(monkeypatch)
    _capture_events(monkeypatch)
    result = runner.invoke(app, ["telemetry", "status"])
    assert result.exit_code == 0
    assert "Telemetry is enabled." in result.output
    assert "Consent: granted. Intake token configured: yes." in result.output


def test_disable_persists_and_confirms_json():
    result = runner.invoke(app, ["telemetry", "disable", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == {"telemetry_enabled": False}
    assert config.get_telemetry_enabled() is False


def test_enable_persists_and_confirms_json():
    config.set_telemetry_enabled(enabled=False)
    result = runner.invoke(app, ["telemetry", "enable", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == {"telemetry_enabled": True}
    assert config.get_telemetry_enabled() is True


def test_enable_and_disable_human(monkeypatch):
    _human(monkeypatch)
    result = runner.invoke(app, ["telemetry", "disable"])
    assert result.exit_code == 0
    assert "Telemetry disabled." in result.output
    result = runner.invoke(app, ["telemetry", "enable"])
    assert result.exit_code == 0
    assert "Telemetry enabled." in result.output


def test_flush_is_hidden_plumbing():
    # `flush` exists for dispatch() to spawn, but users shouldn't be steered to it.
    result = runner.invoke(app, ["telemetry", "--help"])
    assert result.exit_code == 0
    assert "status" in result.output
    assert "flush" not in result.output


# --- run_command integration -------------------------------------------------


def test_command_run_is_tracked_with_full_command_path(monkeypatch):
    captured = _capture_events(monkeypatch)
    result = runner.invoke(app, ["telemetry", "status", "--json"])
    assert result.exit_code == 0
    (event,) = captured
    assert event["command"] == "aai telemetry status"
    assert event["outcome"] == "success"
    assert event["exit_code"] == 0


def test_failed_command_is_tracked_with_error_type(monkeypatch):
    # No stored session and a non-interactive runner: `aai balance` fails with
    # NotAuthenticated before any network call, and telemetry records that class.
    captured = _capture_events(monkeypatch)
    result = runner.invoke(app, ["balance", "--json"])
    assert result.exit_code == 4
    (event,) = captured
    assert event["command"] == "aai balance"
    assert event["outcome"] == "not_authenticated"
    assert event["exit_code"] == 4


def test_inert_telemetry_tracks_nothing(monkeypatch):
    captured = []
    monkeypatch.setattr(telemetry, "dispatch", captured.append)
    result = runner.invoke(app, ["telemetry", "status", "--json"])
    assert result.exit_code == 0
    assert captured == []
