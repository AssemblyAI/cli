"""Tests for `assembly telemetry status/enable/disable` and the run_command integration."""

import json
import re

from typer.testing import CliRunner

from aai_cli.core import config, telemetry
from aai_cli.main import app

runner = CliRunner()


def _capture_events(monkeypatch, *, token="pub_test"):
    monkeypatch.setenv(telemetry.ENV_CLIENT_TOKEN, token)
    # Pre-mint the device id: the one-time first-run disclosure (covered in
    # test_telemetry.py) would otherwise interleave with the output assertions here.
    config.get_device_id()
    captured = []
    monkeypatch.setattr(telemetry, "dispatch", captured.append)
    return captured


def test_status_json_when_inert():
    result = runner.invoke(app, ["telemetry", "status", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "enabled": False,
        "consent": "granted",
        "source": "default",
        "token_configured": False,
    }


def test_status_json_when_enabled(monkeypatch):
    _capture_events(monkeypatch)
    result = runner.invoke(app, ["telemetry", "status", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "enabled": True,
        "consent": "granted",
        "source": "default",
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
        "source": "config",
        "token_configured": True,
    }


def test_status_json_source_for_env_kill_switch(monkeypatch):
    # The docstring promises status says *why*: an env kill-switch silently beating
    # a persisted `telemetry enable` must be visible as the source.
    _capture_events(monkeypatch)
    config.set_telemetry_enabled(enabled=True)
    monkeypatch.setenv("AAI_TELEMETRY_DISABLED", "1")
    result = runner.invoke(app, ["telemetry", "status", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "enabled": False,
        "consent": "denied",
        "source": "env:AAI_TELEMETRY_DISABLED",
        "token_configured": True,
    }


def test_status_human_disabled():
    result = runner.invoke(app, ["telemetry", "status"])
    assert result.exit_code == 0
    assert "Telemetry is disabled." in result.output
    assert "Consent: granted (source: default). Intake token configured: no." in result.output
    # When already disabled the actionable direction is re-enabling, not opting out.
    assert "Re-enable with 'assembly telemetry enable'." in result.output
    assert "Opt out any time" not in result.output


def test_status_human_enabled(monkeypatch):
    _capture_events(monkeypatch)
    result = runner.invoke(app, ["telemetry", "status"])
    assert result.exit_code == 0
    assert "Telemetry is enabled." in result.output
    assert "Consent: granted (source: default). Intake token configured: yes." in result.output
    assert "Opt out any time: 'assembly telemetry disable'" in result.output
    assert "Re-enable with" not in result.output


def test_status_human_says_why_when_env_overrides_persisted_enable(monkeypatch):
    _capture_events(monkeypatch)
    config.set_telemetry_enabled(enabled=True)
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    result = runner.invoke(app, ["telemetry", "status"])
    assert result.exit_code == 0
    assert "Telemetry is disabled." in result.output
    assert "(source: env:DO_NOT_TRACK)" in result.output
    assert "Re-enable with 'assembly telemetry enable'." in result.output


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


def test_enable_and_disable_human():
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


def test_bare_telemetry_shows_help_not_missing_command():
    result = runner.invoke(app, ["telemetry"])
    assert result.exit_code == 2
    # CI forces color on (Rich under GITHUB_ACTIONS), interleaving style codes
    # mid-message, so assert on the color-free render (see test_help_rendering.py).
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "Missing command" not in plain
    assert "Usage: assembly telemetry" in plain
    assert "status" in plain
    assert "disable" in plain


# --- run_command integration -------------------------------------------------


def test_command_run_is_tracked_with_full_command_path(monkeypatch):
    captured = _capture_events(monkeypatch)
    result = runner.invoke(app, ["telemetry", "status", "--json"])
    assert result.exit_code == 0
    (event,) = captured
    assert event["command"] == "assembly telemetry status"
    assert event["outcome"] == "success"
    assert event["exit_code"] == 0


def test_failed_command_is_tracked_with_error_type(monkeypatch):
    # No stored session and a non-interactive runner: `assembly balance` fails with
    # NotAuthenticated before any network call, and telemetry records that class.
    captured = _capture_events(monkeypatch)
    result = runner.invoke(app, ["balance", "--json"])
    assert result.exit_code == 4
    (event,) = captured
    assert event["command"] == "assembly balance"
    assert event["outcome"] == "not_authenticated"
    assert event["exit_code"] == 4


def test_inert_telemetry_tracks_nothing(monkeypatch):
    captured = []
    monkeypatch.setattr(telemetry, "dispatch", captured.append)
    result = runner.invoke(app, ["telemetry", "status", "--json"])
    assert result.exit_code == 0
    assert captured == []
