from __future__ import annotations

import pytest
from typer.testing import CliRunner

from aai_cli import config
from aai_cli.commands import onboard as onboard_cmd
from aai_cli.main import app


def test_status_shows_progress_without_running_wizard() -> None:
    config.record_request("default")
    config.record_request("default")
    result = CliRunner().invoke(app, ["onboard", "--status"])
    assert result.exit_code == 0, result.output
    assert "2 of 100" in result.output


def test_onboard_is_listed_in_help() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert "onboard" in result.output


def test_onboard_runs_wizard_and_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(onboard_cmd.wizard, "run_onboarding", lambda p, c: 0)
    result = CliRunner().invoke(app, ["onboard"])
    assert result.exit_code == 0, result.output


def test_onboard_propagates_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(onboard_cmd.wizard, "run_onboarding", lambda p, c: 4)
    result = CliRunner().invoke(app, ["onboard"])
    assert result.exit_code == 4


def test_build_prompter_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    assert isinstance(onboard_cmd._build_prompter(), onboard_cmd.InteractivePrompter)


def test_build_prompter_noninteractive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    assert isinstance(onboard_cmd._build_prompter(), onboard_cmd.NonInteractivePrompter)
