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
    assert isinstance(onboard_cmd.build_prompter(), onboard_cmd.InteractivePrompter)


def test_build_prompter_noninteractive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    assert isinstance(onboard_cmd.build_prompter(), onboard_cmd.NonInteractivePrompter)


def test_onboard_sorts_first_in_quick_start() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.output.index("onboard") < result.output.index("init")


def test_bare_aai_with_key_shows_help_no_offer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk_test")
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 0, result.output
    assert "Usage" in result.output or "Commands" in result.output


def test_bare_aai_offers_wizard_when_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from aai_cli import main as main_mod

    monkeypatch.setattr(main_mod, "_interactive_session", lambda: True)
    monkeypatch.setattr(main_mod.typer, "confirm", lambda *a, **k: True)
    ran = {"called": False}

    def _fake_run(prompter: object, ctx: object) -> int:
        ran["called"] = True
        return 0

    monkeypatch.setattr(main_mod.wizard, "run_onboarding", _fake_run)
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 0, result.output
    assert ran["called"] is True


def test_bare_aai_interactive_with_key_shows_help_no_offer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aai_cli import main as main_mod

    # Interactive session but a key is already present: _profile_has_key returns True,
    # so the wizard is never offered and help is printed instead.
    monkeypatch.setattr(main_mod, "_interactive_session", lambda: True)
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk_test")
    called = {"confirm": False}
    monkeypatch.setattr(
        main_mod.typer, "confirm", lambda *a, **k: called.__setitem__("confirm", True)
    )
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 0, result.output
    assert called["confirm"] is False
    assert "Usage" in result.output or "Commands" in result.output


def test_bare_aai_declined_offer_shows_help(monkeypatch: pytest.MonkeyPatch) -> None:
    from aai_cli import main as main_mod

    monkeypatch.setattr(main_mod, "_interactive_session", lambda: True)
    monkeypatch.setattr(main_mod.typer, "confirm", lambda *a, **k: False)
    called = {"v": False}

    def _fake_run(prompter: object, ctx: object) -> int:
        called["v"] = True
        return 0

    monkeypatch.setattr(main_mod.wizard, "run_onboarding", _fake_run)
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 0, result.output
    assert called["v"] is False
    assert "Usage" in result.output or "Commands" in result.output
