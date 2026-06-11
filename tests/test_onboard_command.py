from __future__ import annotations

import pytest
from typer.testing import CliRunner

from aai_cli.commands import onboard as onboard_cmd
from aai_cli.main import app
from aai_cli.onboard.prompter import InteractivePrompter, NonInteractivePrompter


def test_onboard_is_listed_in_help() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert "onboard" in result.output


def test_onboard_runs_wizard_and_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aai_cli.commands.onboard.wizard.run_onboarding", lambda p, c: 0)
    result = CliRunner().invoke(app, ["onboard"])
    assert result.exit_code == 0, result.output


def test_onboard_propagates_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aai_cli.commands.onboard.wizard.run_onboarding", lambda p, c: 4)
    result = CliRunner().invoke(app, ["onboard"])
    assert result.exit_code == 4


def test_onboard_propagates_exit_code_one(monkeypatch: pytest.MonkeyPatch) -> None:
    # Exit code 1 specifically pins the `code != 0` guard: a `!= 1` mutant would
    # swallow this and exit 0 instead.
    monkeypatch.setattr("aai_cli.commands.onboard.wizard.run_onboarding", lambda p, c: 1)
    result = CliRunner().invoke(app, ["onboard"])
    assert result.exit_code == 1


def test_onboard_does_not_auto_login_on_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # auto_login=False: an unauthenticated wizard surfaces the auth error (exit 4)
    # rather than kicking off a browser login. A True mutant would instead try to
    # log in and never exit 4 here.
    from aai_cli.errors import NotAuthenticated

    def _raise(p: object, c: object) -> int:
        raise NotAuthenticated("nope")

    monkeypatch.setattr("aai_cli.commands.onboard.wizard.run_onboarding", _raise)
    result = CliRunner().invoke(app, ["onboard"])
    assert result.exit_code == 4
    assert "browser login" not in result.output


def test_build_prompter_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    assert isinstance(onboard_cmd.build_prompter(), InteractivePrompter)


def test_build_prompter_noninteractive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    assert isinstance(onboard_cmd.build_prompter(), NonInteractivePrompter)


def test_build_prompter_forced_noninteractive_on_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    # `--non-interactive` wins even with both ends a real TTY: a mutant that ignored
    # the flag (or `or`-ed it with the TTY check) would hand back an InteractivePrompter.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    assert isinstance(onboard_cmd.build_prompter(non_interactive=True), NonInteractivePrompter)


def _spy_forced(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Capture the `non_interactive` value the command hands `build_prompter`.

    Spying on the argument (rather than the prompter type) is what pins the
    `forced = non_interactive or is_agentic()` expression: under CliRunner stdout is
    never a TTY, so the resolved prompter would read NonInteractive either way.
    """
    captured: dict[str, object] = {}

    def _fake_build(*, non_interactive: bool) -> NonInteractivePrompter:
        captured["forced"] = non_interactive
        return NonInteractivePrompter()

    monkeypatch.setattr("aai_cli.commands.onboard.build_prompter", _fake_build)
    monkeypatch.setattr("aai_cli.commands.onboard.wizard.run_onboarding", lambda p, c: 0)
    return captured


def test_onboard_non_interactive_flag_forces_noninteractive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `--non-interactive` forces non-interactive mode even when no agent is detected.
    monkeypatch.setattr("aai_cli.output.is_agentic", lambda: False)
    captured = _spy_forced(monkeypatch)
    result = CliRunner().invoke(app, ["onboard", "--non-interactive"])
    assert result.exit_code == 0, result.output
    assert captured["forced"] is True


def test_onboard_defaults_to_noninteractive_when_agent_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No flag, but an agent is detected: the wizard still defaults to non-interactive.
    # A mutant dropping the `is_agentic()` term would leave `forced` False here.
    monkeypatch.setattr("aai_cli.output.is_agentic", lambda: True)
    captured = _spy_forced(monkeypatch)
    result = CliRunner().invoke(app, ["onboard"])
    assert result.exit_code == 0, result.output
    assert captured["forced"] is True


def test_onboard_stays_interactive_without_flag_or_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No flag, no agent: `forced` is False, so build_prompter is free to drive real
    # prompts. An `and` mutant on the `or` would also land here, but the two cases
    # above (each True via a different operand) pin the operator.
    monkeypatch.setattr("aai_cli.output.is_agentic", lambda: False)
    captured = _spy_forced(monkeypatch)
    result = CliRunner().invoke(app, ["onboard"])
    assert result.exit_code == 0, result.output
    assert captured["forced"] is False


def test_onboard_sorts_first_in_quick_start() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.output.index("onboard") < result.output.index("init")


def test_interactive_session_requires_both_ends_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    from aai_cli import main as main_mod

    # Both TTY -> interactive.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    assert main_mod._interactive_session() is True
    # Only one end a TTY -> NOT interactive. An `or` mutant would call this interactive.
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    assert main_mod._interactive_session() is False
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    assert main_mod._interactive_session() is False


def test_bare_aai_with_key_shows_help_no_offer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk_test")
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 0, result.output
    assert "Usage" in result.output or "Commands" in result.output


def test_bare_aai_prints_welcome_header(monkeypatch: pytest.MonkeyPatch) -> None:
    # The welcome screen leads with the version + tagline header line.
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk_test")
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 0, result.output
    assert "AssemblyAI from your terminal" in result.output


def test_bare_aai_quiet_suppresses_banner(monkeypatch: pytest.MonkeyPatch) -> None:
    # `--quiet` drops the decorative header but still prints help. A mutant that
    # ignores `quiet` (always banners) would leave the tagline in the output.
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk_test")
    result = CliRunner().invoke(app, ["--quiet"])
    assert result.exit_code == 0, result.output
    assert "AssemblyAI from your terminal" not in result.output
    assert "Usage" in result.output or "Commands" in result.output


def test_bare_aai_offers_wizard_when_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from aai_cli import main as main_mod
    from aai_cli.onboard.sections import WizardContext

    monkeypatch.setattr(main_mod, "_interactive_session", lambda: True)
    monkeypatch.setattr("aai_cli.main.typer.confirm", lambda *a, **k: True)
    captured: dict[str, object] = {}

    def _fake_run(prompter: object, ctx: WizardContext) -> int:
        captured["called"] = True
        captured["json_mode"] = ctx.json_mode
        return 0

    monkeypatch.setattr("aai_cli.main.wizard.run_onboarding", _fake_run)
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 0, result.output
    assert captured["called"] is True
    # The wizard is built in human (non-JSON) mode; a `json_mode=True` mutant flips this.
    assert captured["json_mode"] is False


def test_bare_aai_empty_confirm_defaults_to_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    # The offer prompt defaults to Yes: an empty <Enter> answer runs the wizard.
    # A `default=False` mutant would instead decline and print help.
    from aai_cli import main as main_mod

    monkeypatch.setattr(main_mod, "_interactive_session", lambda: True)
    ran = {"called": False}

    def _fake_run(prompter: object, ctx: object) -> int:
        ran["called"] = True
        return 0

    monkeypatch.setattr("aai_cli.main.wizard.run_onboarding", _fake_run)
    result = CliRunner().invoke(app, [], input="\n")
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
        "aai_cli.main.typer.confirm", lambda *a, **k: called.__setitem__("confirm", True)
    )
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 0, result.output
    assert called["confirm"] is False
    assert "Usage" in result.output or "Commands" in result.output


def test_bare_aai_declined_offer_shows_help(monkeypatch: pytest.MonkeyPatch) -> None:
    from aai_cli import main as main_mod

    monkeypatch.setattr(main_mod, "_interactive_session", lambda: True)
    monkeypatch.setattr("aai_cli.main.typer.confirm", lambda *a, **k: False)
    called = {"v": False}

    def _fake_run(prompter: object, ctx: object) -> int:
        called["v"] = True
        return 0

    monkeypatch.setattr("aai_cli.main.wizard.run_onboarding", _fake_run)
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 0, result.output
    assert called["v"] is False
    assert "Usage" in result.output or "Commands" in result.output
