from __future__ import annotations

import pytest

from aai_cli import output
from aai_cli.context import AppState
from aai_cli.onboard import sections, wizard
from aai_cli.onboard.prompter import NonInteractivePrompter, WizardCancelled
from aai_cli.onboard.sections import SectionResult, WizardContext


@pytest.fixture
def ctx() -> WizardContext:
    return WizardContext(state=AppState(), profile="default", json_mode=False)


def test_auth_failure_stops_the_wizard(ctx: WizardContext, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sections, "welcome", lambda p, c: SectionResult.DONE)
    monkeypatch.setattr(sections, "auth", lambda p, c: SectionResult.FAILED)
    ran_after = False

    def _first(p: object, c: object) -> SectionResult:
        nonlocal ran_after
        ran_after = True
        return SectionResult.DONE

    monkeypatch.setattr(sections, "first_request", _first)
    printed: list[str] = []
    monkeypatch.setattr(
        output.error_console, "print", lambda *a, **k: printed.append(str(a[0]) if a else "")
    )
    code = wizard.run_onboarding(NonInteractivePrompter(), ctx)
    assert code == 4  # NotAuthenticated exit code
    assert ran_after is False
    assert any("Sign-in didn't complete" in line for line in printed)


def test_happy_path_runs_all_sections(ctx: WizardContext, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "welcome",
        "auth",
        "first_request",
        "environment",
        "build_path",
        "claude_code",
        "next_steps",
    ):
        monkeypatch.setattr(sections, name, lambda p, c: SectionResult.DONE)
    assert wizard.run_onboarding(NonInteractivePrompter(), ctx) == 0


def test_cancel_returns_130(ctx: WizardContext, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sections, "welcome", lambda p, c: SectionResult.DONE)

    def _cancel(p: object, c: object) -> SectionResult:
        raise WizardCancelled

    monkeypatch.setattr(sections, "auth", _cancel)
    assert wizard.run_onboarding(NonInteractivePrompter(), ctx) == 130


def test_cursor_is_always_restored(ctx: WizardContext, monkeypatch: pytest.MonkeyPatch) -> None:
    # The finally block must re-show the cursor (show=True), even on cancellation.
    shown: list[bool] = []
    monkeypatch.setattr(output.console, "show_cursor", lambda *, show: shown.append(show))
    monkeypatch.setattr(sections, "welcome", lambda p, c: SectionResult.DONE)

    def _cancel(p: object, c: object) -> SectionResult:
        raise WizardCancelled

    monkeypatch.setattr(sections, "auth", _cancel)
    wizard.run_onboarding(NonInteractivePrompter(), ctx)
    assert shown == [True]
