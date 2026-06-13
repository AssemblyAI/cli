from __future__ import annotations

import json

import pytest

from aai_cli.app.context import AppState
from aai_cli.onboard import sections, wizard
from aai_cli.onboard.prompter import NonInteractivePrompter, WizardCancelled
from aai_cli.onboard.sections import SectionResult, WizardContext
from aai_cli.ui import output

ALL_SECTIONS = (
    "welcome",
    "auth",
    "first_request",
    "environment",
    "build_path",
    "claude_code",
    "next_steps",
    "launch_app",
)


@pytest.fixture
def ctx() -> WizardContext:
    return WizardContext(state=AppState(), profile="default", json_mode=False)


@pytest.fixture
def json_ctx() -> WizardContext:
    return WizardContext(state=AppState(), profile="default", json_mode=True)


def _patch_sections(monkeypatch: pytest.MonkeyPatch, **overrides: SectionResult) -> None:
    """Stub every section to DONE, with per-section result overrides."""

    def _const(result: SectionResult):
        return lambda p, c: result

    for name in ALL_SECTIONS:
        monkeypatch.setattr(sections, name, _const(overrides.get(name, SectionResult.DONE)))


def _capture_stderr(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    printed: list[str] = []
    monkeypatch.setattr(
        output.error_console, "print", lambda *a, **k: printed.append(str(a[0]) if a else "")
    )
    return printed


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
    ran: list[str] = []

    def _record(name: str):
        def _section(p: object, c: object) -> SectionResult:
            ran.append(name)
            return SectionResult.DONE

        return _section

    for name in (
        "welcome",
        "auth",
        "first_request",
        "environment",
        "build_path",
        "claude_code",
        "next_steps",
        "launch_app",
    ):
        monkeypatch.setattr(sections, name, _record(name))
    assert wizard.run_onboarding(NonInteractivePrompter(), ctx) == 0
    # launch_app must run, and run last: its dev server blocks until Ctrl-C, so any
    # section ordered after it would never execute.
    assert ran == [
        "welcome",
        "auth",
        "first_request",
        "environment",
        "build_path",
        "claude_code",
        "next_steps",
        "launch_app",
    ]


def test_failed_section_exits_one_with_closing_line(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A failed non-auth section must not end in a cheery exit 0: the run exits 1 and
    # the closing line names what failed.
    _patch_sections(monkeypatch, first_request=SectionResult.FAILED)
    printed = _capture_stderr(monkeypatch)
    assert wizard.run_onboarding(NonInteractivePrompter(), ctx) == 1
    assert any("Set up with 1 issue (first transcription failed)." in line for line in printed)


def test_two_failed_sections_pluralize_and_name_both(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_sections(
        monkeypatch, first_request=SectionResult.FAILED, claude_code=SectionResult.FAILED
    )
    printed = _capture_stderr(monkeypatch)
    assert wizard.run_onboarding(NonInteractivePrompter(), ctx) == 1
    assert any(
        "Set up with 2 issues (first transcription, coding agent failed)." in line
        for line in printed
    )


def test_clean_run_prints_no_closing_failure_line(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_sections(monkeypatch)
    printed = _capture_stderr(monkeypatch)
    assert wizard.run_onboarding(NonInteractivePrompter(), ctx) == 0
    assert not any("Set up with" in line for line in printed)


def test_json_summary_on_success(
    json_ctx: WizardContext, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # --json emits one machine-readable summary on stdout; the exact section map pins
    # every label and the SectionResult value strings (auth here exercises "skipped").
    _patch_sections(monkeypatch, auth=SectionResult.SKIPPED)
    assert wizard.run_onboarding(NonInteractivePrompter(), json_ctx) == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload == {
        "ok": True,
        "exit_code": 0,
        "sections": {
            "welcome": "done",
            "sign-in": "skipped",
            "first transcription": "done",
            "environment": "done",
            "build path": "done",
            "coding agent": "done",
            "next steps": "done",
            "launch app": "done",
        },
        "failed": [],
    }


def test_json_summary_on_failure(
    json_ctx: WizardContext, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_sections(monkeypatch, first_request=SectionResult.FAILED)
    printed = _capture_stderr(monkeypatch)
    assert wizard.run_onboarding(NonInteractivePrompter(), json_ctx) == 1
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["ok"] is False
    assert payload["exit_code"] == 1
    assert payload["failed"] == ["first transcription"]
    assert payload["sections"]["first transcription"] == "failed"
    # JSON mode keeps the human closing line off stderr (the error envelope is the
    # machine-readable failure signal).
    assert not any("Set up with" in line for line in printed)


def test_json_summary_on_auth_stop(
    json_ctx: WizardContext, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The auth hard stop keeps its own exit code (4, not the generic 1) and still
    # emits the summary; the human "Sign-in didn't complete." line stays off JSON runs.
    _patch_sections(monkeypatch, auth=SectionResult.FAILED)
    printed = _capture_stderr(monkeypatch)
    assert wizard.run_onboarding(NonInteractivePrompter(), json_ctx) == 4
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["ok"] is False
    assert payload["exit_code"] == 4
    assert payload["sections"] == {"welcome": "done", "sign-in": "failed"}
    assert payload["failed"] == ["sign-in"]
    assert not any("Sign-in didn't complete" in line for line in printed)


def test_human_mode_emits_no_json_summary(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_sections(monkeypatch)
    assert wizard.run_onboarding(NonInteractivePrompter(), ctx) == 0
    assert capsys.readouterr().out.strip() == ""


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
