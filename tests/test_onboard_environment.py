"""Tests for the onboarding wizard's environment section summary.

Split out of test_onboard_sections.py to keep modules under the 500-line gate.
"""

from __future__ import annotations

import pytest

from aai_cli.context import AppState
from aai_cli.onboard import sections
from aai_cli.onboard.prompter import NonInteractivePrompter
from aai_cli.onboard.sections import SectionResult, WizardContext
from tests.test_onboard_sections import _capture_console, _ScriptedPrompter


@pytest.fixture
def ctx() -> WizardContext:
    return WizardContext(state=AppState(), profile="default", json_mode=False)


def _patch_checks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    python: str = "ok",
    ffmpeg: str = "ok",
    audio: str = "ok",
) -> None:
    """Pin the three doctor checks the wizard runs to the given statuses."""

    def _mk(name: str, status: str):
        fix = None if status == "ok" else f"fix the {name}"
        check: dict[str, object] = {
            "name": name,
            "status": status,
            "affects": [],
            "detail": f"{name} detail",
            "fix": fix,
        }
        return lambda: check

    monkeypatch.setattr("aai_cli.doctor_checks.check_python", _mk("python", python))
    monkeypatch.setattr("aai_cli.doctor_checks.check_ffmpeg", _mk("ffmpeg", ffmpeg))
    monkeypatch.setattr("aai_cli.doctor_checks.check_audio", _mk("audio", audio))


def test_environment_all_ok_says_everything_looks_good(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_checks(monkeypatch)
    printed = _capture_console(monkeypatch)
    assert sections.environment(NonInteractivePrompter(), ctx) is SectionResult.DONE
    flat = "\n".join(printed)
    assert "Everything looks good." in flat
    assert "Ready —" not in flat
    assert "found — see fixes above" not in flat


def test_environment_warnings_only_uses_soft_summary(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A warning must not sit under a green "Everything looks good." — the summary is
    # computed from the actual check statuses.
    _patch_checks(monkeypatch, audio="warn")
    printed = _capture_console(monkeypatch)
    assert sections.environment(NonInteractivePrompter(), ctx) is SectionResult.DONE
    flat = "\n".join(printed)
    assert "Ready — 1 warning (only affects streaming/agent)." in flat
    assert "Everything looks good" not in flat
    assert "fix: fix the audio" in flat  # per-check fix hints still render


def test_environment_two_warnings_pluralize(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_checks(monkeypatch, ffmpeg="warn", audio="warn")
    printed = _capture_console(monkeypatch)
    assert sections.environment(NonInteractivePrompter(), ctx) is SectionResult.DONE
    assert "Ready — 2 warnings (only affects streaming/agent)." in "\n".join(printed)


def test_environment_failed_check_fails_section(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_checks(monkeypatch, python="fail", audio="warn")
    printed = _capture_console(monkeypatch)
    assert sections.environment(NonInteractivePrompter(), ctx) is SectionResult.FAILED
    flat = "\n".join(printed)
    assert "1 problem found — see fixes above." in flat
    assert "Everything looks good" not in flat


def test_environment_two_failures_pluralize(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_checks(monkeypatch, python="fail", ffmpeg="fail")
    printed = _capture_console(monkeypatch)
    assert sections.environment(NonInteractivePrompter(), ctx) is SectionResult.FAILED
    assert "2 problems found — see fixes above." in "\n".join(printed)


def test_environment_human_mode_notes_streaming_caveat(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_checks(monkeypatch)
    _capture_console(monkeypatch)
    prompter = _ScriptedPrompter()
    assert sections.environment(prompter, ctx) is SectionResult.DONE
    assert any("only affect live streaming" in note for note in prompter.notes)


def test_environment_json_mode_keeps_stdout_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    # Under --json the final summary owns stdout: no human render, no note — but the
    # section result is still computed from the checks.
    json_ctx = WizardContext(state=AppState(), profile="default", json_mode=True)
    _patch_checks(monkeypatch, python="fail")
    printed = _capture_console(monkeypatch)
    prompter = _ScriptedPrompter()
    assert sections.environment(prompter, json_ctx) is SectionResult.FAILED
    assert printed == []
    assert prompter.notes == []
