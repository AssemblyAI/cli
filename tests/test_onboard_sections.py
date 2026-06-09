from __future__ import annotations

import contextlib
from collections.abc import Generator
from pathlib import Path

import pytest
import typer

from aai_cli import output, transcribe_exec, transcribe_render
from aai_cli.commands import init as init_cmd
from aai_cli.commands import setup as setup_cmd
from aai_cli.context import AppState
from aai_cli.onboard import sections
from aai_cli.onboard.prompter import NonInteractivePrompter
from aai_cli.onboard.sections import SectionResult, WizardContext
from aai_cli.steps import Step


class _FakeTranscript:
    id = "t_1"
    status = "completed"
    text = "hello"
    utterances = None


class _ScriptedPrompter:
    """A Prompter test-double whose answers are pinned at construction time."""

    def __init__(self, *, select: str = "skip", confirm: bool = True, text: str = "k") -> None:
        self._select = select
        self._confirm = confirm
        self._text = text
        self.confirm_defaults: list[bool] = []

    def section(self, title: str) -> None:
        pass

    def note(self, message: str) -> None:
        pass

    def confirm(self, title: str, *, default: bool = True) -> bool:
        self.confirm_defaults.append(default)
        return self._confirm

    def select(
        self, title: str, options: list[tuple[str, str]], *, default: str | None = None
    ) -> str:
        return self._select

    def text(self, title: str, *, default: str | None = None) -> str:
        return self._text


def _capture_status(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the messages passed to output.status (the transcription label)."""
    messages: list[str] = []

    @contextlib.contextmanager
    def _fake_status(message: str, *, json_mode: bool) -> Generator[None]:
        messages.append(message)
        yield

    monkeypatch.setattr(output, "status", _fake_status)
    return messages


@pytest.fixture
def ctx() -> WizardContext:
    return WizardContext(state=AppState(), profile="default", json_mode=False)


def test_auth_skips_when_key_already_present(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk_test")
    assert sections.auth(NonInteractivePrompter(), ctx) is SectionResult.SKIPPED


def test_first_request_uses_sample_on_empty_input(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk_test")
    seen: dict[str, object] = {}

    def _fake(
        api_key: str, source: object, *, sample: bool, transcription_config: object
    ) -> _FakeTranscript:
        seen["source"] = source
        seen["sample"] = sample
        return _FakeTranscript()

    monkeypatch.setattr(transcribe_exec, "run_transcription", _fake)
    monkeypatch.setattr(transcribe_render, "render_transcript_result", lambda *a, **k: None)
    status_messages = _capture_status(monkeypatch)
    # NonInteractivePrompter.text returns its default ("") → Enter → sample.
    assert sections.first_request(NonInteractivePrompter(), ctx) is SectionResult.DONE
    assert seen["source"] is None
    assert seen["sample"] is True
    assert status_messages == ["Transcribing the sample clip…"]


def test_first_request_uses_custom_source(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk_test")
    seen: dict[str, object] = {}

    def _fake(
        api_key: str, source: object, *, sample: bool, transcription_config: object
    ) -> _FakeTranscript:
        seen["source"] = source
        seen["sample"] = sample
        return _FakeTranscript()

    monkeypatch.setattr(transcribe_exec, "run_transcription", _fake)
    monkeypatch.setattr(transcribe_render, "render_transcript_result", lambda *a, **k: None)
    status_messages = _capture_status(monkeypatch)
    assert sections.first_request(_ScriptedPrompter(text="meeting.mp3"), ctx) is SectionResult.DONE
    assert seen["source"] == "meeting.mp3"
    assert seen["sample"] is False
    assert status_messages == ["Transcribing meeting.mp3…"]


def test_first_request_handles_failure(ctx: WizardContext, monkeypatch: pytest.MonkeyPatch) -> None:
    from aai_cli.errors import APIError

    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk_test")

    def _boom(*a: object, **k: object) -> _FakeTranscript:
        raise APIError("nope")

    monkeypatch.setattr(transcribe_exec, "run_transcription", _boom)
    assert sections.first_request(_ScriptedPrompter(text="bad.mp3"), ctx) is SectionResult.FAILED


def test_environment_is_non_blocking(ctx: WizardContext, monkeypatch: pytest.MonkeyPatch) -> None:
    # Even if checks warn/fail, the section never blocks the wizard.
    seen: dict[str, object] = {}

    def _capture_render(payload: dict[str, object]) -> str:
        seen.update(payload)
        return ""

    monkeypatch.setattr("aai_cli.commands.doctor._render", _capture_render)
    assert sections.environment(NonInteractivePrompter(), ctx) is SectionResult.DONE
    # The environment section always renders as a non-fatal report (ok=True).
    assert seen["ok"] is True


def test_build_path_skip_choice_does_nothing(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def _fake_run_init(*a: object, **k: object) -> Path:
        nonlocal called
        called = True
        return Path()

    monkeypatch.setattr(init_cmd, "run_init", _fake_run_init)
    # NonInteractivePrompter.select returns the default; build_path's default is "skip".
    assert sections.build_path(NonInteractivePrompter(), ctx) is SectionResult.SKIPPED
    assert called is False


def test_next_steps(ctx: WizardContext) -> None:
    assert sections.next_steps(NonInteractivePrompter(), ctx) is SectionResult.DONE


def test_welcome_cold_start(ctx: WizardContext) -> None:
    assert sections.welcome(NonInteractivePrompter(), ctx) is SectionResult.DONE


def test_auth_browser_path(ctx: WizardContext, monkeypatch: pytest.MonkeyPatch) -> None:
    # Onboarding signs in via the browser only — there is no API-key paste path.
    monkeypatch.setattr(sections, "persist_browser_login", lambda *a, **k: None)
    assert sections.auth(_ScriptedPrompter(), ctx) is SectionResult.DONE


def test_build_path_scaffolds(ctx: WizardContext, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    seen: dict[str, object] = {}

    def _fake_run_init(*a: object, **k: object) -> Path:
        nonlocal calls
        calls += 1
        seen.update(k)
        return Path()

    monkeypatch.setattr(init_cmd, "run_init", _fake_run_init)
    prompter = _ScriptedPrompter(select="audio-transcription", confirm=True)
    result = sections.build_path(prompter, ctx)
    assert result is SectionResult.DONE
    assert calls == 1
    # The scaffold confirmation defaults to Yes (a False mutant would change the prompt).
    assert prompter.confirm_defaults == [True]
    # Pin the exact run_init kwargs the wizard relies on (each is a mutated literal):
    # a non-blocking, non-opening scaffold of the chosen template on the default port.
    assert seen["template"] == "audio-transcription"
    assert seen["directory"] is None
    assert seen["no_install"] is False
    assert seen["no_open"] is True
    assert seen["force"] is False
    assert seen["here"] is False
    assert seen["port"] == 3000
    assert seen["json_mode"] is False
    assert seen["launch"] is False


def test_build_path_declined_after_select(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def _fake_run_init(*a: object, **k: object) -> Path:
        nonlocal called
        called = True
        return Path()

    monkeypatch.setattr(init_cmd, "run_init", _fake_run_init)
    result = sections.build_path(_ScriptedPrompter(select="voice-agent", confirm=False), ctx)
    assert result is SectionResult.SKIPPED
    assert called is False


def test_build_path_run_init_failure(ctx: WizardContext, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*a: object, **k: object) -> Path:
        raise typer.Exit(code=1)

    monkeypatch.setattr(init_cmd, "run_init", _boom)
    result = sections.build_path(_ScriptedPrompter(select="live-captions", confirm=True), ctx)
    assert result is SectionResult.FAILED


def test_claude_code_skipped(ctx: WizardContext) -> None:
    assert sections.claude_code(NonInteractivePrompter(), ctx) is SectionResult.SKIPPED


def _passing_step(*a: object, **k: object) -> Step:
    return {"name": "x", "status": "installed", "detail": "ok"}


def test_claude_code_done(ctx: WizardContext, monkeypatch: pytest.MonkeyPatch) -> None:
    forces: dict[str, object] = {}

    def _mcp(scope: str, *, force: bool) -> Step:
        forces["mcp"] = force
        return _passing_step()

    def _skill(*, force: bool) -> Step:
        forces["skill"] = force
        return _passing_step()

    def _cli_skill(*, force: bool) -> Step:
        forces["cli_skill"] = force
        return _passing_step()

    monkeypatch.setattr(setup_cmd, "_install_mcp", _mcp)
    monkeypatch.setattr(setup_cmd, "_install_skill", _skill)
    monkeypatch.setattr(setup_cmd, "_install_cli_skill", _cli_skill)
    assert sections.claude_code(_ScriptedPrompter(confirm=True), ctx) is SectionResult.DONE
    # The wizard never force-overwrites existing installs (force=False everywhere).
    assert forces == {"mcp": False, "skill": False, "cli_skill": False}


def test_claude_code_failed(ctx: WizardContext, monkeypatch: pytest.MonkeyPatch) -> None:
    def _failing_step(*a: object, **k: object) -> Step:
        return {"name": "x", "status": "failed", "detail": "no npx"}

    monkeypatch.setattr(setup_cmd, "_install_mcp", _passing_step)
    monkeypatch.setattr(setup_cmd, "_install_skill", _failing_step)
    monkeypatch.setattr(setup_cmd, "_install_cli_skill", _passing_step)
    assert sections.claude_code(_ScriptedPrompter(confirm=True), ctx) is SectionResult.FAILED
