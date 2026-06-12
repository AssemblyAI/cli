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
from aai_cli.errors import CLIError
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

    interactive = True

    def __init__(self, *, select: str = "skip", confirm: bool = True, text: str = "k") -> None:
        self._select = select
        self._confirm = confirm
        self._text = text
        self.confirm_defaults: list[bool] = []
        self.text_titles: list[str] = []
        self.notes: list[str] = []

    def section(self, title: str) -> None:
        pass

    def note(self, message: str) -> None:
        self.notes.append(message)

    def confirm(self, title: str, *, default: bool = True) -> bool:
        self.confirm_defaults.append(default)
        return self._confirm

    def select(
        self, title: str, options: list[tuple[str, str]], *, default: str | None = None
    ) -> str:
        return self._select

    def text(self, title: str, *, default: str | None = None) -> str:
        self.text_titles.append(title)
        return self._text


def _capture_status(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the messages passed to output.status (the transcription label)."""
    messages: list[str] = []

    @contextlib.contextmanager
    def _fake_status(message: str, *, json_mode: bool, quiet: bool = False) -> Generator[None]:
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
    prompter = _ScriptedPrompter(text="meeting.mp3")
    assert sections.first_request(prompter, ctx) is SectionResult.DONE
    assert seen["source"] == "meeting.mp3"
    assert seen["sample"] is False
    assert status_messages == ["Transcribing meeting.mp3…"]
    # The prompt advertises every accepted source kind, including podcast pages.
    assert any("YouTube/podcast URL" in t for t in prompter.text_titles)


def test_first_request_handles_failure(ctx: WizardContext, monkeypatch: pytest.MonkeyPatch) -> None:
    from aai_cli.errors import APIError

    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk_test")

    def _boom(*a: object, **k: object) -> _FakeTranscript:
        raise APIError("nope")

    monkeypatch.setattr(transcribe_exec, "run_transcription", _boom)
    assert sections.first_request(_ScriptedPrompter(text="bad.mp3"), ctx) is SectionResult.FAILED


def _capture_console(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    printed: list[str] = []
    monkeypatch.setattr(
        output.console, "print", lambda *a, **k: printed.append(str(a[0]) if a else "")
    )
    return printed


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


def test_next_steps(ctx: WizardContext, monkeypatch: pytest.MonkeyPatch) -> None:
    printed = _capture_console(monkeypatch)
    assert sections.next_steps(NonInteractivePrompter(), ctx) is SectionResult.DONE
    flat = "\n".join(printed)
    # Human mode prints the three next-step hints.
    assert "assembly transcribe" in flat
    assert "assembly stream" in flat
    assert "assembly init" in flat


def test_next_steps_json_mode_keeps_stdout_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    json_ctx = WizardContext(state=AppState(), profile="default", json_mode=True)
    printed = _capture_console(monkeypatch)
    assert sections.next_steps(NonInteractivePrompter(), json_ctx) is SectionResult.DONE
    assert printed == []


def test_first_request_json_mode_skips_human_transcript_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Under --json the wizard's summary owns stdout; the Rich transcript render
    # would corrupt it.
    json_ctx = WizardContext(state=AppState(), profile="default", json_mode=True)
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk_test")
    monkeypatch.setattr(transcribe_exec, "run_transcription", lambda *a, **k: _FakeTranscript())
    rendered = {"n": 0}
    monkeypatch.setattr(
        transcribe_render,
        "render_transcript_result",
        lambda *a, **k: rendered.__setitem__("n", rendered["n"] + 1),
    )
    assert sections.first_request(NonInteractivePrompter(), json_ctx) is SectionResult.DONE
    assert rendered["n"] == 0


def test_first_request_human_mode_renders_transcript(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk_test")
    monkeypatch.setattr(transcribe_exec, "run_transcription", lambda *a, **k: _FakeTranscript())
    rendered = {"n": 0}
    monkeypatch.setattr(
        transcribe_render,
        "render_transcript_result",
        lambda *a, **k: rendered.__setitem__("n", rendered["n"] + 1),
    )
    assert sections.first_request(NonInteractivePrompter(), ctx) is SectionResult.DONE
    assert rendered["n"] == 1


def test_welcome_cold_start(ctx: WizardContext) -> None:
    assert sections.welcome(NonInteractivePrompter(), ctx) is SectionResult.DONE


def test_auth_browser_path(ctx: WizardContext, monkeypatch: pytest.MonkeyPatch) -> None:
    # Onboarding signs in via the browser only — there is no API-key paste path.
    monkeypatch.setattr(sections, "persist_browser_login", lambda *a, **k: None)
    assert sections.auth(_ScriptedPrompter(), ctx) is SectionResult.DONE


def test_auth_noninteractive_fails_without_blocking_on_browser(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A non-interactive session (agent/CI) must NOT start a browser login that would
    # block for two minutes on a callback no one can produce — it fails fast with the
    # actionable next step instead.
    def _must_not_run(*a: object, **k: object) -> None:
        raise AssertionError("non-interactive onboarding must not start a browser login")

    monkeypatch.setattr(sections, "persist_browser_login", _must_not_run)

    class _RecordingNonInteractive(NonInteractivePrompter):
        def __init__(self) -> None:
            self.notes: list[str] = []

        def note(self, message: str) -> None:
            self.notes.append(message)

    prompter = _RecordingNonInteractive()
    assert sections.auth(prompter, ctx) is SectionResult.FAILED
    assert any("ASSEMBLYAI_API_KEY" in note for note in prompter.notes)


def test_build_path_scaffolds(ctx: WizardContext, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    seen: dict[str, object] = {}

    def _fake_run_init(
        state: object,
        *,
        template: str | None,
        directory: str | None,
        no_install: bool,
        no_open: bool,
        force: bool,
        here: bool,
        port: int,
        json_mode: bool,
        launch: bool = True,
    ) -> Path:
        nonlocal calls
        calls += 1
        seen.update(
            template=template,
            directory=directory,
            no_install=no_install,
            no_open=no_open,
            force=force,
            here=here,
            port=port,
            json_mode=json_mode,
            launch=launch,
        )
        return Path("/scaffolded/app")

    monkeypatch.setattr(init_cmd, "run_init", _fake_run_init)
    prompter = _ScriptedPrompter(select="audio-transcription", confirm=True)
    result = sections.build_path(prompter, ctx)
    assert result is SectionResult.DONE
    assert calls == 1
    # The scaffold target is recorded so launch_app can start it at the end of the wizard.
    assert ctx.scaffolded == Path("/scaffolded/app")
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
    # Nothing scaffolded, so launch_app must have nothing to launch.
    assert ctx.scaffolded is None


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

    monkeypatch.setattr(setup_cmd, "install_mcp", _mcp)
    monkeypatch.setattr(setup_cmd, "install_skill", _skill)
    monkeypatch.setattr(setup_cmd, "install_cli_skill", _cli_skill)
    assert sections.claude_code(_ScriptedPrompter(confirm=True), ctx) is SectionResult.DONE
    # The wizard never force-overwrites existing installs (force=False everywhere).
    assert forces == {"mcp": False, "skill": False, "cli_skill": False}


def test_claude_code_failed(ctx: WizardContext, monkeypatch: pytest.MonkeyPatch) -> None:
    def _failing_step(*a: object, **k: object) -> Step:
        return {"name": "x", "status": "failed", "detail": "no npx"}

    monkeypatch.setattr(setup_cmd, "install_mcp", _passing_step)
    monkeypatch.setattr(setup_cmd, "install_skill", _failing_step)
    monkeypatch.setattr(setup_cmd, "install_cli_skill", _passing_step)
    assert sections.claude_code(_ScriptedPrompter(confirm=True), ctx) is SectionResult.FAILED


def _spy_launch(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Replace init's launch_app with a recorder; returns the captured target + kwargs."""
    captured: dict[str, object] = {}

    def _fake_launch(
        target: Path, *, port: int, use_uv: bool, no_open: bool, json_mode: bool
    ) -> None:
        captured["target"] = target
        captured.update(port=port, use_uv=use_uv, no_open=no_open, json_mode=json_mode)

    monkeypatch.setattr(init_cmd, "launch_app", _fake_launch)
    return captured


def test_launch_app_skipped_when_nothing_scaffolded(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _spy_launch(monkeypatch)
    assert sections.launch_app(_ScriptedPrompter(confirm=True), ctx) is SectionResult.SKIPPED
    assert captured == {}


def test_launch_app_noninteractive_hints_instead_of_blocking(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A headless run must never start a blocking dev server; it leaves the run command.
    captured = _spy_launch(monkeypatch)
    ctx.scaffolded = Path("/scaffolded/app")

    class _RecordingNonInteractive(NonInteractivePrompter):
        def __init__(self) -> None:
            self.notes: list[str] = []

        def note(self, message: str) -> None:
            self.notes.append(message)

    prompter = _RecordingNonInteractive()
    assert sections.launch_app(prompter, ctx) is SectionResult.SKIPPED
    assert captured == {}
    assert any("cd /scaffolded/app && assembly dev" in note for note in prompter.notes)


def test_launch_app_declined_leaves_hint(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _spy_launch(monkeypatch)
    ctx.scaffolded = Path("/scaffolded/app")
    prompter = _ScriptedPrompter(confirm=False)
    assert sections.launch_app(prompter, ctx) is SectionResult.SKIPPED
    assert captured == {}
    assert any("cd /scaffolded/app && assembly dev" in note for note in prompter.notes)


@pytest.mark.parametrize("uv", [True, False])
def test_launch_app_launches_scaffolded_app(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch, uv: bool
) -> None:
    captured = _spy_launch(monkeypatch)
    monkeypatch.setattr("aai_cli.init.runner.has_uv", lambda: uv)
    ctx.scaffolded = Path("/scaffolded/app")
    prompter = _ScriptedPrompter(confirm=True)
    assert sections.launch_app(prompter, ctx) is SectionResult.DONE
    # Launching defaults to Yes (an Enter keypress starts the server).
    assert prompter.confirm_defaults == [True]
    # Pin the exact launch kwargs (each is a mutable literal): the default port, the
    # detected runner, a browser that actually opens, and the wizard's human output mode.
    assert captured["target"] == Path("/scaffolded/app")
    assert captured["port"] == 3000
    assert captured["use_uv"] is uv
    assert captured["no_open"] is False
    assert captured["json_mode"] is False


@pytest.mark.parametrize("exc", [typer.Exit(code=1), CLIError("boom")])
def test_launch_app_failure(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    def _boom(*a: object, **k: object) -> None:
        raise exc

    monkeypatch.setattr(init_cmd, "launch_app", _boom)
    ctx.scaffolded = Path("/scaffolded/app")
    assert sections.launch_app(_ScriptedPrompter(confirm=True), ctx) is SectionResult.FAILED
