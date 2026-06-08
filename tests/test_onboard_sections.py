from __future__ import annotations

from pathlib import Path

import pytest

from aai_cli import client, config, transcribe_render
from aai_cli.commands import init as init_cmd
from aai_cli.context import AppState
from aai_cli.onboard import sections
from aai_cli.onboard.prompter import NonInteractivePrompter
from aai_cli.onboard.sections import SectionResult, WizardContext


class _FakeTranscript:
    id = "t_1"
    status = "completed"
    text = "hello"
    utterances = None


@pytest.fixture
def ctx() -> WizardContext:
    return WizardContext(state=AppState(), profile="default", json_mode=False)


def test_auth_skips_when_key_already_present(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk_test")
    assert sections.auth(NonInteractivePrompter(), ctx) is SectionResult.SKIPPED


def test_first_request_transcribes_sample_and_counts(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk_test")
    monkeypatch.setattr(client, "transcribe", lambda *a, **k: _FakeTranscript())
    # Stub the rich render so a minimal fake transcript suffices; we're testing the
    # counter + result, not rendering.
    monkeypatch.setattr(transcribe_render, "render_transcript_result", lambda *a, **k: None)
    assert sections.first_request(NonInteractivePrompter(), ctx) is SectionResult.DONE
    assert config.get_requests_made("default") == 1


def test_environment_is_non_blocking(ctx: WizardContext) -> None:
    # Even if checks warn/fail, the section never blocks the wizard.
    assert sections.environment(NonInteractivePrompter(), ctx) is SectionResult.DONE


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


def test_next_steps_renders_progress(ctx: WizardContext) -> None:
    assert sections.next_steps(NonInteractivePrompter(), ctx) is SectionResult.DONE
