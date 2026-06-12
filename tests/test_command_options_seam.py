"""Direct tests of the options/run seams (transcribe/agent/speak/llm exec modules).

Each command module parses argv into a frozen <Cmd>Options dataclass; everything
after that is a module-level run function of plain data. These tests construct
options directly (dataclasses.replace off a defaults instance) instead of
round-tripping argv through CliRunner. The stream seam's tests live in
test_stream_exec.py.
"""

from __future__ import annotations

import dataclasses

import pytest
import typer

from aai_cli import agent_exec, choices, config, llm, llm_exec, speak_exec, transcribe_exec
from aai_cli.agent.session import DEFAULT_GREETING, DEFAULT_PROMPT
from aai_cli.agent.voices import DEFAULT_VOICE
from aai_cli.context import AppState
from aai_cli.errors import CLIError, UsageError
from aai_cli.options import DEFAULT_BATCH_CONCURRENCY

# The CLI's flag defaults, as data. Tests override per-case with dataclasses.replace.
TRANSCRIBE_DEFAULTS = transcribe_exec.TranscribeOptions(
    source=None,
    sample=False,
    from_stdin=False,
    concurrency=DEFAULT_BATCH_CONCURRENCY,
    force=False,
    speech_model=None,
    language_code=None,
    language_detection=None,
    keyterms_prompt=None,
    temperature=None,
    prompt=None,
    punctuate=None,
    format_text=None,
    disfluencies=None,
    speaker_labels=False,
    speakers_expected=None,
    multichannel=None,
    redact_pii=None,
    redact_pii_policy=None,
    redact_pii_sub=None,
    redact_pii_audio=None,
    filter_profanity=None,
    content_safety=None,
    content_safety_confidence=None,
    speech_threshold=None,
    summarization=None,
    summary_model=None,
    summary_type=None,
    auto_chapters=None,
    sentiment_analysis=None,
    entity_detection=None,
    auto_highlights=None,
    topic_detection=None,
    word_boost=None,
    custom_spelling_file=None,
    audio_start=None,
    audio_end=None,
    download_sections=None,
    webhook_url=None,
    webhook_auth_header=None,
    translate_to=None,
    config_kv=None,
    config_file=None,
    llm_prompt=None,
    model=llm.DEFAULT_MODEL,
    max_tokens=llm.DEFAULT_MAX_TOKENS,
    output_field=None,
    chars_per_caption=None,
    out=None,
    show_code=False,
)

AGENT_DEFAULTS = agent_exec.AgentOptions(
    source=None,
    sample=False,
    voice=DEFAULT_VOICE,
    system_prompt=DEFAULT_PROMPT,
    system_prompt_file=None,
    greeting=DEFAULT_GREETING,
    device=None,
    output_field=None,
    show_code=False,
)

SPEAK_DEFAULTS = speak_exec.SpeakOptions(
    text=None,
    voice=[],
    language=speak_exec.DEFAULT_LANGUAGE,
    sample_rate=None,
    out=None,
)

LLM_DEFAULTS = llm_exec.LlmOptions(
    prompt=None,
    model=llm.DEFAULT_MODEL,
    transcript_id=None,
    system=None,
    follow=False,
    output_field=None,
    max_tokens=llm.DEFAULT_MAX_TOKENS,
)


@pytest.mark.parametrize(
    "defaults",
    [TRANSCRIBE_DEFAULTS, AGENT_DEFAULTS, SPEAK_DEFAULTS, LLM_DEFAULTS],
    ids=["transcribe", "agent", "speak", "llm"],
)
def test_options_are_immutable(defaults):
    field_name = dataclasses.fields(defaults)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(defaults, field_name, None)


def test_run_transcribe_validates_flags_before_credentials():
    # No API key configured: a flag conflict surfaces as a usage error, not
    # NotAuthenticated — validation runs before any credential resolution.
    with pytest.raises(UsageError):
        transcribe_exec.run_transcribe(
            dataclasses.replace(
                TRANSCRIBE_DEFAULTS, language_code="en_us", language_detection=True
            ),
            AppState(),
            json_mode=False,
        )


def test_transcribe_flags_drop_unset_speaker_labels():
    # The boolean --speaker-labels flag maps to None when unset (so the request
    # omits the field entirely), and True only when explicitly enabled.
    assert TRANSCRIBE_DEFAULTS.flags(None)["speaker_labels"] is None
    enabled = dataclasses.replace(TRANSCRIBE_DEFAULTS, speaker_labels=True)
    assert enabled.flags(None)["speaker_labels"] is True


def test_run_agent_session_config_without_cli(monkeypatch):
    config.set_api_key("default", "sk_live")
    seen = {}

    def fake_run_session(api_key, *, renderer, player, mic, config):
        seen["api_key"] = api_key
        seen["config"] = config

    monkeypatch.setattr(agent_exec, "run_session", fake_run_session)
    monkeypatch.setattr(agent_exec, "DuplexAudio", _FakeDuplex)

    agent_exec.run_agent(
        dataclasses.replace(AGENT_DEFAULTS, greeting="Ahoy"), AppState(), json_mode=True
    )
    assert seen["api_key"] == "sk_live"
    run_config = seen["config"]
    assert run_config.voice == DEFAULT_VOICE
    assert run_config.greeting == "Ahoy"
    assert run_config.full_duplex is True
    assert run_config.exit_after_reply is False


class _FakeDuplex:
    def __init__(self, *, target_rate=None, device=None):
        self.mic = object()
        self.player = object()


def test_run_agent_ctrl_c_stops_cleanly(monkeypatch):
    # Ctrl-C is the normal "user hung up" signal: the session ends without an error.
    config.set_api_key("default", "sk_live")

    def raise_interrupt(api_key, *, renderer, player, mic, config):
        raise KeyboardInterrupt

    monkeypatch.setattr(agent_exec, "run_session", raise_interrupt)
    monkeypatch.setattr(agent_exec, "DuplexAudio", _FakeDuplex)
    agent_exec.run_agent(AGENT_DEFAULTS, AppState(), json_mode=True)  # no exception


def test_run_agent_broken_pipe_exits_zero(monkeypatch):
    # A closed downstream pipe (`assembly agent | head`) is a clean stop, not a failure.
    config.set_api_key("default", "sk_live")

    def raise_broken_pipe(api_key, *, renderer, player, mic, config):
        raise BrokenPipeError

    monkeypatch.setattr(agent_exec, "run_session", raise_broken_pipe)
    monkeypatch.setattr(agent_exec, "DuplexAudio", _FakeDuplex)
    with pytest.raises(typer.Exit) as exc:
        agent_exec.run_agent(AGENT_DEFAULTS, AppState(), json_mode=True)
    assert exc.value.exit_code == 0


def test_run_speak_requires_sandbox():
    # The active environment defaults to production, which has no streaming-TTS host.
    with pytest.raises(CLIError) as exc:
        speak_exec.run_speak(SPEAK_DEFAULTS, AppState(), json_mode=False)
    assert exc.value.exit_code == 2
    assert "--sandbox" in (exc.value.suggestion or "")


def test_run_llm_follow_rejects_output_field():
    with pytest.raises(UsageError):
        llm_exec.run_llm(
            dataclasses.replace(
                LLM_DEFAULTS, follow=True, prompt="x", output_field=choices.TextOrJson.text
            ),
            AppState(),
            json_mode=False,
        )
