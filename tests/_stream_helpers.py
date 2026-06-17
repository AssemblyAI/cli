"""Shared building blocks for the `assembly stream` run-path tests.

Split out of test_stream_exec.py so the save-flag suites (test_stream_exec.py and
test_stream_save_dir.py) share one set of fakes — a mic, turn events, a frozen clock,
and the StreamOptions defaults — instead of duplicating them per file.
"""

from __future__ import annotations

from datetime import datetime

from aai_cli.commands.stream import DEFAULT_SPEECH_MODEL
from aai_cli.commands.stream import _exec as stream_exec
from aai_cli.core import llm

# The CLI's flag defaults, as data. Tests override per-case with dataclasses.replace.
DEFAULTS = stream_exec.StreamOptions(
    source=None,
    sample=False,
    from_stdin=False,
    sample_rate=None,
    device=None,
    system_audio=False,
    system_audio_only=False,
    speech_model=DEFAULT_SPEECH_MODEL,
    encoding=None,
    language_detection=None,
    domain=None,
    prompt=None,
    keyterms_prompt=None,
    end_of_turn_confidence_threshold=None,
    min_turn_silence=None,
    max_turn_silence=None,
    turn_detection=None,
    vad_threshold=None,
    format_turns=None,
    include_partial_turns=None,
    speaker_labels=None,
    max_speakers=None,
    voice_focus=None,
    voice_focus_threshold=None,
    inactivity_timeout=None,
    filter_profanity=None,
    redact_pii=None,
    redact_pii_policy=None,
    redact_pii_sub=None,
    webhook_url=None,
    webhook_auth_header=None,
    llm_prompt=None,
    llm_interval=10.0,
    model=llm.DEFAULT_MODEL,
    max_tokens=llm.DEFAULT_MAX_TOKENS,
    config_kv=None,
    config_file=None,
    output_field=None,
    show_code=False,
    save_audio=None,
    save_transcript=None,
    save_dir=None,
    name=None,
    auto_name=False,
    no_save_audio=False,
)


class FakeMic:
    """Mirrors MicrophoneSource's keyword signature (see microphone.py)."""

    def __init__(self, *, target_rate=None, device=None, capture_rate=None, on_open=None):
        self.sample_rate = capture_rate or 16000
        self.device = device

    def __iter__(self):
        return iter([b"\x00\x00"])


class RecordingMic(FakeMic):
    """A mic that yields known PCM so the tee'd WAV's contents can be asserted."""

    PCM = b"\x01\x02\x03\x04\x05\x06\x07\x08"

    def __iter__(self):
        return iter([self.PCM])


class FakeTurn:
    """A streaming turn event with just the attributes the session reads."""

    def __init__(self, transcript, *, end_of_turn=True, speaker_label=None):
        self.transcript = transcript
        self.end_of_turn = end_of_turn
        self.speaker_label = speaker_label


def emit_turns(*events):
    """A fake client.stream_audio that drains the audio (driving any tee) then fires
    each turn through the session's on_turn callback, like the real SDK reader."""

    def _fake(api_key, source, *, params, on_turn, **_kwargs):
        b"".join(source)  # draining is what writes the tee'd WAV, if any
        for event in events:
            on_turn(event)

    return _fake


class FixedDatetime:
    """Freezes datetime.now() so the auto-assembled filename is deterministic."""

    @staticmethod
    def now(*_args, **_kwargs):
        # Naive local wall-clock; _exec's .astimezone() keeps the same 14:30:05.
        return datetime(2026, 6, 16, 14, 30, 5)
