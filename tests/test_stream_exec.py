"""Direct tests of the `assembly stream` options/run seam (aai_cli.commands.stream._exec).

The command module only parses argv into a StreamOptions; everything after that is
run_stream, a plain function of data. These tests drive validation, flag mapping,
and session wiring by constructing options directly — no CliRunner argv round-trip,
no merged-stream output parsing. The --save-dir suite lives in test_stream_save_dir.py;
the shared fakes (mic, turns, defaults) live in tests/_stream_helpers.py.
"""

from __future__ import annotations

import dataclasses
import wave
from pathlib import Path

import pytest

from aai_cli.app.context import AppState
from aai_cli.commands.stream import _exec as stream_exec
from aai_cli.core import config
from aai_cli.core.errors import CLIError, UsageError
from aai_cli.streaming.turn_presets import TurnDetectionPreset
from tests._stream_helpers import DEFAULTS, FakeMic, FakeTurn, RecordingMic, emit_turns


def test_run_stream_maps_flags_to_params_without_cli(monkeypatch):
    # The seam's payoff: assert the flag->StreamingParameters mapping by constructing
    # options directly, instead of threading a giant argv through CliRunner.
    config.set_api_key("default", "sk_live")
    seen = {}

    def fake_stream_audio(api_key, source, *, params, **_kwargs):
        seen["api_key"] = api_key
        seen["params"] = params

    monkeypatch.setattr(stream_exec.client, "stream_audio", fake_stream_audio)
    monkeypatch.setattr(stream_exec, "MicrophoneSource", FakeMic)

    stream_exec.run_stream(
        dataclasses.replace(
            DEFAULTS,
            domain="medical-v1",
            prompt="expect drug names",
            keyterms_prompt=["AssemblyAI"],
        ),
        AppState(),
        json_mode=True,
    )
    assert seen["api_key"] == "sk_live"
    params = seen["params"]
    assert params.domain == "medical-v1"
    assert params.prompt == "expect drug names"
    assert params.keyterms_prompt == ["AssemblyAI"]


def test_run_stream_validates_before_resolving_credentials():
    # No API key is configured: a flag conflict must surface as a usage error, not
    # as NotAuthenticated — validation runs before any credential resolution.
    with pytest.raises(UsageError):
        stream_exec.run_stream(
            dataclasses.replace(DEFAULTS, system_audio=True, system_audio_only=True),
            AppState(),
            json_mode=False,
        )


def test_redact_pii_sub_enum_maps_to_its_string_value():
    # --redact-pii-sub is an SDK enum (validated at parse time), so base_flags must
    # unwrap it to the canonical string the streaming config expects, not pass the
    # enum member through.
    from assemblyai import PIISubstitutionPolicy

    opts = dataclasses.replace(DEFAULTS, redact_pii_sub=PIISubstitutionPolicy.hash)
    assert opts.base_flags()["redact_pii_sub"] == "hash"
    assert DEFAULTS.base_flags()["redact_pii_sub"] is None  # unset stays None


def test_turn_detection_preset_fills_base_flags():
    # --turn-detection balanced supplies the documented (0.4, 400, 1280) trio.
    opts = dataclasses.replace(DEFAULTS, turn_detection=TurnDetectionPreset.balanced)
    flags = opts.base_flags()
    assert flags["end_of_turn_confidence_threshold"] == 0.4
    assert flags["min_turn_silence"] == 400
    assert flags["max_turn_silence"] == 1280


def test_explicit_turn_flag_overrides_the_preset_slot():
    # A raw --min-turn-silence wins over the preset's value; the other slots stay.
    opts = dataclasses.replace(
        DEFAULTS, turn_detection=TurnDetectionPreset.balanced, min_turn_silence=900
    )
    flags = opts.base_flags()
    assert flags["min_turn_silence"] == 900
    assert flags["max_turn_silence"] == 1280


def test_no_preset_leaves_turn_flags_unset():
    flags = DEFAULTS.base_flags()
    assert flags["end_of_turn_confidence_threshold"] is None
    assert flags["min_turn_silence"] is None
    assert flags["max_turn_silence"] is None


def test_stream_options_are_immutable():
    field_name = "sample"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(DEFAULTS, field_name, True)


def test_source_options_are_immutable():
    # The input carrier is frozen too, so a validation/dispatch step can't mutate which
    # source a run reads from after the flags are resolved.
    from aai_cli.streaming.validate import SourceOptions

    opts = SourceOptions(
        source=None,
        sample=False,
        sample_rate=None,
        device=None,
        system_audio=False,
        system_audio_only=False,
    )
    field_name = "system_audio"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(opts, field_name, True)


def test_save_targets_are_immutable():
    # The resolved save destinations are a frozen carrier (like StreamOptions), so a
    # later step can't quietly retarget a file mid-run.
    field_name = "audio"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(stream_exec.SaveTargets(), field_name, Path("x.wav"))


# --- batch streaming (--from-stdin) validation -----------------------------
# Each conflict is rejected before stdin is read, so these raise without a pipe.
@pytest.mark.parametrize(
    "overrides",
    [
        {"from_stdin": True, "source": "a.wav"},  # a positional source
        {"from_stdin": True, "sample": True},  # the hosted sample
        {"from_stdin": True, "system_audio": True},  # live system capture
        {"from_stdin": True, "system_audio_only": True},
        {"from_stdin": True, "device": 2},  # mic-only capture flags
        {"from_stdin": True, "sample_rate": 44100},
        {"from_stdin": True, "show_code": True},  # renders one source
        {"from_stdin": True, "save_audio": Path("out.wav")},  # tees one stream
        {"from_stdin": True, "save_transcript": Path("out.txt")},  # saves one transcript
        {"from_stdin": True, "save_dir": Path("rec")},  # auto-names one run
        {"from_stdin": True, "name": "Standup"},  # --name needs --save-dir
        {"from_stdin": True, "auto_name": True},  # --auto-name names one run
        {"from_stdin": True, "no_save_audio": True},  # --no-save-audio is a single-run flag
    ],
)
def test_from_stdin_rejects_incompatible_flags(overrides):
    with pytest.raises(UsageError):
        stream_exec.run_stream(
            dataclasses.replace(DEFAULTS, **overrides), AppState(), json_mode=False
        )


def test_from_stdin_rejects_llm_with_text_output():
    # --llm renders a live panel; -o text is a contradictory output shape.
    from aai_cli.core import choices

    with pytest.raises(UsageError):
        stream_exec.run_stream(
            dataclasses.replace(
                DEFAULTS,
                from_stdin=True,
                llm_prompt=["summarize"],
                output_field=choices.TextOrJson.text,
            ),
            AppState(),
            json_mode=False,
        )


def test_from_stdin_empty_stdin_is_a_usage_error(monkeypatch):
    # An empty pipe (nothing to stream) is a clean usage error, not a silent no-op.
    monkeypatch.setattr(stream_exec.stdio, "iter_piped_stdin_lines", lambda: iter([]))
    with pytest.raises(UsageError):
        stream_exec.run_stream(
            dataclasses.replace(DEFAULTS, from_stdin=True), AppState(), json_mode=True
        )


def test_from_stdin_dedupes_sources_keeping_order(monkeypatch):
    # Duplicate lines stream once, in first-seen order — the batch driver receives the
    # deduped list (mirrors `transcribe --from-stdin`).
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr(
        stream_exec.stdio, "iter_piped_stdin_lines", lambda: iter(["a.wav", "a.wav", "b.wav"])
    )
    seen: dict[str, list[str]] = {"sources": []}

    def fake_stream_batch(sources, *, make_session, open_source, renderer, json_mode):
        seen["sources"] = list(sources)

    monkeypatch.setattr(stream_exec, "stream_batch_sources", fake_stream_batch)
    stream_exec.run_stream(
        dataclasses.replace(DEFAULTS, from_stdin=True), AppState(), json_mode=True
    )
    assert seen["sources"] == ["a.wav", "b.wav"]


# --- --save-audio (tee the streamed PCM to a WAV) --------------------------
def test_save_audio_tees_streamed_pcm_to_a_wav(monkeypatch, tmp_path):
    # The bytes the streaming API receives are also written to --save-audio, verbatim,
    # as a 16-bit mono WAV at the source's sample rate.
    config.set_api_key("default", "sk_live")
    out = tmp_path / "rec.wav"

    def fake_stream_audio(api_key, source, *, params, **_kwargs):
        # Draining the iterable is what drives the tee — mirror the real SDK consuming it.
        sent = b"".join(source)
        assert sent == RecordingMic.PCM  # the API still sees the unaltered audio

    monkeypatch.setattr(stream_exec.client, "stream_audio", fake_stream_audio)
    monkeypatch.setattr(stream_exec, "MicrophoneSource", RecordingMic)

    stream_exec.run_stream(
        dataclasses.replace(DEFAULTS, save_audio=out), AppState(), json_mode=True
    )

    assert out.is_file()
    with wave.open(str(out), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 16000  # FakeMic's reported rate
        assert w.readframes(w.getnframes()) == RecordingMic.PCM


def test_save_audio_not_written_when_flag_unset(monkeypatch, tmp_path):
    # Without --save-audio, the default run leaves no stray WAV behind (kills a mutant
    # that tees unconditionally).
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr(stream_exec.client, "stream_audio", lambda *a, **k: b"".join(a[1]))
    monkeypatch.setattr(stream_exec, "MicrophoneSource", RecordingMic)

    stream_exec.run_stream(DEFAULTS, AppState(), json_mode=True)

    assert list(tmp_path.glob("*.wav")) == []


def test_save_audio_rejects_system_audio():
    # The mic + system streams can't share one file, so the combo is a usage error
    # (raised before credentials).
    with pytest.raises(UsageError):
        stream_exec.run_stream(
            dataclasses.replace(DEFAULTS, save_audio=Path("rec.wav"), system_audio=True),
            AppState(),
            json_mode=False,
        )


def test_save_audio_allows_system_audio_only(monkeypatch, tmp_path):
    # --save-audio is rejected for the two-stream --system-audio, but --system-audio-only
    # is a single stream, so it tees to the one explicit WAV like any other source.
    config.set_api_key("default", "sk_live")
    out = tmp_path / "rec.wav"

    class FakeSystemAudio:
        def __init__(self, *, on_open=None):
            self.sample_rate = 16000

        def __iter__(self):
            return iter([RecordingMic.PCM])

    def fake_stream_audio(api_key, source, *, params, **_kwargs):
        assert b"".join(source) == RecordingMic.PCM

    monkeypatch.setattr(stream_exec, "MacSystemAudioSource", FakeSystemAudio)
    monkeypatch.setattr(stream_exec.client, "stream_audio", fake_stream_audio)

    stream_exec.run_stream(
        dataclasses.replace(DEFAULTS, save_audio=out, system_audio_only=True),
        AppState(),
        json_mode=True,
    )

    with wave.open(str(out), "rb") as w:
        assert w.readframes(w.getnframes()) == RecordingMic.PCM


def test_save_audio_rejects_show_code():
    # --show-code emits SDK code that doesn't tee audio, so the combo is rejected.
    with pytest.raises(UsageError):
        stream_exec.run_stream(
            dataclasses.replace(DEFAULTS, save_audio=Path("rec.wav"), show_code=True),
            AppState(),
            json_mode=False,
        )


def test_save_audio_rejects_missing_parent_dir(tmp_path):
    # A path under a directory that doesn't exist is a clean path error, before auth.
    config.set_api_key("default", "sk_live")
    with pytest.raises(CLIError) as excinfo:
        stream_exec.run_stream(
            dataclasses.replace(DEFAULTS, save_audio=tmp_path / "nope" / "rec.wav"),
            AppState(),
            json_mode=False,
        )
    assert excinfo.value.error_type == "save_audio_path"


# --- --save-transcript (write the finalized turn text) ---------------------
def test_save_transcript_writes_only_finalized_nonempty_turns(monkeypatch, tmp_path):
    # Each finalized, non-empty turn is one line; partials and empty turns are skipped.
    config.set_api_key("default", "sk_live")
    out = tmp_path / "notes.txt"
    monkeypatch.setattr(
        stream_exec.client,
        "stream_audio",
        emit_turns(
            FakeTurn("partial", end_of_turn=False),  # not finalized -> skipped
            FakeTurn("hello world"),
            FakeTurn("", end_of_turn=True),  # finalized but empty -> skipped
            FakeTurn("goodbye"),
        ),
    )
    monkeypatch.setattr(stream_exec, "MicrophoneSource", FakeMic)

    stream_exec.run_stream(
        dataclasses.replace(DEFAULTS, save_transcript=out), AppState(), json_mode=True
    )

    assert out.read_text(encoding="utf-8") == "hello world\ngoodbye\n"


def test_save_transcript_prefixes_diarized_speaker(monkeypatch, tmp_path):
    # A diarized turn is saved with the same "Speaker A:" prefix the text renderer uses.
    config.set_api_key("default", "sk_live")
    out = tmp_path / "notes.txt"
    monkeypatch.setattr(
        stream_exec.client, "stream_audio", emit_turns(FakeTurn("hi", speaker_label="A"))
    )
    monkeypatch.setattr(stream_exec, "MicrophoneSource", FakeMic)

    stream_exec.run_stream(
        dataclasses.replace(DEFAULTS, save_transcript=out), AppState(), json_mode=True
    )

    assert out.read_text(encoding="utf-8") == "Speaker A: hi\n"


def test_no_transcript_file_written_when_flag_unset(monkeypatch, tmp_path):
    # Without a save flag the default run leaves no stray .txt (kills a mutant that
    # writes unconditionally).
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr(stream_exec.client, "stream_audio", emit_turns(FakeTurn("hi")))
    monkeypatch.setattr(stream_exec, "MicrophoneSource", FakeMic)

    stream_exec.run_stream(DEFAULTS, AppState(), json_mode=True)

    assert list(tmp_path.glob("*.txt")) == []


def test_save_transcript_rejects_missing_parent_dir(tmp_path):
    config.set_api_key("default", "sk_live")
    with pytest.raises(CLIError) as excinfo:
        stream_exec.run_stream(
            dataclasses.replace(DEFAULTS, save_transcript=tmp_path / "nope" / "notes.txt"),
            AppState(),
            json_mode=False,
        )
    assert excinfo.value.error_type == "save_transcript_path"
