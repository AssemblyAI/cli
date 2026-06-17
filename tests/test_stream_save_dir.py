"""End-to-end tests of `assembly stream --save-dir` through run_stream.

Split from test_stream_exec.py: this file drives the auto-name / note / sidecar /
--no-save-audio behavior over the real session + savedir finalization (only the LLM
gateway is faked). Shared fakes live in tests/_stream_helpers.py.
"""

from __future__ import annotations

import dataclasses
import json
import wave
from pathlib import Path

import pytest

from aai_cli.app.context import AppState
from aai_cli.commands.stream import _exec as stream_exec
from aai_cli.commands.stream import _save as stream_save
from aai_cli.core import config
from aai_cli.core.errors import UsageError
from tests._stream_helpers import DEFAULTS, FakeTurn, FixedDatetime, RecordingMic, emit_turns


def test_save_dir_auto_names_transcript_and_matching_wav(monkeypatch, tmp_path):
    # --save-dir buckets by date and shares one timestamp+slug stem across the .txt and
    # the .wav, so both land together under DIR/YYYY-MM-DD/.
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr(stream_save, "datetime", FixedDatetime)
    monkeypatch.setattr(stream_exec.client, "stream_audio", emit_turns(FakeTurn("hi there")))
    monkeypatch.setattr(stream_exec, "MicrophoneSource", RecordingMic)

    stream_exec.run_stream(
        dataclasses.replace(DEFAULTS, save_dir=tmp_path / "rec", name="My Meeting"),
        AppState(),
        json_mode=True,
    )

    bucket = tmp_path / "rec" / "2026-06-16"
    txt = bucket / "2026-06-16-143005-my-meeting.txt"
    wav = bucket / "2026-06-16-143005-my-meeting.wav"
    assert txt.read_text(encoding="utf-8") == "hi there\n"
    with wave.open(str(wav), "rb") as w:
        assert w.readframes(w.getnframes()) == RecordingMic.PCM
    # The sidecar lands beside them with the same stem.
    assert (bucket / "2026-06-16-143005-my-meeting.aai.json").is_file()


@pytest.mark.parametrize(
    "overrides",
    [
        {"save_dir": Path("rec"), "save_audio": Path("a.wav")},  # save-dir owns the audio name
        {"save_dir": Path("rec"), "save_transcript": Path("a.txt")},  # ...and the transcript
        {"save_dir": Path("rec"), "name": "X", "auto_name": True},  # both set the title
        {"name": "Standup"},  # --name without --save-dir is meaningless
        {"auto_name": True},  # --auto-name needs --save-dir
        {"no_save_audio": True},  # --no-save-audio needs --save-dir
    ],
)
def test_save_dir_rejects_incompatible_flags(overrides):
    with pytest.raises(UsageError):
        stream_exec.run_stream(
            dataclasses.replace(DEFAULTS, **overrides), AppState(), json_mode=False
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"save_transcript": Path("a.txt"), "show_code": True},
        {"save_dir": Path("rec"), "show_code": True},
    ],
)
def test_save_flags_reject_show_code(overrides):
    # The generated SDK code doesn't save to disk, so pairing a save flag with --show-code
    # is a usage error rather than a silently-dropped save.
    with pytest.raises(UsageError):
        stream_exec.run_stream(
            dataclasses.replace(DEFAULTS, **overrides), AppState(), json_mode=False
        )


def test_no_save_audio_writes_transcript_and_sidecar_but_no_wav(monkeypatch, tmp_path):
    # --save-dir --no-save-audio keeps the auto-named transcript + sidecar but writes no WAV.
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr(stream_save, "datetime", FixedDatetime)
    monkeypatch.setattr(stream_exec.client, "stream_audio", emit_turns(FakeTurn("hi there")))
    monkeypatch.setattr(stream_exec, "MicrophoneSource", RecordingMic)

    stream_exec.run_stream(
        dataclasses.replace(DEFAULTS, save_dir=tmp_path / "rec", name="Talk", no_save_audio=True),
        AppState(),
        json_mode=True,
    )

    bucket = tmp_path / "rec" / "2026-06-16"
    assert (bucket / "2026-06-16-143005-talk.txt").read_text(encoding="utf-8") == "hi there\n"
    record = json.loads((bucket / "2026-06-16-143005-talk.aai.json").read_text(encoding="utf-8"))
    assert record["audio"] == []
    assert list(bucket.glob("*.wav")) == []


def test_save_dir_auto_name_and_note_end_to_end(monkeypatch, tmp_path):
    # --save-dir --auto-name --llm: the files are renamed from the LLM-derived title, the
    # final answer lands as a .md note, and the sidecar records the title.
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr(stream_save, "datetime", FixedDatetime)
    monkeypatch.setattr(stream_exec.client, "stream_audio", emit_turns(FakeTurn("hi there")))
    monkeypatch.setattr(stream_exec, "MicrophoneSource", RecordingMic)

    from aai_cli.streaming import savedir

    def fake_run_chain(api_key, prompts, *, transcript_text, model, max_tokens):
        return "Cool Title" if prompts == [savedir.TITLE_PROMPT] else "the summary"

    monkeypatch.setattr("aai_cli.core.llm.run_chain", fake_run_chain)

    stream_exec.run_stream(
        dataclasses.replace(
            DEFAULTS, save_dir=tmp_path / "rec", auto_name=True, llm_prompt=["summarize"]
        ),
        AppState(),
        json_mode=True,
    )

    bucket = tmp_path / "rec" / "2026-06-16"
    stem = "2026-06-16-143005-cool-title"
    assert (bucket / f"{stem}.txt").read_text(encoding="utf-8") == "hi there\n"
    assert (bucket / f"{stem}.md").read_text(encoding="utf-8") == "the summary\n"
    with wave.open(str(bucket / f"{stem}.wav"), "rb") as w:
        assert w.readframes(w.getnframes()) == RecordingMic.PCM
    record = json.loads((bucket / f"{stem}.aai.json").read_text(encoding="utf-8"))
    assert record["title"] == "Cool Title"
    assert record["turns"] == 1
