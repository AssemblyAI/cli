"""Shared builders for the `assembly dub` test modules.

The dub suite is split across test_dub_exec.py (pure helpers + validation),
test_dub_pipeline.py (the faked transcribe → translate → synthesize → mux
runs), and test_dub_command.py (argv parsing); the option defaults, transcript
fakes, and boundary recorders they share live here.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from aai_cli.app import mediafile
from aai_cli.commands.dub._exec import DubOptions
from aai_cli.core import client, config, llm
from aai_cli.tts import session
from aai_cli.tts.session import SpeakResult

# The CLI's flag defaults, as data. Tests override per-case with dataclasses.replace.
DEFAULTS = DubOptions(
    media="talk.mp4",
    language="de",
    source_language=None,
    transcript_id=None,
    voice=[],
    model=llm.DEFAULT_MODEL,
    max_tokens=llm.DEFAULT_MAX_TOKENS,
    out=None,
    video=False,
    download_sections=[],
    from_stdin=False,
    concurrency=4,
    force=False,
)

SAMPLE_RATE = 100  # tiny rate keeps the timeline byte math exact and readable


def utterance(start, speaker, text):
    return SimpleNamespace(start=start, end=None, speaker=speaker, text=text)


def fake_transcript(utterances, *, audio_duration=5):
    return SimpleNamespace(id="tr_dub", utterances=utterances, audio_duration=audio_duration)


def completion(text, finish_reason=None):
    """The slice of an OpenAI ChatCompletion that gateway.content_of and the
    dub truncation check read."""
    choice = SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def write_media(tmp_path: Path) -> Path:
    path = tmp_path / "talk.mp4"
    path.write_bytes(b"\x00fake-media")
    return path


def enable_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session, "is_available", lambda: True)


def patch_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "resolve_api_key", lambda **_: "test-key")


def record_transcribe(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Record the transcription request and return a two-speaker transcript."""
    calls: dict[str, object] = {}

    def _fake(api_key, audio, *, config):
        calls["api_key"] = api_key
        calls["audio"] = audio
        calls["config"] = config
        return fake_transcript([utterance(1000, "A", "Hello."), utterance(3000, "B", "World.")])

    monkeypatch.setattr(client, "transcribe", _fake)
    return calls


def record_translate(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Record each gateway call and reply with a marked 'DE:<text>' translation."""
    calls: list[dict[str, object]] = []

    def _fake(
        api_key,
        *,
        model,
        messages,
        max_tokens=llm.DEFAULT_MAX_TOKENS,
        transcript_id=None,
        extra=None,
    ):
        calls.append({"model": model, "messages": messages, "max_tokens": max_tokens})
        return completion(f"DE:{messages[-1]['content']}")

    monkeypatch.setattr(llm, "complete", _fake)
    return calls


def record_synthesize(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Record each TTS request; segment i comes back as 100 bytes of 0xA1+i."""
    calls: list[object] = []

    def _fake(api_key, cfg, *, connect=None, on_warning=None):
        calls.append(cfg)
        pcm = bytes([0xA0 + len(calls)]) * 100
        return SpeakResult(pcm=pcm, sample_rate=SAMPLE_RATE, audio_duration_seconds=0.5)

    monkeypatch.setattr(session, "synthesize", _fake)
    return calls


def record_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Resolve ffmpeg and record the invocation plus the WAV it was handed.

    The temp WAV is deleted right after the mux, so its contents are captured
    here, while the file still exists.
    """
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    recorded: dict[str, object] = {}

    def run(args: list[str]) -> subprocess.CompletedProcess[str]:
        recorded["args"] = args
        with wave.open(args[8], "rb") as wav:  # args[8] is the dub.wav input
            recorded["wav_params"] = wav.getparams()
            recorded["wav_frames"] = wav.readframes(wav.getnframes())
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mediafile, "run_ffmpeg", run)
    return recorded
