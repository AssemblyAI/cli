from __future__ import annotations

import sys
import types
import wave
from pathlib import Path

import pytest

from aai_cli.errors import CLIError
from aai_cli.microphone import audio_missing_error
from aai_cli.tts import audio


def test_write_wav_produces_mono_16bit_wav(tmp_path: Path):
    # Two missing levels deep, so the write only succeeds if parents are created.
    out = tmp_path / "deep" / "nested" / "out.wav"
    pcm = b"\x01\x02\x03\x04"
    audio.write_wav(out, pcm, 24000)

    with wave.open(str(out), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 24000
        assert wav.readframes(wav.getnframes()) == pcm


def test_write_wav_into_existing_dir(tmp_path: Path):
    # The parent already exists; writing must not error (exist_ok must be set).
    out = tmp_path / "flat.wav"
    audio.write_wav(out, b"\x01\x02", 16000)
    assert out.exists()


class FakeStream:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.written: bytes = b""

    def start(self) -> None:
        self.events.append("start")

    def write(self, data: bytes) -> None:
        self.written += bytes(data)
        self.events.append("write")

    def stop(self) -> None:
        self.events.append("stop")

    def close(self) -> None:
        self.events.append("close")


def test_play_pcm_writes_to_started_stream_then_closes():
    stream = FakeStream()
    audio.play_pcm(b"\x01\x02", 16000, stream_factory=lambda rate: stream)
    assert stream.events == ["start", "write", "stop", "close"]
    assert stream.written == b"\x01\x02"


def test_play_pcm_wraps_device_failure_in_cli_error():
    def _boom(_rate: int):
        raise RuntimeError("no device")

    with pytest.raises(CLIError, match="Could not play audio") as excinfo:
        audio.play_pcm(b"\x01\x02", 16000, stream_factory=_boom)
    assert excinfo.value.exit_code == 1


def test_play_pcm_reraises_cli_error_unchanged(monkeypatch: pytest.MonkeyPatch):
    # A CLIError from the factory (e.g. audio_missing_error) is already user-facing,
    # so it must propagate as-is, NOT get re-wrapped in "Could not play audio".
    def _missing(_rate: int):
        raise audio_missing_error()

    with pytest.raises(CLIError) as excinfo:
        audio.play_pcm(b"\x01\x02", 16000, stream_factory=_missing)
    assert "Could not play audio" not in excinfo.value.message


def test_default_output_stream_opens_raw_int16_mono_stream(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def _raw_output_stream(**kwargs: object) -> str:
        captured.update(kwargs)
        return "stream-sentinel"

    fake_sd = types.ModuleType("sounddevice")
    monkeypatch.setattr(fake_sd, "RawOutputStream", _raw_output_stream, raising=False)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    stream = audio._default_output_stream(24000)
    assert stream == "stream-sentinel"
    assert captured == {"samplerate": 24000, "channels": 1, "dtype": "int16"}


def test_default_output_stream_missing_sounddevice_raises_audio_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # import -> ImportError
    with pytest.raises(CLIError):
        audio._default_output_stream(24000)
