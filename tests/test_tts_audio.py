from __future__ import annotations

import wave
from pathlib import Path

import pytest

from aai_cli.errors import CLIError
from aai_cli.tts import audio


def test_write_wav_produces_mono_16bit_wav(tmp_path: Path):
    out = tmp_path / "nested" / "out.wav"  # parent dirs created on demand
    pcm = b"\x01\x02\x03\x04"
    audio.write_wav(out, pcm, 24000)

    with wave.open(str(out), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 24000
        assert wav.readframes(wav.getnframes()) == pcm


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

    with pytest.raises(CLIError, match="Could not play audio"):
        audio.play_pcm(b"\x01\x02", 16000, stream_factory=_boom)
