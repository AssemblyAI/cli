from __future__ import annotations

import sys
import types
import wave
from pathlib import Path

import pytest

from aai_cli.core.errors import CLIError
from aai_cli.core.microphone import audio_missing_error
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
    def __init__(self, *, raise_on_write: BaseException | None = None) -> None:
        self.events: list[str] = []
        self.written: bytes = b""
        self.writes: list[bytes] = []
        self._raise_on_write = raise_on_write

    def start(self) -> None:
        self.events.append("start")

    def write(self, data: bytes) -> None:
        if self._raise_on_write is not None:
            raise self._raise_on_write
        chunk = bytes(data)
        self.written += chunk
        self.writes.append(chunk)
        self.events.append("write")

    def stop(self) -> None:
        self.events.append("stop")

    def abort(self) -> None:
        self.events.append("abort")

    def close(self) -> None:
        self.events.append("close")


def test_play_pcm_writes_to_started_stream_then_closes():
    stream = FakeStream()
    audio.play_pcm(b"\x01\x02", 16000, stream_factory=lambda rate: stream)
    assert stream.events == ["start", "write", "stop", "close"]
    assert stream.written == b"\x01\x02"


def test_play_pcm_writes_audio_in_bounded_chunks():
    # A buffer larger than one chunk is written in fixed-size pieces (so a Ctrl-C
    # can land between writes); the chunks reassemble to the original audio.
    stream = FakeStream()
    pcm = bytes(range(256)) * 40  # 10240 bytes > 2 * chunk
    audio.play_pcm(pcm, 24000, stream_factory=lambda rate: stream)
    assert [len(c) for c in stream.writes] == [4096, 4096, 2048]
    assert b"".join(stream.writes) == pcm


def test_play_pcm_aborts_and_propagates_on_ctrl_c():
    # Ctrl-C mid-playback must stop the device immediately (abort, not just stop)
    # and re-raise so the cancel reaches the CLI; the stream is still closed.
    stream = FakeStream(raise_on_write=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        audio.play_pcm(b"\x01\x02", 16000, stream_factory=lambda rate: stream)
    assert "abort" in stream.events
    assert "stop" not in stream.events  # aborted, never reached the draining stop()
    assert stream.events[-1] == "close"  # finally still closed it


def test_play_pcm_wraps_write_failure_in_cli_error():
    # A device error mid-stream (not from the factory) maps to the same clean
    # CLIError, and the stream is still closed via the finally block.
    stream = FakeStream(raise_on_write=RuntimeError("device fell over"))
    with pytest.raises(CLIError, match="Could not play audio") as excinfo:
        audio.play_pcm(b"\x01\x02", 16000, stream_factory=lambda rate: stream)
    assert excinfo.value.exit_code == 1
    assert stream.events[-1] == "close"


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


def test_pcm_player_opens_device_once_and_drains_on_exit():
    # Streaming playback: the device is opened lazily on the FIRST feed and reused
    # for every later chunk (one "start", not one per chunk), then drained (stop)
    # and closed on a normal exit. The factory is called exactly once.
    streams: list[FakeStream] = []

    def _factory(rate: int) -> FakeStream:
        streams.append(FakeStream())
        return streams[-1]

    with audio.PcmPlayer(stream_factory=_factory) as player:
        player.feed(b"\x01\x02", 24000)
        player.feed(b"\x03\x04", 24000)
    assert len(streams) == 1  # opened once, not re-opened per chunk
    stream = streams[0]
    assert stream.events == ["start", "write", "write", "stop", "close"]
    assert stream.written == b"\x01\x02\x03\x04"


def test_pcm_player_uses_the_first_feeds_sample_rate():
    captured: dict[str, int] = {}

    def _factory(rate: int) -> FakeStream:
        captured["rate"] = rate
        return FakeStream()

    with audio.PcmPlayer(stream_factory=_factory) as player:
        player.feed(b"\x01\x02", 16000)
    assert captured["rate"] == 16000  # the device is opened at the reported rate


def test_pcm_player_feed_writes_in_bounded_chunks():
    stream = FakeStream()
    pcm = bytes(range(256)) * 40  # 10240 bytes > 2 * chunk
    with audio.PcmPlayer(stream_factory=lambda rate: stream) as player:
        player.feed(pcm, 24000)
    assert [len(c) for c in stream.writes] == [4096, 4096, 2048]
    assert b"".join(stream.writes) == pcm


def test_pcm_player_aborts_and_propagates_when_a_chunk_fails():
    # An error while playing a chunk maps to a clean CLIError; the device is aborted
    # (not drained) and still closed.
    stream = FakeStream(raise_on_write=RuntimeError("device fell over"))
    with pytest.raises(CLIError, match="Could not play audio"):
        with audio.PcmPlayer(stream_factory=lambda rate: stream) as player:
            player.feed(b"\x01\x02", 24000)
    assert "abort" in stream.events
    assert "stop" not in stream.events
    assert stream.events[-1] == "close"


def test_pcm_player_without_any_feed_is_a_clean_noop():
    # No audio ever arrived (e.g. an empty synthesis): the device was never opened,
    # so a clean exit touches nothing and does not crash.
    opened: list[int] = []

    def _factory(rate: int) -> FakeStream:
        opened.append(rate)
        return FakeStream()

    with audio.PcmPlayer(stream_factory=_factory):
        pass
    assert opened == []  # the factory was never called


def test_pcm_player_propagates_a_body_error_when_never_opened():
    # An error before the first feed (device never opened) still propagates — the
    # context manager never swallows it.
    with pytest.raises(ValueError, match="boom"):
        with audio.PcmPlayer(stream_factory=lambda rate: FakeStream()):
            raise ValueError("boom")


def test_default_output_stream_opens_raw_int16_mono_stream(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    sentinel = object()

    def _raw_output_stream(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    fake_sd = types.ModuleType("sounddevice")
    monkeypatch.setattr(fake_sd, "RawOutputStream", _raw_output_stream, raising=False)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    stream: object = audio._default_output_stream(24000)
    assert stream is sentinel  # returns exactly what RawOutputStream produced
    assert captured == {"samplerate": 24000, "channels": 1, "dtype": "int16"}


def test_default_output_stream_missing_sounddevice_raises_audio_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # import -> ImportError
    with pytest.raises(CLIError):
        audio._default_output_stream(24000)


def test_silence_returns_zeroed_pcm_of_the_right_length():
    # 16-bit mono: 100 ms at 16 kHz = 1600 frames = 3200 zero bytes.
    pcm = audio.silence(16000, 0.1)
    assert pcm == b"\x00" * 3200
    # Empty duration -> no bytes.
    assert audio.silence(24000, 0.0) == b""
