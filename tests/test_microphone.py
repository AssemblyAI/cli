import sys
import types

import pytest

from assemblyai_cli.errors import CLIError
from assemblyai_cli.microphone import MicrophoneSource, _default_mic_stream, _SoundDeviceMic


class _FakeRawStream:
    """Stand-in for sounddevice.RawInputStream (no hardware)."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = self.stopped = self.closed = False
        self._chunks = [(b"\x01\x02", False), (b"\x03\x04", False)]

    def start(self):
        self.started = True

    def read(self, frames):
        return self._chunks.pop(0)

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def test_yields_chunks_from_factory_with_rate_and_device():
    seen = {}

    def fake_factory(*, sample_rate, device):
        seen["rate"] = sample_rate
        seen["device"] = device
        return iter([b"aa", b"bb"])

    mic = MicrophoneSource(sample_rate=24000, device=3, stream_factory=fake_factory)
    assert list(mic) == [b"aa", b"bb"]
    assert seen == {"rate": 24000, "device": 3}


def test_missing_dependency_raises_mic_missing():
    def boom(*, sample_rate, device):
        raise ImportError("No module named 'sounddevice'")

    mic = MicrophoneSource(sample_rate=16000, stream_factory=boom)
    with pytest.raises(CLIError) as exc:
        list(mic)
    assert exc.value.error_type == "mic_missing"
    assert exc.value.exit_code == 2
    assert "sounddevice" in exc.value.message.lower()


def test_device_error_raises_mic_error_exit_1():
    def boom(*, sample_rate, device):
        raise OSError("Invalid device")

    mic = MicrophoneSource(sample_rate=16000, device=99, stream_factory=boom)
    with pytest.raises(CLIError) as exc:
        list(mic)
    assert exc.value.error_type == "mic_error"
    assert exc.value.exit_code == 1


def test_closes_closeable_stream_in_finally():
    closed = {"called": False}

    class CloseableStream:
        def __iter__(self):
            return iter([b"x"])

        def close(self):
            closed["called"] = True

    mic = MicrophoneSource(sample_rate=16000, stream_factory=lambda **_k: CloseableStream())
    assert list(mic) == [b"x"]
    assert closed["called"] is True  # close() invoked in the finally


def test_plain_iterator_without_close_is_fine():
    # A factory returning a bare iterator (no .close) must not error in teardown.
    mic = MicrophoneSource(sample_rate=16000, stream_factory=lambda **_k: iter([b"z"]))
    assert list(mic) == [b"z"]


def test_sounddevice_mic_yields_bytes_then_stops_and_closes():
    stream = _FakeRawStream()
    mic = _SoundDeviceMic(stream, blocksize=1024)
    it = iter(mic)
    assert next(it) == b"\x01\x02"
    assert next(it) == b"\x03\x04"
    mic.close()
    assert stream.stopped and stream.closed


def test_default_mic_stream_opens_started_sounddevice_stream(monkeypatch):
    created = {}

    def raw_input_stream(**kwargs):
        created.update(kwargs)
        return _FakeRawStream(**kwargs)

    fake_sd = types.ModuleType("sounddevice")
    fake_sd.RawInputStream = raw_input_stream
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    stream = _default_mic_stream(sample_rate=16000, device=2)
    assert isinstance(stream, _SoundDeviceMic)
    assert created["samplerate"] == 16000
    assert created["device"] == 2
    assert created["blocksize"] == 1600  # ~100 ms at 16 kHz
    assert next(iter(stream)) == b"\x01\x02"


def test_default_mic_stream_missing_sounddevice_raises_mic_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # import -> ImportError
    with pytest.raises(CLIError) as exc:
        _default_mic_stream(sample_rate=16000, device=None)
    assert exc.value.error_type == "mic_missing"
    assert exc.value.exit_code == 2
