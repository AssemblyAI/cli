import sys
import types

import pytest

from assemblyai_cli.errors import CLIError
from assemblyai_cli.microphone import (
    _FALLBACK_RATE,
    MicrophoneSource,
    _default_mic_stream,
    _device_default_rate,
    _SoundDeviceMic,
)


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


def test_yields_chunks_at_capture_rate():
    seen = {}

    def fake_factory(*, sample_rate, device):
        seen["rate"] = sample_rate
        seen["device"] = device
        return iter([b"aa", b"bb"])

    mic = MicrophoneSource(capture_rate=24000, device=3, stream_factory=fake_factory)
    assert mic.sample_rate == 24000  # no target -> reports the capture rate
    assert list(mic) == [b"aa", b"bb"]
    assert seen == {"rate": 24000, "device": 3}  # opened at the capture rate


def test_resamples_capture_rate_to_target():
    frames48 = b"\x00\x00" * 960  # 20 ms of silence at 48 kHz

    def fake_factory(*, sample_rate, device):
        assert sample_rate == 48000  # device opened at its native rate
        return iter([frames48])

    mic = MicrophoneSource(target_rate=24000, capture_rate=48000, stream_factory=fake_factory)
    assert mic.sample_rate == 24000  # callers see the target rate
    out = b"".join(mic)
    assert 0 < len(out) < len(frames48)  # downsampled 48k -> 24k


def test_no_resample_when_target_matches_capture():
    def fake_factory(*, sample_rate, device):
        return iter([b"\x01\x02\x03\x04"])

    mic = MicrophoneSource(target_rate=16000, capture_rate=16000, stream_factory=fake_factory)
    assert mic.sample_rate == 16000
    assert list(mic) == [b"\x01\x02\x03\x04"]  # untouched when rates already match


def test_missing_dependency_raises_mic_missing():
    def boom(*, sample_rate, device):
        raise ImportError("No module named 'sounddevice'")

    mic = MicrophoneSource(capture_rate=16000, stream_factory=boom)
    with pytest.raises(CLIError) as exc:
        list(mic)
    assert exc.value.error_type == "mic_missing"
    assert exc.value.exit_code == 2
    assert "sounddevice" in exc.value.message.lower()


def test_device_error_raises_mic_error_exit_1():
    def boom(*, sample_rate, device):
        raise OSError("Invalid device")

    mic = MicrophoneSource(capture_rate=16000, device=99, stream_factory=boom)
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

    mic = MicrophoneSource(capture_rate=16000, stream_factory=lambda **_k: CloseableStream())
    assert list(mic) == [b"x"]
    assert closed["called"] is True  # close() invoked in the finally


def test_plain_iterator_without_close_is_fine():
    mic = MicrophoneSource(capture_rate=16000, stream_factory=lambda **_k: iter([b"z"]))
    assert list(mic) == [b"z"]


def test_rate_query_resolves_capture_rate_when_not_given():
    seen = {}

    def fake_factory(*, sample_rate, device):
        seen["rate"] = sample_rate
        return iter([b"q"])

    mic = MicrophoneSource(
        device=7, stream_factory=fake_factory, rate_query=lambda _device: 32000
    )
    assert mic.sample_rate == 32000
    assert list(mic) == [b"q"]
    assert seen["rate"] == 32000


def test_device_default_rate_reads_device(monkeypatch):
    fake_sd = types.ModuleType("sounddevice")
    fake_sd.query_devices = lambda device, kind: {"default_samplerate": 44100.0}
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    assert _device_default_rate(2) == 44100


def test_device_default_rate_falls_back_on_query_error(monkeypatch):
    fake_sd = types.ModuleType("sounddevice")

    def boom(*a, **k):
        raise RuntimeError("no input device")

    fake_sd.query_devices = boom
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    assert _device_default_rate(None) == _FALLBACK_RATE


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
