import signal
import sys
import types
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from aai_cli.core import microphone
from aai_cli.core.errors import CLIError
from aai_cli.core.microphone import (
    _FALLBACK_RATE,
    MicrophoneSource,
    _default_mic_stream,
    _device_default_rate,
    _ignore_interrupt_during_shutdown,
    _install_shutdown_interrupt_guard,
    _max_input_channels,
    _RawInputStream,
    _SoundDeviceMic,
    import_sounddevice,
    resample_pcm16,
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


class _FakeSoundDevice(types.ModuleType):
    """A typed `_SoundDeviceModule` double: scripted device info + a RawInputStream factory.

    Subclasses `ModuleType` so it can be slotted into `sys.modules` via `monkeypatch.setitem`,
    and conforms to the protocol so it needs no escape hatches at the call sites that pass it
    to the real `_max_input_channels` / `_default_mic_stream` code under test.
    """

    def __init__(
        self,
        info: Mapping[str, object],
        raw_input_stream: Callable[..., _RawInputStream] = _FakeRawStream,
    ) -> None:
        super().__init__("sounddevice")
        self._info = info
        self.RawInputStream = raw_input_stream

    def query_devices(
        self, device: int | None = None, kind: str | None = None
    ) -> Mapping[str, object]:
        return self._info


def test_audio_missing_error_has_reinstall_suggestion():
    from aai_cli.core.microphone import audio_missing_error

    err = audio_missing_error()
    assert "sounddevice" in err.message
    assert err.suggestion is not None
    assert "pip install" in err.suggestion


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
    assert "microphone device 99" in exc.value.message  # names the explicit device
    assert "Invalid device" in exc.value.message  # keeps the underlying cause
    assert exc.value.suggestion is not None
    assert "--device" in exc.value.suggestion


def test_default_device_error_names_default_microphone():
    # device=None must read as "the default microphone", not the raw "device None",
    # and carry an actionable suggestion (permissions / pick another device).
    def boom(*, sample_rate, device):
        raise OSError("Error querying device -1")

    mic = MicrophoneSource(capture_rate=16000, stream_factory=boom)
    with pytest.raises(CLIError) as exc:
        list(mic)
    assert "the default microphone" in exc.value.message
    assert "device None" not in exc.value.message
    assert exc.value.suggestion is not None
    assert "permissions" in exc.value.suggestion
    assert "python -m sounddevice" in exc.value.suggestion


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


def test_on_open_fires_once_after_device_opens():
    events = []
    mic = MicrophoneSource(
        capture_rate=16000,
        stream_factory=lambda **_k: iter([b"x", b"y"]),
        on_open=lambda: events.append("open"),
    )
    assert events == []  # not signaled until iteration opens the device
    assert list(mic) == [b"x", b"y"]
    assert events == ["open"]  # fired exactly once, when the mic became live


def test_on_open_not_called_when_device_fails_to_open():
    events = []

    def boom(**_k):
        raise OSError("no input device")

    mic = MicrophoneSource(
        capture_rate=16000, stream_factory=boom, on_open=lambda: events.append("open")
    )
    with pytest.raises(CLIError):
        list(mic)
    assert events == []  # never claimed "listening" because recording never started


def test_rate_query_resolves_capture_rate_when_not_given():
    seen = {}

    def fake_factory(*, sample_rate, device):
        seen["rate"] = sample_rate
        return iter([b"q"])

    mic = MicrophoneSource(device=7, stream_factory=fake_factory, rate_query=lambda _device: 32000)
    assert mic.sample_rate == 32000
    assert list(mic) == [b"q"]
    assert seen["rate"] == 32000


def test_device_default_rate_reads_device(monkeypatch) -> None:
    fake_sd: Any = types.ModuleType("sounddevice")
    fake_sd.query_devices = lambda device, kind: {"default_samplerate": 44100.0}
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    assert _device_default_rate(2) == 44100


def test_resample_pcm16_uses_16bit_mono_params():
    # resample_pcm16 must treat the buffer as 16-bit (2-byte) mono (1-channel) PCM.
    # Compare against audioop driven with those exact params; a mutated width/channel
    # count yields different bytes (or rejects the frame count), killing the mutant.
    # (`microphone.audioop` is the module's own import, so both sides agree.)
    chunk = bytes(range(256))  # 128 little-endian 16-bit mono samples (a ramp)
    expected, _ = microphone.audioop.ratecv(chunk, 2, 1, 48000, 24000, None)
    out, _ = resample_pcm16(chunk, None, src_rate=48000, dst_rate=24000)
    assert out == expected
    assert out != chunk  # 48k -> 24k actually changes the data


def test_device_default_rate_falls_back_on_query_error(monkeypatch) -> None:
    fake_sd: Any = types.ModuleType("sounddevice")

    def boom(*a, **k):
        raise RuntimeError("no input device")

    fake_sd.query_devices = boom
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    assert _device_default_rate(None) == _FALLBACK_RATE


def test_device_default_rate_falls_back_on_non_numeric_rate(monkeypatch) -> None:
    fake_sd: Any = types.ModuleType("sounddevice")
    fake_sd.query_devices = lambda device, kind: {"default_samplerate": object()}
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    assert _device_default_rate(None) == _FALLBACK_RATE


def test_device_default_rate_keeps_smallest_positive_rate(monkeypatch) -> None:
    # A reported rate of exactly 1 is positive and must be kept as-is; only a
    # non-positive (<= 0) rate falls back. Pins the `rate > 0` boundary so it can't
    # drift to `rate > 1` and silently discard a legitimate 1 Hz reading.
    fake_sd: Any = types.ModuleType("sounddevice")
    fake_sd.query_devices = lambda device, kind: {"default_samplerate": 1.0}
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    assert _device_default_rate(None) == 1


def test_sounddevice_mic_yields_bytes_then_stops_and_closes():
    stream = _FakeRawStream()
    mic = _SoundDeviceMic(stream, blocksize=1024)
    it = iter(mic)
    assert next(it) == b"\x01\x02"
    assert next(it) == b"\x03\x04"
    mic.close()
    assert stream.stopped and stream.closed


def test_default_mic_stream_opens_started_sounddevice_stream(monkeypatch) -> None:
    created = {}

    def raw_input_stream(**kwargs):
        created.update(kwargs)
        return _FakeRawStream(**kwargs)

    fake_sd: Any = types.ModuleType("sounddevice")
    fake_sd.RawInputStream = raw_input_stream
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    stream = _default_mic_stream(sample_rate=16000, device=2)
    assert isinstance(stream, _SoundDeviceMic)
    assert created["samplerate"] == 16000
    assert created["device"] == 2
    assert created["blocksize"] == 1600  # ~100 ms at 16 kHz
    assert created["channels"] == 1  # mono capture
    assert created["dtype"] == "int16"  # PCM16
    assert next(iter(stream)) == b"\x01\x02"


def test_default_mic_stream_floors_blocksize_at_one(monkeypatch) -> None:
    # A pathologically small sample rate makes `sample_rate // 10` round to 0; the
    # max(1, ...) floor must still open with one frame per read, never 0 (which would
    # make sounddevice read nothing). Pins that floor at 1.
    created: dict[str, Any] = {}

    def raw_input_stream(**kwargs):
        created.update(kwargs)
        return _FakeRawStream(**kwargs)

    fake_sd: Any = types.ModuleType("sounddevice")
    fake_sd.RawInputStream = raw_input_stream
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    _default_mic_stream(sample_rate=5, device=None)  # 5 // 10 == 0
    assert created["blocksize"] == 1


def test_default_mic_stream_missing_sounddevice_raises_mic_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # import -> ImportError
    with pytest.raises(CLIError) as exc:
        _default_mic_stream(sample_rate=16000, device=None)
    assert exc.value.error_type == "mic_missing"
    assert exc.value.exit_code == 2


class _FakeStereoStream(_FakeRawStream):
    """A 2-channel input stream: one interleaved stereo frame (L=256, R=768)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # int16 LE: L=256 (b"\x00\x01"), R=768 (b"\x00\x03"), interleaved one frame.
        self._chunks = [(b"\x00\x01\x00\x03", False)]


def test_sounddevice_mic_downmixes_stereo_to_mono():
    # channels=2 averages L/R per frame: (256 + 768) / 2 == 512 (b"\x00\x02").
    mic = _SoundDeviceMic(_FakeStereoStream(), blocksize=1, channels=2)
    assert next(iter(mic)) == b"\x00\x02"


def _fake_sd_rejecting_mono(max_input_channels: int, opened: list[int]) -> _FakeSoundDevice:
    """A sounddevice whose mono open fails with -9998; query reports ``max_input_channels``."""

    def raw_input_stream(*, channels: int, **kwargs: object) -> _RawInputStream:
        opened.append(channels)
        if channels == 1:
            raise OSError("Error opening RawInputStream: Invalid number of channels [-9998]")
        return _FakeStereoStream(channels=channels, **kwargs)

    return _FakeSoundDevice({"max_input_channels": max_input_channels}, raw_input_stream)


def test_max_input_channels_defaults_to_zero_when_absent_or_non_int():
    # A device dict missing the key, or carrying a non-int value, must read as 0 channels (so
    # the caller raises the actionable no-input error) rather than a truthy bogus count.
    assert _max_input_channels(_FakeSoundDevice({}), None) == 0  # key absent -> 0, not get()'s
    assert _max_input_channels(_FakeSoundDevice({"max_input_channels": None}), None) == 0  # non-int
    assert _max_input_channels(_FakeSoundDevice({"max_input_channels": 2}), None) == 2  # int passes


def test_default_mic_stream_falls_back_to_stereo_downmix(monkeypatch):
    # A multichannel-only input (mono rejected, but >=2 channels available) is reopened at
    # stereo and downmixed to mono — so voice works on devices that won't open as mono.
    opened: list[int] = []
    monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd_rejecting_mono(2, opened))
    stream = _default_mic_stream(sample_rate=16000, device=None)
    assert opened == [1, 2]  # tried mono, then reopened stereo
    assert next(iter(stream)) == b"\x00\x02"  # yields downmixed mono


def test_default_mic_stream_zero_input_channels_raises_permission_error(monkeypatch):
    # 0 input channels can't be salvaged (no mic permission / wrong default device): raise an
    # actionable error pointing at the macOS Microphone privacy setting, not the cryptic code.
    opened: list[int] = []
    monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd_rejecting_mono(0, opened))
    with pytest.raises(CLIError) as exc:
        _default_mic_stream(sample_rate=16000, device=None)
    assert opened == [1]  # only the mono attempt; no pointless stereo retry
    assert exc.value.error_type == "mic_error"
    assert exc.value.exit_code == 1
    assert "no input channels" in exc.value.message.lower()
    assert exc.value.suggestion is not None
    assert "Microphone" in exc.value.suggestion


def test_default_mic_stream_single_channel_failure_reraises_original(monkeypatch):
    # A genuine 1-channel device should accept mono; if it still failed, the channel fallback
    # can't help, so surface the real PortAudio error rather than masking it.
    opened: list[int] = []
    monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd_rejecting_mono(1, opened))
    with pytest.raises(OSError, match="Invalid number of channels"):
        _default_mic_stream(sample_rate=16000, device=None)
    assert opened == [1]  # no stereo retry on a 1-channel device


def test_microphone_source_passes_through_factory_clierror():
    # An actionable CLIError from the factory (e.g. the zero-channel case) must propagate
    # intact, not get re-wrapped into the generic "Could not open" message.
    err = CLIError("no input channels", error_type="mic_error", exit_code=1, suggestion="grant it")

    def boom(**_kwargs):
        raise err

    mic = MicrophoneSource(capture_rate=16000, stream_factory=boom)
    with pytest.raises(CLIError) as exc:
        list(mic)
    assert exc.value is err  # passed through unchanged
    assert exc.value.suggestion == "grant it"


def test_ignore_interrupt_during_shutdown_sets_sig_ign():
    # The guard drops a second Ctrl-C during teardown so it can't raise inside
    # sounddevice's atexit PortAudio terminate. Save/restore the global disposition.
    before = signal.getsignal(signal.SIGINT)
    try:
        _ignore_interrupt_during_shutdown()
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, before)


def test_install_shutdown_interrupt_guard_registers_once(monkeypatch):
    registered = []
    monkeypatch.setattr(microphone, "_shutdown_interrupt_guard_installed", False)
    monkeypatch.setattr(microphone.atexit, "register", lambda fn: registered.append(fn))

    _install_shutdown_interrupt_guard()
    _install_shutdown_interrupt_guard()  # idempotent: the flag short-circuits the second call

    assert registered == [_ignore_interrupt_during_shutdown]


def test_import_sounddevice_installs_shutdown_guard(monkeypatch):
    registered = []
    monkeypatch.setattr(microphone, "_shutdown_interrupt_guard_installed", False)
    monkeypatch.setattr(microphone.atexit, "register", lambda fn: registered.append(fn))
    monkeypatch.setitem(sys.modules, "sounddevice", types.ModuleType("sounddevice"))

    import_sounddevice()

    assert registered == [_ignore_interrupt_during_shutdown]


def test_import_sounddevice_missing_does_not_register_guard(monkeypatch):
    # A broken install raises before the guard is reached, so nothing is registered.
    registered = []
    monkeypatch.setattr(microphone, "_shutdown_interrupt_guard_installed", False)
    monkeypatch.setattr(microphone.atexit, "register", lambda fn: registered.append(fn))
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # import -> ImportError

    with pytest.raises(CLIError) as exc:
        import_sounddevice()

    assert exc.value.error_type == "mic_missing"
    assert registered == []
