import sys
import types
from typing import Any

import pytest

from aai_cli.errors import CLIError


class FakeStream:
    def __init__(self):
        self.writes = []
        self.stopped = False
        self.closed = False
        self.active = True

    def write(self, data):
        self.writes.append(data)

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


from aai_cli.agent.audio import DuplexAudio  # noqa: E402


def test_duplex_opens_at_device_rate_and_closes():
    seen = {}
    fake = FakeStream()

    def factory(*, rate, blocksize, callback, device):
        seen["rate"] = rate
        seen["device"] = device
        seen["blocksize"] = blocksize
        return fake

    d = DuplexAudio(device=3, device_rate=48000, stream_factory=factory)
    d.player.start()
    assert seen["rate"] == 48000 and seen["device"] == 3  # one stream at device rate
    assert seen["blocksize"] == 4800  # ~100 ms at 48 kHz (device_rate // 10)
    d.close()
    assert fake.stopped and fake.closed


def test_duplex_restart_after_close_reopens_stream():
    calls = {"n": 0}

    def factory(**_k):
        calls["n"] += 1
        return FakeStream()

    d = DuplexAudio(device_rate=16000, stream_factory=factory)
    d.start()
    assert calls["n"] == 1
    d.close()
    d.start()  # close() cleared the started flag, so this reopens the stream
    assert calls["n"] == 2


def test_duplex_callback_captures_input_and_zero_fills_idle_output():
    cb = {}

    def factory(*, rate, blocksize, callback, device):
        cb["fn"] = callback
        return FakeStream()

    d = DuplexAudio(target_rate=24000, device_rate=48000, stream_factory=factory)
    d.player.start()
    indata = b"\x11\x11" * 4800  # 100 ms @ 48 kHz
    outdata = bytearray(b"\xff" * 1920)  # nothing queued -> should be zeroed
    cb["fn"](indata, outdata, 4800, None, None)
    assert bytes(outdata) == b"\x00" * 1920  # idle output is silence, not garbage

    chunk = next(iter(d.mic))
    assert 0 < len(chunk) < len(indata)  # captured input resampled 48k -> 24k
    d.close()


def test_duplex_playback_resamples_and_drains_into_output():
    cb = {}

    def factory(*, rate, blocksize, callback, device):
        cb["fn"] = callback
        return FakeStream()

    d = DuplexAudio(target_rate=24000, device_rate=48000, stream_factory=factory)
    d.player.start()
    d.player.enqueue(b"\x01\x02" * 240)  # 24 kHz audio -> upsampled to 48 kHz in the buffer
    assert d.player.pending() > 240  # more samples buffered after upsample
    outdata = bytearray(200)
    cb["fn"](b"\x00\x00" * 10, outdata, 10, None, None)
    assert bytes(outdata) != b"\x00" * 200  # buffered audio was played out
    d.close()


def test_duplex_callback_partial_buffer_zero_fills_exact_remainder():
    cb = {}

    def factory(*, rate, blocksize, callback, device):
        cb["fn"] = callback
        return FakeStream()

    # device == target so playback bytes pass through unresampled and are easy to count.
    d = DuplexAudio(target_rate=16000, device_rate=16000, stream_factory=factory)
    d.player.start()
    d.player.enqueue(b"\x01\x02" * 5)  # 10 bytes buffered
    outdata = bytearray(20)  # request 20 bytes -> 10 real + 10 zero-filled
    cb["fn"](b"\x00\x00" * 5, outdata, 5, None, None)
    # The shortfall is filled with exactly `need - len(take)` zero bytes: the buffer
    # plays out first, then silence, and the output stays exactly `need` bytes long.
    assert len(outdata) == 20
    assert bytes(outdata) == b"\x01\x02" * 5 + b"\x00" * 10
    d.close()


def test_duplex_mic_ends_after_close():
    d = DuplexAudio(target_rate=16000, device_rate=16000, stream_factory=lambda **k: FakeStream())
    d.player.start()
    d.close()
    assert list(d.mic) == []  # capture loop returns on the close sentinel


def test_duplex_start_is_idempotent():
    calls = {"n": 0}

    def factory(**k):
        calls["n"] += 1
        return FakeStream()

    d = DuplexAudio(device_rate=16000, stream_factory=factory)
    d.start()
    d.start()  # second start must be a no-op
    assert calls["n"] == 1


def test_duplex_player_facade_flush_and_close():
    fake = FakeStream()
    d = DuplexAudio(target_rate=16000, device_rate=16000, stream_factory=lambda **k: fake)
    d.player.start()
    d.player.enqueue(b"\x01\x02" * 8)  # 16 bytes, no resample (device == target)
    assert d.player.pending() == 8  # pending() reports samples = bytes // 2
    d.player.flush()
    assert d.player.pending() == 0
    d.player.close()
    assert fake.stopped and fake.closed


from aai_cli.agent.audio import NullPlayer, _default_duplex_stream  # noqa: E402


def test_null_player_is_a_noop_player():
    p = NullPlayer()
    p.start()
    p.enqueue(b"ignored")
    p.flush()
    assert p.pending() == 0
    p.close()  # none of these raise or open a device


def test_default_duplex_stream_opens_started_rawstream(monkeypatch) -> None:
    created = {}

    class FakeRaw:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.started = False

        def start(self):
            self.started = True

    fake_sd: Any = types.ModuleType("sounddevice")
    fake_sd.RawStream = lambda **kw: FakeRaw(**kw)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    stream = _default_duplex_stream(rate=48000, blocksize=4800, callback=lambda *a: None, device=2)
    assert stream.started
    assert created["samplerate"] == 48000
    assert created["device"] == 2
    assert created["channels"] == 1


def test_default_duplex_stream_missing_sounddevice_raises_mic_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # import -> ImportError
    with pytest.raises(CLIError) as exc:
        _default_duplex_stream(rate=24000, blocksize=2400, callback=lambda *a: None, device=None)
    assert exc.value.error_type == "mic_missing"


def test_default_duplex_stream_open_failure_raises_audio_output_error(monkeypatch) -> None:
    def boom(**kw):
        raise OSError("device busy")

    fake_sd: Any = types.ModuleType("sounddevice")
    fake_sd.RawStream = boom
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    with pytest.raises(CLIError) as exc:
        _default_duplex_stream(rate=24000, blocksize=2400, callback=lambda *a: None, device=None)
    assert exc.value.error_type == "audio_output_error"
    assert exc.value.exit_code == 1


def test_capture_frames_raises_when_stream_dies_without_close():
    # A device that vanishes mid-session stops the PortAudio callback without a
    # close(): no sentinel ever arrives, so the consumer must time out and surface
    # a clean error instead of blocking the capture thread forever.
    fake = FakeStream()
    fake.active = False
    d = DuplexAudio(device_rate=24000, stream_factory=lambda **kwargs: fake, poll_timeout=0.01)
    d.player.start()
    with pytest.raises(CLIError) as excinfo:
        next(d.capture_frames())
    assert excinfo.value.error_type == "audio_input_error"
    assert excinfo.value.exit_code == 1


def test_poll_timeout_defaults_to_one_second():
    # The default must stay long enough to never fire during normal silence but
    # short enough that a dead device surfaces quickly.
    d = DuplexAudio(device_rate=24000, stream_factory=lambda **kwargs: FakeStream())
    assert d._poll_timeout == 1.0


def test_capture_frames_tolerates_streams_without_active_flag():
    # Fake/file-driven streams may not expose `active`; an empty queue then means
    # "keep waiting", not "device died" — only the sentinel ends capture.
    import threading

    class MinimalStream:
        pass

    d = DuplexAudio(
        device_rate=24000, stream_factory=lambda **kwargs: MinimalStream(), poll_timeout=0.01
    )
    d.player.start()
    threading.Timer(0.05, d._in.put, args=(None,)).start()
    assert list(d.capture_frames()) == []


def test_capture_frames_keeps_waiting_while_stream_is_alive():
    # An empty queue with a healthy stream is just silence: the poll loop keeps
    # waiting (no error) and only raises once the stream actually reports dead.
    class FlippingStream:
        def __init__(self):
            self.polls = 0

        @property
        def active(self):
            self.polls += 1
            return self.polls == 1  # alive on the first poll, dead afterwards

    fake = FlippingStream()
    d = DuplexAudio(device_rate=24000, stream_factory=lambda **kwargs: fake, poll_timeout=0.01)
    d.player.start()
    with pytest.raises(CLIError):
        next(d.capture_frames())
    assert fake.polls >= 2  # first timeout passed the alive check and kept waiting
