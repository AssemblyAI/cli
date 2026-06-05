import sys
import types
from typing import Any

import pytest

from aai_cli.agent.audio import Player, _default_output_stream
from aai_cli.errors import CLIError


class FakeStream:
    def __init__(self):
        self.writes = []
        self.stopped = False
        self.closed = False

    def write(self, data):
        self.writes.append(data)

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def test_player_writes_enqueued_audio():
    fake = FakeStream()
    p = Player(sample_rate=24000, output_rate=24000, stream_factory=lambda rate: fake)
    p.start()
    p.enqueue(b"\x01\x02")
    p.enqueue(b"\x03\x04")
    p.close()  # drains the queue, then tears down
    assert b"\x01\x02" in fake.writes
    assert b"\x03\x04" in fake.writes
    assert fake.stopped
    assert fake.closed


def test_player_flush_discards_pending_audio():
    fake = FakeStream()
    p = Player(sample_rate=24000, output_rate=24000, stream_factory=lambda rate: fake)
    # Do NOT start the worker; queue items directly so flush is deterministic.
    p.enqueue(b"stale-1")
    p.enqueue(b"stale-2")
    p.flush()
    assert p.pending() == 0


def test_player_worker_survives_write_error():
    class BoomStream(FakeStream):
        def write(self, data):
            raise RuntimeError("device gone")

    p = Player(sample_rate=24000, output_rate=24000, stream_factory=lambda rate: BoomStream())
    p.start()
    p.enqueue(b"\x01\x02")
    p.close()  # must return (join has a timeout); thread must not be alive
    assert p._thread is not None and not p._thread.is_alive()


def test_default_output_stream_opens_started_sounddevice_stream(monkeypatch):
    created = {}

    class FakeOut:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.started = False

        def start(self):
            self.started = True

    fake_sd: Any = types.ModuleType("sounddevice")
    fake_sd.RawOutputStream = lambda **kw: FakeOut(**kw)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    stream = _default_output_stream(24000)
    assert stream.started
    assert created["samplerate"] == 24000
    assert created["channels"] == 1


def test_default_output_stream_missing_sounddevice_raises_mic_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # import -> ImportError
    with pytest.raises(CLIError) as exc:
        _default_output_stream(24000)
    assert exc.value.error_type == "mic_missing"


def test_default_output_stream_open_failure_raises_audio_output_error(monkeypatch):
    def boom(**kw):
        raise OSError("no output device")

    fake_sd: Any = types.ModuleType("sounddevice")
    fake_sd.RawOutputStream = boom
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    with pytest.raises(CLIError) as exc:
        _default_output_stream(24000)
    assert exc.value.error_type == "audio_output_error"
    assert exc.value.exit_code == 1


def test_player_opens_stream_at_device_rate():
    seen = {}

    def factory(rate):
        seen["rate"] = rate
        return FakeStream()

    p = Player(sample_rate=24000, output_rate=48000, stream_factory=factory)
    p.start()
    p.close()
    assert seen["rate"] == 48000  # speaker opened at its native rate, not forced to 24 kHz


def test_player_resamples_source_to_device_rate():
    # Agent audio is 24 kHz; when the speaker opens at 48 kHz the worker upsamples.
    fake = FakeStream()
    p = Player(sample_rate=24000, output_rate=48000, stream_factory=lambda rate: fake)
    p.start()
    p.enqueue(b"\x00\x00" * 240)  # 10 ms of 24 kHz silence
    p.close()
    written = b"".join(fake.writes)
    assert len(written) > 240 * 2  # upsampled to ~48 kHz -> more bytes than the 24 kHz input


from aai_cli.agent.audio import DuplexAudio  # noqa: E402


def test_duplex_opens_at_device_rate_and_closes():
    seen = {}
    fake = FakeStream()

    def factory(*, rate, blocksize, callback, device):
        seen["rate"] = rate
        seen["device"] = device
        return fake

    d = DuplexAudio(device=3, device_rate=48000, stream_factory=factory)
    d.player.start()
    assert seen["rate"] == 48000 and seen["device"] == 3  # one stream at device rate
    d.close()
    assert fake.stopped and fake.closed


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


def test_duplex_mic_ends_after_close():
    d = DuplexAudio(target_rate=16000, device_rate=16000, stream_factory=lambda **k: FakeStream())
    d.player.start()
    d.close()
    assert list(d.mic) == []  # capture loop returns on the close sentinel
