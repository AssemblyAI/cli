import sys
import types

import pytest

from assemblyai_cli.agent.audio import Player, _default_output_stream
from assemblyai_cli.errors import CLIError


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
    p = Player(sample_rate=24000, stream_factory=lambda rate: fake)
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
    p = Player(sample_rate=24000, stream_factory=lambda rate: fake)
    # Do NOT start the worker; queue items directly so flush is deterministic.
    p.enqueue(b"stale-1")
    p.enqueue(b"stale-2")
    p.flush()
    assert p.pending() == 0


def test_player_worker_survives_write_error():
    class BoomStream(FakeStream):
        def write(self, data):
            raise RuntimeError("device gone")

    p = Player(sample_rate=24000, stream_factory=lambda rate: BoomStream())
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

    fake_sd = types.ModuleType("sounddevice")
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

    fake_sd = types.ModuleType("sounddevice")
    fake_sd.RawOutputStream = boom
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    with pytest.raises(CLIError) as exc:
        _default_output_stream(24000)
    assert exc.value.error_type == "audio_output_error"
    assert exc.value.exit_code == 1
