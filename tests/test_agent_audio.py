import pytest

from assemblyai_cli.agent.audio import MicCapture, Player
from assemblyai_cli.errors import CLIError


class FakeStream:
    def __init__(self):
        self.writes = []
        self.stopped = False
        self.closed = False

    def write(self, data):
        self.writes.append(data)

    def stop_stream(self):
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


def test_miccapture_yields_chunks_from_factory():
    def fake_factory(*, sample_rate, device):
        assert sample_rate == 24000
        return iter([b"aa", b"bb"])

    mic = MicCapture(sample_rate=24000, device=None, stream_factory=fake_factory)
    assert list(mic) == [b"aa", b"bb"]


def test_miccapture_missing_dependency_raises_cli_error():
    def boom(*, sample_rate, device):
        raise ImportError("no pyaudio")

    mic = MicCapture(sample_rate=24000, device=None, stream_factory=boom)
    with pytest.raises(CLIError) as excinfo:
        list(mic)
    assert excinfo.value.exit_code == 2
    assert "pyaudio" in excinfo.value.message.lower()


def test_player_worker_survives_write_error():
    class BoomStream(FakeStream):
        def write(self, data):
            raise RuntimeError("device gone")

    p = Player(sample_rate=24000, stream_factory=lambda rate: BoomStream())
    p.start()
    p.enqueue(b"\x01\x02")
    p.close()  # must return (join has a timeout); thread must not be alive
    assert p._thread is not None and not p._thread.is_alive()


def test_miccapture_closes_closeable_stream():
    closed = {"called": False}

    class CloseableStream:
        def __iter__(self):
            return iter([b"x"])

        def close(self):
            closed["called"] = True

    mic = MicCapture(stream_factory=lambda *, sample_rate, device: CloseableStream())
    assert list(mic) == [b"x"]
    assert closed["called"] is True  # stream.close() invoked in the finally


def test_miccapture_device_error_raises_cli_error_exit_1():
    def boom(*, sample_rate, device):
        raise RuntimeError("bad device")

    mic = MicCapture(sample_rate=24000, device=None, stream_factory=boom)
    with pytest.raises(CLIError) as excinfo:
        list(mic)
    assert excinfo.value.exit_code == 1
