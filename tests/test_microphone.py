import pytest

from assemblyai_cli.errors import CLIError
from assemblyai_cli.microphone import MicrophoneSource


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
        raise ImportError("No module named 'pyaudio'")

    mic = MicrophoneSource(sample_rate=16000, stream_factory=boom)
    with pytest.raises(CLIError) as exc:
        list(mic)
    assert exc.value.error_type == "mic_missing"
    assert exc.value.exit_code == 2
    assert "pyaudio" in exc.value.message.lower()


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
