import io
import wave

import pytest

from assemblyai_cli.errors import CLIError
from assemblyai_cli.streaming import sources
from assemblyai_cli.streaming.sources import FileSource


def _write_wav(path, *, seconds=0.5, rate=16000):
    frames = int(rate * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x01" * frames)  # 2 bytes/frame, mono 16-bit


def test_filesource_streams_wav_chunks(tmp_path):
    p = tmp_path / "clip.wav"
    _write_wav(p, seconds=0.55)  # 0.55s @16k mono 16-bit = 17600 bytes
    src = FileSource(str(p), sleep=lambda _s: None)
    chunks = list(src)
    assert sum(len(c) for c in chunks) == 17600
    assert all(len(c) <= sources.CHUNK_BYTES for c in chunks)
    assert len(chunks) == 6  # 5 full 3200-byte chunks + one 1600-byte tail
    assert len(chunks[-1]) == 1600


def test_filesource_missing_file_raises():
    with pytest.raises(CLIError) as exc:
        FileSource("/no/such/file.wav")
    assert exc.value.exit_code == 2


def test_filesource_non_wav_without_ffmpeg_raises(tmp_path, monkeypatch):
    p = tmp_path / "clip.mp3"
    p.write_bytes(b"not really audio")
    monkeypatch.setattr(sources.shutil, "which", lambda _name: None)
    with pytest.raises(CLIError) as exc:
        FileSource(str(p))
    assert exc.value.error_type == "ffmpeg_missing"


def test_filesource_uses_ffmpeg_for_non_wav(tmp_path, monkeypatch):
    p = tmp_path / "clip.mp3"
    p.write_bytes(b"not really audio")
    monkeypatch.setattr(sources.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    class FakeProc:
        def __init__(self):
            self.stdout = io.BytesIO(b"\x00" * 3200 + b"\x01" * 100)
            self.stderr = io.BytesIO(b"")
            self.returncode = 0

        def terminate(self):
            pass

        def wait(self):
            pass

    monkeypatch.setattr(sources.subprocess, "Popen", lambda *a, **k: FakeProc())
    chunks = list(FileSource(str(p), sleep=lambda _s: None))
    assert chunks == [b"\x00" * 3200, b"\x01" * 100]


def test_filesource_ffmpeg_cleanup_on_early_stop(tmp_path, monkeypatch):
    p = tmp_path / "clip.mp3"
    p.write_bytes(b"x")
    monkeypatch.setattr(sources.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    calls = {"terminated": False, "waited": False, "closed": False}

    class FakeProc:
        def __init__(self):
            self.stdout = self
            self.stderr = io.BytesIO(b"")
            self.returncode = 0

        def read(self, _n):
            return b"\x00" * 3200  # endless

        def close(self):
            calls["closed"] = True

        def terminate(self):
            calls["terminated"] = True

        def wait(self):
            calls["waited"] = True

    monkeypatch.setattr(sources.subprocess, "Popen", lambda *a, **k: FakeProc())
    gen = iter(FileSource(str(p), sleep=lambda _s: None))
    next(gen)  # pull one chunk
    gen.close()  # stop early -> generator cleanup runs the finally
    assert calls["terminated"] and calls["waited"] and calls["closed"]


def test_filesource_ffmpeg_failure_raises(tmp_path, monkeypatch):
    p = tmp_path / "bad.mp3"
    p.write_bytes(b"x")
    monkeypatch.setattr(sources.shutil, "which", lambda _n: "/usr/bin/ffmpeg")

    class FailProc:
        def __init__(self):
            self.stdout = io.BytesIO(b"")
            self.stderr = io.BytesIO(b"Invalid data found")
            self.returncode = 1

        def terminate(self):
            pass

        def wait(self):
            pass

    monkeypatch.setattr(sources.subprocess, "Popen", lambda *a, **k: FailProc())
    from assemblyai_cli.errors import APIError

    with pytest.raises(APIError):
        list(sources.FileSource(str(p), sleep=lambda _s: None))


def test_filesource_empty_wav_raises(tmp_path):
    p = tmp_path / "empty.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"")
    with pytest.raises(CLIError) as exc:
        list(FileSource(str(p), sleep=lambda _s: None))
    assert exc.value.error_type == "empty_audio"


def test_micsource_missing_dependency_raises(monkeypatch):
    def boom():
        raise ImportError("No module named 'pyaudio'")

    monkeypatch.setattr(sources, "_load_microphone_stream", boom)
    with pytest.raises(CLIError) as exc:
        list(sources.MicSource(sample_rate=16000))
    assert exc.value.error_type == "mic_missing"
    assert "assemblyai-cli[mic]" in exc.value.message


def test_micsource_yields_from_microphone_stream(monkeypatch):
    captured = {}

    class FakeMic:
        def __init__(self, sample_rate, device_index):
            captured["rate"] = sample_rate
            captured["device"] = device_index

        def __iter__(self):
            return iter([b"\x00\x01", b"\x02\x03"])

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(sources, "_load_microphone_stream", lambda: FakeMic)
    chunks = list(sources.MicSource(sample_rate=16000, device=2))
    assert chunks == [b"\x00\x01", b"\x02\x03"]
    assert captured == {"rate": 16000, "device": 2, "closed": True}


def test_micsource_missing_dependency_at_construction(monkeypatch):
    class ExtrasMissing(ImportError):
        pass

    class FakeMic:
        def __init__(self, sample_rate, device_index):
            raise ExtrasMissing("You must install the extras")

    monkeypatch.setattr(sources, "_load_microphone_stream", lambda: FakeMic)
    with pytest.raises(CLIError) as exc:
        list(sources.MicSource(sample_rate=16000))
    assert exc.value.error_type == "mic_missing"


def test_micsource_device_error_becomes_clierror(monkeypatch):
    class FakeMic:
        def __init__(self, sample_rate, device_index):
            raise OSError("Invalid device")

    monkeypatch.setattr(sources, "_load_microphone_stream", lambda: FakeMic)
    with pytest.raises(CLIError) as exc:
        list(sources.MicSource(sample_rate=16000, device=99))
    assert exc.value.error_type == "mic_error"
