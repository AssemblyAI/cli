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


def test_filesource_url_skips_local_check_and_streams_via_ffmpeg(monkeypatch):
    monkeypatch.setattr(sources.shutil, "which", lambda _n: "/usr/bin/ffmpeg")
    captured = {}

    class FakeProc:
        def __init__(self):
            self.stdout = io.BytesIO(b"\x00" * 3200)
            self.stderr = io.BytesIO(b"")
            self.returncode = 0

        def terminate(self):
            pass

        def wait(self):
            pass

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(sources.subprocess, "Popen", fake_popen)
    url = "https://example.com/clip.mp3"
    chunks = list(FileSource(url, sleep=lambda _s: None))  # no is_file() check for URLs
    assert chunks == [b"\x00" * 3200]
    assert url in captured["cmd"]  # passed straight to ffmpeg's -i


def test_filesource_url_without_ffmpeg_raises(monkeypatch):
    monkeypatch.setattr(sources.shutil, "which", lambda _n: None)
    with pytest.raises(CLIError) as exc:
        FileSource("https://example.com/clip.mp3")
    assert exc.value.error_type == "ffmpeg_missing"
