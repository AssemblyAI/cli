import io
import wave
from collections.abc import Generator
from typing import cast

import pytest

from aai_cli.errors import CLIError
from aai_cli.streaming import sources
from aai_cli.streaming.sources import FileSource


def _write_wav(path, *, seconds=0.5, rate=16000):
    frames = int(rate * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x01" * frames)  # 2 bytes/frame, mono 16-bit


def _make_wav(path, *, channels, width, rate):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(b"\x00" * (width * channels * 10))


def test_is_streamable_wav_requires_mono_16bit_16k(tmp_path):
    good = tmp_path / "good.wav"
    _make_wav(good, channels=1, width=2, rate=16000)
    assert sources._is_streamable_wav(good) is True
    # Each criterion alone must disqualify the file (pins the full `and` chain — an
    # `or` would accept any of these because the other two clauses still match).
    stereo = tmp_path / "stereo.wav"
    _make_wav(stereo, channels=2, width=2, rate=16000)
    wrong_rate = tmp_path / "rate.wav"
    _make_wav(wrong_rate, channels=1, width=2, rate=8000)
    wrong_width = tmp_path / "width.wav"
    _make_wav(wrong_width, channels=1, width=1, rate=16000)
    assert sources._is_streamable_wav(stereo) is False
    assert sources._is_streamable_wav(wrong_rate) is False
    assert sources._is_streamable_wav(wrong_width) is False


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
    # Wording matches `client.resolve_audio_source` so the CLI speaks with one voice.
    assert "File not found:" in exc.value.message


def test_filesource_non_wav_without_ffmpeg_raises(tmp_path, monkeypatch):
    p = tmp_path / "clip.mp3"
    p.write_bytes(b"not really audio")
    monkeypatch.setattr(sources.shutil, "which", lambda _name: None)
    with pytest.raises(CLIError) as exc:
        FileSource(str(p))
    assert exc.value.error_type == "ffmpeg_missing"
    assert exc.value.exit_code == 2


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
    gen = cast(Generator[bytes, None, None], iter(FileSource(str(p), sleep=lambda _s: None)))
    next(gen)  # pull one chunk
    gen.close()  # stop early -> generator cleanup runs the finally
    assert calls["terminated"] and calls["waited"] and calls["closed"]


def test_filesource_ffmpeg_wait_keyboardinterrupt_is_silenced(tmp_path, monkeypatch):
    # A stray Ctrl-C while the generator is finalized (proc.wait()) must not escape
    # as the noisy "Exception ignored in generator"; the child is killed instead.
    p = tmp_path / "clip.mp3"
    p.write_bytes(b"x")
    monkeypatch.setattr(sources.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    calls = {"killed": False}

    class FakeProc:
        def __init__(self):
            self.stdout = self
            self.stderr = io.BytesIO(b"")
            self.returncode = 0

        def read(self, _n):
            return b"\x00" * 3200  # endless

        def close(self):
            pass

        def terminate(self):
            pass

        def wait(self):
            raise KeyboardInterrupt  # second Ctrl-C lands during cleanup

        def kill(self):
            calls["killed"] = True

    monkeypatch.setattr(sources.subprocess, "Popen", lambda *a, **k: FakeProc())
    gen = cast(Generator[bytes, None, None], iter(FileSource(str(p), sleep=lambda _s: None)))
    next(gen)  # pull one chunk
    gen.close()  # must return cleanly despite wait() raising KeyboardInterrupt
    assert calls["killed"] is True


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
    from aai_cli.errors import APIError

    with pytest.raises(APIError):
        list(sources.FileSource(str(p), sleep=lambda _s: None))


def test_filesource_ffmpeg_not_terminated_on_natural_eof(tmp_path, monkeypatch):
    # On a clean EOF ffmpeg is allowed to exit on its own; terminating it would
    # surface as a spurious exit -15. Pins the `completed = True` flag.
    p = tmp_path / "clip.mp3"
    p.write_bytes(b"x")
    monkeypatch.setattr(sources.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    holder = {}

    class FakeProc:
        def __init__(self):
            self.stdout = io.BytesIO(b"\x00" * 3200)  # one full chunk, then EOF
            self.stderr = io.BytesIO(b"")
            self.returncode = 0
            self.terminated = False
            holder["proc"] = self

        def terminate(self):
            self.terminated = True

        def wait(self):
            pass

    monkeypatch.setattr(sources.subprocess, "Popen", lambda *a, **k: FakeProc())
    chunks = list(FileSource(str(p), sleep=lambda _s: None))
    assert chunks == [b"\x00" * 3200]
    assert holder["proc"].terminated is False


def test_filesource_ffmpeg_failure_empty_stderr_reports_exit_code(tmp_path, monkeypatch):
    # When ffmpeg fails but writes nothing to stderr, the error message falls back to
    # the exit code. Pins the `detail or f'exit {returncode}'` (an `and` would blank it).
    from aai_cli.errors import APIError

    p = tmp_path / "bad.mp3"
    p.write_bytes(b"x")
    monkeypatch.setattr(sources.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    class FailProc:
        def __init__(self):
            self.stdout = io.BytesIO(b"")
            self.stderr = io.BytesIO(b"")  # no diagnostic text
            self.returncode = 3

        def terminate(self):
            pass

        def wait(self):
            pass

    monkeypatch.setattr(sources.subprocess, "Popen", lambda *a, **k: FailProc())
    with pytest.raises(APIError) as exc:
        list(FileSource(str(p), sleep=lambda _s: None))
    assert "exit 3" in exc.value.message


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
    assert exc.value.exit_code == 2


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


def test_missing_ffmpeg_suggests_install(monkeypatch, tmp_path):
    # A non-WAV file with ffmpeg absent must raise with an actionable suggestion.
    f = tmp_path / "audio.mp3"
    f.write_bytes(b"not really audio")
    monkeypatch.setattr(sources.shutil, "which", lambda name: None)
    with pytest.raises(CLIError) as exc:
        sources.FileSource(str(f))
    assert "ffmpeg" in exc.value.message
    assert exc.value.suggestion is not None
    assert "WAV" in exc.value.suggestion or "ffmpeg" in exc.value.suggestion
