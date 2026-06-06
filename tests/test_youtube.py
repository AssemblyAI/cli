import sys
import types

import pytest

from aai_cli import youtube
from aai_cli.errors import CLIError


def test_is_youtube_url_variants():
    assert youtube.is_youtube_url("https://www.youtube.com/watch?v=abc")
    assert youtube.is_youtube_url("http://youtube.com/watch?v=abc")
    assert youtube.is_youtube_url("https://youtu.be/abc123")
    assert youtube.is_youtube_url("youtube.com/watch?v=x")
    assert youtube.is_youtube_url("https://music.youtube.com/watch?v=x")
    assert not youtube.is_youtube_url("https://example.com/clip.mp3")
    assert not youtube.is_youtube_url("/local/file.wav")
    assert not youtube.is_youtube_url(None)
    assert not youtube.is_youtube_url("")


def _fake_ytdlp(monkeypatch, ydl_cls):
    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=ydl_cls))


def test_download_audio_returns_prepared_path(tmp_path, monkeypatch):
    created = tmp_path / "vid123.m4a"

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            created.write_bytes(b"audio")
            return {"id": "vid123", "ext": "m4a"}

        def prepare_filename(self, info):
            return str(created)

    _fake_ytdlp(monkeypatch, FakeYDL)
    out = youtube.download_audio("https://youtu.be/vid123", tmp_path)
    assert out == created
    assert out.is_file()


def test_download_audio_falls_back_to_landed_file(tmp_path, monkeypatch):
    landed = tmp_path / "actual.webm"

    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            landed.write_bytes(b"x")
            return {"id": "x"}

        def prepare_filename(self, info):
            return str(tmp_path / "guessed.m4a")  # wrong extension; file doesn't exist

    _fake_ytdlp(monkeypatch, FakeYDL)
    assert youtube.download_audio("https://youtu.be/x", tmp_path) == landed


def test_download_audio_no_file_produced_raises(tmp_path, monkeypatch):
    # prepare_filename points at a missing file and nothing landed in dest_dir.
    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            return {"id": "x"}  # writes no file

        def prepare_filename(self, info):
            return str(tmp_path / "guessed.m4a")  # doesn't exist

    _fake_ytdlp(monkeypatch, FakeYDL)
    with pytest.raises(CLIError) as exc:
        youtube.download_audio("https://youtu.be/x", tmp_path)
    assert exc.value.error_type == "youtube_error"
    assert "no audio file" in exc.value.message


def test_download_audio_error_raises_cli_error(tmp_path, monkeypatch):
    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            raise RuntimeError("network down")

        def prepare_filename(self, info):
            return ""

    _fake_ytdlp(monkeypatch, FakeYDL)
    with pytest.raises(CLIError) as exc:
        youtube.download_audio("https://youtu.be/x", tmp_path)
    assert exc.value.error_type == "youtube_error"
    assert exc.value.exit_code == 1


def test_download_audio_missing_ytdlp_raises(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yt_dlp", None)  # force ImportError on `import yt_dlp`
    with pytest.raises(CLIError) as exc:
        youtube.download_audio("https://youtu.be/x", tmp_path)
    assert exc.value.error_type == "ytdlp_missing"
    assert exc.value.exit_code == 2


def test_missing_ytdlp_suggests_install(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yt_dlp", None)  # force ImportError on `import yt_dlp`
    with pytest.raises(CLIError) as exc:
        youtube.download_audio("https://youtu.be/x", tmp_path)
    assert "yt-dlp" in exc.value.message
    assert "pip install yt-dlp" in (exc.value.suggestion or "")
