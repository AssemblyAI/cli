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


def test_is_downloadable_url_matches_media_pages():
    # YouTube by shape; podcast pages because a dedicated yt-dlp extractor claims them.
    assert youtube.is_downloadable_url("https://youtu.be/abc123")
    assert youtube.is_downloadable_url(
        "https://podcasts.apple.com/us/podcast/some-show/id1535809341?i=1000123456789"
    )
    assert youtube.is_downloadable_url("https://www.spreaker.com/episode/12345")
    assert youtube.is_downloadable_url("http://www.spreaker.com/episode/12345")
    assert youtube.is_downloadable_url("  https://www.spreaker.com/episode/12345  ")


def test_is_downloadable_url_passes_direct_and_local_sources_through():
    # Direct audio URLs and unknown pages match only yt-dlp's catch-all Generic
    # extractor — the API fetches those itself, so they must not route to a download.
    assert not youtube.is_downloadable_url("https://example.com/episode.mp3")
    assert not youtube.is_downloadable_url("https://example.com/blog/post")
    assert not youtube.is_downloadable_url("/local/file.wav")
    assert not youtube.is_downloadable_url("podcasts.apple.com/no-scheme")
    assert not youtube.is_downloadable_url(None)
    assert not youtube.is_downloadable_url("")


def test_is_downloadable_url_without_ytdlp_still_matches_youtube(monkeypatch):
    # With yt-dlp unimportable, YouTube still matches by URL shape (so download_audio
    # can raise its install hint); extractor-matched hosts degrade to API pass-through.
    monkeypatch.setitem(sys.modules, "yt_dlp", None)  # force ImportError
    monkeypatch.setitem(sys.modules, "yt_dlp.extractor", None)
    assert youtube.is_downloadable_url("https://youtu.be/abc123")
    assert not youtube.is_downloadable_url("https://www.spreaker.com/episode/12345")


def _fake_ytdlp(monkeypatch, ydl_cls):
    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=ydl_cls))


def test_download_audio_returns_prepared_path(tmp_path, monkeypatch):
    created = tmp_path / "vid123.m4a"
    captured = {}

    class FakeYDL:
        def __init__(self, opts):
            captured["opts"] = opts

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            captured["download"] = download
            created.write_bytes(b"audio")
            return {"id": "vid123", "ext": "m4a"}

        def prepare_filename(self, info):
            return str(created)

    _fake_ytdlp(monkeypatch, FakeYDL)
    out = youtube.download_audio("https://youtu.be/vid123", tmp_path)
    assert out == created
    assert out.is_file()
    # yt-dlp is driven quietly (no console noise) and actually downloads the media.
    assert captured["opts"]["quiet"] is True
    assert captured["opts"]["no_warnings"] is True
    assert captured["opts"]["noprogress"] is True
    assert captured["download"] is True


def test_download_audio_routes_ytdlp_output_to_silent_logger(tmp_path, monkeypatch, capsys):
    # yt-dlp's default logger writes its own "ERROR: …" line to stderr before the CLI's
    # clean error, duplicating the message; the passed logger must swallow everything.
    import logging

    captured = {}

    class FakeYDL:
        def __init__(self, opts):
            captured["opts"] = opts

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            (tmp_path / "x.m4a").write_bytes(b"audio")
            return {"id": "x", "ext": "m4a"}

        def prepare_filename(self, info):
            return str(tmp_path / "x.m4a")

    _fake_ytdlp(monkeypatch, FakeYDL)
    youtube.download_audio("https://youtu.be/x", tmp_path)
    logger = captured["opts"]["logger"]
    # Structurally quiet: no propagation to root, only swallow-everything handlers.
    assert logger.name == "aai_cli.youtube.yt_dlp"
    assert logger.propagate is False
    assert logger.handlers
    assert all(isinstance(h, logging.NullHandler) for h in logger.handlers)
    # Behaviorally quiet: even an ERROR record produces no console output.
    logger.error("ERROR: [youtube] nope: Video unavailable")
    logger.warning("WARNING: noisy")
    logger.debug("[debug] noise")
    out = capsys.readouterr()
    assert out.err == ""
    assert out.out == ""


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


def test_download_audio_falls_back_to_largest_file(tmp_path, monkeypatch):
    # yt-dlp can leave sidecars (thumbnail, .info.json) next to the audio track;
    # the fallback must pick the audio (largest), not an arbitrary iterdir() entry.
    audio = tmp_path / "actual.webm"
    thumb = tmp_path / "actual.webp"

    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            thumb.write_bytes(b"\x00" * 16)  # small sidecar
            audio.write_bytes(b"\x00" * 4096)  # the real, much larger track
            return {"id": "x"}

        def prepare_filename(self, info):
            return str(tmp_path / "guessed.m4a")  # wrong extension; file doesn't exist

    _fake_ytdlp(monkeypatch, FakeYDL)
    assert youtube.download_audio("https://youtu.be/x", tmp_path) == audio


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
    assert exc.value.exit_code == 1
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
