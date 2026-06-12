import importlib
import re
import sys
import types

import pytest

from aai_cli import youtube
from aai_cli.errors import CLIError, UsageError


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
    # Cache the real yt_dlp.utils submodule first: _section_timestamp lazily does
    # `from yt_dlp.utils import parse_duration`, and once the parent is replaced by a
    # SimpleNamespace (not a package) that import can only resolve through sys.modules.
    # Without this, the test would depend on whether an earlier (randomly ordered)
    # test had already imported the real yt-dlp.
    importlib.import_module("yt_dlp.utils")
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


def _raising_ydl(message):
    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            raise RuntimeError(message)

        def prepare_filename(self, info):
            return ""

    return FakeYDL


def test_download_audio_error_raises_cli_error(tmp_path, monkeypatch):
    _fake_ytdlp(monkeypatch, _raising_ydl("network down"))
    with pytest.raises(CLIError) as exc:
        youtube.download_audio("https://youtu.be/x", tmp_path)
    assert exc.value.error_type == "youtube_error"
    assert exc.value.exit_code == 1
    # A message without boilerplate passes through untouched.
    assert exc.value.message == "Could not download audio from https://youtu.be/x: network down"


_YTDLP_BOILERPLATE = (
    "please report this issue on  https://github.com/yt-dlp/yt-dlp/issues?q= , filling "
    "out the appropriate issue template. Confirm you are on the latest version using  yt-dlp -U"
)


def test_download_audio_trims_ytdlp_bug_report_boilerplate(tmp_path, monkeypatch):
    # yt-dlp appends report-a-bug boilerplate to extractor errors; only the
    # meaningful part should reach the user, without the "ERROR: " prefix.
    message = f"ERROR: [youtube] abc: Video unavailable; {_YTDLP_BOILERPLATE}"
    _fake_ytdlp(monkeypatch, _raising_ydl(message))
    with pytest.raises(CLIError) as exc:
        youtube.download_audio("https://youtu.be/x", tmp_path)
    assert exc.value.message == (
        "Could not download audio from https://youtu.be/x: [youtube] abc: Video unavailable"
    )
    assert "report this issue" not in exc.value.message
    assert "latest version" not in exc.value.message


def test_download_audio_all_boilerplate_message_falls_back_to_raw_text(tmp_path, monkeypatch):
    # When trimming would leave nothing, keep the original message over an empty error.
    message = _YTDLP_BOILERPLATE[0].upper() + _YTDLP_BOILERPLATE[1:]
    _fake_ytdlp(monkeypatch, _raising_ydl(message))
    with pytest.raises(CLIError) as exc:
        youtube.download_audio("https://youtu.be/x", tmp_path)
    assert message in exc.value.message


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


def test_parse_download_sections_timestamp_ranges():
    # A "*"-prefixed spec is one or more comma-separated start-end timestamp ranges;
    # an omitted/`inf` end means "to the end", and a leading "-" negates a bound.
    assert youtube.parse_download_sections(["*0:00-5:00"]) == ([], [(0.0, 300.0)], False)
    assert youtube.parse_download_sections(["*10:00-inf"]) == ([], [(600.0, float("inf"))], False)
    assert youtube.parse_download_sections(["*1:30-"]) == ([], [(90.0, float("inf"))], False)
    # Comma-separated ranges in one spec, tolerating whitespace around each token.
    assert youtube.parse_download_sections(["*0:30-1:00, 2:00-3:00"]) == (
        [],
        [(30.0, 60.0), (120.0, 180.0)],
        False,
    )
    # "infinite" is accepted as an alias for "inf".
    assert youtube.parse_download_sections(["*0:00-infinite"]) == ([], [(0.0, float("inf"))], False)
    # A leading "-" on a bound negates it (offset from the end) — distinguishes the sign
    # branch from a no-op.
    assert youtube.parse_download_sections(["*-5:00-10:00"]) == ([], [(-300.0, 600.0)], False)


def test_parse_download_sections_chapters_and_from_url():
    # A non-"*" spec is a chapter-title regex; "*from-url" keeps the source's own range.
    assert youtube.parse_download_sections(["intro"]) == (["intro"], [], False)
    assert youtube.parse_download_sections(["*from-url"]) == ([], [], True)
    # Specs combine: a chapter regex plus a timestamp range plus from-url.
    assert youtube.parse_download_sections(["intro", "*0:00-1:00", "*from-url"]) == (
        ["intro"],
        [(0.0, 60.0)],
        True,
    )


@pytest.mark.parametrize(
    ("spec", "needle"),
    [
        ("*abc-def", 'time "abc"'),  # unparseable timestamp
        ("*5:00", 'time range "5:00"'),  # missing the "-" separator
        ("*-", 'time range "-"'),  # a lone "-" is not a range
        ("*1:00--inf", "-inf"),  # "-inf" is not a valid end
        ("(", "regex"),  # malformed chapter regex
    ],
)
def test_parse_download_sections_rejects_malformed(spec, needle):
    with pytest.raises(UsageError) as exc:
        youtube.parse_download_sections([spec])
    assert needle in exc.value.message
    assert exc.value.exit_code == 2


def test_download_audio_with_sections_sets_download_ranges(tmp_path, monkeypatch):
    # --download-sections must reach yt-dlp as download_ranges + force_keyframes_at_cuts
    # (exact cuts, not the nearest keyframe).
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
    youtube.download_audio(
        "https://youtu.be/x", tmp_path, download_sections=["*0:00-5:00", "intro"]
    )
    download_ranges = captured["opts"]["download_ranges"]
    assert download_ranges.ranges == [(0.0, 300.0)]
    # Chapter-regex specs are compiled before reaching yt-dlp.
    assert download_ranges.chapters == [re.compile("intro")]
    assert captured["opts"]["force_keyframes_at_cuts"] is True


def test_download_audio_without_sections_omits_download_ranges(tmp_path, monkeypatch):
    # The default path must not set download_ranges (downloads the whole track).
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
    assert "download_ranges" not in captured["opts"]
    assert "force_keyframes_at_cuts" not in captured["opts"]
