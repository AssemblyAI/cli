"""Tests for `assembly dub`'s YouTube/media-page URL sources: the audio/video
download (`youtube.download_media`, faked), the --video flag rules, and the
cwd-relative default output naming. The local-file pipeline runs live in
test_dub_pipeline.py."""

from __future__ import annotations

import contextlib
import dataclasses
import json
from pathlib import Path

import pytest

from aai_cli.app.context import AppState
from aai_cli.commands.dub import _exec as dub_exec
from aai_cli.core import youtube
from aai_cli.core.errors import UsageError
from tests._dub_helpers import (
    DEFAULTS,
    enable_sandbox,
    patch_api_key,
    record_ffmpeg,
    record_synthesize,
    record_transcribe,
    record_translate,
    write_media,
)

YT_URL = "https://www.youtube.com/watch?v=abc123"


@pytest.fixture
def media(tmp_path: Path) -> Path:
    return write_media(tmp_path)


@pytest.fixture(autouse=True)
def _sandbox_and_key(monkeypatch: pytest.MonkeyPatch):
    enable_sandbox(monkeypatch)
    patch_api_key(monkeypatch)


@pytest.fixture
def fake_transcribe(monkeypatch: pytest.MonkeyPatch):
    return record_transcribe(monkeypatch)


@pytest.fixture
def fake_translate(monkeypatch: pytest.MonkeyPatch):
    return record_translate(monkeypatch)


@pytest.fixture
def fake_synthesize(monkeypatch: pytest.MonkeyPatch):
    return record_synthesize(monkeypatch)


@pytest.fixture
def fake_ffmpeg(monkeypatch: pytest.MonkeyPatch):
    return record_ffmpeg(monkeypatch)


@pytest.fixture
def fake_download(monkeypatch: pytest.MonkeyPatch):
    """Stand in for yt-dlp: 'download' a fixed media file into the temp dir."""
    seen: dict[str, object] = {}

    def download(url, dest_dir, *, video=False, download_sections=None):
        seen["url"] = url
        seen["video"] = video
        seen["download_sections"] = download_sections
        seen["dest_dir"] = dest_dir
        path = dest_dir / ("vid123.mp4" if video else "vid123.m4a")
        path.write_bytes(b"\x00media")
        seen["path"] = path
        return path

    monkeypatch.setattr(youtube, "download_media", download)
    return seen


def _run(opts, *, json_mode):
    dub_exec.run_dub(opts, AppState(), json_mode=json_mode)


def test_run_dub_youtube_downloads_and_dubs_into_cwd(
    tmp_path,
    fake_download,
    fake_transcribe,
    fake_translate,
    fake_synthesize,
    fake_ffmpeg,
    capsys,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    opts = dataclasses.replace(DEFAULTS, media=YT_URL)
    _run(opts, json_mode=True)
    # Audio-only download by default — the whole source, no section slicing —
    # and the downloaded temp file feeds the pipeline.
    assert fake_download["url"] == YT_URL
    assert fake_download["video"] is False
    assert fake_download["download_sections"] == []
    assert Path(fake_download["dest_dir"]).name.startswith("aai-dub-src-")
    assert fake_transcribe["audio"] == str(fake_download["path"])
    # ffmpeg muxes over the downloaded file; the default output lands in the cwd,
    # named after the download (the temp dir is gone after the run).
    args = fake_ffmpeg["args"]
    assert args[6] == str(fake_download["path"])
    out = tmp_path / "vid123.dub.german.m4a"
    assert args[-1] == str(out)
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == YT_URL
    assert payload["out"] == str(out)


def test_run_dub_youtube_video_keeps_the_picture(
    tmp_path,
    fake_download,
    fake_transcribe,
    fake_translate,
    fake_synthesize,
    fake_ffmpeg,
    capsys,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    messages: list[str] = []

    @contextlib.contextmanager
    def fake_status(message, *, json_mode, quiet):
        messages.append(message)
        yield

    monkeypatch.setattr(dub_exec.output, "status", fake_status)
    opts = dataclasses.replace(DEFAULTS, media=YT_URL, video=True)
    _run(opts, json_mode=True)
    # --video fetches the full video; the dubbed default output keeps its extension.
    assert fake_download["video"] is True
    assert messages[0] == "Downloading video…"
    payload = json.loads(capsys.readouterr().out)
    assert payload["out"] == str(tmp_path / "vid123.dub.german.mp4")


def test_run_dub_youtube_audio_download_status_message(
    tmp_path,
    fake_download,
    fake_transcribe,
    fake_translate,
    fake_synthesize,
    fake_ffmpeg,
    capsys,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    messages: list[str] = []

    @contextlib.contextmanager
    def fake_status(message, *, json_mode, quiet):
        messages.append(message)
        yield

    monkeypatch.setattr(dub_exec.output, "status", fake_status)
    _run(dataclasses.replace(DEFAULTS, media=YT_URL), json_mode=True)
    assert messages[0] == "Downloading audio…"


def test_run_dub_youtube_honors_explicit_out(
    tmp_path,
    fake_download,
    fake_transcribe,
    fake_translate,
    fake_synthesize,
    fake_ffmpeg,
    capsys,
):
    out = tmp_path / "dubbed.mp4"
    opts = dataclasses.replace(DEFAULTS, media=YT_URL, out=out)
    _run(opts, json_mode=True)
    assert fake_ffmpeg["args"][-1] == str(out)


def test_run_dub_youtube_download_sections_slice_the_download(
    tmp_path,
    fake_download,
    fake_transcribe,
    fake_translate,
    fake_synthesize,
    fake_ffmpeg,
    capsys,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    opts = dataclasses.replace(DEFAULTS, media=YT_URL, download_sections=["*0:00-15:00"])
    _run(opts, json_mode=True)
    # The specs reach yt-dlp verbatim, so only that slice is fetched (and dubbed).
    assert fake_download["download_sections"] == ["*0:00-15:00"]


def test_run_dub_download_sections_require_a_url_source(media, monkeypatch):
    # A local file is never downloaded, so the slice specs would be a silent
    # no-op — they are rejected instead, with the local-file alternative named.
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    opts = dataclasses.replace(DEFAULTS, media=str(media), download_sections=["*0:00-15:00"])
    with pytest.raises(UsageError) as exc:
        _run(opts, json_mode=False)
    assert "--download-sections only applies to a downloadable URL source" in exc.value.message
    assert "assembly clip" in (exc.value.suggestion or "")


def test_run_dub_video_requires_a_url_source(media, monkeypatch):
    # A local file's video stream is already copied into the dub, so --video
    # would be a silent no-op — it is rejected instead.
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    opts = dataclasses.replace(DEFAULTS, media=str(media), video=True)
    with pytest.raises(UsageError) as exc:
        _run(opts, json_mode=False)
    assert "--video only applies to a downloadable URL source" in exc.value.message


def test_run_dub_rejects_non_downloadable_url(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    opts = dataclasses.replace(DEFAULTS, media="https://example.com/episode.mp3")
    with pytest.raises(UsageError) as exc:
        _run(opts, json_mode=False)
    assert "assembly dub can't fetch this URL" in exc.value.message
    assert "dubs a local file" in exc.value.message
    assert "Download the media first" in (exc.value.suggestion or "")
