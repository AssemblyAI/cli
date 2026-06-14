"""Direct tests of the `assembly caption` options/run seam (aai_cli/caption_exec.py):
the pure helpers (output naming, filtergraph escaping), validation order, and the
faked transcribe → SRT export → ffmpeg burn-in runs. The boundaries are faked at
the modules caption_exec calls into (`client.transcribe`, `client.get_transcript`,
`youtube.download_media`) and at `mediafile.run_ffmpeg`; argv parsing lives in
test_caption_command.py."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from aai_cli.app import mediafile
from aai_cli.app.context import AppState
from aai_cli.commands.caption import _exec as caption_exec
from aai_cli.commands.caption._exec import CaptionOptions
from aai_cli.core import client, config, youtube
from aai_cli.core.errors import CLIError, UsageError
from tests._clip_helpers import plain

# The CLI's flag defaults, as data. Tests override per-case with dataclasses.replace.
DEFAULTS = CaptionOptions(
    media="talk.mp4",
    transcript_id=None,
    chars_per_caption=None,
    font_size=None,
    out=None,
)

SRT = "1\n00:00:00,500 --> 00:00:01,500\nHello.\n\n2\n00:00:02,000 --> 00:00:03,000\nWorld.\n"


def fake_transcript(srt: str = SRT, transcript_id: str = "tr_cap"):
    """A transcript double whose SRT export records the chars_per_caption it got."""
    calls: list[object] = []

    def export(chars_per_caption=None):
        calls.append(chars_per_caption)
        return srt

    return SimpleNamespace(id=transcript_id, export_subtitles_srt=export, export_calls=calls)


def record_ffmpeg(monkeypatch, *, returncode: int = 0, stderr: str = ""):
    """Resolve ffmpeg and record the invocation plus the SRT it was handed.

    The temp SRT is deleted right after the burn, so its contents are captured
    here, while the file still exists (args[8] is the -vf filtergraph).
    """
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    recorded: dict[str, object] = {}

    def run(args: list[str]) -> subprocess.CompletedProcess[str]:
        recorded["args"] = args
        escaped = args[8].removeprefix("subtitles=").split(":force_style")[0]
        # subtitles_filter escapes filtergraph metacharacters (and the Windows drive
        # colon) with a leading backslash; reverse that to recover the real on-disk path.
        srt_path = re.sub(r"\\(.)", r"\1", escaped)
        recorded["srt"] = Path(srt_path).read_text(encoding="utf-8")
        return subprocess.CompletedProcess(
            args=args, returncode=returncode, stdout="", stderr=stderr
        )

    monkeypatch.setattr(mediafile, "run_ffmpeg", run)
    return recorded


@pytest.fixture
def media(tmp_path: Path) -> Path:
    path = tmp_path / "talk.mp4"
    path.write_bytes(b"\x00fake-media")
    return path


@pytest.fixture(autouse=True)
def _fake_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "resolve_api_key", lambda **_: "test-key")


@pytest.fixture
def fake_transcribe(monkeypatch: pytest.MonkeyPatch):
    """Record the transcription request and return the canned transcript."""
    calls: dict[str, object] = {}
    transcript = fake_transcript()

    def _fake(api_key, audio, *, config):
        calls["api_key"] = api_key
        calls["audio"] = audio
        calls["config"] = config
        return transcript

    monkeypatch.setattr(client, "transcribe", _fake)
    calls["transcript"] = transcript
    return calls


@pytest.fixture
def fake_ffmpeg(monkeypatch: pytest.MonkeyPatch):
    return record_ffmpeg(monkeypatch)


def _run(opts, *, json_mode):
    caption_exec.run_caption(opts, AppState(), json_mode=json_mode)


# --- records and pure helpers --------------------------------------------------


def test_options_are_immutable():
    field_name = dataclasses.fields(DEFAULTS)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(DEFAULTS, field_name, None)


def test_default_out_path():
    assert caption_exec.default_out_path(Path("/x/talk.mp4")) == Path("/x/talk.captioned.mp4")


def test_subtitles_filter_plain_path():
    assert caption_exec.subtitles_filter(Path("/tmp/c.srt"), None) == "subtitles=/tmp/c.srt"


def test_subtitles_filter_appends_font_size():
    spec = caption_exec.subtitles_filter(Path("/tmp/c.srt"), 28)
    assert spec == "subtitles=/tmp/c.srt:force_style=FontSize=28"


def test_subtitles_filter_escapes_filtergraph_metacharacters():
    # ffmpeg's filtergraph syntax gives these characters meaning; an unescaped
    # one in a TMPDIR-derived path would corrupt the filter spec.
    spec = caption_exec.subtitles_filter(Path("/tmp/a'b:c,d;e[f]g.srt"), None)
    assert spec == "subtitles=/tmp/a\\'b\\:c\\,d\\;e\\[f\\]g.srt"


# --- validation order (cheap local checks before any credential or network) ----


def test_run_caption_requires_ffmpeg(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(CLIError) as exc:
        _run(DEFAULTS, json_mode=False)
    assert exc.value.error_type == "missing_dependency"
    # The purpose string pins the shared helper's parameterization.
    assert "ffmpeg is required to burn captions into video" in exc.value.message


def test_run_caption_rejects_missing_file(fake_ffmpeg, tmp_path):
    opts = dataclasses.replace(DEFAULTS, media=str(tmp_path / "nope.mp4"))
    with pytest.raises(CLIError) as exc:
        _run(opts, json_mode=False)
    assert exc.value.error_type == "file_not_found"
    assert exc.value.exit_code == 2
    # The command name + kind pin the shared helper's parameterization.
    assert "assembly caption needs a local video file" in (exc.value.suggestion or "")


def test_run_caption_rejects_directory(fake_ffmpeg, tmp_path):
    opts = dataclasses.replace(DEFAULTS, media=str(tmp_path))
    with pytest.raises(CLIError) as exc:
        _run(opts, json_mode=False)
    assert exc.value.error_type == "not_a_file"
    assert exc.value.exit_code == 2
    assert "not a directory" in (exc.value.suggestion or "")


def test_run_caption_refuses_to_overwrite_the_input(fake_ffmpeg, media):
    opts = dataclasses.replace(DEFAULTS, media=str(media), out=media)
    with pytest.raises(UsageError) as exc:
        _run(opts, json_mode=False)
    assert "overwrite the input file" in exc.value.message


def test_run_caption_rejects_non_downloadable_url(fake_ffmpeg):
    opts = dataclasses.replace(DEFAULTS, media="https://example.com/episode.mp3")
    with pytest.raises(UsageError) as exc:
        _run(opts, json_mode=False)
    assert "assembly caption can't fetch this URL" in exc.value.message
    assert "captions a local file" in exc.value.message
    assert "Download the video first" in (exc.value.suggestion or "")


def test_run_caption_rejects_remote_urls_with_the_url_intact(fake_ffmpeg):
    # Path() would collapse "//" and echo a corrupted "s3:/bucket/…" back.
    opts = dataclasses.replace(DEFAULTS, media="s3://bucket/talk.mp4")
    with pytest.raises(UsageError) as exc:
        _run(opts, json_mode=False)
    assert "s3://bucket/talk.mp4" in exc.value.message
    assert "Download the video first" in (exc.value.suggestion or "")


# --- the faked pipeline ---------------------------------------------------------


def test_run_caption_end_to_end(media, fake_transcribe, fake_ffmpeg, capsys):
    opts = dataclasses.replace(DEFAULTS, media=str(media))
    _run(opts, json_mode=True)

    # Transcription: the local file, with the resolved key, no diarization
    # (captions don't need speaker labels).
    assert fake_transcribe["api_key"] == "test-key"
    assert fake_transcribe["audio"] == str(media)
    assert fake_transcribe["config"].speaker_labels is None
    # No --chars-per-caption: the export endpoint gets None (its own default).
    assert fake_transcribe["transcript"].export_calls == [None]

    # The burn: re-encoded video with the SRT filter, audio copied, default out.
    out = media.parent / "talk.captioned.mp4"
    args = fake_ffmpeg["args"]
    filtergraph = args[8]
    assert args == [
        "/usr/bin/ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(media),
        "-vf",
        filtergraph,
        "-map",
        "0:v",
        "-map",
        "0:a?",
        "-c:a",
        "copy",
        str(out),
    ]
    assert filtergraph.startswith("subtitles=")
    assert "aai-caption-" in filtergraph
    assert filtergraph.endswith("captions.srt")
    assert fake_ffmpeg["srt"] == SRT

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "source": str(media),
        "out": str(out),
        "transcript_id": "tr_cap",
        "captions": 2,
    }


def test_run_caption_human_summary(media, fake_transcribe, fake_ffmpeg, capsys):
    opts = dataclasses.replace(DEFAULTS, media=str(media), out=Path("captioned.mp4"))
    _run(opts, json_mode=False)
    out = plain(capsys.readouterr().out)
    assert "captioned.mp4" in out
    assert "2 caption(s) burned in" in out


def test_run_caption_status_messages(media, fake_transcribe, fake_ffmpeg, monkeypatch):
    messages: list[str] = []

    @contextlib.contextmanager
    def fake_status(message, *, json_mode, quiet):
        messages.append(message)
        yield

    monkeypatch.setattr(caption_exec.output, "status", fake_status)
    _run(dataclasses.replace(DEFAULTS, media=str(media)), json_mode=False)
    assert messages == ["Transcribing for captions…", "Fetching captions…", "Burning captions…"]


def test_dash_prefixed_out_is_disambiguated_for_ffmpeg(
    media, fake_transcribe, fake_ffmpeg, monkeypatch, tmp_path
):
    # A bare "-cap.mp4" argv token would be parsed by ffmpeg as an option.
    monkeypatch.chdir(tmp_path)
    opts = dataclasses.replace(DEFAULTS, media=str(media), out=Path("-cap.mp4"))
    _run(opts, json_mode=True)
    assert fake_ffmpeg["args"][-1] == "./-cap.mp4"


def test_run_caption_forwards_chars_per_caption(media, fake_transcribe, fake_ffmpeg):
    opts = dataclasses.replace(DEFAULTS, media=str(media), chars_per_caption=32)
    _run(opts, json_mode=True)
    assert fake_transcribe["transcript"].export_calls == [32]


def test_run_caption_font_size_reaches_the_filtergraph(media, fake_transcribe, fake_ffmpeg):
    opts = dataclasses.replace(DEFAULTS, media=str(media), font_size=28)
    _run(opts, json_mode=True)
    assert fake_ffmpeg["args"][8].endswith(":force_style=FontSize=28")


def test_transcript_id_reuses_existing_transcript(media, fake_ffmpeg, monkeypatch, capsys):
    fetched: dict[str, object] = {}
    transcript = fake_transcript(transcript_id="tr_99")

    def get_transcript(api_key, transcript_id):
        fetched["args"] = (api_key, transcript_id)
        return transcript

    monkeypatch.setattr(client, "get_transcript", get_transcript)
    monkeypatch.setattr(
        client,
        "transcribe",
        lambda *a, **k: pytest.fail("must not re-transcribe with --transcript-id"),
    )
    opts = dataclasses.replace(DEFAULTS, media=str(media), transcript_id="tr_99")
    _run(opts, json_mode=True)
    assert fetched["args"] == ("test-key", "tr_99")
    payload = json.loads(capsys.readouterr().out)
    assert payload["transcript_id"] == "tr_99"


def test_empty_srt_is_a_no_captions_error(media, fake_ffmpeg, monkeypatch):
    monkeypatch.setattr(client, "transcribe", lambda *a, **k: fake_transcript(srt="  \n"))
    opts = dataclasses.replace(DEFAULTS, media=str(media))
    with pytest.raises(CLIError) as exc:
        _run(opts, json_mode=False)
    assert exc.value.error_type == "no_captions"
    assert exc.value.exit_code == 2
    assert "Transcript tr_cap has no captions to burn in" in exc.value.message


def test_ffmpeg_failure_reports_last_stderr_line(media, fake_transcribe, monkeypatch):
    record_ffmpeg(monkeypatch, returncode=1, stderr="noise\nInvalid data found\n")
    opts = dataclasses.replace(DEFAULTS, media=str(media))
    with pytest.raises(CLIError) as exc:
        _run(opts, json_mode=False)
    assert exc.value.error_type == "caption_failed"
    assert "Could not write talk.captioned.mp4" in exc.value.message
    # The last stderr line is the reason ffmpeg gives; earlier noise is dropped.
    assert "Invalid data found" in exc.value.message
    assert "noise" not in exc.value.message
    assert "audio-only media" in (exc.value.suggestion or "")


def test_ffmpeg_silent_failure_reports_exit_code(media, fake_transcribe, monkeypatch):
    record_ffmpeg(monkeypatch, returncode=3)
    opts = dataclasses.replace(DEFAULTS, media=str(media))
    with pytest.raises(CLIError) as exc:
        _run(opts, json_mode=False)
    assert "ffmpeg exited with code 3" in exc.value.message


# --- YouTube / media-page sources ----------------------------------------------

YT_URL = "https://www.youtube.com/watch?v=abc123"


@pytest.fixture
def fake_download(monkeypatch: pytest.MonkeyPatch):
    """Stand in for yt-dlp: 'download' a fixed video file into the temp dir."""
    seen: dict[str, object] = {}

    def download(url, dest_dir, *, video=False, download_sections=None):
        seen["url"] = url
        seen["video"] = video
        seen["download_sections"] = download_sections
        seen["dest_dir"] = dest_dir
        path = dest_dir / "vid123.mp4"
        path.write_bytes(b"\x00video")
        seen["path"] = path
        return path

    monkeypatch.setattr(youtube, "download_media", download)
    return seen


def test_run_caption_youtube_downloads_the_full_video(
    tmp_path, fake_download, fake_transcribe, fake_ffmpeg, capsys, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    opts = dataclasses.replace(DEFAULTS, media=YT_URL)
    _run(opts, json_mode=True)
    # Captions are burned into the picture, so the download is always the video,
    # never a section slice, into the command's own source temp dir.
    assert fake_download["url"] == YT_URL
    assert fake_download["video"] is True
    assert fake_download["download_sections"] is None
    assert Path(fake_download["dest_dir"]).name.startswith("aai-caption-src-")
    assert fake_transcribe["audio"] == str(fake_download["path"])
    # ffmpeg reads the downloaded temp file; the default output lands in the cwd,
    # named after the download (the temp dir is gone after the run).
    out = tmp_path / "vid123.captioned.mp4"
    assert fake_ffmpeg["args"][6] == str(fake_download["path"])
    assert fake_ffmpeg["args"][-1] == str(out)
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == YT_URL
    assert payload["out"] == str(out)


def test_run_caption_youtube_status_messages(
    tmp_path, fake_download, fake_transcribe, fake_ffmpeg, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    messages: list[str] = []

    @contextlib.contextmanager
    def fake_status(message, *, json_mode, quiet):
        messages.append(message)
        yield

    monkeypatch.setattr(caption_exec.output, "status", fake_status)
    _run(dataclasses.replace(DEFAULTS, media=YT_URL), json_mode=False)
    assert messages == [
        "Downloading video…",
        "Transcribing for captions…",
        "Fetching captions…",
        "Burning captions…",
    ]


def test_run_caption_youtube_honors_explicit_out(
    tmp_path, fake_download, fake_transcribe, fake_ffmpeg
):
    out = tmp_path / "with-captions.mp4"
    opts = dataclasses.replace(DEFAULTS, media=YT_URL, out=out)
    _run(opts, json_mode=True)
    assert fake_ffmpeg["args"][-1] == str(out)
