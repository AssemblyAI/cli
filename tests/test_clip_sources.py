"""Tests for `assembly clip`'s alternative sources and LLM-driven selection:
YouTube/media-page downloads, the `-t -` stdin transcript pipe, and `--llm`
segment selection through the LLM Gateway (all boundaries faked)."""

from __future__ import annotations

import contextlib
import dataclasses
import json
from pathlib import Path

import pytest

from aai_cli import client, config
from aai_cli.commands.clip import _exec as clip_exec
from aai_cli.commands.clip import _select as clip_select
from aai_cli.context import AppState
from aai_cli.errors import CLIError, UsageError
from tests._clip_helpers import DEFAULTS, UTTERANCES, fake_transcript, record_ffmpeg


@pytest.fixture
def media(tmp_path: Path) -> Path:
    path = tmp_path / "meeting.mp4"
    path.write_bytes(b"\x00fake-media")
    return path


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    return record_ffmpeg(monkeypatch)


# --- YouTube / media-page sources ---------------------------------------------


@pytest.fixture
def fake_download(monkeypatch):
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

    monkeypatch.setattr(clip_exec.youtube, "download_media", download)
    return seen


YT_URL = "https://www.youtube.com/watch?v=abc123"


def test_run_clip_downloads_youtube_audio_into_cwd(
    tmp_path, fake_ffmpeg, fake_download, capsys, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    opts = dataclasses.replace(DEFAULTS, media=YT_URL, ranges=["1-2"])
    clip_exec.run_clip(opts, AppState(), json_mode=True)
    assert fake_download["url"] == YT_URL
    # No section slicing, into the command's own source temp dir.
    assert fake_download["download_sections"] is None
    assert Path(fake_download["dest_dir"]).name.startswith("aai-clip-src-")
    # ffmpeg reads the downloaded temp file; the clip lands in the cwd, named
    # after the download (the temp dir is gone after the run).
    assert fake_ffmpeg[1][6] == str(fake_download["path"])
    dest = tmp_path / "vid123.clip01.m4a"
    assert fake_ffmpeg[1][-1] == str(dest)
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == YT_URL
    assert payload["clips"][0]["path"] == str(dest)


def test_run_clip_youtube_honors_out_dir(tmp_path, fake_ffmpeg, fake_download, capsys):
    out_dir = tmp_path / "clips"
    out_dir.mkdir()
    opts = dataclasses.replace(DEFAULTS, media=YT_URL, ranges=["1-2"], out_dir=out_dir)
    clip_exec.run_clip(opts, AppState(), json_mode=True)
    assert fake_ffmpeg[1][-1] == str(out_dir / "vid123.clip01.m4a")


def test_run_clip_youtube_transcribes_the_downloaded_file(
    tmp_path, fake_ffmpeg, fake_download, capsys, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config.set_api_key("default", "sk_test")
    seen = {}

    def fake_transcribe(api_key, audio, *, config):
        seen["audio"] = audio
        return fake_transcript(list(UTTERANCES))

    monkeypatch.setattr(client, "transcribe", fake_transcribe)
    monkeypatch.setattr(
        clip_exec.llm, "transform_transcript", lambda *a, **k: '[{"start": 1, "end": 2}]'
    )
    opts = dataclasses.replace(DEFAULTS, media=YT_URL, llm_prompt="best moment")
    clip_exec.run_clip(opts, AppState(), json_mode=True)
    assert seen["audio"] == str(fake_download["path"])
    payload = json.loads(capsys.readouterr().out)
    assert [(c["start"], c["end"]) for c in payload["clips"]] == [(1.0, 2.0)]


def test_run_clip_youtube_download_status_message(
    tmp_path, fake_ffmpeg, fake_download, capsys, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    messages: list[str] = []

    @contextlib.contextmanager
    def fake_status(message, *, json_mode, quiet):
        messages.append(message)
        yield

    monkeypatch.setattr(clip_exec.output, "status", fake_status)
    opts = dataclasses.replace(DEFAULTS, media=YT_URL, ranges=["1-2"])
    clip_exec.run_clip(opts, AppState(), json_mode=False)
    # Without --video only the audio track is fetched.
    assert fake_download["video"] is False
    assert messages == ["Downloading audio…", "Detecting silence…", "Cutting 1 clip(s)…"]


def test_run_clip_video_downloads_the_full_video(
    tmp_path, fake_ffmpeg, fake_download, capsys, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    messages: list[str] = []

    @contextlib.contextmanager
    def fake_status(message, *, json_mode, quiet):
        messages.append(message)
        yield

    monkeypatch.setattr(clip_exec.output, "status", fake_status)
    opts = dataclasses.replace(DEFAULTS, media=YT_URL, ranges=["1-2"], video=True)
    clip_exec.run_clip(opts, AppState(), json_mode=True)
    # --video fetches the full video, and the clips carry its container/extension.
    assert fake_download["video"] is True
    assert messages[0] == "Downloading video…"
    assert fake_ffmpeg[1][-1] == str(tmp_path / "vid123.clip01.mp4")
    payload = json.loads(capsys.readouterr().out)
    assert payload["clips"][0]["path"] == str(tmp_path / "vid123.clip01.mp4")


def test_run_clip_video_requires_a_url_source(media, fake_ffmpeg):
    # A local file already carries its video into every clip, so --video would be
    # a silent no-op — it is rejected instead.
    opts = dataclasses.replace(DEFAULTS, media=str(media), ranges=["1-2"], video=True)
    with pytest.raises(UsageError) as exc:
        clip_exec.run_clip(opts, AppState(), json_mode=False)
    assert "--video only applies to a downloadable URL source" in exc.value.message


# --- transcript piped on stdin (-t -) -------------------------------------------


def _piped_payload():
    return json.dumps(
        {
            "id": "tr_piped",
            "utterances": [
                {"start": 1500, "end": 2500, "speaker": "A", "text": "Let's talk pricing."},
                {"start": 3000, "end": 4000, "speaker": "B", "text": "Sounds good."},
            ],
        }
    )


def test_run_clip_reads_transcript_json_from_stdin(media, fake_ffmpeg, capsys, monkeypatch):
    # No API key configured and no client call: the piped JSON is the transcript.
    monkeypatch.setattr(clip_exec.stdio, "piped_stdin_text", _piped_payload)
    monkeypatch.setattr(
        client,
        "get_transcript",
        lambda *a: pytest.fail("must not fetch when JSON is piped"),
    )
    opts = dataclasses.replace(DEFAULTS, media=str(media), transcript_id="-", speakers=["A"])
    clip_exec.run_clip(opts, AppState(), json_mode=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["transcript_id"] == "tr_piped"
    assert [(c["start"], c["end"]) for c in payload["clips"]] == [(1.5, 2.5)]


def test_run_clip_reads_transcript_id_from_stdin(media, fake_ffmpeg, capsys, monkeypatch):
    config.set_api_key("default", "sk_test")
    monkeypatch.setattr(clip_exec.stdio, "piped_stdin_text", lambda: "tr_999\n")
    seen = {}

    def fake_get(api_key, transcript_id):
        seen["args"] = (api_key, transcript_id)
        return fake_transcript(list(UTTERANCES))

    monkeypatch.setattr(client, "get_transcript", fake_get)
    opts = dataclasses.replace(DEFAULTS, media=str(media), transcript_id="-", speakers=["B"])
    clip_exec.run_clip(opts, AppState(), json_mode=True)
    assert seen["args"] == ("sk_test", "tr_999")


def test_run_clip_stdin_transcript_requires_piped_input(media, fake_ffmpeg, monkeypatch):
    monkeypatch.setattr(clip_exec.stdio, "piped_stdin_text", lambda: None)
    opts = dataclasses.replace(DEFAULTS, media=str(media), transcript_id="-", speakers=["A"])
    with pytest.raises(UsageError) as exc:
        clip_exec.run_clip(opts, AppState(), json_mode=False)
    assert "-t - expects a transcript id or transcript JSON on stdin" in exc.value.message
    assert "assembly clip <file> -t -" in (exc.value.suggestion or "")


def test_run_clip_stdin_transcript_rejects_bad_json(media, fake_ffmpeg, monkeypatch):
    monkeypatch.setattr(clip_exec.stdio, "piped_stdin_text", lambda: '{"id": ')
    opts = dataclasses.replace(DEFAULTS, media=str(media), transcript_id="-", speakers=["A"])
    with pytest.raises(UsageError) as exc:
        clip_exec.run_clip(opts, AppState(), json_mode=False)
    assert "Couldn't parse the transcript JSON on stdin" in exc.value.message


# --- LLM-driven selection -----------------------------------------------------


def test_run_clip_llm_selection_drives_the_cut(media, fake_ffmpeg, capsys, monkeypatch):
    config.set_api_key("default", "sk_test")
    monkeypatch.setattr(client, "transcribe", lambda *a, **k: fake_transcript(list(UTTERANCES)))
    seen = {}

    def fake_transform(api_key, *, prompt, transcript_text, model, max_tokens):
        seen.update(
            api_key=api_key,
            prompt=prompt,
            transcript_text=transcript_text,
            model=model,
            max_tokens=max_tokens,
        )
        return ' [{"start": 1.5, "end": 4.0}] '

    monkeypatch.setattr(clip_exec.llm, "transform_transcript", fake_transform)
    opts = dataclasses.replace(
        DEFAULTS,
        media=str(media),
        llm_prompt="the pricing discussion",
        model="gpt-5",
        max_tokens=64,
    )
    clip_exec.run_clip(opts, AppState(), json_mode=True)
    assert seen["api_key"] == "sk_test"
    # The reply contract is prefixed; the user's instruction closes the prompt.
    assert "Reply with only a JSON array" in seen["prompt"]
    assert seen["prompt"].endswith("Selection instruction: the pricing discussion")
    assert seen["transcript_text"] == clip_select.utterance_listing(list(UTTERANCES))
    assert seen["model"] == "gpt-5"
    assert seen["max_tokens"] == 64
    payload = json.loads(capsys.readouterr().out)
    assert payload["transcript_id"] == "tr_123"
    assert [(c["start"], c["end"]) for c in payload["clips"]] == [(1.5, 4.0)]
    assert fake_ffmpeg[1][7:11] == ["-ss", "1.500", "-to", "4.000"]


def test_run_clip_llm_composes_with_speaker_filter(media, fake_ffmpeg, capsys, monkeypatch):
    # --speaker narrows the utterances first; the LLM only sees what survived.
    config.set_api_key("default", "sk_test")
    monkeypatch.setattr(client, "transcribe", lambda *a, **k: fake_transcript(list(UTTERANCES)))
    seen = {}

    def fake_transform(api_key, *, prompt, transcript_text, model, max_tokens):
        seen["transcript_text"] = transcript_text
        return '[{"start": 5.0, "end": 6.0}]'

    monkeypatch.setattr(clip_exec.llm, "transform_transcript", fake_transform)
    opts = dataclasses.replace(DEFAULTS, media=str(media), speakers=["A"], llm_prompt="hiring talk")
    clip_exec.run_clip(opts, AppState(), json_mode=True)
    assert "B: Sounds good." not in seen["transcript_text"]
    assert "A: Moving on to hiring." in seen["transcript_text"]
    payload = json.loads(capsys.readouterr().out)
    assert [(c["start"], c["end"]) for c in payload["clips"]] == [(5.0, 6.0)]


def test_run_clip_llm_works_with_transcript_id(media, fake_ffmpeg, capsys, monkeypatch):
    # -t with --llm alone is a valid selection (no --speaker/--search needed).
    config.set_api_key("default", "sk_test")
    monkeypatch.setattr(client, "get_transcript", lambda *a: fake_transcript(list(UTTERANCES)))
    monkeypatch.setattr(
        clip_exec.llm,
        "transform_transcript",
        lambda *a, **k: '[{"start": 3.0, "end": 4.0}]',
    )
    opts = dataclasses.replace(DEFAULTS, media=str(media), transcript_id="tr_123", llm_prompt="x")
    clip_exec.run_clip(opts, AppState(), json_mode=True)
    payload = json.loads(capsys.readouterr().out)
    assert [(c["start"], c["end"]) for c in payload["clips"]] == [(3.0, 4.0)]


def test_run_clip_llm_parse_error_surfaces(media, fake_ffmpeg, monkeypatch):
    config.set_api_key("default", "sk_test")
    monkeypatch.setattr(client, "transcribe", lambda *a, **k: fake_transcript(list(UTTERANCES)))
    monkeypatch.setattr(clip_exec.llm, "transform_transcript", lambda *a, **k: "no json, sorry")
    opts = dataclasses.replace(DEFAULTS, media=str(media), llm_prompt="x")
    with pytest.raises(CLIError) as exc:
        clip_exec.run_clip(opts, AppState(), json_mode=False)
    assert exc.value.error_type == "llm_parse_error"


def test_run_clip_llm_status_message_names_the_model(media, fake_ffmpeg, monkeypatch):
    config.set_api_key("default", "sk_test")
    monkeypatch.setattr(client, "transcribe", lambda *a, **k: fake_transcript(list(UTTERANCES)))
    monkeypatch.setattr(
        clip_exec.llm, "transform_transcript", lambda *a, **k: '[{"start": 1, "end": 2}]'
    )
    messages: list[str] = []

    @contextlib.contextmanager
    def fake_status(message, *, json_mode, quiet):
        messages.append(message)
        yield

    monkeypatch.setattr(clip_exec.output, "status", fake_status)
    opts = dataclasses.replace(DEFAULTS, media=str(media), llm_prompt="best bits", model="gpt-5")
    clip_exec.run_clip(opts, AppState(), json_mode=False)
    assert messages == [
        "Transcribing for clip selection…",
        "Selecting segments with gpt-5…",
        "Detecting silence…",
        "Cutting 1 clip(s)…",
    ]
