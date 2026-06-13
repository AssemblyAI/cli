"""Faked end-to-end runs of the `assembly dub` pipeline (aai_cli/dub_exec.py):
the transcribe → translate → synthesize → ffmpeg mux orchestration, voice
assignment, and the failure modes of each boundary. The LLM Gateway, streaming
TTS, and ffmpeg are faked at the modules dub_exec calls into (`llm.complete`,
`session.synthesize`, `client.transcribe`) and at `dub_exec._run_ffmpeg`; the
pure helpers and validation order live in test_dub_exec.py."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from aai_cli import client, dub_exec, llm, youtube
from aai_cli.context import AppState
from aai_cli.errors import APIError, CLIError, UsageError
from aai_cli.tts import session
from aai_cli.tts.session import SpeakResult
from tests._dub_helpers import (
    DEFAULTS,
    SAMPLE_RATE,
    completion,
    enable_sandbox,
    fake_transcript,
    patch_api_key,
    plain,
    record_ffmpeg,
    record_synthesize,
    record_transcribe,
    record_translate,
    utterance,
    write_media,
)


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


def _run(opts, *, json_mode):
    dub_exec.run_dub(opts, AppState(), json_mode=json_mode)


def test_run_dub_pipeline_end_to_end(
    media, fake_transcribe, fake_translate, fake_synthesize, fake_ffmpeg, capsys
):
    opts = dataclasses.replace(DEFAULTS, media=str(media))
    _run(opts, json_mode=True)

    # Transcription: the local file, diarized so speakers keep distinct voices.
    assert fake_transcribe["audio"] == str(media)
    assert fake_transcribe["config"].speaker_labels is True

    # Translation: one gateway call per utterance, in order, with the dubbing
    # system prompt naming the resolved language ("de" -> "German").
    assert [c["messages"][-1]["content"] for c in fake_translate] == ["Hello.", "World."]
    for call in fake_translate:
        assert call["model"] == llm.DEFAULT_MODEL
        assert call["max_tokens"] == llm.DEFAULT_MAX_TOKENS
        system = call["messages"][0]
        assert system["role"] == "system"
        assert "dubbing" in system["content"]
        assert "German" in system["content"]

    # Synthesis: the translated text in the target language, every speaker on
    # German's one native voice (the language selects the voice).
    assert [(cfg.voice, cfg.text) for cfg in fake_synthesize] == [
        ("juergen", "DE:Hello."),
        ("juergen", "DE:World."),
    ]
    assert all(cfg.language == "German" for cfg in fake_synthesize)

    # The dubbed track: silence to 1.0 s, segment 1, silence to 3.0 s, segment 2,
    # then a tail pad out to the source's 5 s duration (rate 100 -> 200 bytes/s).
    expected_track = b"\x00" * 200 + b"\xa1" * 100 + b"\x00" * 300 + b"\xa2" * 100 + b"\x00" * 300
    assert fake_ffmpeg["wav_frames"] == expected_track
    params = fake_ffmpeg["wav_params"]
    assert (params.nchannels, params.sampwidth, params.framerate) == (1, 2, SAMPLE_RATE)

    # The mux: video copied, WAV swapped in as the only audio, default out path.
    out = media.parent / "talk.dub.german.mp4"
    wav_path = fake_ffmpeg["args"][8]
    assert fake_ffmpeg["args"] == [
        "/usr/bin/ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(media),
        "-i",
        wav_path,
        "-map",
        "0:v?",
        "-map",
        "1:a",
        "-c:v",
        "copy",
        str(out),
    ]
    assert wav_path.endswith("dub.wav")

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "source": str(media),
        "out": str(out),
        "language": "German",
        "transcript_id": "tr_dub",
        "utterances": 2,
        "speakers": {"A": "juergen", "B": "juergen"},
        "sample_rate": SAMPLE_RATE,
        "audio_duration_seconds": 5.0,
    }


def test_run_dub_human_summary(
    media, fake_transcribe, fake_translate, fake_synthesize, fake_ffmpeg, capsys
):
    # A short --out keeps the one-line summary under the 80-column console width:
    # with the default (tmp_path-prefixed) out path, Rich would hard-wrap the line
    # mid-word and these substring asserts would depend on where the break lands.
    opts = dataclasses.replace(DEFAULTS, media=str(media), out=Path("dub.de.mp4"))
    _run(opts, json_mode=False)
    # plain(): under FORCE_COLOR (CI) Rich's repr highlighter interleaves style
    # codes inside the line ("(2 utterances" renders with the 2 colored).
    out = plain(capsys.readouterr().out)
    assert "dub.de.mp4" in out
    assert "dubbed to German" in out
    assert "2 utterances" in out
    assert "A=juergen, B=juergen" in out


def test_bare_voice_dubs_every_speaker(
    media, fake_transcribe, fake_translate, fake_synthesize, fake_ffmpeg
):
    opts = dataclasses.replace(DEFAULTS, media=str(media), voice=["paul"])
    _run(opts, json_mode=True)
    assert [cfg.voice for cfg in fake_synthesize] == ["paul", "paul"]


def test_voice_overrides_pin_speakers_without_consuming_rotation(
    media, fake_transcribe, fake_translate, fake_synthesize, fake_ffmpeg
):
    opts = dataclasses.replace(DEFAULTS, media=str(media), voice=["A=mary"])
    _run(opts, json_mode=True)
    # A is pinned; B still takes German's native voice from the rotation.
    assert [cfg.voice for cfg in fake_synthesize] == ["mary", "juergen"]


def test_english_dub_keeps_the_multi_voice_rotation(
    media, fake_transcribe, fake_translate, fake_synthesize, fake_ffmpeg
):
    # English has many voices, so speakers still rotate through the curated set
    # instead of collapsing onto one voice.
    opts = dataclasses.replace(DEFAULTS, media=str(media), language="en")
    _run(opts, json_mode=True)
    assert [cfg.voice for cfg in fake_synthesize] == ["jane", "michael"]


def test_language_without_a_native_voice_falls_back_to_english_rotation(
    media, fake_transcribe, fake_translate, fake_synthesize, fake_ffmpeg
):
    # Japanese is translatable but has no catalog voice: the dub still runs,
    # on the English rotation.
    opts = dataclasses.replace(DEFAULTS, media=str(media), language="ja")
    _run(opts, json_mode=True)
    assert [cfg.voice for cfg in fake_synthesize] == ["jane", "michael"]
    assert all(cfg.language == "Japanese" for cfg in fake_synthesize)


def test_transcript_id_reuses_existing_transcript(
    media, fake_translate, fake_ffmpeg, monkeypatch, capsys
):
    fetched: dict[str, str] = {}

    def get_transcript(api_key, transcript_id):
        fetched["id"] = transcript_id
        return SimpleNamespace(
            id=transcript_id,
            utterances=[utterance(0, "A", "Hello.")],
            audio_duration=None,  # duration unknown -> no tail pad
        )

    monkeypatch.setattr(client, "get_transcript", get_transcript)
    monkeypatch.setattr(
        client,
        "transcribe",
        lambda *a, **k: pytest.fail("must not re-transcribe with --transcript-id"),
    )
    monkeypatch.setattr(
        session,
        "synthesize",
        lambda api_key, cfg, **_: SpeakResult(
            pcm=b"\xaa" * 2000, sample_rate=300, audio_duration_seconds=0.0
        ),
    )

    opts = dataclasses.replace(DEFAULTS, media=str(media), transcript_id="tr_99")
    _run(opts, json_mode=True)
    assert fetched["id"] == "tr_99"
    payload = json.loads(capsys.readouterr().out)
    assert payload["transcript_id"] == "tr_99"
    # 1000 samples at 300 Hz, rounded to milliseconds: 3.3333... -> 3.333.
    assert payload["audio_duration_seconds"] == 3.333


def test_empty_translation_is_an_api_error(media, fake_synthesize, fake_ffmpeg, monkeypatch):
    long_text = "a" * 50 + "TAIL!"
    transcript = fake_transcript([utterance(0, "A", "Hello."), utterance(1000, "B", long_text)])
    monkeypatch.setattr(client, "transcribe", lambda *a, **k: transcript)
    replies = iter(["Hallo.", "   "])
    monkeypatch.setattr(llm, "complete", lambda *a, **k: completion(next(replies)))

    opts = dataclasses.replace(DEFAULTS, media=str(media))
    with pytest.raises(APIError) as exc:
        _run(opts, json_mode=False)
    # The 1-based index and the (50-char) text preview pin which utterance failed.
    assert f"empty translation for utterance 2 ({'a' * 50!r})." in exc.value.message


def test_mixed_sample_rates_are_an_api_error(
    media, fake_transcribe, fake_translate, fake_ffmpeg, monkeypatch
):
    rates = iter([100, 200])
    monkeypatch.setattr(
        session,
        "synthesize",
        lambda api_key, cfg, **_: SpeakResult(
            pcm=b"\x01\x02", sample_rate=next(rates), audio_duration_seconds=0.0
        ),
    )
    opts = dataclasses.replace(DEFAULTS, media=str(media))
    with pytest.raises(APIError) as exc:
        _run(opts, json_mode=False)
    assert "mixed sample rates ([100, 200])" in exc.value.message


def test_ffmpeg_failure_reports_last_stderr_line(
    media, fake_transcribe, fake_translate, fake_synthesize, monkeypatch
):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        dub_exec,
        "_run_ffmpeg",
        lambda args: subprocess.CompletedProcess(
            args=args, returncode=1, stdout="", stderr="noise\nInvalid data found\n"
        ),
    )
    opts = dataclasses.replace(DEFAULTS, media=str(media))
    with pytest.raises(CLIError) as exc:
        _run(opts, json_mode=False)
    assert exc.value.error_type == "dub_failed"
    assert "Could not write talk.dub.german.mp4" in exc.value.message
    # The last stderr line is the reason ffmpeg gives; earlier noise is dropped.
    assert "Invalid data found" in exc.value.message
    assert "noise" not in exc.value.message
    assert "readable audio/video file" in (exc.value.suggestion or "")


def test_ffmpeg_silent_failure_reports_exit_code(
    media, fake_transcribe, fake_translate, fake_synthesize, monkeypatch
):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        dub_exec,
        "_run_ffmpeg",
        lambda args: subprocess.CompletedProcess(args=args, returncode=3, stdout="", stderr=""),
    )
    opts = dataclasses.replace(DEFAULTS, media=str(media))
    with pytest.raises(CLIError) as exc:
        _run(opts, json_mode=False)
    assert "ffmpeg exited with code 3" in exc.value.message


# --- YouTube / media-page sources ----------------------------------------------

YT_URL = "https://www.youtube.com/watch?v=abc123"


@pytest.fixture
def fake_download(monkeypatch: pytest.MonkeyPatch):
    """Stand in for yt-dlp: 'download' a fixed media file into the temp dir."""
    seen: dict[str, object] = {}

    def download(url, dest_dir, *, video=False, download_sections=None):
        seen["url"] = url
        seen["video"] = video
        seen["download_sections"] = download_sections
        path = dest_dir / ("vid123.mp4" if video else "vid123.m4a")
        path.write_bytes(b"\x00media")
        seen["path"] = path
        return path

    monkeypatch.setattr(youtube, "download_media", download)
    return seen


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
    assert "Download the media first" in (exc.value.suggestion or "")
