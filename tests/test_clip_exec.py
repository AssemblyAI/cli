"""Direct tests of the `assembly clip` options/run seam (aai_cli/clip_exec.py):
validation, ffmpeg orchestration, and transcript-backed --speaker/--search
selection. Constructed-options tests (dataclasses.replace off the shared
defaults) avoid any argv round-trip; the ffmpeg boundary is faked at
`mediafile.run_ffmpeg`. The pure selection logic is covered in
test_clip_select.py; YouTube/stdin/LLM sources in test_clip_sources.py."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from aai_cli import client, clip_exec, config, mediafile
from aai_cli.clip_select import Segment
from aai_cli.context import AppState
from aai_cli.errors import CLIError, UsageError
from tests._clip_helpers import (
    DEFAULTS,
    UTTERANCES,
    fake_transcript,
    plain,
    record_ffmpeg,
    utterance,
)


@pytest.fixture
def media(tmp_path: Path) -> Path:
    path = tmp_path / "meeting.mp4"
    path.write_bytes(b"\x00fake-media")
    return path


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    return record_ffmpeg(monkeypatch)


def test_options_are_immutable():
    field_name = dataclasses.fields(DEFAULTS)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(DEFAULTS, field_name, None)


@pytest.mark.parametrize(
    "instance",
    [
        clip_exec.WrittenClip(path=Path("x.mp4"), segment=Segment(0.0, 1.0)),
        clip_exec._PipedTranscript(id="tr_1", utterances=[]),
    ],
    ids=["written_clip", "piped_transcript"],
)
def test_result_records_are_immutable(instance):
    field_name = dataclasses.fields(instance)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field_name, None)


# --- validation errors -------------------------------------------------------


def test_run_clip_rejects_missing_file(tmp_path):
    opts = dataclasses.replace(DEFAULTS, media=str(tmp_path / "nope.mp4"), ranges=["1-2"])
    with pytest.raises(CLIError) as exc:
        clip_exec.run_clip(opts, AppState(), json_mode=False)
    assert exc.value.error_type == "file_not_found"
    assert exc.value.exit_code == 2
    assert "File not found" in exc.value.message
    # The command name pins the shared helper's parameterization.
    assert "assembly clip needs a local audio/video file" in (exc.value.suggestion or "")


def test_run_clip_rejects_directory(tmp_path):
    opts = dataclasses.replace(DEFAULTS, media=str(tmp_path), ranges=["1-2"])
    with pytest.raises(CLIError) as exc:
        clip_exec.run_clip(opts, AppState(), json_mode=False)
    assert exc.value.error_type == "not_a_file"
    assert exc.value.exit_code == 2
    assert "Not a file" in exc.value.message
    assert "not a directory" in (exc.value.suggestion or "")


def test_run_clip_rejects_non_downloadable_url():
    opts = dataclasses.replace(DEFAULTS, media="https://x.test/a.mp4", ranges=["1-2"])
    with pytest.raises(UsageError) as exc:
        clip_exec.run_clip(opts, AppState(), json_mode=False)
    assert "can't fetch this URL" in exc.value.message
    assert "Download the media first" in (exc.value.suggestion or "")


def test_run_clip_requires_a_selector(media):
    with pytest.raises(UsageError) as exc:
        clip_exec.run_clip(
            dataclasses.replace(DEFAULTS, media=str(media)), AppState(), json_mode=False
        )
    assert "Nothing selects a segment" in exc.value.message
    assert "--range" in (exc.value.suggestion or "")


def test_run_clip_rejects_transcript_id_without_a_transcript_selector(media):
    opts = dataclasses.replace(DEFAULTS, media=str(media), transcript_id="tr_1", ranges=["1-2"])
    with pytest.raises(UsageError) as exc:
        clip_exec.run_clip(opts, AppState(), json_mode=False)
    assert "--transcript-id only applies with --speaker/--search" in exc.value.message
    assert "drop --transcript-id" in (exc.value.suggestion or "")


def test_run_clip_rejects_missing_out_dir(media, tmp_path):
    opts = dataclasses.replace(
        DEFAULTS, media=str(media), ranges=["1-2"], out_dir=tmp_path / "missing"
    )
    with pytest.raises(UsageError) as exc:
        clip_exec.run_clip(opts, AppState(), json_mode=False)
    assert "--out-dir doesn't exist" in exc.value.message
    assert "Create it first" in (exc.value.suggestion or "")


def test_run_clip_requires_ffmpeg(media, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    opts = dataclasses.replace(DEFAULTS, media=str(media), ranges=["1-2"])
    with pytest.raises(CLIError) as exc:
        clip_exec.run_clip(opts, AppState(), json_mode=False)
    assert exc.value.error_type == "missing_dependency"
    # The purpose string pins the shared helper's parameterization.
    assert "ffmpeg is required to cut media" in exc.value.message
    assert "Install it" in (exc.value.suggestion or "")


# --- range-only cutting (no transcript, no network) --------------------------


def test_run_clip_range_only_cuts_and_emits_json(media, fake_ffmpeg, capsys):
    opts = dataclasses.replace(DEFAULTS, media=str(media), ranges=["5-12.5"])
    clip_exec.run_clip(opts, AppState(), json_mode=True)
    dest = media.parent / "meeting.clip01.mp4"
    assert fake_ffmpeg == [
        [
            "/usr/bin/ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(media),
            "-af",
            "silencedetect=noise=-30dB:d=0.2",
            "-f",
            "null",
            "-",
        ],
        [
            "/usr/bin/ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(media),
            "-ss",
            "5.000",
            "-to",
            "12.500",
            str(dest),
        ],
    ]
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "source": str(media),
        "transcript_id": None,
        "clips": [{"path": str(dest), "start": 5.0, "end": 12.5, "duration": 7.5}],
    }


def test_run_clip_human_mode_prints_one_line_per_clip(tmp_path, fake_ffmpeg, capsys, monkeypatch):
    # A relative source keeps each rendered line under the 80-column console
    # width — an absolute tmp_path would wrap and split the asserted text.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "meeting.mp4").write_bytes(b"\x00fake-media")
    opts = dataclasses.replace(DEFAULTS, media="meeting.mp4", ranges=["5-12.5678", "90-100"])
    clip_exec.run_clip(opts, AppState(), json_mode=False)
    out = plain(capsys.readouterr().out)
    assert "meeting.clip01.mp4" in out
    assert "0:05.0 - 0:12.6" in out
    # The duration rounds at 3 decimals (7.5678 -> 7.568).
    assert "(7.568s)" in out
    assert "meeting.clip02.mp4" in out
    assert "1:30.0 - 1:40.0" in out
    # Human mode prints lines, not a JSON object.
    assert not out.lstrip().startswith("{")


def test_run_clip_applies_padding_to_explicit_ranges(media, fake_ffmpeg, capsys):
    opts = dataclasses.replace(DEFAULTS, media=str(media), ranges=["5-10"], padding=1.0)
    clip_exec.run_clip(opts, AppState(), json_mode=True)
    assert fake_ffmpeg[1][7:11] == ["-ss", "4.000", "-to", "11.000"]
    clips = json.loads(capsys.readouterr().out)["clips"]
    assert (clips[0]["start"], clips[0]["end"]) == (4.0, 11.0)


def test_run_clip_rounds_payload_times_to_milliseconds(media, fake_ffmpeg, capsys):
    # Both endpoints carry sub-millisecond noise, so every rounded field
    # (start, end, duration) actually changes at 3 decimal places.
    opts = dataclasses.replace(DEFAULTS, media=str(media), ranges=["0.1234-1.5678"])
    clip_exec.run_clip(opts, AppState(), json_mode=True)
    clips = json.loads(capsys.readouterr().out)["clips"]
    assert clips[0] == {
        "path": str(media.parent / "meeting.clip01.mp4"),
        "start": 0.123,
        "end": 1.568,
        "duration": 1.444,
    }


def test_run_clip_honors_out_dir(media, tmp_path, fake_ffmpeg, capsys):
    out_dir = tmp_path / "clips"
    out_dir.mkdir()
    opts = dataclasses.replace(DEFAULTS, media=str(media), ranges=["1-2"], out_dir=out_dir)
    clip_exec.run_clip(opts, AppState(), json_mode=True)
    dest = out_dir / "meeting.clip01.mp4"
    assert fake_ffmpeg[1][-1] == str(dest)
    assert json.loads(capsys.readouterr().out)["clips"][0]["path"] == str(dest)


# --- silence snapping ---------------------------------------------------------

DETECT_LOG = (
    "[silencedetect @ 0x1] silence_start: 4\n"
    "[silencedetect @ 0x1] silence_end: 4.6 | silence_duration: 0.6\n"
    "[silencedetect @ 0x1] silence_start: 13\n"
    "[silencedetect @ 0x1] silence_end: 14 | silence_duration: 1.0\n"
)


def test_run_clip_snaps_boundaries_into_detected_silence(media, capsys, monkeypatch):
    # Both 5.0 and 12.5 land on speech: the start snaps back into the 4.0-4.6
    # silence (0.25 before speech resumes at 4.6), the end snaps forward into
    # the 13.0-14.0 silence (0.25 past where speech stops at 13.0).
    calls = record_ffmpeg(monkeypatch, detect_log=DETECT_LOG)
    opts = dataclasses.replace(DEFAULTS, media=str(media), ranges=["5-12.5"])
    clip_exec.run_clip(opts, AppState(), json_mode=True)
    assert calls[1][7:11] == ["-ss", "4.350", "-to", "13.250"]
    clips = json.loads(capsys.readouterr().out)["clips"]
    assert (clips[0]["start"], clips[0]["end"]) == (4.35, 13.25)


def test_run_clip_failed_silence_detection_cuts_at_selected_times(media, capsys, monkeypatch):
    # Snapping is best-effort: a broken silencedetect pass must not fail (or
    # shift) the cut.
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")

    def run(args: list[str]) -> subprocess.CompletedProcess[str]:
        if "-af" in args:
            return subprocess.CompletedProcess(
                args=args, returncode=1, stdout="", stderr=DETECT_LOG
            )
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mediafile, "run_ffmpeg", run)
    opts = dataclasses.replace(DEFAULTS, media=str(media), ranges=["5-12.5"])
    clip_exec.run_clip(opts, AppState(), json_mode=True)
    clips = json.loads(capsys.readouterr().out)["clips"]
    assert (clips[0]["start"], clips[0]["end"]) == (5.0, 12.5)


def test_run_clip_no_snap_skips_detection(media, capsys, monkeypatch):
    calls = record_ffmpeg(monkeypatch, detect_log=DETECT_LOG)
    opts = dataclasses.replace(DEFAULTS, media=str(media), ranges=["5-12.5"], snap=False)
    clip_exec.run_clip(opts, AppState(), json_mode=True)
    # Exactly one ffmpeg call: the cut, at the exact selected times.
    assert len(calls) == 1
    assert "-af" not in calls[0]
    assert calls[0][7:11] == ["-ss", "5.000", "-to", "12.500"]
    clips = json.loads(capsys.readouterr().out)["clips"]
    assert (clips[0]["start"], clips[0]["end"]) == (5.0, 12.5)


def test_run_clip_surfaces_ffmpeg_failure(media, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")

    def fail(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args, returncode=1, stdout="", stderr="noise\nInvalid data found\n"
        )

    monkeypatch.setattr(mediafile, "run_ffmpeg", fail)
    opts = dataclasses.replace(DEFAULTS, media=str(media), ranges=["1-2"])
    with pytest.raises(CLIError) as exc:
        clip_exec.run_clip(opts, AppState(), json_mode=False)
    assert exc.value.error_type == "clip_failed"
    assert "Could not cut meeting.clip01.mp4" in exc.value.message
    # The last stderr line is the reason ffmpeg gives; earlier noise is dropped.
    assert "Invalid data found" in exc.value.message
    assert "noise" not in exc.value.message
    assert "readable audio/video file" in (exc.value.suggestion or "")


def test_run_clip_reports_exit_code_when_ffmpeg_is_silent(media, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        mediafile,
        "run_ffmpeg",
        lambda args: subprocess.CompletedProcess(args=args, returncode=3, stdout="", stderr=""),
    )
    opts = dataclasses.replace(DEFAULTS, media=str(media), ranges=["1-2"])
    with pytest.raises(CLIError) as exc:
        clip_exec.run_clip(opts, AppState(), json_mode=False)
    assert "ffmpeg exited with code 3" in exc.value.message


def test_run_ffmpeg_captures_output_and_does_not_raise():
    # The real boundary (not the fake): output is captured as text and a non-zero
    # exit must not raise — the callers turn the exit code into a CLIError.
    result = mediafile.run_ffmpeg(
        [
            sys.executable,
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)",
        ]
    )
    assert result.returncode == 3
    assert result.stdout == "out\n"
    assert result.stderr == "err\n"


# --- transcript-backed selection ---------------------------------------------


def test_run_clip_transcribes_with_speaker_labels(media, fake_ffmpeg, capsys, monkeypatch):
    config.set_api_key("default", "sk_test")
    seen = {}

    def fake_transcribe(api_key, audio, *, config):
        seen["api_key"] = api_key
        seen["audio"] = audio
        seen["config"] = config
        return fake_transcript(list(UTTERANCES))

    monkeypatch.setattr(client, "transcribe", fake_transcribe)
    opts = dataclasses.replace(DEFAULTS, media=str(media), speakers=["a"])
    clip_exec.run_clip(opts, AppState(), json_mode=True)
    assert seen["api_key"] == "sk_test"
    assert seen["audio"] == str(media)
    assert seen["config"].speaker_labels is True
    payload = json.loads(capsys.readouterr().out)
    assert payload["transcript_id"] == "tr_123"
    # Speaker A's two utterances: 1.5-2.5s and 5-6s.
    assert [(c["start"], c["end"]) for c in payload["clips"]] == [(1.5, 2.5), (5.0, 6.0)]
    assert fake_ffmpeg[1][-1] == str(media.parent / "meeting.clip01.mp4")
    assert fake_ffmpeg[2][-1] == str(media.parent / "meeting.clip02.mp4")


def test_run_clip_reuses_transcript_by_id(media, fake_ffmpeg, capsys, monkeypatch):
    config.set_api_key("default", "sk_test")
    seen = {}

    def fake_get(api_key, transcript_id):
        seen["args"] = (api_key, transcript_id)
        return fake_transcript(list(UTTERANCES))

    monkeypatch.setattr(client, "get_transcript", fake_get)
    monkeypatch.setattr(
        client,
        "transcribe",
        lambda *a, **k: pytest.fail("must not re-transcribe when -t is given"),
    )
    opts = dataclasses.replace(DEFAULTS, media=str(media), transcript_id="tr_123", search="pricing")
    clip_exec.run_clip(opts, AppState(), json_mode=True)
    assert seen["args"] == ("sk_test", "tr_123")
    payload = json.loads(capsys.readouterr().out)
    assert [(c["start"], c["end"]) for c in payload["clips"]] == [(1.5, 2.5)]


def test_run_clip_merges_transcript_matches_with_explicit_ranges(
    media, fake_ffmpeg, capsys, monkeypatch
):
    config.set_api_key("default", "sk_test")
    utterances = [utterance(5000, 8000, "A", "hello")]
    monkeypatch.setattr(client, "transcribe", lambda *a, **k: fake_transcript(utterances))
    opts = dataclasses.replace(DEFAULTS, media=str(media), speakers=["A"], ranges=["7-12"])
    clip_exec.run_clip(opts, AppState(), json_mode=True)
    clips = json.loads(capsys.readouterr().out)["clips"]
    assert [(c["start"], c["end"]) for c in clips] == [(5.0, 12.0)]


def test_run_clip_errors_when_transcript_has_no_utterances(media, fake_ffmpeg, monkeypatch):
    config.set_api_key("default", "sk_test")
    monkeypatch.setattr(client, "transcribe", lambda *a, **k: fake_transcript(None))
    opts = dataclasses.replace(DEFAULTS, media=str(media), speakers=["A"])
    with pytest.raises(CLIError) as exc:
        clip_exec.run_clip(opts, AppState(), json_mode=False)
    assert exc.value.error_type == "no_utterances"
    assert exc.value.exit_code == 2
    assert "tr_123 has no utterances" in exc.value.message
    assert "--speaker-labels" in (exc.value.suggestion or "")


def test_run_clip_errors_when_nothing_matches(media, fake_ffmpeg, monkeypatch):
    config.set_api_key("default", "sk_test")
    monkeypatch.setattr(client, "transcribe", lambda *a, **k: fake_transcript(list(UTTERANCES)))
    opts = dataclasses.replace(DEFAULTS, media=str(media), speakers=["Z"])
    with pytest.raises(CLIError) as exc:
        clip_exec.run_clip(opts, AppState(), json_mode=False)
    assert exc.value.error_type == "no_match"
    assert "No transcript segments matched" in exc.value.message
    assert "-o utterances" in (exc.value.suggestion or "")


def test_run_clip_status_messages(media, fake_ffmpeg, monkeypatch):
    config.set_api_key("default", "sk_test")
    monkeypatch.setattr(client, "transcribe", lambda *a, **k: fake_transcript(list(UTTERANCES)))
    messages: list[str] = []

    @contextlib.contextmanager
    def fake_status(message, *, json_mode, quiet):
        messages.append(message)
        yield

    monkeypatch.setattr(clip_exec.output, "status", fake_status)
    opts = dataclasses.replace(DEFAULTS, media=str(media), speakers=["A"])
    clip_exec.run_clip(opts, AppState(), json_mode=False)
    assert messages == [
        "Transcribing for clip selection…",
        "Detecting silence…",
        "Cutting 2 clip(s)…",
    ]
