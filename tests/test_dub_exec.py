"""Direct tests of the `assembly dub` options/run seam (aai_cli/dub_exec.py):
the pure helpers (language resolution, output naming, timeline assembly,
utterance extraction) and run_dub's validation order. Constructed-options
tests (dataclasses.replace off the shared defaults) avoid any argv
round-trip. The faked pipeline runs live in test_dub_pipeline.py; argv
parsing in test_dub_command.py."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from aai_cli import dub_exec
from aai_cli.context import AppState
from aai_cli.errors import CLIError, UsageError
from tests._dub_helpers import (
    DEFAULTS,
    enable_sandbox,
    fake_transcript,
    patch_api_key,
    utterance,
    write_media,
)


@pytest.fixture
def media(tmp_path: Path) -> Path:
    return write_media(tmp_path)


@pytest.fixture
def sandbox(monkeypatch: pytest.MonkeyPatch):
    enable_sandbox(monkeypatch)


@pytest.fixture(autouse=True)
def _fake_key(monkeypatch: pytest.MonkeyPatch):
    patch_api_key(monkeypatch)


# --- records and pure helpers --------------------------------------------------


@pytest.mark.parametrize(
    "instance",
    [DEFAULTS, dub_exec._Utterance(start_ms=0, speaker="A", text="hi")],
    ids=["options", "utterance"],
)
def test_records_are_immutable(instance):
    field_name = dataclasses.fields(instance)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field_name, None)


def test_language_names_map_codes_to_names():
    # An independent copy of the expected table: a silently edited entry in the
    # shipped map must fail here, not just round-trip through itself.
    assert dub_exec.LANGUAGE_NAMES == {
        "ar": "Arabic",
        "de": "German",
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "hi": "Hindi",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "nl": "Dutch",
        "pl": "Polish",
        "pt": "Portuguese",
        "ru": "Russian",
        "tr": "Turkish",
        "vi": "Vietnamese",
        "zh": "Chinese",
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("de", "German"),
        (" DE ", "German"),  # codes are trimmed and case-insensitive
        ("German", "German"),  # a full name passes through
        (" Klingon ", "Klingon"),  # unlisted languages pass through, trimmed
    ],
)
def test_resolve_language(value, expected):
    assert dub_exec.resolve_language(value) == expected


def test_resolve_language_rejects_blank():
    with pytest.raises(UsageError) as exc:
        dub_exec.resolve_language("   ")
    assert "--lang needs a language" in exc.value.message
    assert "--lang de" in (exc.value.suggestion or "")


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("German", "talk.dub.german.mp4"),
        ("Brazilian Portuguese", "talk.dub.brazilian-portuguese.mp4"),
    ],
)
def test_default_out_path(language, expected):
    out = dub_exec.default_out_path(Path("/x/talk.mp4"), language)
    assert out == Path("/x") / expected


def test_assemble_timeline_fills_gaps_and_pads_tail():
    # rate 1000: one second of 16-bit mono PCM is 2000 bytes.
    track = dub_exec.assemble_timeline([(500, b"\x01\x02")], 1000, total_seconds=1.0)
    # 0.5 s leading silence, the segment, then a 0.499 s tail pad to 1.0 s.
    assert track == b"\x00" * 1000 + b"\x01\x02" + b"\x00" * 998


def test_assemble_timeline_overlap_appends_without_silence():
    # The first segment runs to 0.1 s; the second "starts" at 0.05 s, so it is
    # appended immediately (the dub drifts) rather than overlapping or crashing.
    placed = [(0, b"\x01" * 200), (50, b"\x02\x02")]
    track = dub_exec.assemble_timeline(placed, 1000, total_seconds=None)
    assert track == b"\x01" * 200 + b"\x02\x02"


def test_assemble_timeline_skips_tail_when_track_is_long_enough():
    track = dub_exec.assemble_timeline([(0, b"\x01" * 200)], 1000, total_seconds=0.05)
    assert track == b"\x01" * 200


def test_utterances_of_defaults_and_filtering():
    transcript = fake_transcript(
        [
            SimpleNamespace(speaker=None, text="Hi"),  # no start attr, no speaker label
            utterance(2000, "B", None),  # no text -> dropped
            utterance(3000, "B", "   "),  # blank text -> dropped
            utterance(4000, "C", "  Bye  "),
        ]
    )
    assert dub_exec._utterances_of(transcript, "tr_dub") == [
        dub_exec._Utterance(start_ms=0, speaker="A", text="Hi"),
        dub_exec._Utterance(start_ms=4000, speaker="C", text="Bye"),
    ]


@pytest.mark.parametrize(
    "utterances",
    [None, [], [utterance(0, "A", "")]],
    ids=["missing", "empty", "all-blank"],
)
def test_utterances_of_requires_spoken_utterances(utterances):
    with pytest.raises(CLIError) as exc:
        dub_exec._utterances_of(SimpleNamespace(utterances=utterances), "tr_x")
    assert exc.value.error_type == "no_utterances"
    assert exc.value.exit_code == 2
    assert "Transcript tr_x has no utterances to dub" in exc.value.message
    assert "--speaker-labels" in (exc.value.suggestion or "")


@pytest.mark.parametrize(
    ("duration", "expected"),
    [(12, 12.0), (4.5, 4.5), (None, None), (True, None), ("90", None)],
    ids=["int", "float", "none", "bool", "str"],
)
def test_total_seconds(duration, expected):
    transcript = SimpleNamespace(audio_duration=duration)
    assert dub_exec._total_seconds(transcript) == expected


# --- validation order (cheap local checks before any credential or network) ----


def test_run_dub_rejects_blank_language_first():
    opts = dataclasses.replace(DEFAULTS, language=" ")
    with pytest.raises(UsageError):  # not the sandbox CLIError: language wins
        dub_exec.run_dub(opts, AppState(), json_mode=False)


def test_run_dub_requires_sandbox():
    # The active environment defaults to production, which has no streaming-TTS host.
    with pytest.raises(CLIError) as exc:
        dub_exec.run_dub(DEFAULTS, AppState(), json_mode=False)
    assert exc.value.error_type == "unsupported_environment"
    assert exc.value.exit_code == 2
    assert "only available in the sandbox" in exc.value.message
    assert "Re-run as: assembly --sandbox dub" in (exc.value.suggestion or "")


def test_run_dub_rejects_missing_file(sandbox, tmp_path):
    opts = dataclasses.replace(DEFAULTS, media=str(tmp_path / "nope.mp4"))
    with pytest.raises(CLIError) as exc:
        dub_exec.run_dub(opts, AppState(), json_mode=False)
    assert exc.value.error_type == "file_not_found"
    assert exc.value.exit_code == 2
    # The command name pins the shared helper's parameterization.
    assert "assembly dub needs a local audio/video file" in (exc.value.suggestion or "")


def test_run_dub_rejects_directory(sandbox, tmp_path):
    opts = dataclasses.replace(DEFAULTS, media=str(tmp_path))
    with pytest.raises(CLIError) as exc:
        dub_exec.run_dub(opts, AppState(), json_mode=False)
    assert exc.value.error_type == "not_a_file"
    assert exc.value.exit_code == 2
    assert "not a directory" in (exc.value.suggestion or "")


def test_run_dub_refuses_to_overwrite_the_input(sandbox, media):
    opts = dataclasses.replace(DEFAULTS, media=str(media), out=media)
    with pytest.raises(UsageError) as exc:
        dub_exec.run_dub(opts, AppState(), json_mode=False)
    assert "overwrite the input file" in exc.value.message


def test_run_dub_requires_ffmpeg(sandbox, media, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    opts = dataclasses.replace(DEFAULTS, media=str(media))
    with pytest.raises(CLIError) as exc:
        dub_exec.run_dub(opts, AppState(), json_mode=False)
    assert exc.value.error_type == "missing_dependency"
    # The purpose string pins the shared helper's parameterization.
    assert "ffmpeg is required to write the dubbed file" in exc.value.message
