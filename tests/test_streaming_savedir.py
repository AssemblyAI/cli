"""Unit tests for aai_cli.streaming.savedir — the --save-dir finalization.

``write_outputs`` is pure file I/O (no gateway), so the rename/note/sidecar behavior
is asserted directly; the LLM title call (``derive_title``) is exercised against a
patched ``llm.run_chain``.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path

import pytest

from aai_cli.core.errors import CLIError
from aai_cli.streaming import naming, savedir

NOW = datetime(2026, 6, 16, 14, 30, 5)


def _plan(tmp_path: Path, *, name=None, auto_name=False, write_note=False):
    return savedir.SaveDirPlan(
        save_dir=tmp_path / "rec",
        now=NOW,
        name=name,
        auto_name=auto_name,
        write_note=write_note,
    )


def _seed(plan: savedir.SaveDirPlan, *, audio=True) -> tuple[naming.SavePaths, list[Path]]:
    """Create the provisional .txt (+ .wav when audio is saved) a live run would leave.

    Returns the provisional paths and the audio-file list to hand ``write_outputs`` (the
    session passes the WAV(s) it actually teed, or none under ``--no-save-audio``).
    """
    paths = plan.paths
    naming.ensure_dir(paths.directory)
    paths.transcript.write_text("Speaker A: hello\n", encoding="utf-8")
    if not audio:
        return paths, []
    paths.audio.write_bytes(b"RIFFFAKE")
    return paths, [paths.audio]


def test_save_dir_plan_is_immutable(tmp_path):
    plan = _plan(tmp_path, auto_name=True)
    field_name = "auto_name"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(plan, field_name, False)


def test_write_outputs_writes_sidecar_with_metadata(tmp_path):
    # The sidecar carries title/date/duration/speakers/turns plus the file names, so a
    # list UI needs no transcript parsing (a whole-dict assert kills key/value mutants).
    plan = _plan(tmp_path, name="My Meeting")
    _, audio = _seed(plan)

    final = savedir.write_outputs(
        plan, title=None, note=None, audio=audio, speakers=["A", "B"], duration_seconds=42, turns=3
    )

    raw = final.sidecar.read_text(encoding="utf-8")
    assert '\n  "title"' in raw  # pretty-printed with a 2-space indent
    assert json.loads(raw) == {
        "title": "My Meeting",
        "date": "2026-06-16T14:30:05",
        "duration_seconds": 42,
        "speakers": ["A", "B"],
        "turns": 3,
        "transcript": "2026-06-16-143005-my-meeting.txt",
        "audio": ["2026-06-16-143005-my-meeting.wav"],
        "note": None,
    }


def test_write_outputs_empty_audio_list_when_not_saved(tmp_path):
    # --no-save-audio: the sidecar's "audio" is an empty list, not a phantom WAV name.
    plan = _plan(tmp_path, name="Talk")
    _seed(plan, audio=False)

    final = savedir.write_outputs(
        plan, title=None, note=None, audio=[], speakers=[], duration_seconds=0, turns=0
    )

    assert json.loads(final.sidecar.read_text(encoding="utf-8"))["audio"] == []


def test_write_outputs_lists_each_channel_wav_for_system_audio(tmp_path):
    # --save-dir --system-audio tees one WAV per channel; the sidecar lists both, and
    # --auto-name re-stems each alongside the transcript.
    plan = _plan(tmp_path, auto_name=True)
    paths = plan.paths
    naming.ensure_dir(paths.directory)
    paths.transcript.write_text("You: hi\nSystem: hi\n", encoding="utf-8")
    you = naming.channel_audio(paths.audio, "you")
    system = naming.channel_audio(paths.audio, "system")
    you.write_bytes(b"YOU0")
    system.write_bytes(b"SYS0")

    final = savedir.write_outputs(
        plan,
        title="Sync Up",
        note=None,
        audio=[you, system],
        speakers=[],
        duration_seconds=3,
        turns=2,
    )

    record = json.loads(final.sidecar.read_text(encoding="utf-8"))
    assert record["audio"] == [
        "2026-06-16-143005-sync-up-you.wav",
        "2026-06-16-143005-sync-up-system.wav",
    ]
    # Both channel WAVs moved to the new stem; the old-stem files are gone.
    assert (final.directory / "2026-06-16-143005-sync-up-you.wav").read_bytes() == b"YOU0"
    assert (final.directory / "2026-06-16-143005-sync-up-system.wav").read_bytes() == b"SYS0"
    assert not you.exists()


def test_write_outputs_writes_llm_note_as_markdown(tmp_path):
    # With --llm a note lands as <stem>.md next to the transcript, and the sidecar
    # records its name.
    plan = _plan(tmp_path, name="Talk", write_note=True)
    _, audio = _seed(plan)

    final = savedir.write_outputs(
        plan,
        title=None,
        note="- ship the release",
        audio=audio,
        speakers=[],
        duration_seconds=1,
        turns=1,
    )

    assert final.note.read_text(encoding="utf-8") == "- ship the release\n"
    assert json.loads(final.sidecar.read_text(encoding="utf-8"))["note"] == final.note.name


def test_write_outputs_skips_note_file_when_no_answer(tmp_path):
    # write_note was requested but no answer was produced (no turns) -> no .md, null in sidecar.
    plan = _plan(tmp_path, name="Talk", write_note=True)
    _, audio = _seed(plan)

    final = savedir.write_outputs(
        plan, title=None, note=None, audio=audio, speakers=[], duration_seconds=0, turns=0
    )

    assert not final.note.exists()
    assert json.loads(final.sidecar.read_text(encoding="utf-8"))["note"] is None


def test_write_outputs_renames_files_for_auto_name(tmp_path):
    # --auto-name: the provisional timestamp-only files are renamed to carry the title slug;
    # the WAV moves with the transcript and the old stem is gone.
    plan = _plan(tmp_path, auto_name=True)
    provisional, audio = _seed(plan)

    final = savedir.write_outputs(
        plan,
        title="Quarterly Review",
        note=None,
        audio=audio,
        speakers=[],
        duration_seconds=5,
        turns=2,
    )

    assert final.transcript.name == "2026-06-16-143005-quarterly-review.txt"
    assert final.transcript.read_text(encoding="utf-8") == "Speaker A: hello\n"
    assert final.audio.name == "2026-06-16-143005-quarterly-review.wav"
    assert final.audio.read_bytes() == b"RIFFFAKE"
    assert not provisional.transcript.exists()
    assert not provisional.audio.exists()
    assert json.loads(final.sidecar.read_text(encoding="utf-8"))["title"] == "Quarterly Review"


def test_write_outputs_keeps_timestamp_when_title_slugs_to_nothing(tmp_path):
    # An auto-name title with no usable characters slugs to "" -> keep the timestamp stem
    # rather than rename to a bare-hyphen name.
    plan = _plan(tmp_path, auto_name=True)
    provisional, audio = _seed(plan)

    final = savedir.write_outputs(
        plan, title="???", note=None, audio=audio, speakers=[], duration_seconds=0, turns=1
    )

    assert final.transcript == provisional.transcript
    assert final.transcript.exists()


def test_write_outputs_skips_missing_wav_on_rename(tmp_path):
    # --auto-name with a recording that left no WAV (e.g. interrupted before audio) renames
    # the transcript without crashing on the absent audio file.
    plan = _plan(tmp_path, auto_name=True)
    paths = plan.paths
    naming.ensure_dir(paths.directory)
    paths.transcript.write_text("hi\n", encoding="utf-8")  # no .wav written

    final = savedir.write_outputs(
        plan,
        title="A Title",
        note=None,
        audio=[paths.audio],
        speakers=[],
        duration_seconds=0,
        turns=1,
    )

    assert final.transcript.read_text(encoding="utf-8") == "hi\n"
    assert not final.audio.exists()


def test_write_outputs_ignores_title_without_auto_name(tmp_path):
    # A title is only adopted under --auto-name; without it the explicit --name stem wins
    # and the sidecar keeps that name (pins the `auto_name and title` selection).
    plan = _plan(tmp_path, name="Orig", auto_name=False)
    provisional, audio = _seed(plan)

    final = savedir.write_outputs(
        plan, title="Ignored", note=None, audio=audio, speakers=[], duration_seconds=0, turns=0
    )

    assert final.stem == provisional.stem
    assert json.loads(final.sidecar.read_text(encoding="utf-8"))["title"] == "Orig"


def test_write_outputs_raises_clean_error_when_sidecar_write_blocked(tmp_path):
    # A failed sidecar write (its directory was never created) is a save_dir_path CLIError,
    # not a raw OSError — pins the _write error wrapper.
    plan = _plan(tmp_path, name="Talk")
    # No ensure_dir / provisional files: the bucket directory doesn't exist, so writing
    # the sidecar into it fails.
    with pytest.raises(CLIError) as excinfo:
        savedir.write_outputs(
            plan, title=None, note=None, audio=[], speakers=[], duration_seconds=0, turns=0
        )
    assert excinfo.value.error_type == "save_dir_path"
    assert excinfo.value.exit_code == 2


def test_write_outputs_raises_clean_error_when_rename_blocked(tmp_path):
    # A failed rename surfaces as a save_dir_path CLIError, not a raw OSError.
    plan = _plan(tmp_path, auto_name=True)
    naming.ensure_dir(plan.paths.directory)
    # No provisional transcript exists, so the rename of a missing file fails.
    with pytest.raises(CLIError) as excinfo:
        savedir.write_outputs(
            plan, title="A Title", note=None, audio=[], speakers=[], duration_seconds=0, turns=0
        )
    assert excinfo.value.error_type == "save_dir_path"
    assert excinfo.value.exit_code == 2


def test_derive_title_strips_and_prompts_over_transcript(monkeypatch):
    # derive_title runs the title prompt over the transcript text and trims the answer.
    seen = {}

    def fake_run_chain(api_key, prompts, *, transcript_text, model, max_tokens):
        seen["prompts"] = prompts
        seen["transcript_text"] = transcript_text
        return "  Quarterly Review  \n"

    monkeypatch.setattr(savedir.llm, "run_chain", fake_run_chain)
    title = savedir.derive_title("sk", "we shipped the release", model="m", max_tokens=32)

    assert title == "Quarterly Review"
    assert seen["prompts"] == [savedir.TITLE_PROMPT]
    assert seen["transcript_text"] == "we shipped the release"
