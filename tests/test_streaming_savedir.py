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


def test_save_dir_plan_is_immutable(tmp_path):
    plan = savedir.SaveDirPlan(
        save_dir=tmp_path, now=NOW, name=None, auto_name=True, save_audio=True, write_note=False
    )
    field_name = "save_audio"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(plan, field_name, False)


def _plan(tmp_path: Path, *, name=None, auto_name=False, save_audio=True, write_note=False):
    return savedir.SaveDirPlan(
        save_dir=tmp_path / "rec",
        now=NOW,
        name=name,
        auto_name=auto_name,
        save_audio=save_audio,
        write_note=write_note,
    )


def _seed_capture(plan: savedir.SaveDirPlan) -> naming.SavePaths:
    """Create the provisional .txt (+ .wav when audio is saved) a live run would leave."""
    paths = plan.paths
    naming.ensure_dir(paths.directory)
    paths.transcript.write_text("Speaker A: hello\n", encoding="utf-8")
    if plan.save_audio:
        paths.audio.write_bytes(b"RIFFFAKE")
    return paths


def test_write_outputs_writes_sidecar_with_metadata(tmp_path):
    # The sidecar carries title/date/duration/speakers/turns plus the file names, so a
    # list UI needs no transcript parsing (a whole-dict assert kills key/value mutants).
    plan = _plan(tmp_path, name="My Meeting")
    _seed_capture(plan)

    final = savedir.write_outputs(
        plan, title=None, note=None, speakers=["A", "B"], duration_seconds=42, turns=3
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
        "audio": "2026-06-16-143005-my-meeting.wav",
        "note": None,
    }


def test_write_outputs_omits_audio_in_sidecar_when_not_saved(tmp_path):
    # --no-save-audio: the sidecar's "audio" is null, not a phantom WAV name.
    plan = _plan(tmp_path, name="Talk", save_audio=False)
    _seed_capture(plan)

    final = savedir.write_outputs(
        plan, title=None, note=None, speakers=[], duration_seconds=0, turns=0
    )

    assert json.loads(final.sidecar.read_text(encoding="utf-8"))["audio"] is None


def test_write_outputs_writes_llm_note_as_markdown(tmp_path):
    # With --llm a note lands as <stem>.md next to the transcript, and the sidecar
    # records its name.
    plan = _plan(tmp_path, name="Talk", write_note=True)
    _seed_capture(plan)

    final = savedir.write_outputs(
        plan, title=None, note="- ship the release", speakers=[], duration_seconds=1, turns=1
    )

    assert final.note.read_text(encoding="utf-8") == "- ship the release\n"
    assert json.loads(final.sidecar.read_text(encoding="utf-8"))["note"] == final.note.name


def test_write_outputs_skips_note_file_when_no_answer(tmp_path):
    # write_note was requested but no answer was produced (no turns) -> no .md, null in sidecar.
    plan = _plan(tmp_path, name="Talk", write_note=True)
    _seed_capture(plan)

    final = savedir.write_outputs(
        plan, title=None, note=None, speakers=[], duration_seconds=0, turns=0
    )

    assert not final.note.exists()
    assert json.loads(final.sidecar.read_text(encoding="utf-8"))["note"] is None


def test_write_outputs_renames_files_for_auto_name(tmp_path):
    # --auto-name: the provisional timestamp-only files are renamed to carry the title slug;
    # the WAV moves with the transcript and the old stem is gone.
    plan = _plan(tmp_path, auto_name=True)
    provisional = _seed_capture(plan)

    final = savedir.write_outputs(
        plan, title="Quarterly Review", note=None, speakers=[], duration_seconds=5, turns=2
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
    provisional = _seed_capture(plan)

    final = savedir.write_outputs(
        plan, title="???", note=None, speakers=[], duration_seconds=0, turns=1
    )

    assert final.transcript == provisional.transcript
    assert final.transcript.exists()


def test_write_outputs_skips_missing_wav_on_rename(tmp_path):
    # --auto-name with a recording that left no WAV (e.g. interrupted before audio) renames
    # the transcript without crashing on the absent audio file.
    plan = _plan(tmp_path, auto_name=True, save_audio=True)
    paths = plan.paths
    naming.ensure_dir(paths.directory)
    paths.transcript.write_text("hi\n", encoding="utf-8")  # no .wav written

    final = savedir.write_outputs(
        plan, title="A Title", note=None, speakers=[], duration_seconds=0, turns=1
    )

    assert final.transcript.read_text(encoding="utf-8") == "hi\n"
    assert not final.audio.exists()


def test_write_outputs_ignores_title_without_auto_name(tmp_path):
    # A title is only adopted under --auto-name; without it the explicit --name stem wins
    # and the sidecar keeps that name (pins the `auto_name and title` selection).
    plan = _plan(tmp_path, name="Orig", auto_name=False)
    provisional = _seed_capture(plan)

    final = savedir.write_outputs(
        plan, title="Ignored", note=None, speakers=[], duration_seconds=0, turns=0
    )

    assert final.stem == provisional.stem
    assert json.loads(final.sidecar.read_text(encoding="utf-8"))["title"] == "Orig"


def test_write_outputs_raises_clean_error_when_sidecar_write_blocked(tmp_path):
    # A failed sidecar write (its directory was never created) is a save_dir_path CLIError,
    # not a raw OSError — pins the _write error wrapper.
    plan = _plan(tmp_path, name="Talk", save_audio=False)
    # No ensure_dir / provisional files: the bucket directory doesn't exist, so writing
    # the sidecar into it fails.
    with pytest.raises(CLIError) as excinfo:
        savedir.write_outputs(plan, title=None, note=None, speakers=[], duration_seconds=0, turns=0)
    assert excinfo.value.error_type == "save_dir_path"
    assert excinfo.value.exit_code == 2


def test_write_outputs_raises_clean_error_when_rename_blocked(tmp_path):
    # A failed rename surfaces as a save_dir_path CLIError, not a raw OSError.
    plan = _plan(tmp_path, auto_name=True, save_audio=False)
    naming.ensure_dir(plan.paths.directory)
    # No provisional transcript exists, so the rename of a missing file fails.
    with pytest.raises(CLIError) as excinfo:
        savedir.write_outputs(
            plan, title="A Title", note=None, speakers=[], duration_seconds=0, turns=0
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
