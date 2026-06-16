"""Unit tests for aai_cli.streaming.transcript — the --save-transcript text writer."""

from __future__ import annotations

import pytest

from aai_cli.core.errors import CLIError
from aai_cli.streaming.transcript import TranscriptWriter, validate_target


def test_validate_target_rejects_missing_parent_dir(tmp_path):
    with pytest.raises(CLIError) as excinfo:
        validate_target(tmp_path / "nope" / "notes.txt")
    assert excinfo.value.error_type == "save_transcript_path"


def test_validate_target_accepts_existing_parent(tmp_path):
    validate_target(tmp_path / "notes.txt")  # parent exists -> no raise


def test_writer_appends_one_line_per_turn_and_flushes(tmp_path):
    out = tmp_path / "notes.txt"
    writer = TranscriptWriter(out)
    try:
        writer.write_turn("hello world")
        # Flushed per turn: the line is on disk before close, so a Ctrl-C keeps it.
        assert out.read_text(encoding="utf-8") == "hello world\n"
        writer.write_turn("Speaker A: second turn")
    finally:
        writer.close()
    assert out.read_text(encoding="utf-8") == "hello world\nSpeaker A: second turn\n"


def test_writer_raises_clean_error_on_unopenable_path(tmp_path):
    # The parent doesn't exist, so opening for write fails -> a clean CLIError, not OSError.
    with pytest.raises(CLIError) as excinfo:
        TranscriptWriter(tmp_path / "missing" / "notes.txt")
    assert excinfo.value.error_type == "save_transcript_path"
