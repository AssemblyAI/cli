"""Unit tests for aai_cli.streaming.naming — the --save-dir filename assembly."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from aai_cli.core.errors import CLIError
from aai_cli.streaming import naming

NOW = datetime(2026, 6, 16, 14, 30, 5)


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Weekly Standup", "weekly-standup"),
        ("Weekly  Standup!! (Eng)", "weekly-standup-eng"),  # runs collapse, edges stripped
        ("ALL CAPS", "all-caps"),  # lowercased
        ("---trim---", "trim"),  # leading/trailing separators dropped
        ("!!!", ""),  # nothing usable -> empty, so the caller drops the suffix
    ],
)
def test_slugify(title, expected):
    assert naming.slugify(title) == expected


def test_slugify_caps_length_without_trailing_hyphen():
    # A long title is capped at 80 chars; a hyphen landing on the cut boundary is stripped.
    slug = naming.slugify("a" * 79 + " more words here")
    assert slug == "a" * 79  # the space would have become the 80th char's hyphen, trimmed


def test_resolve_buckets_by_date_with_slugged_name():
    paths = naming.resolve(Path("rec"), "My Meeting", now=NOW)
    assert paths.transcript == Path("rec/2026-06-16/2026-06-16-143005-my-meeting.txt")
    assert paths.audio == Path("rec/2026-06-16/2026-06-16-143005-my-meeting.wav")


def test_resolve_derives_note_and_sidecar_from_the_same_stem():
    # The .md note and .aai.json sidecar share the transcript's stem so a browse UI can
    # match all four files of one recording by name (pins each suffix).
    paths = naming.resolve(Path("rec"), "My Meeting", now=NOW)
    bucket = Path("rec/2026-06-16")
    assert paths.note == bucket / "2026-06-16-143005-my-meeting.md"
    assert paths.sidecar == bucket / "2026-06-16-143005-my-meeting.aai.json"
    assert paths.directory == bucket


def test_resolve_without_name_is_just_the_timestamp():
    # No --name (or a name that slugs to nothing) -> the stem is the bare timestamp,
    # never a trailing-hyphen filename.
    assert naming.resolve(Path("rec"), None, now=NOW).transcript.name == "2026-06-16-143005.txt"
    assert naming.resolve(Path("rec"), "!!!", now=NOW).transcript.name == "2026-06-16-143005.txt"


def test_ensure_dir_creates_nested_dirs(tmp_path):
    target = tmp_path / "rec" / "2026-06-16"
    naming.ensure_dir(target)
    assert target.is_dir()


def test_ensure_dir_raises_clean_error_when_path_is_blocked(tmp_path):
    # A parent component that is a file, not a directory, can't be turned into a dir.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")
    with pytest.raises(CLIError) as excinfo:
        naming.ensure_dir(blocker / "2026-06-16")
    assert excinfo.value.error_type == "save_dir_path"
