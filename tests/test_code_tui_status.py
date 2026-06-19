"""Tests for the coding-agent TUI's pure status/text helpers (`tui_status`).

Split from test_code_tui.py (which drives the Textual app) to keep each file under the
500-line gate; these need no pilot, just the plain functions.
"""

from __future__ import annotations

from pathlib import Path

import pyperclip

from aai_cli.code_agent import tui_status
from aai_cli.ui import theme


def test_spinner_text_formats_frame_and_elapsed() -> None:
    assert tui_status._spinner_text(46, "✶") == "✶ Working… (46s)"
    assert tui_status._spinner_text(0, "✷") == "✷ Working… (0s)"


def test_abbrev_home() -> None:
    assert tui_status._abbrev_home(Path.home() / "proj") == "~/proj"
    # A path outside home renders as-is; compare to the platform-native string so this
    # holds on Windows (where str(Path(...)) uses backslashes) as well as POSIX.
    outside = Path("/etc/hosts")
    assert tui_status._abbrev_home(outside) == str(outside)


def test_git_branch_and_status(tmp_path: Path) -> None:
    assert tui_status._git_branch(tmp_path) is None  # no .git
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/feature-x\n")
    assert tui_status._git_branch(tmp_path) == "feature-x"
    (tmp_path / ".git" / "HEAD").write_text("a1b2c3d4e5f6\n")  # detached
    assert tui_status._git_branch(tmp_path) == "a1b2c3d4"

    status = tui_status._status_text(tmp_path, auto_approve=True)
    assert "auto" in status and "a1b2c3d4" in status
    assert "manual" in tui_status._status_text(tmp_path, auto_approve=False)


def test_voicebar_markup_per_phase_carries_label_meter_accent_and_hint() -> None:
    # Each phase renders its own label + accent color; the meter frame and any trailing hint
    # are passed through verbatim. Assert the literal accents (not the dict value) so a mutated
    # color literal is caught — reading from the dict would mutate in lockstep and survive.
    listening = tui_status.voicebar_markup("listening", "▁▃▅", hint=" (Ctrl-V)")
    assert "Listening" in listening and "▁▃▅" in listening and " (Ctrl-V)" in listening
    assert theme.BRAND in listening  # blue accent while listening
    thinking = tui_status.voicebar_markup("thinking", "▃▅▇")
    assert "Thinking" in thinking and "#f59e0b" in thinking  # amber, no hint
    speaking = tui_status.voicebar_markup("speaking", "▅▇▆")
    assert "Speaking" in speaking and "#22c55e" in speaking  # green


def test_copy_note_copies_and_confirms() -> None:
    # The happy path: the reply is handed to the copier and a confirmation note returned.
    copied: list[str] = []
    note = tui_status.copy_note("a reply", copied.append)
    assert copied == ["a reply"]
    assert "copied" in note


def test_copy_note_when_nothing_to_copy() -> None:
    # No reply yet: don't touch the clipboard, and tell the user there's nothing to copy.
    copied: list[str] = []
    note = tui_status.copy_note("", copied.append)
    assert copied == []  # copier never called
    assert "nothing to copy" in note


def test_copy_note_degrades_when_no_clipboard() -> None:
    # A headless/clipboard-less box: pyperclip raises; copy_note must absorb it and return a
    # note rather than letting the raise propagate (which would tear down the TUI).
    def _boom(_text: str) -> None:
        raise pyperclip.PyperclipException("no copy/paste mechanism")

    note = tui_status.copy_note("a reply", _boom)
    assert "no clipboard available" in note


def test_status_text_renders_voice_badge(tmp_path: Path) -> None:
    # No voice front-end -> no voice badge (the dot glyphs are absent); on/off render the
    # state so the Ctrl-V toggle shows. (Asserts on the dots, not the word — the tmp_path name
    # itself can contain "voice".)
    none = tui_status._status_text(tmp_path, auto_approve=False)
    assert "●" not in none and "○" not in none
    on = tui_status._status_text(tmp_path, auto_approve=False, voice_state="on")
    off = tui_status._status_text(tmp_path, auto_approve=False, voice_state="off")
    assert "voice on" in on and "●" in on  # filled dot when on
    assert "voice off" in off and "○" in off  # hollow dot when off
