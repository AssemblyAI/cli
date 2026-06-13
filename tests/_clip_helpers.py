"""Shared builders for the `assembly clip` test modules.

The clip suite is split across test_clip_select.py (pure selection logic),
test_clip_exec.py (validation + cutting), test_clip_sources.py (YouTube, stdin
pipe, LLM selection), and test_clip_command.py (argv parsing); the option
defaults and transcript fakes they all share live here.
"""

from __future__ import annotations

import re
import subprocess
from types import SimpleNamespace

import pytest

from aai_cli import llm, mediafile
from aai_cli.commands.clip._exec import ClipOptions

_ANSI_SGR = re.compile(r"\x1b\[[0-9;]*m")

# The CLI's flag defaults, as data. Tests override per-case with dataclasses.replace.
DEFAULTS = ClipOptions(
    media="meeting.mp4",
    transcript_id=None,
    speakers=[],
    search=None,
    llm_prompt=None,
    model=llm.DEFAULT_MODEL,
    max_tokens=llm.DEFAULT_MAX_TOKENS,
    ranges=[],
    padding=0.0,
    snap=True,
    out_dir=None,
    video=False,
)


def plain(text: str) -> str:
    """Strip SGR color codes (CI forces color on) for substring assertions."""
    return _ANSI_SGR.sub("", text)


def utterance(start, end, speaker, text):
    return SimpleNamespace(start=start, end=end, speaker=speaker, text=text)


UTTERANCES = [
    utterance(1500, 2500, "A", "Let's talk pricing today."),
    utterance(3000, 4000, "B", "Sounds good."),
    utterance(5000, 6000, "A", "Moving on to hiring."),
]


def fake_transcript(utterances):
    return SimpleNamespace(id="tr_123", utterances=utterances)


def record_ffmpeg(monkeypatch: pytest.MonkeyPatch, detect_log: str = "") -> list[list[str]]:
    """Resolve ffmpeg and record every invocation, succeeding with no output.

    With snapping on (the default) the first recorded call is the silencedetect
    pass; ``detect_log`` is what it reports on stderr (empty: no silences, so
    snapping is a no-op and cut times stay exactly as selected).
    """
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    calls: list[list[str]] = []

    def run(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        stderr = detect_log if "-af" in args else ""
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr=stderr)

    monkeypatch.setattr(mediafile, "run_ffmpeg", run)
    return calls
