"""Real ffmpeg parser/rendering regressions; only the SDK boundary is faked."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from aai_cli.commands.caption import _exec as caption_exec
from aai_cli.core import client, config
from aai_cli.main import app

SRT = "1\n00:00:00,000 --> 00:00:02,000\nHello captions\n"
DIRECTORIES = ["plain", "o'brien", "comma,dir", "semi;dir", "bracket[dir]", "space dir"]
if os.name == "posix":
    DIRECTORIES += ["tmp:dir", r"back\slash"]


def _ffmpeg(*args: str) -> str:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _hashes(*args: str) -> list[str]:
    return [
        line
        for line in _ffmpeg(*args, "-f", "framehash", "-").splitlines()
        if not line.startswith("#")
    ]


@pytest.mark.parametrize("dirname", DIRECTORIES)
@pytest.mark.parametrize("font_size", [None, 28])
def test_caption_cli_renders_subtitles(tmp_path, monkeypatch, dirname, font_size):
    temp_root = tmp_path / dirname
    temp_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_root))
    monkeypatch.setattr(config, "resolve_api_key", lambda **_: "test-key")
    calls = []

    def export(chars_per_caption=None):
        calls.append(("srt", chars_per_caption))
        return SRT

    def transcribe(api_key, audio, *, config):
        calls.append(("transcribe", Path(audio).name))
        return SimpleNamespace(id="test-caption", export_subtitles_srt=export)

    monkeypatch.setattr(client, "transcribe", transcribe)
    media = tmp_path / "talk.mp4"
    _ffmpeg("-f", "lavfi", "-i", "color=c=black:s=320x240:r=1:d=2", "-c:v", "mpeg4", str(media))
    args = ["caption", str(media), "--chars-per-caption", "32", "--json"]
    if font_size is not None:
        args += ["--font-size", str(font_size)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert calls == [("transcribe", "talk.mp4"), ("srt", 32)]
    baseline = _hashes("-i", str(media))
    captioned = _hashes("-i", str(tmp_path / "talk.captioned.mp4"))
    assert len(baseline) == len(captioned) == 2
    assert captioned != baseline


def test_windows_path_filter_contract(tmp_path, monkeypatch):
    # On POSIX a literal C: directory tests the parser, NOT native Windows I/O.
    monkeypatch.chdir(tmp_path)
    windows = PureWindowsPath(r"C:\captions\captions.srt")
    srt = Path(windows.as_posix()) if os.name == "posix" else tmp_path / "captions.srt"
    srt.parent.mkdir(parents=True, exist_ok=True)
    srt.write_text(SRT, encoding="utf-8")
    filter_path = windows if os.name == "posix" else PureWindowsPath(srt)
    spec = caption_exec.subtitles_filter(filter_path, None)
    if os.name == "posix":
        # Avoid libavformat treating C: as a URL protocol on this POSIX host.
        # Keep the generated drive colon and separators intact for both parsers.
        spec = spec.replace("subtitles=", "subtitles=./", 1)
    source = ("-f", "lavfi", "-i", "color=c=black:s=320x240:r=1:d=2")
    baseline = _hashes(*source)
    captioned = _hashes(*source, "-vf", spec)
    assert len(baseline) == len(captioned) == 2
    assert captioned != baseline
