"""CLI-level tests for `assembly caption`: argv → CaptionOptions parsing, error
rendering, and the command's placement in the root help. The pipeline itself is
covered in test_caption_exec.py."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aai_cli.commands.caption import _exec as caption_exec
from aai_cli.commands.caption._exec import CaptionOptions
from aai_cli.main import app

runner = CliRunner()

_ANSI_SGR = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI_SGR.sub("", text)


def _capture_run_caption(monkeypatch):
    captured = {}

    def fake_run_caption(opts, state, *, json_mode):
        captured["opts"] = opts
        captured["json_mode"] = json_mode

    monkeypatch.setattr(caption_exec, "run_caption", fake_run_caption)
    return captured


def test_caption_parses_every_flag_into_options(monkeypatch):
    captured = _capture_run_caption(monkeypatch)
    result = runner.invoke(
        app,
        [
            "caption",
            "talk.mp4",
            "-t",
            "tr_abc",
            "--chars-per-caption",
            "32",
            "--font-size",
            "28",
            "--out",
            "captioned.mp4",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["opts"] == CaptionOptions(
        media="talk.mp4",
        transcript_id="tr_abc",
        chars_per_caption=32,
        font_size=28,
        out=Path("captioned.mp4"),
    )
    assert captured["json_mode"] is True


def test_caption_defaults_when_only_media_is_given(monkeypatch):
    captured = _capture_run_caption(monkeypatch)
    result = runner.invoke(app, ["caption", "talk.mp4"])
    assert result.exit_code == 0, result.output
    assert captured["opts"] == CaptionOptions(
        media="talk.mp4",
        transcript_id=None,
        chars_per_caption=None,
        font_size=None,
        out=None,
    )
    assert captured["json_mode"] is False


def test_caption_accepts_the_minimum_flag_values(monkeypatch):
    # Both numeric flags declare min=1; the boundary value must parse.
    captured = _capture_run_caption(monkeypatch)
    result = runner.invoke(
        app, ["caption", "talk.mp4", "--chars-per-caption", "1", "--font-size", "1"]
    )
    assert result.exit_code == 0, result.output
    assert captured["opts"].chars_per_caption == 1
    assert captured["opts"].font_size == 1


def test_caption_rejects_zero_flag_values():
    result = runner.invoke(app, ["caption", "talk.mp4", "--font-size", "0"])
    assert result.exit_code == 2
    result = runner.invoke(app, ["caption", "talk.mp4", "--chars-per-caption", "0"])
    assert result.exit_code == 2


def test_caption_requires_the_media_argument():
    result = runner.invoke(app, ["caption"])
    assert result.exit_code == 2


def test_caption_missing_file_renders_clean_error(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    result = runner.invoke(app, ["caption", str(tmp_path / "nope.mp4")])
    assert result.exit_code == 2
    plain = _plain(result.output)
    assert "File not found" in plain
    assert "Traceback" not in plain


def test_caption_json_error_shape(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    result = runner.invoke(app, ["caption", str(tmp_path / "nope.mp4"), "--json"])
    assert result.exit_code == 2
    err = json.loads(_plain(result.output).strip())
    assert err["error"]["type"] == "file_not_found"


@pytest.mark.usefixtures("internal_profile")  # dub is sandbox-only, hidden from external help
def test_caption_is_listed_between_dub_and_eval_in_root_help():
    # Pins caption's slot in _COMMAND_ORDER: it renders in the "Run AssemblyAI"
    # panel after dub, not alphabetically at the end of the help.
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    plain = _plain(result.output)

    def row(name: str) -> int:
        match = re.search(rf"^[│|\s]*{name}\s", plain, flags=re.MULTILINE)
        assert match is not None, f"{name} not in root help"
        return match.start()

    assert row("dub") < row("caption") < row("eval")
