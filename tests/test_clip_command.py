"""CLI-level tests for `assembly clip`: argv → ClipOptions parsing, error rendering,
and the command's placement in the root help."""

from __future__ import annotations

import json
import re
import subprocess

from typer.testing import CliRunner

from aai_cli import clip_exec, llm, mediafile
from aai_cli.clip_exec import ClipOptions
from aai_cli.main import app

runner = CliRunner()

_ANSI_SGR = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI_SGR.sub("", text)


def _capture_run_clip(monkeypatch):
    captured = {}

    def fake_run_clip(opts, state, *, json_mode):
        captured["opts"] = opts
        captured["json_mode"] = json_mode

    monkeypatch.setattr(clip_exec, "run_clip", fake_run_clip)
    return captured


def test_clip_parses_every_flag_into_options(monkeypatch, tmp_path):
    captured = _capture_run_clip(monkeypatch)
    result = runner.invoke(
        app,
        [
            "clip",
            "meeting.mp4",
            "-t",
            "tr_abc",
            "--speaker",
            "A",
            "--speaker",
            "B",
            "--search",
            "pricing",
            "--llm",
            "best moments",
            "--model",
            "gpt-5",
            "--max-tokens",
            "64",
            "--range",
            "5-10",
            "--range",
            "1:30-2:00",
            "--padding",
            "0.5",
            "--no-snap",
            "--out-dir",
            str(tmp_path),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["opts"] == ClipOptions(
        media="meeting.mp4",
        transcript_id="tr_abc",
        speakers=["A", "B"],
        search="pricing",
        llm_prompt="best moments",
        model="gpt-5",
        max_tokens=64,
        ranges=["5-10", "1:30-2:00"],
        padding=0.5,
        snap=False,
        out_dir=tmp_path,
    )
    assert captured["json_mode"] is True


def test_clip_defaults_when_only_media_is_given(monkeypatch):
    captured = _capture_run_clip(monkeypatch)
    result = runner.invoke(app, ["clip", "meeting.mp4"])
    assert result.exit_code == 0, result.output
    assert captured["opts"] == ClipOptions(
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
    )
    assert captured["json_mode"] is False


def test_clip_requires_the_media_argument():
    result = runner.invoke(app, ["clip"])
    assert result.exit_code == 2


def test_clip_rejects_negative_padding():
    result = runner.invoke(app, ["clip", "meeting.mp4", "--padding", "-1"])
    assert result.exit_code == 2


def test_clip_missing_file_renders_clean_error(tmp_path):
    result = runner.invoke(app, ["clip", str(tmp_path / "nope.mp4"), "--range", "1-2"])
    assert result.exit_code == 2
    plain = _plain(result.output)
    assert "File not found" in plain
    assert "Traceback" not in plain


def test_clip_json_error_shape(tmp_path):
    result = runner.invoke(app, ["clip", str(tmp_path / "nope.mp4"), "--range", "1-2", "--json"])
    assert result.exit_code == 2
    err = json.loads(_plain(result.output).strip())
    assert err["error"]["type"] == "file_not_found"


def test_clip_end_to_end_range_cut_via_cli(tmp_path, monkeypatch):
    media = tmp_path / "talk.mp3"
    media.write_bytes(b"\x00")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg")
    calls: list[list[str]] = []

    def fake_run(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mediafile, "run_ffmpeg", fake_run)
    result = runner.invoke(app, ["clip", str(media), "--range", "1-2", "--json"])
    assert result.exit_code == 0, result.output
    # calls[0] is the silencedetect pass; calls[1] the cut.
    assert calls[1][-1] == str(tmp_path / "talk.clip01.mp3")
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["clips"][0]["duration"] == 1.0


def test_clip_is_listed_between_llm_and_eval_in_root_help():
    # Pins clip's slot in _COMMAND_ORDER: it renders in the "Run AssemblyAI"
    # panel after llm, not alphabetically at the end of the help.
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    plain = _plain(result.output)

    def row(name: str) -> int:
        match = re.search(rf"^[│|\s]*{name}\s", plain, flags=re.MULTILINE)
        assert match is not None, f"{name} not in root help"
        return match.start()

    assert row("llm") < row("clip") < row("eval")
