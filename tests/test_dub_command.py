"""Argv parsing tests for `assembly dub` (aai_cli/commands/dub.py): the command
module only builds a DubOptions and hands it to dub_exec.run_dub, so these
tests pin the flag -> options mapping and the end-to-end sandbox guard; the
pipeline itself is covered in test_dub_exec.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from aai_cli.commands.dub import _exec as dub_exec
from aai_cli.core import llm
from aai_cli.main import app
from tests._clip_helpers import plain
from tests._tts_session_helpers import use_no_tts_production

runner = CliRunner()


@pytest.fixture
def captured_run(monkeypatch: pytest.MonkeyPatch):
    """Capture the (opts, json_mode) the command hands to run_dub."""
    seen: dict[str, object] = {}

    def fake_run(opts, state, *, json_mode):
        seen["opts"] = opts
        seen["json_mode"] = json_mode

    monkeypatch.setattr(dub_exec, "run_dub", fake_run)
    return seen


def test_lang_is_required():
    result = runner.invoke(app, ["dub", "talk.mp4"])
    assert result.exit_code == 2
    assert "--lang" in plain(result.output)


def test_env_without_tts_host_is_rejected(monkeypatch):
    use_no_tts_production(monkeypatch)
    result = runner.invoke(app, ["dub", "talk.mp4", "--lang", "de"])
    assert result.exit_code == 2
    output = " ".join(plain(result.output).split())
    assert "assembly dub needs streaming TTS" in output
    assert "to run assembly dub in production" in output


def test_defaults_map_to_options(captured_run):
    result = runner.invoke(app, ["dub", "talk.mp4", "--lang", "de"])
    assert result.exit_code == 0
    assert captured_run["json_mode"] is False
    assert captured_run["opts"] == dub_exec.DubOptions(
        media="talk.mp4",
        language="de",
        source_language=None,
        transcript_id=None,
        voice=[],
        model=llm.DEFAULT_MODEL,
        max_tokens=llm.DEFAULT_MAX_TOKENS,
        out=None,
        video=False,
        download_sections=[],
        from_stdin=False,
        concurrency=4,
        force=False,
    )


def test_every_flag_maps_to_options(captured_run):
    result = runner.invoke(
        app,
        [
            "dub",
            "talk.mp4",
            "--lang",
            "German",
            "--source-lang",
            "fr",
            "-t",
            "tr_1",
            "--voice",
            "A=jane",
            "--voice",
            "paul",
            "--model",
            "gpt-5",
            "--max-tokens",
            "7",
            "--out",
            "dubbed.mp4",
            "--video",
            "--download-sections",
            "*0:00-15:00",
            "--download-sections",
            "intro",
            "--json",
        ],
    )
    assert result.exit_code == 0
    assert captured_run["json_mode"] is True
    assert captured_run["opts"] == dub_exec.DubOptions(
        media="talk.mp4",
        language="German",
        source_language="fr",
        transcript_id="tr_1",
        voice=["A=jane", "paul"],
        model="gpt-5",
        max_tokens=7,
        out=Path("dubbed.mp4"),
        video=True,
        download_sections=["*0:00-15:00", "intro"],
        from_stdin=False,
        concurrency=4,
        force=False,
    )
