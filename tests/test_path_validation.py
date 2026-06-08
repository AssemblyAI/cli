"""File-path options reject a missing path up front via Typer's `exists=True`.

Typer validates the path before the command body runs (no API key needed), and
its message — "does not exist" with exit code 2 — is distinct from the loaders'
own not-found handling, so these also pin the `exists=True` option kwarg.
"""

from __future__ import annotations

from typer.testing import CliRunner

from aai_cli.main import app

runner = CliRunner()


def test_transcribe_config_file_must_exist(tmp_path):
    result = runner.invoke(
        app, ["transcribe", "x.mp3", "--config-file", str(tmp_path / "nope.json")]
    )
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_transcribe_custom_spelling_file_must_exist(tmp_path):
    result = runner.invoke(
        app, ["transcribe", "x.mp3", "--custom-spelling-file", str(tmp_path / "nope.json")]
    )
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_stream_config_file_must_exist(tmp_path):
    result = runner.invoke(app, ["stream", "--config-file", str(tmp_path / "nope.json")])
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_agent_system_prompt_file_must_exist(tmp_path):
    result = runner.invoke(app, ["agent", "--system-prompt-file", str(tmp_path / "nope.txt")])
    assert result.exit_code == 2
    assert "does not exist" in result.output
