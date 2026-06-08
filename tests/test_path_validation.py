"""File-path options reject a missing path up front via Typer's `exists=True`.

Typer validates the path before the command body runs (no API key needed), and
its message — "does not exist" with exit code 2 — is distinct from the loaders'
own not-found handling, so these also pin the `exists=True` option kwarg.
"""

from __future__ import annotations

import re

from typer.testing import CliRunner

from aai_cli.main import app

runner = CliRunner()


def _flatten(output: str) -> str:
    """Collapse a Rich error panel into one line for width-independent matching.

    Typer renders option errors in a bordered panel; at narrow terminal widths the
    message wraps and the box borders ("│") split a phrase like "does not exist"
    across lines. Drop the borders and collapse whitespace so the substring check
    doesn't depend on the sandbox's terminal width.
    """
    return re.sub(r"\s+", " ", output.replace("│", " "))


def test_transcribe_config_file_must_exist(tmp_path):
    result = runner.invoke(
        app, ["transcribe", "x.mp3", "--config-file", str(tmp_path / "nope.json")]
    )
    assert result.exit_code == 2
    assert "does not exist" in _flatten(result.output)


def test_transcribe_custom_spelling_file_must_exist(tmp_path):
    result = runner.invoke(
        app, ["transcribe", "x.mp3", "--custom-spelling-file", str(tmp_path / "nope.json")]
    )
    assert result.exit_code == 2
    assert "does not exist" in _flatten(result.output)


def test_stream_config_file_must_exist(tmp_path):
    result = runner.invoke(app, ["stream", "--config-file", str(tmp_path / "nope.json")])
    assert result.exit_code == 2
    assert "does not exist" in _flatten(result.output)


def test_agent_system_prompt_file_must_exist(tmp_path):
    result = runner.invoke(app, ["agent", "--system-prompt-file", str(tmp_path / "nope.txt")])
    assert result.exit_code == 2
    assert "does not exist" in _flatten(result.output)
