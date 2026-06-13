"""`assembly t` — the hidden one-letter alias for transcribe."""

from typer.testing import CliRunner

from aai_cli.main import app

runner = CliRunner()


def test_t_runs_the_transcribe_command():
    # --show-code needs no auth/network and proves the full flag surface is shared.
    alias = runner.invoke(app, ["t", "--sample", "--show-code"])
    full = runner.invoke(app, ["transcribe", "--sample", "--show-code"])
    assert alias.exit_code == 0
    assert "import assemblyai" in alias.output
    assert alias.output == full.output  # same command, same behavior


def test_t_is_hidden_from_root_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # No bare "t" row in the command table (transcribe itself still lists).
    assert "\n│ t " not in result.output
