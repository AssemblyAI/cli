from typer.testing import CliRunner

from assemblyai_cli.main import app

runner = CliRunner()


def test_help_runs():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "AssemblyAI" in result.output


def test_version_command():
    from assemblyai_cli import __version__

    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == __version__


def test_global_flags_parse():
    # --profile is a global option accepted before a subcommand
    assert runner.invoke(app, ["--profile", "staging", "version"]).exit_code == 0


def test_stream_registered_top_level():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "stream" in result.output
