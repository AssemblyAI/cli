from typer.testing import CliRunner

from aai_cli.main import app

runner = CliRunner()


def test_help_runs():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "AssemblyAI" in result.output


def test_version_command():
    from aai_cli import __version__

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


def test_help_lists_commands_in_workflow_order():
    from typer.core import TyperGroup
    from typer.main import get_command

    cmd = get_command(app)
    # Typer (>=0.13) vendors its own click; the root command is a TyperGroup.
    assert isinstance(cmd, TyperGroup)
    ctx = cmd.make_context("aai", [], resilient_parsing=True)
    names = cmd.list_commands(ctx)  # the order shown under --help
    # Core transcription first, then voice/LLM, account, tooling, version last.
    assert names == [
        "transcribe",
        "stream",
        "transcripts",
        "agent",
        "llm",
        "login",
        "logout",
        "whoami",
        "doctor",
        "samples",
        "init",
        "claude",
        "version",
    ]
