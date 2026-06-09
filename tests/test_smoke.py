from typer.testing import CliRunner

from aai_cli.main import app

runner = CliRunner()


def test_help_runs():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "AssemblyAI" in result.output


def test_version_flag_prints_and_exits():
    # `aai --version` / `-V` is the reflex every CLI answers; the eager callback prints
    # the version and exits before any command runs.
    from aai_cli import __version__

    for flag in ("--version", "-V"):
        result = runner.invoke(app, [flag])
        assert result.exit_code == 0
        assert result.output.strip() == __version__


def test_quiet_suppresses_env_override_warning(monkeypatch):
    # --env contradicting the profile normally warns on stderr; --quiet silences it.
    from aai_cli import config

    config.set_api_key("default", "sk_live")
    config.set_profile_env("default", "production")
    # The warning is emitted by the root callback (before any subcommand), so a bare
    # invocation that falls through to the help screen exercises it without network.
    noisy = runner.invoke(app, ["--env", "sandbox000"])
    quiet = runner.invoke(app, ["--quiet", "--env", "sandbox000"])
    assert "may be rejected" in noisy.output
    assert "may be rejected" not in quiet.output


def test_shell_completion_is_available(monkeypatch):
    # add_completion=True ships `--show-completion` (and --install-completion), the
    # discoverability affordance gh/kubectl/docker users reach for. Typer detects the
    # active shell via shellingham, which needs a controlling TTY — absent under
    # CliRunner/CI — so pin it to a known shell to test the affordance deterministically.
    import typer.completion

    monkeypatch.setattr(typer.completion, "_get_shell_name", lambda: "zsh")
    result = runner.invoke(app, ["--show-completion"])
    assert result.exit_code == 0
    assert "_aai_completion" in result.output  # the emitted zsh completion script


def test_global_flags_parse():
    # --profile is a global option accepted before the eager --version flag
    assert runner.invoke(app, ["--profile", "staging", "--version"]).exit_code == 0


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
    # Grouped into Rich help panels (see help_panels.py): Quick Start, Build an app,
    # Run AssemblyAI, Setup & Tools, History, then Account.
    assert names == [
        # Quick Start
        "onboard",
        # Build an app
        "init",
        # Run AssemblyAI
        "transcribe",
        "stream",
        "agent",
        "llm",
        # Setup & Tools
        "doctor",
        "setup",
        # History
        "transcripts",
        "sessions",
        # Account
        "login",
        "logout",
        "whoami",
        "balance",
        "usage",
        "limits",
        "keys",
        "audit",
    ]
