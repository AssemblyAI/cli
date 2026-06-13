from typer.testing import CliRunner

from aai_cli.main import app

runner = CliRunner()


def test_help_runs():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "AssemblyAI" in result.output


def test_version_flag_prints_and_exits():
    # `assembly --version` / `-V` is the reflex every CLI answers; the eager callback prints
    # the version and exits before any command runs.
    from aai_cli import __version__

    for flag in ("--version", "-V"):
        result = runner.invoke(app, [flag])
        assert result.exit_code == 0
        assert result.output.strip() == __version__


def test_quiet_suppresses_env_override_warning(monkeypatch):
    # --env contradicting the profile normally warns on stderr; --quiet silences it.
    from aai_cli.core import config

    config.set_api_key("default", "sk_live")
    config.set_profile_env("default", "production")
    # The warning is emitted by the root callback (before any subcommand), so a bare
    # invocation that falls through to the help screen exercises it without network.
    noisy = runner.invoke(app, ["--env", "sandbox000"])
    quiet = runner.invoke(app, ["--quiet", "--env", "sandbox000"])
    assert "may be rejected" in noisy.output
    assert "may be rejected" not in quiet.output


def test_env_override_warning_is_structured_in_json_mode(monkeypatch, mocker):
    # In --json mode the warning ships as a {"warning": …} object on stderr, so a
    # pipeline gets a machine-readable hint instead of an unexplained auth failure.
    import json

    from aai_cli.core import config

    config.set_api_key("default", "sk_live")
    config.set_profile_env("default", "production")
    mocker.patch(
        "aai_cli.commands.transcripts.client.list_transcripts", autospec=True, return_value=[]
    )
    result = runner.invoke(app, ["--env", "sandbox000", "transcripts", "list", "--json"])
    warning_line = next(
        line for line in result.output.splitlines() if line.strip().startswith('{"warning"')
    )
    assert "may be rejected" in json.loads(warning_line)["warning"]


def test_sandbox_alone_targets_sandbox():
    from aai_cli.core import environments

    result = runner.invoke(app, ["--sandbox"])
    assert result.exit_code == 0
    assert environments.active().name == "sandbox000"


def test_sandbox_with_agreeing_env_is_fine():
    from aai_cli.core import environments

    result = runner.invoke(app, ["--sandbox", "--env", "sandbox000"])
    assert result.exit_code == 0
    assert environments.active().name == "sandbox000"


def test_sandbox_flag_conflicting_env_warns():
    # Credentials are environment-bound, so `--sandbox --env production` must not pick
    # one silently: --env wins, with a warning. Agreeing flags and --quiet stay silent.
    noisy = runner.invoke(app, ["--sandbox", "--env", "production"])
    assert "--sandbox ignored: --env production takes precedence." in noisy.output
    assert '{"warning"' not in noisy.output  # human mode renders text, not JSON
    quiet = runner.invoke(app, ["--quiet", "--sandbox", "--env", "production"])
    assert "--sandbox ignored" not in quiet.output
    agreeing = runner.invoke(app, ["--sandbox", "--env", "sandbox000"])
    assert "--sandbox ignored" not in agreeing.output
    env_only = runner.invoke(app, ["--env", "production"])
    assert "--sandbox ignored" not in env_only.output


def test_sandbox_conflict_warning_is_structured_in_json_mode(mocker):
    # Same {"warning": …} contract as the env-override warning: a --json pipeline gets
    # a machine-readable hint instead of a decorated human line.
    import json

    from aai_cli.core import config

    config.set_api_key("default", "sk_live")
    mocker.patch(
        "aai_cli.commands.transcripts.client.list_transcripts", autospec=True, return_value=[]
    )
    result = runner.invoke(
        app, ["--sandbox", "--env", "production", "transcripts", "list", "--json"]
    )
    warning_line = next(
        line for line in result.output.splitlines() if line.strip().startswith('{"warning"')
    )
    assert "--sandbox ignored" in json.loads(warning_line)["warning"]


def test_shell_completion_is_available(monkeypatch):
    # add_completion=True ships `--show-completion` (and --install-completion), the
    # discoverability affordance gh/kubectl/docker users reach for. Typer detects the
    # active shell via shellingham, which needs a controlling TTY — absent under
    # CliRunner/CI — so pin it to a known shell to test the affordance deterministically.
    import typer.completion

    monkeypatch.setattr(typer.completion, "_get_shell_name", lambda: "zsh")
    result = runner.invoke(app, ["--show-completion"])
    assert result.exit_code == 0
    assert "_assembly_completion" in result.output  # the emitted zsh completion script


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
    ctx = cmd.make_context("assembly", [], resilient_parsing=True)
    # The order shown under --help; hidden internal commands (e.g. _update-check,
    # the detached update-check refresh) aren't displayed, so exclude them.
    names = [n for n in cmd.list_commands(ctx) if not cmd.commands[n].hidden]
    # Grouped into Rich help panels (see help_panels.py): Quick Start, Build an App,
    # Run AssemblyAI, Setup & Tools, History, then Account.
    assert names == [
        # Quick Start
        "onboard",
        # Build an App
        "init",
        "dev",
        "share",
        "deploy",
        # Run AssemblyAI
        "transcribe",
        "stream",
        "dictate",
        "agent",
        "speak",
        "llm",
        "clip",
        "dub",
        "caption",
        "eval",
        "webhooks",
        # Setup & Tools
        "doctor",
        "setup",
        "config",
        "update",
        "telemetry",
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
