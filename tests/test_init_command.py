import io
import subprocess
import sys
import types

import pytest
import typer
from typer.testing import CliRunner

from aai_cli.commands import init as init_cmd
from aai_cli.errors import CLIError
from aai_cli.main import app

runner = CliRunner()
TEMPLATE = "audio-transcription"


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_init_scaffold_only_creates_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "myapp" / "api" / "index.py").exists()
    assert (tmp_path / "myapp" / ".env").exists()


def test_init_writes_key_from_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk-from-env")
    runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"])
    assert "ASSEMBLYAI_API_KEY=sk-from-env" in (tmp_path / "myapp" / ".env").read_text()


def test_init_logged_out_installs_but_skips_launch_with_hint(tmp_path, monkeypatch):
    # Logged out + install: deps install, but the server can't start without a key —
    # the report must say so (and must not launch) rather than exiting silently.

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "aai_cli.init.runner.run_setup",
        lambda *a, **k: subprocess.CompletedProcess([], 0, "", ""),
    )
    launched = {"v": False}
    monkeypatch.setattr(
        "aai_cli.init.runner.launch_and_open",
        lambda *a, **k: launched.__setitem__("v", True) or 0,
    )
    result = runner.invoke(app, ["init", TEMPLATE, "app"])  # no --no-install, logged out
    assert result.exit_code == 0, result.output
    assert launched["v"] is False
    assert "aai login" in result.output


def test_init_writes_base_url_for_active_env(tmp_path, monkeypatch):
    # The generated .env pins the app to the CLI's active environment hosts.
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"])
    env = (tmp_path / "myapp" / ".env").read_text()
    assert "ASSEMBLYAI_BASE_URL=https://" in env
    assert "ASSEMBLYAI_LLM_GATEWAY_URL=https://" in env


def test_init_placeholder_key_when_logged_out(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"])
    env = (tmp_path / "myapp" / ".env").read_text()
    assert "your_assemblyai_api_key_here" in env


def test_init_unknown_template_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "nope", "myapp", "--no-install"])
    assert result.exit_code == 1


def test_init_refuses_nonempty_dir_without_force(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"]).exit_code == 0
    result = runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"])
    assert result.exit_code == 1


def test_init_force_overwrites(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"]).exit_code == 0
    result = runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install", "--force"])
    assert result.exit_code == 0


def test_init_no_template_non_interactive_errors(tmp_path, monkeypatch):
    # CliRunner has no TTY, so the picker can't run; bare `aai init` must error helpfully.
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init"])
    assert result.exit_code != 0
    assert TEMPLATE in result.output  # lists the available templates


def test_init_default_dir_is_template_slug(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", TEMPLATE, "--no-install"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / TEMPLATE / "api" / "index.py").exists()


def test_init_appears_in_help():
    result = runner.invoke(app, ["--help"])
    assert "init" in result.output


def test_init_prints_cli_banner_in_human_mode(tmp_path, monkeypatch):
    # Vercel-style header at the top of an interactive run (human output only).
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("aai_cli.output._is_agentic", lambda: False)
    result = runner.invoke(app, ["init", TEMPLATE, "x", "--no-install"])
    assert result.exit_code == 0, result.output
    assert "AssemblyAI CLI" in result.output


def test_init_here_scaffolds_into_cwd(tmp_path, monkeypatch):
    work = tmp_path / "work"  # a fresh empty dir (tmp_path itself holds the test config/)
    work.mkdir()
    monkeypatch.chdir(work)
    result = runner.invoke(app, ["init", TEMPLATE, "--here", "--no-install"])
    assert result.exit_code == 0, result.output
    assert (work / "api" / "index.py").exists()  # scaffolded into cwd, no subdir


def test_init_here_with_directory_errors(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    result = runner.invoke(app, ["init", TEMPLATE, "mydir", "--here", "--no-install"])
    assert result.exit_code != 0  # ambiguous: directory AND --here both given


def test_init_json_output_is_machine_readable(tmp_path, monkeypatch):
    import json

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert any(step["name"] == "scaffold" and step["status"] == "created" for step in payload)


def test_init_unregistered_template_errors_cleanly(tmp_path, monkeypatch):
    # `llm` isn't a separate template (it's folded into audio-transcription), so
    # picking it must give a clean error, not a FileNotFoundError.
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "llm", "x", "--no-install"])
    assert result.exit_code == 1
    assert "llm" in result.output


def _fake_questionary(choice):
    """A minimal stand-in for the questionary module's select(...).ask() chain."""

    class _Choice:
        def __init__(self, title, value):
            self.value = value

    class _Select:
        def ask(self):
            return choice

    return types.SimpleNamespace(Choice=_Choice, select=lambda *a, **k: _Select())


def test_pick_template_interactive_returns_choice(monkeypatch):
    monkeypatch.setattr("sys.stdin", _Tty())
    monkeypatch.setattr("sys.stdout", _Tty())
    monkeypatch.setitem(sys.modules, "questionary", _fake_questionary(TEMPLATE))
    assert init_cmd._pick_template() == TEMPLATE


def test_pick_template_ctrl_c_exits_130(monkeypatch):
    # questionary returns None when the user presses Ctrl-C at the prompt.
    monkeypatch.setattr("sys.stdin", _Tty())
    monkeypatch.setattr("sys.stdout", _Tty())
    monkeypatch.setitem(sys.modules, "questionary", _fake_questionary(None))
    with pytest.raises(typer.Exit) as exc:
        init_cmd._pick_template()
    assert exc.value.exit_code == 130


def test_pick_template_missing_questionary_errors(monkeypatch):
    # A broken/stale install missing the declared 'questionary' dep.
    monkeypatch.setattr("sys.stdin", _Tty())
    monkeypatch.setattr("sys.stdout", _Tty())
    monkeypatch.setitem(sys.modules, "questionary", None)  # makes `import questionary` raise
    with pytest.raises(CLIError) as exc:
        init_cmd._pick_template()
    assert exc.value.error_type == "missing_dependency"


def test_init_install_failure_reports_and_exits(tmp_path, monkeypatch):
    # A failing dependency install is reported and exits non-zero (no launch).
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "aai_cli.init.runner.run_setup",
        lambda *a, **k: subprocess.CompletedProcess([], 1, "", "pip exploded"),
    )
    launched = {"v": False}
    monkeypatch.setattr(
        "aai_cli.init.runner.launch_and_open",
        lambda *a, **k: launched.__setitem__("v", True) or 0,
    )
    result = runner.invoke(app, ["init", TEMPLATE, "app", "--json"])
    assert result.exit_code == 1
    assert launched["v"] is False
    assert "pip exploded" in result.output


def test_init_launches_when_key_present(tmp_path, monkeypatch):
    # Key present + install succeeds -> the server is launched and the browser opens.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk-real-key")
    monkeypatch.setattr("aai_cli.output._is_agentic", lambda: False)  # exercise human banner
    monkeypatch.setattr(
        "aai_cli.init.runner.run_setup",
        lambda *a, **k: subprocess.CompletedProcess([], 0, "ok", ""),
    )
    monkeypatch.setattr("aai_cli.init.runner.find_free_port", lambda preferred: 4321)
    captured = {}
    monkeypatch.setattr(
        "aai_cli.init.runner.launch_and_open",
        lambda target, **k: captured.update(k) or 0,
    )
    result = runner.invoke(app, ["init", TEMPLATE, "app"])
    assert result.exit_code == 0, result.output
    assert captured["port"] == 4321
    assert captured["open_browser"] is True
    assert "Starting" in result.output
    assert "4321" in result.output


def test_init_no_open_keeps_browser_closed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk-real-key")
    monkeypatch.setattr(
        "aai_cli.init.runner.run_setup",
        lambda *a, **k: subprocess.CompletedProcess([], 0, "ok", ""),
    )
    monkeypatch.setattr("aai_cli.init.runner.find_free_port", lambda preferred: 4321)
    captured = {}
    monkeypatch.setattr(
        "aai_cli.init.runner.launch_and_open",
        lambda target, **k: captured.update(k) or 0,
    )
    result = runner.invoke(app, ["init", TEMPLATE, "app", "--no-open"])
    assert result.exit_code == 0, result.output
    assert captured["open_browser"] is False


def test_init_launch_failure_propagates_exit_code(tmp_path, monkeypatch):
    # A non-zero server exit code is surfaced as the command's exit code.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk-real-key")
    monkeypatch.setattr(
        "aai_cli.init.runner.run_setup",
        lambda *a, **k: subprocess.CompletedProcess([], 0, "ok", ""),
    )
    monkeypatch.setattr("aai_cli.init.runner.find_free_port", lambda preferred: 4321)
    monkeypatch.setattr("aai_cli.init.runner.launch_and_open", lambda *a, **k: 7)
    result = runner.invoke(app, ["init", TEMPLATE, "app"])
    assert result.exit_code == 7
