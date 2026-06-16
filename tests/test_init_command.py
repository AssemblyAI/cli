import io
import subprocess
import sys
import types

import pytest
import typer
from typer.testing import CliRunner

from aai_cli.app import init_exec
from aai_cli.commands import init as init_cmd
from aai_cli.core.errors import CLIError
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


def test_init_rejects_dir_and_here_together(tmp_path, monkeypatch):
    # DIRECTORY and --here are mutually exclusive; passing both is a usage error
    # exiting 2 (pins that exit_code on the conflict).
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", TEMPLATE, "somedir", "--here", "--no-install"])
    assert result.exit_code == 2
    assert "not both" in result.output


def test_init_writes_key_from_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk-from-env")
    result = runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"])
    assert "ASSEMBLYAI_API_KEY=sk-from-env" in (tmp_path / "myapp" / ".env").read_text()
    # A resolved key means no skipped-key row (pins `if api_key is None`).
    assert "no API key found" not in result.output


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
    assert "assembly login" in result.output
    # Deps installed but no key -> a launch-skipped row with the manual run command
    # (pins `not no_install and api_key is None`).
    assert "assembly dev" in result.output


def test_init_logged_out_install_emits_launch_skipped_step_json(tmp_path, monkeypatch):
    # In --json mode the only signal for the no-key launch guard is the structured
    # "launch"/"skipped" step (human mode prints the uvicorn hint regardless), so this
    # specifically pins the `api_key is None` half of the guard.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "aai_cli.init.runner.run_setup",
        lambda *a, **k: subprocess.CompletedProcess([], 0, "", ""),
    )
    result = runner.invoke(app, ["init", TEMPLATE, "app", "--json"])  # install, logged out
    assert result.exit_code == 0, result.output
    assert '"name": "launch"' in result.output
    assert '"status": "skipped"' in result.output
    assert "no API key" in result.output


def test_init_writes_base_url_for_active_env(tmp_path, monkeypatch):
    # The generated .env pins the app to the CLI's active environment hosts.
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"])
    env = (tmp_path / "myapp" / ".env").read_text()
    assert "ASSEMBLYAI_BASE_URL=https://" in env
    assert "ASSEMBLYAI_LLM_GATEWAY_URL=https://" in env


def test_init_placeholder_key_when_logged_out(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # --json: human output always prints a "run it with uvicorn" next-step hint, so the
    # launch-skipped row only shows up as a distinct step in the structured output.
    result = runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install", "--json"])
    env = (tmp_path / "myapp" / ".env").read_text()
    assert "your_assemblyai_api_key_here" in env
    # --no-install means no deps were installed, so there's no launch-skipped row even
    # without a key (pins the `not no_install` half of the launch guard).
    assert "assembly dev" not in result.output


def test_init_unknown_template_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "nope", "myapp", "--no-install"])
    assert result.exit_code == 2


def test_init_refuses_nonempty_dir_without_force(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"]).exit_code == 0
    result = runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"])
    assert result.exit_code == 2


def test_init_no_template_non_interactive_errors(tmp_path, monkeypatch):
    # CliRunner has no TTY, so the picker can't run; bare `assembly init` must error helpfully.
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init"])
    assert result.exit_code != 0
    assert TEMPLATE in result.output  # lists the available templates


def test_init_port_out_of_range_is_rejected_before_scaffolding(tmp_path, monkeypatch):
    # A bad --port used to surface only at launch (after a wasted scaffold+install) as an
    # internal "report a bug" error. Typer now rejects it up front with a usage error (2),
    # before any directory is created. Pins the max=65535 bound.
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install", "--port", "65536"])
    assert result.exit_code == 2
    assert not (tmp_path / "myapp").exists()  # rejected before scaffolding


def test_init_port_zero_is_accepted(tmp_path, monkeypatch):
    # Port 0 means "let the OS assign a free port" — it must stay valid (pins min=0, not 1).
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install", "--port", "0"])
    assert result.exit_code == 0, result.output


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
    result = runner.invoke(app, ["init", TEMPLATE, "x", "--no-install"])
    assert result.exit_code == 0, result.output
    # Decoration goes to stderr (data → stdout), so a piped stdout never sees it.
    assert "AssemblyAI CLI" in result.stderr
    assert "AssemblyAI CLI" not in result.stdout


def test_init_banner_skipped_on_error_only_runs(tmp_path, monkeypatch):
    # The banner prints only after validation passes: a pure error run (unknown
    # template) stays undecorated like the sibling commands, and stdout stays empty.
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "nope", "x", "--no-install"])
    assert result.exit_code == 2
    assert "AssemblyAI CLI" not in result.stderr
    assert result.stdout == ""


def test_init_banner_skipped_on_target_conflict_error(tmp_path, monkeypatch):
    # Target validation failures are error-only runs too: no banner.
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"]).exit_code == 0
    result = runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"])
    assert result.exit_code == 2
    assert "AssemblyAI CLI" not in result.stderr


def test_init_help_enumerates_template_names():
    import re

    from aai_cli.init import templates

    result = runner.invoke(app, ["init", "--help"])
    assert result.exit_code == 0
    # Strip ANSI (CI forces color) and unwrap lines before matching. Every template
    # name from the registry must be visible (live-captions appears nowhere else in
    # the help, so this pins the enumeration, not the Examples epilog).
    flat = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", result.output).split())
    for name in templates.TEMPLATE_ORDER:
        assert name in flat


def test_init_template_arg_help_is_derived_from_registry():
    # The exact help string, render-independent: derived from TEMPLATE_ORDER so the
    # enumeration can never drift from the templates that actually ship.
    import inspect

    from typer.models import ArgumentInfo

    default = inspect.signature(init_cmd.init).parameters["template"].default
    assert isinstance(default, ArgumentInfo)
    assert default.help == (
        "Template to scaffold: audio-transcription, live-captions, voice-agent, "
        "agent-cascade (omit to pick interactively)"
    )


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
    assert result.exit_code == 2
    assert "llm" in result.output


def _fake_questionary(choice):
    """A minimal stand-in for the questionary module's select(...).ask() chain.

    The returned namespace records the choices select(...) was called with on its
    ``choices`` attribute (read it back in a test to inspect titles/descriptions)."""

    ns = types.SimpleNamespace(Choice=None, select=None, choices=None)

    class _Choice:
        def __init__(self, title, value, description=None):
            self.title = title
            self.value = value
            self.description = description

    class _Select:
        def ask(self):
            return choice

    def _select(*_a, choices=None, **_k):
        ns.choices = choices
        return _Select()

    ns.Choice = _Choice
    ns.select = _select
    return ns


def test_pick_template_interactive_returns_choice(monkeypatch):
    monkeypatch.setattr("sys.stdin", _Tty())
    monkeypatch.setattr("sys.stdout", _Tty())
    monkeypatch.setitem(sys.modules, "questionary", _fake_questionary(TEMPLATE))
    assert init_exec._pick_template() == TEMPLATE


def test_pick_template_choices_carry_descriptions(monkeypatch):
    # Each picker choice wires the registry's title + description for that template.
    from aai_cli.init import templates

    monkeypatch.setattr("sys.stdin", _Tty())
    monkeypatch.setattr("sys.stdout", _Tty())
    fake = _fake_questionary(TEMPLATE)
    monkeypatch.setitem(sys.modules, "questionary", fake)
    init_exec._pick_template()
    choices = fake.choices
    assert [c.value for c in choices] == list(templates.TEMPLATE_ORDER)
    assert all(c.description == templates.description_for(c.value) for c in choices)
    assert all(c.description for c in choices)  # every template advertises a description


def test_pick_template_ctrl_c_exits_130(monkeypatch):
    # questionary returns None when the user presses Ctrl-C at the prompt.
    monkeypatch.setattr("sys.stdin", _Tty())
    monkeypatch.setattr("sys.stdout", _Tty())
    monkeypatch.setitem(sys.modules, "questionary", _fake_questionary(None))
    with pytest.raises(typer.Exit) as exc:
        init_exec._pick_template()
    assert exc.value.exit_code == 130


def test_pick_template_missing_questionary_errors(monkeypatch):
    # A broken/stale install missing the declared 'questionary' dep.
    monkeypatch.setattr("sys.stdin", _Tty())
    monkeypatch.setattr("sys.stdout", _Tty())
    monkeypatch.setitem(sys.modules, "questionary", None)  # makes `import questionary` raise
    with pytest.raises(CLIError) as exc:
        init_exec._pick_template()
    assert exc.value.error_type == "missing_dependency"
    assert exc.value.exit_code == 1


@pytest.mark.parametrize(("stdin_tty", "stdout_tty"), [(True, False), (False, True)])
def test_pick_template_errors_when_either_stream_not_a_tty(monkeypatch, stdin_tty, stdout_tty):
    # The picker needs BOTH stdin and stdout interactive; if either is piped it must
    # bail with a usage error (pins the `or`, which an `and` would weaken to "both
    # piped"). questionary is stubbed out so a mutated fall-through is observable as a
    # *different* error rather than the usage error this asserts.
    monkeypatch.setattr("sys.stdin", _Tty() if stdin_tty else io.StringIO())
    monkeypatch.setattr("sys.stdout", _Tty() if stdout_tty else io.StringIO())
    monkeypatch.setitem(sys.modules, "questionary", None)
    with pytest.raises(CLIError) as exc:
        init_exec._pick_template()
    assert exc.value.error_type == "usage_error"
    assert exc.value.exit_code == 2


def test_active_env_vars_agents_host_replaces_only_first_streaming(monkeypatch):
    # The agents host is derived by swapping the FIRST "streaming" token for "agents"
    # (replace count=1); a host containing it twice must keep the later occurrence.
    fake_env = types.SimpleNamespace(
        api_base="https://api.x",
        llm_gateway_base="https://llm.x",
        streaming_host="streaming.streaming.example.com",
        streaming_tts_host="",
    )
    monkeypatch.setattr(init_exec.environments, "active", lambda: fake_env)
    assert init_exec._active_env_vars()["ASSEMBLYAI_AGENTS_HOST"] == "agents.streaming.example.com"


def test_active_env_vars_includes_streaming_tts_host(monkeypatch):
    fake_env = types.SimpleNamespace(
        api_base="https://api.x",
        llm_gateway_base="https://llm.x/v1",
        streaming_host="streaming.x",
        streaming_tts_host="streaming-tts.x",
    )
    monkeypatch.setattr(init_exec.environments, "active", lambda: fake_env)
    assert init_exec._active_env_vars()["ASSEMBLYAI_TTS_HOST"] == "streaming-tts.x"


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


def test_init_install_failure_does_not_launch_even_with_key(tmp_path, monkeypatch):
    # A failed install must flip will_launch off so the server never starts -- even
    # when a key is present (which would otherwise satisfy the launch guard). Pins the
    # literal `False` returned on the failure branch.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk-real-key")
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


def test_init_install_failure_detail_is_truncated(tmp_path, monkeypatch):
    # A pathologically long install error is capped at 300 chars in the report detail
    # so it can't flood the terminal; pins the [:300] slice.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "aai_cli.init.runner.run_setup",
        lambda *a, **k: subprocess.CompletedProcess([], 1, "", "x" * 500),
    )
    result = runner.invoke(app, ["init", TEMPLATE, "app", "--json"])
    assert result.exit_code == 1
    assert "x" * 300 in result.output
    assert "x" * 301 not in result.output


def test_init_launches_when_key_present(tmp_path, monkeypatch):
    # Key present + install succeeds -> the server is launched and the browser opens.
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
