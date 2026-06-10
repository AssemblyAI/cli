import json
import subprocess

import pytest
from typer.testing import CliRunner

from aai_cli.main import app
from tests.setup_helpers import (
    FakeRun,
    _all_tools_present,
    _cli_skill_path,
    _skill_path,
    _statuses,
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Keep skill writes/reads inside a temp HOME so tests never touch ~/.claude."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)


@pytest.fixture(autouse=True)
def _force_json(monkeypatch):
    """These tests pin the structured step/status JSON. The CLI now defaults to human
    text everywhere (JSON is opt-in), so force the machine output the assertions parse —
    the equivalent of invoking each command with --json."""
    monkeypatch.setattr("aai_cli.output.resolve_json", lambda *, explicit: True)


# --- install: all three steps ------------------------------------------------


def test_install_happy_path_runs_all_steps(monkeypatch):
    _all_tools_present(monkeypatch)
    # MCP not yet present -> `mcp get` returns non-zero.
    fake = FakeRun({("claude", "mcp", "get"): 1})
    monkeypatch.setattr("aai_cli.commands.setup.subprocess.run", fake)

    result = runner.invoke(app, ["setup", "install"])
    assert result.exit_code == 0

    statuses = _statuses(result)
    assert statuses == {"mcp": "installed", "skill": "installed", "aai-cli skill": "installed"}

    assert [
        "claude",
        "mcp",
        "add",
        "--transport",
        "http",
        "--scope",
        "user",
        "assemblyai-docs",
        "https://mcp.assemblyai.com/docs",
    ] in fake.calls
    assert [
        "npx",
        "-y",
        "skills",
        "add",
        "AssemblyAI/assemblyai-skill",
        "--global",
        "--yes",
    ] in fake.calls
    # The bundled aai-cli skill was copied into HOME (no subprocess involved).
    assert (_cli_skill_path() / "SKILL.md").exists()


def test_install_skill_failed_when_npx_succeeds_but_nothing_installed(monkeypatch):
    # Regression: `install` must verify the skill landed, not trust npx's exit
    # code — otherwise install says "installed" while status says "not_installed".
    _all_tools_present(monkeypatch)
    fake = FakeRun({("claude", "mcp", "get"): 1}, creates_skill=False)
    monkeypatch.setattr("aai_cli.commands.setup.subprocess.run", fake)

    result = runner.invoke(app, ["setup", "install"])
    assert result.exit_code == 1  # skill step failed
    assert _statuses(result)["skill"] == "failed"
    # The detail quotes the install command starting at `add` (_SKILL_ADD[3:]), so the
    # user sees exactly what to retry -- pins that slice start.
    skill_detail = next(
        s["detail"] for s in json.loads(result.output)["steps"] if s["name"] == "skill"
    )
    assert "'add AssemblyAI/assemblyai-skill --global --yes'" in skill_detail

    # And status agrees: still not installed.
    status_result = runner.invoke(app, ["setup", "status"])
    assert _statuses(status_result)["skill"] == "not_installed"


def test_install_detaches_stdin_and_sets_timeout(monkeypatch):
    """Regression: subprocess children must not inherit stdin, or an interactive
    prompt (npx, claude) hangs the CLI forever. Each call must pass a timeout too."""
    _all_tools_present(monkeypatch)
    seen = []

    def record(cmd, *args, **kwargs):
        seen.append((list(cmd), kwargs))
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="")

    monkeypatch.setattr("aai_cli.commands.setup.subprocess.run", record)
    result = runner.invoke(app, ["setup", "install"])
    assert result.exit_code in (0, 1)
    assert seen, "expected subprocess.run to be called"
    for _cmd, kwargs in seen:
        assert kwargs.get("stdin") is subprocess.DEVNULL
        assert kwargs.get("timeout")
        assert kwargs.get("capture_output") is True  # stdout/stderr must be captured
        assert kwargs.get("text") is True  # decoded to str, not bytes
        assert kwargs.get("check") is False  # we inspect returncode, never raise

    # The skill download gets the longer 300s timeout (vs the 120s default elsewhere).
    add_calls = [kw for cmd, kw in seen if cmd[:1] == ["npx"] and "add" in cmd]
    assert add_calls and add_calls[0]["timeout"] == 300


def test_install_scope_passthrough(monkeypatch):
    _all_tools_present(monkeypatch)
    fake = FakeRun({("claude", "mcp", "get"): 1})
    monkeypatch.setattr("aai_cli.commands.setup.subprocess.run", fake)

    result = runner.invoke(app, ["setup", "install", "--scope", "project"])
    assert result.exit_code == 0
    assert [
        "claude",
        "mcp",
        "add",
        "--transport",
        "http",
        "--scope",
        "project",
        "assemblyai-docs",
        "https://mcp.assemblyai.com/docs",
    ] in fake.calls


def test_install_scope_local_passthrough(monkeypatch):
    _all_tools_present(monkeypatch)
    fake = FakeRun({("claude", "mcp", "get"): 1})
    monkeypatch.setattr("aai_cli.commands.setup.subprocess.run", fake)

    result = runner.invoke(app, ["setup", "install", "--scope", "local"])
    assert result.exit_code == 0
    assert [
        "claude",
        "mcp",
        "add",
        "--transport",
        "http",
        "--scope",
        "local",
        "assemblyai-docs",
        "https://mcp.assemblyai.com/docs",
    ] in fake.calls


def test_install_invalid_scope_exits_2(monkeypatch):
    _all_tools_present(monkeypatch)
    monkeypatch.setattr("aai_cli.commands.setup.subprocess.run", FakeRun())
    result = runner.invoke(app, ["setup", "install", "--scope", "bogus"])
    assert result.exit_code == 2


def test_install_idempotent_when_mcp_present(monkeypatch):
    _all_tools_present(monkeypatch)
    # `mcp get` returns 0 -> already registered.
    fake = FakeRun({("claude", "mcp", "get"): 0})
    monkeypatch.setattr("aai_cli.commands.setup.subprocess.run", fake)

    result = runner.invoke(app, ["setup", "install"])
    assert result.exit_code == 0
    assert _statuses(result)["mcp"] == "already"
    # No `mcp add` should have run.
    assert not any(c[:3] == ["claude", "mcp", "add"] for c in fake.calls)


def test_install_failure_exits_nonzero(monkeypatch):
    _all_tools_present(monkeypatch)
    # mcp not present, but `mcp add` fails.
    fake = FakeRun({("claude", "mcp", "get"): 1, ("claude", "mcp", "add"): 1})
    monkeypatch.setattr("aai_cli.commands.setup.subprocess.run", fake)

    result = runner.invoke(app, ["setup", "install"])
    assert result.exit_code == 1
    assert _statuses(result)["mcp"] == "failed"


def test_install_force_remove_failure_reports_failed(monkeypatch):
    _all_tools_present(monkeypatch)
    # present, but the forced remove fails
    fake = FakeRun({("claude", "mcp", "get"): 0, ("claude", "mcp", "remove"): 1})
    monkeypatch.setattr("aai_cli.commands.setup.subprocess.run", fake)

    result = runner.invoke(app, ["setup", "install", "--force"])
    assert result.exit_code == 1
    assert _statuses(result)["mcp"] == "failed"
    assert not any(c[:3] == ["claude", "mcp", "add"] for c in fake.calls)


def test_install_force_removes_then_adds(monkeypatch):
    _all_tools_present(monkeypatch)
    fake = FakeRun({("claude", "mcp", "get"): 0})
    monkeypatch.setattr("aai_cli.commands.setup.subprocess.run", fake)

    result = runner.invoke(app, ["setup", "install", "--force"])
    assert result.exit_code == 0
    assert ["claude", "mcp", "remove", "assemblyai-docs"] in fake.calls
    assert any(c[:3] == ["claude", "mcp", "add"] for c in fake.calls)


def test_install_skips_mcp_when_claude_missing(monkeypatch):
    monkeypatch.setattr(
        "aai_cli.commands.setup.shutil.which",
        lambda tool: None if tool == "claude" else f"/usr/bin/{tool}",
    )
    fake = FakeRun()
    monkeypatch.setattr("aai_cli.commands.setup.subprocess.run", fake)

    result = runner.invoke(app, ["setup", "install"])
    assert result.exit_code == 0  # skip is not a failure
    statuses = _statuses(result)
    assert statuses["mcp"] == "skipped"
    assert statuses["skill"] == "installed"
    # The bundled aai-cli skill installs regardless of claude/npx.
    assert statuses["aai-cli skill"] == "installed"
    assert not any(c[0] == "claude" for c in fake.calls)


# --- assemblyai skill (npx-based) --------------------------------------------


def test_install_skill_idempotent_when_present(monkeypatch):
    # Regression: a repeat install must report the skill as `already` (like MCP),
    # not re-run `npx skills add` and claim `installed` every time.
    _all_tools_present(monkeypatch)
    skill = _skill_path()
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# AssemblyAI")
    fake = FakeRun({("claude", "mcp", "get"): 1})
    monkeypatch.setattr("aai_cli.commands.setup.subprocess.run", fake)

    result = runner.invoke(app, ["setup", "install"])
    assert result.exit_code == 0
    assert _statuses(result)["skill"] == "already"
    # No `npx … add` should have run — the skill was already present.
    assert not any(c[0] == "npx" and "add" in c for c in fake.calls)


def test_install_force_reinstalls_skill(monkeypatch):
    # --force must re-run `npx skills add` even when the skill is already present.
    _all_tools_present(monkeypatch)
    skill = _skill_path()
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# AssemblyAI")
    fake = FakeRun({("claude", "mcp", "get"): 1})
    monkeypatch.setattr("aai_cli.commands.setup.subprocess.run", fake)

    result = runner.invoke(app, ["setup", "install", "--force"])
    assert result.exit_code == 0
    assert _statuses(result)["skill"] == "installed"
    assert [
        "npx",
        "-y",
        "skills",
        "add",
        "AssemblyAI/assemblyai-skill",
        "--global",
        "--yes",
    ] in fake.calls


def test_install_skips_skill_when_npx_missing(monkeypatch):
    monkeypatch.setattr(
        "aai_cli.commands.setup.shutil.which",
        lambda tool: None if tool == "npx" else f"/usr/bin/{tool}",
    )
    fake = FakeRun({("claude", "mcp", "get"): 1})
    monkeypatch.setattr("aai_cli.commands.setup.subprocess.run", fake)

    result = runner.invoke(app, ["setup", "install"])
    assert result.exit_code == 0
    statuses = _statuses(result)
    assert statuses["skill"] == "skipped"
    assert statuses["mcp"] == "installed"
    # aai-cli skill copies in regardless (no npx needed).
    assert statuses["aai-cli skill"] == "installed"
    assert not any(c[0] == "npx" for c in fake.calls)


# --- aai-cli skill (bundled, copied) -----------------------------------------


def test_install_aai_cli_skill_idempotent_when_present(monkeypatch):
    _all_tools_present(monkeypatch)
    cli_skill = _cli_skill_path()
    cli_skill.mkdir(parents=True)
    (cli_skill / "SKILL.md").write_text("# old")
    fake = FakeRun({("claude", "mcp", "get"): 1})
    monkeypatch.setattr("aai_cli.commands.setup.subprocess.run", fake)

    result = runner.invoke(app, ["setup", "install"])
    assert result.exit_code == 0
    assert _statuses(result)["aai-cli skill"] == "already"
    # Not overwritten without --force.
    assert (cli_skill / "SKILL.md").read_text() == "# old"


def test_install_aai_cli_skill_force_reinstalls(monkeypatch):
    _all_tools_present(monkeypatch)
    cli_skill = _cli_skill_path()
    cli_skill.mkdir(parents=True)
    (cli_skill / "SKILL.md").write_text("# old")
    fake = FakeRun({("claude", "mcp", "get"): 1})
    monkeypatch.setattr("aai_cli.commands.setup.subprocess.run", fake)

    result = runner.invoke(app, ["setup", "install", "--force"])
    assert result.exit_code == 0
    assert _statuses(result)["aai-cli skill"] == "installed"
    # Overwritten with the bundled copy (references/ exist; placeholder gone).
    assert (cli_skill / "references").is_dir()
    assert "# old" not in (cli_skill / "SKILL.md").read_text()


# --- status ------------------------------------------------------------------


def test_status_reports_all_installed(monkeypatch, tmp_path):
    _all_tools_present(monkeypatch)
    for name in ("assemblyai", "aai-cli"):
        d = tmp_path / ".claude" / "skills" / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("# x")
    # `mcp get` returns 0 -> present.
    monkeypatch.setattr(
        "aai_cli.commands.setup.subprocess.run",
        FakeRun({("claude", "mcp", "get"): 0}),
    )

    result = runner.invoke(app, ["setup", "status"])
    assert result.exit_code == 0
    assert _statuses(result) == {
        "mcp": "installed",
        "skill": "installed",
        "aai-cli skill": "installed",
    }


def test_status_reports_not_installed(monkeypatch):
    _all_tools_present(monkeypatch)  # no skill dirs created
    monkeypatch.setattr(
        "aai_cli.commands.setup.subprocess.run",
        FakeRun({("claude", "mcp", "get"): 1}),
    )

    result = runner.invoke(app, ["setup", "status"])
    assert result.exit_code == 0
    assert _statuses(result) == {
        "mcp": "not_installed",
        "skill": "not_installed",
        "aai-cli skill": "not_installed",
    }


def test_status_mcp_unknown_when_claude_missing(monkeypatch):
    monkeypatch.setattr(
        "aai_cli.commands.setup.shutil.which",
        lambda tool: None if tool == "claude" else f"/usr/bin/{tool}",
    )
    monkeypatch.setattr("aai_cli.commands.setup.subprocess.run", FakeRun())

    result = runner.invoke(app, ["setup", "status"])
    assert result.exit_code == 0
    assert _statuses(result)["mcp"] == "unknown"
