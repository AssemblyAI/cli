import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from assemblyai_cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Keep skill writes/reads inside a temp HOME so tests never touch ~/.claude."""
    monkeypatch.setenv("HOME", str(tmp_path))


class FakeRun:
    """Records subprocess calls and returns canned CompletedProcess results.

    `returncodes` maps a command prefix tuple (the first N argv tokens) to a
    return code; the longest matching prefix wins, default 0. When
    `creates_skill` is set, a successful `npx … add` materializes a SKILL.md
    under HOME — mimicking a real install so `_install_skill`'s filesystem
    verification passes.
    """

    def __init__(self, returncodes=None, *, creates_skill=True):
        self.calls = []
        self.returncodes = returncodes or {}
        self.creates_skill = creates_skill

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append(cmd)
        rc = 0
        best = -1
        for prefix, code in self.returncodes.items():
            n = len(prefix)
            if tuple(cmd[:n]) == prefix and n > best:
                rc, best = code, n
        if rc == 0 and self.creates_skill and cmd[:1] == ["npx"] and "add" in cmd:
            skill = Path.home() / ".claude" / "skills" / "assemblyai"
            skill.mkdir(parents=True, exist_ok=True)
            (skill / "SKILL.md").write_text("# AssemblyAI")
        return subprocess.CompletedProcess(args=cmd, returncode=rc, stdout="", stderr="boom")


def _all_tools_present(monkeypatch):
    monkeypatch.setattr(
        "assemblyai_cli.commands.claude.shutil.which",
        lambda tool: f"/usr/bin/{tool}",
    )


def test_install_happy_path_runs_both_steps(monkeypatch):
    _all_tools_present(monkeypatch)
    # MCP not yet present -> `mcp get` returns non-zero.
    fake = FakeRun({("claude", "mcp", "get"): 1})
    monkeypatch.setattr("assemblyai_cli.commands.claude.subprocess.run", fake)

    result = runner.invoke(app, ["claude", "install"])
    assert result.exit_code == 0

    payload = json.loads(result.output)
    statuses = {s["name"]: s["status"] for s in payload["steps"]}
    assert statuses == {"mcp": "installed", "skill": "installed"}

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
    assert ["npx", "-y", "skills", "add", "AssemblyAI/assemblyai-skill"] in fake.calls


def test_install_skill_failed_when_npx_succeeds_but_nothing_installed(monkeypatch):
    # Regression: `install` must verify the skill landed, not trust npx's exit
    # code — otherwise install says "installed" while status says "not_installed".
    _all_tools_present(monkeypatch)
    fake = FakeRun({("claude", "mcp", "get"): 1}, creates_skill=False)
    monkeypatch.setattr("assemblyai_cli.commands.claude.subprocess.run", fake)

    result = runner.invoke(app, ["claude", "install"])
    assert result.exit_code == 1  # skill step failed
    statuses = {s["name"]: s["status"] for s in json.loads(result.output)["steps"]}
    assert statuses["skill"] == "failed"

    # And status agrees: still not installed.
    status_result = runner.invoke(app, ["claude", "status"])
    skill = {s["name"]: s["status"] for s in json.loads(status_result.output)["steps"]}["skill"]
    assert skill == "not_installed"


def test_install_detaches_stdin_and_sets_timeout(monkeypatch):
    """Regression: subprocess children must not inherit stdin, or an interactive
    prompt (npx, claude) hangs the CLI forever. Each call must pass a timeout too."""
    _all_tools_present(monkeypatch)
    seen = []

    def record(cmd, *args, **kwargs):
        seen.append(kwargs)
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="")

    monkeypatch.setattr("assemblyai_cli.commands.claude.subprocess.run", record)
    result = runner.invoke(app, ["claude", "install"])
    assert result.exit_code in (0, 1)
    assert seen, "expected subprocess.run to be called"
    for kwargs in seen:
        assert kwargs.get("stdin") is subprocess.DEVNULL
        assert kwargs.get("timeout")


def test_install_scope_passthrough(monkeypatch):
    _all_tools_present(monkeypatch)
    fake = FakeRun({("claude", "mcp", "get"): 1})
    monkeypatch.setattr("assemblyai_cli.commands.claude.subprocess.run", fake)

    result = runner.invoke(app, ["claude", "install", "--scope", "project"])
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


def test_install_invalid_scope_exits_2(monkeypatch):
    _all_tools_present(monkeypatch)
    monkeypatch.setattr("assemblyai_cli.commands.claude.subprocess.run", FakeRun())
    result = runner.invoke(app, ["claude", "install", "--scope", "bogus"])
    assert result.exit_code == 2


def test_install_idempotent_when_mcp_present(monkeypatch):
    _all_tools_present(monkeypatch)
    # `mcp get` returns 0 -> already registered.
    fake = FakeRun({("claude", "mcp", "get"): 0})
    monkeypatch.setattr("assemblyai_cli.commands.claude.subprocess.run", fake)

    result = runner.invoke(app, ["claude", "install"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    statuses = {s["name"]: s["status"] for s in payload["steps"]}
    assert statuses["mcp"] == "already"
    # No `mcp add` should have run.
    assert not any(c[:3] == ["claude", "mcp", "add"] for c in fake.calls)


def test_install_force_removes_then_adds(monkeypatch):
    _all_tools_present(monkeypatch)
    fake = FakeRun({("claude", "mcp", "get"): 0})
    monkeypatch.setattr("assemblyai_cli.commands.claude.subprocess.run", fake)

    result = runner.invoke(app, ["claude", "install", "--force"])
    assert result.exit_code == 0
    assert ["claude", "mcp", "remove", "assemblyai-docs"] in fake.calls
    assert any(c[:3] == ["claude", "mcp", "add"] for c in fake.calls)


def test_install_skips_mcp_when_claude_missing(monkeypatch):
    monkeypatch.setattr(
        "assemblyai_cli.commands.claude.shutil.which",
        lambda tool: None if tool == "claude" else f"/usr/bin/{tool}",
    )
    fake = FakeRun()
    monkeypatch.setattr("assemblyai_cli.commands.claude.subprocess.run", fake)

    result = runner.invoke(app, ["claude", "install"])
    assert result.exit_code == 0  # skip is not a failure
    payload = json.loads(result.output)
    statuses = {s["name"]: s["status"] for s in payload["steps"]}
    assert statuses["mcp"] == "skipped"
    assert statuses["skill"] == "installed"
    assert not any(c[0] == "claude" for c in fake.calls)


def test_install_skips_skill_when_npx_missing(monkeypatch):
    monkeypatch.setattr(
        "assemblyai_cli.commands.claude.shutil.which",
        lambda tool: None if tool == "npx" else f"/usr/bin/{tool}",
    )
    fake = FakeRun({("claude", "mcp", "get"): 1})
    monkeypatch.setattr("assemblyai_cli.commands.claude.subprocess.run", fake)

    result = runner.invoke(app, ["claude", "install"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    statuses = {s["name"]: s["status"] for s in payload["steps"]}
    assert statuses["skill"] == "skipped"
    assert statuses["mcp"] == "installed"
    assert not any(c[0] == "npx" for c in fake.calls)


def test_install_failure_exits_nonzero(monkeypatch):
    _all_tools_present(monkeypatch)
    # mcp not present, but `mcp add` fails.
    fake = FakeRun({("claude", "mcp", "get"): 1, ("claude", "mcp", "add"): 1})
    monkeypatch.setattr("assemblyai_cli.commands.claude.subprocess.run", fake)

    result = runner.invoke(app, ["claude", "install"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    statuses = {s["name"]: s["status"] for s in payload["steps"]}
    assert statuses["mcp"] == "failed"


def test_install_force_remove_failure_reports_failed(monkeypatch):
    _all_tools_present(monkeypatch)
    # present, but the forced remove fails
    fake = FakeRun({("claude", "mcp", "get"): 0, ("claude", "mcp", "remove"): 1})
    monkeypatch.setattr("assemblyai_cli.commands.claude.subprocess.run", fake)

    result = runner.invoke(app, ["claude", "install", "--force"])
    assert result.exit_code == 1
    statuses = {s["name"]: s["status"] for s in json.loads(result.output)["steps"]}
    assert statuses["mcp"] == "failed"
    assert not any(c[:3] == ["claude", "mcp", "add"] for c in fake.calls)


def test_status_reports_both_installed(monkeypatch, tmp_path):
    _all_tools_present(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    skill = tmp_path / ".claude" / "skills" / "assemblyai"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# AssemblyAI")
    # `mcp get` returns 0 -> present.
    monkeypatch.setattr(
        "assemblyai_cli.commands.claude.subprocess.run",
        FakeRun({("claude", "mcp", "get"): 0}),
    )

    result = runner.invoke(app, ["claude", "status"])
    assert result.exit_code == 0
    statuses = {s["name"]: s["status"] for s in json.loads(result.output)["steps"]}
    assert statuses == {"mcp": "installed", "skill": "installed"}


def test_status_reports_not_installed(monkeypatch, tmp_path):
    _all_tools_present(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))  # no skill dir created
    monkeypatch.setattr(
        "assemblyai_cli.commands.claude.subprocess.run",
        FakeRun({("claude", "mcp", "get"): 1}),
    )

    result = runner.invoke(app, ["claude", "status"])
    assert result.exit_code == 0
    statuses = {s["name"]: s["status"] for s in json.loads(result.output)["steps"]}
    assert statuses == {"mcp": "not_installed", "skill": "not_installed"}


def test_status_mcp_unknown_when_claude_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "assemblyai_cli.commands.claude.shutil.which",
        lambda tool: None if tool == "claude" else f"/usr/bin/{tool}",
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("assemblyai_cli.commands.claude.subprocess.run", FakeRun())

    result = runner.invoke(app, ["claude", "status"])
    assert result.exit_code == 0
    statuses = {s["name"]: s["status"] for s in json.loads(result.output)["steps"]}
    assert statuses["mcp"] == "unknown"


def test_remove_unwinds_both(monkeypatch, tmp_path):
    _all_tools_present(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    skill = tmp_path / ".claude" / "skills" / "assemblyai"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# AssemblyAI")
    fake = FakeRun({("claude", "mcp", "get"): 0})  # present -> removable
    monkeypatch.setattr("assemblyai_cli.commands.claude.subprocess.run", fake)

    result = runner.invoke(app, ["claude", "remove"])
    assert result.exit_code == 0
    statuses = {s["name"]: s["status"] for s in json.loads(result.output)["steps"]}
    assert statuses == {"mcp": "removed", "skill": "removed"}
    assert ["claude", "mcp", "remove", "assemblyai-docs"] in fake.calls
    assert not skill.exists()


def test_remove_when_absent_is_not_an_error(monkeypatch, tmp_path):
    _all_tools_present(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))  # no skill dir
    fake = FakeRun({("claude", "mcp", "get"): 1})  # absent
    monkeypatch.setattr("assemblyai_cli.commands.claude.subprocess.run", fake)

    result = runner.invoke(app, ["claude", "remove"])
    assert result.exit_code == 0
    statuses = {s["name"]: s["status"] for s in json.loads(result.output)["steps"]}
    assert statuses == {"mcp": "not_installed", "skill": "not_installed"}
    assert not any(c[:3] == ["claude", "mcp", "remove"] for c in fake.calls)


def test_remove_skill_failure_reports_failed(monkeypatch, tmp_path):
    _all_tools_present(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    skill = tmp_path / ".claude" / "skills" / "assemblyai"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# AssemblyAI")
    # MCP absent so only the skill step can fail.
    monkeypatch.setattr(
        "assemblyai_cli.commands.claude.subprocess.run",
        FakeRun({("claude", "mcp", "get"): 1}),
    )

    def boom(_path):
        raise PermissionError("locked")

    monkeypatch.setattr("assemblyai_cli.commands.claude.shutil.rmtree", boom)

    result = runner.invoke(app, ["claude", "remove"])
    assert result.exit_code == 1
    statuses = {s["name"]: s["status"] for s in json.loads(result.output)["steps"]}
    assert statuses["skill"] == "failed"


def test_install_scope_local_passthrough(monkeypatch):
    _all_tools_present(monkeypatch)
    fake = FakeRun({("claude", "mcp", "get"): 1})
    monkeypatch.setattr("assemblyai_cli.commands.claude.subprocess.run", fake)

    result = runner.invoke(app, ["claude", "install", "--scope", "local"])
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


def test_remove_scope_passthrough(monkeypatch, tmp_path):
    _all_tools_present(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    fake = FakeRun({("claude", "mcp", "get"): 0})  # present
    monkeypatch.setattr("assemblyai_cli.commands.claude.subprocess.run", fake)

    result = runner.invoke(app, ["claude", "remove", "--scope", "project"])
    assert result.exit_code == 0
    assert ["claude", "mcp", "remove", "assemblyai-docs", "--scope", "project"] in fake.calls


def test_claude_help_lists_all_subcommands():
    result = runner.invoke(app, ["claude", "--help"])
    assert result.exit_code == 0
    assert "install" in result.output
    assert "status" in result.output
    assert "remove" in result.output


def test_claude_no_subcommand_lists_commands():
    # Bare `aai claude` should show its commands instead of "Missing command".
    result = runner.invoke(app, ["claude"])
    assert "install" in result.output
    assert "status" in result.output
    assert "remove" in result.output
