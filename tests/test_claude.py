import json
import subprocess

from typer.testing import CliRunner

from assemblyai_cli.main import app

runner = CliRunner()


class FakeRun:
    """Records subprocess calls and returns canned CompletedProcess results.

    `returncodes` maps a command prefix tuple (the first N argv tokens) to a
    return code; the longest matching prefix wins, default 0.
    """

    def __init__(self, returncodes=None):
        self.calls = []
        self.returncodes = returncodes or {}

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append(cmd)
        rc = 0
        best = -1
        for prefix, code in self.returncodes.items():
            n = len(prefix)
            if tuple(cmd[:n]) == prefix and n > best:
                rc, best = code, n
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
    assert ["npx", "skills", "add", "AssemblyAI/assemblyai-skill"] in fake.calls


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
