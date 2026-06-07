import json
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aai_cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Keep skill writes/reads inside a temp HOME so tests never touch ~/.claude."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)


def test_proc_detail_prefers_stderr_then_falls_back_to_stdout():
    from aai_cli.commands import setup

    # stderr wins when present (pins `proc.stderr or proc.stdout`); stdout is the
    # fallback when stderr is empty.
    both = subprocess.CompletedProcess([], 1, stdout="out text", stderr="err text")
    assert setup._proc_detail(both) == "err text"
    only_out = subprocess.CompletedProcess([], 1, stdout="only out", stderr="")
    assert setup._proc_detail(only_out) == "only out"


def _skill_path() -> Path:
    return Path.home() / ".claude" / "skills" / "assemblyai"


def _cli_skill_path() -> Path:
    return Path.home() / ".claude" / "skills" / "aai-cli"


class FakeRun:
    """Records subprocess calls and returns canned CompletedProcess results.

    `returncodes` maps a command prefix tuple (the first N argv tokens) to a
    return code; the longest matching prefix wins, default 0. To mimic the real
    `skills` CLI, a successful `npx … add` materializes the assemblyai skill under
    HOME (so `_install_skill`'s filesystem check passes) and `npx … remove`
    deletes it — toggle with `creates_skill` / `removes_skill`. The aai-cli skill
    is bundled and copied directly (no subprocess), so it never goes through here.
    """

    def __init__(self, returncodes=None, *, creates_skill=True, removes_skill=True):
        self.calls = []
        self.returncodes = returncodes or {}
        self.creates_skill = creates_skill
        self.removes_skill = removes_skill

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append(cmd)
        rc = 0
        best = -1
        for prefix, code in self.returncodes.items():
            n = len(prefix)
            if tuple(cmd[:n]) == prefix and n > best:
                rc, best = code, n
        if rc == 0 and cmd[:1] == ["npx"]:
            if "add" in cmd and self.creates_skill:
                _skill_path().mkdir(parents=True, exist_ok=True)
                (_skill_path() / "SKILL.md").write_text("# AssemblyAI")
            elif "remove" in cmd and self.removes_skill:
                shutil.rmtree(_skill_path(), ignore_errors=True)
        return subprocess.CompletedProcess(args=cmd, returncode=rc, stdout="", stderr="boom")


def _all_tools_present(monkeypatch):
    monkeypatch.setattr(
        "aai_cli.commands.setup.shutil.which",
        lambda tool: f"/usr/bin/{tool}",
    )


def _statuses(result):
    return {s["name"]: s["status"] for s in json.loads(result.output)["steps"]}


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

    # And status agrees: still not installed.
    status_result = runner.invoke(app, ["setup", "status"])
    assert _statuses(status_result)["skill"] == "not_installed"


def test_install_detaches_stdin_and_sets_timeout(monkeypatch):
    """Regression: subprocess children must not inherit stdin, or an interactive
    prompt (npx, claude) hangs the CLI forever. Each call must pass a timeout too."""
    _all_tools_present(monkeypatch)
    seen = []

    def record(cmd, *args, **kwargs):
        seen.append(kwargs)
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="")

    monkeypatch.setattr("aai_cli.commands.setup.subprocess.run", record)
    result = runner.invoke(app, ["setup", "install"])
    assert result.exit_code in (0, 1)
    assert seen, "expected subprocess.run to be called"
    for kwargs in seen:
        assert kwargs.get("stdin") is subprocess.DEVNULL
        assert kwargs.get("timeout")


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


def test_remove_skill_failure_reports_failed(monkeypatch):
    _all_tools_present(monkeypatch)
    skill = _skill_path()
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# AssemblyAI")
    # MCP absent (so only the skill step can fail) and `npx skills remove` runs but
    # leaves the skill in place -> remove must report it as failed, not removed.
    monkeypatch.setattr(
        "aai_cli.commands.setup.subprocess.run",
        FakeRun({("claude", "mcp", "get"): 1}, removes_skill=False),
    )

    result = runner.invoke(app, ["setup", "remove"])
    assert result.exit_code == 1
    assert _statuses(result)["skill"] == "failed"


def test_remove_skill_skipped_when_npx_missing(monkeypatch):
    # The assemblyai skill is present but npx is gone -> we can't drive `skills
    # remove`, so report skipped (not failed).
    monkeypatch.setattr(
        "aai_cli.commands.setup.shutil.which",
        lambda tool: None if tool == "npx" else f"/usr/bin/{tool}",
    )
    skill = _skill_path()
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# AssemblyAI")
    monkeypatch.setattr(
        "aai_cli.commands.setup.subprocess.run",
        FakeRun({("claude", "mcp", "get"): 1}),
    )

    result = runner.invoke(app, ["setup", "remove"])
    assert result.exit_code == 0
    assert _statuses(result)["skill"] == "skipped"


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


# --- remove ------------------------------------------------------------------


def test_remove_unwinds_all(monkeypatch, tmp_path):
    _all_tools_present(monkeypatch)
    for name in ("assemblyai", "aai-cli"):
        d = tmp_path / ".claude" / "skills" / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("# x")
    fake = FakeRun({("claude", "mcp", "get"): 0})  # present -> removable
    monkeypatch.setattr("aai_cli.commands.setup.subprocess.run", fake)

    result = runner.invoke(app, ["setup", "remove"])
    assert result.exit_code == 0
    assert _statuses(result) == {"mcp": "removed", "skill": "removed", "aai-cli skill": "removed"}
    assert ["claude", "mcp", "remove", "assemblyai-docs"] in fake.calls
    assert ["npx", "-y", "skills", "remove", "assemblyai", "--global"] in fake.calls
    assert not _skill_path().exists()
    assert not _cli_skill_path().exists()


def test_remove_when_absent_is_not_an_error(monkeypatch):
    _all_tools_present(monkeypatch)  # no skill dirs
    fake = FakeRun({("claude", "mcp", "get"): 1})  # absent
    monkeypatch.setattr("aai_cli.commands.setup.subprocess.run", fake)

    result = runner.invoke(app, ["setup", "remove"])
    assert result.exit_code == 0
    assert _statuses(result) == {
        "mcp": "not_installed",
        "skill": "not_installed",
        "aai-cli skill": "not_installed",
    }
    assert not any(c[:3] == ["claude", "mcp", "remove"] for c in fake.calls)


def test_remove_scope_passthrough(monkeypatch):
    _all_tools_present(monkeypatch)
    fake = FakeRun({("claude", "mcp", "get"): 0})  # present
    monkeypatch.setattr("aai_cli.commands.setup.subprocess.run", fake)

    result = runner.invoke(app, ["setup", "remove", "--scope", "project"])
    assert result.exit_code == 0
    assert ["claude", "mcp", "remove", "assemblyai-docs", "--scope", "project"] in fake.calls


def test_remove_invalid_scope_exits_2(monkeypatch):
    _all_tools_present(monkeypatch)
    monkeypatch.setattr("aai_cli.commands.setup.subprocess.run", FakeRun())
    result = runner.invoke(app, ["setup", "remove", "--scope", "bogus"])
    assert result.exit_code == 2


def test_remove_skips_mcp_when_claude_missing(monkeypatch):
    monkeypatch.setattr(
        "aai_cli.commands.setup.shutil.which",
        lambda tool: None if tool == "claude" else f"/usr/bin/{tool}",
    )
    fake = FakeRun()
    monkeypatch.setattr("aai_cli.commands.setup.subprocess.run", fake)

    result = runner.invoke(app, ["setup", "remove"])
    assert result.exit_code == 0
    assert _statuses(result)["mcp"] == "skipped"
    assert not any(c[0] == "claude" for c in fake.calls)


def test_remove_mcp_failure_reports_failed(monkeypatch):
    _all_tools_present(monkeypatch)
    # present, but `mcp remove` fails -> the mcp step is failed and exit is non-zero.
    fake = FakeRun({("claude", "mcp", "get"): 0, ("claude", "mcp", "remove"): 1})
    monkeypatch.setattr("aai_cli.commands.setup.subprocess.run", fake)

    result = runner.invoke(app, ["setup", "remove"])
    assert result.exit_code == 1
    assert _statuses(result)["mcp"] == "failed"


def test_copy_tree_skips_pycache_and_pyc(tmp_path):
    # _copy_tree must not copy compiled-Python detritus into the agent's skills dir.
    from aai_cli.commands import setup

    src = tmp_path / "src"
    (src / "references").mkdir(parents=True)
    (src / "SKILL.md").write_text("# skill")
    (src / "references" / "a.md").write_text("a")
    (src / "stale.pyc").write_text("junk")
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "x.cpython-312.pyc").write_text("junk")

    dest = tmp_path / "dest"
    setup._copy_tree(src, dest)

    assert (dest / "SKILL.md").read_text() == "# skill"
    assert (dest / "references" / "a.md").read_text() == "a"
    assert not (dest / "stale.pyc").exists()
    assert not (dest / "__pycache__").exists()


# --- help --------------------------------------------------------------------


def test_setup_help_lists_all_subcommands():
    result = runner.invoke(app, ["setup", "--help"])
    assert result.exit_code == 0
    assert "install" in result.output
    assert "status" in result.output
    assert "remove" in result.output


def test_setup_no_subcommand_lists_commands():
    # Bare `aai setup` should show its commands instead of "Missing command".
    result = runner.invoke(app, ["setup"])
    assert "install" in result.output
    assert "status" in result.output
    assert "remove" in result.output


# --- aai-cli skill: defensive failure branches --------------------------------


def test_install_cli_skill_fails_when_bundle_missing(monkeypatch, tmp_path):
    from aai_cli.commands import setup

    monkeypatch.setattr(setup, "_bundled_cli_skill", lambda: tmp_path / "nonexistent")
    step = setup._install_cli_skill(force=False)
    assert step["status"] == "failed"
    assert "packaging bug" in step["detail"]


def test_install_cli_skill_fails_when_copy_lacks_skill_md(monkeypatch, tmp_path):
    from aai_cli.commands import setup

    empty = tmp_path / "emptybundle"
    empty.mkdir()
    monkeypatch.setattr(setup, "_bundled_cli_skill", lambda: empty)
    step = setup._install_cli_skill(force=False)
    assert step["status"] == "failed"
    assert "SKILL.md" in step["detail"]


def test_remove_cli_skill_fails_when_rmtree_noops(monkeypatch):
    from aai_cli.commands import setup

    dest = _cli_skill_path()
    dest.mkdir(parents=True)
    (dest / "SKILL.md").write_text("# x")
    monkeypatch.setattr(setup.shutil, "rmtree", lambda *a, **k: None)
    step = setup._remove_cli_skill()
    assert step["status"] == "failed"
    assert "still present" in step["detail"]
