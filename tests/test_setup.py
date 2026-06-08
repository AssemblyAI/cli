import subprocess

import pytest
from typer.testing import CliRunner

from aai_cli.main import app
from tests.setup_helpers import FakeRun, _all_tools_present, _cli_skill_path, _skill_path, _statuses

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


# --- remove: assemblyai skill ------------------------------------------------


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


# --- remove: all three steps -------------------------------------------------


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


# --- aai-cli skill helpers ----------------------------------------------------


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
