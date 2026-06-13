import json
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


@pytest.fixture(autouse=True)
def _force_json(monkeypatch):
    """These tests pin the structured step/status JSON. The CLI now defaults to human
    text everywhere (JSON is opt-in), so force the machine output the assertions parse —
    the equivalent of invoking each command with --json."""
    monkeypatch.setattr("aai_cli.output.resolve_json", lambda *, explicit: True)


def test_proc_detail_prefers_stderr_then_falls_back_to_stdout():
    from aai_cli import setup_exec as setup

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
        "aai_cli.setup_exec.subprocess.run",
        FakeRun({("claude", "mcp", "get"): 1}, removes_skill=False),
    )

    result = runner.invoke(app, ["setup", "remove"])
    assert result.exit_code == 1
    assert _statuses(result)["skill"] == "failed"
    # The failure detail surfaces the subprocess's stderr ("boom"), preferring it over
    # the generic "still present" fallback (pins `_proc_detail(proc) or ...`).
    skill_detail = next(
        s["detail"] for s in json.loads(result.output)["steps"] if s["name"] == "skill"
    )
    assert "boom" in skill_detail


def test_remove_skill_skipped_when_npx_missing(monkeypatch):
    # The assemblyai skill is present but npx is gone -> we can't drive `skills
    # remove`, so report skipped (not failed).
    monkeypatch.setattr(
        "aai_cli.setup_exec.shutil.which",
        lambda tool: None if tool == "npx" else f"/usr/bin/{tool}",
    )
    skill = _skill_path()
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# AssemblyAI")
    monkeypatch.setattr(
        "aai_cli.setup_exec.subprocess.run",
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
    monkeypatch.setattr("aai_cli.setup_exec.subprocess.run", fake)

    result = runner.invoke(app, ["setup", "remove"])
    assert result.exit_code == 0
    assert _statuses(result) == {"mcp": "removed", "skill": "removed", "aai-cli skill": "removed"}
    assert ["claude", "mcp", "remove", "assemblyai-docs"] in fake.calls
    assert ["npx", "-y", "skills", "remove", "assemblyai", "--global"] in fake.calls
    assert not _skill_path().exists()
    assert not _cli_skill_path().exists()
    # The skill-remove subprocess uses the explicit 120s timeout backstop.
    remove_calls = [kw for cmd, kw in fake.invocations if cmd[:1] == ["npx"] and "remove" in cmd]
    assert remove_calls and remove_calls[0]["timeout"] == 120


def test_remove_when_absent_is_not_an_error(monkeypatch):
    _all_tools_present(monkeypatch)  # no skill dirs
    fake = FakeRun({("claude", "mcp", "get"): 1})  # absent
    monkeypatch.setattr("aai_cli.setup_exec.subprocess.run", fake)

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
    monkeypatch.setattr("aai_cli.setup_exec.subprocess.run", fake)

    result = runner.invoke(app, ["setup", "remove", "--scope", "project"])
    assert result.exit_code == 0
    assert ["claude", "mcp", "remove", "assemblyai-docs", "--scope", "project"] in fake.calls


def test_remove_invalid_scope_exits_2(monkeypatch):
    _all_tools_present(monkeypatch)
    monkeypatch.setattr("aai_cli.setup_exec.subprocess.run", FakeRun())
    result = runner.invoke(app, ["setup", "remove", "--scope", "bogus"])
    assert result.exit_code == 2


def test_remove_skips_mcp_when_claude_missing(monkeypatch):
    monkeypatch.setattr(
        "aai_cli.setup_exec.shutil.which",
        lambda tool: None if tool == "claude" else f"/usr/bin/{tool}",
    )
    fake = FakeRun()
    monkeypatch.setattr("aai_cli.setup_exec.subprocess.run", fake)

    result = runner.invoke(app, ["setup", "remove"])
    assert result.exit_code == 0
    assert _statuses(result)["mcp"] == "skipped"
    assert not any(c[0] == "claude" for c in fake.calls)


def test_remove_mcp_failure_reports_failed(monkeypatch):
    _all_tools_present(monkeypatch)
    # present, but `mcp remove` fails -> the mcp step is failed and exit is non-zero.
    fake = FakeRun({("claude", "mcp", "get"): 0, ("claude", "mcp", "remove"): 1})
    monkeypatch.setattr("aai_cli.setup_exec.subprocess.run", fake)

    result = runner.invoke(app, ["setup", "remove"])
    assert result.exit_code == 1
    assert _statuses(result)["mcp"] == "failed"


# --- aai-cli skill helpers ----------------------------------------------------


def test_copy_tree_skips_pycache_and_pyc(tmp_path):
    # _copy_tree must not copy compiled-Python detritus into the agent's skills dir.
    from aai_cli import setup_exec as setup

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


def test_copy_tree_creates_missing_parent_dirs(tmp_path):
    # The destination's parents may not exist yet (~/.claude/skills on a fresh
    # machine); _copy_tree must create the whole chain (mkdir parents=True).
    from aai_cli import setup_exec as setup

    src = tmp_path / "src"
    src.mkdir()
    (src / "SKILL.md").write_text("# skill")

    dest = tmp_path / "a" / "b" / "c" / "dest"  # none of a/b/c exist yet
    setup._copy_tree(src, dest)
    assert (dest / "SKILL.md").read_text() == "# skill"


def test_copy_tree_into_existing_dir_is_tolerated(tmp_path):
    # _copy_tree may run with the destination already present (a forced reinstall over
    # an existing skill dir); the mkdir must tolerate it (exist_ok=True), not raise.
    from aai_cli import setup_exec as setup

    src = tmp_path / "src"
    src.mkdir()
    (src / "SKILL.md").write_text("# skill")

    dest = tmp_path / "dest"
    dest.mkdir()  # already exists before the copy
    setup._copy_tree(src, dest)
    assert (dest / "SKILL.md").read_text() == "# skill"


# --- help --------------------------------------------------------------------


def test_setup_help_lists_all_subcommands():
    result = runner.invoke(app, ["setup", "--help"])
    assert result.exit_code == 0
    assert "install" in result.output
    assert "status" in result.output
    assert "remove" in result.output


def test_setup_help_install_summary_is_a_complete_sentence():
    # The panel shows install's docstring first line; it used to be cut mid-sentence
    # at a colon ("…by installing three things:"). Pin a standalone summary.
    import re

    result = runner.invoke(app, ["setup", "--help"])
    # Strip ANSI (CI forces color) and unwrap lines before matching.
    flat = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", result.output).split())
    assert "Set up your coding agent for AssemblyAI (docs MCP server + skills)." in flat
    assert "three things:" not in flat


def test_setup_no_subcommand_lists_commands():
    # Bare `assembly setup` should show its commands instead of "Missing command".
    result = runner.invoke(app, ["setup"])
    assert "install" in result.output
    assert "status" in result.output
    assert "remove" in result.output


# --- aai-cli skill: defensive failure branches --------------------------------


def test_install_cli_skill_fails_when_bundle_missing(monkeypatch, tmp_path):
    from aai_cli import setup_exec as setup

    monkeypatch.setattr(setup, "_bundled_cli_skill", lambda: tmp_path / "nonexistent")
    step = setup.install_cli_skill(force=False)
    assert step["status"] == "failed"
    assert "packaging bug" in step["detail"]


def test_install_cli_skill_fails_when_copy_lacks_skill_md(monkeypatch, tmp_path):
    from aai_cli import setup_exec as setup

    empty = tmp_path / "emptybundle"
    empty.mkdir()
    monkeypatch.setattr(setup, "_bundled_cli_skill", lambda: empty)
    step = setup.install_cli_skill(force=False)
    assert step["status"] == "failed"
    assert "SKILL.md" in step["detail"]


def test_remove_cli_skill_fails_when_rmtree_noops(monkeypatch):
    from aai_cli import setup_exec as setup

    dest = _cli_skill_path()
    dest.mkdir(parents=True)
    (dest / "SKILL.md").write_text("# x")
    monkeypatch.setattr(setup.shutil, "rmtree", lambda *a, **k: None)
    step = setup.remove_cli_skill()
    assert step["status"] == "failed"
    assert "still present" in step["detail"]


def test_remove_cli_skill_tolerates_rmtree_error(monkeypatch):
    # Removal is best-effort (ignore_errors=True): a deletion failure must surface as a
    # clean "failed" step (skill still present), never an uncaught OSError. Without
    # ignore_errors, rmtree would raise instead of returning.
    from aai_cli import setup_exec as setup

    dest = _cli_skill_path()
    dest.mkdir(parents=True)
    (dest / "SKILL.md").write_text("# x")

    def rmtree(path, ignore_errors=False, **kwargs):
        if not ignore_errors:
            raise OSError("permission denied")  # what a non-ignoring rmtree would do

    monkeypatch.setattr(setup.shutil, "rmtree", rmtree)
    step = setup.remove_cli_skill()
    assert step["status"] == "failed"
    assert "still present" in step["detail"]
