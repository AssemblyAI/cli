"""Unit tests for aai_cli.app.coding_agent — the setup/doctor shared presence probes."""

import subprocess

import pytest

from aai_cli.app import coding_agent


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Keep skill reads inside a temp HOME so tests never touch ~/.claude."""
    monkeypatch.setenv("HOME", str(tmp_path))
    # Path.home() reads USERPROFILE on Windows, not HOME, so isolate both.
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)


# --- run() ---------------------------------------------------------------------


def test_run_passes_safe_subprocess_defaults(monkeypatch):
    seen = {}

    def record(cmd, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(coding_agent.subprocess, "run", record)
    proc = coding_agent.run(["claude", "--version"])
    assert proc.returncode == 0
    # stdin detached so a prompting child fails fast instead of hanging; output
    # captured and decoded; returncode inspected rather than raised; 120s backstop.
    assert seen["stdin"] is subprocess.DEVNULL
    assert seen["capture_output"] is True
    assert seen["text"] is True
    assert seen["check"] is False
    assert seen["timeout"] == 120


def test_run_timeout_becomes_clean_failure(monkeypatch):
    def hang(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 0))

    monkeypatch.setattr(coding_agent.subprocess, "run", hang)
    proc = coding_agent.run(["claude", "mcp", "get"], timeout=5)
    assert proc.returncode == 124
    assert proc.stdout == ""
    assert proc.stderr == "timed out after 5s: claude mcp get"


# --- skill locations -------------------------------------------------------------


def test_skills_root_honors_claude_config_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "agent-config"))
    assert coding_agent.skills_root() == tmp_path / "agent-config" / "skills"


def test_skills_root_defaults_to_home_dot_claude(tmp_path):
    assert coding_agent.skills_root() == tmp_path / ".claude" / "skills"


def test_skill_presence_requires_skill_md(tmp_path):
    root = tmp_path / ".claude" / "skills"
    assert coding_agent.skill_dir() == root / "assemblyai"
    assert coding_agent.cli_skill_dir() == root / "aai-cli"
    assert not coding_agent.skill_installed()
    assert not coding_agent.cli_skill_installed()
    for name in ("assemblyai", "aai-cli"):
        d = root / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("# x")
    assert coding_agent.skill_installed()
    assert coding_agent.cli_skill_installed()


# --- mcp_present -----------------------------------------------------------------


def test_mcp_present_when_claude_mcp_get_succeeds(monkeypatch):
    calls = []

    def fake_run(cmd, *, timeout=120):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(coding_agent, "run", fake_run)
    assert coding_agent.mcp_present() is True
    assert calls == [["claude", "mcp", "get", "assemblyai-docs"]]


def test_mcp_absent_when_claude_mcp_get_fails(monkeypatch):
    monkeypatch.setattr(
        coding_agent,
        "run",
        lambda cmd, *, timeout=120: subprocess.CompletedProcess(
            args=cmd, returncode=1, stdout="", stderr=""
        ),
    )
    assert coding_agent.mcp_present() is False


# --- missing_components -----------------------------------------------------------


def _presence(monkeypatch, *, mcp, skill, cli_skill):
    monkeypatch.setattr(coding_agent, "mcp_present", lambda: mcp)
    monkeypatch.setattr(coding_agent, "skill_installed", lambda: skill)
    monkeypatch.setattr(coding_agent, "cli_skill_installed", lambda: cli_skill)


def test_missing_components_lists_every_absent_artifact(monkeypatch):
    _presence(monkeypatch, mcp=False, skill=False, cli_skill=False)
    assert coding_agent.missing_components() == [
        "docs MCP",
        "assemblyai skill",
        "aai-cli skill",
    ]


def test_missing_components_empty_when_fully_installed(monkeypatch):
    _presence(monkeypatch, mcp=True, skill=True, cli_skill=True)
    assert coding_agent.missing_components() == []


def test_missing_components_reports_only_the_absent_ones(monkeypatch):
    _presence(monkeypatch, mcp=True, skill=False, cli_skill=True)
    assert coding_agent.missing_components() == ["assemblyai skill"]
    _presence(monkeypatch, mcp=False, skill=True, cli_skill=False)
    assert coding_agent.missing_components() == ["docs MCP", "aai-cli skill"]
