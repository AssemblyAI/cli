"""Tests for the live agent's project-context loader (aai_cli.agent_cascade.project_context).

`assembly live` reads the launch directory's AGENTS.md/CLAUDE.md into its system prompt so a
spoken answer is grounded in the project it's run from — the same convention coding agents follow.
"""

from __future__ import annotations

import types

from aai_cli.agent_cascade import project_context
from aai_cli.app.context import AppState
from aai_cli.commands.agent_cascade import _exec
from aai_cli.commands.agent_cascade._exec import run_agent_cascade
from aai_cli.core import config
from tests.test_agent_cascade_command import _opts


def test_returns_none_when_no_instruction_files(tmp_path):
    # An empty directory has nothing to inject, so the prompt stays the plain persona.
    assert project_context.load_project_context(tmp_path) is None


def test_reads_agents_md_under_a_heading(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Use uv run for everything.", encoding="utf-8")
    loaded = project_context.load_project_context(tmp_path)
    # The content is included verbatim under a per-file heading naming its source.
    assert loaded == "# AGENTS.md\n\nUse uv run for everything."


def test_reads_claude_md_when_agents_md_absent(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("Project rules here.", encoding="utf-8")
    loaded = project_context.load_project_context(tmp_path)
    assert loaded == "# CLAUDE.md\n\nProject rules here."


def test_includes_both_files_in_precedence_order_when_they_differ(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Agents rules.", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("Claude rules.", encoding="utf-8")
    loaded = project_context.load_project_context(tmp_path)
    # Both distinct files are present, AGENTS.md first (its precedence), then CLAUDE.md.
    assert loaded == "# AGENTS.md\n\nAgents rules.\n\n# CLAUDE.md\n\nClaude rules."


def test_identical_content_is_included_once(tmp_path):
    # CLAUDE.md is commonly a symlink to AGENTS.md (as in this repo); identical content must not
    # be duplicated into the prompt. We assert the dedup on content, so it covers the symlink case
    # without depending on symlink support being available on the test platform.
    (tmp_path / "AGENTS.md").write_text("Same guidance.", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("Same guidance.", encoding="utf-8")
    loaded = project_context.load_project_context(tmp_path)
    assert loaded == "# AGENTS.md\n\nSame guidance."
    assert loaded.count("Same guidance.") == 1


def test_whitespace_only_file_is_skipped(tmp_path):
    # A blank instruction file carries no guidance, so it's treated as absent (None, not an
    # empty heading) — the stripped-empty branch.
    (tmp_path / "AGENTS.md").write_text("   \n\t\n", encoding="utf-8")
    assert project_context.load_project_context(tmp_path) is None


def test_oversized_content_is_truncated_to_the_budget(tmp_path):
    body = "x" * (project_context.MAX_CONTEXT_CHARS + 5000)
    (tmp_path / "AGENTS.md").write_text(body, encoding="utf-8")
    loaded = project_context.load_project_context(tmp_path)
    assert loaded is not None
    # Capped at the budget plus the truncation marker, so a huge file can't crowd out the chat.
    assert loaded.endswith("[project context truncated]")
    assert len(loaded) == project_context.MAX_CONTEXT_CHARS + len("\n\n[project context truncated]")
    assert len(loaded) < len(body)


def test_content_at_the_budget_is_left_whole(tmp_path):
    # A file exactly at the cap is included untruncated (the boundary is inclusive).
    # Account for the "# AGENTS.md\n\n" heading so the combined string lands exactly at the cap.
    heading = "# AGENTS.md\n\n"
    body = "y" * (project_context.MAX_CONTEXT_CHARS - len(heading))
    (tmp_path / "AGENTS.md").write_text(body, encoding="utf-8")
    loaded = project_context.load_project_context(tmp_path)
    assert loaded is not None
    assert "truncated" not in loaded
    assert len(loaded) == project_context.MAX_CONTEXT_CHARS


def test_defaults_to_the_current_working_directory(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("cwd guidance", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    # No directory argument -> reads cwd, so the live command picks up the project it's launched in.
    assert project_context.load_project_context() == "# AGENTS.md\n\ncwd guidance"


def test_missing_directory_reads_as_no_context(tmp_path):
    # A nonexistent base directory raises OSError per candidate, which is swallowed -> None.
    assert project_context.load_project_context(tmp_path / "does-not-exist") is None


def test_context_filenames_order():
    # AGENTS.md (the cross-agent standard) takes precedence over CLAUDE.md.
    assert project_context.CONTEXT_FILENAMES == ("AGENTS.md", "CLAUDE.md")


# --- command wiring: run_agent_cascade reads the loader into the config ------


def test_run_reads_project_context_into_config(monkeypatch):
    monkeypatch.setattr(_exec.tts_session, "require_available", lambda _c: None)
    monkeypatch.setattr(config, "resolve_api_key", lambda **_: "k")
    monkeypatch.setattr(_exec, "FileSource", lambda src: types.SimpleNamespace(sample_rate=16000))
    monkeypatch.setattr(_exec.client, "resolve_audio_source", lambda source, sample: "clip.wav")
    # Stub the loader so the assertion doesn't depend on the repo's own (large) instruction file.
    monkeypatch.setattr(_exec, "load_project_context", lambda: "# AGENTS.md\n\nProject background.")
    captured = {}

    def fake_real(api_key, config, *, audio, stt_params, approver=None):
        captured["config"] = config
        return "deps"

    monkeypatch.setattr(_exec.engine.CascadeDeps, "real", fake_real)
    monkeypatch.setattr(_exec.engine, "run_cascade", lambda **kwargs: None)
    run_agent_cascade(_opts(source="clip.wav"), AppState(), json_mode=False)
    # The launch directory's AGENTS.md/CLAUDE.md rides into the cascade config.
    assert captured["config"].project_context == "# AGENTS.md\n\nProject background."
