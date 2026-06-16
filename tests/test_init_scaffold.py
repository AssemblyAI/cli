import os
import stat

import pytest

from aai_cli.core.errors import CLIError
from aai_cli.init import scaffold


def test_scaffold_env_is_owner_only_readable(tmp_path):
    # The .env holds the real API key, so it must not be world/group-readable.
    target = tmp_path / "app"
    scaffold.scaffold("audio-transcription", target, api_key="sk-real-key")
    env_path = target / ".env"
    assert env_path.is_file()
    # POSIX permission bits are meaningful only on POSIX; Windows has no 0600 mode,
    # so the scaffolder's chmod is a best-effort no-op there.
    if os.name == "posix":
        mode = stat.S_IMODE(env_path.stat().st_mode)
        assert mode == 0o600
        assert not mode & (stat.S_IRGRP | stat.S_IROTH)  # no group/other read of the key


def test_scaffold_tightens_existing_env_on_overwrite(tmp_path):
    # `assembly init --force` re-scaffolds over an existing project; a stale, loosely
    # permissioned .env must be tightened to 0600 rather than left as-is.
    target = tmp_path / "app"
    target.mkdir()
    stale = target / ".env"
    stale.write_text("ASSEMBLYAI_API_KEY=old\n")
    if os.name == "posix":
        stale.chmod(0o644)
    scaffold.scaffold("audio-transcription", target, api_key="sk-real-key")
    # The rewrite lands on every platform; the 0600 tightening is POSIX-only.
    assert "ASSEMBLYAI_API_KEY=sk-real-key" in stale.read_text()
    if os.name == "posix":
        assert stat.S_IMODE(stale.stat().st_mode) == 0o600


def test_scaffold_omits_template_root_init_but_keeps_api_init(tmp_path):
    # The in-repo template dir is an importable package (templates/<name>/__init__.py),
    # but that root marker is repo-only and must NOT ship into the scaffolded project —
    # while api/'s own __init__.py must, since the shipped app's `from . import settings`
    # needs `api` to be a package.
    target = tmp_path / "app"
    scaffold.scaffold("agent-framework", target, api_key="sk-real-key")
    assert not (target / "__init__.py").exists()  # root marker skipped
    assert (target / "api" / "__init__.py").is_file()  # api package kept


def test_scaffold_copies_files_and_renames_dotfiles(tmp_path):
    target = tmp_path / "app"
    scaffold.scaffold("audio-transcription", target, api_key="sk-real-key")
    assert (target / "api" / "index.py").exists()
    assert (target / "static" / "index.html").exists()
    # vercel.json ships in the scaffold: it pins the FastAPI framework preset so the
    # `assembly deploy` -> `vercel deploy` path doesn't auto-detect the "services" framework.
    assert (target / "vercel.json").exists()
    # dotfile templates are renamed to their dotted names
    assert (target / ".gitignore").exists()
    assert (target / ".env.example").exists()
    # the plain-named source files are NOT copied verbatim
    assert not (target / "gitignore").exists()
    assert not (target / "env.example").exists()
    # the container build ships, and dockerignore is renamed to its dotted name
    assert (target / "Dockerfile").exists()
    assert (target / ".dockerignore").exists()
    assert not (target / "dockerignore").exists()
    # .dockerignore must exclude .env so the real key isn't baked into the image
    assert ".env" in (target / ".dockerignore").read_text().splitlines()


def test_scaffold_writes_env_with_key(tmp_path):
    target = tmp_path / "app"
    scaffold.scaffold("audio-transcription", target, api_key="sk-real-key")
    env = (target / ".env").read_text()
    assert "ASSEMBLYAI_API_KEY=sk-real-key" in env


def test_scaffold_writes_env_vars(tmp_path):
    target = tmp_path / "app"
    scaffold.scaffold(
        "audio-transcription",
        target,
        api_key="k",
        env_vars={
            "ASSEMBLYAI_BASE_URL": "https://api.sb.example",
            "ASSEMBLYAI_LLM_GATEWAY_URL": "https://llm.sb.example/v1",
        },
    )
    env = (target / ".env").read_text()
    assert "ASSEMBLYAI_BASE_URL=https://api.sb.example" in env
    assert "ASSEMBLYAI_LLM_GATEWAY_URL=https://llm.sb.example/v1" in env


def test_scaffold_omits_env_vars_when_none(tmp_path):
    target = tmp_path / "app"
    scaffold.scaffold("audio-transcription", target, api_key="k")
    env = (target / ".env").read_text()
    assert "ASSEMBLYAI_BASE_URL" not in env
    assert "ASSEMBLYAI_LLM_GATEWAY_URL" not in env


def test_scaffold_skips_pycache(tmp_path):
    # Importing a template's api/index.py during our own test run leaves a
    # __pycache__ next to it; the scaffolder must not copy that into the project.
    target = tmp_path / "app"
    scaffold.scaffold("audio-transcription", target, api_key=None)
    assert not list(target.rglob("__pycache__"))
    assert not list(target.rglob("*.pyc"))


def test_scaffold_writes_placeholder_when_no_key(tmp_path):
    target = tmp_path / "app"
    scaffold.scaffold("audio-transcription", target, api_key=None)
    env = (target / ".env").read_text()
    assert scaffold.PLACEHOLDER_KEY in env


def test_scaffold_unknown_template_raises(tmp_path):
    with pytest.raises(CLIError) as exc:
        scaffold.scaffold("nope", tmp_path / "app", api_key=None)
    assert exc.value.error_type == "unknown_template"
    assert exc.value.exit_code == 1


def test_scaffold_registered_but_missing_files_raises(tmp_path, monkeypatch):
    # Defense in depth: the registry lists a template whose on-disk dir is gone.
    monkeypatch.setattr("aai_cli.init.templates.is_template", lambda _t: True)
    with pytest.raises(CLIError) as exc:
        scaffold.scaffold("ghost-template", tmp_path / "app", api_key=None)
    assert exc.value.error_type == "template_missing"
    assert exc.value.exit_code == 1


def test_scaffold_creates_nested_target_parents(tmp_path):
    # `assembly init <tmpl> a/b/app` targets a path whose parents don't exist yet; scaffold
    # must create the whole chain (target.mkdir parents=True).
    target = tmp_path / "a" / "b" / "app"  # a/ and b/ do not exist
    scaffold.scaffold("audio-transcription", target, api_key="k")
    assert (target / "api" / "index.py").exists()


def test_scaffold_is_idempotent_over_existing_tree(tmp_path):
    # Re-scaffolding (e.g. `--force`) runs over an already-populated tree, so every
    # mkdir along the copy walk must tolerate existing dirs (exist_ok=True).
    target = tmp_path / "app"
    scaffold.scaffold("audio-transcription", target, api_key="k")
    scaffold.scaffold("audio-transcription", target, api_key="k2")  # dirs already exist
    assert (target / "api" / "index.py").exists()
    assert "ASSEMBLYAI_API_KEY=k2" in (target / ".env").read_text()


def test_existing_env_key_none_when_env_missing(tmp_path):
    assert scaffold.existing_env_key(tmp_path) is None


def test_existing_env_key_none_for_placeholder(tmp_path):
    (tmp_path / ".env").write_text(f"ASSEMBLYAI_API_KEY={scaffold.PLACEHOLDER_KEY}\n")
    assert scaffold.existing_env_key(tmp_path) is None


def test_existing_env_key_none_for_blank_value(tmp_path):
    (tmp_path / ".env").write_text("ASSEMBLYAI_API_KEY=\n")
    assert scaffold.existing_env_key(tmp_path) is None


def test_existing_env_key_none_when_key_line_absent(tmp_path):
    (tmp_path / ".env").write_text("ASSEMBLYAI_BASE_URL=https://api.example\n")
    assert scaffold.existing_env_key(tmp_path) is None


def test_existing_env_key_returns_configured_key(tmp_path):
    # Other lines before/after the key line are skipped; trailing whitespace is trimmed.
    (tmp_path / ".env").write_text(
        "OTHER=1\nASSEMBLYAI_API_KEY=sk-configured \nASSEMBLYAI_BASE_URL=https://api.example\n"
    )
    assert scaffold.existing_env_key(tmp_path) == "sk-configured"


def test_target_conflict_detects_nonempty_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert scaffold.target_conflict(empty) is False
    assert scaffold.target_conflict(tmp_path / "missing") is False
    nonempty = tmp_path / "full"
    nonempty.mkdir()
    (nonempty / "x.txt").write_text("hi")
    assert scaffold.target_conflict(nonempty) is True
