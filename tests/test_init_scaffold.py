import stat

import pytest

from aai_cli.errors import CLIError
from aai_cli.init import scaffold


def test_scaffold_env_is_owner_only_readable(tmp_path):
    # The .env holds the real API key, so it must not be world/group-readable. (CI is
    # POSIX; the project gates its Windows-specific paths in scripts/check.sh, not here.)
    target = tmp_path / "app"
    scaffold.scaffold("audio-transcription", target, api_key="sk-real-key")
    mode = stat.S_IMODE((target / ".env").stat().st_mode)
    assert mode == 0o600
    assert not mode & (stat.S_IRGRP | stat.S_IROTH)  # no group/other read of the key


def test_scaffold_tightens_existing_env_on_overwrite(tmp_path):
    # `aai init --force` re-scaffolds over an existing project; a stale, loosely
    # permissioned .env must be tightened to 0600 rather than left as-is.
    target = tmp_path / "app"
    target.mkdir()
    stale = target / ".env"
    stale.write_text("ASSEMBLYAI_API_KEY=old\n")
    stale.chmod(0o644)
    scaffold.scaffold("audio-transcription", target, api_key="sk-real-key")
    assert stat.S_IMODE(stale.stat().st_mode) == 0o600


def test_scaffold_copies_files_and_renames_dotfiles(tmp_path):
    target = tmp_path / "app"
    scaffold.scaffold("audio-transcription", target, api_key="sk-real-key")
    assert (target / "api" / "index.py").exists()
    assert (target / "static" / "index.html").exists()
    assert not (target / "vercel.json").exists()
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
    # `aai init <tmpl> a/b/app` targets a path whose parents don't exist yet; scaffold
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


def test_target_conflict_detects_nonempty_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert scaffold.target_conflict(empty) is False
    assert scaffold.target_conflict(tmp_path / "missing") is False
    nonempty = tmp_path / "full"
    nonempty.mkdir()
    (nonempty / "x.txt").write_text("hi")
    assert scaffold.target_conflict(nonempty) is True
