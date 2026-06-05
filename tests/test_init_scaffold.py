import pytest

from aai_cli.errors import CLIError
from aai_cli.init import scaffold


def test_scaffold_copies_files_and_renames_dotfiles(tmp_path):
    target = tmp_path / "app"
    scaffold.scaffold("audio-transcription", target, api_key="sk-real-key")
    assert (target / "api" / "index.py").exists()
    assert (target / "index.html").exists()
    assert (target / "vercel.json").exists()
    # dotfile templates are renamed to their dotted names
    assert (target / ".gitignore").exists()
    assert (target / ".env.example").exists()
    # the plain-named source files are NOT copied verbatim
    assert not (target / "gitignore").exists()
    assert not (target / "env.example").exists()


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
    with pytest.raises(CLIError):
        scaffold.scaffold("nope", tmp_path / "app", api_key=None)


def test_target_conflict_detects_nonempty_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert scaffold.target_conflict(empty) is False
    assert scaffold.target_conflict(tmp_path / "missing") is False
    nonempty = tmp_path / "full"
    nonempty.mkdir()
    (nonempty / "x.txt").write_text("hi")
    assert scaffold.target_conflict(nonempty) is True
