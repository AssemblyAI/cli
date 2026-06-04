from typer.testing import CliRunner

from aai_cli.main import app

runner = CliRunner()


def test_init_scaffold_only_creates_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "transcribe", "myapp", "--no-install"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "myapp" / "api" / "index.py").exists()
    assert (tmp_path / "myapp" / ".env").exists()


def test_init_writes_key_from_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk-from-env")
    runner.invoke(app, ["init", "transcribe", "myapp", "--no-install"])
    assert "ASSEMBLYAI_API_KEY=sk-from-env" in (tmp_path / "myapp" / ".env").read_text()


def test_init_placeholder_key_when_logged_out(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "transcribe", "myapp", "--no-install"])
    env = (tmp_path / "myapp" / ".env").read_text()
    assert "your_assemblyai_api_key_here" in env


def test_init_unknown_template_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "nope", "myapp", "--no-install"])
    assert result.exit_code == 1


def test_init_refuses_nonempty_dir_without_force(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "transcribe", "myapp", "--no-install"]).exit_code == 0
    result = runner.invoke(app, ["init", "transcribe", "myapp", "--no-install"])
    assert result.exit_code == 1


def test_init_force_overwrites(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "transcribe", "myapp", "--no-install"]).exit_code == 0
    result = runner.invoke(app, ["init", "transcribe", "myapp", "--no-install", "--force"])
    assert result.exit_code == 0


def test_init_no_template_non_interactive_errors(tmp_path, monkeypatch):
    # CliRunner has no TTY, so the picker can't run; bare `aai init` must error helpfully.
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init"])
    assert result.exit_code != 0
    assert "transcribe" in result.output  # lists the available templates


def test_init_default_dir_is_template_app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "transcribe", "--no-install"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "transcribe-app" / "api" / "index.py").exists()


def test_init_appears_in_help():
    result = runner.invoke(app, ["--help"])
    assert "init" in result.output
