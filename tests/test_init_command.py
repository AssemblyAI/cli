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


def test_init_writes_base_url_for_active_env(tmp_path, monkeypatch):
    # The generated .env pins the app to the CLI's active environment hosts.
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "transcribe", "myapp", "--no-install"])
    env = (tmp_path / "myapp" / ".env").read_text()
    assert "ASSEMBLYAI_BASE_URL=https://" in env
    assert "ASSEMBLYAI_LLM_GATEWAY_URL=https://" in env


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


def test_init_here_scaffolds_into_cwd(tmp_path, monkeypatch):
    work = tmp_path / "work"  # a fresh empty dir (tmp_path itself holds the test config/)
    work.mkdir()
    monkeypatch.chdir(work)
    result = runner.invoke(app, ["init", "transcribe", "--here", "--no-install"])
    assert result.exit_code == 0, result.output
    assert (work / "api" / "index.py").exists()  # scaffolded into cwd, no subdir


def test_init_here_with_directory_errors(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    result = runner.invoke(app, ["init", "transcribe", "mydir", "--here", "--no-install"])
    assert result.exit_code != 0  # ambiguous: directory AND --here both given


def test_init_json_output_is_machine_readable(tmp_path, monkeypatch):
    import json

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "transcribe", "myapp", "--no-install", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert any(step["name"] == "scaffold" and step["status"] == "created" for step in payload)


def test_init_unshipped_template_errors_cleanly(tmp_path, monkeypatch):
    # `stream` is in the design but its template doesn't ship yet, so it must not be
    # registered — and picking it must give a clean error, not a FileNotFoundError.
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "stream", "s", "--no-install"])
    assert result.exit_code == 1
    assert "stream" in result.output
