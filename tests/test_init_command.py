from typer.testing import CliRunner

from aai_cli.main import app

runner = CliRunner()
TEMPLATE = "audio-transcription"


def test_init_scaffold_only_creates_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "myapp" / "api" / "index.py").exists()
    assert (tmp_path / "myapp" / ".env").exists()


def test_init_writes_key_from_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk-from-env")
    runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"])
    assert "ASSEMBLYAI_API_KEY=sk-from-env" in (tmp_path / "myapp" / ".env").read_text()


def test_init_logged_out_installs_but_skips_launch_with_hint(tmp_path, monkeypatch):
    # Logged out + install: deps install, but the server can't start without a key —
    # the report must say so (and must not launch) rather than exiting silently.
    import subprocess

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "aai_cli.init.runner.run_setup",
        lambda *a, **k: subprocess.CompletedProcess([], 0, "", ""),
    )
    launched = {"v": False}
    monkeypatch.setattr(
        "aai_cli.init.runner.launch_and_open",
        lambda *a, **k: launched.__setitem__("v", True) or 0,
    )
    result = runner.invoke(app, ["init", TEMPLATE, "app"])  # no --no-install, logged out
    assert result.exit_code == 0, result.output
    assert launched["v"] is False
    assert "aai login" in result.output


def test_init_writes_base_url_for_active_env(tmp_path, monkeypatch):
    # The generated .env pins the app to the CLI's active environment hosts.
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"])
    env = (tmp_path / "myapp" / ".env").read_text()
    assert "ASSEMBLYAI_BASE_URL=https://" in env
    assert "ASSEMBLYAI_LLM_GATEWAY_URL=https://" in env


def test_init_placeholder_key_when_logged_out(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"])
    env = (tmp_path / "myapp" / ".env").read_text()
    assert "your_assemblyai_api_key_here" in env


def test_init_unknown_template_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "nope", "myapp", "--no-install"])
    assert result.exit_code == 1


def test_init_refuses_nonempty_dir_without_force(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"]).exit_code == 0
    result = runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"])
    assert result.exit_code == 1


def test_init_force_overwrites(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"]).exit_code == 0
    result = runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install", "--force"])
    assert result.exit_code == 0


def test_init_no_template_non_interactive_errors(tmp_path, monkeypatch):
    # CliRunner has no TTY, so the picker can't run; bare `aai init` must error helpfully.
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init"])
    assert result.exit_code != 0
    assert TEMPLATE in result.output  # lists the available templates


def test_init_default_dir_is_template_slug(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", TEMPLATE, "--no-install"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / TEMPLATE / "api" / "index.py").exists()


def test_init_appears_in_help():
    result = runner.invoke(app, ["--help"])
    assert "init" in result.output


def test_init_prints_cli_banner_in_human_mode(tmp_path, monkeypatch):
    # Vercel-style header at the top of an interactive run (human output only).
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("aai_cli.output._is_agentic", lambda: False)
    result = runner.invoke(app, ["init", TEMPLATE, "x", "--no-install"])
    assert result.exit_code == 0, result.output
    assert "AssemblyAI CLI" in result.output


def test_init_here_scaffolds_into_cwd(tmp_path, monkeypatch):
    work = tmp_path / "work"  # a fresh empty dir (tmp_path itself holds the test config/)
    work.mkdir()
    monkeypatch.chdir(work)
    result = runner.invoke(app, ["init", TEMPLATE, "--here", "--no-install"])
    assert result.exit_code == 0, result.output
    assert (work / "api" / "index.py").exists()  # scaffolded into cwd, no subdir


def test_init_here_with_directory_errors(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    result = runner.invoke(app, ["init", TEMPLATE, "mydir", "--here", "--no-install"])
    assert result.exit_code != 0  # ambiguous: directory AND --here both given


def test_init_json_output_is_machine_readable(tmp_path, monkeypatch):
    import json

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert any(step["name"] == "scaffold" and step["status"] == "created" for step in payload)


def test_init_unregistered_template_errors_cleanly(tmp_path, monkeypatch):
    # `llm` isn't a separate template (it's folded into audio-transcription), so
    # picking it must give a clean error, not a FileNotFoundError.
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "llm", "x", "--no-install"])
    assert result.exit_code == 1
    assert "llm" in result.output
