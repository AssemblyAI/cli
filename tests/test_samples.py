from pathlib import Path

from typer.testing import CliRunner

from aai_cli.main import app

runner = CliRunner()

_ENV_KEY = 'os.environ["ASSEMBLYAI_API_KEY"]'


def test_samples_list_shows_transcribe():
    result = runner.invoke(app, ["samples", "list"])
    assert result.exit_code == 0
    assert "transcribe" in result.output


def test_samples_list_shows_templates():
    result = runner.invoke(app, ["samples", "list"])
    assert result.exit_code == 0
    assert "transcribe" in result.output
    assert "stream" in result.output
    assert "agent" in result.output


def test_samples_create_agent_uses_env_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["samples", "create", "agent"])
    assert result.exit_code == 0
    body = Path(tmp_path, "agent", "agent.py").read_text()
    assert _ENV_KEY in body  # reads the key from the environment, no secret in the file
    assert "session.update" in body  # the voice-agent handshake
    assert "sounddevice" in body  # audio backend (PortAudio bundled in the wheel)
    assert "pyaudio" not in body


def test_samples_no_subcommand_lists_commands():
    # Bare `aai samples` should show its commands instead of erroring out.
    result = runner.invoke(app, ["samples"])
    assert "list" in result.output and "create" in result.output


def test_samples_create_stream_uses_env_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["samples", "create", "stream"])
    assert result.exit_code == 0
    body = Path(tmp_path, "stream", "stream.py").read_text()
    assert _ENV_KEY in body
    assert "MicrophoneStream" in body


def test_samples_create_transcribe_uses_env_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["samples", "create", "transcribe"])
    assert result.exit_code == 0
    body = Path(tmp_path, "transcribe", "transcribe.py").read_text()
    assert _ENV_KEY in body
    assert "import assemblyai as aai" in body


def test_samples_create_needs_no_auth(tmp_path, monkeypatch):
    # Scaffolding writes no secret, so it works without being logged in.
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["samples", "create", "transcribe"])
    assert result.exit_code == 0


def test_samples_create_unknown_name_errors():
    result = runner.invoke(app, ["samples", "create", "nope"])
    assert result.exit_code == 1


def test_samples_create_refuses_existing_without_force(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["samples", "create", "transcribe"]).exit_code == 0
    # Second run without --force must refuse.
    result = runner.invoke(app, ["samples", "create", "transcribe"])
    assert result.exit_code == 1


def test_samples_create_force_overwrites(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["samples", "create", "transcribe"]).exit_code == 0
    result = runner.invoke(app, ["samples", "create", "transcribe", "--force"])
    assert result.exit_code == 0


def test_samples_create_is_valid_python(tmp_path, monkeypatch):
    import ast

    monkeypatch.chdir(tmp_path)
    for name in ("transcribe", "stream", "agent"):
        assert runner.invoke(app, ["samples", "create", name]).exit_code == 0
        ast.parse(Path(tmp_path, name, f"{name}.py").read_text())  # generated code parses
