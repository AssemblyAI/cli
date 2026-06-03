from pathlib import Path

from typer.testing import CliRunner

from assemblyai_cli import config
from assemblyai_cli.main import app

runner = CliRunner()


def test_samples_list_shows_transcribe():
    result = runner.invoke(app, ["samples", "list"])
    assert result.exit_code == 0
    assert "transcribe" in result.output


def test_samples_list_shows_templates():
    result = runner.invoke(app, ["samples", "list"])
    assert result.exit_code == 0
    assert "transcribe" in result.output
    assert "stream" in result.output


def test_samples_create_stream_writes_script_with_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config.set_api_key("default", "sk_injected")
    result = runner.invoke(app, ["samples", "create", "stream"])
    assert result.exit_code == 0
    script = Path(tmp_path, "stream", "stream.py")
    assert script.exists()
    body = script.read_text()
    assert "sk_injected" in body
    assert "{{API_KEY}}" not in body
    assert "MicrophoneStream" in body


def test_samples_create_writes_script_with_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config.set_api_key("default", "sk_injected")
    result = runner.invoke(app, ["samples", "create", "transcribe"])
    assert result.exit_code == 0
    script = Path(tmp_path, "transcribe", "transcribe.py")
    assert script.exists()
    body = script.read_text()
    assert "sk_injected" in body
    assert "{{API_KEY}}" not in body


def test_samples_create_unknown_name_errors():
    result = runner.invoke(app, ["samples", "create", "nope"])
    assert result.exit_code == 1


def test_samples_create_unauthenticated_exits_2(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["samples", "create", "transcribe"])
    assert result.exit_code == 2


def test_samples_create_refuses_existing_without_force(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config.set_api_key("default", "sk_injected")
    assert runner.invoke(app, ["samples", "create", "transcribe"]).exit_code == 0
    # Second run without --force must refuse.
    result = runner.invoke(app, ["samples", "create", "transcribe"])
    assert result.exit_code == 1


def test_samples_create_force_overwrites(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config.set_api_key("default", "sk_injected")
    assert runner.invoke(app, ["samples", "create", "transcribe"]).exit_code == 0
    result = runner.invoke(app, ["samples", "create", "transcribe", "--force"])
    assert result.exit_code == 0


def test_samples_create_file_is_owner_only(tmp_path, monkeypatch):
    import stat

    monkeypatch.chdir(tmp_path)
    config.set_api_key("default", "sk_injected")
    assert runner.invoke(app, ["samples", "create", "transcribe"]).exit_code == 0
    from pathlib import Path

    mode = stat.S_IMODE(Path(tmp_path, "transcribe", "transcribe.py").stat().st_mode)
    assert mode == 0o600
