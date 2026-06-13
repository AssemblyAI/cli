"""Tests for `assembly init` --force semantics, the key report row, and port hints.

Split out of test_init_command.py to keep modules under the 500-line gate.
"""

import subprocess

from typer.testing import CliRunner

from aai_cli.main import app

runner = CliRunner()
TEMPLATE = "audio-transcription"


def test_init_force_overwrites(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"]).exit_code == 0
    result = runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install", "--force"])
    assert result.exit_code == 0


def test_init_target_is_existing_file_usage_error(tmp_path, monkeypatch):
    # A target that exists but is a FILE is a clean usage error (exit 2), not the
    # "Unexpected error: [Errno 17] File exists" internal-bug path mkdir would hit.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "myapp").write_text("I am a file")
    result = runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"])
    assert result.exit_code == 2
    assert "exists and is not a directory" in result.output
    assert "Unexpected error" not in result.output
    assert (tmp_path / "myapp").read_text() == "I am a file"  # left untouched


def test_init_force_warns_existing_files_are_overwritten(tmp_path, monkeypatch):
    # --force overlays the template onto a non-empty target; the run must say so
    # (on stderr) instead of silently clobbering files.
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"]).exit_code == 0
    result = runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install", "--force"])
    assert result.exit_code == 0
    flat = " ".join(result.stderr.split())
    assert "overwriting existing files" in flat
    assert "files not in the template are kept" in flat
    assert "overwriting existing files" not in result.stdout


def test_init_force_no_overwrite_notice_for_fresh_target(tmp_path, monkeypatch):
    # --force against a missing/empty target overwrites nothing, so no notice.
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", TEMPLATE, "fresh", "--no-install", "--force"])
    assert result.exit_code == 0, result.output
    assert "overwriting existing files" not in result.output


def test_init_force_overwrite_notice_is_structured_in_json(tmp_path, monkeypatch):
    import json

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"]).exit_code == 0
    result = runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install", "--force", "--json"])
    assert result.exit_code == 0, result.output
    warning = json.loads(result.stderr.strip().splitlines()[0])
    assert "overwriting existing files" in warning["warning"]


def test_init_force_preserves_configured_env_key(tmp_path, monkeypatch):
    # A real key the user configured in .env must survive a keyless --force re-run
    # (previously it was silently reset to the placeholder).
    import json

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk-configured")
    assert runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"]).exit_code == 0
    monkeypatch.delenv("ASSEMBLYAI_API_KEY")
    result = runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install", "--force", "--json"])
    assert result.exit_code == 0, result.output
    assert "ASSEMBLYAI_API_KEY=sk-configured" in (tmp_path / "myapp" / ".env").read_text()
    payload = json.loads(result.stdout)
    key_row = next(s for s in payload if s["name"] == "key")
    assert key_row["status"] == "kept"
    assert "preserved" in key_row["detail"]


def test_init_force_over_placeholder_still_writes_placeholder(tmp_path, monkeypatch):
    # Nothing worth preserving: a placeholder .env re-scaffolds to a placeholder
    # with the usual skipped-key row.
    import json

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"]).exit_code == 0
    result = runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install", "--force", "--json"])
    assert result.exit_code == 0, result.output
    assert "your_assemblyai_api_key_here" in (tmp_path / "myapp" / ".env").read_text()
    payload = json.loads(result.stdout)
    key_row = next(s for s in payload if s["name"] == "key")
    assert key_row["status"] == "skipped"
    assert "no API key found" in key_row["detail"]


def test_init_force_with_preserved_key_still_launches(tmp_path, monkeypatch):
    # The preserved .env key counts as having a key: deps install and the server
    # launches, with no bogus "no API key" launch-skipped row.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk-configured")
    assert runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"]).exit_code == 0
    monkeypatch.delenv("ASSEMBLYAI_API_KEY")
    monkeypatch.setattr(
        "aai_cli.init.runner.run_setup",
        lambda *a, **k: subprocess.CompletedProcess([], 0, "", ""),
    )
    monkeypatch.setattr("aai_cli.init.runner.find_free_port", lambda preferred: 4321)
    launched = {"v": False}
    monkeypatch.setattr(
        "aai_cli.init.runner.launch_and_open",
        lambda *a, **k: launched.__setitem__("v", True) or 0,
    )
    result = runner.invoke(app, ["init", TEMPLATE, "myapp", "--force"])
    assert result.exit_code == 0, result.output
    assert launched["v"] is True
    assert "no API key" not in result.output


def test_init_reports_key_written_from_environment(tmp_path, monkeypatch):
    import json

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk-env")
    result = runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    key_row = next(s for s in payload if s["name"] == "key")
    assert key_row["status"] == "written"
    assert key_row["detail"] == "from environment"


def test_init_reports_key_written_from_keyring(tmp_path, monkeypatch):
    import json

    from aai_cli import config

    monkeypatch.chdir(tmp_path)
    config.set_api_key("default", "sk-stored")
    result = runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    key_row = next(s for s in payload if s["name"] == "key")
    assert key_row["status"] == "written"
    assert key_row["detail"] == "from keyring"


def test_init_blank_env_var_reports_keyring_source(tmp_path, monkeypatch):
    # A whitespace-only ASSEMBLYAI_API_KEY is "unset" to the key chain, so a key
    # that actually resolved from the keyring must not be attributed to the env.
    import json

    from aai_cli import config

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "   ")
    config.set_api_key("default", "sk-stored")
    result = runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    key_row = next(s for s in payload if s["name"] == "key")
    assert key_row["status"] == "written"
    assert key_row["detail"] == "from keyring"


def test_init_no_install_hint_carries_custom_port(tmp_path, monkeypatch):
    # `--no-install --port N` signs off with `assembly dev --port N`, not a bare
    # `assembly dev` that would boot the default port instead.
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install", "--port", "5005"])
    assert result.exit_code == 0, result.output
    packed = "".join(result.output.split())  # the hint line wraps on long tmp paths
    assert "assemblydev--port5005" in packed


def test_init_no_install_hint_default_port_needs_no_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", TEMPLATE, "myapp", "--no-install"])
    assert result.exit_code == 0, result.output
    packed = "".join(result.output.split())
    assert "assemblydev`" in packed
    assert "--port" not in result.output


def test_init_launch_skipped_detail_carries_custom_port(tmp_path, monkeypatch):
    # Logged out + install: the launch-skipped row's run command keeps the chosen port.
    import json

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "aai_cli.init.runner.run_setup",
        lambda *a, **k: subprocess.CompletedProcess([], 0, "", ""),
    )
    result = runner.invoke(app, ["init", TEMPLATE, "app", "--port", "5005", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    launch_row = next(s for s in payload if s["name"] == "launch")
    assert "assembly dev --port 5005" in launch_row["detail"]
