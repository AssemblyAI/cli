"""`assembly update` — install-channel dispatching self-update."""

import json
import types

from typer.testing import CliRunner

from aai_cli import __version__, config, update_check
from aai_cli.main import app

runner = CliRunner()


def _unwrapped(text: str) -> str:
    """Collapse console soft-wrapping (the dev version string is long)."""
    return " ".join(text.split())


def _pin_latest(monkeypatch, latest):
    """Make the explicit fetch land ``latest`` in the cache (no network)."""

    def fake_fetch():
        config.set_update_cache(last_check=1.0, latest_version=latest)

    monkeypatch.setattr(update_check, "fetch_and_cache", fake_fetch)


def _record_subprocess(monkeypatch, returncode=0):
    calls = []

    def fake_run(argv, check):
        calls.append((argv, check))
        return types.SimpleNamespace(returncode=returncode)

    monkeypatch.setattr("aai_cli.commands.update.subprocess.run", fake_run)
    return calls


def test_update_check_reports_available(monkeypatch):
    _pin_latest(monkeypatch, "999.0.0")
    result = runner.invoke(app, ["update", "--check"])
    assert result.exit_code == 0
    out = _unwrapped(result.output)
    assert "Update available" in out
    assert "999.0.0" in out
    assert "assembly update" in out  # points at the action


def test_update_check_json_shape(monkeypatch):
    _pin_latest(monkeypatch, "999.0.0")
    result = runner.invoke(app, ["update", "--check", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "current": __version__,
        "latest": "999.0.0",
        "update_available": True,
    }


def test_update_already_up_to_date_skips_the_upgrade(monkeypatch):
    _pin_latest(monkeypatch, __version__)
    calls = _record_subprocess(monkeypatch)
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0
    assert "Already up to date" in result.output
    assert __version__ in result.output
    assert calls == []  # no channel command ran


def test_update_fetch_failure_is_a_clean_api_error(monkeypatch):
    _pin_latest(monkeypatch, None)
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 1
    assert "Couldn't determine the latest version" in result.output


def test_update_unknown_install_channel_exits_2_with_docs(monkeypatch):
    _pin_latest(monkeypatch, "999.0.0")
    monkeypatch.setattr(update_check, "detect_upgrade_command", lambda: "")
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 2
    assert "Couldn't detect how this CLI was installed" in result.output
    assert update_check.DOCS_URL in result.output


def test_update_runs_the_channel_command(monkeypatch):
    _pin_latest(monkeypatch, "999.0.0")
    monkeypatch.setattr(update_check, "detect_upgrade_command", lambda: "brew upgrade assembly")
    calls = _record_subprocess(monkeypatch, returncode=0)
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0
    # shlex-split argv, check=False so the exit code is inspected, not raised.
    assert calls == [(["brew", "upgrade", "assembly"], False)]
    out = _unwrapped(result.output)
    assert "Updated" in out
    assert "999.0.0" in out
    assert "brew upgrade assembly" in out


def test_update_json_reports_versions_and_command(monkeypatch):
    _pin_latest(monkeypatch, "999.0.0")
    monkeypatch.setattr(update_check, "detect_upgrade_command", lambda: "pipx upgrade aai-cli")
    _record_subprocess(monkeypatch, returncode=0)
    result = runner.invoke(app, ["update", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "updated": True,
        "from": __version__,
        "to": "999.0.0",
        "command": "pipx upgrade aai-cli",
    }


def test_update_failed_upgrade_surfaces_exit_status(monkeypatch):
    _pin_latest(monkeypatch, "999.0.0")
    monkeypatch.setattr(update_check, "detect_upgrade_command", lambda: "brew upgrade assembly")
    _record_subprocess(monkeypatch, returncode=3)
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 1
    out = _unwrapped(result.output)
    assert "'brew upgrade assembly' exited with status 3" in out
    assert "Re-run it directly" in out
