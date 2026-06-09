import subprocess

from typer.testing import CliRunner

from aai_cli.main import app

runner = CliRunner()
WEB = "web: uvicorn api.index:app --host 0.0.0.0 --port ${PORT:-3000}\n"


def _make_project(tmp_path):
    (tmp_path / "Procfile").write_text(WEB)


def _stub_runner(monkeypatch, *, use_uv=True, setup_rc=0):
    monkeypatch.setattr("aai_cli.init.runner.has_uv", lambda: use_uv)
    monkeypatch.setattr("aai_cli.init.runner.find_free_port", lambda port, **k: port)
    monkeypatch.setattr(
        "aai_cli.init.runner.run_setup",
        lambda *a, **k: subprocess.CompletedProcess([], setup_rc, "", "boom"),
    )
    captured: dict = {}

    def fake_run_server(target, *, command, port, env, open_browser):
        captured.update(
            target=target, command=command, port=port, env=env, open_browser=open_browser
        )
        return 0

    monkeypatch.setattr("aai_cli.init.runner.run_server", fake_run_server)
    return captured


def test_dev_boots_procfile_command_with_reload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    captured = _stub_runner(monkeypatch)
    result = runner.invoke(app, ["dev", "--no-open"])
    assert result.exit_code == 0, result.output
    cmd = captured["command"]
    assert cmd[:4] == ["uv", "run", "uvicorn", "api.index:app"]
    assert "--host" in cmd
    assert cmd[-3:] == ["--port", "3000", "--reload"]
    assert captured["env"]["PORT"] == "3000"
    assert captured["open_browser"] is False
    assert "Starting" in result.output
    assert "localhost:3000" in result.output


def test_dev_opens_browser_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    captured = _stub_runner(monkeypatch)
    result = runner.invoke(app, ["dev"])
    assert result.exit_code == 0, result.output
    assert captured["open_browser"] is True


def test_dev_custom_port_expands_and_flows_through(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    captured = _stub_runner(monkeypatch)
    result = runner.invoke(app, ["dev", "--port", "8123", "--no-open"])
    assert result.exit_code == 0, result.output
    assert captured["port"] == 8123
    assert captured["env"]["PORT"] == "8123"
    assert "8123" in captured["command"]
    assert "3000" not in captured["command"]  # default was overridden, not used


def test_dev_venv_command_when_no_uv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    captured = _stub_runner(monkeypatch, use_uv=False)
    result = runner.invoke(app, ["dev", "--no-open", "--json"])
    assert result.exit_code == 0, result.output
    assert captured["command"][1] == "-m"
    assert captured["command"][0].endswith("python") or ".venv" in captured["command"][0]
    assert captured["command"][2] == "uvicorn"
    assert captured["command"][-1] == "--reload"
    assert '"detail": "venv + pip"' in result.output


def test_dev_no_install_skips_setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    captured = _stub_runner(monkeypatch)
    called = {"setup": False}
    monkeypatch.setattr(
        "aai_cli.init.runner.run_setup",
        lambda *a, **k: (
            called.__setitem__("setup", True) or subprocess.CompletedProcess([], 0, "", "")
        ),
    )
    result = runner.invoke(app, ["dev", "--no-install", "--no-open"])
    assert result.exit_code == 0, result.output
    assert called["setup"] is False
    assert captured["command"][-1] == "--reload"


def test_dev_missing_procfile_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no Procfile
    captured = _stub_runner(monkeypatch)
    result = runner.invoke(app, ["dev"])
    assert result.exit_code == 1
    assert "aai init" in result.output
    assert captured == {}  # never launched


def test_dev_install_failure_exits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    captured = _stub_runner(monkeypatch, setup_rc=1)
    result = runner.invoke(app, ["dev"])
    assert result.exit_code == 1
    assert "boom" in result.output
    assert captured == {}  # install failed -> no launch


def test_dev_install_failure_detail_truncated_to_300(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    _stub_runner(monkeypatch)
    long_stderr = "x" * 500
    monkeypatch.setattr(
        "aai_cli.init.runner.run_setup",
        lambda *a, **k: subprocess.CompletedProcess([], 1, "", long_stderr),
    )
    result = runner.invoke(app, ["dev", "--json"])
    assert result.exit_code == 1
    assert '"detail": "' + "x" * 300 + '"' in result.output
    assert "x" * 301 not in result.output


def test_dev_server_nonzero_exit_propagates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    _stub_runner(monkeypatch)
    monkeypatch.setattr("aai_cli.init.runner.run_server", lambda *a, **k: 3)
    result = runner.invoke(app, ["dev", "--no-open"])
    assert result.exit_code == 3


def test_dev_json_emits_install_step(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    _stub_runner(monkeypatch)
    result = runner.invoke(app, ["dev", "--no-open", "--json"])
    assert result.exit_code == 0, result.output
    assert '"name": "install"' in result.output
    assert '"detail": "uv"' in result.output
