import subprocess

from typer.testing import CliRunner

from aai_cli.main import app

runner = CliRunner()


def _make_app(tmp_path):
    """Scaffold the minimal marker `aai dev` looks for: api/index.py."""
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "index.py").write_text("app = object()\n")


def _stub_runner(monkeypatch, *, use_uv=True, setup_rc=0):
    """Stub the runner boundary; return a dict capturing launch_and_open kwargs."""
    monkeypatch.setattr("aai_cli.init.runner.has_uv", lambda: use_uv)
    monkeypatch.setattr("aai_cli.init.runner.find_free_port", lambda port, **k: port)
    monkeypatch.setattr(
        "aai_cli.init.runner.run_setup",
        lambda *a, **k: subprocess.CompletedProcess([], setup_rc, "", "boom"),
    )
    captured: dict = {}

    def fake_launch(target, *, port, use_uv, open_browser, reload):
        captured.update(
            target=target, port=port, use_uv=use_uv, open_browser=open_browser, reload=reload
        )
        return 0

    monkeypatch.setattr("aai_cli.init.runner.launch_and_open", fake_launch)
    return captured


def test_dev_launches_with_reload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_app(tmp_path)
    captured = _stub_runner(monkeypatch)
    result = runner.invoke(app, ["dev", "--no-open"])
    assert result.exit_code == 0, result.output
    assert captured["reload"] is True
    assert captured["open_browser"] is False
    assert captured["port"] == 3000
    assert "Starting" in result.output
    assert "localhost:3000" in result.output


def test_dev_opens_browser_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_app(tmp_path)
    captured = _stub_runner(monkeypatch)
    result = runner.invoke(app, ["dev"])
    assert result.exit_code == 0, result.output
    assert captured["open_browser"] is True


def test_dev_no_install_skips_setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_app(tmp_path)
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
    assert captured["reload"] is True


def test_dev_missing_app_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no api/index.py here
    captured = _stub_runner(monkeypatch)
    result = runner.invoke(app, ["dev"])
    assert result.exit_code == 1
    assert "aai init" in result.output
    assert captured == {}  # never launched


def test_dev_install_failure_exits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_app(tmp_path)
    captured = _stub_runner(monkeypatch, setup_rc=1)
    result = runner.invoke(app, ["dev"])
    assert result.exit_code == 1
    assert captured == {}  # install failed -> no launch
    assert "boom" in result.output


def test_dev_server_nonzero_exit_propagates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_app(tmp_path)
    _stub_runner(monkeypatch)
    monkeypatch.setattr("aai_cli.init.runner.launch_and_open", lambda *a, **k: 3)
    result = runner.invoke(app, ["dev", "--no-open"])
    assert result.exit_code == 3


def test_dev_json_emits_install_step(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_app(tmp_path)
    _stub_runner(monkeypatch)
    result = runner.invoke(app, ["dev", "--no-open", "--json"])
    assert result.exit_code == 0, result.output
    assert '"name": "install"' in result.output
    assert '"detail": "uv"' in result.output


def test_dev_venv_path_when_no_uv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_app(tmp_path)
    captured = _stub_runner(monkeypatch, use_uv=False)
    result = runner.invoke(app, ["dev", "--no-open", "--json"])
    assert result.exit_code == 0, result.output
    assert captured["use_uv"] is False
    assert '"detail": "venv + pip"' in result.output


def test_dev_custom_port_flows_through(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_app(tmp_path)
    captured = _stub_runner(monkeypatch)
    result = runner.invoke(app, ["dev", "--port", "8123", "--no-open"])
    assert result.exit_code == 0, result.output
    assert captured["port"] == 8123
