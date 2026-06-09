import subprocess

from typer.testing import CliRunner

from aai_cli.main import app

runner = CliRunner()
WEB = "web: uvicorn api.index:app --host 0.0.0.0 --port ${PORT:-3000}\n"


def _make_project(tmp_path):
    (tmp_path / "Procfile").write_text(WEB)


class _FakeProc:
    def __init__(self, *, wait_rc=0, poll_rc=None, wait_raises=None):
        self._wait_rc = wait_rc
        self._poll_rc = poll_rc
        self._wait_raises = wait_raises
        self.terminated = False

    def wait(self):
        if self._wait_raises is not None:
            raise self._wait_raises
        return self._wait_rc

    def poll(self):
        return self._poll_rc

    def terminate(self):
        self.terminated = True


def _stub(
    monkeypatch,
    *,
    has_cloudflared=True,
    setup_rc=0,
    port_up=True,
    url="https://happy-slug.trycloudflare.com",
    server=None,
    proxy=None,
):
    server = server if server is not None else _FakeProc(poll_rc=0)
    proxy = proxy if proxy is not None else _FakeProc(poll_rc=None)
    monkeypatch.setattr("aai_cli.init.runner.has_uv", lambda: True)
    monkeypatch.setattr("aai_cli.init.runner.find_free_port", lambda p, **k: p)
    monkeypatch.setattr(
        "aai_cli.init.runner.run_setup",
        lambda *a, **k: subprocess.CompletedProcess([], setup_rc, "", "boom"),
    )
    monkeypatch.setattr("aai_cli.init.runner.wait_for_port", lambda p, **k: port_up)
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/cloudflared" if has_cloudflared else None
    )
    monkeypatch.setattr("aai_cli.init.tunnel.await_url", lambda *a, **k: url)
    seq = iter([server, proxy])
    monkeypatch.setattr("aai_cli.init.runner.spawn", lambda *a, **k: next(seq))
    return server, proxy


def test_share_prints_public_url(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    server, proxy = _stub(monkeypatch)
    result = runner.invoke(app, ["share"])
    assert result.exit_code == 0, result.output
    assert "happy-slug.trycloudflare.com" in result.output
    assert "localhost:3000" in result.output
    # proxy still running (poll None) -> terminated; server already exited (poll 0) -> not
    assert proxy.terminated is True
    assert server.terminated is False


def test_share_missing_cloudflared_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    _stub(monkeypatch, has_cloudflared=False)
    result = runner.invoke(app, ["share"])
    assert result.exit_code == 1
    assert "brew install cloudflared" in result.output


def test_share_missing_procfile_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _stub(monkeypatch)
    result = runner.invoke(app, ["share"])
    assert result.exit_code == 1
    assert "aai init" in result.output


def test_share_install_failure_exits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    _stub(monkeypatch, setup_rc=1)
    result = runner.invoke(app, ["share"])
    assert result.exit_code == 1
    assert "boom" in result.output


def test_share_server_didnt_start(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    server, _ = _stub(monkeypatch, port_up=False, server=_FakeProc(poll_rc=None))
    result = runner.invoke(app, ["share"])
    assert result.exit_code == 1
    assert server.terminated is True  # cleaned up even though tunnel never opened


def test_share_no_tunnel_url(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    server, proxy = _stub(
        monkeypatch, url=None, server=_FakeProc(poll_rc=None), proxy=_FakeProc(poll_rc=None)
    )
    result = runner.invoke(app, ["share"])
    assert result.exit_code == 1
    assert server.terminated is True
    assert proxy.terminated is True


def test_share_keyboard_interrupt_is_clean(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    server, proxy = _stub(
        monkeypatch,
        server=_FakeProc(wait_raises=KeyboardInterrupt(), poll_rc=None),
        proxy=_FakeProc(poll_rc=None),
    )
    result = runner.invoke(app, ["share"])
    assert result.exit_code == 0
    assert server.terminated is True
    assert proxy.terminated is True


def test_share_json_emits_url(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    _stub(monkeypatch)
    result = runner.invoke(app, ["share", "--json"])
    assert result.exit_code == 0, result.output
    assert '"url"' in result.output
    assert "happy-slug.trycloudflare.com" in result.output
