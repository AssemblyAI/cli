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
    url: str | None = "https://happy-slug.trycloudflare.com",
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


def test_share_missing_cloudflared_errors_with_brew_hint_on_macos(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    monkeypatch.setattr("sys.platform", "darwin")
    _stub(monkeypatch, has_cloudflared=False)
    result = runner.invoke(app, ["share"])
    assert result.exit_code == 1
    assert "brew install cloudflared" in result.output


def test_share_missing_cloudflared_errors_with_docs_url_on_linux(tmp_path, monkeypatch):
    # brew is useless advice off macOS; Linux gets Cloudflare's official install docs.
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    monkeypatch.setattr("sys.platform", "linux")
    _stub(monkeypatch, has_cloudflared=False)
    result = runner.invoke(app, ["share"])
    assert result.exit_code == 1
    # Rich wraps the long URL mid-token, so compare with all whitespace removed.
    packed = "".join(result.output.split())
    assert (
        "https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
        in packed
    )
    assert "brewinstallcloudflared" not in packed


def test_cloudflared_install_hint_per_platform(monkeypatch):
    from aai_cli.commands import share as share_cmd

    monkeypatch.setattr("sys.platform", "darwin")
    assert share_cmd._cloudflared_install_hint() == "Install it: brew install cloudflared"
    monkeypatch.setattr("sys.platform", "linux")
    assert share_cmd._cloudflared_install_hint() == (
        "Install it: "
        "https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
    )


def test_share_binds_loopback_not_procfile_wildcard(tmp_path, monkeypatch):
    # share serves the LAN-facing side through cloudflared only; the local server
    # itself must bind loopback, not the Procfile's 0.0.0.0.
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    server, proxy = _stub(monkeypatch)
    seq = iter([server, proxy])
    commands = []

    def spawn(command, **kwargs):
        commands.append(command)
        return next(seq)

    monkeypatch.setattr("aai_cli.init.runner.spawn", spawn)
    result = runner.invoke(app, ["share"])
    assert result.exit_code == 0, result.output
    dev_cmd = commands[0]
    assert dev_cmd[dev_cmd.index("--host") + 1] == "127.0.0.1"
    # The Procfile's wildcard bind must not survive into the local server command.
    wildcard_host = WEB.split("--host ")[1].split(maxsplit=1)[0]
    assert wildcard_host not in dev_cmd


def test_share_missing_procfile_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _stub(monkeypatch)
    result = runner.invoke(app, ["share"])
    assert result.exit_code == 1
    assert "assembly init" in result.output


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


def _capture_tunnel_log(monkeypatch, server, proxy):
    """Re-patch runner.spawn to record the cloudflared log path (after _stub)."""
    seq = iter([server, proxy])
    logs = []

    def spawn(command, **kwargs):
        if kwargs.get("log_path") is not None:
            logs.append(kwargs["log_path"])
        return next(seq)

    monkeypatch.setattr("aai_cli.init.runner.spawn", spawn)
    return logs


def test_share_tunnel_timeout_keeps_log_and_points_at_it(tmp_path, monkeypatch):
    # On "didn't report a tunnel URL in time", cloudflared's captured output is the
    # only evidence — the error must name the log file, and the file must survive.
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    server, proxy = _stub(
        monkeypatch, url=None, server=_FakeProc(poll_rc=None), proxy=_FakeProc(poll_rc=None)
    )
    logs = _capture_tunnel_log(monkeypatch, server, proxy)
    result = runner.invoke(app, ["share"])
    assert result.exit_code == 1
    [log] = logs
    try:
        assert log.exists()  # the evidence is kept on the failure path
        packed = "".join(result.output.split())  # the suggestion line may soft-wrap
        assert str(log) in packed
        assert "checkitforerrors" in packed
    finally:
        log.unlink(missing_ok=True)


def test_share_deletes_tunnel_log_on_clean_exit(tmp_path, monkeypatch):
    # A successful share must not leave aai-tunnel-*.log litter in /tmp.
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    server, proxy = _stub(monkeypatch)
    logs = _capture_tunnel_log(monkeypatch, server, proxy)
    result = runner.invoke(app, ["share"])
    assert result.exit_code == 0, result.output
    [log] = logs
    assert not log.exists()
    assert str(log) not in result.output  # nothing points the user at a deleted file


def test_share_log_cleanup_tolerates_already_missing_file(tmp_path, monkeypatch):
    # If the log vanished before cleanup, the unlink must not blow up the command.
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    _stub(monkeypatch)

    def await_and_remove(log_path, **kwargs):
        log_path.unlink()
        return "https://happy-slug.trycloudflare.com"

    monkeypatch.setattr("aai_cli.init.tunnel.await_url", await_and_remove)
    result = runner.invoke(app, ["share"])
    assert result.exit_code == 0, result.output


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


def test_share_busy_port_notice_on_stderr(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    _stub(monkeypatch)
    monkeypatch.setattr("aai_cli.init.runner.find_free_port", lambda p, **k: p + 1)
    result = runner.invoke(app, ["share", "--port", "5000"])
    assert result.exit_code == 0, result.output
    assert "Port 5000 is in use; using 5001." in result.stderr


def test_share_busy_port_notice_suppressed_by_quiet(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    _stub(monkeypatch)
    monkeypatch.setattr("aai_cli.init.runner.find_free_port", lambda p, **k: p + 1)
    result = runner.invoke(app, ["--quiet", "share", "--port", "5000"])
    assert result.exit_code == 0, result.output
    assert "is in use" not in result.output


def test_share_json_emits_url(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    _stub(monkeypatch)
    result = runner.invoke(app, ["share", "--json"])
    assert result.exit_code == 0, result.output
    assert '"url"' in result.output
    assert "happy-slug.trycloudflare.com" in result.output


def test_share_tunnel_env_excludes_api_key(tmp_path, monkeypatch):
    # The tunnel binary only proxies a port; the API key the dev server inherits
    # must not reach cloudflared's environment (logs/diagnostics could leak it).
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk-secret")
    server, proxy = _stub(monkeypatch)
    seq = iter([server, proxy])
    spawn_envs = []

    def spawn(command, **kwargs):
        spawn_envs.append(kwargs.get("env"))
        return next(seq)

    monkeypatch.setattr("aai_cli.init.runner.spawn", spawn)
    result = runner.invoke(app, ["share"])
    assert result.exit_code == 0, result.output
    dev_env, tunnel_env = spawn_envs
    assert dev_env["ASSEMBLYAI_API_KEY"] == "sk-secret"  # the user's app needs it
    assert "ASSEMBLYAI_API_KEY" not in tunnel_env
    assert tunnel_env["PATH"] == dev_env["PATH"]  # everything else passes through
