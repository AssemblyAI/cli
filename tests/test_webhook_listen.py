"""`assembly webhooks listen`: the local sink, event shaping, forwarding, and the
cloudflared tunnel wiring.

The serving tests bind a real loopback server (tight allow_hosts opt-in) and POST
to it from a background thread; pytest-timeout bounds them so a listener that
never shuts down fails instead of wedging the run.
"""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from typer.testing import CliRunner

from aai_cli.commands.webhooks import _listen as webhook_listen
from aai_cli.main import app

runner = CliRunner()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _post_when_up(port, payload, results):
    import httpx2 as httpx

    from aai_cli.init import runner as init_runner

    assert init_runner.wait_for_port(port, timeout=15)
    results.append(httpx.post(f"http://127.0.0.1:{port}/", json=payload, timeout=10))


def _ndjson(result):
    return [json.loads(line) for line in result.output.splitlines() if line.startswith("{")]


# --- receiving deliveries ----------------------------------------------------------


@pytest.mark.allow_hosts(["127.0.0.1"])
@pytest.mark.timeout(60)
def test_listen_no_tunnel_emits_banner_and_event_ndjson():
    port = _free_port()
    results = []
    payload = {"transcript_id": "t_1", "status": "completed"}
    poster = threading.Thread(target=_post_when_up, args=(port, payload, results))
    poster.start()
    try:
        result = runner.invoke(
            app,
            ["webhooks", "listen", "--no-tunnel", "--port", str(port), "--max-events", "1", "-j"],
        )
    finally:
        poster.join()
    assert result.exit_code == 0, result.output
    local = f"http://127.0.0.1:{port}"
    banner = next(o for o in _ndjson(result) if "local" in o)
    assert banner == {"url": local, "local": local, "port": port}
    event = next(o for o in _ndjson(result) if "payload" in o)
    assert event["path"] == "/"
    assert event["payload"] == payload
    assert event["transcript_id"] == "t_1"
    assert event["status"] == "completed"
    assert "forward" not in event
    # The delivery was acknowledged the way AssemblyAI expects: 200 + JSON body.
    assert results[0].status_code == 200
    assert results[0].json() == {"ok": True}


@pytest.mark.allow_hosts(["127.0.0.1"])
@pytest.mark.timeout(60)
def test_listen_human_mode_prints_hint_and_event_line():
    port = _free_port()
    results = []
    poster = threading.Thread(
        target=_post_when_up, args=(port, {"transcript_id": "t_9", "status": "error"}, results)
    )
    poster.start()
    try:
        result = runner.invoke(
            app, ["webhooks", "listen", "--no-tunnel", "--port", str(port), "--max-events", "1"]
        )
    finally:
        poster.join()
    assert result.exit_code == 0, result.output
    assert "Listening for webhooks" in result.output
    assert f"http://127.0.0.1:{port}" in result.output
    assert "--webhook-url" in result.output  # the copy-paste hint
    assert "t_9" in result.output
    assert "status=error" in result.output
    assert "{" not in result.output.replace('{"ok": true}', "")  # no NDJSON in human mode


@pytest.mark.parametrize("bad_port", ["-1", "99999"])
def test_listen_rejects_out_of_range_port(bad_port):
    # An out-of-range port is a user-input error (exit 2, validated at parse time),
    # never an "Unexpected error … report a bug" internal failure from the socket layer.
    result = runner.invoke(app, ["webhooks", "listen", "--port", bad_port])
    assert result.exit_code == 2
    # The exact bounds in the message pin the min=0/max=65535 literals against mutation.
    assert "0<=x<=65535" in result.output
    assert "report it" not in result.output


# --- forwarding --------------------------------------------------------------------


class _Receiver(ThreadingHTTPServer):
    """A target app double for --forward-to: records each POST, answers 204."""

    def __init__(self):
        self.seen = []

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                outer.seen.append((self.path, body, self.headers.get("Content-Type")))
                self.send_response(204)
                self.end_headers()

            def log_request(self, code="-", size="-"):
                pass

        super().__init__(("127.0.0.1", 0), Handler)


@pytest.mark.allow_hosts(["127.0.0.1"])
@pytest.mark.timeout(60)
def test_forward_to_relays_the_body_and_reports_the_status():
    receiver = _Receiver()
    receiver_thread = threading.Thread(target=receiver.serve_forever, daemon=True)
    receiver_thread.start()
    forward_url = f"http://127.0.0.1:{receiver.server_address[1]}/hook"
    port = _free_port()
    results = []
    payload = {"transcript_id": "t_fw", "status": "completed"}
    poster = threading.Thread(target=_post_when_up, args=(port, payload, results))
    poster.start()
    try:
        result = runner.invoke(
            app,
            [
                "webhooks",
                "listen",
                "--no-tunnel",
                "--port",
                str(port),
                "--forward-to",
                forward_url,
                "--max-events",
                "1",
                "--json",
            ],
        )
    finally:
        poster.join()
        receiver.shutdown()
        receiver_thread.join()
        receiver.server_close()
    assert result.exit_code == 0, result.output
    event = next(o for o in _ndjson(result) if "payload" in o)
    assert event["forward"] == {"url": forward_url, "status_code": 204}
    # The body is relayed byte-for-byte (httpx serializes json= compactly).
    compact = json.dumps(payload, separators=(",", ":")).encode()
    assert receiver.seen == [("/hook", compact, "application/json")]


@pytest.mark.allow_hosts(["127.0.0.1"])
@pytest.mark.timeout(60)
def test_forward_failure_is_reported_on_the_event_not_fatal():
    closed = _free_port()  # nothing listens here: the forward is refused
    port = _free_port()
    results = []
    poster = threading.Thread(target=_post_when_up, args=(port, {"transcript_id": "x"}, results))
    poster.start()
    try:
        result = runner.invoke(
            app,
            [
                "webhooks",
                "listen",
                "--no-tunnel",
                "--port",
                str(port),
                "--forward-to",
                f"http://127.0.0.1:{closed}/hook",
                "--max-events",
                "1",
                "--json",
            ],
        )
    finally:
        poster.join()
    assert result.exit_code == 0, result.output  # delivery was still acknowledged
    event = next(o for o in _ndjson(result) if "payload" in o)
    assert event["forward"]["url"] == f"http://127.0.0.1:{closed}/hook"
    assert event["forward"]["error"]
    assert results[0].status_code == 200


# --- the sink and event shaping (no sockets) ---------------------------------------


def test_shape_event_pulls_up_transcript_fields():
    event = webhook_listen.shape_event("/", b'{"transcript_id": "t", "status": "completed"}')
    assert event == {
        "path": "/",
        "payload": {"transcript_id": "t", "status": "completed"},
        "transcript_id": "t",
        "status": "completed",
    }


def test_shape_event_non_json_body_rides_as_raw_text():
    assert webhook_listen.shape_event("/x", b"\xffnot json") == {
        "path": "/x",
        "raw": "�not json",
    }


def test_shape_event_non_dict_payload_has_no_transcript_fields():
    assert webhook_listen.shape_event("/", b"[1, 2]") == {"path": "/", "payload": [1, 2]}


def test_shape_event_dict_without_transcript_id():
    assert webhook_listen.shape_event("/", b'{"hello": 1}') == {
        "path": "/",
        "payload": {"hello": 1},
    }


def test_render_event_shows_forward_status_and_error():
    line = webhook_listen._render_event(
        {
            "path": "/",
            "transcript_id": "t_r",
            "status": "completed",
            "forward": {"url": "http://app/hook", "status_code": 200},
        }
    )
    assert "t_r" in line
    assert "status=completed" in line
    assert "http://app/hook" in line
    assert "200" in line
    line = webhook_listen._render_event(
        {"path": "/", "raw": "plain", "forward": {"url": "u", "error": "boom"}}
    )
    assert "plain" in line
    assert "boom" in line


def _sink_events(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_sink_max_events_zero_never_fires_the_limit(capsys):
    sink = webhook_listen._EventSink(forward_to=None, max_events=0, json_mode=True)
    fired = []
    sink.on_limit = lambda: fired.append(1)
    sink.handle("/", b"{}", "application/json")
    sink.handle("/", b"{}", "application/json")
    assert fired == []
    assert len(_sink_events(capsys)) == 2  # both deliveries still emitted


def test_sink_fires_the_limit_exactly_at_max_events(capsys):
    sink = webhook_listen._EventSink(forward_to=None, max_events=2, json_mode=True)
    fired = []
    sink.on_limit = lambda: fired.append(1)
    sink.handle("/", b"{}", "application/json")
    assert fired == []  # one short of the limit: keep serving
    sink.handle("/", b"{}", "application/json")
    assert fired == [1]
    assert len(_sink_events(capsys)) == 2


# --- tunnel wiring (cloudflared faked) ----------------------------------------------


class _FakeProc:
    def __init__(self):
        self.terminated = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True


def _raise_interrupt(self):
    raise KeyboardInterrupt


def _stub_tunnel(monkeypatch, tmp_path, *, url):
    proc = _FakeProc()
    log = tmp_path / "tunnel.log"
    log.write_text("cloudflared output")
    seen = {}
    real_port = _free_port()

    def fake_find_free_port(preferred, **kwargs):
        seen["preferred"] = preferred
        return real_port

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/cloudflared")
    monkeypatch.setattr("aai_cli.init.runner.find_free_port", fake_find_free_port)
    monkeypatch.setattr(
        "aai_cli.init.tunnel.open_quick_tunnel", lambda port, *, cwd: (proc, url, log)
    )
    # Stop immediately: the listener loop isn't under test here.
    monkeypatch.setattr(
        "aai_cli.commands.webhooks._listen.ThreadingHTTPServer.serve_forever", _raise_interrupt
    )
    return proc, log, seen, real_port


@pytest.mark.allow_hosts(["127.0.0.1"])
def test_listen_public_prints_tunnel_url_and_cleans_up(tmp_path, monkeypatch):
    proc, log, seen, real_port = _stub_tunnel(
        monkeypatch, tmp_path, url="https://hook-slug.trycloudflare.com"
    )
    result = runner.invoke(app, ["webhooks", "listen"])
    # Ctrl-C stops the foreground listener: exit 130 (cancel), still cleaning up below.
    assert result.exit_code == 130, result.output
    assert seen["preferred"] == 8989  # the documented default port
    assert "Listening for webhooks https://hook-slug.trycloudflare.com" in result.output
    # Rich wraps the long hint line mid-token; compare with whitespace removed.
    assert "--webhook-urlhttps://hook-slug.trycloudflare.com" in "".join(result.output.split())
    assert f"127.0.0.1:{real_port}" in result.output
    assert proc.terminated is True
    assert not log.exists()  # a clean exit must not leave aai-tunnel-*.log litter


@pytest.mark.allow_hosts(["127.0.0.1"])
def test_listen_tunnel_url_timeout_errors_and_keeps_the_log(tmp_path, monkeypatch):
    proc, log, _seen, _port = _stub_tunnel(monkeypatch, tmp_path, url=None)
    result = runner.invoke(app, ["webhooks", "listen"])
    assert result.exit_code == 1
    assert "didn't report a tunnel URL" in result.output
    assert str(log) in "".join(result.output.split())  # Rich may wrap the path
    assert log.exists()  # kept: it's the only evidence of why the tunnel failed
    assert proc.terminated is True


@pytest.mark.allow_hosts(["127.0.0.1"])
def test_listen_accepts_explicit_max_events_zero(monkeypatch):
    # 0 is the documented "until Ctrl-C" value; the option's floor must not reject it.
    monkeypatch.setattr(
        "aai_cli.commands.webhooks._listen.ThreadingHTTPServer.serve_forever", _raise_interrupt
    )
    result = runner.invoke(app, ["webhooks", "listen", "--no-tunnel", "--max-events", "0"])
    assert result.exit_code == 130, result.output  # Ctrl-C cancel
    assert "Listening for webhooks" in result.output


def test_listen_missing_cloudflared_errors_before_binding(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    result = runner.invoke(app, ["webhooks", "listen"])
    assert result.exit_code == 1
    assert "cloudflared is required to expose a public webhook URL." in result.output
    assert "Install it:" in result.output


def test_webhooks_no_subcommand_shows_help():
    # no_args_is_help=True: bare `assembly webhooks` prints its help (the subcommand list)
    # rather than the bare "Missing command." usage error that no_args_is_help=False emits.
    result = runner.invoke(app, ["webhooks"])
    assert "Missing command" not in result.output
    assert "listen" in result.output
