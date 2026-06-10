import http.client
import socket
import threading
import time

import pytest

from aai_cli.auth import endpoints, loopback
from aai_cli.errors import APIError

# These tests bind a real loopback HTTP server and connect to it, so they opt back
# into sockets past the suite-wide --disable-socket (see pyproject pytest config).
# Restricting to 127.0.0.1 keeps the external-network block intact.
pytestmark = pytest.mark.allow_hosts(["127.0.0.1"])


@pytest.fixture(autouse=True)
def _unique_loopback_port(monkeypatch):
    """Give every test its own OS-assigned loopback port.

    Production binds the single fixed ``LOOPBACK_PORT`` (one login at a time), but the
    suite runs many capture cycles back-to-back. Sharing one fixed port makes the tests
    flaky under load/random ordering: if one test's server is still bound when the next
    starts, the bind raises inside the worker thread (surfacing as an unhandled-thread
    warning) and cascades into the neighbours. A fresh port per test removes that
    coupling without changing what `capture_callback` does.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((endpoints.LOOPBACK_HOST, 0))
        port = probe.getsockname()[1]
    monkeypatch.setattr(endpoints, "LOOPBACK_PORT", port)


def _hit(path: str) -> int | None:
    """Request `path` against the loopback server, returning the HTTP status code.

    Uses http.client (not urllib) so a 404 comes back as a normal response status
    rather than a raised HTTPError, and so no urllib audit suppression is needed.
    """
    # Retry briefly until the server thread is bound.
    for _ in range(50):
        conn = http.client.HTTPConnection(
            endpoints.LOOPBACK_HOST, endpoints.LOOPBACK_PORT, timeout=2
        )
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            resp.read()
            return resp.status
        except OSError:
            time.sleep(0.05)
        finally:
            conn.close()
    return None


def test_capture_returns_token_and_type():
    result_box = {}

    def run():
        result_box["result"] = loopback.capture_callback(timeout=5.0)

    t = threading.Thread(target=run)
    t.start()
    status = _hit("/callback?stytch_token_type=discovery_oauth&token=tok_abc")
    t.join(timeout=5)

    assert status == 200  # the callback is acknowledged with 200 OK
    result = result_box["result"]
    assert result.token == "tok_abc"
    assert result.token_type == "discovery_oauth"
    assert result.error is None


def test_capture_ignores_unknown_paths():
    # A request to a non-callback path gets a 404 and the server keeps waiting; the
    # real callback that follows still completes the capture.
    result_box = {}

    def run():
        result_box["result"] = loopback.capture_callback(timeout=5.0)

    t = threading.Thread(target=run)
    t.start()
    assert _hit("/favicon.ico") == 404  # unknown path -> 404, capture stays open
    _hit("/callback?stytch_token_type=discovery_oauth&token=tok_late")
    t.join(timeout=5)

    result = result_box["result"]
    assert result.token == "tok_late"


def _body(path: str) -> bytes:
    """Fetch `path` once (no retry) and return the response body.

    Callers first confirm the server is bound via `_hit`, so no readiness loop is
    needed here.
    """
    conn = http.client.HTTPConnection(endpoints.LOOPBACK_HOST, endpoints.LOOPBACK_PORT, timeout=2)
    try:
        conn.request("GET", path)
        return conn.getresponse().read()
    finally:
        conn.close()


def test_success_page_scrubs_token_from_history():
    # The callback URL holds the single-use token in its query string; the success
    # page must drop it from browser history rather than leave it lingering.
    def run():
        loopback.capture_callback(timeout=5.0)

    t = threading.Thread(target=run)
    t.start()
    assert _hit("/favicon.ico") == 404  # wait until the server is bound (keeps capture open)
    body = _body("/callback?stytch_token_type=discovery_oauth&token=tok_abc")
    t.join(timeout=5)

    assert b"replaceState" in body  # the query (token) is scrubbed client-side
    assert b"tok_abc" not in body  # the page never reflects the token itself


def test_capture_rejects_callback_without_token():
    # A callback to /callback that carries no token (a stray/preflight request) is
    # refused with a 400 and does not end the capture; the real callback completes it.
    result_box = {}

    def run():
        result_box["result"] = loopback.capture_callback(timeout=5.0)

    t = threading.Thread(target=run)
    t.start()
    assert _hit("/callback?stytch_token_type=discovery_oauth") == 400
    _hit("/callback?stytch_token_type=discovery_oauth&token=tok_ok")
    t.join(timeout=5)

    result = result_box["result"]
    assert result.token == "tok_ok"  # the tokenless request never ended the capture


def test_capture_times_out_without_callback():
    result = loopback.capture_callback(timeout=0.3)
    assert result.error == "timeout"
    assert result.token is None


def test_capture_raises_clean_error_when_port_unavailable(monkeypatch):
    # Occupy a port, then point the callback server at it: binding must fail with a
    # clean APIError, not a raw OSError traceback escaping run_login_flow.
    busy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    busy.bind((endpoints.LOOPBACK_HOST, 0))
    busy.listen(1)
    port = busy.getsockname()[1]
    monkeypatch.setattr(endpoints, "LOOPBACK_PORT", port)
    try:
        with pytest.raises(APIError):
            loopback.capture_callback(timeout=1.0)
    finally:
        busy.close()
