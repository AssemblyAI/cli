import http.client
import socket
import threading
import time

import pytest

from aai_cli.auth import endpoints, loopback
from aai_cli.errors import APIError


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
        result_box["result"] = loopback.capture_callback("st8", timeout=5.0)

    t = threading.Thread(target=run)
    t.start()
    status = _hit("/callback?state=st8&stytch_token_type=discovery_oauth&token=tok_abc")
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
        result_box["result"] = loopback.capture_callback("st8", timeout=5.0)

    t = threading.Thread(target=run)
    t.start()
    assert _hit("/favicon.ico") == 404  # unknown path -> 404, capture stays open
    _hit("/callback?state=st8&stytch_token_type=discovery_oauth&token=tok_late")
    t.join(timeout=5)

    result = result_box["result"]
    assert result.token == "tok_late"


def test_capture_rejects_mismatched_state():
    # A callback with the wrong state nonce (a forged/login-CSRF attempt) is refused
    # with a 400 and does not end the capture; the genuine callback then completes it.
    result_box = {}

    def run():
        result_box["result"] = loopback.capture_callback("good", timeout=5.0)

    t = threading.Thread(target=run)
    t.start()
    assert _hit("/callback?state=evil&stytch_token_type=discovery_oauth&token=tok_bad") == 400
    _hit("/callback?state=good&stytch_token_type=discovery_oauth&token=tok_ok")
    t.join(timeout=5)

    result = result_box["result"]
    assert result.token == "tok_ok"  # the forged token was never captured


def test_capture_rejects_missing_state():
    # A callback with no state at all is refused (400) and never captured.
    result_box = {}

    def run():
        result_box["result"] = loopback.capture_callback("good", timeout=0.8)

    t = threading.Thread(target=run)
    t.start()
    assert _hit("/callback?stytch_token_type=discovery_oauth&token=tok_bad") == 400
    t.join(timeout=5)

    result = result_box["result"]
    assert result.error == "timeout"
    assert result.token is None


def test_capture_times_out_without_callback():
    result = loopback.capture_callback("st8", timeout=0.3)
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
            loopback.capture_callback("st8", timeout=1.0)
    finally:
        busy.close()
