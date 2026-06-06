import socket
import threading
import time
import urllib.request

import pytest

from aai_cli.auth import endpoints, loopback
from aai_cli.errors import APIError


def _hit(path: str) -> None:
    url = f"http://{endpoints.LOOPBACK_HOST}:{endpoints.LOOPBACK_PORT}{path}"
    # Retry briefly until the server thread is bound.
    for _ in range(50):
        try:
            urllib.request.urlopen(url, timeout=2).read()  # noqa: S310 - fixed localhost URL
            return
        except OSError:
            time.sleep(0.05)


def test_capture_returns_token_and_type():
    result_box = {}

    def run():
        result_box["result"] = loopback.capture_callback(timeout=5.0)

    t = threading.Thread(target=run)
    t.start()
    _hit("/callback?stytch_token_type=discovery_oauth&token=tok_abc")
    t.join(timeout=5)

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
    _hit("/favicon.ico")  # unknown path -> 404, capture stays open
    _hit("/callback?stytch_token_type=discovery_oauth&token=tok_late")
    t.join(timeout=5)

    result = result_box["result"]
    assert result.token == "tok_late"


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
