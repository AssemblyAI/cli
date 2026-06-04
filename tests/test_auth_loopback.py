import threading
import time
import urllib.request

from assemblyai_cli.auth import endpoints, loopback


def _hit(path: str) -> None:
    url = f"http://{endpoints.LOOPBACK_HOST}:{endpoints.LOOPBACK_PORT}{path}"
    # Retry briefly until the server thread is bound.
    for _ in range(50):
        try:
            urllib.request.urlopen(url, timeout=2).read()
            return
        except Exception:
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


def test_capture_times_out_without_callback():
    result = loopback.capture_callback(timeout=0.3)
    assert result.error == "timeout"
    assert result.token is None
