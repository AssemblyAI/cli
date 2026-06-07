from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from aai_cli.auth import endpoints
from aai_cli.errors import APIError

# The callback URL carries the single-use OAuth token (and the state nonce) in its
# query string, so it would otherwise linger in the browser's history and address
# bar. Scrub it from the current history entry with replaceState the moment the page
# loads — no extra request to race the server shutdown, unlike a redirect. The token
# is already spent server-side by the time the user reads this, but keeping it out of
# history is the OAuth-for-native-apps (RFC 8252) hygiene. The page reflects no query
# data, so there is nothing to inject; the script is a static literal.
_SUCCESS_HTML = (
    b"<html><body style='font-family:sans-serif'>"
    b"<script>history.replaceState(null,'',location.pathname)</script>"
    b"<h2>Signed in.</h2><p>You can close this tab and return to the terminal.</p>"
    b"</body></html>"
)


@dataclass
class CallbackResult:
    token: str | None = None
    token_type: str | None = None
    error: str | None = None


def capture_callback(
    expected_state: str,
    timeout: float = 120.0,  # pragma: no mutate (default window; tests pass explicit timeouts)
) -> CallbackResult:
    """Bind the fixed loopback port, capture one OAuth callback, return its token.

    Only a callback whose `state` query parameter equals `expected_state` is
    accepted; any other request (wrong/missing state, or a different path) gets a
    4xx and the server keeps waiting, so a forged callback can't complete someone
    else's login. Returns a CallbackResult; `error="timeout"` if no matching
    callback arrives in time.
    """
    result = CallbackResult()
    done = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # stdlib API name
            parsed = urlparse(self.path)
            if parsed.path != endpoints.LOOPBACK_PATH:
                self.send_response(404)
                self.end_headers()
                return
            qs = parse_qs(parsed.query)
            state = next(iter(qs.get("state", [])), None)
            # Constant-time compare so a forged callback can't probe the nonce by
            # timing. A mismatch is rejected without ending the capture: the real
            # callback can still arrive (otherwise it falls through to timeout).
            if state is None or not secrets.compare_digest(state, expected_state):
                self.send_response(400)
                self.end_headers()
                return
            result.token = next(iter(qs.get("token", [])), None)
            result.token_type = next(iter(qs.get("stytch_token_type", [])), None)
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_SUCCESS_HTML)
            done.set()

        def log_message(self, format: str, *args: object) -> None:  # silence stderr logging
            pass

    try:
        server = HTTPServer((endpoints.LOOPBACK_HOST, endpoints.LOOPBACK_PORT), Handler)
    except OSError as exc:
        raise APIError(
            f"Could not start the login callback server on "
            f"{endpoints.LOOPBACK_HOST}:{endpoints.LOOPBACK_PORT} ({exc}). "
            "Close whatever is using that port and run 'aai login' again."
        ) from exc
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        if not done.wait(timeout):
            result.error = "timeout"
    finally:
        server.shutdown()  # stop serve_forever()
        thread.join(timeout=5)
        server.server_close()  # close the listening socket (shutdown() leaves it open)
    return result
