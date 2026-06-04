from __future__ import annotations

import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from aai_cli.auth import endpoints

_SUCCESS_HTML = (
    b"<html><body style='font-family:sans-serif'>"
    b"<h2>Signed in.</h2><p>You can close this tab and return to the terminal.</p>"
    b"</body></html>"
)


@dataclass
class CallbackResult:
    token: str | None = None
    token_type: str | None = None
    error: str | None = None


def capture_callback(timeout: float = 120.0) -> CallbackResult:
    """Bind the fixed loopback port, capture one OAuth callback, return its token.

    Returns a CallbackResult; `error="timeout"` if no callback arrives in time.
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
            result.token = next(iter(qs.get("token", [])), None)
            result.token_type = next(iter(qs.get("stytch_token_type", [])), None)
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_SUCCESS_HTML)
            done.set()

        def log_message(self, *args: object) -> None:  # silence stderr logging
            pass

    server = HTTPServer((endpoints.LOOPBACK_HOST, endpoints.LOOPBACK_PORT), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        if not done.wait(timeout):
            result.error = "timeout"
    finally:
        server.shutdown()
        thread.join(timeout=5)
    return result
