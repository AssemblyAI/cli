from __future__ import annotations

import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from aai_cli.auth import endpoints
from aai_cli.errors import APIError

# The callback URL carries the single-use OAuth token in its
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


@dataclass
class CallbackCapture:
    """A loopback callback server that is already bound and serving.

    Splitting the bind (`start_capture`) from the blocking wait lets the login flow
    fail on a taken port *before* it sends the user's browser into the OAuth flow.
    `wait()` blocks for one matching callback and always shuts the server down.
    """

    result: CallbackResult
    done: threading.Event
    server: HTTPServer
    thread: threading.Thread
    lock: threading.Lock

    def wait(
        self,
        timeout: float = 120.0,  # pragma: no mutate (default window; tests pass explicit timeouts)
    ) -> CallbackResult:
        """Block for one OAuth callback (or the timeout), then shut the server down.

        Returns the CallbackResult; `error="timeout"` if no matching callback
        arrived in time.
        """
        try:
            if not self.done.wait(timeout):
                # Claim the capture under the lock: the handler thread may be
                # processing a callback that arrived right at the deadline. If it
                # already claimed (done set), its token result stands; otherwise the
                # timeout claims it, and a late callback can no longer mutate the
                # result this method is about to hand to the caller.
                with self.lock:
                    if not self.done.is_set():
                        self.result.error = "timeout"
                        self.done.set()
        finally:
            self.server.shutdown()  # stop serve_forever()
            self.thread.join(timeout=5)  # pragma: no mutate (cleanup grace period only)
            self.server.server_close()  # close the listening socket (shutdown() leaves it open)
        return self.result


def start_capture() -> CallbackCapture:
    """Bind the fixed loopback port and start serving; the returned capture's
    ``wait()`` collects one OAuth callback.

    Raises a clean APIError when the bind fails (port taken) so callers can abort
    before opening the browser. Only a callback to the registered path that carries
    a `token` is accepted; any other request (a different path, or no token) gets a
    4xx and the server keeps waiting, so a stray request can't end the capture early.
    The first matching callback wins: a duplicate (browser reload/double-click, or
    anything else hitting the loopback port afterwards) is acknowledged but can never
    overwrite the captured token.
    """
    result = CallbackResult()
    done = threading.Event()
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # stdlib API name
            parsed = urlparse(self.path)
            if parsed.path != endpoints.LOOPBACK_PATH:
                self.send_response(404)
                self.end_headers()
                return
            qs = parse_qs(parsed.query)
            token = next(iter(qs.get("token", [])), None)
            # A callback with no token (a stray or preflight request) is rejected
            # without ending the capture: the genuine callback can still arrive
            # (otherwise it falls through to timeout).
            if token is None:
                self.send_response(400)
                self.end_headers()
                return
            # First claim wins: once the capture is done (a prior callback, or the
            # timeout in wait()), the result is already in the caller's hands, so a
            # late or duplicate callback must not mutate it. The lock pairs with
            # wait()'s timeout claim so the two threads can't interleave mid-write.
            with lock:
                if not done.is_set():
                    result.token = token
                    result.token_type = next(iter(qs.get("stytch_token_type", [])), None)
                    done.set()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_SUCCESS_HTML)

        def log_message(self, format: str, *args: object) -> None:  # silence stderr logging
            pass

    port = endpoints.loopback_port()
    try:
        server = HTTPServer((endpoints.LOOPBACK_HOST, port), Handler)
    except OSError as exc:
        raise APIError(
            f"Could not start the login callback server on "
            f"{endpoints.LOOPBACK_HOST}:{port} ({exc}). "
            "Close whatever is using that port and run 'assembly login' again."
        ) from exc
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return CallbackCapture(result=result, done=done, server=server, thread=thread, lock=lock)


def capture_callback(
    timeout: float = 120.0,  # pragma: no mutate (default window; tests pass explicit timeouts)
) -> CallbackResult:
    """Bind the port, capture one OAuth callback, and shut down (one-shot helper)."""
    return start_capture().wait(timeout)
