"""`assembly webhooks listen` engine: a local sink for AssemblyAI webhook deliveries.

Binds a threaded HTTP server on 127.0.0.1 and exposes it through a cloudflared
quick tunnel — the printed public URL is what ``--webhook-url`` wants. Each
delivery is acknowledged with HTTP 200 immediately and printed as it arrives
(one NDJSON record per delivery under ``--json``); ``--forward-to`` re-POSTs
the body to a local app, with forwarding failures reported on the event rather
than to AssemblyAI. Ctrl-C (or ``--max-events``) stops the listener.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

from rich.markup import escape

from aai_cli.core import jsonshape
from aai_cli.core.errors import CLIError
from aai_cli.init import runner, tunnel
from aai_cli.ui import output

if TYPE_CHECKING:
    import subprocess


@dataclass(frozen=True)
class ListenOptions:
    """Every `assembly webhooks listen` flag as plain data (options/run split)."""

    port: int
    forward_to: str | None
    public: bool  # False = --no-tunnel (local sink only)
    max_events: int  # 0 = serve until Ctrl-C


def shape_event(path: str, body: bytes) -> dict[str, object]:
    """The emitted record for one delivery: the parsed payload, plus pulled-up
    ``transcript_id``/``status`` when the body looks like an AssemblyAI webhook."""
    event: dict[str, object] = {"path": path}
    try:
        payload = json.loads(body.decode())
    except ValueError:  # includes UnicodeDecodeError: non-JSON bodies ride as raw text
        event["raw"] = body.decode(errors="replace")
        return event
    event["payload"] = payload
    record = jsonshape.as_mapping(payload)
    if record is not None and "transcript_id" in record:
        event["transcript_id"] = record["transcript_id"]
        event["status"] = record.get("status")
    return event


def forward(url: str, body: bytes, content_type: str) -> dict[str, object]:
    """Re-POST a delivery to ``url``; the per-event forwarding-outcome record.

    A refused/failed forward is data on the event, never an exception — the
    delivery was already acknowledged to AssemblyAI.
    """
    import httpx2 as httpx  # deferred: imported per delivery, keeps CLI startup light

    try:
        response = httpx.post(
            url,
            content=body,
            headers={"content-type": content_type},
            timeout=10,  # pragma: no mutate (tuning constant; no unit-observable behavior)
        )
    except httpx.HTTPError as err:
        return {"url": url, "error": str(err)}
    return {"url": url, "status_code": response.status_code}


def _render_event(event: dict[str, object]) -> str:
    parts = [f"[aai.heading]→ POST[/aai.heading] {escape(str(event['path']))}"]
    if "transcript_id" in event:
        parts.append(f"transcript_id={escape(str(event['transcript_id']))}")
        parts.append(f"status={escape(str(event['status']))}")
    elif "raw" in event:
        parts.append(escape(str(event["raw"])))
    fwd = jsonshape.as_mapping(event.get("forward"))
    if fwd is not None:
        outcome = fwd.get("error") if "error" in fwd else fwd.get("status_code")
        parts.append(
            f"[aai.muted]→ forwarded to {escape(str(fwd.get('url')))}: "
            f"{escape(str(outcome))}[/aai.muted]"
        )
    return "  ".join(parts)


class _EventSink:
    """Serializes deliveries from handler threads into one printed record each."""

    def __init__(self, *, forward_to: str | None, max_events: int, json_mode: bool) -> None:
        self._forward_to = forward_to
        self._max_events = max_events
        self._json_mode = json_mode
        self._lock = threading.Lock()
        self._count = 0
        self.on_limit: Callable[[], None] = lambda: None  # the listen loop's shutdown

    def handle(self, path: str, body: bytes, content_type: str) -> None:
        event = shape_event(path, body)
        if self._forward_to is not None:
            event["forward"] = forward(self._forward_to, body, content_type)
        with self._lock:
            self._count += 1
            output.emit(event, _render_event, json_mode=self._json_mode)
            if self._max_events and self._count >= self._max_events:
                self.on_limit()


def _make_handler(sink: _EventSink) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            # Acknowledge before printing/forwarding: AssemblyAI only needs a 2xx.
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok": true}')
            sink.handle(self.path, body, self.headers.get("Content-Type") or "application/json")

        def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
            """BaseHTTPRequestHandler logs every request to stderr; stay quiet."""
            del code, size

    return Handler


def _render_listening(data: dict[str, object]) -> str:
    return (
        f"[aai.heading]Listening for webhooks[/aai.heading] "
        f"[aai.url]{escape(str(data['url']))}[/aai.url]\n"
        f"[aai.muted]→ receiving on[/aai.muted] [aai.url]{escape(str(data['local']))}[/aai.url]"
        "  [aai.muted](Ctrl-C to stop)[/aai.muted]\n"
        f"[aai.muted]Try:[/aai.muted] assembly transcribe --sample "
        f"--webhook-url {escape(str(data['url']))}"
    )


def _announce(public_url: str | None, port: int, *, json_mode: bool) -> None:
    local = f"http://127.0.0.1:{port}"
    payload: dict[str, object] = {"url": public_url or local, "local": local, "port": port}
    output.emit(payload, _render_listening, json_mode=json_mode)


def run_listen(opts: ListenOptions, *, json_mode: bool) -> None:
    """Bind the sink, open the tunnel, and serve deliveries until stopped."""
    if opts.public:
        tunnel.require_cloudflared("expose a public webhook URL")
    port = runner.find_free_port(opts.port)
    sink = _EventSink(forward_to=opts.forward_to, max_events=opts.max_events, json_mode=json_mode)
    server = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(sink))
    # shutdown() is called from a handler thread; serve_forever (main thread)
    # notices within its poll interval and returns.
    sink.on_limit = server.shutdown
    proxy: subprocess.Popen[str] | None = None
    log_path: Path | None = None
    keep_log = False
    try:
        public_url: str | None = None
        if opts.public:
            proxy, public_url, log_path = tunnel.open_quick_tunnel(port, cwd=Path.cwd())
            if public_url is None:
                # Keep the captured cloudflared output: it's the only evidence of
                # why the tunnel never came up.
                keep_log = True
                raise CLIError(
                    "cloudflared didn't report a tunnel URL in time.",
                    error_type="tunnel_error",
                    exit_code=1,
                    suggestion=f"cloudflared's output was kept at {log_path} — "
                    "check it for errors.",
                )
        _announce(public_url, port, json_mode=json_mode)
        # Ctrl-C is the expected way to stop a foreground listener; let it propagate so
        # the command exits 130 (cancel). The finally below still closes the socket and
        # tears down the tunnel.
        server.serve_forever()
    finally:
        # Close here, not via `with server:` — a tunnel failure raises before the
        # serve block, and the bound listening socket must not outlive the command.
        server.server_close()
        tunnel.terminate(proxy)
        if log_path is not None and not keep_log:
            log_path.unlink(missing_ok=True)
