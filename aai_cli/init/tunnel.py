# aai_cli/init/tunnel.py
from __future__ import annotations

import re
import time
from collections.abc import Callable
from pathlib import Path

# cloudflared binary name; resolved via shutil.which by callers.
CLOUDFLARED = "cloudflared"

# A cloudflared quick tunnel prints an ephemeral https://<slug>.trycloudflare.com URL.
_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def tunnel_command(port: int) -> list[str]:
    """The cloudflared quick-tunnel command pointing at the local server."""
    return [CLOUDFLARED, "tunnel", "--url", f"http://localhost:{port}"]


def find_url(text: str) -> str | None:
    """The first trycloudflare.com URL in `text`, or None."""
    match = _URL.search(text)
    return match.group(0) if match else None


def await_url(
    log_path: Path,
    *,
    timeout: float = 30.0,
    interval: float = 0.2,
    sleep: Callable[[float], None] = time.sleep,
) -> str | None:
    """Poll `log_path` (cloudflared's captured output) for the tunnel URL.

    Returns the URL once it appears, or None if it hasn't within `timeout` seconds.
    `sleep` is injectable so tests don't wait on the wall clock.
    """
    deadline = time.monotonic() + timeout
    while True:
        url = find_url(log_path.read_text())
        if url is not None:
            return url
        if time.monotonic() >= deadline:
            return None
        sleep(interval)
