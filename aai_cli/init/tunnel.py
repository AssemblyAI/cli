from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from aai_cli.core import config
from aai_cli.core.errors import CLIError
from aai_cli.init import runner

# cloudflared binary name; resolved via shutil.which by callers.
CLOUDFLARED = "cloudflared"

# brew exists only on macOS; everywhere else point at Cloudflare's install docs.
_CLOUDFLARED_DOCS = (
    "https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
)

# A cloudflared quick tunnel prints an ephemeral https://<slug>.trycloudflare.com URL.
_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def install_hint() -> str:
    # A ternary (not an if/return) so neither branch reads as unreachable under
    # mypy --warn-unreachable, which targets one platform at a time: on macOS the
    # second return looked dead, on Linux the first would.
    hint = "brew install cloudflared" if sys.platform == "darwin" else _CLOUDFLARED_DOCS
    return f"Install it: {hint}"


def require_cloudflared(purpose: str) -> None:
    """Raise a clean missing-dependency error when cloudflared isn't on PATH."""
    if shutil.which(CLOUDFLARED) is None:
        raise CLIError(
            f"cloudflared is required to {purpose}.",
            error_type="missing_dependency",
            exit_code=1,
            suggestion=install_hint(),
        )


def tunnel_command(port: int) -> list[str]:
    """The cloudflared quick-tunnel command pointing at the local server."""
    return [CLOUDFLARED, "tunnel", "--url", f"http://localhost:{port}"]


def open_quick_tunnel(port: int, *, cwd: Path) -> tuple[subprocess.Popen[str], str | None, Path]:
    """Spawn a cloudflared quick tunnel for ``port``: (process, URL or None, log path).

    The tunnel binary only proxies the port, so the API key is stripped from its
    environment (keeps the secret out of cloudflared's logs/diagnostics). A None
    URL means cloudflared never reported one — the caller should keep the log
    file (the only evidence of why) and name it; on success it should unlink it.
    """
    fd, name = tempfile.mkstemp(prefix="aai-tunnel-", suffix=".log")
    os.close(fd)
    log_path = Path(name)
    env = {k: v for k, v in os.environ.items() if k != config.ENV_API_KEY}
    process = runner.spawn(tunnel_command(port), cwd=cwd, env=env, log_path=log_path)
    return process, await_url(log_path), log_path


def terminate(process: subprocess.Popen[str] | None) -> None:
    """Terminate a spawned process if it's still running (None / exited: no-op)."""
    if process is not None and process.poll() is None:
        process.terminate()


def find_url(text: str) -> str | None:
    """The first trycloudflare.com URL in `text`, or None."""
    match = _URL.search(text)
    return match.group(0) if match else None


def await_url(
    log_path: Path,
    *,
    timeout: float = 30.0,  # pragma: no mutate -- tuning constant; no unit-observable behavior
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
