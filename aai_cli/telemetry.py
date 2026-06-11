"""Anonymous usage telemetry, modeled on the Supabase CLI's design.

One allow-listed event per command run (command path, outcome, duration — never
arguments, file paths, ids, or account data) is shipped to the Datadog logs
intake using a write-only *client* token (``pub…``), the credential class
Datadog designs to be embedded in client apps. ``SHIPPED_CLIENT_TOKEN`` carries
it (it is public by design — never put an API key there);
``AAI_TELEMETRY_CLIENT_TOKEN`` overrides it without a release.

Telemetry is opt-out (``AAI_TELEMETRY_DISABLED=1``, the cross-tool
``DO_NOT_TRACK=1``, or ``aai telemetry disable``) and must never slow down or
break the command it observes: delivery happens in a detached flusher process,
and every send-side failure is swallowed.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager

import typer

from aai_cli import __version__, config
from aai_cli.errors import CLIError

ENV_DISABLED = "AAI_TELEMETRY_DISABLED"
ENV_DO_NOT_TRACK = "DO_NOT_TRACK"
ENV_CLIENT_TOKEN = "AAI_TELEMETRY_CLIENT_TOKEN"
ENV_INTAKE_URL = "AAI_TELEMETRY_INTAKE_URL"

# A Datadog *client token*: the write-only, embeddable credential class (it can
# submit events but read nothing), committed deliberately — the same way PostHog
# `phc_` keys ship in open-source CLIs. An API key (account secret) must never
# appear here. Rotate in Datadog (Organization Settings → Client Tokens) if abused;
# AAI_TELEMETRY_CLIENT_TOKEN overrides without a release.
SHIPPED_CLIENT_TOKEN = "pub0d633113b9f7d22faff215fefaf30b43"

# Datadog's client-token log intake (US1 site). Orgs on another site override via
# AAI_TELEMETRY_INTAKE_URL without needing a new CLI release.
DEFAULT_INTAKE_URL = "https://browser-intake-datadoghq.com/api/v2/logs"

_SEND_TIMEOUT_SECONDS = 5.0


def client_token() -> str:
    """The write-only intake token: env override first, then the shipped one."""
    return os.environ.get(ENV_CLIENT_TOKEN) or SHIPPED_CLIENT_TOKEN


def intake_url() -> str:
    return os.environ.get(ENV_INTAKE_URL) or DEFAULT_INTAKE_URL


def consent_granted() -> bool:
    """Opt-out consent: the env kill-switches win, then the persisted choice.

    Every non-empty value disables — ``DO_NOT_TRACK`` is conventionally ``1`` but
    tools commonly export ``true``, and treating those as "still tracking" would
    betray the user's stated intent.
    """
    if os.environ.get(ENV_DISABLED) or os.environ.get(ENV_DO_NOT_TRACK):
        return False
    return config.get_telemetry_enabled() is not False


def is_enabled() -> bool:
    """Telemetry runs only with both a token to send with and consent to send."""
    return bool(client_token()) and consent_granted()


def build_event(
    command: str, *, outcome: str, exit_code: int, duration_ms: int
) -> dict[str, object]:
    """One invocation event, shaped for the Datadog logs intake.

    Every field is allow-listed and anonymous: the command *path* (never its
    arguments or options), the outcome class, and coarse machine facts. The
    device id is a random UUID minted locally — no account id, email, or
    hostname ever rides along.
    """
    return {
        "ddsource": "aai-cli",
        "service": "aai-cli",
        "ddtags": f"version:{__version__}",
        "message": f"{command} {outcome}",
        "command": command,
        "outcome": outcome,
        "exit_code": exit_code,
        "duration_ms": duration_ms,
        "cli_version": __version__,
        "os": platform.system().lower(),
        "python_version": platform.python_version(),
        "ci": bool(os.environ.get("CI")),
        "device_id": config.get_device_id(),
    }


def dispatch(event: Mapping[str, object]) -> None:
    """Hand one event to a detached `aai telemetry flush` process; return immediately.

    The child is the CLI's own (hidden) ``telemetry flush`` subcommand — an explicit,
    reviewable entry point, the same shape as the Vercel CLI's ``telemetry flush``.
    The payload travels via argv and carries only the event + the write-only public
    token, so argv visibility is acceptable. Detaching (own session, stdio discarded)
    is what keeps the user's command from ever waiting on the network; the child's
    env disables telemetry so a flush can never spawn another flusher.
    """
    payload = json.dumps({"url": intake_url(), "token": client_token(), "event": event})
    subprocess.Popen(
        [sys.executable, "-m", "aai_cli", "telemetry", "flush", payload],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env={**os.environ, ENV_DISABLED: "1"},
    )


def flush_payload(raw: str) -> None:
    """POST one serialized payload to the intake.

    The token rides both as the ``DD-API-KEY`` header (the v2 logs API form) and
    the ``dd-api-key`` query param (the browser-intake form the client-token
    endpoints expect), so either intake host accepts it. Runs in the detached
    flusher (`aai telemetry flush`) with stdio discarded, so failures need no
    handling here.
    """
    import httpx2 as httpx

    payload = json.loads(raw)
    token = str(payload["token"])
    with httpx.Client(timeout=_SEND_TIMEOUT_SECONDS) as client:
        client.post(
            str(payload["url"]),
            params={"dd-api-key": token},
            headers={"DD-API-KEY": token},
            json=[payload["event"]],
        )


def _safe_dispatch(command: str, started: float, *, outcome: str, exit_code: int) -> None:
    duration_ms = int((time.monotonic() - started) * 1000)
    try:
        dispatch(
            build_event(command, outcome=outcome, exit_code=exit_code, duration_ms=duration_ms)
        )
    except (OSError, CLIError):
        # Best-effort by contract: a config/spawn failure while *recording* a command
        # must never surface in the command itself.
        return


@contextmanager
def track(command: str) -> Generator[None]:
    """Record one command run, deriving the outcome from whatever escapes the body.

    CLIErrors keep their machine-readable ``error_type`` as the outcome; a
    deliberate ``typer.Exit`` maps through its code; anything else is the
    catch-all ``internal_error``. The body's exception always re-raises —
    tracking observes control flow, never alters it.
    """
    if not is_enabled():
        yield
        return
    started = time.monotonic()
    try:
        yield
    except CLIError as err:
        _safe_dispatch(command, started, outcome=err.error_type, exit_code=err.exit_code)
        raise
    except typer.Exit as exc:
        code = exc.exit_code
        outcome = "success" if code == 0 else "error"
        _safe_dispatch(command, started, outcome=outcome, exit_code=code)
        raise
    except BaseException:
        _safe_dispatch(command, started, outcome="internal_error", exit_code=1)
        raise
    _safe_dispatch(command, started, outcome="success", exit_code=0)
