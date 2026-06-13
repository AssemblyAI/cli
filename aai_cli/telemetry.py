"""Anonymous usage telemetry, modeled on the Supabase CLI's design.

One allow-listed event per command run (command path, outcome, duration — never
arguments, ids, or account data; a failure also carries the error message,
capped at 500 chars) is shipped to the Datadog logs
intake using a write-only *client* token (``pub…``), the credential class
Datadog designs to be embedded in client apps. ``SHIPPED_CLIENT_TOKEN`` carries
it (it is public by design — never put an API key there);
``AAI_TELEMETRY_CLIENT_TOKEN`` overrides it without a release.

Telemetry is opt-out (``AAI_TELEMETRY_DISABLED=1``, the cross-tool
``DO_NOT_TRACK=1``, or ``assembly telemetry disable``) and must never slow down or
break the command it observes: delivery happens in a detached flusher process,
and every send-side failure is swallowed.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager

import typer

from aai_cli import __version__, argscan, config, procs
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

# Cap on the error message shipped with a failure event: enough for any CLI
# error line, while bounding the payload if an upstream message embeds a body.
_ERROR_MESSAGE_MAX_CHARS = 500


def client_token() -> str:
    """The write-only intake token: env override first, then the shipped one."""
    return os.environ.get(ENV_CLIENT_TOKEN) or SHIPPED_CLIENT_TOKEN


def intake_url() -> str:
    """The Datadog logs-intake URL events are posted to (``AAI_TELEMETRY_INTAKE_URL``
    overrides the default, for tests/staging)."""
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


def consent_source() -> str:
    """Which layer decided :func:`consent_granted`, in the order that layer wins:
    an env kill-switch (``env:AAI_TELEMETRY_DISABLED`` / ``env:DO_NOT_TRACK``), the
    choice persisted by ``assembly telemetry enable/disable`` (``config``), or the
    opt-out ``default``."""
    if os.environ.get(ENV_DISABLED):
        return f"env:{ENV_DISABLED}"
    if os.environ.get(ENV_DO_NOT_TRACK):
        return f"env:{ENV_DO_NOT_TRACK}"
    if config.get_telemetry_enabled() is not None:
        return "config"
    return "default"


def is_enabled() -> bool:
    """Telemetry runs only with both a token to send with and consent to send."""
    return bool(client_token()) and consent_granted()


FIRST_RUN_NOTICE = (
    "Anonymous usage data is collected to improve the CLI; opt out with "
    "'assembly telemetry disable' (or DO_NOT_TRACK=1)."
)


def _notice_suppressed(raw_args: list[str]) -> bool:
    """Whether the invocation asked for quiet or machine-readable output.

    The one-time disclosure is human-facing chrome: it must not decorate a
    ``--quiet`` run nor pollute the machine-readable stderr a ``--json`` (or
    ``-o json``) pipeline relies on.
    """
    return argscan.requests_quiet(raw_args) or argscan.requests_json(raw_args)


def _maybe_emit_first_run_notice() -> None:
    """Disclose collection once, when the anonymous device id is first minted.

    Printed to stderr so stdout stays pipeline-clean. Minting the id here makes the
    disclosure at-most-once-ever: every later run sees the persisted id and stays
    silent (including when the first run suppressed the line via --quiet/--json).
    Wrapped like every other telemetry side effect — a config failure must never
    break the command being recorded.
    """
    try:
        if config.has_device_id():
            return
        config.get_device_id()
        if _notice_suppressed(sys.argv[1:]):
            return
        sys.stderr.write(FIRST_RUN_NOTICE + "\n")
    except (OSError, CLIError):
        return


def build_event(
    command: str,
    *,
    outcome: str,
    exit_code: int,
    duration_ms: int,
    error_message: str | None = None,
) -> dict[str, object]:
    """One invocation event, shaped for the Datadog logs intake.

    Every field is allow-listed and anonymous: the command *path* (never its
    arguments or options), the outcome class, and coarse machine facts. The
    device id is a random UUID minted locally — no account id, email, or
    hostname ever rides along.

    A failure additionally sets ``status: error`` and the reserved
    ``error.kind``/``error.message`` so the event feeds Datadog **Error
    Tracking** (issue grouping), not just log search. ``error.kind`` reuses the
    anonymous ``outcome`` (the ``CLIError.error_type``); ``error.message`` is
    the one-line message the user saw (capped at ``_ERROR_MESSAGE_MAX_CHARS``).
    Stack traces are still deliberately omitted.
    """
    succeeded = outcome == "success"
    event: dict[str, object] = {
        "ddsource": "aai-cli",
        "service": "aai-cli",
        "ddtags": f"version:{__version__}",
        "message": f"{command} {outcome}",
        "status": "info" if succeeded else "error",
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
    if not succeeded:
        error: dict[str, object] = {"kind": outcome}
        if error_message:
            error["message"] = error_message[:_ERROR_MESSAGE_MAX_CHARS]
        event["error"] = error
    return event


def dispatch(event: Mapping[str, object]) -> None:
    """Hand one event to a detached `assembly telemetry flush` process; return immediately.

    The child is the CLI's own (hidden) ``telemetry flush`` subcommand — an explicit,
    reviewable entry point, the same shape as the Vercel CLI's ``telemetry flush``.
    The payload travels via argv and carries only the event + the write-only public
    token, so argv visibility is acceptable. Detaching (own session, stdio discarded)
    is what keeps the user's command from ever waiting on the network; the child's
    env disables telemetry so a flush can never spawn another flusher.
    """
    payload = json.dumps({"url": intake_url(), "token": client_token(), "event": event})
    procs.spawn_detached(["telemetry", "flush", payload], disable_env_var=ENV_DISABLED)


def flush_payload(raw: str) -> None:
    """POST one serialized payload to the intake.

    The token rides both as the ``DD-API-KEY`` header (the v2 logs API form) and
    the ``dd-api-key`` query param (the browser-intake form the client-token
    endpoints expect), so either intake host accepts it. Runs in the detached
    flusher (`assembly telemetry flush`) with stdio discarded, so failures need no
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


def _safe_dispatch(
    command: str,
    started: float,
    *,
    outcome: str,
    exit_code: int,
    error_message: str | None = None,
) -> None:
    duration_ms = int((time.monotonic() - started) * 1000)
    try:
        dispatch(
            build_event(
                command,
                outcome=outcome,
                exit_code=exit_code,
                duration_ms=duration_ms,
                error_message=error_message,
            )
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
    _maybe_emit_first_run_notice()
    started = time.monotonic()
    try:
        yield
    except CLIError as err:
        _safe_dispatch(
            command,
            started,
            outcome=err.error_type,
            exit_code=err.exit_code,
            error_message=err.message,
        )
        raise
    except typer.Exit as exc:
        code = exc.exit_code
        outcome = "success" if code == 0 else "error"
        _safe_dispatch(command, started, outcome=outcome, exit_code=code)
        raise
    except BaseException as exc:
        _safe_dispatch(
            command, started, outcome="internal_error", exit_code=1, error_message=str(exc)
        )
        raise
    _safe_dispatch(command, started, outcome="success", exit_code=0)
