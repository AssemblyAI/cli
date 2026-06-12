"""Shared plumbing for the CLI's WebSocket-backed sessions (voice agent, streaming TTS).

Both sessions classify connect/session failures the same way and silence the same
library loggers; keeping that here means a change to either behavior lands in
`assembly agent` and `assembly speak` together instead of drifting apart.
"""

from __future__ import annotations

import logging

from aai_cli.errors import APIError, CLIError, auth_failure, is_auth_failure

# A pre-upgrade HTTP 403 on the WebSocket handshake is NOT a rejected key (it also
# covers WAF/region/plan blocks) — mirrors how `stream` classifies handshakes.
_HTTP_FORBIDDEN = 403

# Handshake statuses that mean the server refused the connection outright.
_HANDSHAKE_AUTH_STATUSES = (401, 403)

# The sync websockets client logs through these; both are silenced for a session
# (the parent covers any future child logger, the client logger is the one that fires).
WEBSOCKETS_LOGGERS = ("websockets", "websockets.client")


def silence_websockets_logging() -> None:
    """Keep websockets' internal logging off the user's stderr for the session.

    The sync client's background reader thread logs unhandled teardown errors (e.g.
    ``EOFError: stream ended``) as "unexpected internal error" + traceback through the
    ``websockets.client`` logger, which would land on stderr right next to our clean
    CLIError. Those internals are never user-actionable from the CLI, so raise the
    loggers above every level they emit at. Idempotent: re-setting the level is a no-op.
    """
    for name in WEBSOCKETS_LOGGERS:
        logging.getLogger(name).setLevel(logging.CRITICAL)


def handshake_status(exc: object) -> int | None:
    """The HTTP status of a rejected WebSocket handshake (401/403), else None.

    Reads the two structured shapes only — the assemblyai SDK's StreamingError
    carries the status on ``.code``; websockets' InvalidStatus carries it on
    ``.response.status_code`` — never the message text. The single classifier for
    every realtime path (stream, agent, speak), so 401-vs-403 handling can't drift.
    """
    code = getattr(exc, "code", None)
    if code in _HANDSHAKE_AUTH_STATUSES:
        return int(code)
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in _HANDSHAKE_AUTH_STATUSES:
        return int(status)
    return None


def is_rejected_key(exc: object) -> bool:
    """Is this connect/session failure auth-shaped (the key itself was rejected)?

    Mirrors how `stream` classifies handshake failures: a plain HTTP 403 on the
    WebSocket upgrade stays an API error there ("Streaming error: WebSocket handshake
    rejected (HTTP 403)"), so it must not become "Your API key was rejected" here —
    403 also covers non-credential blocks (WAF, region, plan). Only 401, the Voice
    Agent's 1008 policy-violation close, or an explicitly auth-worded message
    (`is_auth_failure`'s text hints) count as a rejected key.
    """
    if handshake_status(exc) == _HTTP_FORBIDDEN:
        return False
    return is_auth_failure(exc)


def auth_or_api_error(exc: object, message: str) -> CLIError:
    """Map a connect/session exception to the right CLIError: a rejected key becomes
    auth_failure(), anything else becomes APIError(f"{message}: {exc}")."""
    if is_rejected_key(exc):
        return auth_failure()
    return APIError(f"{message}: {exc}")
