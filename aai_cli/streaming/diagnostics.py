"""Hygiene helpers shared by the realtime session paths (stream, agent, speak).

Two concerns every WebSocket-backed command shares, kept in one place so the
three paths fail identically: silencing library loggers that would dirty stderr
next to the CLI's own normalized error, and turning a rejected WebSocket
handshake (HTTP 401/403) into a CLIError that carries an actionable suggestion
instead of a bare status line.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from aai_cli.core import debuglog
from aai_cli.core import ws as wsutil
from aai_cli.core.errors import APIError, CLIError, NotAuthenticated

# The assemblyai SDK's streaming client logs its own connection failures at ERROR
# ("Connection failed: WebSocket handshake rejected (HTTP 403) (code=403)") through
# child loggers of this one. The CLI configures no logging, so Python's last-resort
# handler would print that line to stderr right before the CLI's normalized error —
# a duplicate in human mode and a non-JSON line polluting --json stderr. The
# CLIError already carries the message, so the logger is raised above ERROR.
SDK_STREAMING_LOGGER = "assemblyai.streaming"

_UNAUTHORIZED = 401


def silence_streaming_logging() -> None:
    """Silence the library loggers that would dirty stderr during a realtime run.

    Extends the shared websockets silencing (``aai_cli.ws``) with the assemblyai
    SDK's streaming logger, which only the `stream` path uses. Idempotent. Stands
    down (like ``aai_cli.ws``) under the root ``-v/--verbose`` flag, where library
    logs are the requested output.
    """
    wsutil.silence_websockets_logging()
    if debuglog.active():
        return
    logging.getLogger(SDK_STREAMING_LOGGER).setLevel(logging.CRITICAL)


def handshake_suggestion(host: str) -> str:
    """The actionable next steps for a rejected streaming handshake."""
    target = host or "the streaming endpoint"
    return (
        "Check 'assembly whoami', that your key matches this environment "
        f"(--sandbox?), and network/proxy access to {target}."
    )


def handshake_error(error: object, message: str, *, host: str) -> CLIError | None:
    """An auth-flavored CLIError for a handshake 401/403, else None.

    401 means the credential itself was rejected -> NotAuthenticated (exit 4,
    ``rejected_key=True`` so auto-login won't retry an env-provided key). 403
    stays an APIError — it also covers WAF/region/plan blocks, mirroring
    ``aai_cli.ws.is_rejected_key`` — but now carries the same suggestion.
    """
    status = wsutil.handshake_status(error)
    if status is None:
        return None
    if status == _UNAUTHORIZED:
        return NotAuthenticated(
            f"{message}: {error}",
            suggestion=handshake_suggestion(host),
            rejected_key=True,
        )
    return APIError(f"{message}: {error}", suggestion=handshake_suggestion(host))


def classify_error(error: object, message: str, *, host: str) -> CLIError:
    """The one CLIError classification for a realtime WebSocket failure.

    Shared by every realtime path (stream, agent, speak) for connect failures and
    recorded stream Error events alike, so they can't drift: a rejected handshake
    (HTTP 401/403) carries the actionable suggestion via ``handshake_error``;
    anything else keeps the ``aai_cli.ws`` mapping — a rejected key becomes
    ``auth_failure()``, the rest ``APIError(f"{message}: …")``.
    """
    rejected = handshake_error(error, message, host=host)
    if rejected is not None:
        return rejected
    return wsutil.auth_or_api_error(error, message)


def open_authorized_ws[T](
    connect: Callable[..., T],
    api_key: str,
    url: str,
    *,
    message: str,
    host: str,
    bearer: bool = True,
    **connect_kwargs: object,
) -> T:
    """Open an ``Authorization``-headered WebSocket, mapping a connect failure via
    ``classify_error``.

    The one connect path for the raw-websocket sessions (agent, speak), so a
    rejected handshake (HTTP 401/403) carries the same actionable suggestion in
    both and everything else keeps the shared classification.

    ``bearer`` selects the AssemblyAI auth scheme for the endpoint: the Voice Agent
    socket expects a ``Bearer <key>`` token (the default), while the streaming
    sockets (STT, TTS) authenticate with the **raw** key — pass ``bearer=False``
    for those, or the server refuses the session with an in-band Error frame.
    """
    token = f"Bearer {api_key}" if bearer else api_key
    try:
        return connect(url, additional_headers={"Authorization": token}, **connect_kwargs)
    except Exception as exc:
        raise classify_error(exc, message, host=host) from exc
