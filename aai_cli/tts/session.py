from __future__ import annotations

import base64
import contextlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlencode

from aai_cli import environments
from aai_cli.errors import APIError, CLIError, auth_failure, is_auth_failure


class _WebSocket(Protocol):
    """The slice of a websockets sync connection this module drives — named as a
    Protocol so the untyped library boundary is structurally typed, not opaque."""

    def recv(self) -> str | bytes: ...
    def send(self, data: str, /) -> None: ...  # positional-only: matches ws send(message)
    def close(self) -> None: ...


# The connect factory: returns a fresh _WebSocket. websockets' real sync client
# matches structurally; tests inject a fake with the same surface.
_Connect = Callable[..., _WebSocket]

# A pre-upgrade HTTP 403 on the WebSocket handshake is NOT a rejected key (it also
# covers WAF/region/plan blocks) — mirrors how agent/stream classify handshakes.
_HTTP_FORBIDDEN = 403

# websockets' sync client logs teardown errors through these; silence them so an
# internal "stream ended" traceback never lands on stderr next to our clean error.
_WEBSOCKETS_LOGGERS = ("websockets", "websockets.client")


@dataclass(frozen=True)
class SpeakConfig:
    """Per-run TTS parameters. Optional fields are only sent when set, so the
    server applies its own defaults (voice/language/sample_rate) otherwise."""

    text: str
    voice: str | None = None
    language: str | None = None
    sample_rate: int | None = None

    def query_params(self) -> dict[str, str]:
        params: dict[str, str] = {}
        if self.voice is not None:
            params["voice"] = self.voice
        if self.language is not None:
            params["language"] = self.language
        if self.sample_rate is not None:
            params["sample_rate"] = str(self.sample_rate)
        return params


@dataclass(frozen=True)
class SpeakResult:
    """The synthesized audio: raw 16-bit mono PCM plus its sample rate."""

    pcm: bytes
    sample_rate: int
    audio_duration_seconds: float


def is_available() -> bool:
    """True only where the active environment has a streaming-TTS host (sandbox)."""
    return bool(environments.active().streaming_tts_host)


def ws_url(params: dict[str, str]) -> str:
    """The streaming-TTS socket URL for the active environment, with query params."""
    base = f"wss://{environments.active().streaming_tts_host}/v1/ws/"
    return f"{base}?{urlencode(params)}" if params else base


def _silence_websockets_logging() -> None:
    """Keep websockets' internal teardown logging off the user's stderr."""
    for name in _WEBSOCKETS_LOGGERS:
        logging.getLogger(name).setLevel(logging.CRITICAL)


def _is_rejected_key(exc: Exception) -> bool:
    """Auth-shaped failure (the key was rejected) vs a generic block. A plain HTTP
    403 on the upgrade is NOT a rejected key (WAF/region/plan also 403)."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == _HTTP_FORBIDDEN:
        return False
    return is_auth_failure(exc)


def _auth_or_api_error(exc: Exception, message: str) -> CLIError:
    """A rejected key becomes auth_failure(); anything else becomes an APIError."""
    if _is_rejected_key(exc):
        return auth_failure()
    return APIError(f"{message}: {exc}")


def _default_connect(
    url: str, *, additional_headers: dict[str, str], max_size: int | None
) -> _WebSocket:
    """The real websockets sync client, imported lazily so tests can inject a fake."""
    from websockets.sync.client import connect

    return connect(url, additional_headers=additional_headers, max_size=max_size)


def _open_ws(connect: _Connect, api_key: str, url: str) -> _WebSocket:
    """Open the TTS socket, mapping a connect failure to a clean CLIError."""
    try:
        return connect(
            url,
            additional_headers={"Authorization": f"Bearer {api_key}"},
            max_size=None,
        )
    except Exception as exc:
        raise _auth_or_api_error(exc, "Could not connect to the TTS service") from exc


def _run_protocol(
    ws: _WebSocket, config: SpeakConfig, on_warning: Callable[[str], None] | None
) -> SpeakResult:
    """Send Generate+Flush, collect Audio until is_final_for_flush, then Terminate."""
    begin = json.loads(ws.recv())
    if begin.get("type") != "Begin":
        raise APIError(f"TTS service did not start the session (got {begin.get('type')!r}).")

    ws.send(json.dumps({"type": "Generate", "text": config.text}))
    ws.send(json.dumps({"type": "Flush"}))

    pcm = bytearray()
    # Bound only so the post-loop duration calc has a name; the value is never
    # observed (the sole loop break is inside the Audio branch, after sample_rate
    # is reassigned from the frame), so its literal can't be asserted.
    sample_rate = 0  # pragma: no mutate
    while True:
        msg = json.loads(ws.recv())
        mtype = msg.get("type")
        if mtype == "Audio":
            pcm.extend(base64.b64decode(msg["audio"]))
            sample_rate = int(msg["sample_rate"])
            if msg.get("is_final_for_flush"):
                break
        elif mtype == "Error":
            raise APIError(
                f"TTS error ({msg.get('error_code', '')}): {msg.get('error') or 'unknown'}"
            )
        elif mtype == "Warning" and on_warning is not None:
            on_warning(str(msg.get("warning", "")))

    with contextlib.suppress(Exception):
        ws.send(json.dumps({"type": "Terminate"}))

    duration = len(pcm) / 2 / sample_rate
    return SpeakResult(bytes(pcm), sample_rate, duration)


def synthesize(
    api_key: str,
    config: SpeakConfig,
    *,
    connect: _Connect | None = None,
    on_warning: Callable[[str], None] | None = None,
) -> SpeakResult:
    """Open the streaming-TTS socket and synthesize ``config.text`` to PCM.

    ``connect`` defaults to websockets' synchronous client; injectable for tests.
    Connect/session failures map to a clean CLIError (a rejected key -> exit 4).
    """
    _silence_websockets_logging()
    if connect is None:
        connect = _default_connect

    ws = _open_ws(connect, api_key, ws_url(config.query_params()))
    try:
        return _run_protocol(ws, config, on_warning)
    except (CLIError, KeyboardInterrupt, BrokenPipeError):
        raise  # clean CLI errors, Ctrl-C, and a closed pipe are handled upstream
    except Exception as exc:
        raise _auth_or_api_error(exc, "TTS session failed") from exc
    finally:
        with contextlib.suppress(Exception):
            ws.close()
