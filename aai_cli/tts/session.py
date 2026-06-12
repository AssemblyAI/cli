from __future__ import annotations

import base64
import binascii
import contextlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlencode

from aai_cli import environments
from aai_cli import ws as wsutil
from aai_cli.errors import APIError, CLIError
from aai_cli.streaming import diagnostics
from aai_cli.tts import audio


class _WebSocket(Protocol):
    """The slice of a websockets sync connection this module drives — named as a
    Protocol so the untyped library boundary is structurally typed, not opaque."""

    def recv(self, timeout: float | None = None) -> str | bytes: ...
    def send(self, data: str, /) -> None: ...  # positional-only: matches ws send(message)
    def close(self) -> None: ...


# The connect factory: returns a fresh _WebSocket. websockets' real sync client
# matches structurally; tests inject a fake with the same surface.
_Connect = Callable[..., _WebSocket]

# The streaming-TTS server synthesizes at 24 kHz unless a sample_rate is requested,
# and echoes the resolved value back in the Begin frame's configuration. Audio frames
# carry only the PCM payload, so Begin is the single source of truth for the rate; this
# is the fallback if that field is ever absent.
_DEFAULT_SAMPLE_RATE = 24000

# Pause inserted between speaker turns in a multi-voice dialogue, for natural pacing.
_INTER_TURN_SILENCE_SECONDS = 0.25

# Bound on the wait for each protocol frame. The server streams frames continuously
# while a synthesis is in flight, so a gap this long means it went silent mid-session;
# without a bound, `assembly speak` would hang forever instead of failing cleanly.
_RECV_TIMEOUT_SECONDS = 60.0


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


def _pcm_duration_seconds(pcm: bytes | bytearray, sample_rate: int) -> float:
    """Seconds of audio in 16-bit mono PCM: two bytes per sample."""
    return len(pcm) / 2 / sample_rate


def _decode_audio_frame(msg: dict[str, object]) -> bytes:
    """The PCM payload of an Audio frame, with a malformed frame (missing or
    undecodable ``audio``) mapped to an APIError that names the defect — not a
    bare KeyError/binascii.Error that surfaces as a cryptic "TTS session failed"."""
    encoded = msg.get("audio")
    if not isinstance(encoded, str):
        raise APIError("TTS service sent an Audio frame without an audio payload.")
    try:
        # validate=True: junk characters must fail loudly, not be silently dropped
        # (the default discards them, corrupting the PCM byte stream).
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise APIError(f"TTS service sent an Audio frame that is not valid base64: {exc}") from exc


def _recv_raw(ws: _WebSocket) -> str | bytes:
    """One frame off the socket, with a bounded wait: a server that goes silent
    mid-session (e.g. never sends the final Audio frame) must fail the command,
    not hang it forever on an unbounded recv()."""
    try:
        return ws.recv(timeout=_RECV_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        raise APIError(
            f"TTS service stopped responding (no frame for {_RECV_TIMEOUT_SECONDS:g}s)."
        ) from exc


def _default_connect(
    url: str, *, additional_headers: dict[str, str], max_size: int | None
) -> _WebSocket:
    """The real websockets sync client, imported lazily so tests can inject a fake."""
    from websockets.sync.client import connect

    return connect(url, additional_headers=additional_headers, max_size=max_size)


def _open_ws(connect: _Connect, api_key: str, url: str) -> _WebSocket:
    """Open the TTS socket, mapping a connect failure to a clean CLIError.

    A rejected handshake (HTTP 401/403) gets the shared actionable suggestion
    (whoami / environment / network); anything else keeps the wsutil mapping.
    """
    message = "Could not connect to the TTS service"
    try:
        return connect(
            url,
            additional_headers={"Authorization": f"Bearer {api_key}"},
            max_size=None,
        )
    except Exception as exc:
        rejected = diagnostics.handshake_error(
            exc, message, host=environments.active().streaming_tts_host
        )
        if rejected is not None:
            raise rejected from exc
        raise wsutil.auth_or_api_error(exc, message) from exc


def _run_protocol(
    ws: _WebSocket, config: SpeakConfig, on_warning: Callable[[str], None] | None
) -> SpeakResult:
    """Send Generate + ForceFlushTextBuffer, collect Audio until is_final, then Terminate."""
    begin = json.loads(_recv_raw(ws))
    if begin.get("type") != "Begin":
        raise APIError(f"TTS service did not start the session (got {begin.get('type')!r}).")
    sample_rate = int(begin.get("configuration", {}).get("sample_rate", _DEFAULT_SAMPLE_RATE))

    ws.send(json.dumps({"type": "Generate", "text": config.text}))
    ws.send(json.dumps({"type": "ForceFlushTextBuffer"}))

    pcm = bytearray()
    while True:
        msg = json.loads(_recv_raw(ws))
        mtype = msg.get("type")
        if mtype == "Audio":
            pcm.extend(_decode_audio_frame(msg))
            if msg.get("is_final"):
                break
        elif mtype == "Error":
            raise APIError(
                f"TTS error ({msg.get('error_code', '')}): {msg.get('error') or 'unknown'}"
            )
        elif mtype == "Warning" and on_warning is not None:
            on_warning(str(msg.get("warning", "")))

    with contextlib.suppress(Exception):
        ws.send(json.dumps({"type": "Terminate"}))

    return SpeakResult(bytes(pcm), sample_rate, _pcm_duration_seconds(pcm, sample_rate))


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
    wsutil.silence_websockets_logging()
    if connect is None:
        connect = _default_connect

    ws = _open_ws(connect, api_key, ws_url(config.query_params()))
    try:
        return _run_protocol(ws, config, on_warning)
    except (CLIError, KeyboardInterrupt, BrokenPipeError):
        raise  # clean CLI errors, Ctrl-C, and a closed pipe are handled upstream
    except Exception as exc:
        raise wsutil.auth_or_api_error(exc, "TTS session failed") from exc
    finally:
        with contextlib.suppress(Exception):
            ws.close()


def synthesize_dialogue(
    api_key: str,
    segments: list[tuple[str, str]],
    *,
    language: str | None = None,
    sample_rate: int | None = None,
    connect: _Connect | None = None,
    on_warning: Callable[[str], None] | None = None,
) -> SpeakResult:
    """Synthesize each ``(voice, text)`` segment and concatenate the PCM.

    Each segment opens its own connection (the voice is fixed at connect time).
    A short silence is inserted between turns — never at the ends. The result's
    sample rate is the rate the server reported for the segments.
    """
    pcm = bytearray()
    sample_rate_out = _DEFAULT_SAMPLE_RATE
    for index, (voice, text) in enumerate(segments):
        config = SpeakConfig(text=text, voice=voice, language=language, sample_rate=sample_rate)
        result = synthesize(api_key, config, connect=connect, on_warning=on_warning)
        if index:
            pcm.extend(audio.silence(result.sample_rate, _INTER_TURN_SILENCE_SECONDS))
        pcm.extend(result.pcm)
        sample_rate_out = result.sample_rate
    return SpeakResult(bytes(pcm), sample_rate_out, _pcm_duration_seconds(pcm, sample_rate_out))
