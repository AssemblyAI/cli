from __future__ import annotations

import base64
import contextlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

from aai_cli import environments
from aai_cli.errors import APIError, CLIError, auth_failure, is_auth_failure

# The websocket connection and the `connect` factory come from a library with no
# usable type stubs; alias that untyped boundary so each role is named and Any
# stays in one place. (Server frames remain dict[str, Any].)
_WebSocket = Any
_Connect = Any

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
