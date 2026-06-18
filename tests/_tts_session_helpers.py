"""Shared test doubles for the streaming-TTS session suites.

Split out of ``test_tts_session.py`` so the single-synthesis tests and the
``synthesize_dialogue`` tests can each stay under the 500-line file gate while
driving the same fake socket and frame builders.
"""

from __future__ import annotations

import base64
import json

from aai_cli.core import environments


def use_env(name: str) -> None:
    environments.set_active(environments.get(name))


class FakeWS:
    """A minimal stand-in for a websockets sync connection."""

    def __init__(self, incoming: list[str]) -> None:
        self._incoming = list(incoming)
        self.sent: list[dict[str, object]] = []
        self.closed = False
        self.recv_timeouts: list[float | None] = []

    def recv(self, timeout: float | None = None) -> str:
        self.recv_timeouts.append(timeout)
        return self._incoming.pop(0)

    def send(self, data: str) -> None:
        self.sent.append(json.loads(data))

    def close(self) -> None:
        self.closed = True


def audio_frame(pcm: bytes, *, final: bool) -> str:
    # An Audio frame with an explicit is_final flag — the defensive end-of-stream path
    # (the live server instead omits is_final and ends with FlushDone, see audio_chunk).
    # The sample rate is reported once, up front, in the Begin frame's configuration.
    return json.dumps(
        {
            "type": "Audio",
            "audio": base64.b64encode(pcm).decode("ascii"),
            "is_final": final,
        }
    )


def audio_chunk(pcm: bytes) -> str:
    # The real server's Audio frames carry the PCM payload and a flush_id, but NO
    # is_final flag — completion is signalled by a separate FlushDone frame.
    return json.dumps(
        {"type": "Audio", "audio": base64.b64encode(pcm).decode("ascii"), "flush_id": 0}
    )


def flush_done_frame() -> str:
    return json.dumps({"type": "FlushDone", "flush_id": 0, "audio_duration_ms": 880})


def begin_frame(*, sample_rate: int = 24000) -> str:
    return json.dumps(
        {
            "type": "Begin",
            "id": "s1",
            "expires_at": 1,
            "configuration": {"voice": "jane", "language": "english", "sample_rate": sample_rate},
        }
    )


def connect_returning(ws: FakeWS, captured: dict[str, object]):
    def _connect(url: str, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return ws

    return _connect
